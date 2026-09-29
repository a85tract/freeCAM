"""Segmented execution of a stage: the original Fortran, paused at each replaced kernel.

A stage whose kernels are all original runs its Fortran action whole (see
``NativeStage.select_mode``).  A stage with a replacement runs the same
Fortran through a *segment runner*: the runner executes the original code
continuously and returns to Python only where a replaced kernel would have
been called, with a *frame* describing that call's arguments in place.
Python runs the model on the frame, writes the answer back, and tells the
runner to resume from where it stopped.  The runner never calls Python;
every transition is a call Python makes.  This module is the Python half of
that protocol -- events, frames, the drive loop and its lifecycle rules --
written against a small runner interface, so it is tested here with a fake
runner and later bound to the image's ``pycam_stage_*`` entries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .errors import PhysicsError


class SegmentEvent(IntEnum):
    """What a runner reports after start() or resume()."""

    DONE = 0
    NEEDS_PYTHON_KERNEL = 1
    ERROR = 2


@dataclass(frozen=True, slots=True)
class FrameArgument:
    """One argument of a paused kernel call: the array where the Fortran holds it."""

    name: str
    array: np.ndarray
    intent: str          # "in", "out" or "inout"

    @property
    def is_input(self) -> bool:
        return self.intent in ("in", "inout")

    @property
    def is_output(self) -> bool:
        return self.intent in ("out", "inout")


@dataclass(frozen=True, slots=True)
class KernelFrame:
    """A paused kernel call, as the runner describes it.

    ``arguments`` are views of the Fortran storage the call would have read
    and written, chunk-shaped (``pcols`` leading).  Only the first ``ncol``
    lanes are live; a model never sees a padding lane and never writes one.
    ``token`` identifies this pause: the frame is good for exactly one
    write-back and one resume.  ``ncol`` is the chunk's live column count,
    the first axis of every array; a frame without one (``ncol <= 0``: a
    routine with no ``ncol`` dummy, served at a hook) is live in full.
    """

    kernel: str
    call_index: int
    lchnk: int
    ncol: int
    substep: int
    arguments: tuple[FrameArgument, ...]
    token: int

    def argument(self, name: str) -> FrameArgument:
        for argument in self.arguments:
            if argument.name == name:
                return argument
        raise PhysicsError(f"kernel {self.kernel!r} has no argument {name!r}")

    def batch(self) -> dict[str, np.ndarray]:
        """The live lanes of every input, copied: what the model is handed."""

        batch: dict[str, np.ndarray] = {}
        for argument in self.arguments:
            if not argument.is_input:
                continue
            batch[argument.name] = _lanes(argument.array, self.ncol).copy()
        return batch

    def write_back(self, answer: Mapping[str, Any]) -> tuple[str, ...]:
        """Put the model's answer into the live lanes of every output, exactly.

        Every output the kernel declares must be answered; the answer must
        have the output's dtype and its live-lane shape.  ``casting="no"``:
        a value that would need conversion is a contract error, not a
        rounding.  Returns the names written.
        """

        outputs = [argument for argument in self.arguments if argument.is_output]
        missing = [argument.name for argument in outputs if argument.name not in answer]
        if missing:
            raise PhysicsError(
                f"kernel {self.kernel!r}: the model answered {sorted(answer)} but the "
                f"kernel also writes {missing}")
        written = []
        for argument in outputs:
            target = _lanes(argument.array, self.ncol)
            value = np.asarray(answer[argument.name])
            if target.size == 0:
                # an output the routine has no room for -- a field this
                # configuration never registered, packed as a zero-size array,
                # or a chunk with no cloudy column: nothing to write, whatever
                # extents the model gave its empty answer
                if value.size != 0:
                    raise PhysicsError(
                        f"kernel {self.kernel!r}: {argument.name} has no storage in this call "
                        f"(shape {target.shape}), the model returned {value.shape}")
                written.append(argument.name)
                continue
            if value.shape != target.shape:
                raise PhysicsError(
                    f"kernel {self.kernel!r}: {argument.name} must be {target.shape}, "
                    f"the model returned {value.shape}")
            if value.dtype != target.dtype:
                raise PhysicsError(
                    f"kernel {self.kernel!r}: {argument.name} must be {target.dtype}, "
                    f"the model returned {value.dtype}")
            np.copyto(target, value, casting="no")
            written.append(argument.name)
        return tuple(written)


class OriginalKernel:
    """Put in a kernel slot: the original Fortran kernel, but called from Python.

    A test's replacement.  The stage runs segmented -- the runner pauses at
    the kernel, hands Python the frame -- and Python answers with the
    original direct kernel run on the frame's own values.  Bit-for-bit
    against the whole stage is then the proof that the pause, the frame and
    the write-back are right, since the arithmetic is the same routine.
    """

    def __repr__(self) -> str:
        return "OriginalKernel()"


class OriginalByChunk:
    """Put in a kernel slot: the original, answering every waiting chunk in one call.

    Resolved by the stage to :class:`OriginalAtPauseByChunk` when its runner runs
    the original at a pause, and otherwise to :class:`ByChunk` around its original
    kernel through Python: the batched mode's stacking and split, around arithmetic
    that is the original routine chunk by chunk -- bit-for-bit is the proof of that path.
    """

    def __repr__(self) -> str:
        return "OriginalByChunk()"


class OriginalAtPause:
    """Put in a kernel slot: the original call, run by the runner itself at the pause.

    The runner's `original` entry executes the very call statement on the
    frame's storage.  This model then reads every output the frame declares,
    zeroes it, and answers with what it read -- so the frame's write-back
    puts the original's values back exactly, and the gate exercises the
    pause, the frame's slots, the write-back and the resume, with no direct
    kernel or standalone image required.  Bit-for-bit output is then the
    proof of the path.
    """

    takes_frame = True

    def __call__(self, frame: "KernelFrame", runner: "SegmentRunner", context: int) -> dict[str, np.ndarray]:
        runner.run_original(context, frame.kernel)          # type: ignore[attr-defined]
        answer: dict[str, np.ndarray] = {}
        for argument in frame.arguments:
            if not argument.is_output:
                continue
            target = _lanes(argument.array, frame.ncol)
            answer[argument.name] = np.array(target, copy=True)
            if target.size:
                target[...] = 0
        return answer

    def __repr__(self) -> str:
        return "OriginalAtPause()"


class OriginalAtPauseByChunk:
    """What :class:`OriginalByChunk` resolves to when the runner runs the original at a pause.

    One chunk-batch call for every chunk waiting at the kernel, answered chunk by
    chunk by the original call at that chunk's own pause: each waiting slot is made
    live, its frame's inputs must be, bit for bit, what the batch stacked for it, and
    the original runs on its storage as :class:`OriginalAtPause` runs it.  The
    answers are stacked chunk after chunk, and the batch's split puts each back into
    its own slot -- so bit-for-bit output proves the stacking, the split and the
    write-back of the batched path, with the original's own arithmetic.
    """

    takes_chunk_batches = True
    #: called with the runner, the context and the waiting slots, in the batch's order
    takes_slots = True

    def __call__(self, batch: "ChunkBatch", runner: "SegmentRunner", context: int,
                 slots: Sequence[int]) -> dict[str, np.ndarray]:
        answers: list[dict[str, np.ndarray]] = []
        start = 0
        for index, (slot, rows) in enumerate(zip(slots, batch.rows)):
            runner.batch_select(context, slot)              # type: ignore[attr-defined]
            frame = runner.frame(context)
            _require_stacked(batch, index, start, frame)
            answers.append(OriginalAtPause()(frame, runner, context))
            start += rows
        return {name: np.concatenate([answer[name] for answer in answers], axis=0) for name in answers[0]}

    def __repr__(self) -> str:
        return "OriginalAtPauseByChunk()"


def _require_stacked(batch: "ChunkBatch", index: int, start: int, frame: KernelFrame) -> None:
    """The ``index``-th chunk of ``batch`` is, bit for bit, what its live frame holds as inputs."""

    rows = batch.rows[index]
    changed = []
    for argument in frame.arguments:
        if not argument.is_input:
            continue
        if argument.name in batch.stacked:
            given = batch.inputs[argument.name][start:start + rows]
        elif argument.name in batch.inputs:
            given = batch.inputs[argument.name]
        elif argument.name in batch.per_chunk:
            given = batch.per_chunk[argument.name][index]
        else:
            changed.append(f"{argument.name} is not in the batch")
            continue
        live = np.ascontiguousarray(_lanes(argument.array, frame.ncol))
        given = np.ascontiguousarray(given)
        if live.shape != given.shape or live.dtype != given.dtype or live.tobytes() != given.tobytes():
            changed.append(f"{argument.name} (batch {given.dtype}{given.shape}, frame {live.dtype}{live.shape})")
    if changed:
        raise PhysicsError(f"kernel {batch.kernel!r}: chunk {index} of the batch (lchnk {frame.lchnk}) is not "
                           f"what its frame holds: " + "; ".join(changed))


class FrameCapture:
    """Put in a kernel slot: the original at the pause, with every call's frame recorded.

    Before the original runs, every input the frame declares is copied; after
    it runs, every output is copied and answered exactly as
    :class:`OriginalAtPause` does, so the stage stays bit-for-bit and the
    record holds what the kernel was given and what it returned, call by
    call.  :meth:`save` writes the rank's record as an ``.npz``: one array per
    call and argument (``in/<call>/<name>``, ``out/<call>/<name>``, the live
    lanes only) and a JSON ``meta`` array with the step, chunk column count
    and token of each call.  ``step`` reports the model step when called.
    """

    takes_frame = True

    def __init__(self, kernel: str, every: int = 1) -> None:
        self.kernel = kernel
        #: record one call in ``every``; the others are answered by the original and not kept
        self.every = max(1, int(every))
        self.seen = 0
        #: the model step in flight, set by the stage before each run
        self.current_step: int | None = None
        self.inputs: list[dict[str, np.ndarray]] = []
        self.outputs: list[dict[str, np.ndarray]] = []
        self.meta: list[dict[str, Any]] = []
        self._original = OriginalAtPause()

    def __call__(self, frame: "KernelFrame", runner: "SegmentRunner", context: int) -> dict[str, np.ndarray]:
        self.seen += 1
        if (self.seen - 1) % self.every:
            return self._original(frame, runner, context)
        before: dict[str, np.ndarray] = {}
        for argument in frame.arguments:
            if argument.is_output and argument.intent == "out":
                continue
            before[argument.name] = np.array(_live(argument.array, frame.ncol), copy=True)
        answer = self._original(frame, runner, context)
        self.inputs.append(before)
        self.outputs.append({name: np.array(value, copy=True) for name, value in answer.items()})
        self.meta.append({
            "step": None if self.current_step is None else int(self.current_step),
            "ncol": int(frame.ncol),
            "token": int(frame.token),
            "kernel": frame.kernel,
        })
        return answer

    @property
    def calls(self) -> int:
        return len(self.meta)

    def save(self, path: str | Path) -> Path:
        import json

        arrays: dict[str, np.ndarray] = {}
        for index, (before, after) in enumerate(zip(self.inputs, self.outputs)):
            for name, value in before.items():
                arrays[f"in/{index}/{name}"] = value
            for name, value in after.items():
                arrays[f"out/{index}/{name}"] = value
        arrays["meta"] = np.array(json.dumps(self.meta))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **arrays)
        return target

    def __repr__(self) -> str:
        return f"FrameCapture({self.kernel!r}, calls={self.calls}, every={self.every})"


def _lanes(array: np.ndarray, ncol: int) -> np.ndarray:
    """The live lanes of a frame array: the first ``ncol`` along the first axis.

    A scalar has no lanes.  A frame without a column count (``ncol <= 0``)
    is live in full: the routine has no ``ncol`` dummy and every element it
    was given counts -- slicing to ``[:0]`` would drop the whole array, as it
    did for ``fluxbelowinv``'s profiles in capture run 7343844.
    """

    if array.ndim == 0 or ncol <= 0:
        return array
    return array[:ncol] if array.shape[0] >= ncol else array


def _live(array: np.ndarray, ncol: int) -> np.ndarray:
    """The live lanes of a frame array (see :func:`_lanes`)."""

    return _lanes(array, ncol)


class SegmentRunner(Protocol):
    """What the image (or a fake) offers for one stage.

    The runner owns a rank-local context per stage.  ``start`` runs the
    original code from the top of the stage with ``mask`` naming the kernels
    Python computes; ``frame`` describes the pause ``start``/``resume``
    reported; ``resume`` continues past the kernel named.  ``reset`` returns
    a context to idle after a failure; ``destroy`` frees it.  A runner with a
    batched mode (``offers_batch``) also has ``batch_begin`` (the slot count),
    ``batch_select``, ``batch_advance`` and ``batch_end``; frame, resume and
    the original then act on the live slot.
    """

    def create(self, stage: str) -> int: ...
    def start(self, context: int, mask: Mapping[str, bool]) -> SegmentEvent: ...
    def frame(self, context: int) -> KernelFrame: ...
    def resume(self, context: int, kernel: str, token: int) -> SegmentEvent: ...
    def error(self, context: int) -> str: ...
    def reset(self, context: int) -> None: ...
    def destroy(self, context: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ChunkBatch:
    """Every chunk waiting at one kernel, as one batch: what a chunk-batch model is handed.

    ``inputs`` stacks the live lanes of every column-shaped input chunk after chunk,
    ``rows[i]`` of them from the i-th; an input with no column axis (a time step, a
    per-constituent flag) is there once when every chunk has the same, and otherwise
    only in ``per_chunk`` (a chunk's own count or id), chunk by chunk.  The model
    answers every output the kernel writes, stacked the same way.
    """

    kernel: str
    inputs: Mapping[str, np.ndarray]
    rows: tuple[int, ...]
    lchnks: tuple[int, ...]
    per_chunk: Mapping[str, tuple[np.ndarray, ...]]
    #: the inputs stacked chunk after chunk; the rest of ``inputs`` is every chunk's alike
    stacked: frozenset[str] = frozenset()

    @property
    def columns(self) -> int:
        return int(sum(self.rows))


class ByChunk:
    """A per-chunk model taking chunk batches: each chunk's rows handed to it alone.

    The stacking and the split around the model, without changing what it
    computes -- so the original kernel through Python, wrapped here, proves the
    batched path bit-for-bit.
    """

    takes_chunk_batches = True

    def __init__(self, model: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> None:
        self.model = model

    def __call__(self, batch: ChunkBatch) -> dict[str, np.ndarray]:
        pieces: list[Mapping[str, Any]] = []
        start = 0
        for index, rows in enumerate(batch.rows):
            one: dict[str, Any] = {}
            for name, value in batch.inputs.items():
                one[name] = value[start:start + rows] if name in batch.stacked else value
            for name, values in batch.per_chunk.items():
                one[name] = values[index]
            pieces.append(self.model(one))
            start += rows
        return {name: np.concatenate([np.asarray(piece[name]) for piece in pieces], axis=0) for name in pieces[0]}

    def __repr__(self) -> str:
        return f"ByChunk({self.model!r})"


@dataclass(slots=True)
class _Waiting:
    """A slot paused at a kernel in batched mode, and what was read from it before it was parked."""

    kernel: str
    token: int
    lchnk: int
    ncol: int
    batch: dict[str, np.ndarray] | None
    lanes: frozenset[str]
    #: the outputs the call has no storage for (an unassociated pointer, a zero-size field)
    empty: frozenset[str]
    #: the frame's inputs as parked, when the stage verifies what comes back (verify_frames)
    parked: dict[str, np.ndarray] | None = None


def _verify_restored(stage: str, slot: int, frame: KernelFrame, parked: Mapping[str, np.ndarray]) -> None:
    """A slot's frame inputs as they were parked, bit for bit, or a report of what came back changed."""

    changed = []
    for argument in frame.arguments:
        if argument.name not in parked:
            continue
        now = np.ascontiguousarray(_lanes(argument.array, frame.ncol))
        then = np.ascontiguousarray(parked[argument.name])
        if now.shape != then.shape:
            changed.append(f"{argument.name} shape {then.shape} -> {now.shape}")
            continue
        if now.size == 0 or now.tobytes() == then.tobytes():
            continue
        width = now.dtype.itemsize
        differ = (now.view(np.uint8).reshape(now.size, width) != then.view(np.uint8).reshape(then.size, width)).any(axis=1)
        first = int(np.flatnonzero(differ)[0])
        index = np.unravel_index(first, now.shape) if now.ndim else ()
        changed.append(f"{argument.name}: {int(differ.sum())} of {now.size} differ, first at {tuple(int(i) for i in index)} "
                       f"(parked {then.reshape(-1)[first]!r}, back {now.reshape(-1)[first]!r})")
    if changed:
        raise PhysicsError(f"{stage}: chunk slot {slot} (lchnk {frame.lchnk}) came back with its frame "
                           f"inputs changed: " + "; ".join(changed))


def _lane_names(frame: KernelFrame) -> frozenset[str]:
    """The frame's column-shaped arguments: those whose first extent is the chunk's (pcols)."""

    extents = [a.array.shape[0] for a in frame.arguments if a.array.ndim > 0]
    if frame.ncol <= 0 or not extents:
        return frozenset()
    pcols = max(extents)
    return frozenset(a.name for a in frame.arguments if a.array.ndim > 0 and a.array.shape[0] == pcols >= frame.ncol)


@dataclass(slots=True)
class SegmentCounters:
    """What one segmented run cost the framework, apart from the models."""

    starts: int = 0
    pauses: int = 0
    resumes: int = 0
    model_calls: int = 0
    crossings: int = 0           # Python -> Fortran calls: start + frame + resume ...
    bytes_copied_in: int = 0
    bytes_copied_out: int = 0
    #: model calls by the kernel they answered, so a run can show where its pauses were
    calls_by_kernel: dict[str, int] = field(default_factory=dict)


def _live_bytes(array: np.ndarray, ncol: int) -> int:
    """The bytes of an output's live lanes; a scalar served where it lives has no lanes."""

    return _lanes(array, ncol).nbytes


class SegmentedStage:
    """Drives one stage's segment runner through a step.

    Lifecycle: the context is created on first use and kept; it is *idle*
    between steps and *paused* between a NEEDS_PYTHON_KERNEL event and the
    matching resume.  Kernel slots may only change while idle -- the
    caller checks that -- and a model failure destroys the context and
    marks the stage tainted, since the Fortran already executed up to the
    pause cannot be undone.
    """

    def __init__(self, stage_name: str, runner: SegmentRunner) -> None:
        self.stage_name = stage_name
        self.runner = runner
        self.context: int | None = None
        self.paused_on: KernelFrame | None = None
        self.tainted: str | None = None
        self.generation = 0
        self.counters = SegmentCounters()
        #: batched mode: a slot's frame inputs, copied when it is parked, must come back bit
        #: for bit when it is made live again -- a check of the runner's chunk store
        #: (FREECAM_BATCH_VERIFY=1 turns it on)
        self.verify_frames = os.environ.get("FREECAM_BATCH_VERIFY") == "1"

    @property
    def idle(self) -> bool:
        return self.paused_on is None and self.tainted is None

    def run(self, kernels: Mapping[str, Callable[..., Mapping[str, Any]] | None], *, whole: bool = False,
            batched: bool = False) -> None:
        """One step of the stage: the original Fortran with ``kernels`` at their pauses.

        ``whole`` runs the driver through the runner with no pause armed: for a stage whose
        process slot inside the runner (a compiled plugin bound in the image) answers a
        branch, so the run is one crossing and no Python in the step.  ``batched`` runs
        every chunk of the rank to the kernel before any is answered, so a chunk-batch
        model answers all of them in one call (see :meth:`_run_batched`).
        """

        if self.tainted is not None:
            raise PhysicsError(
                f"{self.stage_name}: a model left the stage tainted and it cannot run "
                f"again:\n{self.tainted}")
        if self.paused_on is not None:
            raise PhysicsError(
                f"{self.stage_name}: still paused on {self.paused_on.kernel!r}; a step "
                f"cannot start inside another")
        mask = {name: kernel is not None for name, kernel in kernels.items()}
        if not any(mask.values()) and not whole:
            raise PhysicsError(
                f"{self.stage_name}: nothing is replaced; run the original stage whole")
        spec = getattr(self.runner, "spec", None)
        conflicts = spec.replacement_conflicts(mask) if spec is not None and hasattr(spec, "replacement_conflicts") else []
        if conflicts:
            detail = "; ".join(f"{inner!r} is called inside {outer!r}" for inner, outer in conflicts)
            raise PhysicsError(
                f"{self.stage_name}: a kernel and the kernel it runs inside are both replaced: {detail}. "
                f"Replace one or the other; a replaced outer kernel never reaches the inner call")
        if batched and not getattr(self.runner, "offers_batch", False):
            raise PhysicsError(
                f"{self.stage_name}: batched chunks were asked for, but the image's runner has no "
                f"batched mode (batch_chunks in its pausable spec)")
        if self.context is None:
            self.context = self.runner.create(self.stage_name)
            self.counters.crossings += 1
        self.generation += 1
        if batched:
            self._run_batched(kernels, mask)
            return
        counters = self.counters
        pauses_before = counters.pauses
        event = self.runner.start(self.context, mask)
        counters.starts += 1
        counters.crossings += 1
        try:
            while event != SegmentEvent.DONE:
                if event == SegmentEvent.ERROR:
                    detail = self.runner.error(self.context)
                    raise PhysicsError(
                        f"{self.stage_name}: the runner failed after {counters.pauses - pauses_before} "
                        f"pause(s) this run: {detail}")
                frame = self.runner.frame(self.context)
                counters.crossings += 1
                counters.pauses += 1
                self.paused_on = frame
                model = kernels.get(frame.kernel)
                if model is None:
                    raise PhysicsError(
                        f"{self.stage_name}: the runner paused on {frame.kernel!r}, which "
                        f"is not replaced")
                if getattr(model, "takes_frame", False):
                    answer = model(frame, self.runner, self.context)
                else:
                    batch = frame.batch()
                    counters.bytes_copied_in += sum(v.nbytes for v in batch.values())
                    answer = model(batch)
                counters.model_calls += 1
                counters.calls_by_kernel[frame.kernel] = counters.calls_by_kernel.get(frame.kernel, 0) + 1
                written = frame.write_back(answer)
                counters.bytes_copied_out += sum(
                    _live_bytes(frame.argument(name).array, frame.ncol) for name in written)
                event = self.runner.resume(self.context, frame.kernel, frame.token)
                counters.resumes += 1
                counters.crossings += 1
                self.paused_on = None
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise

    def _run_batched(self, kernels: Mapping[str, Callable[..., Mapping[str, Any]] | None],
                     mask: Mapping[str, bool]) -> None:
        """Every chunk of the rank to its next pause, then every waiting chunk answered, until all finish.

        Each chunk runs in a slot of the runner's own; one is live at a time.  A round
        runs every slot to its next pause or the end of its chunk.  The slots waiting at
        a kernel whose model takes chunk batches are answered by one call with their
        lanes stacked (:class:`ChunkBatch`); any other model -- the original at the
        pause, a capture, a per-chunk model -- answers each slot on its own frame.  The
        slots then resume, the last parked first (it is live already).
        """

        runner, counters, context = self.runner, self.counters, self.context
        pauses_before = counters.pauses
        slots = runner.batch_begin(context, mask)
        counters.starts += 1
        counters.crossings += 1
        waiting: dict[int, _Waiting] = {}

        def note(slot: int, event: SegmentEvent) -> None:
            if event == SegmentEvent.ERROR:
                raise PhysicsError(
                    f"{self.stage_name}: the runner failed in chunk slot {slot} after "
                    f"{counters.pauses - pauses_before} pause(s) this run: {runner.error(context)}")
            if event != SegmentEvent.NEEDS_PYTHON_KERNEL:
                return
            frame = runner.frame(context)
            counters.crossings += 1
            counters.pauses += 1
            model = kernels.get(frame.kernel)
            if model is None:
                raise PhysicsError(f"{self.stage_name}: the runner paused on {frame.kernel!r}, which is not replaced")
            stacked = getattr(model, "takes_chunk_batches", False)
            batch = frame.batch() if stacked else None
            if batch is not None:
                counters.bytes_copied_in += sum(v.nbytes for v in batch.values())
            parked = ({a.name: np.array(_lanes(a.array, frame.ncol), copy=True) for a in frame.arguments if a.is_input}
                      if self.verify_frames else None)
            empty = frozenset(a.name for a in frame.arguments if a.is_output and _lanes(a.array, frame.ncol).size == 0)
            waiting[slot] = _Waiting(kernel=frame.kernel, token=frame.token, lchnk=frame.lchnk, ncol=frame.ncol,
                                     batch=batch, lanes=_lane_names(frame), empty=empty, parked=parked)

        try:
            for slot in range(slots):
                runner.batch_select(context, slot)
                event = runner.batch_advance(context)
                counters.crossings += 2
                note(slot, event)
            while waiting:
                current, waiting = waiting, {}
                answers: dict[int, dict[str, np.ndarray]] = {}
                for kernel in dict.fromkeys(w.kernel for w in current.values()):
                    model = kernels[kernel]
                    if not getattr(model, "takes_chunk_batches", False):
                        continue
                    group = [slot for slot in sorted(current) if current[slot].kernel == kernel]
                    answers.update(self._answer_chunks(kernel, model, group, current))
                    counters.model_calls += 1
                    counters.calls_by_kernel[kernel] = counters.calls_by_kernel.get(kernel, 0) + 1
                for slot in sorted(current, reverse=True):
                    parked = current[slot]
                    runner.batch_select(context, slot)
                    frame = runner.frame(context)
                    counters.crossings += 2
                    if frame.kernel != parked.kernel or frame.token != parked.token:
                        raise PhysicsError(
                            f"{self.stage_name}: chunk slot {slot} came back paused on {frame.kernel!r} "
                            f"(token {frame.token}), not where it was parked ({parked.kernel!r}, {parked.token})")
                    model = kernels[frame.kernel]
                    # a slot-taking model checked the slot's inputs itself when it made
                    # the slot live, before its original wrote the outputs
                    checked = slot in answers and getattr(model, "takes_slots", False)
                    if parked.parked is not None and not checked:
                        _verify_restored(self.stage_name, slot, frame, parked.parked)
                    self.paused_on = frame
                    if slot in answers:
                        answer = answers[slot]
                    else:
                        if getattr(model, "takes_frame", False):
                            answer = model(frame, runner, context)
                        else:
                            batch = frame.batch()
                            counters.bytes_copied_in += sum(v.nbytes for v in batch.values())
                            answer = model(batch)
                        counters.model_calls += 1
                        counters.calls_by_kernel[frame.kernel] = counters.calls_by_kernel.get(frame.kernel, 0) + 1
                    written = frame.write_back(answer)
                    counters.bytes_copied_out += sum(
                        _live_bytes(frame.argument(name).array, frame.ncol) for name in written)
                    event = runner.resume(context, frame.kernel, frame.token)
                    counters.resumes += 1
                    counters.crossings += 1
                    self.paused_on = None
                    note(slot, event)
            runner.batch_end(context)
            counters.crossings += 1
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise

    def _answer_chunks(self, kernel: str, model: Callable[[ChunkBatch], Mapping[str, Any]], group: list[int],
                       waiting: Mapping[int, _Waiting]) -> dict[int, dict[str, np.ndarray]]:
        """One chunk-batch call for the slots waiting at ``kernel``, its answer split back by slot."""

        parked = [waiting[slot] for slot in group]
        rows = tuple(w.ncol for w in parked)
        lanes = frozenset.intersection(*(w.lanes for w in parked))
        inputs: dict[str, np.ndarray] = {}
        per_chunk: dict[str, tuple[np.ndarray, ...]] = {}
        for name in parked[0].batch:
            values = [w.batch[name] for w in parked]
            if name in lanes:
                inputs[name] = np.concatenate(values, axis=0)
            elif all(np.array_equal(values[0], v) for v in values[1:]):
                inputs[name] = values[0]
            else:
                per_chunk[name] = tuple(values)
        batch = ChunkBatch(kernel=kernel, inputs=inputs, rows=rows, lchnks=tuple(w.lchnk for w in parked),
                           per_chunk=per_chunk, stacked=frozenset(name for name in inputs if name in lanes))
        if getattr(model, "takes_slots", False):
            answer = model(batch, self.runner, self.context, group)       # type: ignore[call-arg]
        else:
            answer = model(batch)
        total = batch.columns
        empty = frozenset.intersection(*(w.empty for w in parked))
        split: dict[int, dict[str, np.ndarray]] = {slot: {} for slot in group}
        for name, value in answer.items():
            value = np.asarray(value)
            if name in empty:
                # no waiting chunk has storage for it: each is handed the empty answer
                if value.size != 0:
                    raise PhysicsError(
                        f"{self.stage_name}: {kernel}'s chunk-batch answer {name} has shape {value.shape}; "
                        f"no waiting chunk has storage for it")
                for slot in group:
                    split[slot][name] = value
                continue
            if name not in lanes or value.ndim == 0 or value.shape[0] != total:
                raise PhysicsError(
                    f"{self.stage_name}: {kernel}'s chunk-batch answer {name} has shape {value.shape}; "
                    f"an output is answered for every stacked column ({total})")
            start = 0
            for slot, count in zip(group, rows):
                split[slot][name] = value[start:start + count]
                start += count
        return split

    def _fail(self, detail: str) -> None:
        """A failure mid-stage: the context is gone and the stage is tainted."""

        self.tainted = detail
        self.paused_on = None
        if self.context is not None:
            try:
                self.runner.destroy(self.context)
            finally:
                self.context = None

    def close(self) -> None:
        """Release the context at finalize; refused while paused."""

        if self.paused_on is not None:
            raise PhysicsError(f"{self.stage_name}: cannot finalize while paused on {self.paused_on.kernel!r}")
        if self.context is not None:
            self.runner.destroy(self.context)
            self.context = None


__all__ = ["ByChunk", "ChunkBatch", "FrameArgument", "KernelFrame", "OriginalAtPause", "OriginalAtPauseByChunk",
           "OriginalByChunk", "OriginalKernel", "SegmentCounters",
           "SegmentEvent", "SegmentRunner", "SegmentedStage"]
