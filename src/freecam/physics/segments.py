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

from dataclasses import dataclass, field
from enum import IntEnum
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
    write-back and one resume.
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
            array = argument.array
            batch[argument.name] = array[:self.ncol].copy() if array.ndim else array.copy()
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
            target = argument.array[:self.ncol] if argument.array.ndim else argument.array
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


class SegmentRunner(Protocol):
    """What the image (or a fake) offers for one stage.

    The runner owns a rank-local context per stage.  ``start`` runs the
    original code from the top of the stage with ``mask`` naming the kernels
    Python computes; ``frame`` describes the pause ``start``/``resume``
    reported; ``resume`` continues past the kernel named.  ``reset`` returns
    a context to idle after a failure; ``destroy`` frees it.
    """

    def create(self, stage: str) -> int: ...
    def start(self, context: int, mask: Mapping[str, bool]) -> SegmentEvent: ...
    def frame(self, context: int) -> KernelFrame: ...
    def resume(self, context: int, kernel: str, token: int) -> SegmentEvent: ...
    def error(self, context: int) -> str: ...
    def reset(self, context: int) -> None: ...
    def destroy(self, context: int) -> None: ...


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

    @property
    def idle(self) -> bool:
        return self.paused_on is None and self.tainted is None

    def run(self, kernels: Mapping[str, Callable[..., Mapping[str, Any]] | None]) -> None:
        """One step of the stage: the original Fortran with ``kernels`` at their pauses."""

        if self.tainted is not None:
            raise PhysicsError(
                f"{self.stage_name}: a model left the stage tainted and it cannot run "
                f"again:\n{self.tainted}")
        if self.paused_on is not None:
            raise PhysicsError(
                f"{self.stage_name}: still paused on {self.paused_on.kernel!r}; a step "
                f"cannot start inside another")
        mask = {name: kernel is not None for name, kernel in kernels.items()}
        if not any(mask.values()):
            raise PhysicsError(
                f"{self.stage_name}: nothing is replaced; run the original stage whole")
        if self.context is None:
            self.context = self.runner.create(self.stage_name)
            self.counters.crossings += 1
        self.generation += 1
        counters = self.counters
        event = self.runner.start(self.context, mask)
        counters.starts += 1
        counters.crossings += 1
        try:
            while event != SegmentEvent.DONE:
                if event == SegmentEvent.ERROR:
                    detail = self.runner.error(self.context)
                    raise PhysicsError(f"{self.stage_name}: the runner failed: {detail}")
                frame = self.runner.frame(self.context)
                counters.crossings += 1
                counters.pauses += 1
                self.paused_on = frame
                model = kernels.get(frame.kernel)
                if model is None:
                    raise PhysicsError(
                        f"{self.stage_name}: the runner paused on {frame.kernel!r}, which "
                        f"is not replaced")
                batch = frame.batch()
                counters.bytes_copied_in += sum(v.nbytes for v in batch.values())
                answer = model(batch)
                counters.model_calls += 1
                counters.calls_by_kernel[frame.kernel] = counters.calls_by_kernel.get(frame.kernel, 0) + 1
                written = frame.write_back(answer)
                counters.bytes_copied_out += sum(
                    frame.argument(name).array[:frame.ncol].nbytes for name in written)
                event = self.runner.resume(self.context, frame.kernel, frame.token)
                counters.resumes += 1
                counters.crossings += 1
                self.paused_on = None
        except BaseException as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            raise

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


__all__ = ["FrameArgument", "KernelFrame", "OriginalKernel", "SegmentCounters", "SegmentEvent",
           "SegmentRunner", "SegmentedStage"]
