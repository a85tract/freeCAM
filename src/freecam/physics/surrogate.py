"""Stand where mmacro_pcond stands, from Python.

`macrop_driver_tend` is cut at its kernel call, and the two halves are
separate workflow actions (`cam_run1.macro_pre` and `cam_run1.macro_post`).
Between them the routine's whole argument boundary is a StatePool record:
`macro_split.in_*` is what the kernel reads, `macro_split.out_*` what it
returns, `macro_split.ref_*` what the original computed when it is being
shadowed.  A process inserted between the two halves is therefore the kernel,
for that timestep, on every column of every chunk this rank owns.

Three of them ship here, and they are meant to be used in this order:

``IdentityKernel``   copies out_* back onto itself.  With the kernel mode left
                     at 0 the original still computes the answer, so a run
                     with this process inserted must stay bit-for-bit with the
                     oracle.  That is the gate which separates a plumbing
                     mistake from a model error, and nothing after it is
                     trustworthy until it passes.
``ShadowSurrogate``  runs a model and scores it against ref_*, writing
                     nothing.  The integration is still the original's, so a
                     shadowed run is also bit-for-bit -- this is how a
                     surrogate earns a drift record inside a real year.
``MacroSurrogate``   runs a model and returns its answer.  Not bit-for-bit,
                     by construction.

Lanes ``ncol..pcols-1`` of a chunk are padding the routine never fills, and
none of these touch them.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..pi_cam.facade import Physics
from .errors import PhysicsError
from .spec import load_function_spec

FUNCTION = "mmacro_pcond"


def _boundary() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The kernel's own argument names, from the reviewed specification."""

    spec = load_function_spec(FUNCTION)
    reads = tuple(
        item.name for item in spec.arguments
        if item.role in ("structural", "input", "inout")
    )
    returns = tuple(
        [item.name for item in spec.arguments if item.role == "output"]
        + [item.name for item in spec.arguments if item.role == "inout"]
    )
    return reads, returns


INPUTS, RETURNED = _boundary()


class _Lanes:
    """Which (lane, chunk) pairs are live, and how to gather and scatter them.

    A StatePool field is ``(pcols, pver, nchunks)`` or ``(pcols, nchunks)`` or
    ``(nchunks,)``; only lanes ``0..ncol-1`` of each chunk carry a column.  One
    pair of index arrays turns any of those into a flat batch of columns and
    back again, with no Python loop over columns and no copy of the padding.
    """

    def __init__(self, ncol: np.ndarray) -> None:
        counts = np.asarray(ncol, dtype=np.int64).reshape(-1)
        if np.any(counts < 0):
            raise PhysicsError(f"macro_split.in_ncol is negative: {counts.tolist()}")
        self.counts = counts
        self.lane = np.concatenate([np.arange(n) for n in counts]) if counts.size else np.empty(0, np.int64)
        self.chunk = np.repeat(np.arange(counts.size), counts)
        self.columns = int(self.lane.size)

    def gather(self, field: np.ndarray) -> np.ndarray:
        if field.ndim == 3:
            return field[self.lane, :, self.chunk]
        if field.ndim == 2:
            return field[self.lane, self.chunk]
        return field[self.chunk]

    def scatter(self, field: np.ndarray, values: np.ndarray) -> None:
        values = np.asarray(values)
        if field.ndim == 3:
            if values.shape != (self.columns, field.shape[1]):
                raise PhysicsError(
                    f"expected {(self.columns, field.shape[1])}, got {values.shape}"
                )
            field[self.lane, :, self.chunk] = values
        elif field.ndim == 2:
            field[self.lane, self.chunk] = values.reshape(self.columns)
        else:
            field[self.chunk] = values.reshape(self.columns)


class MacroKernel(Physics):
    """Base class: the boundary, the batching, and the write-back."""

    after = "cam_run1.macro_pre"
    reads = tuple(f"macro_split.in_{name}" for name in INPUTS)
    writes = tuple(f"macro_split.out_{name}" for name in RETURNED)

    def columns(self, state: Any) -> tuple["_Lanes", dict[str, np.ndarray]]:
        """One batch of live columns: ``(lanes, {argument: values})``."""

        lanes = _Lanes(np.asarray(state["macro_split.in_ncol"]))
        batch = {
            name: lanes.gather(np.asarray(state[f"macro_split.in_{name}"]))
            for name in INPUTS
        }
        return lanes, batch

    def predict(self, batch: Mapping[str, np.ndarray], columns: int) -> Mapping[str, np.ndarray]:
        """What the kernel returns for this batch; keys are RETURNED."""

        raise NotImplementedError

    def write(self, state: Any, lanes: "_Lanes", answer: Mapping[str, np.ndarray]) -> None:
        missing = [name for name in RETURNED if name not in answer]
        if missing:
            raise PhysicsError(
                f"{type(self).__name__} returned {len(answer)} of {len(RETURNED)} "
                f"values; missing: {', '.join(missing)}"
            )
        for name in RETURNED:
            lanes.scatter(np.asarray(state[f"macro_split.out_{name}"]), answer[name])

    def run(self, state: Any, context: Any) -> None:
        lanes, batch = self.columns(state)
        if lanes.columns == 0:
            return
        self.write(state, lanes, self.predict(batch, lanes.columns))


class IdentityKernel(MacroKernel):
    """Read the kernel's own answer and put it straight back.

    Every value makes the whole round trip -- native chunk to NumPy view to
    batch and back into the lane it came from -- and none of them may change.
    Run it with the kernel mode at 0 and the result has to stay bit-for-bit
    with the oracle; if it does not, the fault is in the plumbing, and no
    surrogate result obtained afterwards would mean anything.
    """

    name = "macro_identity"
    reads = MacroKernel.reads + tuple(f"macro_split.out_{name}" for name in RETURNED)

    def run(self, state: Any, context: Any) -> None:
        lanes = _Lanes(np.asarray(state["macro_split.in_ncol"]))
        if lanes.columns == 0:
            return
        for name in RETURNED:
            field = np.asarray(state[f"macro_split.out_{name}"])
            lanes.scatter(field, lanes.gather(field))


class TorchSurrogate(MacroKernel):
    """A ``torch.nn.Module`` in the kernel's place.

    The module is called once per action with the whole rank's live columns.
    It takes the 35 inputs as a mapping of ``(columns, lev)`` and ``(columns,)``
    arrays and returns the 23 the kernel returns, under the same names -- the
    layout ``examples/generate_mmacro_pcond_dataset.py`` writes as ``input__*``
    and ``output__*``/``updated__*``, so training and substitution share one
    boundary and nothing has to be re-mapped.
    """

    name = "macro_surrogate"

    def __init__(self, model: Any, *, adapter: Callable[..., Any] | None = None) -> None:
        self.model = model
        self.adapter = adapter

    def predict(self, batch: Mapping[str, np.ndarray], columns: int) -> Mapping[str, np.ndarray]:
        import torch

        with torch.inference_mode():
            tensors = {name: torch.from_numpy(np.ascontiguousarray(value))
                       for name, value in batch.items()}
            answer = self.adapter(self.model, tensors) if self.adapter else self.model(tensors)
        return {name: np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value)
                for name, value in dict(answer).items()}


class ShadowSurrogate(TorchSurrogate):
    """Score a model against the original without letting it near the run.

    With the kernel mode at 2 the original computes the answer and publishes it
    in ``ref_*``; this reads it, runs the model on the same inputs, and keeps
    per-argument error statistics.  It writes nothing, so the integration is
    still the original's and a shadowed run is bit-for-bit -- which is what
    makes it safe to leave on for a whole year.
    """

    name = "macro_shadow"
    reads = MacroKernel.reads + tuple(f"macro_split.ref_{name}" for name in RETURNED)
    writes = ()

    def __init__(self, model: Any, *, adapter: Callable[..., Any] | None = None) -> None:
        super().__init__(model, adapter=adapter)
        self.samples = 0
        self.error: dict[str, dict[str, float]] = {}

    def run(self, state: Any, context: Any) -> None:
        lanes, batch = self.columns(state)
        if lanes.columns == 0:
            return
        answer = self.predict(batch, lanes.columns)
        for name in RETURNED:
            if name not in answer:
                continue
            truth = lanes.gather(np.asarray(state[f"macro_split.ref_{name}"]))
            self._accumulate(name, np.asarray(answer[name], dtype=np.float64), truth)
        self.samples += lanes.columns

    def _accumulate(self, name: str, predicted: np.ndarray, truth: np.ndarray) -> None:
        difference = predicted - truth
        record = self.error.setdefault(name, {"n": 0.0, "sum_squared": 0.0, "max_absolute": 0.0,
                                              "truth_sum_squared": 0.0})
        record["n"] += difference.size
        record["sum_squared"] += float(np.sum(difference * difference))
        record["max_absolute"] = max(record["max_absolute"], float(np.max(np.abs(difference))))
        record["truth_sum_squared"] += float(np.sum(truth * truth))

    def drift_report(self) -> dict[str, Any]:
        """Root-mean-square error per returned argument, and relative to its own scale."""

        rows = {}
        for name, record in sorted(self.error.items()):
            count = max(record["n"], 1.0)
            rmse = (record["sum_squared"] / count) ** 0.5
            scale = (record["truth_sum_squared"] / count) ** 0.5
            rows[name] = {
                "rmse": rmse,
                "max_absolute": record["max_absolute"],
                "relative_rmse": rmse / scale if scale > 0 else float("nan"),
            }
        return {"function": FUNCTION, "columns": self.samples, "arguments": rows}


STAGE = "cam_run1.cloud_macro_microphysics"
FIRST_HALF = "cam_run1.macro_pre_leaf"
SECOND_HALF = "cam_run1.macro_post_leaf"


def split_macro_stage(workflow: Any, *, split: bool = True) -> tuple[str, ...]:
    """Swap the whole macrophysics stage for its two halves, or back.

    The stage and its halves are alternatives, never both: leaving the stage
    enabled beside its halves would run the macrophysics twice per timestep.
    Doing it in one call is the only way to be sure of that, so nothing else
    here touches the three actions individually.
    """

    stage = workflow.process(STAGE)
    halves = [workflow.process(FIRST_HALF), workflow.process(SECOND_HALF)]
    if split:
        for half in halves:
            half.enable()
        stage.disable()
        return (FIRST_HALF, SECOND_HALF)
    stage.enable()
    for half in halves:
        half.disable()
    return (STAGE,)


__all__ = [
    "FIRST_HALF", "INPUTS", "RETURNED", "SECOND_HALF", "STAGE",
    "IdentityKernel", "MacroKernel", "ShadowSurrogate", "TorchSurrogate", "split_macro_stage",
]
