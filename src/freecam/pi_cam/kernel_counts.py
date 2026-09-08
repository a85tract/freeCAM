"""The kernel execution counters of a counting image, read and attributed from Python.

A counting image routes every instrumented kernel call through a three-
instruction trampoline that increments ``counts[kernel_index][context]``
(``native/pi_cam/pycam_kcount.c``).  Python owns the context: the driver sets
the slot when a workflow action starts and restores the previous slot when it
ends -- paired operations, exception-safe, nesting naturally into a stack --
so a shared kernel's calls are counted separately per process.  Slot 0 is
initialization/unattributed (the table's default before the driver installs
the counters), slot 1 is finalize, slot 2 catches a run-phase action without
an assigned slot, and the workflow actions take the remaining slots in sorted
order, identically on every rank.

The counters are plain per-rank increments, valid only for the admitted
single-threaded-per-rank configuration; the CLI refuses to observe with
OpenMP threads.  Reading is zero-copy (a NumPy view over the table); nothing
here touches the model's hot path.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from .errors import PICAMConfigurationError

SLOT_INITIALIZATION = 0
SLOT_FINALIZE = 1
SLOT_RUN_UNATTRIBUTED = 2
RESERVED_SLOTS = {
    SLOT_INITIALIZATION: "initialization",
    SLOT_FINALIZE: "finalize",
    SLOT_RUN_UNATTRIBUTED: "run-unattributed",
}
FIRST_ACTION_SLOT = 3
STEP_SENTINEL = -1


@dataclass
class KernelCounters:
    """One rank's view of the counting table, with process-slot attribution."""

    table: np.ndarray                                   # int64 [max_kernels, slots], zero-copy
    kernels: int                                        # instrumented candidate indices are < this
    set_context: Callable[[int], int]                   # returns the previous slot
    get_context: Callable[[], int]
    slot_names: dict[int, str] = field(default_factory=lambda: dict(RESERVED_SLOTS))
    slot_of_action: dict[str, int] = field(default_factory=dict)
    first_step: np.ndarray | None = None
    last_step: np.ndarray | None = None
    _step_baseline: np.ndarray | None = None

    @classmethod
    def from_library(cls, library: Any) -> "KernelCounters | None":
        """The counters of a loaded image, or None when it carries no trampolines."""

        try:
            info = library.pycam_kcount_info_v1
            data = library.pycam_kcount_data_v1
            setter = library.pycam_kcount_context_set_v1
            getter = library.pycam_kcount_context_get_v1
        except AttributeError:
            return None
        info.restype = ctypes.c_int32
        info.argtypes = [ctypes.POINTER(ctypes.c_int32)] * 3
        data.restype = ctypes.POINTER(ctypes.c_int64)
        setter.restype = ctypes.c_int32
        setter.argtypes = [ctypes.c_int32]
        getter.restype = ctypes.c_int32
        kernels, slots, max_kernels = (ctypes.c_int32(0) for _ in range(3))
        if info(ctypes.byref(kernels), ctypes.byref(slots), ctypes.byref(max_kernels)) != 0:
            return None
        if kernels.value <= 0:
            return None
        table = np.ctypeslib.as_array(data(), shape=(max_kernels.value, slots.value))
        return cls(table=table, kernels=int(kernels.value),
                   set_context=lambda slot: int(setter(int(slot))),
                   get_context=lambda: int(getter()))

    @property
    def slots(self) -> int:
        return int(self.table.shape[1])

    def assign_slots(self, action_ids: list[str]) -> None:
        """One slot per workflow action, in sorted order: identical on every rank."""

        ordered = sorted(set(action_ids))
        if FIRST_ACTION_SLOT + len(ordered) > self.slots:
            raise PICAMConfigurationError(
                f"{len(ordered)} workflow actions do not fit the counting table's "
                f"{self.slots} slots ({FIRST_ACTION_SLOT} reserved)")
        for offset, action_id in enumerate(ordered):
            slot = FIRST_ACTION_SLOT + offset
            self.slot_of_action[action_id] = slot
            self.slot_names[slot] = action_id

    def enter_action(self, qualified_name: str) -> int:
        """Set the action's slot; returns the previous slot for the paired restore."""

        return self.set_context(self.slot_of_action.get(qualified_name, SLOT_RUN_UNATTRIBUTED))

    def restore(self, previous: int) -> None:
        self.set_context(previous)

    # ------------------------------------------------------------------ #
    # Step attribution: first and last executing step, tracked by diffing
    # per-kernel totals at step boundaries (the hot path stays untouched).
    # ------------------------------------------------------------------ #

    def rank_totals(self) -> np.ndarray:
        return self.table[: self.kernels].sum(axis=1)

    def begin_step_tracking(self) -> None:
        self.first_step = np.full(self.kernels, STEP_SENTINEL, dtype=np.int64)
        self.last_step = np.full(self.kernels, STEP_SENTINEL, dtype=np.int64)
        self._step_baseline = self.rank_totals().copy()

    def note_step(self, model_step: int) -> None:
        if self._step_baseline is None:
            return
        totals = self.rank_totals()
        moved = totals > self._step_baseline
        if moved.any():
            assert self.first_step is not None and self.last_step is not None
            self.first_step[moved & (self.first_step == STEP_SENTINEL)] = model_step
            self.last_step[moved] = model_step
        self._step_baseline = totals.copy()

    # ------------------------------------------------------------------ #
    # Global reduction: collective, so a completed reduction proves every
    # rank contributed; nothing is inferred from missing data.
    # ------------------------------------------------------------------ #

    def reduce(self, world: Any, operations: Any = None) -> "dict[str, Any] | None":
        """Global statistics on rank 0 (None elsewhere); collective on ``world``."""

        if operations is None:
            from mpi4py import MPI as operations

        MPI = operations
        active = np.ascontiguousarray(self.table[: self.kernels])
        table_sum = np.zeros_like(active)
        world.Reduce(active, table_sum, op=MPI.SUM, root=0)
        totals = self.rank_totals()
        totals_sum = np.zeros_like(totals)
        totals_min = np.zeros_like(totals)
        totals_max = np.zeros_like(totals)
        world.Reduce(totals, totals_sum, op=MPI.SUM, root=0)
        world.Reduce(totals, totals_min, op=MPI.MIN, root=0)
        world.Reduce(totals, totals_max, op=MPI.MAX, root=0)
        called = (totals > 0).astype(np.int64)
        ranks_with_calls = np.zeros_like(called)
        world.Reduce(called, ranks_with_calls, op=MPI.SUM, root=0)
        first = self.first_step if self.first_step is not None \
            else np.full(self.kernels, STEP_SENTINEL, dtype=np.int64)
        last = self.last_step if self.last_step is not None \
            else np.full(self.kernels, STEP_SENTINEL, dtype=np.int64)
        first_masked = np.where(first == STEP_SENTINEL, np.iinfo(np.int64).max, first)
        first_min = np.zeros_like(first_masked)
        last_max = np.zeros_like(last)
        world.Reduce(first_masked, first_min, op=MPI.MIN, root=0)
        world.Reduce(last, last_max, op=MPI.MAX, root=0)
        if world.Get_rank() != 0:
            return None
        return {
            "mpi_ranks": int(world.Get_size()),
            "table_sum": table_sum,
            "totals_sum": totals_sum,
            "totals_min": totals_min,
            "totals_max": totals_max,
            "ranks_with_calls": ranks_with_calls,
            "first_step": np.where(first_min == np.iinfo(np.int64).max, STEP_SENTINEL, first_min),
            "last_step": last_max,
        }

    def observation_record(self, reduced: Mapping[str, Any],
                           instrumented: list[Mapping[str, Any]]) -> dict[str, Any]:
        """The raw per-run observation: compact, no per-call event stream."""

        by_index = {int(row["index"]): row for row in instrumented}
        kernels = []
        table = reduced["table_sum"]
        for index in sorted(by_index):
            row = by_index[index]
            slots = {self.slot_names.get(slot, f"slot-{slot}"): int(count)
                     for slot, count in enumerate(table[index]) if count}
            kernels.append({
                "index": index,
                "qualified": row["qualified"],
                "calls_total": int(reduced["totals_sum"][index]),
                "calls_by_context": slots,
                "first_step": int(reduced["first_step"][index]),
                "last_step": int(reduced["last_step"][index]),
                "ranks_with_calls": int(reduced["ranks_with_calls"][index]),
                "rank_calls_min": int(reduced["totals_min"][index]),
                "rank_calls_max": int(reduced["totals_max"][index]),
            })
        return {
            "mpi_ranks": reduced["mpi_ranks"],
            "slot_names": {str(slot): name for slot, name in sorted(self.slot_names.items())},
            "kernels": kernels,
        }


__all__ = ["KernelCounters", "RESERVED_SLOTS", "SLOT_FINALIZE", "SLOT_INITIALIZATION",
           "SLOT_RUN_UNATTRIBUTED", "STEP_SENTINEL"]
