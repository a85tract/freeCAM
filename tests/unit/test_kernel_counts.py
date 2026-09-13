"""Kernel execution counters: slot assignment, paired context, step attribution, reduction."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from freecam.pi_cam.driver import PICAMDriver
from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.kernel_counts import (KernelCounters, SLOT_FINALIZE, SLOT_INITIALIZATION,
                                          SLOT_RUN_UNATTRIBUTED, STEP_SENTINEL)


def _counters(kernels=4, slots=8) -> tuple[KernelCounters, SimpleNamespace]:
    state = SimpleNamespace(context=0)
    table = np.zeros((16, slots), dtype=np.int64)

    def setter(slot: int) -> int:
        previous, state.context = state.context, slot if 0 <= slot < slots else 0
        return previous

    counters = KernelCounters(table=table, kernels=kernels,
                              set_context=setter, get_context=lambda: state.context)
    return counters, state


def test_slots_are_assigned_deterministically_and_overflow_is_refused() -> None:
    counters, _ = _counters(slots=8)
    counters.assign_slots(["cam_run1.b", "cam_run1.a", "cam_run1.b"])
    assert counters.slot_of_action == {"cam_run1.a": 3, "cam_run1.a_python": 3,
                                       "cam_run1.b": 4, "cam_run1.b_python": 4}
    assert counters.slot_names[3] == "cam_run1.a" and counters.slot_names[0] == "initialization"
    with pytest.raises(PICAMConfigurationError, match="do not fit"):
        counters.assign_slots([f"cam_run1.p{i}" for i in range(9)])


def test_a_python_stage_wrapper_and_its_native_action_share_one_canonical_slot() -> None:
    counters, state = _counters(slots=8)
    counters.assign_slots(["cam_run1.cloud_python_stage_python", "cam_run1.a"])
    assert counters.slot_names[4] == "cam_run1.cloud_python_stage"
    outer = counters.enter_action("cam_run1.cloud_python_stage_python")
    inner = counters.enter_action("cam_run1.cloud_python_stage")   # run_action inside the class
    assert state.context == 4                                       # same slot, not unattributed
    counters.restore(inner)
    counters.restore(outer)
    assert state.context == SLOT_INITIALIZATION


def test_enter_and_restore_pair_and_nest_and_unknown_actions_stay_unattributed() -> None:
    counters, state = _counters()
    counters.assign_slots(["cam_run1.a", "cam_run1.b"])
    outer = counters.enter_action("cam_run1.a")
    assert (outer, state.context) == (SLOT_INITIALIZATION, 3)
    inner = counters.enter_action("cam_run1.b")
    assert (inner, state.context) == (3, 4)
    counters.restore(inner)
    assert state.context == 3                          # the parent's slot resumes
    counters.restore(outer)
    assert state.context == SLOT_INITIALIZATION
    unknown = counters.enter_action("cam_run9.mystery")
    assert state.context == SLOT_RUN_UNATTRIBUTED      # never guessed, never misattributed
    counters.restore(unknown)


def test_the_driver_restores_the_context_even_when_the_action_fails() -> None:
    counters, state = _counters()
    counters.assign_slots(["cam_run1.a"])

    @contextmanager
    def region(_name):
        yield

    calls = []

    def execute_action(action):
        calls.append((action.qualified_name, state.context))
        if action.qualified_name == "cam_run1.boom":
            raise RuntimeError("the action failed")
        return "trace"

    fake = SimpleNamespace(kernel_counters=counters, profiler=SimpleNamespace(region=region),
                           _execute_action=execute_action)
    action = SimpleNamespace(qualified_name="cam_run1.a", operation="a")
    assert PICAMDriver._execute(fake, action) == "trace"
    assert calls == [("cam_run1.a", 3)] and state.context == SLOT_INITIALIZATION
    with pytest.raises(RuntimeError):
        PICAMDriver._execute(fake, SimpleNamespace(qualified_name="cam_run1.boom", operation="boom"))
    assert state.context == SLOT_INITIALIZATION        # restored despite the exception
    fake.kernel_counters = None
    assert PICAMDriver._execute(fake, action) == "trace"
    assert state.context == SLOT_INITIALIZATION


def test_step_tracking_records_first_and_last_step_per_kernel_and_context() -> None:
    counters, _ = _counters(kernels=3)
    counters.begin_step_tracking()
    counters.table[0, 3] += 5                          # kernel 0 in context 3, step 1
    counters.note_step(1)
    counters.note_step(2)                              # nothing moved
    counters.table[0, 4] += 1                          # kernel 0 in context 4, step 3
    counters.table[0, 3] += 1                          # and again in context 3
    counters.table[2, 0] += 7                          # kernel 2 first moves in step 3
    counters.note_step(3)
    # spans are per [kernel, context]: context 3 spans 1..3, context 4 only step 3
    assert counters.first_step[0, 3] == 1 and counters.last_step[0, 3] == 3
    assert counters.first_step[0, 4] == 3 and counters.last_step[0, 4] == 3
    assert counters.first_step[1].tolist() == [STEP_SENTINEL] * 8
    assert counters.first_step[2, 0] == 3 and counters.last_step[2, 0] == 3


class _SingleRankWorld:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Reduce(self, send, recv, op=None, root=0):
        np.copyto(recv, send)


def test_reduction_and_the_observation_record_report_per_context_counts() -> None:
    counters, _ = _counters(kernels=4)
    counters.assign_slots(["cam_run1.a", "cam_run1.b"])
    counters.begin_step_tracking()
    counters.table[1, 3] = 100                         # kernel 1 in process a
    counters.table[1, 0] = 2                           # and twice during initialization
    counters.table[3, 4] = 7                           # kernel 3 in process b
    counters.note_step(1)
    ops = SimpleNamespace(SUM=None, MIN=None, MAX=None)
    reduced = counters.reduce(_SingleRankWorld(), operations=ops)
    record = counters.observation_record(reduced, [
        {"index": 1, "qualified": "m::one"}, {"index": 3, "qualified": "m::three"},
        {"index": 2, "qualified": "m::silent"},
    ])
    rows = {row["qualified"]: row for row in record["kernels"]}
    assert rows["m::one"]["calls_total"] == 102
    assert rows["m::one"]["calls_by_context"] == {"initialization": 2, "cam_run1.a": 100}
    assert rows["m::one"]["steps_by_context"]["cam_run1.a"] == [1, 1]
    assert rows["m::three"]["calls_by_context"] == {"cam_run1.b": 7}
    # zero calls stay zero: an uncalled instrumented kernel is reported, never dropped
    assert rows["m::silent"]["calls_total"] == 0 and rows["m::silent"]["calls_by_context"] == {}
    assert rows["m::silent"]["first_step"] == STEP_SENTINEL
    assert rows["m::one"]["ranks_with_calls"] == 1 and rows["m::one"]["rank_calls_max"] == 102
    assert record["slot_names"]["1"] == "finalize" and record["mpi_ranks"] == 1
    assert SLOT_FINALIZE == 1
