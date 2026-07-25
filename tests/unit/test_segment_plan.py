from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycam_sima import (
    BranchSpec,
    FieldEdit,
    MoveScheme,
    ObserveFields,
    PrepareInitialStep,
    RunPhase,
    RunScheme,
    RunSchemeGroup,
    RunSteps,
    SegmentPlan,
    SetSchemeEnabled,
)
from pycam_sima.model import CCPPSuitePlan, StatePool, execute_segment_plan
from pycam_sima.model.comm import SerialComm
from pycam_sima.model.grid import dimensions_for_rank


ROOT = Path(__file__).resolve().parents[2]
KESSLER_SUITE = (
    ROOT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


class _RecordingDriver:
    phase_names = ("dynamics_to_physics", "physics_timestep_initial")

    def __init__(self) -> None:
        self.pool = StatePool(dimensions_for_rank(0, 24))
        self.pool.set("air_temperature", 240.0)
        self.scheme_plan = CCPPSuitePlan.from_xml(KESSLER_SUITE)
        self.comm = SerialComm()
        self.clock = SimpleNamespace(nstep=0)
        self.backend = SimpleNamespace(call_count=0)
        self._last_phase = None
        self._last_scheme = None
        self._last_scheme_group = None
        self.calls = []

    def prepare_initial_step(self) -> None:
        self.calls.append(("prepare",))

    def run_phase(self, name: str) -> None:
        self.calls.append(("phase", name))
        self._last_phase = name

    def run_scheme(self, name: str, *, group=None) -> None:
        scheme = self.scheme_plan.scheme(name, group=group)
        self.calls.append(("scheme", scheme.key))
        self._last_scheme = scheme.key
        self._last_scheme_group = scheme.group
        if scheme.name == "kessler":
            self.backend.call_count += 1

    def run_scheme_group(self, group: str) -> None:
        self.calls.append(("group", group))

    def step(self) -> None:
        self.calls.append(("step",))
        self.clock.nstep += 1


def test_segment_plan_json_round_trip_covers_every_action() -> None:
    plan = SegmentPlan(
        "complete",
        (
            PrepareInitialStep(),
            SetSchemeEnabled("kessler", False),
            MoveScheme("kessler", to_group="physics_after_coupler"),
            FieldEdit("air_temperature", "add", 1.0),
            RunScheme("kessler", group="physics_after_coupler"),
            RunSchemeGroup("physics_after_coupler"),
            RunPhase("dynamics_to_physics"),
            RunSteps(2),
            ObserveFields(("air_temperature",), ("mean",)),
        ),
        unsafe=True,
    )

    assert SegmentPlan.from_mapping(plan.as_dict()) == plan
    assert plan.step_count == 2


def test_legacy_branch_converts_to_edit_then_steps() -> None:
    branch = BranchSpec(
        "legacy",
        steps=3,
        disable_schemes=("kessler",),
        field_edits=(FieldEdit("air_temperature", "add", 2.0),),
    )

    plan = branch.to_segment_plan()

    assert plan.unsafe is True
    assert isinstance(plan.actions[0], SetSchemeEnabled)
    assert isinstance(plan.actions[1], FieldEdit)
    assert plan.actions[-1] == RunSteps(3)

    unsafe_edit = BranchSpec(
        "unsafe-edit",
        steps=0,
        field_edits=(FieldEdit("earth_radius", "set", 1.0, unsafe=True),),
    )
    assert unsafe_edit.to_segment_plan().unsafe is True


def test_segment_executor_preserves_order_and_observes_fields() -> None:
    driver = _RecordingDriver()
    plan = SegmentPlan(
        "ordered",
        (
            FieldEdit("air_temperature", "add", 1.0),
            RunScheme("kessler", group="physics_before_coupler"),
            RunPhase("dynamics_to_physics"),
            RunSteps(2),
            ObserveFields(("air_temperature",)),
        ),
        unsafe=True,
    )

    trace = execute_segment_plan(driver, plan)

    assert driver.calls == [
        ("scheme", driver.scheme_plan.scheme("kessler").key),
        ("phase", "dynamics_to_physics"),
        ("step",),
        ("step",),
    ]
    assert driver.clock.nstep == 2
    assert driver.backend.call_count == 1
    assert trace[1]["native_calls_delta"] == 1
    observation = trace[-1]["observations"][0]
    assert observation["global"] == {
        "min": 241.0,
        "max": 241.0,
        "mean": 241.0,
    }


def test_invalid_plan_is_rejected_before_field_mutation() -> None:
    driver = _RecordingDriver()
    plan = SegmentPlan(
        "invalid",
        (
            FieldEdit("air_temperature", "add", 1.0),
            RunPhase("not-a-phase"),
        ),
        unsafe=True,
    )

    with pytest.raises(ValueError, match="unknown model phase"):
        execute_segment_plan(driver, plan)
    assert np.all(driver.pool.get("air_temperature") == 240.0)


def test_empty_plan_is_a_valid_no_op() -> None:
    driver = _RecordingDriver()

    trace = execute_segment_plan(driver, SegmentPlan("empty"))

    assert trace == ()
    assert driver.calls == []
    assert driver.clock.nstep == 0


def test_segment_plan_rejects_unknown_json_action_type() -> None:
    with pytest.raises(ValueError, match="unknown segment action type"):
        SegmentPlan.from_mapping(
            {
                "schema_version": 1,
                "name": "invalid-action",
                "unsafe": True,
                "actions": [{"type": "not_an_action"}],
            }
        )


@pytest.mark.parametrize(
    ("action", "exception", "message"),
    (
        (RunPhase("not-a-phase"), ValueError, "unknown model phase"),
        (
            RunScheme("not-a-scheme"),
            ValueError,
            "unknown scheme",
        ),
        (RunSchemeGroup("not-a-group"), ValueError, "unknown scheme group"),
        (ObserveFields(("not-a-field",)), KeyError, "unknown state field"),
    ),
)
def test_segment_plan_rejects_unknown_action_names(
    action, exception, message
) -> None:
    with pytest.raises(exception, match=message):
        execute_segment_plan(
            _RecordingDriver(),
            SegmentPlan("bad-name", (action,), unsafe=True),
        )


@pytest.mark.parametrize(
    "action",
    (
        RunPhase("dynamics_to_physics"),
        RunScheme("kessler", group="physics_before_coupler"),
        RunSchemeGroup("physics_before_coupler"),
    ),
)
def test_granular_model_actions_require_unsafe(action) -> None:
    driver = _RecordingDriver()

    with pytest.raises(ValueError, match="unsafe=True"):
        execute_segment_plan(driver, SegmentPlan("safe", (action,)))
