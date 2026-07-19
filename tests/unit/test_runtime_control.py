import numpy as np
import pytest

from pycam_sima import DEFAULT_STEP_PHASES, RuntimeOptions, StepPlan
from pycam_sima.config import CaseConfig
from pycam_sima.driver import FKesslerDriver
from pycam_sima.dynamics import IdentityDynamics
from pycam_sima.mpi_runtime import SerialComm
from pycam_sima.native import RecordingBackend


def make_driver(
    *, options: RuntimeOptions | None = None, step_plan: StepPlan | None = None
) -> FKesslerDriver:
    config = CaseConfig.from_yaml("configs/fkessler_ne3pg3.yaml")
    return FKesslerDriver(
        config,
        SerialComm(),
        backend=RecordingBackend(),
        dynamics=IdentityDynamics(),
        options=options,
        step_plan=step_plan,
    )


def test_step_plan_lists_effective_phases_and_protects_required_phases() -> None:
    options = RuntimeOptions(timestep_seconds=1800, dynamics=False)
    plan = StepPlan.default()

    rows = plan.describe(options)
    assert [row["name"] for row in rows] == list(plan.names)
    assert next(row for row in rows if row["name"] == "se_dynamics")["enabled"] is False
    assert plan.sequence_safe

    plan.disable("kessler_after_coupler")
    assert not plan.phase("kessler_after_coupler").enabled
    assert plan.sequence_safe

    with pytest.raises(ValueError, match="required by the validated model-step lifecycle"):
        plan.disable("advance_clock")
    plan.disable("advance_clock", unsafe=True)
    assert not plan.sequence_safe


def test_reordering_requires_explicit_unsafe_acknowledgement() -> None:
    plan = StepPlan.default()
    with pytest.raises(ValueError, match="requires unsafe=True"):
        plan.move("se_dynamics", before="physics_to_dynamics")

    plan.move("se_dynamics", before="physics_to_dynamics", unsafe=True)
    assert plan.names.index("se_dynamics") < plan.names.index("physics_to_dynamics")
    assert not plan.sequence_safe

    plan.reset()
    assert plan.names == tuple(phase.name for phase in DEFAULT_STEP_PHASES)
    assert plan.sequence_safe


def test_step_uses_the_visible_plan_order() -> None:
    plan = StepPlan.default()
    plan.move("se_dynamics", before="physics_to_dynamics", unsafe=True)
    driver = make_driver(step_plan=plan)
    seen: list[str] = []
    driver.observe("phase_begin:*", lambda context: seen.append(context.task_name))

    driver.initialize()
    driver.step()

    assert seen == list(plan.names)
    assert driver.clock.step == 1


def test_physics_and_dynamics_can_be_disabled_independently() -> None:
    options = RuntimeOptions(
        timestep_seconds=1800,
        physics_before=False,
        physics_after=False,
        dynamics=False,
    )
    driver = make_driver(options=options)

    driver.initialize()
    driver.step()

    assert driver.backend.calls == [
        "lifecycle:register",
        "lifecycle:initialize",
        "lifecycle:timestep_initial",
        "lifecycle:timestep_final",
        "lifecycle:timestep_initial",
    ]
    assert driver.dynamics.calls == [
        "initialize",
        "dynamics_to_physics",
        "physics_to_dynamics",
        "dynamics_to_physics",
    ]


def test_parameters_are_live_views_and_timestep_changes_apply_at_next_step() -> None:
    driver = make_driver()
    driver.allocate_minimal_state(ncol=2)

    driver.parameters.surface_reference_pressure = 98_500.0
    driver.parameters.dycore_energy_adjustment = False
    driver.parameters.constituent_minimum_values = (2.0e-12, 3.0e-12, 4.0e-12)

    assert driver.pool["surface_reference_pressure"][0] == 98_500.0
    assert driver.pool["flag_for_dycore_energy_consistency_adjustment"][0] == 0
    np.testing.assert_array_equal(
        driver.pool["ccpp_constituent_minimum_values"],
        (2.0e-12, 3.0e-12, 4.0e-12),
    )

    driver.initialize()
    driver.options.timestep_seconds = 900
    driver.step()

    assert driver.clock.dt_seconds == 900
    assert driver.clock.elapsed_seconds == 900
    assert driver.pool["timestep_for_physics"][0] == 900

    driver.parameters.timestep_seconds = 600
    assert driver.clock.dt_seconds == 600
    assert driver.pool["timestep_for_physics"][0] == 600


def test_invalid_runtime_parameter_is_rejected_without_changing_the_value() -> None:
    driver = make_driver()
    with pytest.raises(ValueError, match="positive"):
        driver.parameters.timestep_seconds = 0
    assert driver.parameters.timestep_seconds == 1800

    with pytest.raises(RuntimeError, match="allocate_minimal_state"):
        _ = driver.parameters.surface_reference_pressure
