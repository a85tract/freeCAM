from pathlib import Path

import numpy as np
import pytest

from pycam_sima.model import (
    CAMDriver,
    DriverState,
    KesslerSchemePlan,
    ModelConfig,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
)
from pycam_sima.model.driver import (
    INITIAL_PREP_PHASES,
)
from pycam_sima.model.contracts import default_contracts
from pycam_sima.model.grid import local_elements


PROJECT = Path(__file__).resolve().parents[2]
ATM_IN = (
    Path("/glade/derecho/scratch/ruitong/pycam-sima/")
    / "FKESSLER_ne3pg3_gnu_24x50/FKESSLER_ne3pg3_gnu_24x50/run/atm_in"
)


class FixedCaseComm:
    rank = 0
    size = 24

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value]

    def allgather(self, value):
        return [
            [item.global_id for item in local_elements(rank)] for rank in range(24)
        ]

    def barrier(self):
        return None


@pytest.fixture(scope="module")
def session():
    config = ModelConfig.from_yaml(
        PROJECT / "configs/fkessler_model.yaml"
    ).with_overrides(atm_in=str(ATM_IN), history_enabled=False)
    return CAMDriver(config, run_dir=PROJECT, comm=FixedCaseComm()).start()


def test_initialize_is_python_owned_and_zero_native_calls(session):
    assert session.state == DriverState.INITIALIZED
    assert session.backend.call_count == 0
    assert session.backend._abi_checked is False
    assert all(
        device._abi_checked is False
        for device in session.backend.devices.devices.values()
    )
    assert len(session.pool.inventory()) == len(default_contracts()) == 222
    assert all(item["owner"] == "python" for item in session.pool.inventory())
    assert np.isfinite(session.get_field("air_temperature")).all()
    assert set(session._scheme_handlers()) == set(session.scheme_names)


def test_original_kessler_device_preserves_addresses(session):
    before = session.pool.pointer_records()
    session.run_scheme("kessler", group=PHYSICS_BEFORE_COUPLER)
    session.pool.assert_pointer_stability(before)
    assert session.backend.call_count == 1
    assert session.backend.devices.devices["kessler"]._abi_checked is True


def test_group_execution_follows_a_cross_group_move() -> None:
    plan = KesslerSchemePlan.default()
    plan.move("kessler", to_group=PHYSICS_AFTER_COUPLER, unsafe=True)
    driver = object.__new__(CAMDriver)
    driver.scheme_plan = plan
    calls = []
    driver.run_scheme = calls.append

    CAMDriver.run_scheme_group(driver, PHYSICS_BEFORE_COUPLER)
    assert "physics_before_coupler.kessler" not in calls
    CAMDriver.run_scheme_group(driver, PHYSICS_AFTER_COUPLER)
    assert calls[-1] == "physics_before_coupler.kessler"


def test_step_uses_complete_python_orchestrator(session, monkeypatch):
    session.run_scheme("kessler_update", group=PHYSICS_BEFORE_COUPLER)
    session.state = DriverState.PRIMED
    calls = []

    def record_phase(name):
        calls.append(f"phase:{name}")

    def record_scheme(name, *, group=None):
        del group
        calls.append(f"scheme:{name}")

    monkeypatch.setattr(session, "run_phase", record_phase)
    monkeypatch.setattr(session, "run_scheme", record_scheme)
    session.step()
    expected = [
        f"scheme:{scheme.key}"
        for scheme in session.scheme_plan.active(PHYSICS_AFTER_COUPLER)
    ]
    expected.extend(
        ["phase:physics_to_dynamics", "phase:scale_physics_forcing"]
    )
    for nsubstep in (1, 2):
        expected.append(
            "phase:apply_cam_forcing_substep_2"
            if nsubstep == 2
            else "phase:apply_cam_forcing"
        )
        for rstep in range(1, 4):
            if rstep > 1:
                expected.append("phase:update_time_levels")
            expected.extend(
                f"phase:{name}"
                for name in (
                    "initialize_prim_step", "se_type4_rk",
                    "advance_hyperviscosity", "update_surface_dry_air_pressure",
                    "advance_se_tracers",
                )
            )
        expected.extend(
            f"phase:{name}"
            for name in (
                "advance_fvm_tracers", "vertical_remap_se", "vertical_remap_fvm"
            )
        )
        if nsubstep == 2:
            expected.append("phase:compute_final_omega")
        expected.append("phase:update_time_levels")
    expected.append("phase:physics_timestep_final")
    expected.extend(f"phase:{name}" for name in INITIAL_PREP_PHASES)
    expected.extend(
        f"scheme:{scheme.key}"
        for scheme in session.scheme_plan.active(PHYSICS_BEFORE_COUPLER)
    )
    assert calls == expected
    assert calls.count("phase:se_type4_rk") == 6
    assert calls.count("phase:advance_fvm_tracers") == 2
    assert calls.count("phase:vertical_remap_se") == 2
    assert calls.count("phase:vertical_remap_fvm") == 2
    assert calls.count("phase:compute_final_omega") == 1
    assert session.state == DriverState.RUNNING
