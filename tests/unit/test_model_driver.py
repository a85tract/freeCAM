from pathlib import Path

import numpy as np
import pytest

from pycam_sima.model import (
    CAMDriver,
    CCPPSuitePlan,
    DriverState,
    ModelConfig,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
)
from pycam_sima.model.driver import (
    INITIAL_PREP_PHASES,
)
from pycam_sima.model.clock import ModelClock
from pycam_sima.model.contracts import default_contracts
from pycam_sima.model.grid import local_elements
import pycam_sima.model.driver as driver_module


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
    assert len(session.pool.inventory()) == len(
        session.state_schema.pool_contracts()
    )
    assert len(session.pool.inventory()) > len(default_contracts())
    assert all(item["owner"] == "python" for item in session.pool.inventory())
    assert np.isfinite(session.get_field("air_temperature")).all()
    assert {
        item.name for item in session.scheme_plan.schemes
    } <= set(session.processes.process_names)


def test_original_kessler_device_preserves_addresses(session):
    before = session.pool.pointer_records()
    session.run_scheme("kessler", group=PHYSICS_BEFORE_COUPLER)
    session.pool.assert_pointer_stability(before)
    assert session.backend.call_count == 1
    assert session.backend.devices.devices["kessler"]._abi_checked is True


def test_group_execution_follows_a_cross_group_move() -> None:
    plan = CCPPSuitePlan.from_xml(
        PROJECT
        / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
    )
    kessler_key = plan.scheme("kessler").key
    plan.move("kessler", to_group=PHYSICS_AFTER_COUPLER, unsafe=True)
    driver = object.__new__(CAMDriver)
    driver.scheme_plan = plan
    driver.pool = type("Pool", (), {"dimensions": {}})()
    calls = []
    driver.run_scheme = calls.append

    CAMDriver.run_scheme_group(driver, PHYSICS_BEFORE_COUPLER)
    assert kessler_key not in calls
    CAMDriver.run_scheme_group(driver, PHYSICS_AFTER_COUPLER)
    assert calls[-1] == kessler_key


def test_driver_compiles_plan_and_state_from_selected_non_kessler_suite() -> None:
    config = ModelConfig.from_yaml(
        PROJECT / "configs/fkessler_model.yaml"
    ).with_overrides(
        physics_suite="held_suarez_1994",
        dt_seconds=900,
        stop_n=4,
        case_name="held-suarez-control",
        history_enabled=False,
    )
    driver = CAMDriver(config, run_dir=PROJECT, comm=FixedCaseComm())

    assert driver.scheme_plan.name == "held_suarez_1994"
    assert "held_suarez_1994" in {
        scheme.name for scheme in driver.scheme_plan.schemes
    }
    field_names = {
        contract.standard_name
        for contract in driver.state_schema.pool_contracts()
    }
    assert "air_temperature_previous_timestep" not in field_names
    assert "large_scale_precipitation_rate" not in field_names


def test_driver_accepts_a_custom_suite_xml_not_named_in_catalog(
    tmp_path: Path,
) -> None:
    suite_xml = tmp_path / "suite_my_experiment.xml"
    suite_xml.write_text(
        '<?xml version="1.0"?>\n'
        '<suite name="my_experiment" version="1.0">\n'
        '  <group name="physics_before_coupler">\n'
        '    <scheme>held_suarez_1994</scheme>\n'
        '  </group>\n'
        '  <group name="physics_after_coupler"/>\n'
        '</suite>\n'
    )
    config = ModelConfig.from_yaml(
        PROJECT / "configs/fkessler_model.yaml"
    ).with_overrides(
        physics_suite="my_experiment",
        suite_xml=str(suite_xml),
        case_name="custom-suite",
        history_enabled=False,
    )
    driver = CAMDriver(config, run_dir=PROJECT, comm=FixedCaseComm())

    assert driver.scheme_plan.name == "my_experiment"
    assert [scheme.name for scheme in driver.scheme_plan.schemes] == [
        "held_suarez_1994"
    ]
    assert not driver.state_schema.unresolved_schemes
    assert driver.processes.provider_for(
        driver.scheme_plan.scheme("held_suarez_1994")
    ) == "fortran-device"


def test_missing_optional_coupler_group_is_skipped() -> None:
    plan = CCPPSuitePlan.from_xml(
        PROJECT
        / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_musica.xml"
    )
    assert PHYSICS_BEFORE_COUPLER not in plan.group_names
    driver = object.__new__(CAMDriver)
    driver.scheme_plan = plan
    calls = []
    driver.run_scheme_group = (
        lambda group, callback=None: calls.append(group)
    )

    CAMDriver._run_optional_scheme_group(driver, PHYSICS_BEFORE_COUPLER)
    CAMDriver._run_optional_scheme_group(driver, PHYSICS_AFTER_COUPLER)
    assert calls == [PHYSICS_AFTER_COUPLER]


def test_physics_timestep_initial_synchronizes_ccpp_step(
    monkeypatch,
) -> None:
    class Pool:
        def __init__(self):
            self.values = {
                "is_first_timestep": np.array(True),
                "current_timestep_number": np.array(0, dtype=np.int32),
                "fractional_calendar_days_on_end_of_current_timestep": (
                    np.array(0.0)
                ),
                "fractional_calendar_days_on_end_of_next_timestep": (
                    np.array(0.0)
                ),
                (
                    "next_calendar_day_to_perform_shortwave_radiation_for_"
                    "surface_models"
                ): np.array(0.0),
                (
                    "number_of_seconds_until_next_shortwave_radiation_"
                    "timestep"
                ): np.array(0, dtype=np.int32),
            }

        def ccpp_field_name(self, standard_name):
            if standard_name not in self.values:
                raise KeyError(standard_name)
            return standard_name

        def set(self, name, value):
            self.values[name][...] = value

        def get(self, name):
            return self.values[name]

    driver = object.__new__(CAMDriver)
    driver.clock = ModelClock(nstep=4, seconds=7200, dt_seconds=1800)
    driver.config = type("Config", (), {"physics_suite": "cam4"})()
    driver.backend = object()
    driver.orbital_service = type(
        "OrbitalService",
        (),
        {"update": lambda self, pool, clock, orbital_year=None: None},
    )()
    driver.config.orbital_year = 2000
    def run_suite_lifecycle(phase):
        assert phase == "timestep_initial"
        pool.values[
            "number_of_seconds_until_next_shortwave_radiation_timestep"
        ][...] = 3600
        return ()

    driver.run_suite_lifecycle = run_suite_lifecycle
    monkeypatch.setattr(
        driver_module,
        "physics_timestep_initial",
        lambda pool, backend: None,
    )
    pool = Pool()

    driver._physics_timestep_initial(pool)

    assert pool.get("current_timestep_number").item() == 4
    assert pool.get("is_first_timestep").item() is False
    assert (
        pool.get(
            "next_calendar_day_to_perform_shortwave_radiation_for_"
            "surface_models"
        ).item()
        == driver.clock.fractional_calendar_day(3600)
    )


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
    expected.extend(
        f"scheme:{scheme.key}"
        for scheme in session.scheme_plan.active(PHYSICS_AFTER_COUPLER)
    )
    assert calls == expected
    assert calls.count("phase:se_type4_rk") == 6
    assert calls.count("phase:advance_fvm_tracers") == 2
    assert calls.count("phase:vertical_remap_se") == 2
    assert calls.count("phase:vertical_remap_fvm") == 2
    assert calls.count("phase:compute_final_omega") == 1
    assert session.state == DriverState.RUNNING
