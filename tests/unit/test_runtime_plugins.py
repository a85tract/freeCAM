from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from pycam_sima import (
    ActivatePhysics,
    DefineVariable,
    InstallPhysics,
    ObserveFields,
    PhysicsPluginSpec,
    RunScheme,
    SchemePlacement,
    SegmentPlan,
    VariableSpec,
)
from pycam_sima.model import CCPPSuitePlan, ModelConfig, ModelSnapshot
from pycam_sima.model.checkpoint import deserialize_snapshot, serialize_snapshot
from pycam_sima.model.clock import NoLeapClock
from pycam_sima.model.comm import SerialComm
from pycam_sima.model.devices import DeviceRegistry
from pycam_sima.model.experiment import validate_segment_plan
from pycam_sima.model.grid import dimensions_for_rank
from pycam_sima.model.plugins import PhysicsPluginManager
from pycam_sima.model.state import StatePool


ROOT = Path(__file__).resolve().parents[2]
KESSLER_SUITE = (
    ROOT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


class _Driver:
    def __init__(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.pool = StatePool(dimensions_for_rank(0, 24))
        self.comm = SerialComm()
        self.state = SimpleNamespace(value="RUNNING")
        self.clock = SimpleNamespace(nstep=0)
        self._last_phase = "physics_timestep_initial"
        self._last_scheme = None
        self._last_scheme_group = None
        self._native_call_depth = 0
        self._boundary_index = 0
        self.scheme_plan = CCPPSuitePlan.from_xml(KESSLER_SUITE)
        empty = tmp_path / "empty-devices"
        empty.mkdir(exist_ok=True)
        self.backend = SimpleNamespace(
            devices=DeviceRegistry(empty),
            call_count=0,
        )
        self.plugins = PhysicsPluginManager(
            self, cache_dir=tmp_path / "cache"
        )

    @property
    def execution_cursor(self):
        return (
            self.state.value,
            self.clock.nstep,
            self._last_phase,
            self._last_scheme,
            self._last_scheme_group,
            self._boundary_index,
            self._native_call_depth,
        )


def _runtime_probe_descriptor(tmp_path: Path) -> Path:
    source = tmp_path / "runtime_probe.F90"
    source.write_text(
        "module runtime_probe\n"
        "  use ccpp_kinds, only: kind_phys\n"
        "  implicit none\n"
        "contains\n"
        "  !> \\section arg_table_runtime_probe_initialize Argument Table\n"
        "  !! \\htmlinclude runtime_probe_initialize.html\n"
        "  subroutine runtime_probe_initialize(lifecycle, errmsg, errflg)\n"
        "    real(kind_phys), intent(out) :: lifecycle\n"
        "    character(len=*), intent(out) :: errmsg\n"
        "    integer, intent(out) :: errflg\n"
        "    lifecycle = 100.0_kind_phys\n"
        "    errmsg = ''; errflg = 0\n"
        "  end subroutine runtime_probe_initialize\n"
        "  !> \\section arg_table_runtime_probe_run Argument Table\n"
        "  !! \\htmlinclude runtime_probe_run.html\n"
        "  subroutine runtime_probe_run(input, output, errmsg, errflg)\n"
        "    real(kind_phys), intent(in) :: input(:)\n"
        "    real(kind_phys), intent(out) :: output(:)\n"
        "    character(len=*), intent(out) :: errmsg\n"
        "    integer, intent(out) :: errflg\n"
        "    output = input + 1.0_kind_phys\n"
        "    errmsg = ''; errflg = 0\n"
        "  end subroutine runtime_probe_run\n"
        "  !> \\section arg_table_runtime_probe_finalize Argument Table\n"
        "  !! \\htmlinclude runtime_probe_finalize.html\n"
        "  subroutine runtime_probe_finalize(lifecycle, errmsg, errflg)\n"
        "    real(kind_phys), intent(out) :: lifecycle\n"
        "    character(len=*), intent(out) :: errmsg\n"
        "    integer, intent(out) :: errflg\n"
        "    lifecycle = -100.0_kind_phys\n"
        "    errmsg = ''; errflg = 0\n"
        "  end subroutine runtime_probe_finalize\n"
        "end module runtime_probe\n"
    )
    metadata = tmp_path / "runtime_probe.meta"
    metadata.write_text(
        "[ccpp-table-properties]\n"
        "  name = runtime_probe\n"
        "  type = scheme\n"
        "[ccpp-arg-table]\n"
        "  name = runtime_probe_initialize\n"
        "  type = scheme\n"
        "[ lifecycle ]\n"
        "  standard_name = runtime_probe_lifecycle\n"
        "  units = K\n"
        "  type = real | kind = kind_phys\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errmsg ]\n"
        "  standard_name = ccpp_error_message\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errflg ]\n"
        "  standard_name = ccpp_error_code\n"
        "  units = 1\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ccpp-arg-table]\n"
        "  name = runtime_probe_run\n"
        "  type = scheme\n"
        "[ input ]\n"
        "  standard_name = runtime_probe_input\n"
        "  units = K\n"
        "  type = real | kind = kind_phys\n"
        "  dimensions = (horizontal_loop_extent)\n"
        "  intent = in\n"
        "[ output ]\n"
        "  standard_name = runtime_probe_output\n"
        "  units = K\n"
        "  type = real | kind = kind_phys\n"
        "  dimensions = (horizontal_loop_extent)\n"
        "  intent = out\n"
        "[ errmsg ]\n"
        "  standard_name = ccpp_error_message\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errflg ]\n"
        "  standard_name = ccpp_error_code\n"
        "  units = 1\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ccpp-arg-table]\n"
        "  name = runtime_probe_finalize\n"
        "  type = scheme\n"
        "[ lifecycle ]\n"
        "  standard_name = runtime_probe_lifecycle\n"
        "  units = K\n"
        "  type = real | kind = kind_phys\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errmsg ]\n"
        "  standard_name = ccpp_error_message\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errflg ]\n"
        "  standard_name = ccpp_error_code\n"
        "  units = 1\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
    )
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "runtime_probe",
                "fortran_module": "runtime_probe",
                "sources": [str(source)],
                "metadata": [str(metadata)],
                "source_modules": ["runtime_probe"],
                "providers": {
                    "ccpp_kinds": str(
                        ROOT / "native/devices/support/ccpp_kinds.F90"
                    )
                },
                "state_policy": "stateless",
                "dimension_bindings": {
                    "horizontal_loop_extent": "nphys_local"
                },
                "entrypoints": {
                    "initialize": {"table": "runtime_probe_initialize"},
                    "run": {"table": "runtime_probe_run"},
                    "finalize": {"table": "runtime_probe_finalize"},
                },
                "processes": {
                    "runtime_probe": "run",
                    "runtime_probe:initialize": "initialize",
                    "runtime_probe:finalize": "finalize",
                },
            },
            sort_keys=False,
        )
    )
    return descriptor


def test_dynamic_variable_keeps_old_addresses_and_round_trips_snapshot(
    tmp_path: Path,
):
    driver = _Driver(tmp_path)
    before = driver.pool.pointer_records()
    spec = VariableSpec(
        name="plugin_diagnostic",
        standard_name="plugin_diagnostic",
        dtype="float64",
        dimensions=("nphys_local", "pver"),
        units="K",
    )

    values = driver.plugins.define_variable(spec, initial=7.0)

    driver.pool.assert_pointer_stability(before)
    assert values.flags.f_contiguous
    assert np.all(values == 7.0)
    assert "plugin_diagnostic" in driver.pool.dynamic_fields

    snapshot_driver = SimpleNamespace(
        pool=driver.pool,
        clock=NoLeapClock(nstep=2, seconds=3600),
        state=SimpleNamespace(value="RUNNING"),
        comm=SimpleNamespace(rank=0, size=24),
        config=ModelConfig(),
        scheme_plan=driver.scheme_plan,
        _last_phase=driver._last_phase,
        _last_scheme=None,
        _last_scheme_group=None,
        _boundary_index=driver._boundary_index,
        backend=driver.backend,
        plugins=driver.plugins,
    )
    metadata, content = serialize_snapshot(
        ModelSnapshot.capture(snapshot_driver)
    )
    restored = deserialize_snapshot(metadata, content).new_pool()

    assert restored.contract("plugin_diagnostic") == spec.contract()
    assert np.array_equal(restored.get("plugin_diagnostic"), values)
    assert "plugin_diagnostic" in restored.dynamic_fields


def test_source_and_prebuilt_plugins_share_the_same_runtime_contract(
    tmp_path: Path,
):
    descriptor = _runtime_probe_descriptor(tmp_path)
    placement = SchemePlacement(
        "runtime_probe",
        group="physics_before_coupler",
        after="kessler",
    )
    source_driver = _Driver(tmp_path / "source")
    source_record = source_driver.plugins.install(
        PhysicsPluginSpec(
            str(descriptor),
            placements=(placement,),
            project_root=str(tmp_path),
        ),
        initial_values={"runtime_probe_input": 4.0},
        unsafe=True,
    )

    source_driver.backend.devices.invoke(
        "runtime_probe", source_driver.pool
    )
    assert np.all(
        source_driver.pool.get_ccpp("runtime_probe_output") == 5.0
    )
    assert source_driver.pool.is_initialized(
        source_driver.pool.ccpp_field_name("runtime_probe_output")
    )
    assert (
        source_driver.scheme_plan.scheme("runtime_probe").group
        == "physics_before_coupler"
    )
    source_driver.plugins.deactivate("runtime_probe", unsafe=True)
    assert np.all(
        source_driver.pool.get_ccpp("runtime_probe_lifecycle") == -100.0
    )
    source_driver.plugins.activate("runtime_probe", unsafe=True)
    assert np.all(
        source_driver.pool.get_ccpp("runtime_probe_lifecycle") == 100.0
    )
    source_driver.backend.devices.invoke(
        "runtime_probe", source_driver.pool
    )

    prebuilt_driver = _Driver(tmp_path / "prebuilt")
    prebuilt_record = prebuilt_driver.plugins.install(
        PhysicsPluginSpec(
            source_record.manifest_path,
            placements=(placement,),
        ),
        initial_values={"runtime_probe_input": 4.0},
        unsafe=True,
    )
    prebuilt_driver.backend.devices.invoke(
        "runtime_probe", prebuilt_driver.pool
    )

    assert source_record.source_hash == prebuilt_record.source_hash
    assert np.array_equal(
        source_driver.pool.get_ccpp("runtime_probe_output"),
        prebuilt_driver.pool.get_ccpp("runtime_probe_output"),
    )
    source_driver.plugins.finalize_all()
    assert np.all(
        source_driver.pool.get_ccpp("runtime_probe_lifecycle") == -100.0
    )


def test_dynamic_actions_are_json_serializable():
    variable = VariableSpec(
        "runtime_control",
        "float64",
        ("nphys_local",),
        standard_name="runtime_control",
    )
    plugin = PhysicsPluginSpec(
        "/shared/runtime_probe/device.json",
        placements=(SchemePlacement("runtime_probe"),),
        name="runtime_probe",
    )
    plan = SegmentPlan(
        "dynamic",
        (
            DefineVariable(variable, np.float64(2.0)),
            InstallPhysics(
                plugin,
                initial_values={"runtime_probe_input": np.float64(4.0)},
            ),
            ActivatePhysics("runtime_probe"),
        ),
        unsafe=True,
    )

    assert SegmentPlan.from_mapping(plan.as_dict()) == plan


def test_plan_can_observe_an_explicit_plugin_variable(tmp_path: Path):
    driver = _Driver(tmp_path)
    variable = VariableSpec(
        "runtime_plugin_temperature",
        "float64",
        ("nphys_local", "pver"),
        standard_name="runtime_plugin_temperature",
        units="K",
    )
    plan = SegmentPlan(
        "install-run-observe",
        (
            InstallPhysics(
                PhysicsPluginSpec(
                    "/shared/runtime/device.json",
                    name="runtime_temperature_offset",
                    placements=(
                        SchemePlacement("runtime_temperature_offset"),
                    ),
                    variables=(variable,),
                ),
                initial_values={"runtime_plugin_temperature": 240.0},
            ),
            RunScheme(
                "runtime_temperature_offset",
                group="physics_before_coupler",
            ),
            ObserveFields(("runtime_plugin_temperature",)),
        ),
        unsafe=True,
    )

    validate_segment_plan(driver, plan)
