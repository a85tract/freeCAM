from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    PICAMVariableSpec,
    RecordingCAMBackend,
)
from freecam.pi_cam import runtime_fortran as runtime_fortran_module
from freecam.pi_cam.errors import BoundaryReplayError, PICAMConfigurationError


def _driver() -> tuple[PICAMDriver, RecordingCAMBackend, InMemoryBoundaryProvider]:
    config = PICAMConfig(
        case_name="unit-pi-cam",
        source_root=Path("/tmp/source"),
        mpi_size=1,
        stop_n=2,
    )
    boundary = InMemoryBoundaryProvider(
        {
            (0, 0): {"sst": np.full((2,), 280.0)},
            (1, 0): {"sst": np.full((2,), 281.0)},
            (2, 0): {"sst": np.full((2,), 282.0)},
            (3, 0): {"sst": np.full((2,), 283.0)},
        }
    )
    backend = RecordingCAMBackend()
    return (
        PICAMDriver(config, boundary, backend, rank=0, size=1),
        backend,
        boundary,
    )


def test_rank_local_boundary_failure_is_raised_on_every_rank() -> None:
    class FailureComm:
        rank = 1
        size = 3

        @staticmethod
        def allgather(value):
            del value
            return ["rank zero failed", None, None]

    config = PICAMConfig(
        case_name="collective-boundary-test",
        source_root=Path("/tmp/source"),
        mpi_size=3,
        stop_n=2,
    )
    driver = PICAMDriver(
        config,
        InMemoryBoundaryProvider(),
        RecordingCAMBackend(),
        rank=1,
        size=3,
        communicator=FailureComm(),
    )

    with pytest.raises(BoundaryReplayError, match="failed collectively; see rank 0"):
        driver._collective_boundary_call("test boundary", lambda: None)


def test_complete_step_is_ordered_by_python_and_advances_1800_seconds() -> None:
    driver, backend, boundary = _driver()
    driver.initialize()
    trace = driver.step()

    assert trace[0].operation == "boundary_import"
    assert trace[-1].operation == "boundary_export"
    assert driver.clock.nstep == 1
    assert driver.clock.seconds == 1800
    assert driver.coupling_step == 1
    assert backend.calls[:3] == [
        "initialize",
        "boundary_export",
        "initial_priming",
    ]
    assert "advance_timestep" in backend.calls
    assert (0, 0) in boundary.exports


def test_python_control_routes_only_numerical_actions_through_generic_execute() -> None:
    driver, backend, _ = _driver()
    driver.initialize()

    driver.step()

    channels = {channel for channel, _ in backend.dispatches}
    assert channels == {
        "boundary-kernel",
        "clock-mirror",
        "io-service",
        "numerical-kernel",
        "state-service",
    }
    assert ("boundary-kernel", "boundary_import") in backend.dispatches
    assert ("boundary-kernel", "boundary_export") in backend.dispatches
    assert ("clock-mirror", "advance_timestep") in backend.dispatches
    assert ("state-service", "leaf_pbuf_deallocate") in backend.dispatches
    assert ("io-service", "wshist") in backend.dispatches
    assert not any(
        channel == "numerical-kernel"
        and operation in {
            "boundary_import",
            "boundary_export",
            "wshist",
            "restart",
            "advance_timestep",
            "leaf_pbuf_deallocate",
            "leaf_pbuf_update_tim_idx",
            "leaf_diag_deallocate",
        }
        for channel, operation in backend.dispatches
    )


def test_restart_alarm_is_decided_by_python_before_calling_io_service() -> None:
    driver, backend, _ = _driver()
    driver.initialize()
    restart = driver.step_plan.select("restart", phase="cam_run4")

    driver._execute(restart)
    assert ("io-service", "restart") not in backend.dispatches

    driver._native_step = driver.config.stop_n
    driver._execute(restart)
    assert ("io-service", "restart") in backend.dispatches


def test_python_output_cadence_can_skip_history_and_schedule_restart() -> None:
    driver, backend, _ = _driver()
    driver.initialize()
    history = driver.step_plan.select("wshist", phase="cam_run4")
    restart = driver.step_plan.select("restart", phase="cam_run4")
    driver.configure_output(
        history_every=2,
        restart_every=2,
        restart_at_end=False,
    )

    driver._native_step = 0
    driver._execute(history)
    driver._execute(restart)
    assert ("io-service", "wshist") not in backend.dispatches
    assert ("io-service", "restart") not in backend.dispatches

    driver._native_step = 1
    driver._execute(history)
    driver._execute(restart)
    assert ("io-service", "wshist") in backend.dispatches
    assert ("io-service", "restart") in backend.dispatches


def test_fine_grained_step_holds_native_import_when_coupling_input_is_held() -> None:
    class HeldBoundary(InMemoryBoundaryProvider):
        def has_fresh_import(self, step: int, rank: int) -> bool:
            del rank
            return step != 2

    template, _, _ = _driver()
    boundary = HeldBoundary(
        {
            (0, 0): {"sst": np.full((2,), 280.0)},
            (1, 0): {"sst": np.full((2,), 281.0)},
            (2, 0): {"sst": np.full((2,), 282.0)},
        }
    )
    backend = RecordingCAMBackend()
    driver = PICAMDriver(template.config, boundary, backend, rank=0, size=1)
    driver.initialize()

    trace = driver.step()

    assert trace[0].operation == "boundary_import"
    assert "boundary_import" not in backend.calls
    assert "stepon_run2" in backend.calls
    assert (0, 0) in boundary.exports


def test_initialize_primes_run1_before_the_first_normal_run2() -> None:
    driver, backend, boundary = _driver()

    driver.initialize()

    assert backend.calls == [
        "initialize",
        "boundary_export",
        "initial_priming",
    ]
    assert [item.operation for item in driver.trace[-3:]] == [
        "boundary_import",
        "initial_priming",
        "boundary_export",
    ]
    assert driver.clock.nstep == 0
    assert driver.coupling_step == 0
    assert (0, 0) in boundary.exports


def test_backend_preallocates_python_state_before_native_initialize() -> None:
    class PreparingBackend(RecordingCAMBackend):
        def prepare_state(self, pool, config, *, rank, size):
            del config, rank, size
            pool.ensure_from_array(
                "phys_state.t",
                np.zeros((16, 30, 1), order="F"),
                category="native_cam_state",
            )
            self.calls.append("prepare_state")

        def initialize(self, pool, *, fcomm):
            assert "phys_state.t" in pool
            super().initialize(pool, fcomm=fcomm)

    driver, _, boundary = _driver()
    backend = PreparingBackend()
    driver = PICAMDriver(driver.config, boundary, backend, rank=0, size=1)

    assert backend.calls == []
    driver.initialize()

    assert backend.calls[:2] == ["prepare_state", "initialize"]
    address = driver.python_initialized_addresses["phys_state.t"]
    assert driver.pool["phys_state.t"].ctypes.data == address


def test_individual_phase_and_scheme_are_exposed_without_advancing_time() -> None:
    driver, backend, _ = _driver()
    driver.initialize()

    result = driver.physics.dadadj.run()
    phase = driver.phases.cam_run3.run()

    assert result.operation == "dadadj"
    assert [item.operation for item in phase] == ["stepon_run3"]
    assert driver.clock.nstep == 0
    assert backend.calls[-2:] == ["dadadj", "stepon_run3"]


def test_isolated_physics_call_prefers_direct_statepool_kernel() -> None:
    class DirectKernelBackend(RecordingCAMBackend):
        direct_kernels = ("dadadj",)

        def execute_kernel(self, name, pool, *, fcomm):
            del pool, fcomm
            self.calls.append(f"direct:{name}")

    template, _, boundary = _driver()
    backend = DirectKernelBackend()
    driver = PICAMDriver(template.config, boundary, backend, rank=0, size=1)
    driver.initialize()

    trace = driver.physics.dry_adjustment.run()

    assert trace.phase == "direct_kernel"
    assert trace.operation == "dadadj"
    assert backend.calls[-1] == "direct:dadadj"


def test_pythonic_phase_and_action_handles_edit_the_same_step_plan() -> None:
    driver, _, _ = _driver()

    assert "rayleigh_friction" in dir(driver.physics)
    assert "rayleigh_friction_tend" in dir(driver.physics)
    assert len(driver.physics) == 298
    assert driver.physics.coverage == {
        "interfaces": 298,
        "runnable": 36,
        "catalog_only": 262,
        "source_reachable": 372,
        "source_catalog": 371,
        "physical_processes": 276,
        "compiled_process_adapters": 276,
        "formerly_catalog_only_interfaces": 262,
        "catalog_adapters_compiled": 262,
            "catalog_current_case_loadable": 0,
            "runtime_templates": 262,
            "runtime_templates_loadable": 0,
            "runtime_bound": 0,
            "runtime_inserted": 0,
            "current_case_loadable": 0,
        "configuration_specific": 276,
        "helper_routines": 95,
        "runtime_overlap": 14,
        "excluded_lifecycle": 1,
        "enabled": 31,
        "disabled": 5,
        "leaf": 15,
        "stage": 21,
    }
    assert driver.physics.names[0] == "surface_fluxes_and_emissions"
    assert driver.physics.dry_adjustment.operation == "dadadj"
    assert driver.physics.dadadj.operation == "dadadj"
    assert driver.physics.deep_convection.metadata["source_procedures"] == (
        "convect_deep::convect_deep_tend",
    )
    assert driver.physics.cloud_fraction_fice.qualified_name == (
        "cloud_fraction::cldfrc_fice"
    )
    assert driver.physics.cldfrc_fice.name == "cloud_fraction_fice"
    assert driver.physics.cloud_fraction_fice.runnable is False
    assert driver.physics.cloud_fraction_fice.level == "process"
    assert driver.physics.zm_conv_evap.name == "zm_conv_evap"
    assert "gamma" not in driver.physics.names
    assert driver.physics.catalog.process("math_lib::gamma").level == "helper"
    with pytest.raises(PICAMConfigurationError, match="not an independently runnable"):
        driver.physics.zm_conv_evap.run()
    assert "cam_run2" in dir(driver.phases)
    expanded = driver.phases.cam_run2.expand()
    driver.physics.rayleigh_friction.enabled = False

    assert len(expanded) == 19
    assert driver.physics.rayleigh_friction.enabled is False
    assert "tracers_and_chemistry" not in {
        action.name for action in driver.step_plan.in_phase("cam_run2")
    }


def test_driver_can_replace_cam_run1_composites_with_ordered_leaf_calls() -> None:
    driver, backend, _ = _driver()
    driver.expand_cam_run1_leaves(experimental=True)
    driver.initialize()

    driver.step()

    assert "aero_model_wetdep" not in backend.calls
    assert "physics_diagnostics" not in backend.calls
    assert "cam_export" not in backend.calls
    for operation in (
        "leaf_modal_aero_prepare",
        "leaf_aero_model_wetdep",
        "leaf_carma_wetdep_tend",
        "leaf_convect_deep_tend_2",
        "leaf_diag_phys_writeout",
        "leaf_cloud_diagnostics_calc",
        "leaf_tropopause_output",
        "leaf_cam_export",
        "leaf_diag_export",
    ):
        assert operation in backend.calls


def test_driver_can_expand_cam_run2_and_run4_leaf_calls() -> None:
    driver, backend, _ = _driver()
    driver.expand_cam_run2_run4_leaves(experimental=True)
    driver.initialize()

    driver.step()

    for composite in (
        "tracers_chemistry",
        "aero_model_drydep",
        "finish",
        "wrapup",
    ):
        assert composite not in backend.calls
    for operation in (
        "leaf_tracers_timestep_tend",
        "leaf_aoa_tracers_timestep_tend",
        "leaf_chem_timestep_tend",
        "leaf_aero_model_drydep",
        "leaf_carma_timestep_tend",
        "leaf_carma_accumulate_stats",
        "leaf_pbuf_deallocate",
        "leaf_pbuf_update_tim_idx",
        "leaf_diag_deallocate",
        "stepon_run3",
        "leaf_cam_run4_wrapup",
        "leaf_cam_run4_step_cost",
        "leaf_cam_run4_flush",
    ):
        assert operation in backend.calls


def test_isolated_io_leaf_is_an_executable_workflow_action() -> None:
    driver, backend, _ = _driver()
    driver.expand_cam_run4_leaves(experimental=True)
    driver.initialize()

    trace = driver.run_action(
        "flush_leaf", phase="cam_run4", experimental=True
    )

    assert trace.operation == "leaf_cam_run4_flush"
    assert backend.calls[-1] == "leaf_cam_run4_flush"


def test_native_backend_can_fuse_only_the_unchanged_default_step() -> None:
    class FusedRecordingCAMBackend(RecordingCAMBackend):
        def execute_source_step(self, pool, *, fcomm, apply_import=True):
            del pool, fcomm, apply_import
            self.calls.append("source_step")

    config = PICAMConfig(
        case_name="unit-pi-cam",
        source_root=Path("/tmp/source"),
        mpi_size=1,
        stop_n=2,
        execution_mode="source_compat",
    )
    boundary = InMemoryBoundaryProvider(
        {
            (0, 0): {"sst": np.full((2,), 280.0)},
            (1, 0): {"sst": np.full((2,), 281.0)},
            (2, 0): {"sst": np.full((2,), 282.0)},
            (3, 0): {"sst": np.full((2,), 283.0)},
        }
    )
    backend = FusedRecordingCAMBackend()
    driver = PICAMDriver(config, boundary, backend, rank=0, size=1)
    driver.initialize()

    trace = driver.step()

    assert backend.calls[-1] == "source_step"
    assert [item.operation for item in trace] == [
        item.operation for item in driver.step_plan
    ]
    assert driver.clock.nstep == 1

    driver.step_plan.set_enabled(
        "dadadj", False, phase="cam_run1", experimental=True
    )
    driver.step()
    assert backend.calls.count("source_step") == 1
    assert "stepon_run2" in backend.calls


def test_direct_kernel_is_explicit_and_does_not_advance_model_time() -> None:
    class DirectRecordingCAMBackend(RecordingCAMBackend):
        direct_kernels = ("sample",)

        def execute_kernel(self, name, pool, *, fcomm):
            del pool, fcomm
            self.calls.append(f"kernel:{name}")

    driver, _, boundary = _driver()
    backend = DirectRecordingCAMBackend()
    driver = PICAMDriver(driver.config, boundary, backend, rank=0, size=1)
    driver.initialize()

    with pytest.raises(PICAMConfigurationError, match="experimental=True"):
        driver.kernels.sample.run()
    trace = driver.kernels.sample.run(experimental=True)

    assert trace.phase == "direct_kernel"
    assert trace.operation == "sample"
    assert backend.calls[-1] == "kernel:sample"
    assert driver.clock.nstep == 0


def test_dynamic_field_and_python_process_are_part_of_the_step_plan() -> None:
    driver, backend, _ = _driver()
    driver.initialize()
    tracer = driver.define_variable(
        PICAMVariableSpec(
            "experiment_tracer",
            ("pcols", "pver"),
            initial=2.0,
            aliases=("tracer",),
        )
    )

    def add_timestep(fields, context, *, scale):
        fields["tracer"][...] += scale * context.timestep_seconds

    process = driver.physics.install_python(
        add_timestep,
        name="notebook_heating",
        after="dadadj",
        writes=("tracer",),
        parameters={"scale": 1.0e-3},
    )
    address = tracer.ctypes.data

    process.run()
    assert np.array_equal(tracer, np.full(tracer.shape, 3.8))
    driver.step()

    assert np.array_equal(tracer, np.full(tracer.shape, 5.6))
    assert tracer.ctypes.data == address
    assert "notebook_heating" not in backend.calls
    with pytest.raises(Exception, match="used by runtime processes"):
        driver.delete_variable("tracer")
    process.remove()
    driver.delete_variable("tracer")
    assert "experiment_tracer" not in driver.pool


def test_physics_run_uses_attribute_style_rank_local_state() -> None:
    from freecam.pi_cam import Physics

    driver, _, _ = _driver()
    driver.initialize()
    tracer = driver.define_variable(
        PICAMVariableSpec("experiment_tracer", ("pcols", "pver"), initial=2.0)
    )

    class Heating(Physics):
        writes = ("experiment_tracer",)

        def run(self, state, context):
            state.experiment_tracer += 1.0e-3 * context.timestep_seconds

    process = driver.physics.install_python(
        Heating().tendency,
        name="attribute_heating",
        after="dadadj",
        writes=Heating.writes,
    )
    process.run()

    assert np.array_equal(tracer, np.full(tracer.shape, 3.8))


def test_rank_independent_numpy_array_is_copied_into_statepool() -> None:
    driver, _, _ = _driver()
    driver.initialize()
    source = np.linspace(0.0, 1.0, 30)

    values = driver.define_array("rh", source)

    assert np.array_equal(values, source)
    assert values.ctypes.data != source.ctypes.data
    assert driver.pool.contract("rh").dynamic is True
    driver.delete_variable("rh")
    assert "rh" not in driver.pool


def test_failed_python_process_restores_declared_writes() -> None:
    driver, _, _ = _driver()
    driver.initialize()
    values = driver.define_variable(
        PICAMVariableSpec("runtime_value", ("pver",), initial=7.0)
    )

    def fail_after_write(fields, context):
        del context
        fields["runtime_value"][...] = -1.0
        raise RuntimeError("intentional callback failure")

    process = driver.physics.install_python(
        fail_after_write,
        name="failure_probe",
        after="dadadj",
        writes=("runtime_value",),
    )
    with pytest.raises(Exception, match="were restored"):
        process.run()

    assert np.array_equal(values, np.full(values.shape, 7.0))


def test_runtime_fortran_process_uses_the_same_mutable_step_plan(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "device.json"
    library = tmp_path / "device.so"
    manifest.write_text("{}")
    library.write_bytes(b"device")

    class FakeDevice:
        name = "runtime_offset"
        source_hash = "source"
        processes = {"runtime_offset": "run"}
        entrypoints = {"run": {"arguments": []}}

        def __init__(self, path):
            self.manifest_path = Path(path)
            self.library_path = library

        def _ensure_abi(self):
            return None

        def invoke_process(self, name, pool):
            assert name == "runtime_offset"
            pool["runtime_value"][...] += 2.0

    driver, _, _ = _driver()
    driver.initialize()
    values = driver.define_variable(
        PICAMVariableSpec("runtime_value", ("pver",), initial=1.0)
    )
    monkeypatch.setattr(runtime_fortran_module, "FortranDevice", FakeDevice)
    monkeypatch.setattr(
        driver.fortran_processes,
        "_prepare_manifest",
        lambda spec: manifest,
    )

    process = driver.physics.install_fortran(
        manifest,
        process="runtime_offset",
        after="dadadj",
        unsafe=True,
    )
    process.run()

    assert np.array_equal(values, np.full(values.shape, 3.0))
    process.disable()
    assert not process.enabled
    process.enable()
    process.move(before="deep_convection")
    process.remove()
    driver.delete_variable("runtime_value")
