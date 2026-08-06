from pathlib import Path

import numpy as np

from pycam_sima.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    RecordingCAMBackend,
)


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


def test_individual_phase_and_scheme_are_exposed_without_advancing_time() -> None:
    driver, backend, _ = _driver()
    driver.initialize()

    result = driver.physics.dadadj.run()
    phase = driver.phases.cam_run3.run()

    assert result.operation == "dadadj"
    assert [item.operation for item in phase] == ["stepon_run3"]
    assert driver.clock.nstep == 0
    assert backend.calls[-2:] == ["dadadj", "stepon_run3"]


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
