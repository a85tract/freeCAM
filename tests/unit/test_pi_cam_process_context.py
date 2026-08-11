from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    PICAMFieldContract,
    RecordingCAMBackend,
)
from freecam.pi_cam.errors import PICAMConfigurationError, PICAMStateError
from freecam.pi_cam.kernel_codegen import generate_direct_kernel_module
from freecam.pi_cam.process_codegen import generated_promoted_kernels


class _PromotedBackend(RecordingCAMBackend):
    direct_kernels = ("cloud_fraction_fice",)

    def prepare_state(self, pool, config, *, rank, size):
        del config, rank, size
        pool.dimensions.update({"chunks": 2, "pverp": 31, "pcnst": 4})
        pool.create(PICAMFieldContract("grid.chunk_ncols", ("chunks",), "int32"))
        pool.create(PICAMFieldContract("grid.chunk_id", ("chunks",), "int32"))
        pool.create(
            PICAMFieldContract(
                "phys_state.t", ("pcols", "pver", "chunks"), "float64"
            ),
            initial=240.0,
        )
        pool["grid.chunk_ncols"][:] = (16, 9)
        pool["grid.chunk_id"][:] = (1, 2)

    def execute_promoted_process(self, name, pool, *, bindings, fcomm):
        del fcomm
        assert name == "cloud_fraction_fice"
        temperature = pool[bindings["process_context.cloud_fraction_fice.t"]]
        fice = pool[bindings["process_context.cloud_fraction_fice.fice"]]
        fsnow = pool[bindings["process_context.cloud_fraction_fice.fsnow"]]
        fice[...] = temperature * 0.0 + 0.25
        fsnow[...] = temperature * 0.0 + 0.75
        self.calls.append("promoted:cloud_fraction_fice")


def _driver(backend=None) -> PICAMDriver:
    config = PICAMConfig(
        case_name="process-context",
        source_root=Path("/tmp/source"),
        mpi_size=1,
        stop_n=2,
    )
    boundary = InMemoryBoundaryProvider(
        {
            (0, 0): {"sst": np.full((2,), 280.0)},
            (1, 0): {"sst": np.full((2,), 281.0)},
            (2, 0): {"sst": np.full((2,), 282.0)},
        }
    )
    driver = PICAMDriver(
        config,
        boundary,
        backend or _PromotedBackend(),
        rank=0,
        size=1,
    )
    driver.initialize()
    return driver


def test_catalog_process_can_promote_caller_arguments_into_statepool() -> None:
    driver = _driver()

    result = driver.physics.cloud_fraction_fice()
    process = result.process

    assert process.runnable is True
    assert process.capability == "statepool_bound"
    assert driver.physics.coverage["standalone"] == 1
    assert process.metadata["native_available"] is True
    bindings = {item["argument"]: item["field"] for item in process.bindings}
    assert bindings["ncol"] == "grid.chunk_ncols"
    assert bindings["t"] == "phys_state.t"
    assert next(item["source"] for item in process.bindings if item["argument"] == "t") == "inferred"
    assert bindings["fice"] == "process_context.cloud_fraction_fice.fice"
    assert driver.pool[bindings["fice"]].shape == (16, 30, 2)
    assert driver.pool[bindings["fice"]].flags.f_contiguous

    assert result.trace.phase == "promoted_process"
    assert tuple(result) == ("fice", "fsnow")
    assert np.all(result.fice == 0.25)
    assert np.all(result.fsnow == 0.75)
    assert driver.clock.nstep == 0

    result.remove()
    assert bindings["fice"] not in driver.pool
    assert "phys_state.t" in driver.pool


def test_assumed_shape_process_uses_explicit_statepool_bindings() -> None:
    driver = _driver()
    for name in ("u", "v", "u_n", "v_n", "mag"):
        driver.define_variable(
            {
                "schema_version": 1,
                "name": f"unit_vector.{name}",
                "dimensions": ["pcols", "chunks"],
                "dtype": "float64",
                "initial": 1.0 if name in {"u", "v"} else 0.0,
            }
        )

    process = driver.physics.get_unit_vector.promote(
        bindings={name: f"unit_vector.{name}" for name in ("u", "v", "u_n", "v_n", "mag")}
    )

    assert process.runnable is False
    assert process.capability == "statepool_bound_no_native"
    assert process.metadata["native_available"] is False
    assert process.metadata["created_fields"] == ()


def test_process_call_accepts_an_explicit_statepool_array_override() -> None:
    driver = _driver()

    result = driver.physics.cloud_fraction_fice(t=driver.pool["phys_state.t"])
    temperature = next(
        binding for binding in result.bindings if binding.argument == "t"
    )

    assert temperature.field == "phys_state.t"
    assert temperature.source == "explicit"
    result.remove()


def test_promotion_requires_real_inputs_and_rolls_back_partial_state() -> None:
    driver = _driver()
    before = tuple(driver.pool)

    with pytest.raises(PICAMStateError, match="needs initial values or field bindings"):
        driver.physics.zm_conv_evap.promote()

    assert tuple(driver.pool) == before
    assert "zm_conv_evap" not in driver.process_contexts


def test_scalar_caller_context_can_be_explicitly_initialized() -> None:
    driver = _driver()
    inputs = {
        "ths": 300.0,
        "thvs": 301.0,
        "qflx": 1.0e-5,
        "shflx": 10.0,
        "rrho": 0.8,
        "ustar": 0.4,
    }

    process = driver.physics.calc_obklen.promote(initials=inputs)

    assert len(process.bindings) == 10
    for name, value in inputs.items():
        field = next(
            item["field"] for item in process.bindings if item["argument"] == name
        )
        np.testing.assert_array_equal(driver.pool[field], np.full((2,), value))
    with pytest.raises(PICAMConfigurationError, match="built without its direct adapter"):
        process.run()


def test_generated_promoted_descriptor_covers_all_simple_candidate_processes() -> None:
    kernels = generated_promoted_kernels()
    names = tuple(kernel.name for kernel in kernels)

    assert len(kernels) == 21
    assert "cloud_fraction_fice" in names
    assert "zm_conv_evap" in names
    assert "compute_uwshcu_inv" in names
    assert "get_unit_vector" in names
    cloud = next(kernel for kernel in kernels if kernel.name == "cloud_fraction_fice")
    source = generate_direct_kernel_module((cloud,))
    assert cloud.arguments[0].rank == 1
    assert cloud.arguments[1].rank == 3
    assert "use cloud_fraction, only: cldfrc_fice" in source
    assert "call cldfrc_fice(" in source
