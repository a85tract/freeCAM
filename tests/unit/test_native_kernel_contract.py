from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest

from freecam.model.backend import FVMKernelConfig, KernelBackend
from freecam.model.config import ModelConfig
from freecam.model.errors import MissingKernelError, StateOwnershipError


ROOT = Path(__file__).resolve().parents[2]


class _Pool:
    dimensions = {
        "fv_nphys": 3,
        "pver": 30,
        "nconst": 3,
        "ntrac": 3,
        "np": 4,
        "fvm_reconstruction": 5,
        "fvm_internal": 5,
        "fvm_interp_span": 7,
        "fvm_stretch": 7,
        "fvm_halo": 9,
    }


def test_fvm_dimensions_and_controls_are_python_owned() -> None:
    config = FVMKernelConfig.from_pool(_Pool())
    assert config.nc == 3
    assert config.nlev == 30
    assert config.ntrac == 3
    assert config.np == 4
    assert config.level_begin == 1
    assert config.level_end == 30
    assert config.large_courant is True
    assert config.irecons_levels.dtype == np.int32
    assert np.array_equal(config.irecons_levels, np.full(30, 6, np.int32))

    native = (ROOT / "native/kernels/fvm_transport_kernel.F90").read_text()
    assert "type(fvm_dimensions_c), intent(in) :: config" in native
    assert "subflux(3,3,4,30)" not in native
    assert "tracer(9,9,30,3)" not in native
    assert "level_begin,config%level_end" in native


def test_limiter_flattened_grid_dimension_is_validated_separately() -> None:
    backend = (ROOT / "src/freecam/model/backend.py").read_text()
    native = (ROOT / "native/kernels/se_startup_kernel.F90").read_text()
    manifest = (ROOT / "native/kernels/abi-v2.json").read_text()
    assert "ngp_value = tracer_mass.shape[0] * tracer_mass.shape[1]" in backend
    assert "pycam_sima_validate_se_dimensions_v2" in native
    assert '"pycam_sima_validate_se_dimensions_v2"' in manifest
    assert "if (ngp/=build_ngp) return" not in native

    runtime = KernelBackend(ROOT / "build/libpycam_sima_kernels.so")
    runtime._require_se_dimensions(4, 16)
    with pytest.raises(StateOwnershipError, match="np=4, ngp=4"):
        runtime._require_se_dimensions(4, 4)


def test_kernel_reports_and_checks_its_model_specialization() -> None:
    runtime = KernelBackend(ROOT / "build/libpycam_sima_kernels.so")
    reference = ModelConfig()
    assert runtime.specialization == reference.kernel_specialization
    runtime.validate_specialization(reference)
    with pytest.raises(MissingKernelError, match="specialized for"):
        runtime.validate_specialization(
            reference.with_overrides(
                grid="ne3np5.pg3",
                np=5,
            )
        )


def test_qwater_preparation_uses_native_array_expression() -> None:
    runtime = KernelBackend(ROOT / "build/libpycam_sima_kernels.so")
    constituent_mass = np.asfortranarray(
        np.full((4, 4, 2, 1, 3), 2.0, dtype=np.float64)
    )
    pressure_thickness = np.asfortranarray(
        np.full((4, 4, 2, 1), 4.0, dtype=np.float64)
    )
    qwater = np.asfortranarray(
        np.full((4, 4, 2, 1, 2), np.nan, dtype=np.float64)
    )

    runtime.prepare_qwater(
        constituent_mass=constituent_mass,
        pressure_thickness=pressure_thickness,
        qwater=qwater,
        qsize=1,
    )

    assert np.array_equal(qwater[..., 0], np.full(qwater[..., 0].shape, 0.5))
    assert np.array_equal(qwater[..., 1], np.zeros(qwater[..., 1].shape))


def test_kernel_library_has_no_model_framework_dependency() -> None:
    library = ROOT / "build/libpycam_sima_kernels.so"
    if not library.exists():
        pytest.skip("build the native libraries before checking ELF dependencies")
    dynamic = subprocess.run(
        ("readelf", "-d", str(library)),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.lower()
    for forbidden in (
        "mpi", "pmi", "pals", "esmf", "pio", "netcdf", "hdf5", "libsci",
        "rpath", "runpath",
    ):
        assert forbidden not in dynamic

    makefile = (ROOT / "native/kernels/Makefile").read_text()
    assert "env -i" in makefile
    assert ".env_mach_specific" not in makefile

    main_symbols = subprocess.run(
        ("nm", "-D", "--defined-only", str(library)),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    assert "pycam_sima_kessler" not in main_symbols

    for device in ("kessler", "kessler_update"):
        device_library = (
            ROOT
            / f"build/devices/{device}/libpycam_device_{device}.so"
        )
        assert device_library.is_file()
        device_dynamic = subprocess.run(
            ("readelf", "-d", str(device_library)),
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.lower()
        for forbidden in (
            "mpi", "pmi", "pals", "esmf", "pio", "netcdf", "hdf5",
            "libsci", "rpath", "runpath",
        ):
            assert forbidden not in device_dynamic
