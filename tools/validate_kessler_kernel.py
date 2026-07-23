#!/usr/bin/env python3
"""Compare the stateless ABI bit-for-bit with the pinned CAM Kessler source."""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/schemes/kessler/kessler.F90"
)
KERNEL_LIBRARY = (
    ROOT / "build/devices/kessler/libpycam_device_kessler.so"
)
F64F = np.ctypeslib.ndpointer(
    dtype=np.float64, flags=("F_CONTIGUOUS", "ALIGNED")
)


def _reference_signature(function: ctypes._CFuncPtr) -> None:
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        *([F64F] * 11),
        ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = None


def _device_signature(function: ctypes._CFuncPtr) -> None:
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        *([F64F] * 11),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
    ]
    function.restype = ctypes.c_int


def _inputs() -> list[np.ndarray]:
    ncol, nz = 5, 30
    level = np.linspace(0.0, 1.0, nz)

    def tile(value: np.ndarray) -> np.ndarray:
        return np.asfortranarray(np.tile(value, (ncol, 1)))

    cpair = tile(np.full(nz, 1004.64))
    rair = tile(np.full(nz, 287.0423113650487))
    rho = tile(0.02 + 1.18 * level)
    z = tile(25000.0 - 24900.0 * level)
    pk = tile(0.2 + 0.8 * level)
    theta = tile(310.0 - 15.0 * level)
    qv = tile(1.0e-6 + 0.012 * level)
    qc = tile(2.0e-4 * np.exp(-((level - 0.75) / 0.15) ** 2))
    qr = tile(2.0e-5 * np.exp(-((level - 0.85) / 0.10) ** 2))
    precl = np.empty(ncol, dtype=np.float64, order="F")
    relhum = np.empty((ncol, nz), dtype=np.float64, order="F")
    return [cpair, rair, rho, z, pk, theta, qv, qc, qr, precl, relhum]


def main() -> int:
    subprocess.run(
        ("make", "-C", str(ROOT / "native/kernels"), "all"), check=True
    )
    with tempfile.TemporaryDirectory(prefix="pycam-kessler-reference-") as temp:
        temp_dir = Path(temp)
        reference_library = temp_dir / "libkessler_reference.so"
        subprocess.run(
            (
                "gfortran",
                "-O2",
                "-fPIC",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-ffree-line-length-none",
                "-shared",
                "-J",
                str(temp_dir),
                "-o",
                str(reference_library),
                str(ROOT / "tests/fortran/ccpp_kinds.F90"),
                str(SOURCE),
                str(ROOT / "tests/fortran/kessler_reference_adapter.F90"),
            ),
            check=True,
        )
        native = ctypes.CDLL(str(KERNEL_LIBRARY))
        reference = ctypes.CDLL(str(reference_library))
        initialize = native.pycam_device_kessler_initialize_v1
        initialize.argtypes = [
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
        ]
        initialize.restype = ctypes.c_int
        actual_function = native.pycam_device_kessler_run_v1
        expected_function = reference.kessler_reference_v1
        _device_signature(actual_function)
        _reference_signature(expected_function)
        actual, expected = _inputs(), _inputs()
        expected_error = ctypes.c_int()
        scalars = (5, 30, 1800.0, 2.501e6, 100000.0, 1000.0)
        message = ctypes.create_string_buffer(2048)
        actual_error = initialize(
            *scalars[3:], message, len(message)
        )
        if actual_error:
            raise SystemExit(
                "generated Kessler initialize failed: "
                f"{message.value.decode(errors='replace')}"
            )
        actual_error = actual_function(
            scalars[0],
            scalars[1],
            scalars[2],
            scalars[1],
            1,
            *actual,
            message,
            len(message),
        )
        expected_function(
            *scalars, *expected, ctypes.byref(expected_error)
        )
        if actual_error != expected_error.value:
            raise SystemExit(
                f"error flag mismatch: {actual_error} != "
                f"{expected_error.value}"
            )
        names = (
            "cpair", "rair", "rho", "z", "pk", "theta", "qv", "qc", "qr",
            "precl", "relhum",
        )
        for name, actual_value, expected_value in zip(names, actual, expected):
            equal = actual_value.view(np.uint64) == expected_value.view(np.uint64)
            if np.all(equal):
                continue
            index = tuple(np.argwhere(~equal)[0])
            actual_bits = int(actual_value[index].view(np.uint64))
            expected_bits = int(expected_value[index].view(np.uint64))
            raise SystemExit(
                f"{name} differs at {index}: actual=0x{actual_bits:016x}, "
                f"expected=0x{expected_bits:016x}"
            )
    print("KESSLER_SOURCE_BFB fields=11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
