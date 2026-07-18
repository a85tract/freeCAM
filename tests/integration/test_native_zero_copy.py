from pathlib import Path

import numpy as np
import pytest


LIBRARY = Path("build/native/libpycam_sima_kessler.so")


@pytest.mark.skipif(not LIBRARY.is_file(), reason="native library has not been built")
def test_calc_exner_modifies_python_output_without_copy():
    from cffi import FFI

    ffi = FFI()
    ffi.cdef(
        "int pycam_calc_exner_run(int, int, void *, void *, double, void *, void *);"
    )
    lib = ffi.dlopen(str(LIBRARY.resolve()))
    ncol, nz = 2, 3
    cpair = np.full((ncol, nz), 1004.64, dtype=np.float64, order="F")
    rair = np.full((ncol, nz), 287.0, dtype=np.float64, order="F")
    pmid = np.asfortranarray(np.linspace(20000.0, 90000.0, ncol * nz).reshape((ncol, nz), order="F"))
    exner = np.empty((ncol, nz), dtype=np.float64, order="F")
    pointer = exner.ctypes.data

    def ptr(array):
        return ffi.cast("void *", array.ctypes.data)

    ierr = lib.pycam_calc_exner_run(ncol, nz, ptr(cpair), ptr(rair), 100000.0, ptr(pmid), ptr(exner))
    assert ierr == 0
    assert exner.ctypes.data == pointer
    # NumPy and the Fortran compiler use different pow entry points, so this
    # smoke test checks the physical result and zero-copy pointer. BFB is
    # checked against the native CAM-SIMA capture, never against NumPy math.
    expected = (pmid / 100000.0) ** (rair / cpair)
    np.testing.assert_allclose(exner, expected, rtol=2.0e-15, atol=0.0)
