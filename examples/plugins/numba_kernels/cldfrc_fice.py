"""cldfrc_fice as a Python kernel for the hook: the ice and snow fractions from temperature.

Bound with ``--kernel-plugin cldfrc_fice=examples/plugins/numba_kernels/cldfrc_fice.py:cldfrc_fice``,
the image compiles this with Numba and calls it at cloud_fraction::cldfrc_fice's hook on every call,
with no Python in the step.  The arrays are the hook's: ``t`` in, ``fice`` and ``fsnow`` out, indexed
``[column, level]``, all columns of the chunk including the padding ones (the hook writes back the
live columns only).  The arithmetic is the original's, statement for statement, so a run with this
kernel in the slot can be compared bit-for-bit with the oracle.
"""
import numpy as np

TMELT = 273.15          # shr_const_tkfrz
TMAX_FICE = TMELT - 10.0
TMIN_FICE = TMAX_FICE - 30.0
TMAX_FSNOW = TMELT
TMIN_FSNOW = TMELT - 5.0


def cldfrc_fice(t, fice, fsnow):
    ncol, pver = t.shape
    for k in range(pver):
        for i in range(ncol):
            tik = t[i, k]
            if tik > TMAX_FICE:
                fice[i, k] = 0.0
            elif tik < TMIN_FICE:
                fice[i, k] = 1.0
            else:
                fice[i, k] = (TMAX_FICE - tik) / (TMAX_FICE - TMIN_FICE)
            if tik > TMAX_FSNOW:
                fsnow[i, k] = 0.0
            elif tik < TMIN_FSNOW:
                fsnow[i, k] = 1.0
            else:
                fsnow[i, k] = (TMAX_FSNOW - tik) / (TMAX_FSNOW - TMIN_FSNOW)


def cldfrc_fice_reference(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The same in NumPy, for checking the compiled kernel offline."""

    fice = np.where(t > TMAX_FICE, 0.0, np.where(t < TMIN_FICE, 1.0, (TMAX_FICE - t) / (TMAX_FICE - TMIN_FICE)))
    fsnow = np.where(t > TMAX_FSNOW, 0.0, np.where(t < TMIN_FSNOW, 1.0, (TMAX_FSNOW - t) / (TMAX_FSNOW - TMIN_FSNOW)))
    return fice, fsnow
