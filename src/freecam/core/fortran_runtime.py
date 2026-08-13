"""Process-wide initialization for Intel Fortran libraries loaded by Python."""

from __future__ import annotations

import ctypes
import sys


_RUNTIME: ctypes.CDLL | None = None
_READY = False


def prepare_fortran_runtime() -> None:
    """Run Intel's generated-main initialization at most once per process."""

    global _READY, _RUNTIME
    if _READY:
        return
    try:
        runtime = ctypes.CDLL("libifcore.so.5", mode=ctypes.RTLD_GLOBAL)
        initialize = runtime.for_rtl_init_
    except (OSError, AttributeError):
        _READY = True
        return
    initialize.argtypes = (ctypes.POINTER(ctypes.c_int),)
    initialize.restype = None
    argc = ctypes.c_int(len(sys.argv))
    initialize(ctypes.byref(argc))
    _RUNTIME = runtime
    _READY = True
