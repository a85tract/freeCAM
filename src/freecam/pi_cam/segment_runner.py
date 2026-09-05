"""The image's segment runner for tphysbc stage 7, spoken from Python.

``pycam_stage7_runner`` (native/pi_cam/support/pycam_stage7_runner.F90) runs
the original stage-7 Fortran and pauses at ``mmacro_pcond`` when Python
says it is replaced.  This module binds its seven ``bind(C)`` entries and
presents them as the :class:`~freecam.physics.segments.SegmentRunner` a
:class:`~freecam.physics.segments.SegmentedStage` drives: a start, a frame
describing the paused call's arguments as views of the Fortran storage, a
resume.  The frame's argument names and intents come from the same reviewed
descriptor the direct kernel is built from, in the call's order.
"""

from __future__ import annotations

import ctypes
from typing import Any, Mapping

import numpy as np

from freecam.physics.segments import FrameArgument, KernelFrame, SegmentEvent
from .errors import NativeCAMError
from .kernel_codegen import load_direct_kernels

STAGE = "cam_run1.cloud_macro_microphysics"
KERNELS = ("mmacro_pcond",)            # kernel id = position + 1, as the module numbers them
DTYPES = {1: np.float64, 2: np.int32}
INTENTS = {0: "in", 1: "out", 2: "inout"}
ENTRIES = ("pycam_stage7_create_v1", "pycam_stage7_start_v1", "pycam_stage7_frame_v1",
           "pycam_stage7_resume_v1", "pycam_stage7_error_v1", "pycam_stage7_reset_v1",
           "pycam_stage7_destroy_v1")


def image_offers_runner(library: Any) -> bool:
    """Whether ``library`` exports the stage-7 runner's entries."""

    return all(hasattr(library, name) for name in ENTRIES)


class StageSevenRunner:
    """The ``pycam_stage7_*`` entries as a SegmentRunner."""

    #: the kernels this runner can pause at; a stage chooses segmented
    #: execution under ``auto`` only when every replaced kernel is one
    kernels = KERNELS

    def __init__(self, library: Any, descriptors: str | Any) -> None:
        self.library = library
        kernels = {k.name: k for k in load_direct_kernels(descriptors)}
        arguments = kernels["mmacro_pcond"].arguments
        #: the frame's argument names in the call's order, without the stage prefix
        self.names = tuple(a.field.split(".", 1)[1] for a in arguments)
        self.slots = len(self.names)
        self._bind()

    def _bind(self) -> None:
        lib = self.library
        c_int, c_int64, c_ptr = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
        pi = ctypes.POINTER(c_int)
        for name in ENTRIES:
            getattr(lib, name).restype = c_int
        lib.pycam_stage7_create_v1.argtypes = [pi]
        lib.pycam_stage7_start_v1.argtypes = [c_int, c_int, pi, pi]
        lib.pycam_stage7_frame_v1.argtypes = [c_int, pi, pi, pi, pi, pi, pi, c_int,
                                              ctypes.POINTER(c_ptr), pi, ctypes.POINTER(c_int64), pi, pi]
        lib.pycam_stage7_resume_v1.argtypes = [c_int, c_int, c_int, pi]
        lib.pycam_stage7_error_v1.argtypes = [c_int, ctypes.c_char_p, c_int]
        lib.pycam_stage7_reset_v1.argtypes = [c_int]
        lib.pycam_stage7_destroy_v1.argtypes = [c_int]

    # -- the SegmentRunner protocol -------------------------------------------

    def create(self, stage: str) -> int:
        if stage != STAGE:
            raise NativeCAMError(f"the image's stage-7 runner does not run {stage!r}")
        context = ctypes.c_int(0)
        status = self.library.pycam_stage7_create_v1(ctypes.byref(context))
        if status:
            raise NativeCAMError(f"stage 7 runner refused to create a context ({status}): {self.error(0)}")
        return int(context.value)

    def start(self, context: int, mask: Mapping[str, bool]) -> SegmentEvent:
        flags = (ctypes.c_int * len(KERNELS))(*(1 if mask.get(name) else 0 for name in KERNELS))
        event = ctypes.c_int(2)
        status = self.library.pycam_stage7_start_v1(context, len(KERNELS), flags, ctypes.byref(event))
        if status:
            raise NativeCAMError(f"stage 7 runner refused to start ({status}): {self.error(context)}")
        return SegmentEvent(event.value)

    def frame(self, context: int) -> KernelFrame:
        n = self.slots
        kernel, index, lchnk, ncol, substep, token = (ctypes.c_int(0) for _ in range(6))
        pointers = (ctypes.c_void_p * n)()
        ndims = (ctypes.c_int * n)()
        shapes = (ctypes.c_int64 * (3 * n))()
        dtypes = (ctypes.c_int * n)()
        intents = (ctypes.c_int * n)()
        status = self.library.pycam_stage7_frame_v1(
            context, ctypes.byref(kernel), ctypes.byref(index), ctypes.byref(lchnk), ctypes.byref(ncol),
            ctypes.byref(substep), ctypes.byref(token), n, pointers, ndims, shapes, dtypes, intents)
        if status:
            raise NativeCAMError(f"stage 7 runner has no frame ({status}): {self.error(context)}")
        arguments = []
        for i, name in enumerate(self.names):
            rank = ndims[i]
            shape = tuple(int(shapes[3 * i + axis]) for axis in range(rank))
            dtype = np.dtype(DTYPES[dtypes[i]])
            count = int(np.prod(shape)) if shape else 1
            buffer = (ctypes.c_byte * (count * dtype.itemsize)).from_address(pointers[i])
            array = np.ndarray(shape, dtype=dtype, buffer=buffer, order="F")
            arguments.append(FrameArgument(name, array, INTENTS[intents[i]]))
        return KernelFrame(kernel=KERNELS[kernel.value - 1], call_index=index.value, lchnk=lchnk.value,
                           ncol=ncol.value, substep=substep.value, arguments=tuple(arguments),
                           token=token.value)

    def resume(self, context: int, kernel: str, token: int) -> SegmentEvent:
        event = ctypes.c_int(2)
        status = self.library.pycam_stage7_resume_v1(context, KERNELS.index(kernel) + 1, int(token),
                                                     ctypes.byref(event))
        if status:
            raise NativeCAMError(f"stage 7 runner refused to resume ({status}): {self.error(context)}")
        return SegmentEvent(event.value)

    def error(self, context: int) -> str:
        buffer = ctypes.create_string_buffer(256)
        self.library.pycam_stage7_error_v1(context, buffer, len(buffer))
        return buffer.value.decode(errors="replace")

    def reset(self, context: int) -> None:
        self.library.pycam_stage7_reset_v1(context)

    def destroy(self, context: int) -> None:
        self.library.pycam_stage7_destroy_v1(context)


__all__ = ["ENTRIES", "KERNELS", "STAGE", "StageSevenRunner", "image_offers_runner"]
