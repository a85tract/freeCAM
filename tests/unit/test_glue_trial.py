"""The compiled direct callers, when built: the same calls as ctypes makes, checked against ctypes callbacks."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

_glue = pytest.importorskip("freecam.core._glue")


def test_the_kernel_call_hands_the_tables_it_was_given() -> None:
    seen: list[tuple] = []
    CB = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
                          ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64), ctypes.c_int32,
                          ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32)

    @CB
    def kernel(action, nfields, pointers, ndims, shapes, max_rank, fcomm, error, error_len):
        seen.append((action, nfields, pointers[0], pointers[1], ndims[0], shapes[0], shapes[1], max_rank, fcomm, error_len))
        return 7

    x = np.zeros((4, 3), order="F")
    pointers = (ctypes.c_void_p * 2)(x.ctypes.data, x.ctypes.data + 8)
    ndims = (ctypes.c_int32 * 2)(2, 2)
    shapes = (ctypes.c_int64 * 4)(4, 3, 4, 3)
    error = ctypes.create_string_buffer(64)
    call = _glue.KernelCall(_glue.address_of(kernel), 5, 2, ctypes.addressof(pointers), ctypes.addressof(ndims),
                            ctypes.addressof(shapes), 2, 9, ctypes.addressof(error), 64, (pointers, ndims, shapes, error))
    assert call() == 7
    assert seen == [(5, 2, x.ctypes.data, x.ctypes.data + 8, 2, 4, 3, 2, 9, 64)]
    pointers[1] = x.ctypes.data + 16                          # retargeted through the same table
    call()
    assert seen[-1][3] == x.ctypes.data + 16


def test_the_probes_answer_address_rank_and_shape() -> None:
    x = np.zeros((4, 3, 2), order="F")
    P2 = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p),
                          ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int64))

    @P2
    def view(lchnk, code, ptr, ndims, extents):
        if code != 1:
            return 2
        ptr[0] = x.ctypes.data
        ndims[0] = 3
        extents[0], extents[1], extents[2] = 4, 3, 2
        return 0

    probe = _glue.Probe2(_glue.address_of(view))
    assert probe(10, 1) == (0, x.ctypes.data, 3, (4, 3, 2))
    assert probe(10, 2)[0] == 2
    B1 = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                          ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64))
    B2 = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                          ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32),
                          ctypes.POINTER(ctypes.c_int64))

    @B1
    def plane(chunk, index, sliced, ptr, extents):
        ptr[0] = x.ctypes.data; extents[0], extents[1] = 4, 3
        return 0

    @B2
    def any_field(chunk, index, sliced, rank, integer, ptr, ndims, extents):
        ptr[0] = x.ctypes.data; ndims[0] = rank
        for i in range(rank):
            extents[i] = x.shape[i]
        return 0

    buffer = _glue.PBufProbe(_glue.address_of(plane), _glue.address_of(any_field))
    assert buffer.plane(10, 5, 1) == (0, x.ctypes.data, (4, 3))
    assert buffer.any(10, 5, 0, 3, 0) == (0, x.ctypes.data, 3, (4, 3, 2))


def test_the_history_call_passes_the_name_and_the_address() -> None:
    seen: list[tuple] = []
    OF = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_double),
                          ctypes.c_int32, ctypes.c_int32)

    @OF
    def outfld(name, n, data, idim, lchnk):
        seen.append((name[:n], n, data[0], idim, lchnk))
        return 0

    x = np.full((4,), 2.5)
    call = _glue.Outfld(_glue.address_of(outfld))
    assert call(b"FIELD   ", 8, x.ctypes.data, 16, 10) == 0
    assert seen == [(b"FIELD   ", 8, 2.5, 16, 10)]
