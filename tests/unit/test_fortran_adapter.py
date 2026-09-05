from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from freecam.core import FortranAdapterError, PointerTableAdapter


_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_char_p,
    ctypes.c_int32,
)


def test_pointer_table_adapter_builds_zero_copy_fortran_contract() -> None:
    observed: dict[str, object] = {}

    @_CALLBACK
    def kernel(action, nfields, pointers, ndims, shapes, max_rank, fcomm, error, error_len):
        del error, error_len
        observed.update(
            action=action,
            nfields=nfields,
            rank=ndims[0],
            shape=(shapes[0], shapes[1]),
            max_rank=max_rank,
            fcomm=fcomm,
        )
        values = np.ctypeslib.as_array(
            (ctypes.c_double * (shapes[0] * shapes[1])).from_address(pointers[0])
        )
        values += action
        return 0

    library = SimpleNamespace(pycam_test_v1=kernel)
    adapter = PointerTableAdapter(
        library,  # type: ignore[arg-type]
        {
            "run": {
                "symbol": "pycam_test_v1",
                "action_id": 3,
                "arguments": [
                    {
                        "field": "temperature",
                        "dtype": "float64",
                        "rank": 2,
                        "intent": "inout",
                    }
                ],
            }
        },
        library_name="test.so",
    )
    values = np.zeros((2, 3), order="F")

    adapter.call("run", {"temperature": values}, fcomm=17)

    assert observed == {
        "action": 3,
        "nfields": 1,
        "rank": 2,
        "shape": (2, 3),
        "max_rank": 2,
        "fcomm": 17,
    }
    assert np.array_equal(values, np.full((2, 3), 3.0))


def test_pointer_table_adapter_rejects_wrong_dtype_before_native_call() -> None:
    library = SimpleNamespace(pycam_test_v1=_CALLBACK(lambda *args: 0))
    adapter = PointerTableAdapter(
        library,  # type: ignore[arg-type]
        {
            "run": {
                "symbol": "pycam_test_v1",
                "arguments": [{"field": "x", "dtype": "float64"}],
            }
        },
        library_name="test.so",
    )

    with pytest.raises(FortranAdapterError, match="dtype"):
        adapter.call("run", {"x": np.zeros(2, dtype=np.float32)}, fcomm=0)


def test_a_bound_call_marshals_once_and_runs_the_kernel_every_time() -> None:
    """bind() is call() with the tables built once.

    A stage hands the same scratch arrays to the same kernel on every chunk
    of every step, so the pointer, rank and extent tables need building
    once; each call then only invokes and re-checks that no array's
    descriptor changed while Fortran had its address -- the check the
    one-shot path has always made after the call.
    """

    seen: list[tuple] = []

    @_CALLBACK
    def kernel(action, nfields, pointers, ndims, shapes, max_rank, fcomm, error, error_len):
        del error, error_len
        seen.append((action, nfields, ndims[0], (shapes[0], shapes[1]), max_rank, fcomm))
        values = np.ctypeslib.as_array(
            (ctypes.c_double * (shapes[0] * shapes[1])).from_address(pointers[0])
        )
        values += action
        return 0

    adapter = PointerTableAdapter(
        SimpleNamespace(pycam_test_v1=kernel),  # type: ignore[arg-type]
        {"run": {"symbol": "pycam_test_v1", "action_id": 3, "arguments": [
            {"field": "temperature", "dtype": "float64", "rank": 2, "intent": "inout"}]}},
        library_name="fake",
    )
    temperature = np.zeros((4, 3), order="F")
    bound = adapter.bind("run", {"temperature": temperature}, fcomm=7)
    bound()
    bound()

    assert seen == [(3, 1, 2, (4, 3), 2, 7)] * 2
    assert np.all(temperature == 6.0)            # the same storage, written in place twice


def test_a_bound_call_refuses_an_array_whose_descriptor_moved() -> None:
    @_CALLBACK
    def kernel(action, nfields, pointers, ndims, shapes, max_rank, fcomm, error, error_len):
        del action, nfields, pointers, ndims, shapes, max_rank, fcomm, error, error_len
        return 0

    adapter = PointerTableAdapter(
        SimpleNamespace(pycam_test_v1=kernel),  # type: ignore[arg-type]
        {"run": {"symbol": "pycam_test_v1", "action_id": 1, "arguments": [
            {"field": "x", "dtype": "float64", "rank": 1, "intent": "inout"}]}},
        library_name="fake",
    )
    x = np.zeros(6)
    bound = adapter.bind("run", {"x": x}, fcomm=0)
    bound()
    x.shape = (2, 3)                              # the tables still describe (6,)
    with pytest.raises(FortranAdapterError, match="changed a Python array descriptor"):
        bound()


def test_a_bound_call_can_be_pointed_at_another_array_of_the_same_form() -> None:
    """retarget() moves one argument's pointer without rebuilding the tables."""

    seen: list[int] = []

    @_CALLBACK
    def kernel(action, nfields, pointers, ndims, shapes, max_rank, fcomm, error, error_len):
        del action, nfields, ndims, shapes, max_rank, fcomm, error, error_len
        seen.append(pointers[0])
        return 0

    adapter = PointerTableAdapter(
        SimpleNamespace(pycam_test_v1=kernel),  # type: ignore[arg-type]
        {"run": {"symbol": "pycam_test_v1", "action_id": 1, "arguments": [
            {"field": "x", "dtype": "float64", "rank": 2, "intent": "in"}]}},
        library_name="fake",
    )
    first = np.zeros((4, 3), order="F")
    second = np.ones((4, 3), order="F")
    bound = adapter.bind("run", {"x": first}, fcomm=0)
    bound()
    bound.retarget(0, second)
    bound()
    assert seen == [first.ctypes.data, second.ctypes.data]
    assert bound.arrays[0] is second                       # the post-call check follows it
    with pytest.raises(FortranAdapterError, match="cannot be retargeted"):
        bound.retarget(0, np.zeros((3, 4), order="F"))     # another shape: the extents table would lie
    with pytest.raises(FortranAdapterError, match="cannot be retargeted"):
        bound.retarget(0, np.zeros((4, 3)))                # C order: not what the tables describe
