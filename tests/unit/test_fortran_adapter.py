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
