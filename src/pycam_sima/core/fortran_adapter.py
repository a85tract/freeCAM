"""Manifest-driven pointer-table ABI for Python-owned Fortran arguments."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class FortranAdapterError(RuntimeError):
    """A manifest or native call violated the stable pointer-table ABI."""


@dataclass(frozen=True, slots=True)
class FortranArgument:
    """One Python-owned array passed through the generic ABI."""

    field: str
    dtype: str | None = None
    rank: int | None = None
    intent: str = "inout"
    contiguous: str = "F"

    @classmethod
    def from_payload(cls, payload: str | Mapping[str, Any]) -> "FortranArgument":
        if isinstance(payload, str):
            return cls(field=payload)
        intent = str(payload.get("intent", "inout")).lower()
        if intent not in {"in", "out", "inout"}:
            raise FortranAdapterError(f"invalid Fortran argument intent {intent!r}")
        contiguous = str(payload.get("contiguous", "F")).upper()
        if contiguous not in {"F", "C", "ANY"}:
            raise FortranAdapterError(
                f"invalid contiguous requirement {contiguous!r}"
            )
        dtype = payload.get("dtype")
        rank = payload.get("rank")
        return cls(
            field=str(payload["field"]),
            dtype=None if dtype is None else np.dtype(str(dtype)).str,
            rank=None if rank is None else int(rank),
            intent=intent,
            contiguous=contiguous,
        )


@dataclass(frozen=True, slots=True)
class FortranCall:
    symbol: str
    action_id: int
    arguments: tuple[FortranArgument, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "FortranCall":
        raw_arguments = payload.get("arguments")
        if raw_arguments is None:
            # Schema-v1 PI-CAM manifests used a compact list of field names.
            raw_arguments = payload.get("fields", ())
        return cls(
            symbol=str(payload["symbol"]),
            action_id=int(payload.get("action_id", 0)),
            arguments=tuple(
                FortranArgument.from_payload(argument)
                for argument in raw_arguments
            ),
        )


class PointerTableAdapter:
    """Invoke any generated ``bind(C)`` pointer-table entry point.

    The native function has one stable ABI regardless of the scientific
    routine's original Fortran signature.  Its manifest selects StatePool
    arrays and supplies their addresses, ranks, and extents without copies.
    """

    def __init__(
        self,
        library: ctypes.CDLL,
        operations: Mapping[str, Mapping[str, Any]],
        *,
        library_name: str,
    ) -> None:
        self.library = library
        self.library_name = str(library_name)
        self.operations = {
            str(name): FortranCall.from_payload(payload)
            for name, payload in operations.items()
        }

    @staticmethod
    def _validate(argument: FortranArgument, array: np.ndarray) -> None:
        if argument.dtype is not None and array.dtype != np.dtype(argument.dtype):
            raise FortranAdapterError(
                f"field {argument.field!r} has dtype {array.dtype}; "
                f"expected {np.dtype(argument.dtype)}"
            )
        if argument.rank is not None and array.ndim != argument.rank:
            raise FortranAdapterError(
                f"field {argument.field!r} has rank {array.ndim}; "
                f"expected {argument.rank}"
            )
        if argument.contiguous == "F" and not array.flags.f_contiguous:
            raise FortranAdapterError(
                f"field {argument.field!r} must be Fortran contiguous"
            )
        if argument.contiguous == "C" and not array.flags.c_contiguous:
            raise FortranAdapterError(
                f"field {argument.field!r} must be C contiguous"
            )
        if argument.intent in {"out", "inout"} and not array.flags.writeable:
            raise FortranAdapterError(
                f"field {argument.field!r} is read-only but intent is {argument.intent}"
            )

    def call(self, operation: str, pool: Mapping[str, np.ndarray], *, fcomm: int) -> None:
        try:
            call = self.operations[operation]
        except KeyError as exc:
            raise FortranAdapterError(
                f"native operation {operation!r} is not declared"
            ) from exc
        try:
            function = getattr(self.library, call.symbol)
        except AttributeError as exc:
            raise FortranAdapterError(
                f"{self.library_name} does not export {call.symbol!r}"
            ) from exc

        arrays = [np.asarray(pool[argument.field]) for argument in call.arguments]
        for argument, array in zip(call.arguments, arrays):
            self._validate(argument, array)
        invariants = tuple(
            (int(array.ctypes.data), array.shape, array.dtype.str) for array in arrays
        )
        max_rank = max((array.ndim for array in arrays), default=0)
        pointers = (ctypes.c_void_p * len(arrays))(
            *(ctypes.c_void_p(int(array.ctypes.data)) for array in arrays)
        )
        ndims = (ctypes.c_int32 * len(arrays))(*(array.ndim for array in arrays))
        shapes = (ctypes.c_int64 * (len(arrays) * max(1, max_rank)))()
        for field_index, array in enumerate(arrays):
            for axis, extent in enumerate(array.shape):
                shapes[field_index * max(1, max_rank) + axis] = extent

        error_buffer = ctypes.create_string_buffer(4096)
        function.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        function.restype = ctypes.c_int32
        status = int(
            function(
                call.action_id,
                len(arrays),
                pointers,
                ndims,
                shapes,
                max_rank,
                int(fcomm),
                error_buffer,
                len(error_buffer),
            )
        )
        if status:
            detail = error_buffer.value.decode(errors="replace")
            raise FortranAdapterError(
                f"native operation {operation!r} failed ({status}): {detail}"
            )
        for array, invariant in zip(arrays, invariants):
            actual = (int(array.ctypes.data), array.shape, array.dtype.str)
            if actual != invariant:
                raise FortranAdapterError(
                    f"native operation {operation!r} changed a Python array descriptor"
                )
