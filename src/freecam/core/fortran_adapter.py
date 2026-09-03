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
    character: bool = False

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
            character=bool(payload.get("character", False)),
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

    def bind(self, operation: str, pool: Mapping[str, np.ndarray], *, fcomm: int) -> "BoundCall":
        """Prepare ``operation`` on ``pool`` once, for calling many times.

        :meth:`call` does the same work on every invocation: resolve the
        entry point, validate each argument, build the pointer, rank and
        extent tables, allocate the error buffer, set the signature.  None
        of it changes while the arrays stay where they are, and a stage's
        scratch never moves -- so a stage that calls a hundred kernels a step
        was paying 135-500 microseconds of table-building per call.  The
        bound call keeps the tables and checks, on each call, only that the
        arrays are still the ones it was built on.
        """

        call, function = self._resolve(operation)
        arrays = [np.asarray(pool[argument.field]) for argument in call.arguments]
        for argument, array in zip(call.arguments, arrays):
            self._validate(argument, array)
        return BoundCall(operation, call, function, arrays, fcomm)

    def _resolve(self, operation: str):
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
        return call, function

    def call(self, operation: str, pool: Mapping[str, np.ndarray], *, fcomm: int) -> None:
        call, function = self._resolve(operation)

        arrays = [np.asarray(pool[argument.field]) for argument in call.arguments]
        for argument, array in zip(call.arguments, arrays):
            self._validate(argument, array)
        invariants = tuple(
            (int(array.ctypes.data), array.shape, array.dtype.str) for array in arrays
        )
        max_rank = max(
            (
                array.ndim + (1 if argument.character else 0)
                for argument, array in zip(call.arguments, arrays)
            ),
            default=0,
        )
        pointers = (ctypes.c_void_p * len(arrays))(
            *(ctypes.c_void_p(int(array.ctypes.data)) for array in arrays)
        )
        ndims = (ctypes.c_int32 * len(arrays))(*(array.ndim for array in arrays))
        shapes = (ctypes.c_int64 * (len(arrays) * max(1, max_rank)))()
        for field_index, array in enumerate(arrays):
            for axis, extent in enumerate(array.shape):
                shapes[field_index * max(1, max_rank) + axis] = extent
            if call.arguments[field_index].character:
                shapes[
                    field_index * max(1, max_rank) + array.ndim
                ] = array.dtype.itemsize

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


class BoundCall:
    """One operation, prepared once for arrays it will be handed again and again.

    Everything :meth:`PointerTableAdapter.call` rebuilds per call -- the
    pointer, rank and extent tables, the error buffer, the signature -- is
    built here once and kept, together with references to the arrays so they
    cannot be freed underneath the tables.  A call is then the native
    invocation and the same post-call check the adapter has always made: that
    no array's descriptor changed while Fortran had its address.
    """

    _ARGTYPES = [
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

    def __init__(self, operation: str, call: FortranCall, function: Any,
                 arrays: list[np.ndarray], fcomm: int) -> None:
        self.operation = operation
        self.arrays = list(arrays)
        self.invariants = tuple(
            (int(array.ctypes.data), array.shape, array.dtype.str) for array in self.arrays
        )
        self.max_rank = max(
            (array.ndim + (1 if argument.character else 0)
             for argument, array in zip(call.arguments, self.arrays)),
            default=0,
        )
        width = max(1, self.max_rank)
        self.pointers = (ctypes.c_void_p * len(self.arrays))(
            *(ctypes.c_void_p(pointer) for pointer, _, _ in self.invariants)
        )
        self.ndims = (ctypes.c_int32 * len(self.arrays))(*(array.ndim for array in self.arrays))
        self.shapes = (ctypes.c_int64 * (len(self.arrays) * width))()
        for index, array in enumerate(self.arrays):
            for axis, extent in enumerate(array.shape):
                self.shapes[index * width + axis] = extent
            if call.arguments[index].character:
                self.shapes[index * width + array.ndim] = array.dtype.itemsize
        self.error_buffer = ctypes.create_string_buffer(4096)
        function.argtypes = self._ARGTYPES
        function.restype = ctypes.c_int32
        self.function = function
        self.action_id = call.action_id
        self.count = len(self.arrays)
        self.fcomm = int(fcomm)

    def __call__(self) -> None:
        self.error_buffer[0] = b"\x00"
        status = int(self.function(
            self.action_id, self.count, self.pointers, self.ndims, self.shapes,
            self.max_rank, self.fcomm, self.error_buffer, len(self.error_buffer),
        ))
        if status:
            detail = self.error_buffer.value.decode(errors="replace")
            raise FortranAdapterError(
                f"native operation {self.operation!r} failed ({status}): {detail}"
            )
        for array, invariant in zip(self.arrays, self.invariants):
            if (int(array.ctypes.data), array.shape, array.dtype.str) != invariant:
                raise FortranAdapterError(
                    f"native operation {self.operation!r} changed a Python array descriptor"
                )
