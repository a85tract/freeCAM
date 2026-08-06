"""Thin ctypes ABI for original iCESM CAM numerical routines."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys
from typing import Mapping, Protocol

import numpy as np

from .errors import NativeCAMError
from .plan import PICAMAction
from .state import PICAMStatePool


_FORTRAN_RUNTIME: ctypes.CDLL | None = None
_FORTRAN_RUNTIME_READY = False


def _prepare_fortran_runtime() -> None:
    """Initialize Intel Fortran when Python, rather than Fortran, is main."""

    global _FORTRAN_RUNTIME, _FORTRAN_RUNTIME_READY
    if _FORTRAN_RUNTIME_READY:
        return
    try:
        runtime = ctypes.CDLL("libifcore.so.5", mode=ctypes.RTLD_GLOBAL)
        initialize = runtime.for_rtl_init_
    except (OSError, AttributeError):
        _FORTRAN_RUNTIME_READY = True
        return
    initialize.argtypes = (ctypes.POINTER(ctypes.c_int),)
    initialize.restype = None
    argc = ctypes.c_int(len(sys.argv))
    initialize(ctypes.byref(argc))
    _FORTRAN_RUNTIME = runtime
    _FORTRAN_RUNTIME_READY = True


class CAMNumericalBackend(Protocol):
    def initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None: ...
    def execute(self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int) -> None: ...
    def finalize(self, pool: PICAMStatePool, *, fcomm: int) -> None: ...


class RecordingCAMBackend:
    """Deterministic no-numerics backend used for plan and boundary tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        del pool, fcomm
        self.calls.append("initialize")

    def execute(self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int) -> None:
        del pool, fcomm
        self.calls.append(action.operation)

    def finalize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        del pool, fcomm
        self.calls.append("finalize")


class NativeCAMDevice:
    """Load generated bind(C) adapters while Python retains array ownership."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest_path = Path(manifest)
        payload = json.loads(self.manifest_path.read_text())
        if int(payload.get("schema_version", 1)) != 1:
            raise NativeCAMError("unsupported native CAM manifest schema")
        library = Path(payload["library"])
        if not library.is_absolute():
            library = self.manifest_path.parent / library
        if not library.is_file():
            raise NativeCAMError(f"native CAM library does not exist: {library}")
        self.library_path = library
        _prepare_fortran_runtime()
        self._library = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        self._operations = dict(payload.get("operations", {}))

    def _call(self, operation: str, pool: PICAMStatePool, fcomm: int) -> None:
        try:
            record = self._operations[operation]
        except KeyError as exc:
            raise NativeCAMError(f"native CAM operation {operation!r} is not built") from exc
        symbol_name = str(record["symbol"])
        field_names = tuple(str(name) for name in record.get("fields", ()))
        try:
            function = getattr(self._library, symbol_name)
        except AttributeError as exc:
            raise NativeCAMError(
                f"{self.library_path} does not export {symbol_name!r}"
            ) from exc
        arrays = [pool[name] for name in field_names]
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
        action_id = int(record.get("action_id", 0))
        status = int(
            function(
                action_id,
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
            message = error_buffer.value.decode(errors="replace")
            raise NativeCAMError(
                f"native CAM operation {operation!r} failed ({status}): {message}"
            )

    def initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        if "initialize" in self._operations:
            self._call("initialize", pool, fcomm)

    def execute(self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int) -> None:
        self._call(action.operation, pool, fcomm)

    def execute_source_step(
        self, pool: PICAMStatePool, *, fcomm: int, apply_import: bool = True
    ) -> None:
        """Execute the original CAM timestep boundary used by BFB runs."""

        operation = "source_step" if apply_import else "source_step_held_import"
        self._call(operation, pool, fcomm)

    def finalize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        if "finalize" in self._operations:
            self._call("finalize", pool, fcomm)
