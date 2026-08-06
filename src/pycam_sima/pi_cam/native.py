"""Thin ctypes ABI for original iCESM CAM numerical routines."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
import sys
from typing import Mapping, Protocol

import numpy as np

from pycam_sima.core.fortran_adapter import (
    FortranAdapterError,
    PointerTableAdapter,
)

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
        self._abi = PointerTableAdapter(
            self._library,
            self._operations,
            library_name=str(self.library_path),
        )
        state_bridge = payload.get("state_bridge")
        self._state_bridge = (
            None
            if not isinstance(state_bridge, Mapping)
            else _NativeStateBridge(self._library, state_bridge, str(self.library_path))
        )

    def _call(self, operation: str, pool: PICAMStatePool, fcomm: int) -> None:
        try:
            self._abi.call(operation, pool, fcomm=fcomm)
        except FortranAdapterError as exc:
            raise NativeCAMError(str(exc)) from exc

    def initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        if "initialize" in self._operations:
            self._call("initialize", pool, fcomm)
        if self._state_bridge is not None:
            self._state_bridge.attach(pool)

    def execute(self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int) -> None:
        if self._state_bridge is not None:
            self._state_bridge.copy_to_native(pool)
        self._call(action.operation, pool, fcomm)
        if self._state_bridge is not None:
            self._state_bridge.copy_from_native(pool)

    def execute_source_step(
        self, pool: PICAMStatePool, *, fcomm: int, apply_import: bool = True
    ) -> None:
        """Execute the original CAM timestep boundary used by BFB runs."""

        operation = "source_step" if apply_import else "source_step_held_import"
        if self._state_bridge is not None:
            self._state_bridge.copy_to_native(pool)
        self._call(operation, pool, fcomm)
        if self._state_bridge is not None:
            self._state_bridge.copy_from_native(pool)

    def finalize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        if self._state_bridge is not None:
            self._state_bridge.copy_to_native(pool)
        if "finalize" in self._operations:
            self._call("finalize", pool, fcomm)


class _NativeStateBridge:
    """Synchronize Python arrays with temporary legacy CAM derived types."""

    def __init__(
        self,
        library: ctypes.CDLL,
        payload: Mapping[str, object],
        library_name: str,
    ) -> None:
        if int(payload.get("schema_version", 0)) != 1:
            raise NativeCAMError("unsupported PI-CAM native state bridge schema")
        symbols = payload.get("symbols")
        fields = payload.get("fields")
        if not isinstance(symbols, Mapping) or not isinstance(fields, list):
            raise NativeCAMError("PI-CAM native state bridge manifest is incomplete")
        self.library = library
        self.library_name = library_name
        self.fields = tuple(dict(field) for field in fields if isinstance(field, Mapping))
        if len(self.fields) != len(fields):
            raise NativeCAMError("PI-CAM native state field record is invalid")
        try:
            self._count = getattr(library, str(symbols["count"]))
            self._metadata = getattr(library, str(symbols["metadata"]))
            self._transfer = getattr(library, str(symbols["transfer"]))
        except (KeyError, AttributeError) as exc:
            raise NativeCAMError(
                f"{library_name} does not implement its declared state bridge"
            ) from exc
        self._count.argtypes = []
        self._count.restype = ctypes.c_int32
        self._metadata.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self._metadata.restype = ctypes.c_int32
        self._transfer.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
        self._transfer.restype = ctypes.c_int32
        self._active_fields: tuple[
            tuple[dict[str, object], tuple[int, ...], np.dtype], ...
        ] = ()

    def _field_metadata(self, field: Mapping[str, object]) -> tuple[np.dtype, tuple[int, ...], bool]:
        dtype_code = ctypes.c_int32()
        field_rank = ctypes.c_int32()
        active = ctypes.c_int32()
        extents = (ctypes.c_int64 * 8)()
        status = int(
            self._metadata(
                int(field["field_id"]),
                ctypes.byref(dtype_code),
                ctypes.byref(field_rank),
                extents,
                len(extents),
                ctypes.byref(active),
            )
        )
        if status:
            raise NativeCAMError(
                f"native state metadata for {field['name']!r} failed ({status})"
            )
        expected = np.dtype(str(field["dtype"]))
        if dtype_code.value not in {1, 2}:
            raise NativeCAMError(
                f"native state field {field['name']!r} has unknown dtype code "
                f"{dtype_code.value}"
            )
        actual = np.dtype("float64" if dtype_code.value == 1 else "int32")
        if actual != expected or field_rank.value != int(field["rank"]):
            raise NativeCAMError(
                f"native state contract for {field['name']!r} disagrees with manifest"
            )
        shape = tuple(int(extents[index]) for index in range(field_rank.value))
        if active.value and (not shape or min(shape) < 1):
            raise NativeCAMError(f"native state field {field['name']!r} has invalid shape")
        return expected, shape, bool(active.value)

    def _copy(self, field: Mapping[str, object], array: np.ndarray, direction: int) -> None:
        status = int(
            self._transfer(
                int(field["field_id"]),
                int(direction),
                ctypes.c_void_p(int(array.ctypes.data)),
                int(array.size),
            )
        )
        if status:
            label = "native-to-Python" if direction == 1 else "Python-to-native"
            raise NativeCAMError(
                f"{label} transfer for {field['name']!r} failed ({status})"
            )

    def attach(self, pool: PICAMStatePool) -> None:
        count = int(self._count())
        if count != len(self.fields):
            raise NativeCAMError(
                f"native state bridge exports {count} fields; manifest has {len(self.fields)}"
            )
        active: list[tuple[dict[str, object], tuple[int, ...], np.dtype]] = []
        for field in self.fields:
            dtype, shape, present = self._field_metadata(field)
            if not present:
                continue
            name = str(field["name"])
            if name in pool:
                raise NativeCAMError(f"native state field {name!r} already exists")
            pool.ensure_from_array(
                name,
                np.empty(shape, dtype=dtype, order="F"),
                category="native_cam_state",
            )
            self._copy(field, pool[name], 1)
            active.append((field, shape, dtype))
        self._active_fields = tuple(active)

    @staticmethod
    def _validated_array(
        pool: PICAMStatePool,
        field: Mapping[str, object],
        shape: tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        array = pool[str(field["name"])]
        if array.shape != shape or array.dtype != dtype:
            raise NativeCAMError(
                f"Python-owned state field {field['name']!r} changed its native contract"
            )
        if array.ndim > 1 and not array.flags.f_contiguous:
            raise NativeCAMError(
                f"Python-owned state field {field['name']!r} is no longer Fortran contiguous"
            )
        return array

    def copy_to_native(self, pool: PICAMStatePool) -> None:
        for field, shape, dtype in self._active_fields:
            self._copy(field, self._validated_array(pool, field, shape, dtype), 2)

    def copy_from_native(self, pool: PICAMStatePool) -> None:
        for field, shape, dtype in self._active_fields:
            self._copy(field, self._validated_array(pool, field, shape, dtype), 1)
