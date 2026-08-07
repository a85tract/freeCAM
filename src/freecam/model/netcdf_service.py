"""Python-owned NetCDF service callbacks for standalone Fortran devices."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

from netCDF4 import Dataset
import numpy as np


_CHAR_POINTER = ctypes.POINTER(ctypes.c_char)
_INT_POINTER = ctypes.POINTER(ctypes.c_int)

_OPEN_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    _CHAR_POINTER,
    ctypes.c_int,
    _INT_POINTER,
    _CHAR_POINTER,
    ctypes.c_int,
)
_CLOSE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    _CHAR_POINTER,
    ctypes.c_int,
)
_SHAPE_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    _CHAR_POINTER,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    _INT_POINTER,
    _INT_POINTER,
    _INT_POINTER,
    _INT_POINTER,
    _CHAR_POINTER,
    ctypes.c_int,
)
_READ_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_int,
    _CHAR_POINTER,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    _INT_POINTER,
    _INT_POINTER,
    ctypes.c_void_p,
    ctypes.c_size_t,
    _CHAR_POINTER,
    ctypes.c_int,
)


def _decode_text(pointer: _CHAR_POINTER, length: int) -> str:
    return ctypes.string_at(pointer, int(length)).decode(
        "utf-8", errors="strict"
    )


def _write_error(
    pointer: _CHAR_POINTER,
    capacity: int,
    message: str,
    *,
    status: int = 1,
) -> int:
    encoded = str(message).encode("utf-8", errors="replace")
    count = min(len(encoded), max(0, int(capacity) - 1))
    if count:
        ctypes.memmove(pointer, encoded, count)
    if capacity:
        pointer[count] = b"\0"
    return int(status)


class NetCDFVariableMissing(KeyError):
    """Signal CCPP's portable missing-variable status code."""


class NetCDFCallbackService:
    """Keep NetCDF handles and ctypes callbacks alive for one device `.so`."""

    def __init__(self) -> None:
        self._datasets: dict[int, Dataset] = {}
        self._next_handle = 1
        self.open_callback = _OPEN_CALLBACK(self._open)
        self.close_callback = _CLOSE_CALLBACK(self._close)
        self.shape_callback = _SHAPE_CALLBACK(self._shape)
        self.read_int_callback = _READ_CALLBACK(self._read_int)
        self.read_real_callback = _READ_CALLBACK(self._read_real)
        self.read_char_callback = _READ_CALLBACK(self._read_char)

    def _dataset(self, handle: int) -> Dataset:
        try:
            return self._datasets[int(handle)]
        except KeyError as exc:
            raise RuntimeError(
                f"unknown Python NetCDF reader handle {handle}"
            ) from exc

    def _open(
        self,
        path_pointer,
        path_length,
        handle_pointer,
        error_pointer,
        error_length,
    ) -> int:
        try:
            path = Path(_decode_text(path_pointer, path_length)).resolve()
            dataset = Dataset(path, "r")
            dataset.set_auto_mask(False)
            dataset.set_auto_scale(False)
            dataset.set_auto_chartostring(False)
            handle = self._next_handle
            self._next_handle += 1
            self._datasets[handle] = dataset
            handle_pointer[0] = handle
            return 0
        except Exception as exc:
            return _write_error(error_pointer, error_length, exc)

    def _close(self, handle, error_pointer, error_length) -> int:
        try:
            dataset = self._datasets.pop(int(handle), None)
            if dataset is not None:
                dataset.close()
            return 0
        except Exception as exc:
            return _write_error(error_pointer, error_length, exc)

    @staticmethod
    def _selection(
        shape: tuple[int, ...],
        rank: int,
        subset: int,
        start_pointer,
        count_pointer,
    ) -> tuple[tuple[slice, ...], tuple[int, ...]]:
        if len(shape) != int(rank):
            raise ValueError(
                f"file variable rank is {len(shape)}, requested {rank}"
            )
        fortran_shape = tuple(reversed(shape))
        if not subset:
            return tuple(slice(None) for _ in shape), fortran_shape
        starts = tuple(int(start_pointer[index]) for index in range(rank))
        counts = tuple(int(count_pointer[index]) for index in range(rank))
        if any(start < 1 for start in starts):
            raise ValueError("NetCDF subset starts use one-based indices")
        if any(count < 1 for count in counts):
            raise ValueError("NetCDF subset counts must be positive")
        for start, count, extent in zip(starts, counts, fortran_shape):
            if start + count - 1 > extent:
                raise ValueError(
                    f"NetCDF subset {start}:{start + count - 1} exceeds "
                    f"dimension extent {extent}"
                )
        slices = tuple(
            slice(start - 1, start - 1 + count)
            for start, count in reversed(tuple(zip(starts, counts)))
        )
        return slices, counts

    def _variable_and_selection(
        self,
        handle: int,
        name_pointer,
        name_length: int,
        rank: int,
        subset: int,
        start_pointer,
        count_pointer,
    ) -> tuple[Any, tuple[slice, ...], tuple[int, ...]]:
        name = _decode_text(name_pointer, name_length)
        dataset = self._dataset(handle)
        try:
            variable = dataset.variables[name]
        except KeyError as exc:
            raise NetCDFVariableMissing(
                f"{Path(dataset.filepath()).name}: no variable {name!r}"
            ) from exc
        selection, dimensions = self._selection(
            tuple(variable.shape),
            int(rank),
            int(subset),
            start_pointer,
            count_pointer,
        )
        return variable, selection, dimensions

    def _shape(
        self,
        handle,
        name_pointer,
        name_length,
        rank,
        subset,
        start_pointer,
        count_pointer,
        dimensions_pointer,
        character_length_pointer,
        error_pointer,
        error_length,
    ) -> int:
        try:
            _variable, _selection, dimensions = (
                self._variable_and_selection(
                    handle,
                    name_pointer,
                    name_length,
                    rank,
                    subset,
                    start_pointer,
                    count_pointer,
                )
            )
            for index, extent in enumerate(dimensions):
                dimensions_pointer[index] = int(extent)
            character_length_pointer[0] = (
                int(dimensions[0]) if dimensions else 1
            )
            return 0
        except NetCDFVariableMissing as exc:
            return _write_error(
                error_pointer,
                error_length,
                exc,
                status=3,
            )
        except Exception as exc:
            return _write_error(error_pointer, error_length, exc)

    def _read_array(
        self,
        handle,
        name_pointer,
        name_length,
        rank,
        subset,
        start_pointer,
        count_pointer,
    ) -> np.ndarray:
        variable, selection, _dimensions = self._variable_and_selection(
            handle,
            name_pointer,
            name_length,
            rank,
            subset,
            start_pointer,
            count_pointer,
        )
        return np.asarray(variable[selection])

    def _read_numeric(
        self,
        dtype,
        handle,
        name_pointer,
        name_length,
        rank,
        subset,
        start_pointer,
        count_pointer,
        data_pointer,
        elements,
        error_pointer,
        error_length,
    ) -> int:
        try:
            values = np.ascontiguousarray(
                self._read_array(
                    handle,
                    name_pointer,
                    name_length,
                    rank,
                    subset,
                    start_pointer,
                    count_pointer,
                ),
                dtype=dtype,
            )
            if values.size != int(elements):
                raise ValueError(
                    f"NetCDF callback produced {values.size} values, "
                    f"Fortran allocated {elements}"
                )
            ctypes.memmove(data_pointer, values.ctypes.data, values.nbytes)
            return 0
        except NetCDFVariableMissing as exc:
            return _write_error(
                error_pointer,
                error_length,
                exc,
                status=3,
            )
        except Exception as exc:
            return _write_error(error_pointer, error_length, exc)

    def _read_int(self, *arguments) -> int:
        return self._read_numeric(np.int32, *arguments)

    def _read_real(self, *arguments) -> int:
        return self._read_numeric(np.float64, *arguments)

    def _read_char(
        self,
        handle,
        name_pointer,
        name_length,
        rank,
        subset,
        start_pointer,
        count_pointer,
        data_pointer,
        elements,
        error_pointer,
        error_length,
    ) -> int:
        try:
            values = self._read_array(
                handle,
                name_pointer,
                name_length,
                rank,
                subset,
                start_pointer,
                count_pointer,
            )
            if values.dtype.kind == "S":
                payload = np.ascontiguousarray(values).tobytes(order="C")
            elif values.dtype.kind == "U":
                payload = "".join(
                    np.ascontiguousarray(values).reshape(-1).tolist()
                ).encode("utf-8")
            else:
                raise TypeError(
                    f"NetCDF character variable has dtype {values.dtype}"
                )
            if len(payload) != int(elements):
                raise ValueError(
                    f"NetCDF callback produced {len(payload)} character "
                    f"bytes, Fortran allocated {elements}"
                )
            ctypes.memmove(data_pointer, payload, len(payload))
            return 0
        except NetCDFVariableMissing as exc:
            return _write_error(
                error_pointer,
                error_length,
                exc,
                status=3,
            )
        except Exception as exc:
            return _write_error(error_pointer, error_length, exc)

    def close(self) -> None:
        for dataset in tuple(self._datasets.values()):
            dataset.close()
        self._datasets.clear()


def register_netcdf_reader_callbacks(
    library: ctypes.CDLL,
) -> NetCDFCallbackService:
    """Register Python callbacks in one generated device library."""

    service = NetCDFCallbackService()
    register = library.pycam_register_netcdf_reader_callbacks
    register.argtypes = [ctypes.c_void_p] * 6
    register.restype = None
    register(
        *(
            ctypes.cast(callback, ctypes.c_void_p)
            for callback in (
                service.open_callback,
                service.close_callback,
                service.shape_callback,
                service.read_int_callback,
                service.read_real_callback,
                service.read_char_callback,
            )
        )
    )
    return service
