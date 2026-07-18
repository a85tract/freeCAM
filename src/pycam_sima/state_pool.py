from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    standard_name: str
    dtype: np.dtype[Any]
    dimensions: tuple[str, ...]
    intent: str = "inout"
    lifetime: str = "run"
    owner: str = "python"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.owner not in {"python", "native_view"}:
            raise ValueError("owner must be python or native_view")
        if self.intent not in {"in", "out", "inout"}:
            raise ValueError(f"invalid intent {self.intent!r}")


class StatePool(MutableMapping[str, np.ndarray[Any, Any]]):
    """Python state registry keyed by CCPP standard name.

    Kernel-mode fields are allocated by Python.  Full-CAM fields are zero-copy
    NumPy views of CAM's long-lived allocations and are marked ``native_view``.
    In both modes users inspect and edit the same arrays through this registry.
    """

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray[Any, Any]] = {}
        self._specs: dict[str, FieldSpec] = {}

    def allocate(
        self,
        spec: FieldSpec,
        shape: tuple[int, ...],
        *,
        fill: float | int | None = 0,
    ) -> np.ndarray[Any, Any]:
        if spec.standard_name in self._arrays:
            raise KeyError(f"field already allocated: {spec.standard_name}")
        array = np.empty(shape, dtype=spec.dtype, order="F")
        if fill is not None:
            array.fill(fill)
        self._arrays[spec.standard_name] = array
        self._specs[spec.standard_name] = spec
        return array

    def register(
        self, spec: FieldSpec, array: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        if not isinstance(array, np.ndarray):
            raise TypeError("StatePool fields must be NumPy arrays")
        if array.dtype != spec.dtype:
            raise TypeError(
                f"{spec.standard_name}: expected {spec.dtype}, got {array.dtype}"
            )
        if array.ndim > 1 and not array.flags.f_contiguous:
            raise ValueError(f"{spec.standard_name} is not Fortran-contiguous")
        self._arrays[spec.standard_name] = array
        self._specs[spec.standard_name] = spec
        return array

    def require(self, name: str) -> np.ndarray[Any, Any]:
        try:
            return self._arrays[name]
        except KeyError as exc:
            raise KeyError(f"required state field is missing: {name}") from exc

    def spec(self, name: str) -> FieldSpec:
        return self._specs[name]

    def release_scope(self, lifetime: str) -> None:
        for name in [n for n, spec in self._specs.items() if spec.lifetime == lifetime]:
            del self._arrays[name]
            del self._specs[name]

    def validate(self) -> None:
        for name, array in self._arrays.items():
            spec = self._specs[name]
            if array.dtype != spec.dtype:
                raise TypeError(f"{name}: dtype changed from {spec.dtype} to {array.dtype}")
            if array.ndim > 1 and not array.flags.f_contiguous:
                raise ValueError(f"{name}: array is no longer Fortran-contiguous")

    def pointer(self, name: str) -> int:
        return int(self.require(name).__array_interface__["data"][0])

    @contextmanager
    def callback_access(self, writable: bool) -> Iterator[None]:
        previous = {name: array.flags.writeable for name, array in self._arrays.items()}
        try:
            if not writable:
                for array in self._arrays.values():
                    array.flags.writeable = False
            yield
        finally:
            for name, was_writeable in previous.items():
                self._arrays[name].flags.writeable = was_writeable

    def __getitem__(self, key: str) -> np.ndarray[Any, Any]:
        return self.require(key)

    def __setitem__(self, key: str, value: np.ndarray[Any, Any]) -> None:
        if key not in self._specs:
            raise KeyError("register new fields with StatePool.register")
        self.register(self._specs[key], value)

    def __delitem__(self, key: str) -> None:
        del self._arrays[key]
        del self._specs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)
