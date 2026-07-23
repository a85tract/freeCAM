"""Manifest-driven zero-copy connection between StatePool and Fortran devices."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import DeviceContractError, MissingKernelError


DEVICE_ABI_VERSION = 1
_DTYPES: Mapping[str, tuple[Any, Any]] = {
    "float64": (np.dtype("float64"), ctypes.c_double),
    "int32": (np.dtype("int32"), ctypes.c_int),
    "bool": (np.dtype("bool"), ctypes.c_bool),
}
_UNIT_ALIASES = {
    "%": "percent",
    "1": "1",
    "count": "count",
}


def _normalized_units(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().split())
    return _UNIT_ALIASES.get(normalized, normalized)


class FortranDevice:
    """One generated adapter library plus its machine-readable contract."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path).resolve()
        try:
            payload = json.loads(self.manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DeviceContractError(
                f"cannot read device manifest {self.manifest_path}: {exc}"
            ) from exc
        if payload.get("schema_version") != 1:
            raise DeviceContractError(
                f"{self.manifest_path}: unsupported manifest schema "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("abi_version") != DEVICE_ABI_VERSION:
            raise DeviceContractError(
                f"{self.manifest_path}: unsupported device ABI "
                f"{payload.get('abi_version')!r}"
            )
        self.manifest = payload
        self.name = str(payload["name"])
        self.state_policy = str(payload["state_policy"])
        self.initialize_entrypoint = payload.get("initialize_entrypoint")
        self.entrypoints = dict(payload["entrypoints"])
        self.processes = dict(payload["processes"])
        self.dimension_bindings = dict(payload["dimension_bindings"])
        library = self.manifest_path.parent / str(payload["library"])
        if not library.is_file():
            raise MissingKernelError(
                f"device {self.name!r} library does not exist: {library}"
            )
        self.library_path = library
        self.lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
        self._functions: dict[str, Any] = {}
        self._abi_checked = False
        self._initialized = False

    @property
    def source_hash(self) -> str:
        return str(self.manifest["source"]["sha256"])

    def _ensure_abi(self) -> None:
        if self._abi_checked:
            return
        try:
            version = self.lib.pycam_device_abi_version
        except AttributeError as exc:
            raise DeviceContractError(
                f"device {self.name!r} does not export "
                "'pycam_device_abi_version'"
            ) from exc
        version.argtypes = []
        version.restype = ctypes.c_int
        if version() != DEVICE_ABI_VERSION:
            raise DeviceContractError(
                f"device {self.name!r} returned an incompatible ABI version"
            )
        self._abi_checked = True

    def _function(self, entrypoint: str):
        if entrypoint in self._functions:
            return self._functions[entrypoint]
        try:
            contract = self.entrypoints[entrypoint]
        except KeyError as exc:
            raise DeviceContractError(
                f"device {self.name!r} has no entrypoint {entrypoint!r}"
            ) from exc
        try:
            function = getattr(self.lib, contract["symbol"])
        except AttributeError as exc:
            raise DeviceContractError(
                f"device {self.name!r} does not export "
                f"{contract['symbol']!r}"
            ) from exc
        argtypes = [
            self._ctypes_argument(argument)
            for argument in contract["arguments"]
        ]
        argtypes.extend(
            [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
        )
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        self._functions[entrypoint] = function
        return function

    @staticmethod
    def _ctypes_argument(argument: Mapping[str, Any]) -> Any:
        try:
            numpy_dtype, ctype = _DTYPES[str(argument["dtype"])]
        except KeyError as exc:
            raise DeviceContractError(
                f"unsupported ABI dtype {argument.get('dtype')!r}"
            ) from exc
        rank = int(argument["rank"])
        passing = str(argument["passing"])
        if rank:
            return np.ctypeslib.ndpointer(
                dtype=numpy_dtype,
                ndim=rank,
                flags=("F_CONTIGUOUS", "ALIGNED"),
            )
        if passing == "value":
            return ctype
        return ctypes.POINTER(ctype)

    def _expected_shape(
        self, argument: Mapping[str, Any], pool: Any
    ) -> tuple[int, ...]:
        shape: list[int] = []
        for standard_name in argument["dimensions"]:
            if str(standard_name).isdigit():
                shape.append(int(standard_name))
                continue
            try:
                pool_dimension = self.dimension_bindings[standard_name]
                shape.append(int(pool.dimensions[pool_dimension]))
            except KeyError as exc:
                raise DeviceContractError(
                    f"device {self.name!r} cannot resolve dimension "
                    f"{standard_name!r}"
                ) from exc
        return tuple(shape)

    def _resolve_field(
        self, argument: Mapping[str, Any], pool: Any
    ) -> np.ndarray:
        binding = argument["binding"]
        source = binding["source"]
        if source == "field":
            field_name = str(binding["name"])
            values = pool.get(field_name)
            contract = pool.contract(field_name)
        elif source == "standard_name":
            standard_name = str(binding["name"])
            field_name = pool.ccpp_field_name(standard_name)
            values = pool.get(field_name)
            contract = pool.contract(field_name)
        else:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']}: binding source "
                f"{source!r} is not a StatePool field"
            )
        expected_dtype = _DTYPES[str(argument["dtype"])][0]
        expected_shape = self._expected_shape(argument, pool)
        if values.dtype != expected_dtype:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} maps to {field_name!r} "
                f"with dtype {values.dtype}, expected {expected_dtype}"
            )
        if values.shape != expected_shape:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} maps to {field_name!r} "
                f"with shape {values.shape}, expected {expected_shape}"
            )
        if values.ndim and not values.flags.f_contiguous:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} maps to non-Fortran-"
                f"contiguous field {field_name!r}"
            )
        if argument["intent"] in {"out", "inout"} and not values.flags.writeable:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} maps an output to "
                f"read-only field {field_name!r}"
            )
        expected_units = _normalized_units(argument["units"])
        actual_units = _normalized_units(contract.units)
        if expected_units not in {"", "none", "count"} and (
            actual_units != expected_units
        ):
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} maps to {field_name!r} "
                f"with units {contract.units!r}, expected "
                f"{argument['units']!r}"
            )
        return values

    def _resolve_argument(
        self, argument: Mapping[str, Any], pool: Any
    ) -> Any:
        binding = argument["binding"]
        source = binding["source"]
        dtype, ctype = _DTYPES[str(argument["dtype"])]
        if source == "dimension":
            value: Any = int(pool.dimensions[str(binding["name"])])
        elif source == "literal":
            value = binding["value"]
        else:
            values = self._resolve_field(argument, pool)
            if int(argument["rank"]):
                return values
            if argument["passing"] == "reference":
                return values.ctypes.data_as(ctypes.POINTER(ctype))
            value = values.item()
        if dtype.kind in {"i", "u"}:
            return int(value)
        if dtype.kind == "b":
            return bool(value)
        return float(value)

    def call(self, entrypoint: str, pool: Any) -> None:
        self._ensure_abi()
        try:
            contract = self.entrypoints[entrypoint]
        except KeyError as exc:
            raise DeviceContractError(
                f"device {self.name!r} has no entrypoint {entrypoint!r}"
            ) from exc
        arguments = [
            self._resolve_argument(argument, pool)
            for argument in contract["arguments"]
        ]
        message = ctypes.create_string_buffer(2048)
        status = self._function(entrypoint)(
            *arguments, message, len(message)
        )
        if status:
            detail = message.value.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"device {self.name!r} entrypoint {entrypoint!r} failed "
                f"with code {status}{suffix}"
            )

    def invoke_process(self, process: str, pool: Any) -> None:
        try:
            entrypoint = self.processes[process]
        except KeyError as exc:
            raise DeviceContractError(
                f"device {self.name!r} has no process {process!r}"
            ) from exc
        if self.state_policy == "reinitialize_each_run":
            if self.initialize_entrypoint is not None:
                self.call(self.initialize_entrypoint, pool)
        elif self.state_policy == "initialize_once":
            if not self._initialized and self.initialize_entrypoint is not None:
                self.call(self.initialize_entrypoint, pool)
                self._initialized = True
        elif self.state_policy != "stateless":
            raise DeviceContractError(
                f"device {self.name!r} has unknown state policy "
                f"{self.state_policy!r}"
            )
        self.call(entrypoint, pool)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "library": str(self.library_path),
            "state_policy": self.state_policy,
            "processes": dict(self.processes),
            "entrypoints": tuple(self.entrypoints),
            "source_hash": self.source_hash,
            "abi_checked": self._abi_checked,
        }


class DeviceRegistry:
    """Discover generated devices and route named processes through them."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.devices: dict[str, FortranDevice] = {}
        self._processes: dict[str, FortranDevice] = {}
        if self.root.is_dir():
            for manifest in sorted(self.root.glob("*/device.json")):
                self.register(FortranDevice(manifest))

    def register(self, device: FortranDevice) -> None:
        if device.name in self.devices:
            raise DeviceContractError(
                f"duplicate device name {device.name!r}"
            )
        duplicates = set(device.processes) & set(self._processes)
        if duplicates:
            raise DeviceContractError(
                f"duplicate device processes: {sorted(duplicates)}"
            )
        self.devices[device.name] = device
        for process in device.processes:
            self._processes[process] = device

    @property
    def process_names(self) -> frozenset[str]:
        return frozenset(self._processes)

    def invoke(self, process: str, pool: Any) -> None:
        try:
            device = self._processes[process]
        except KeyError as exc:
            raise MissingKernelError(
                f"no generated device provides process {process!r}; "
                f"available processes are {sorted(self._processes)}"
            ) from exc
        before = pool.pointer_records()
        device.invoke_process(process, pool)
        pool.assert_pointer_stability(before)

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self.devices[name].describe() for name in sorted(self.devices)
        )
