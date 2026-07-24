"""Manifest-driven zero-copy connection between StatePool and Fortran devices."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .errors import DeviceContractError, MissingKernelError
from .state import NativeObjectHandle


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
        argtypes = []
        for argument in contract["arguments"]:
            if str(argument["dtype"]) == "character":
                argtypes.extend(
                    [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
                )
            else:
                argtypes.append(self._ctypes_argument(argument))
        argtypes.extend(
            [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
        )
        function.argtypes = argtypes
        function.restype = ctypes.c_int
        self._functions[entrypoint] = function
        return function

    @staticmethod
    def _ctypes_argument(argument: Mapping[str, Any]) -> Any:
        if str(argument["dtype"]) == "character":
            return ctypes.POINTER(ctypes.c_char)
        if str(argument["dtype"]) == "opaque":
            return ctypes.c_void_p
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
        if str(argument["dtype"]) == "character":
            if values.dtype.kind != "S":
                raise DeviceContractError(
                    f"{self.name}.{argument['abi_name']} maps to "
                    f"{field_name!r} with dtype {values.dtype}; character "
                    "fields require a fixed-width NumPy bytes dtype"
                )
            expected_dtype = values.dtype
        else:
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

    def _new_opaque_handle(
        self,
        argument: Mapping[str, Any],
        pool: Any,
        shape: tuple[int, ...],
    ) -> NativeObjectHandle:
        try:
            contract = argument["opaque"]
            factory = getattr(self.lib, contract["factory_symbol"])
            destroy = getattr(self.lib, contract["destroy_symbol"])
        except (KeyError, AttributeError) as exc:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} has no usable opaque "
                "object factory"
            ) from exc
        dimension_types = [ctypes.c_int] * len(shape)
        factory.argtypes = dimension_types
        factory.restype = ctypes.c_void_p
        destroy.argtypes = [ctypes.c_void_p, *dimension_types]
        destroy.restype = None
        address = int(factory(*shape) or 0)
        if not address:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} failed to allocate "
                f"{argument['fortran_type']} with shape {shape}"
            )

        def release() -> None:
            destroy(ctypes.c_void_p(address), *shape)

        return NativeObjectHandle(
            address=address,
            fortran_type=str(argument["fortran_type"]).lower(),
            shape=shape,
            owner=self,
            destroy=release,
        )

    def _resolve_opaque(
        self, argument: Mapping[str, Any], pool: Any
    ) -> ctypes.c_void_p:
        binding = argument["binding"]
        if binding["source"] not in {"standard_name", "field"}:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']}: opaque arguments must "
                "bind to persistent process state"
            )
        standard_name = str(binding["name"]).lower()
        shape = self._expected_shape(argument, pool)
        try:
            handle = pool.get_process_state(standard_name)
        except KeyError:
            if argument["intent"] != "out":
                raise DeviceContractError(
                    f"{self.name}.{argument['abi_name']} requires opaque "
                    f"state {standard_name!r} before its {argument['intent']} "
                    "call; run the producing register/initialize/scheme first"
                ) from None
            handle = self._new_opaque_handle(argument, pool, shape)
            pool.set_process_state(standard_name, handle)
        expected_type = str(argument["fortran_type"]).lower()
        if handle.fortran_type != expected_type or handle.shape != shape:
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} expects opaque "
                f"{expected_type}{shape}, got "
                f"{handle.fortran_type}{handle.shape}"
            )
        return ctypes.c_void_p(handle.address)

    def _resolve_argument(
        self, argument: Mapping[str, Any], pool: Any
    ) -> Any:
        binding = argument["binding"]
        source = binding["source"]
        if str(argument["dtype"]) == "opaque":
            return self._resolve_opaque(argument, pool)
        if str(argument["dtype"]) == "character":
            values = self._resolve_field(argument, pool)
            return (
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),
                int(values.dtype.itemsize),
            )
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
        arguments = []
        for argument in contract["arguments"]:
            resolved = self._resolve_argument(argument, pool)
            if str(argument["dtype"]) == "character":
                arguments.extend(resolved)
            else:
                arguments.append(resolved)
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
        for argument in contract["arguments"]:
            if argument["intent"] not in {"out", "inout"}:
                continue
            if argument["dtype"] == "opaque":
                # Opaque derived-type state lives in StatePool's native
                # handle registry, not in its NumPy field schema.
                continue
            binding = argument["binding"]
            if binding["source"] == "field":
                pool.mark_initialized(str(binding["name"]))
            elif binding["source"] == "standard_name":
                pool.mark_initialized(
                    pool.ccpp_field_name(str(binding["name"]))
                )

    def invoke_process(self, process: str, pool: Any) -> None:
        try:
            entrypoint = self.processes[process]
        except KeyError as exc:
            raise DeviceContractError(
                f"device {self.name!r} has no process {process!r}"
            ) from exc
        if self.state_policy == "reinitialize_each_run":
            # Generated descriptors also expose explicit lifecycle processes.
            # Reinitialize only before the numerical run entrypoint; applying
            # this policy to ``scheme:initialize`` would call initialize twice.
            if (
                entrypoint == "run"
                and self.initialize_entrypoint is not None
            ):
                self.call(self.initialize_entrypoint, pool)
        elif self.state_policy == "initialize_once":
            if (
                entrypoint != self.initialize_entrypoint
                and not self._initialized
                and self.initialize_entrypoint is not None
            ):
                self.call(self.initialize_entrypoint, pool)
                self._initialized = True
        elif self.state_policy != "stateless":
            raise DeviceContractError(
                f"device {self.name!r} has unknown state policy "
                f"{self.state_policy!r}"
            )
        self.call(entrypoint, pool)
        if (
            self.state_policy == "initialize_once"
            and entrypoint == self.initialize_entrypoint
        ):
            self._initialized = True

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

    def __init__(self, root: str | Path | Iterable[str | Path]):
        roots = (
            (root,)
            if isinstance(root, (str, Path))
            else tuple(root)
        )
        self.roots = tuple(Path(item).resolve() for item in roots)
        self.root = self.roots[0] if self.roots else Path(".").resolve()
        self.devices: dict[str, FortranDevice] = {}
        self._processes: dict[str, FortranDevice] = {}
        self._retired_devices: list[FortranDevice] = []
        for device_root in self.roots:
            if not device_root.is_dir():
                continue
            for manifest in sorted(device_root.glob("*/device.json")):
                device = FortranDevice(manifest)
                # Earlier build roots have priority when a focused core build
                # and a full-catalog build contain the same generated device.
                if device.name in self.devices:
                    continue
                self.register(device)

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

    def unregister(self, name: str) -> FortranDevice:
        """Stop routing a device while keeping its shared library mapped."""

        try:
            device = self.devices.pop(name)
        except KeyError as exc:
            raise DeviceContractError(
                f"unknown device name {name!r}"
            ) from exc
        self._processes = {
            process: owner
            for process, owner in self._processes.items()
            if owner is not device
        }
        # ctypes has no portable, safe hot-unload contract for Fortran module
        # code. Keep the object alive until the registry itself is finalized.
        self._retired_devices.append(device)
        return device

    @property
    def process_names(self) -> frozenset[str]:
        return frozenset(self._processes)

    def has_process(self, process: str) -> bool:
        return process in self._processes

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

    @staticmethod
    def release_pool(pool: Any) -> None:
        pool.release_process_state()
