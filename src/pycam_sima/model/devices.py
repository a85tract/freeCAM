"""Manifest-driven zero-copy connection between StatePool and Fortran devices."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import shutil
import subprocess
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
    "rad": "radian",
}
_UNIT_FACTORS = {
    ("pa", "hpa"): np.float64(1.0e-2),
    ("hpa", "pa"): np.float64(1.0e2),
    ("percent", "fraction"): np.float64(1.0e-2),
    ("fraction", "percent"): np.float64(1.0e2),
    ("radian", "degrees"): np.float64(180.0) / np.float64(np.pi),
    ("degrees", "radian"): np.float64(np.pi) / np.float64(180.0),
}
_CONSTITUENT_STANDARD_NAMES = {
    "cl": "Cl",
    "cl2": "Cl2",
    "o2": "O2",
    "o3": "O3",
    "cloud_ice": (
        "cloud_ice_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
    "cloud_liquid_water": (
        "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
    "rain_water": (
        "rain_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
    "water_vapor": (
        "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
}
_WATER_CONSTITUENTS = frozenset(_CONSTITUENT_STANDARD_NAMES)


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
        self.host_entrypoints = dict(payload.get("host_entrypoints", {}))
        self.dimension_bindings = dict(payload["dimension_bindings"])
        library = self.manifest_path.parent / str(payload["library"])
        if not library.is_file():
            raise MissingKernelError(
                f"device {self.name!r} library does not exist: {library}"
            )
        self.library_path = library
        self._lib: Any | None = None
        self.host_services = tuple(payload.get("host_services", ()))
        self._netcdf_service = None
        self._functions: dict[str, Any] = {}
        self._external_runtime_handles: list[Any] = []
        self._abi_checked = False
        self._initialized = False

    def _preload_external_runtime_libraries(self) -> None:
        """Resolve declared optional runtimes without embedding an RPATH.

        Device builds intentionally reject RPATH/RUNPATH entries.  On HPC
        systems a compiler can still link against a module-provided library
        whose directory is not exported in ``LD_LIBRARY_PATH``.  Resolve only
        libraries explicitly declared by the manifest, including NetCDF
        Fortran directories reported by ``nf-config``.
        """

        directories = [
            Path(value)
            for variable in ("LD_LIBRARY_PATH", "LIBRARY_PATH")
            for value in os.environ.get(variable, "").split(os.pathsep)
            if value
        ]
        nf_config = shutil.which("nf-config")
        if nf_config is not None:
            try:
                flags = subprocess.run(
                    (nf_config, "--flibs"),
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.split()
            except (OSError, subprocess.CalledProcessError):
                flags = ()
            directories.extend(
                Path(flag[2:]) for flag in flags if flag.startswith("-L")
            )

        # Descriptors list direct libraries before their transitive
        # dependencies (for example MUSICA before NetCDF).  Preload in reverse
        # so an absolute dependency without RPATH is already resident when
        # its consumer is opened.
        declared_libraries = tuple(
            self.manifest.get("external", {}).get("libraries", ())
        )
        for declared in reversed(declared_libraries):
            value = str(declared)
            if value.endswith(".a"):
                continue
            if "/" in value:
                path = Path(value)
                names = [str(path)] if path.is_file() else []
            else:
                candidates: list[Path] = []
                for directory in dict.fromkeys(directories):
                    candidates.extend(
                        sorted(directory.glob(f"lib{value}.so*"))
                    )
                system_name = ctypes.util.find_library(value)
                names = [str(path) for path in candidates]
                if system_name:
                    names.append(system_name)
            for name in names:
                try:
                    handle = ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
                self._external_runtime_handles.append(handle)
                break

    @property
    def lib(self) -> Any:
        """Load the numerical device only when this model first uses it.

        A catalog may contain optional external stacks such as MUSICA even
        when the active suite is Kessler.  Eagerly dlopen'ing every catalog
        library would make an unrelated model depend on every optional
        runtime library.
        """

        if self._lib is None:
            try:
                library = ctypes.CDLL(
                    str(self.library_path), mode=ctypes.RTLD_LOCAL
                )
            except OSError:
                self._preload_external_runtime_libraries()
                try:
                    library = ctypes.CDLL(
                        str(self.library_path), mode=ctypes.RTLD_LOCAL
                    )
                except OSError as exc:
                    raise MissingKernelError(
                        f"cannot load device {self.name!r} from "
                        f"{self.library_path}: {exc}"
                    ) from exc
            self._lib = library
            if "netcdf_reader" in self.host_services:
                from .netcdf_service import (
                    register_netcdf_reader_callbacks,
                )

                self._netcdf_service = register_netcdf_reader_callbacks(
                    library
                )
        return self._lib

    @staticmethod
    def _pool_string(pool: Any, standard_name: str) -> bytes:
        field_name = pool.ccpp_field_name(standard_name)
        if not pool.is_initialized(field_name):
            raise DeviceContractError(
                f"host entrypoint reads uninitialized StatePool field "
                f"{field_name!r}"
            )
        value = np.asarray(pool.get(field_name)).reshape(-1)
        if value.size != 1:
            raise DeviceContractError(
                f"host configuration {standard_name!r} must be scalar"
            )
        item = value[0]
        return (
            bytes(item).rstrip(b"\0 ")
            if isinstance(item, (bytes, np.bytes_))
            else str(item).encode("utf-8")
        )

    def invoke_host_entrypoint(
        self, entrypoint: str, pool: Any
    ) -> None:
        """Run a declared Python-orchestrated native host boundary."""

        try:
            contract = self.host_entrypoints[entrypoint]
        except KeyError as exc:
            raise DeviceContractError(
                f"device {self.name!r} has no host entrypoint {entrypoint!r}"
            ) from exc
        if contract.get("kind") != "musica_constituent_registry":
            raise DeviceContractError(
                f"device {self.name!r} has unsupported host entrypoint kind "
                f"{contract.get('kind')!r}"
            )
        standard_name = str(contract["standard_name"]).lower()
        if standard_name in pool.process_state_names:
            return

        values = [
            self._pool_string(pool, str(name))
            for name in contract["configuration_standard_names"]
        ]
        configuration_names = tuple(
            str(name) for name in contract["configuration_standard_names"]
        )
        for name, value in zip(configuration_names[1:], values[1:]):
            path = Path(value.decode("utf-8", errors="strict"))
            if path.name.lower() == "none":
                continue
            if not path.is_file():
                raise DeviceContractError(
                    f"device {self.name!r} host configuration {name!r} "
                    f"is not a readable file: {path}"
                )
        configure = getattr(self.lib, str(contract["configure_symbol"]))
        configure.argtypes = [
            item
            for _ in values
            for item in (ctypes.POINTER(ctypes.c_char), ctypes.c_int)
        ]
        configure.restype = ctypes.c_int
        configure_args: list[Any] = []
        buffers = [ctypes.create_string_buffer(value) for value in values]
        for buffer, value in zip(buffers, values):
            configure_args.extend(
                (
                    ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)),
                    len(value),
                )
            )
        status = int(configure(*configure_args))
        if status:
            raise DeviceContractError(
                f"device {self.name!r} host configuration failed with "
                f"code {status}"
            )

        register = getattr(self.lib, str(contract["register_symbol"]))
        register.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
        ]
        register.restype = ctypes.c_int
        address = ctypes.c_void_p()
        count = ctypes.c_int()
        message = ctypes.create_string_buffer(2048)
        status = int(
            register(
                ctypes.byref(address),
                ctypes.byref(count),
                message,
                len(message),
            )
        )
        if status:
            detail = message.value.decode("utf-8", errors="replace").strip()
            raise DeviceContractError(
                f"device {self.name!r} constituent registration failed with "
                f"code {status}: {detail}"
            )

        release = getattr(self.lib, str(contract["release_symbol"]))
        release.argtypes = [ctypes.c_void_p]
        release.restype = None
        registered_address = int(address.value or 0)
        registered_count = int(count.value)
        try:
            if registered_count > int(pool.dimensions["nconst"]):
                raise DeviceContractError(
                    f"MUSICA registered {registered_count} constituents, but "
                    f"ModelConfig allocated only {pool.dimensions['nconst']}"
                )
            metadata = getattr(self.lib, str(contract["metadata_symbol"]))
            metadata.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_int,
            ]
            metadata.restype = ctypes.c_int
            registered_names: list[str] = []
            registered_minima: list[float] = []
            registered_molar_masses: list[float] = []
            for index in range(1, registered_count + 1):
                name = ctypes.create_string_buffer(512)
                error = ctypes.create_string_buffer(2048)
                minimum = ctypes.c_double()
                molar_mass = ctypes.c_double()
                status = int(
                    metadata(
                        index,
                        name,
                        len(name),
                        ctypes.byref(minimum),
                        ctypes.byref(molar_mass),
                        error,
                        len(error),
                    )
                )
                if status:
                    detail = error.value.decode(
                        "utf-8", errors="replace"
                    ).strip()
                    raise DeviceContractError(
                        f"MUSICA constituent metadata {index} failed: {detail}"
                    )
                registered_names.append(
                    name.value.decode("utf-8", errors="strict").lower()
                )
                registered_minima.append(float(minimum.value))
                registered_molar_masses.append(float(molar_mass.value))
            pool_standard_names = tuple(
                _CONSTITUENT_STANDARD_NAMES.get(name, name).lower()
                for name in pool.constituent_names
            )
            try:
                registered_positions = tuple(
                    pool_standard_names.index(name)
                    for name in registered_names
                )
            except ValueError as exc:
                raise DeviceContractError(
                    "MUSICA registered a constituent absent from ModelConfig: "
                    f"registered={registered_names}, "
                    f"configured={list(pool_standard_names)}"
                ) from exc
            if len(set(registered_positions)) != registered_count:
                raise DeviceContractError(
                    "MUSICA registered duplicate constituent names"
                )
            configured_minima = np.asarray(pool.get("constituent_minimum"))
            configured_molar = np.asarray(
                pool.get("constituent_molecular_weight")
            ) * np.float64(1.0e-3)
            if not np.allclose(
                registered_minima,
                configured_minima[list(registered_positions)],
                rtol=0.0,
                atol=np.finfo(np.float64).eps,
            ):
                raise DeviceContractError(
                    "MUSICA constituent minima differ from ModelConfig"
                )
            if not np.allclose(
                registered_molar_masses,
                configured_molar[list(registered_positions)],
                rtol=0.0,
                atol=np.finfo(np.float64).eps,
            ):
                raise DeviceContractError(
                    "MUSICA constituent molar masses differ from ModelConfig"
                )
        except Exception:
            release(address)
            raise
        # ``musica_ccpp_register`` must run because it also initializes
        # MUSICA's internal species inventory.  Its returned Fortran pointer
        # wrappers are not C-interoperable objects, however.  Use them only
        # through the metadata accessors above, then create the ABI-facing
        # registry with the generated constructor/configure functions.
        release(ctypes.c_void_p(registered_address))
        opaque_argument = next(
            (
                argument
                for entrypoint in self.entrypoints.values()
                for argument in entrypoint["arguments"]
                if argument["dtype"] == "opaque"
                and str(argument["standard_name"]).lower()
                == standard_name
                and "configure_symbol" in argument["opaque"]
            ),
            None,
        )
        if opaque_argument is None:
            raise DeviceContractError(
                f"device {self.name!r} has no configurable ABI registry for "
                f"{standard_name!r}"
            )
        handle = self._new_opaque_handle(
            opaque_argument,
            pool,
            (int(pool.dimensions["nconst"]),),
        )
        pool.set_process_state(standard_name, handle)

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
                extent = int(pool.dimensions[pool_dimension])
            except KeyError as exc:
                raise DeviceContractError(
                    f"device {self.name!r} cannot resolve dimension "
                    f"{standard_name!r}"
                ) from exc
            try:
                runtime_field = pool.ccpp_field_name(standard_name)
            except KeyError:
                pass
            else:
                if pool.is_initialized(runtime_field):
                    runtime_extent = int(pool.get(runtime_field).item())
                    if not 0 <= runtime_extent <= extent:
                        raise DeviceContractError(
                            f"device {self.name!r} runtime dimension "
                            f"{standard_name!r}={runtime_extent} is outside "
                            f"its allocated capacity {extent}"
                        )
                    extent = runtime_extent
            shape.append(extent)
        return tuple(shape)

    def _resolve_field(
        self,
        argument: Mapping[str, Any],
        pool: Any,
        copybacks: list[tuple[np.ndarray, np.ndarray, np.float64]]
        | None = None,
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
        if (
            argument["intent"] in {"in", "inout"}
            and not pool.is_initialized(field_name)
        ):
            raise DeviceContractError(
                f"{self.name}.{argument['abi_name']} reads uninitialized "
                f"StatePool field {field_name!r}; initialize CCPP standard "
                f"name {argument['standard_name']!r} first"
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
        storage_values = values
        dynamic_copyback: tuple[np.ndarray, np.ndarray, np.float64] | None = None
        if values.shape != expected_shape:
            runtime_axes = []
            for axis, (stored, expected, dimension) in enumerate(
                zip(values.shape, expected_shape, argument["dimensions"])
            ):
                if stored == expected:
                    continue
                try:
                    runtime_field = pool.ccpp_field_name(str(dimension))
                except KeyError:
                    runtime_field = None
                if (
                    runtime_field is None
                    or not pool.is_initialized(runtime_field)
                    or expected < 0
                    or expected > stored
                ):
                    raise DeviceContractError(
                        f"{self.name}.{argument['abi_name']} maps to "
                        f"{field_name!r} with shape {values.shape}, "
                        f"expected {expected_shape}"
                    )
                runtime_axes.append(axis)
            if len(values.shape) != len(expected_shape) or not runtime_axes:
                raise DeviceContractError(
                    f"{self.name}.{argument['abi_name']} maps to "
                    f"{field_name!r} with shape {values.shape}, "
                    f"expected {expected_shape}"
                )
            destination = storage_values[
                tuple(slice(0, extent) for extent in expected_shape)
            ]
            if argument["intent"] == "out":
                values = np.empty(expected_shape, dtype=values.dtype, order="F")
            else:
                values = np.asfortranarray(destination)
            if argument["intent"] in {"out", "inout"}:
                dynamic_copyback = (
                    values,
                    destination,
                    np.float64(1.0),
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
            input_factor = _UNIT_FACTORS.get((actual_units, expected_units))
            output_factor = _UNIT_FACTORS.get((expected_units, actual_units))
            intent = str(argument["intent"])
            if (
                intent in {"in", "inout"} and input_factor is None
            ) or (
                intent in {"out", "inout"} and output_factor is None
            ):
                raise DeviceContractError(
                    f"{self.name}.{argument['abi_name']} maps to "
                    f"{field_name!r} with units {contract.units!r}, "
                    f"expected {argument['units']!r}; no safe "
                    "conversion is registered"
                )
            if intent == "out":
                converted = np.empty_like(values, order="F")
            else:
                converted = np.asfortranarray(values * input_factor)
            if intent in {"out", "inout"}:
                if copybacks is None:
                    raise DeviceContractError(
                        f"{self.name}.{argument['abi_name']} requires an "
                        "output-unit conversion without a call context"
                    )
                copybacks.append((converted, values, output_factor))
            values = converted
        if dynamic_copyback is not None:
            if copybacks is None:
                raise DeviceContractError(
                    f"{self.name}.{argument['abi_name']} requires a dynamic-"
                    "extent output copy without a call context"
                )
            copybacks.append(dynamic_copyback)
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
        configure_symbol = argument["opaque"].get("configure_symbol")
        if configure_symbol is not None:
            if (
                str(argument["fortran_type"]).lower()
                != "ccpp_constituent_prop_ptr_t"
                or shape != (int(pool.dimensions["nconst"]),)
            ):
                destroy(ctypes.c_void_p(address), *shape)
                raise DeviceContractError(
                    f"{self.name}.{argument['abi_name']} has an invalid "
                    "CCPP constituent registry shape"
                )
            configure = getattr(self.lib, str(configure_symbol))
            configure.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_int,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.c_bool,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_int,
            ]
            configure.restype = ctypes.c_int
            minima = np.asarray(pool.get("constituent_minimum"))
            molar_masses = np.asarray(
                pool.get("constituent_molecular_weight")
            ) * np.float64(1.0e-3)
            try:
                for index, short_name in enumerate(
                    pool.constituent_names,
                    start=1,
                ):
                    standard_name = _CONSTITUENT_STANDARD_NAMES.get(
                        short_name,
                        short_name,
                    )
                    encoded = standard_name.encode("utf-8")
                    name = ctypes.create_string_buffer(encoded)
                    error = ctypes.create_string_buffer(2048)
                    is_water = short_name in _WATER_CONSTITUENTS
                    status = int(
                        configure(
                            ctypes.c_void_p(address),
                            shape[0],
                            index,
                            name,
                            len(encoded),
                            float(minima[index - 1]),
                            float(molar_masses[index - 1]),
                            is_water,
                            True,
                            is_water,
                            error,
                            len(error),
                        )
                    )
                    if status:
                        detail = error.value.decode(
                            "utf-8",
                            errors="replace",
                        ).strip()
                        raise DeviceContractError(
                            f"{self.name}: cannot configure constituent "
                            f"{index} {standard_name!r}: {detail}"
                        )
            except Exception:
                destroy(ctypes.c_void_p(address), *shape)
                raise

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
            if (
                argument["intent"] != "out"
                and "configure_symbol" not in argument["opaque"]
            ):
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
        self,
        argument: Mapping[str, Any],
        pool: Any,
        copybacks: list[tuple[np.ndarray, np.ndarray, np.float64]]
        | None = None,
    ) -> Any:
        binding = argument["binding"]
        source = binding["source"]
        if str(argument["dtype"]) == "opaque":
            return self._resolve_opaque(argument, pool)
        if str(argument["dtype"]) == "character":
            values = self._resolve_field(argument, pool, copybacks)
            return (
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_char)),
                int(values.dtype.itemsize),
            )
        dtype, ctype = _DTYPES[str(argument["dtype"])]
        if source == "dimension":
            # Public CCPP scalar dimensions can be runtime values with a fixed
            # allocation capacity.  Prefer that explicit field when present;
            # injected ABI dimensions remain fixed StatePool dimensions.
            runtime_field = None
            if not bool(argument.get("injected", False)):
                try:
                    runtime_field = pool.ccpp_field_name(
                        str(argument["standard_name"])
                    )
                except KeyError:
                    pass
            if runtime_field is not None:
                values = self._resolve_field(
                    {
                        **argument,
                        "binding": {
                            "source": "field",
                            "name": runtime_field,
                        },
                    },
                    pool,
                    copybacks,
                )
                if argument["passing"] == "reference":
                    return values.ctypes.data_as(ctypes.POINTER(ctype))
                value = values.item()
            else:
                value = int(pool.dimensions[str(binding["name"])])
                if argument["passing"] == "reference":
                    # Fixed register/init dimensions are writable ABI
                    # temporaries; their capacity was inferred before pool
                    # allocation and cannot be changed by a scheme.
                    return ctypes.pointer(ctype(value))
        elif source == "literal":
            value = binding["value"]
            if argument["passing"] == "reference":
                return ctypes.pointer(ctype(value))
        else:
            # CCPP metadata is not consistent about tagging scalar dimension
            # arguments.  A producer may declare ``nbndlw`` as an ordinary
            # standard-name scalar while consumers use the same standard
            # name as an array dimension.  StatePool deliberately stores
            # dimensions once in ``pool.dimensions`` rather than allocating
            # duplicate scalar arrays.  Resolve that exact case here.
            dimension_name = str(binding["name"])
            if (
                source == "standard_name"
                and int(argument["rank"]) == 0
                and dimension_name in pool.dimensions
            ):
                try:
                    pool.ccpp_field_name(dimension_name)
                except KeyError:
                    value = int(pool.dimensions[dimension_name])
                    if argument["passing"] == "reference":
                        return ctypes.pointer(ctype(value))
                    if dtype.kind in {"i", "u"}:
                        return int(value)
                    return float(value)
            values = self._resolve_field(argument, pool, copybacks)
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
        copybacks: list[
            tuple[np.ndarray, np.ndarray, np.float64]
        ] = []
        for argument in contract["arguments"]:
            resolved = self._resolve_argument(argument, pool, copybacks)
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
        for converted, destination, factor in copybacks:
            np.multiply(converted, factor, out=destination)
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
                standard_name = str(binding["name"])
                try:
                    field_name = pool.ccpp_field_name(standard_name)
                except KeyError:
                    if standard_name in pool.dimensions:
                        # File-sized output dimensions were inferred before
                        # StatePool allocation and are not duplicate fields.
                        continue
                    raise
                pool.mark_initialized(field_name)
            elif binding["source"] == "dimension" and not bool(
                argument.get("injected", False)
            ):
                try:
                    field_name = pool.ccpp_field_name(
                        str(argument["standard_name"])
                    )
                except KeyError:
                    continue
                pool.mark_initialized(field_name)

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
            "host_services": self.host_services,
            "host_entrypoints": tuple(self.host_entrypoints),
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

    def invoke_host_entrypoint(
        self, device_name: str, entrypoint: str, pool: Any
    ) -> None:
        try:
            device = self.devices[device_name]
        except KeyError as exc:
            raise MissingKernelError(
                f"no generated device named {device_name!r}"
            ) from exc
        before = pool.pointer_records()
        device.invoke_host_entrypoint(entrypoint, pool)
        pool.assert_pointer_stability(before)

    def initialize_constituent_registry(
        self,
        pool: Any,
        *,
        device_names: Iterable[str],
        constituent_standard_names: Iterable[str] | None = None,
    ) -> tuple[str, ...]:
        """Give active original schemes the Python-owned constituent order.

        The generated CAM cap normally creates this module-global lookup
        table.  PyCAM-SIMA replaces only that host-owned registry boundary;
        the numerical schemes and their calls to ``ccpp_constituent_index``
        remain unchanged.
        """

        active_libraries: dict[Path, FortranDevice] = {}
        for name in device_names:
            device = self.devices.get(name)
            if device is not None:
                active_libraries.setdefault(device.library_path, device)
        if not active_libraries:
            return ()

        standard_names = tuple(
            str(name)
            for name in (
                (
                    _CONSTITUENT_STANDARD_NAMES.get(name, name)
                    for name in pool.constituent_names
                )
                if constituent_standard_names is None
                else constituent_standard_names
            )
        )
        if not standard_names:
            return ()
        encoded = tuple(name.encode("utf-8") for name in standard_names)
        width = max(len(name) for name in encoded) + 1
        packed = bytearray(width * len(encoded))
        for index, name in enumerate(encoded):
            begin = index * width
            packed[begin : begin + len(name)] = name
        names_buffer = (ctypes.c_char * len(packed)).from_buffer(packed)

        initialized: list[str] = []
        for library_path, device in active_libraries.items():
            try:
                configure = getattr(
                    device.lib,
                    "pycam_ccpp_scheme_registry_initialize_v1",
                )
            except AttributeError:
                # Older/plugin libraries may not use ccpp_scheme_utils.
                continue
            configure.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_char),
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_int,
            ]
            configure.restype = ctypes.c_int
            error = ctypes.create_string_buffer(2048)
            status = int(
                configure(
                    len(encoded),
                    width,
                    names_buffer,
                    error,
                    len(error),
                )
            )
            if status:
                detail = error.value.decode(
                    "utf-8", errors="replace"
                ).strip()
                raise DeviceContractError(
                    "cannot initialize the Python-owned CCPP constituent "
                    f"registry in {library_path}: {detail or status}"
                )
            initialized.append(str(library_path))
        return tuple(initialized)

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self.devices[name].describe() for name in sorted(self.devices)
        )

    @staticmethod
    def release_pool(pool: Any) -> None:
        pool.release_process_state()
