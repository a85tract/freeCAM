"""Thin ctypes ABI for CAM kernels and Python-gated primitive services."""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Iterator, Mapping, Protocol

import numpy as np

from freecam.core.fortran_adapter import (
    FortranAdapterError,
    PointerTableAdapter,
)
from freecam.core.fortran_runtime import prepare_fortran_runtime

from .errors import NativeCAMError
from .plan import PICAMAction
from .process_devices import (
    PICAMProcessDeviceCatalog,
    PICAMProcessDeviceRuntime,
)
from .state import PICAMFieldContract, PICAMStatePool


def _prepare_fortran_runtime() -> None:
    """Initialize Intel Fortran when Python, rather than Fortran, is main."""

    prepare_fortran_runtime()


class CAMNumericalBackend(Protocol):
    def configure_shared_coupler(self, enabled: bool) -> None: ...
    def prepare_initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None: ...
    def prepare_state(
        self, pool: PICAMStatePool, config: object, *, rank: int, size: int
    ) -> None: ...
    def initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None: ...
    def execute(self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int) -> None: ...
    def execute_boundary_kernel(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None: ...
    def execute_io_service(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None: ...
    def execute_state_service(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None: ...
    def synchronize_clock(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None: ...
    def finalize(self, pool: PICAMStatePool, *, fcomm: int) -> None: ...


class RecordingCAMBackend:
    """Deterministic no-numerics backend used for plan and boundary tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.dispatches: list[tuple[str, str]] = []

    def configure_shared_coupler(self, enabled: bool) -> None:
        del enabled

    def prepare_initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        del pool, fcomm

    def prepare_state(
        self, pool: PICAMStatePool, config: object, *, rank: int, size: int
    ) -> None:
        del pool, config, rank, size

    def initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        del pool, fcomm
        self.calls.append("initialize")

    def execute(self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int) -> None:
        del pool, fcomm
        if action.kind not in {
            "scheme",
            "coupling",
            "dynamics",
            "kernel",
            "runtime_fortran_process",
        }:
            raise NativeCAMError(
                f"generic numerical execution cannot run {action.kind!r} "
                f"action {action.operation!r}"
            )
        self.calls.append(action.operation)
        self.dispatches.append(("numerical-kernel", action.operation))

    def _service(self, channel: str, action: PICAMAction) -> None:
        self.calls.append(action.operation)
        self.dispatches.append((channel, action.operation))

    def execute_boundary_kernel(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        del pool, fcomm
        self._service("boundary-kernel", action)

    def execute_io_service(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        del pool, fcomm
        self._service("io-service", action)

    def execute_state_service(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        del pool, fcomm
        self._service("state-service", action)

    def synchronize_clock(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        del pool, fcomm
        self._service("clock-mirror", action)

    def finalize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        del pool, fcomm
        self.calls.append("finalize")


class _BoundStatePool(Mapping[str, np.ndarray]):
    """Translate generated logical argument names to real StatePool fields."""

    def __init__(
        self,
        pool: Mapping[str, np.ndarray],
        bindings: Mapping[str, str],
    ) -> None:
        self.pool = pool
        self.bindings = dict(bindings)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.pool[self.bindings.get(name, name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self.bindings)

    def __len__(self) -> int:
        return len(self.bindings)


class _ChunkProcessStatePool(Mapping[str, np.ndarray]):
    """Expose one CAM chunk from aggregate rank-local StatePool storage."""

    def __init__(
        self,
        pool: PICAMStatePool,
        bindings: Mapping[str, str],
        arguments: Mapping[str, Mapping[str, object]],
        chunk: int,
    ) -> None:
        self.pool = pool
        self.bindings = dict(bindings)
        self.arguments = dict(arguments)
        self.chunk = int(chunk)

    def __getitem__(self, logical_name: str) -> np.ndarray:
        field = self.bindings[logical_name]
        values = np.asarray(self.pool[field])
        argument = self.arguments[logical_name]
        rank = int(argument.get("rank", 0))
        fortran_type = str(argument.get("fortran_type", ""))
        chunks = int(self.pool.dimensions["chunks"])
        if fortran_type.startswith("type:"):
            if values.dtype.kind == "V":
                if values.ndim != rank + 1 or values.shape[-1] != chunks:
                    raise NativeCAMError(
                        f"opaque record binding {field!r} must have rank "
                        f"{rank + 1} with chunks last"
                    )
                return values[(..., self.chunk)]
            if values.dtype != np.dtype("uint8"):
                raise NativeCAMError(
                    f"opaque binding {field!r} must be uint8 owner storage "
                    "or a void-record array"
                )
            if rank != 0:
                raise NativeCAMError(
                    f"opaque array {field!r} needs an explicit void-record view"
                )
            if values.ndim == 1:
                if values.size % chunks:
                    raise NativeCAMError(
                        f"opaque owner {field!r} cannot be divided into {chunks} chunks"
                    )
                record_bytes = values.size // chunks
                return np.ndarray(
                    shape=(),
                    dtype=np.dtype(f"V{record_bytes}"),
                    buffer=values,
                    offset=self.chunk * record_bytes,
                )
            if values.ndim == 2 and values.shape[-1] == chunks:
                record = values[:, self.chunk]
                return np.ndarray(
                    shape=(), dtype=np.dtype(f"V{record.nbytes}"), buffer=record
                )
            raise NativeCAMError(
                f"opaque binding {field!r} is not rank-local owner storage"
            )
        if values.ndim == rank + 1 and values.shape[-1] == chunks:
            if rank == 0:
                return values[self.chunk : self.chunk + 1].reshape(())
            return values[(..., self.chunk)]
        if values.ndim == rank:
            return values
        raise NativeCAMError(
            f"binding {field!r} has aggregate rank {values.ndim}; expected "
            f"{rank + 1} with chunks last"
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.bindings)

    def __len__(self) -> int:
        return len(self.bindings)


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
        leaf_device = payload.get("leaf_device")
        self._leaf_library_path: Path | None = None
        self._leaf_operation_names: frozenset[str] = frozenset()
        if isinstance(leaf_device, Mapping):
            raw_leaf_library = Path(str(leaf_device["library"]))
            if not raw_leaf_library.is_absolute():
                raw_leaf_library = self.manifest_path.parent / raw_leaf_library
            self._leaf_library_path = raw_leaf_library
            self._leaf_operation_names = frozenset(
                str(name) for name in leaf_device.get("operations", ())
            )
        promoted_kernel_device = payload.get("promoted_kernel_device")
        self._promoted_kernel_library_path: Path | None = None
        self._promoted_kernel_operation_names: frozenset[str] = frozenset()
        if isinstance(promoted_kernel_device, Mapping):
            raw_promoted_library = Path(str(promoted_kernel_device["library"]))
            if not raw_promoted_library.is_absolute():
                raw_promoted_library = self.manifest_path.parent / raw_promoted_library
            self._promoted_kernel_library_path = raw_promoted_library
            self._promoted_kernel_operation_names = frozenset(
                str(name)
                for name in promoted_kernel_device.get("operations", ())
            )
        main_operations = {
            name: record
            for name, record in self._operations.items()
            if name not in self._leaf_operation_names
            and name not in self._promoted_kernel_operation_names
        }
        direct_kernels = payload.get("direct_kernels", {})
        kernel_records = (
            direct_kernels.get("kernels", ())
            if isinstance(direct_kernels, Mapping)
            else ()
        )
        self.direct_kernels = tuple(
            str(record["name"])
            for record in kernel_records
            if isinstance(record, Mapping) and "name" in record
        )
        self._abi = PointerTableAdapter(
            self._library,
            main_operations,
            library_name=str(self.library_path),
        )
        self._leaf_library: ctypes.CDLL | None = None
        self._global_library: ctypes.CDLL | None = None
        self._leaf_abi: PointerTableAdapter | None = None
        self._promoted_kernel_library: ctypes.CDLL | None = None
        self._promoted_kernel_abi: PointerTableAdapter | None = None
        self._process_device_runtime: PICAMProcessDeviceRuntime | None = None
        generation_path = payload.get("process_device_generation")
        validation_path = payload.get("process_device_validation")
        loading_path = payload.get("process_device_loading")
        if generation_path is None or validation_path is None:
            for parent in self.manifest_path.resolve().parents:
                generation = parent / "validation/pi_cam_in_module_adapter_generation.json"
                validation = parent / "validation/pi_cam_in_module_adapter_validation.json"
                if generation.is_file() and validation.is_file():
                    generation_path, validation_path = generation, validation
                    break
        if generation_path is not None and validation_path is not None:
            generation = Path(str(generation_path))
            validation = Path(str(validation_path))
            if not generation.is_absolute():
                generation = self.manifest_path.parent / generation
            if not validation.is_absolute():
                validation = self.manifest_path.parent / validation
            loading = None if loading_path is None else Path(str(loading_path))
            if loading is not None and not loading.is_absolute():
                loading = self.manifest_path.parent / loading
            if loading is None:
                for parent in self.manifest_path.resolve().parents:
                    candidate = parent / "validation/pi_cam_process_device_loading.json"
                    if candidate.is_file():
                        loading = candidate
                        break
            if generation.is_file() and validation.is_file():
                catalog = PICAMProcessDeviceCatalog.from_reports(
                    generation, validation, loading
                )
                self._process_device_runtime = PICAMProcessDeviceRuntime(
                    catalog, main_library=self.library_path
                )
        self._native_initialized = False
        state_bridge = payload.get("state_bridge")
        self._state_bridge = (
            None
            if not isinstance(state_bridge, Mapping)
            else _NativeStateBridge(self._library, state_bridge, str(self.library_path))
        )
        self.python_initialized_addresses: dict[str, int] = {}

    @property
    def leaf_loaded(self) -> bool:
        """Whether this rank has explicitly loaded the optional leaf device."""

        return self._leaf_abi is not None

    def bind_kernel(self, name: str, pool: Mapping[str, np.ndarray], *, fcomm: int):
        """Prepare direct kernel ``name`` on ``pool`` for repeated calls.

        The same entry :meth:`execute_kernel` reaches, with the argument
        tables built once (see ``BoundCall``); the returned callable takes
        no arguments and raises the same error the one-shot path would.
        """

        if name not in self.direct_kernels:
            raise NativeCAMError(f"unknown direct CAM kernel {name!r}")
        operation = f"direct_kernel.{name}"
        try:
            bound = self._adapter_for(operation).bind(operation, pool, fcomm=fcomm)
        except FortranAdapterError as exc:
            raise NativeCAMError(str(exc)) from exc

        def run() -> None:
            try:
                bound()
            except FortranAdapterError as exc:
                raise NativeCAMError(str(exc)) from exc

        def retarget(index: int, array: np.ndarray) -> None:
            try:
                bound.retarget(index, array)
            except FortranAdapterError as exc:
                raise NativeCAMError(str(exc)) from exc

        run.retarget = retarget  # type: ignore[attr-defined]
        return run

    def _call(
        self, operation: str, pool: Mapping[str, np.ndarray], fcomm: int
    ) -> None:
        try:
            self._adapter_for(operation).call(operation, pool, fcomm=fcomm)
        except FortranAdapterError as exc:
            raise NativeCAMError(str(exc)) from exc

    def _adapter_for(self, operation: str) -> PointerTableAdapter:
        """The adapter that serves ``operation``, loading its library on first use."""

        if operation in self._leaf_operation_names:
            if not self._native_initialized:
                raise NativeCAMError(
                    "cam_run1 leaf actions require an initialized CAM model"
                )
            if self._leaf_abi is None:
                if (
                    self._leaf_library_path is None
                    or not self._leaf_library_path.is_file()
                ):
                    raise NativeCAMError(
                        f"native CAM leaf library does not exist: {self._leaf_library_path}"
                    )
                # Promote the already-loaded CAM image only now, after
                # initialization and only for an explicitly requested
                # leaf action.  Default BFB runs retain RTLD_LOCAL for
                # their entire lifecycle.
                self._global_library = ctypes.CDLL(
                    str(self.library_path),
                    mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL,
                )
                self._leaf_library = ctypes.CDLL(
                    str(self._leaf_library_path), mode=ctypes.RTLD_LOCAL
                )
                leaf_operations = {
                    name: self._operations[name]
                    for name in self._leaf_operation_names
                }
                self._leaf_abi = PointerTableAdapter(
                    self._leaf_library,
                    leaf_operations,
                    library_name=str(self._leaf_library_path),
                )
            return self._leaf_abi
        elif operation in self._promoted_kernel_operation_names:
            if not self._native_initialized:
                raise NativeCAMError(
                    "promoted CAM kernels require an initialized CAM model"
                )
            if self._promoted_kernel_abi is None:
                if (
                    self._promoted_kernel_library_path is None
                    or not self._promoted_kernel_library_path.is_file()
                ):
                    raise NativeCAMError(
                        "native CAM promoted-kernel library does not exist: "
                        f"{self._promoted_kernel_library_path}"
                    )
                # Keep the default CAM lifecycle on the exact BFB main
                # image.  Only an explicit promoted-process call exposes
                # that image's original module procedures and loads the
                # generated adapter add-on.
                if self._global_library is None:
                    self._global_library = ctypes.CDLL(
                        str(self.library_path),
                        mode=os.RTLD_NOLOAD | ctypes.RTLD_GLOBAL,
                    )
                self._promoted_kernel_library = ctypes.CDLL(
                    str(self._promoted_kernel_library_path),
                    mode=ctypes.RTLD_LOCAL,
                )
                promoted_operations = {
                    name: self._operations[name]
                    for name in self._promoted_kernel_operation_names
                }
                self._promoted_kernel_abi = PointerTableAdapter(
                    self._promoted_kernel_library,
                    promoted_operations,
                    library_name=str(self._promoted_kernel_library_path),
                )
            return self._promoted_kernel_abi
        else:
            return self._abi

    def prepare_state(
        self,
        pool: PICAMStatePool,
        config: object,
        *,
        rank: int,
        size: int,
    ) -> None:
        """Allocate every admitted native-state array in Python before CAM init."""

        if self._state_bridge is not None:
            self._state_bridge.preallocate(pool, config, rank=rank, size=size)
            self.python_initialized_addresses = {
                name: int(values.ctypes.data) for name, values in pool.items()
            }

    _MODULE_SYMBOL_TYPES = {"float64": ctypes.c_double}

    def module_symbol(self, symbol: str, dtype: str) -> object | None:
        """Resolve one Fortran module data symbol in the loaded image.

        Returns a ctypes object viewing the module variable's storage, or
        ``None`` when the symbol or dtype is not available -- callers fail
        closed on ``None`` rather than guessing an address.
        """

        ctype = self._MODULE_SYMBOL_TYPES.get(str(dtype))
        if ctype is None:
            return None
        try:
            return ctype.in_dll(self._library, str(symbol))
        except ValueError:
            return None

    def configure_shared_coupler(self, enabled: bool) -> None:
        """Select whether CESM already owns the shared MPI/PIO control plane."""

        try:
            configure = self._library.pycam_pi_cam_set_shared_coupler_v1
        except AttributeError as exc:
            if enabled:
                raise NativeCAMError(
                    "native CAM library lacks the single-CAM coupler ABI"
                ) from exc
            return
        configure.argtypes = [ctypes.c_int32]
        configure.restype = ctypes.c_int32
        status = int(configure(ctypes.c_int32(int(bool(enabled)))))
        if status:
            raise NativeCAMError(
                f"native CAM shared-coupler configuration failed ({status})"
            )

    def prepare_initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        """Run only the native grid/chunk setup needed by Python allocation."""

        if "prepare_initialize" in self._operations:
            self._call("prepare_initialize", pool, fcomm)
        if self._state_bridge is not None:
            self._state_bridge.read_initialization_context()

    def initialize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        if "bind_state" in self._operations:
            self._call("bind_state", pool, fcomm)
        if "initialize" in self._operations:
            self._call("initialize", pool, fcomm)
        if self._state_bridge is not None:
            self._state_bridge.attach(pool)
        self._native_initialized = True

    def _execute_operation(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        if self._state_bridge is not None and not self._state_bridge.zero_copy:
            self._state_bridge.copy_to_native(pool)
        self._call(action.operation, pool, fcomm)
        if self._state_bridge is not None and not self._state_bridge.zero_copy:
            self._state_bridge.copy_from_native(pool)

    def execute(self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int) -> None:
        if action.kind not in {
            "scheme",
            "coupling",
            "dynamics",
            "kernel",
            "runtime_fortran_process",
        }:
            raise NativeCAMError(
                f"generic numerical execution cannot run {action.kind!r} "
                f"action {action.operation!r}"
            )
        self._execute_operation(action, pool, fcomm=fcomm)

    def execute_boundary_kernel(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        if action.kind != "boundary":
            raise NativeCAMError("boundary kernel received a non-boundary action")
        self._execute_operation(action, pool, fcomm=fcomm)

    def execute_io_service(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        if action.kind != "io":
            raise NativeCAMError("I/O service received a non-I/O action")
        self._execute_operation(action, pool, fcomm=fcomm)

    def execute_state_service(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        if action.kind != "service":
            raise NativeCAMError("state service received a non-service action")
        self._execute_operation(action, pool, fcomm=fcomm)

    def synchronize_clock(
        self, action: PICAMAction, pool: PICAMStatePool, *, fcomm: int
    ) -> None:
        if action.kind != "clock":
            raise NativeCAMError("clock mirror received a non-clock action")
        self._execute_operation(action, pool, fcomm=fcomm)

    def execute_source_step(
        self, pool: PICAMStatePool, *, fcomm: int, apply_import: bool = True
    ) -> None:
        """Execute the original CAM timestep boundary used by BFB runs."""

        operation = "source_step" if apply_import else "source_step_held_import"
        if self._state_bridge is not None and not self._state_bridge.zero_copy:
            self._state_bridge.copy_to_native(pool)
        self._call(operation, pool, fcomm)
        if self._state_bridge is not None and not self._state_bridge.zero_copy:
            self._state_bridge.copy_from_native(pool)

    def execute_kernel(
        self, name: str, pool: Mapping[str, np.ndarray], *, fcomm: int
    ) -> None:
        """Run one original leaf routine directly on Python StatePool arrays.

        The generated adapter passes the selected NumPy addresses straight to
        the unchanged Fortran routine.  A compatibility build may synchronize
        the result at its next mirrored CAM action; the zero-copy build keeps
        the same Python-owned addresses bound for the entire lifecycle.
        """

        if name not in self.direct_kernels:
            raise NativeCAMError(f"unknown direct CAM kernel {name!r}")
        self._call(f"direct_kernel.{name}", pool, fcomm)

    def execute_promoted_process(
        self,
        name: str,
        pool: Mapping[str, np.ndarray],
        *,
        bindings: Mapping[str, str],
        fcomm: int,
    ) -> None:
        """Run a generated routine against explicit StatePool argument bindings."""

        if name in self.direct_kernels:
            self._call(
                f"direct_kernel.{name}",
                _BoundStatePool(pool, bindings),
                fcomm,
            )
            return
        raise NativeCAMError(f"unknown promoted CAM process {name!r}")

    def has_process_adapter(self, qualified_name: str, source: str | None = None) -> bool:
        runtime = self._process_device_runtime
        if runtime is None:
            return False
        if source is not None:
            return runtime.catalog.has(qualified_name, source)
        return any(
            name.startswith(f"{qualified_name}@")
            for name in runtime.qualified_names
        )

    @property
    def process_adapters(self) -> tuple[str, ...]:
        runtime = self._process_device_runtime
        return () if runtime is None else runtime.catalog.loadable_names

    @property
    def generated_process_adapters(self) -> tuple[str, ...]:
        """All compiled adapters, including entries from inactive configs."""

        runtime = self._process_device_runtime
        return () if runtime is None else runtime.qualified_names

    def execute_catalog_process(
        self,
        name: str,
        qualified_name: str,
        source: str,
        pool: PICAMStatePool,
        *,
        bindings: Mapping[str, str],
        arguments: tuple[Mapping[str, object], ...],
        fcomm: int,
    ) -> None:
        """Run one generated source process once for each local CAM chunk."""

        if not self._native_initialized:
            raise NativeCAMError("catalog processes require an initialized CAM model")
        runtime = self._process_device_runtime
        if runtime is None or not runtime.catalog.has(qualified_name, source):
            raise NativeCAMError(
                f"no compiled process device for {qualified_name!r}"
            )
        by_logical = {
            f"process_context.{name}.{argument['name']}": argument
            for argument in arguments
            if not bool(argument.get("procedure", False))
        }
        for chunk in range(int(pool.dimensions["chunks"])):
            runtime.call(
                qualified_name,
                source,
                name,
                _ChunkProcessStatePool(pool, bindings, by_logical, chunk),
                fcomm=fcomm,
            )

    def kernel_fields(self, name: str) -> tuple[str, ...]:
        """Return the ordered StatePool contract for one direct kernel."""

        if name not in self.direct_kernels:
            raise NativeCAMError(f"unknown direct CAM kernel {name!r}")
        call = self._abi.operations[f"direct_kernel.{name}"]
        return tuple(argument.field for argument in call.arguments)

    def finalize(self, pool: PICAMStatePool, *, fcomm: int) -> None:
        if self._state_bridge is not None and not self._state_bridge.zero_copy:
            self._state_bridge.copy_to_native(pool)
        if "finalize" in self._operations:
            self._call("finalize", pool, fcomm)


class _NativeStateBridge:
    """Bind Python state directly, with schema-v1 mirror compatibility."""

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
        raw_defaults = payload.get("dimension_defaults", {})
        if not isinstance(raw_defaults, Mapping):
            raise NativeCAMError("PI-CAM state dimension defaults are invalid")
        self.dimension_defaults = {
            str(name): int(value) for name, value in raw_defaults.items()
        }
        ownership = str(payload.get("ownership"))
        self.zero_copy = ownership in {
            "python-owned-zero-copy-pointer-shell",
            "python-owned-zero-copy-derived-shell",
        }
        self.derived_shell = ownership == "python-owned-zero-copy-derived-shell"
        raw_owners = payload.get("owners", ())
        if not isinstance(raw_owners, list):
            raise NativeCAMError("PI-CAM state owner records are invalid")
        self.owners = tuple(
            dict(owner) for owner in raw_owners if isinstance(owner, Mapping)
        )
        if len(self.owners) != len(raw_owners):
            raise NativeCAMError("PI-CAM state owner record is invalid")
        raw_initializers = payload.get("initializers", {})
        if not isinstance(raw_initializers, Mapping):
            raise NativeCAMError("PI-CAM state initializers are invalid")
        self.initializers = dict(raw_initializers)
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
        self._initialization_context: dict[str, object] | None = None
        context_symbol = symbols.get("context")
        self._context = (
            None if context_symbol is None else getattr(library, str(context_symbol), None)
        )
        if self.derived_shell and self._context is None:
            raise NativeCAMError(
                f"{library_name} lacks the Python state initialization context ABI"
            )

    def read_initialization_context(self) -> Mapping[str, object]:
        """Read exact rank-local chunk metadata and native sentinel bits."""

        if self._context is None:
            self._initialization_context = {}
            return self._initialization_context
        chunk_begin = ctypes.c_int32()
        chunk_end = ctypes.c_int32()
        capacity = 65536
        ncols = (ctypes.c_int32 * capacity)()
        inf_bits = ctypes.c_int64()
        posinf_bits = ctypes.c_int64()
        self._context.argtypes = [
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self._context.restype = ctypes.c_int32
        status = int(
            self._context(
                capacity,
                ctypes.byref(chunk_begin),
                ctypes.byref(chunk_end),
                ncols,
                ctypes.byref(inf_bits),
                ctypes.byref(posinf_bits),
            )
        )
        if status:
            raise NativeCAMError(
                f"native PI-CAM initialization context failed ({status})"
            )
        chunks = int(chunk_end.value - chunk_begin.value + 1)
        if chunks < 1 or chunks > capacity:
            raise NativeCAMError("native PI-CAM initialization chunk range is invalid")
        self._initialization_context = {
            "chunk_begin": int(chunk_begin.value),
            "chunk_end": int(chunk_end.value),
            "chunk_ncols": tuple(int(ncols[index]) for index in range(chunks)),
            "inf_bits": int(inf_bits.value),
            "posinf_bits": int(posinf_bits.value),
        }
        return self._initialization_context

    @staticmethod
    def _float_from_bits(value: int) -> np.float64:
        return np.asarray(value, dtype=np.int64).view(np.float64)[()]

    def _initialize_python_state(
        self,
        pool: PICAMStatePool,
        *,
        pcols: int,
    ) -> None:
        context = getattr(self, "_initialization_context", None)
        if self.derived_shell and context is None:
            raise NativeCAMError("PI-CAM state was allocated before native grid setup")
        if context is None:
            return
        chunk_ids = np.arange(
            int(context["chunk_begin"]),
            int(context["chunk_end"]) + 1,
            dtype=np.int32,
        )
        chunk_ncols = np.asarray(context["chunk_ncols"], dtype=np.int32)
        for name, values in (
            ("grid.chunk_id", chunk_ids),
            ("grid.chunk_ncols", chunk_ncols),
        ):
            if name not in pool:
                pool.attach(
                    PICAMFieldContract(
                        name=name,
                        dimensions=("chunks",),
                        dtype="int32",
                        category="native_grid_context",
                        writable=False,
                        restart=False,
                    ),
                    values.copy(order="F"),
                )
        symbolic: dict[str, object] = {
            "chunk_id": chunk_ids,
            "chunk_ncols": chunk_ncols,
            "pcols": int(pcols),
            "inf": self._float_from_bits(int(context["inf_bits"])),
            "posinf": self._float_from_bits(int(context["posinf_bits"])),
        }
        for name, raw_value in getattr(self, "initializers", {}).items():
            if name not in pool:
                continue
            value = symbolic.get(raw_value, raw_value)
            try:
                pool[name][...] = value
            except ValueError as exc:
                raise NativeCAMError(
                    f"cannot apply Python initializer {raw_value!r} to {name!r}"
                ) from exc

    def _owner_layout(
        self, owner: Mapping[str, object]
    ) -> tuple[int, dict[int, int]]:
        """Query the compiler's exact record size and inline member offsets."""

        try:
            function = getattr(self.library, str(owner["layout_symbol"]))
        except (KeyError, AttributeError) as exc:
            raise NativeCAMError(
                f"{self.library_name} lacks layout query for {owner.get('name')!r}"
            ) from exc
        raw_ids = owner.get("inline_field_ids", ())
        if not isinstance(raw_ids, list):
            raise NativeCAMError("PI-CAM inline field ids are invalid")
        expected_ids = tuple(int(value) for value in raw_ids)
        count = max(1, len(expected_ids))
        field_ids = (ctypes.c_int32 * count)()
        offsets = (ctypes.c_int64 * count)()
        record_bytes = ctypes.c_int64()
        function.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int64),
        ]
        function.restype = ctypes.c_int32
        status = int(
            function(
                ctypes.byref(record_bytes),
                len(expected_ids),
                field_ids,
                offsets,
            )
        )
        if status:
            raise NativeCAMError(
                f"native layout query for {owner.get('name')!r} failed ({status})"
            )
        actual_ids = tuple(int(field_ids[index]) for index in range(len(expected_ids)))
        if actual_ids != expected_ids or record_bytes.value < 1:
            raise NativeCAMError(
                f"native layout for {owner.get('name')!r} disagrees with manifest"
            )
        return int(record_bytes.value), {
            field_id: int(offsets[index])
            for index, field_id in enumerate(expected_ids)
        }

    @staticmethod
    def _aligned_zero_bytes(nbytes: int, alignment: int = 64) -> np.ndarray:
        storage = np.zeros(int(nbytes) + int(alignment) - 1, dtype=np.uint8)
        offset = (-int(storage.ctypes.data)) % int(alignment)
        return storage[offset : offset + int(nbytes)]

    def _preallocate_owner_records(
        self,
        pool: PICAMStatePool,
        dimensions: Mapping[str, int],
    ) -> None:
        fields_by_id = {int(field["field_id"]): field for field in self.fields}
        chunks = int(dimensions["chunks"])
        for owner in self.owners:
            owner_name = str(owner["name"])
            raw_name = str(owner["raw_field"])
            record_bytes, offsets = self._owner_layout(owner)
            byte_dimension = f"__native_owner_{owner_name}_bytes"
            total_bytes = record_bytes * chunks
            pool.dimensions[byte_dimension] = total_bytes
            raw = self._aligned_zero_bytes(total_bytes)
            pool.attach(
                PICAMFieldContract(
                    name=raw_name,
                    dimensions=(byte_dimension,),
                    dtype="uint8",
                    category="native_cam_owner",
                    restart=False,
                ),
                raw,
            )
            for field_id, member_offset in offsets.items():
                field = fields_by_id[field_id]
                name = str(field["name"])
                dimension_names = tuple(str(value) for value in field["dimensions"])
                shape = tuple(int(dimensions[value]) for value in dimension_names)
                dtype = np.dtype(str(field["dtype"]))
                component_shape = shape[:-1]
                component_values = int(np.prod(component_shape, dtype=np.int64))
                component_bytes = max(1, component_values) * dtype.itemsize
                if member_offset < 0 or member_offset + component_bytes > record_bytes:
                    raise NativeCAMError(
                        f"inline field {name!r} exceeds its {owner_name} record"
                    )
                component_strides: list[int] = []
                stride = dtype.itemsize
                for extent in component_shape:
                    component_strides.append(stride)
                    stride *= int(extent)
                view = np.ndarray(
                    shape=shape,
                    dtype=dtype,
                    buffer=raw,
                    offset=member_offset,
                    strides=tuple((*component_strides, record_bytes)),
                )
                pool.attach(
                    PICAMFieldContract(
                        name=name,
                        dimensions=dimension_names,
                        dtype=dtype.str,
                        category="native_cam_inline_state",
                        requires_contiguous=False,
                    ),
                    view,
                )

    def preallocate(
        self,
        pool: PICAMStatePool,
        config: object,
        *,
        rank: int,
        size: int,
    ) -> None:
        """Create Python-owned arrays from the generated symbolic contract."""

        defaults = {
            str(name): int(value)
            for name, value in dict(
                getattr(self, "dimension_defaults", {})
            ).items()
        }
        pcols = int(getattr(config, "pcols"))
        pver = int(getattr(config, "pver"))
        resolution = str(getattr(config, "resolution"))
        if not resolution.startswith("ne") or not resolution[2:].isdigit():
            raise NativeCAMError(f"cannot derive PI-CAM grid from {resolution!r}")
        ne = int(resolution[2:])
        element_count = 6 * ne * ne
        base, remainder = divmod(element_count, int(size))
        local_elements = base + int(int(rank) < remainder)
        columns_per_element = int(defaults["physics_columns_per_element"])
        estimated_chunks = (local_elements * columns_per_element + pcols - 1) // pcols
        context = getattr(self, "_initialization_context", None)
        chunks = (
            estimated_chunks
            if context is None
            else len(tuple(context["chunk_ncols"]))
        )
        if context is not None and chunks != estimated_chunks:
            raise NativeCAMError(
                "native PI-CAM chunk count differs from Python grid estimate: "
                f"{chunks} != {estimated_chunks}"
            )
        dimensions = {
            **defaults,
            "pcols": pcols,
            "pver": pver,
            "pverp": pver + 1,
            "chunks": chunks,
            "nphys_local": (
                local_elements * columns_per_element
                if context is None
                else sum(int(value) for value in context["chunk_ncols"])
            ),
        }
        dimensions["column"] = dimensions["nphys_local"]
        dimensions["level"] = pver
        dimensions["interface_level"] = pver + 1
        for name, value in dimensions.items():
            existing = pool.dimensions.setdefault(name, int(value))
            if existing != int(value):
                raise NativeCAMError(
                    f"Python state dimension {name!r}={existing} conflicts with "
                    f"native contract value {value}"
                )
        if self.derived_shell:
            self._preallocate_owner_records(pool, dimensions)
        for field in self.fields:
            if not bool(field.get("active_by_default", True)):
                continue
            name = str(field["name"])
            if name in pool:
                continue
            raw_dimensions = field.get("dimensions")
            if not isinstance(raw_dimensions, list):
                # Old manifests remain readable but cannot preallocate state.
                return
            dimension_names = tuple(str(value) for value in raw_dimensions)
            missing = [value for value in dimension_names if value not in dimensions]
            if missing:
                raise NativeCAMError(
                    f"Python state field {name!r} has unknown dimensions {missing}"
                )
            contract = PICAMFieldContract(
                name=name,
                dimensions=dimension_names,
                dtype=str(field["dtype"]),
                category="native_cam_state",
            )
            if name not in pool:
                pool.create(contract)
        self._initialize_python_state(pool, pcols=pcols)

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
            if name not in pool:
                pool.ensure_from_array(
                    name,
                    np.empty(shape, dtype=dtype, order="F"),
                    category="native_cam_state",
                )
            else:
                array = pool[name]
                if array.shape != shape or array.dtype != dtype:
                    raise NativeCAMError(
                        f"preallocated Python state field {name!r} has "
                        f"{array.shape}/{array.dtype}; native CAM requires "
                        f"{shape}/{dtype}"
                    )
            if not self.zero_copy:
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
