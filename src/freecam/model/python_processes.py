"""Trusted Notebook-defined Python processes executed on rank-local state."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import platform
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import cloudpickle
import numpy as np

from .collective import collective_error_message
from .ccpp_suite import PHYSICS_BEFORE_COUPLER, SuiteScheme
from .errors import (
    PythonProcessContractError,
    PythonProcessExecutionError,
    PythonProcessTaintedError,
    StateOwnershipError,
)

PYTHON_PROCESS_SCHEMA_VERSION = 2
DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_PARAMETER_BYTES = 64 * 1024


def _safe_name(value: str, label: str) -> str:
    text = str(value).strip().lower()
    if not text or not text[0].isalpha():
        raise ValueError(f"{label} must start with a letter")
    if any(not (character.isalnum() or character in "_.-") for character in text):
        raise ValueError(
            f"{label} may contain only letters, digits, dot, dash, and underscore"
        )
    return text


def _payload_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_parameter_value(value: Any, path: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise PythonProcessContractError(
                f"Python process parameter {path} must be finite"
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PythonProcessContractError(
                    f"Python process parameter {path} contains a non-string key"
                )
            normalized[key] = _normalize_parameter_value(item, f"{path}.{key}")
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _normalize_parameter_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise PythonProcessContractError(
        f"Python process parameter {path} has unsupported type "
        f"{type(value).__name__}; use JSON-compatible scalar values or put "
        "large arrays in StatePool fields"
    )


def normalize_python_parameters(
    values: Mapping[str, Any] | None,
    *,
    max_parameter_bytes: int = DEFAULT_MAX_PARAMETER_BYTES,
) -> dict[str, Any]:
    """Validate and copy a small keyword-parameter mapping."""

    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("Python process parameters must be a mapping")
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise PythonProcessContractError(
                f"Python process parameter name {key!r} must be a valid "
                "Python identifier"
            )
        normalized[key] = _normalize_parameter_value(value, key)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > int(max_parameter_bytes):
        raise PythonProcessContractError(
            f"serialized Python process parameters are {len(encoded)} bytes; "
            f"limit is {int(max_parameter_bytes)} bytes. Store large data in "
            "StatePool fields instead"
        )
    return normalized


def _parameter_hash(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_signature(
    function: Callable[..., Any],
    parameters: Mapping[str, Any] | None = None,
) -> None:
    values = dict(parameters or {})
    try:
        signature = inspect.signature(function)
        signature.bind(object(), object(), **values)
    except (TypeError, ValueError) as exc:
        raise PythonProcessContractError(
            "Python process must accept exactly two positional parameters "
            "(fields, context); custom parameters must be declared as "
            "keyword-only arguments and supplied by parameters={...}"
        ) from exc
    declared = tuple(signature.parameters.values())
    if any(
        parameter.kind
        in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in declared
    ):
        raise PythonProcessContractError("Python process may not use *args or **kwargs")
    positional = tuple(
        parameter
        for parameter in declared
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    keyword_only = tuple(
        parameter
        for parameter in declared
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    )
    if len(positional) != 2 or len(declared) != 2 + len(keyword_only):
        raise PythonProcessContractError(
            "Python process must accept exactly two positional parameters "
            "(fields, context); custom parameters must be keyword-only"
        )


@dataclass(frozen=True, slots=True)
class PythonProcessSpec:
    """Serializable contract for one trusted Notebook callback."""

    name: str
    payload: bytes
    payload_hash: str
    group: str = PHYSICS_BEFORE_COUPLER
    before: str | None = None
    after: str | None = None
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    parameters: Mapping[str, Any] | None = None
    max_parameter_bytes: int = DEFAULT_MAX_PARAMETER_BYTES
    enabled: bool = True
    transactional: bool = True
    source: str | None = None
    python_version: str = platform.python_version()
    cloudpickle_version: str = cloudpickle.__version__
    # A native process may call the image's numerical routines directly --
    # direct kernels, bind(C) handles, module variables -- rather than only
    # reading and writing StatePool fields.  It receives the driver's native
    # access object in its context.  Off by default: an ordinary process
    # never sees a pointer.
    native: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_name(self.name, "process name"))
        object.__setattr__(self, "payload", bytes(self.payload))
        object.__setattr__(self, "group", str(self.group).strip().lower())
        object.__setattr__(
            self, "reads", tuple(str(item).strip() for item in self.reads)
        )
        object.__setattr__(
            self, "writes", tuple(str(item).strip() for item in self.writes)
        )
        object.__setattr__(self, "max_parameter_bytes", int(self.max_parameter_bytes))
        if self.max_parameter_bytes <= 0:
            raise ValueError("max_parameter_bytes must be positive")
        object.__setattr__(
            self,
            "parameters",
            normalize_python_parameters(
                self.parameters,
                max_parameter_bytes=self.max_parameter_bytes,
            ),
        )
        if not self.group:
            raise ValueError("Python process group must be non-empty")
        if self.before is not None and self.after is not None:
            raise ValueError("provide at most one of before= or after=")
        if len(set(self.reads)) != len(self.reads):
            raise ValueError("Python process reads contains duplicate fields")
        if len(set(self.writes)) != len(self.writes):
            raise ValueError("Python process writes contains duplicate fields")
        overlap = set(self.reads) & set(self.writes)
        if overlap:
            raise ValueError(
                f"write fields are already readable; remove them from reads: "
                f"{sorted(overlap)}"
            )
        if _payload_hash(self.payload) != self.payload_hash:
            raise PythonProcessContractError(
                f"Python process {self.name!r} payload hash does not match"
            )

    @classmethod
    def from_callable(
        cls,
        function: Callable[["PythonFieldView", "PythonProcessContext"], None],
        *,
        name: str | None = None,
        group: str = PHYSICS_BEFORE_COUPLER,
        before: str | None = None,
        after: str | None = None,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        enabled: bool = True,
        transactional: bool = True,
        native: bool = False,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_parameter_bytes: int = DEFAULT_MAX_PARAMETER_BYTES,
    ) -> "PythonProcessSpec":
        if not callable(function):
            raise TypeError("Python process must be callable")
        parameter_values = normalize_python_parameters(
            parameters,
            max_parameter_bytes=max_parameter_bytes,
        )
        _validate_signature(function, parameter_values)
        payload = cloudpickle.dumps(function)
        if len(payload) > int(max_payload_bytes):
            raise PythonProcessContractError(
                f"serialized Python process is {len(payload)} bytes; limit is "
                f"{int(max_payload_bytes)} bytes. Store large data in StatePool "
                "fields instead of capturing it in the function closure."
            )
        try:
            restored = cloudpickle.loads(payload)
        except BaseException as exc:
            raise PythonProcessContractError(
                f"cannot deserialize Python process: {type(exc).__name__}: {exc}"
            ) from exc
        if not callable(restored):
            raise PythonProcessContractError(
                "serialized Python process did not restore as a callable"
            )
        _validate_signature(restored, parameter_values)
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = None
        return cls(
            name=name or getattr(function, "__name__", "python_process"),
            payload=payload,
            payload_hash=_payload_hash(payload),
            group=group,
            before=before,
            after=after,
            reads=tuple(reads),
            writes=tuple(writes),
            parameters=parameter_values,
            max_parameter_bytes=max_parameter_bytes,
            enabled=enabled,
            transactional=transactional,
            native=native,
            source=source,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PYTHON_PROCESS_SCHEMA_VERSION,
            "name": self.name,
            "payload_base64": base64.b64encode(self.payload).decode("ascii"),
            "payload_hash": self.payload_hash,
            "group": self.group,
            "before": self.before,
            "after": self.after,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "parameters": dict(self.parameters or {}),
            "max_parameter_bytes": self.max_parameter_bytes,
            "enabled": self.enabled,
            "transactional": self.transactional,
            "native": self.native,
            "source": self.source,
            "python_version": self.python_version,
            "cloudpickle_version": self.cloudpickle_version,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PythonProcessSpec":
        version = int(values.get("schema_version", PYTHON_PROCESS_SCHEMA_VERSION))
        if version not in {1, PYTHON_PROCESS_SCHEMA_VERSION}:
            raise ValueError(f"unsupported Python process schema version {version}")
        try:
            payload = base64.b64decode(
                str(values["payload_base64"]).encode("ascii"), validate=True
            )
        except (KeyError, ValueError) as exc:
            raise PythonProcessContractError(
                "Python process payload is not valid base64"
            ) from exc
        return cls(
            name=str(values["name"]),
            payload=payload,
            payload_hash=str(values["payload_hash"]),
            group=str(values.get("group", PHYSICS_BEFORE_COUPLER)),
            before=(None if values.get("before") is None else str(values["before"])),
            after=(None if values.get("after") is None else str(values["after"])),
            reads=tuple(str(item) for item in values.get("reads", ())),
            writes=tuple(str(item) for item in values.get("writes", ())),
            parameters=({} if version == 1 else dict(values.get("parameters", {}))),
            max_parameter_bytes=int(
                values.get("max_parameter_bytes", DEFAULT_MAX_PARAMETER_BYTES)
            ),
            enabled=bool(values.get("enabled", True)),
            transactional=bool(values.get("transactional", True)),
            native=bool(values.get("native", False)),
            source=(None if values.get("source") is None else str(values["source"])),
            python_version=str(values.get("python_version", platform.python_version())),
            cloudpickle_version=str(
                values.get("cloudpickle_version", cloudpickle.__version__)
            ),
        )

    def with_parameters(self, updates: Mapping[str, Any]) -> "PythonProcessSpec":
        values = dict(self.parameters or {})
        values.update(
            normalize_python_parameters(
                updates,
                max_parameter_bytes=self.max_parameter_bytes,
            )
        )
        values = normalize_python_parameters(
            values,
            max_parameter_bytes=self.max_parameter_bytes,
        )
        function = cloudpickle.loads(self.payload)
        _validate_signature(function, values)
        return replace(self, parameters=values)


class NativeAccess:
    """What a native process may reach: the image, the pool, the kernels.

    A Python transliteration of a Fortran driver is not a NumPy process; it
    asks the image to run its routines and hands them the image's own
    storage.  This is the narrowest object that lets it: the loaded library
    for bind(C) entries and module variables, the rank-local pool for the
    chunk table, the Fortran communicator handle, and one call that runs a
    direct kernel on a mapping of arrays and records it in the action trace
    like any other kernel run.  It holds the driver by reference and copies
    nothing.
    """

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    @property
    def library(self) -> Any:
        """The loaded CAM image, for ctypes entry points and module views."""

        backend = self._driver.backend
        library = getattr(backend, "_library", None)
        if library is None:
            raise PythonProcessContractError("the backend exposes no loaded image")
        return library

    @property
    def pool(self) -> Any:
        return self._driver.pool

    @property
    def fcomm(self) -> int:
        return int(self._driver.fcomm)

    @property
    def chunks(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lchnk, ncol)`` for every chunk of this rank, in pool order."""

        pool = self._driver.pool
        return (
            np.asarray(pool["grid.chunk_id"], dtype=np.int64).reshape(-1),
            np.asarray(pool["grid.chunk_ncols"], dtype=np.int64).reshape(-1),
        )

    def kernel_arguments(self, name: str) -> tuple[Mapping[str, Any], ...]:
        """The manifest's argument records for direct kernel ``name``."""

        operations = getattr(self._driver.backend, "_operations", {})
        try:
            return tuple(operations[f"direct_kernel.{name}"]["arguments"])
        except KeyError as exc:
            raise PythonProcessContractError(
                f"the image declares no direct kernel {name!r}"
            ) from exc

    def run_kernel(self, name: str, arrays: Mapping[str, np.ndarray]) -> None:
        """Run direct kernel ``name`` on ``arrays`` (one chunk per last axis)."""

        self._driver.run_kernel(name, experimental=True, pool=arrays)


@dataclass(frozen=True, slots=True)
class PythonProcessContext:
    """Read-only model metadata provided at a Python process boundary."""

    process_name: str
    group: str
    rank: int
    size: int
    step: int
    timestep_seconds: int
    year: int
    month: int
    day: int
    seconds: int
    calendar: str
    # Only for a process installed with native=True: the driver's numerical
    # image, its rank-local pool, and a way to run a direct kernel.  None for
    # every other process.
    native: Any = None

    @property
    def date(self) -> tuple[int, int, int]:
        """Current model date as ``(year, month, day)``."""

        return (self.year, self.month, self.day)


class PythonFieldView(Mapping[str, np.ndarray]):
    """Restricted mapping of declared rank-local StatePool arrays."""

    def __init__(
        self,
        pool: Any,
        *,
        reads: Mapping[str, str],
        writes: Mapping[str, str],
    ) -> None:
        self._pool = pool
        self._reads = dict(reads)
        self._writes = dict(writes)
        self._names = (*self._reads, *self._writes)

    def __getitem__(self, name: str) -> np.ndarray:
        key = str(name)
        if key in self._writes:
            # Return a non-owning writable view.  The callback can update
            # values in place, but NumPy will reject operations such as
            # ``resize`` that would replace StatePool-owned storage.
            return self._pool.get(self._writes[key]).view()
        try:
            value = self._pool.get(self._reads[key])
        except KeyError:
            raise KeyError(
                f"Python process field {key!r} was not declared in reads/writes"
            ) from None
        view = value.view()
        view.flags.writeable = False
        return view

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class PythonStateView(Mapping[str, np.ndarray]):
    """Attribute-style view of one callback's declared StatePool fields.

    ``state.T`` is still the rank-local NumPy view owned by StatePool; this
    class only translates friendly attribute names and enforces the existing
    read/write declaration.  It never copies a field or performs MPI work.
    """

    _COMMON_ALIASES = {
        "T": ("T", "phys_state.t", "air_temperature"),
        "u": ("u", "phys_state.u", "eastward_wind"),
        "v": ("v", "phys_state.v", "northward_wind"),
        "q": ("q", "phys_state.q", "specific_humidity"),
    }

    def __init__(self, fields: PythonFieldView) -> None:
        object.__setattr__(self, "_fields", fields)

    @classmethod
    def field_candidates(cls, name: str) -> tuple[str, ...]:
        """Return canonical candidates for one user-facing state name."""

        token = str(name)
        return cls._COMMON_ALIASES.get(token, (token,))

    def _key(self, name: str) -> str:
        for candidate in self.field_candidates(name):
            if candidate in self._fields:
                return candidate
        leaf_matches = tuple(
            key for key in self._fields if key.rsplit(".", 1)[-1] == str(name)
        )
        if len(leaf_matches) == 1:
            return leaf_matches[0]
        raise KeyError(name)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._fields[self._key(str(name))]

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        target = self[name]
        if value is target:
            # ``state.T += value`` mutates the view first, then Python assigns
            # the same object back to the attribute.
            return
        if not target.flags.writeable:
            raise StateOwnershipError(
                f"Python process field {name!r} was declared read-only"
            )
        np.copyto(target, value, casting="same_kind")

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __dir__(self) -> list[str]:
        names = {name for name in self._fields if str(name).isidentifier()}
        for alias in self._COMMON_ALIASES:
            try:
                self._key(alias)
            except KeyError:
                continue
            names.add(alias)
        return sorted(set(super().__dir__()) | names)


@dataclass(slots=True)
class RegisteredPythonProcess:
    """Rank-local registry record for one installed Python process."""

    spec: PythonProcessSpec
    scheme_key: str
    read_bindings: Mapping[str, str]
    write_bindings: Mapping[str, str]
    #: The callable, materialised from the payload once and kept.  Loading it
    #: per invocation handed every step a fresh copy of whatever it closed
    #: over -- a bound method's object included -- so nothing a process kept
    #: between steps survived, and a stage that caches its runtime rebuilt it
    #: every step.
    function: Any = None

    @property
    def name(self) -> str:
        return self.spec.name

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "scheme_key": self.scheme_key,
            "read_bindings": dict(self.read_bindings),
            "write_bindings": dict(self.write_bindings),
        }


class PythonProcessRegistry:
    """Collectively install and run trusted Python callbacks."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self.installed: dict[str, RegisteredPythonProcess] = {}

    @property
    def process_names(self) -> frozenset[str]:
        return frozenset(self.installed)

    def has_process(self, process: str, *, source_group: str | None = None) -> bool:
        record = self.installed.get(str(process).lower())
        if record is None:
            return False
        return source_group in {None, f"python:{record.name}"}

    def install(
        self,
        spec: PythonProcessSpec | Mapping[str, Any],
        *,
        unsafe: bool = False,
    ) -> RegisteredPythonProcess:
        if not isinstance(spec, PythonProcessSpec):
            spec = PythonProcessSpec.from_mapping(spec)
        if not spec.transactional and not unsafe:
            raise ValueError(
                "transactional=False requires unsafe=True because a failed "
                "callback can leave partially modified arrays"
            )
        self._require_boundary()
        local_error: str | None = None
        read_bindings: dict[str, str] = {}
        write_bindings: dict[str, str] = {}
        try:
            if spec.name in self.installed:
                raise PythonProcessContractError(
                    f"Python process {spec.name!r} is already installed"
                )
            if self.driver.processes.provider_for_process(spec.name) is not None:
                raise PythonProcessContractError(
                    f"process name {spec.name!r} already has a provider"
                )
            if spec.group not in self.driver.scheme_plan.group_names:
                raise PythonProcessContractError(f"suite has no group {spec.group!r}")
            if spec.before is not None:
                self.driver.scheme_plan.scheme(spec.before, group=spec.group)
            if spec.after is not None:
                self.driver.scheme_plan.scheme(spec.after, group=spec.group)
            function = cloudpickle.loads(spec.payload)
            _validate_signature(function, spec.parameters)
            read_bindings = {name: self._resolve_field(name) for name in spec.reads}
            write_bindings = {name: self._resolve_field(name) for name in spec.writes}
            overlap = set(read_bindings.values()) & set(write_bindings.values())
            if overlap:
                raise PythonProcessContractError(
                    "Python process read and write declarations resolve to "
                    f"the same StatePool fields: {sorted(overlap)}. Declare "
                    "those fields only in writes."
                )
            for exposed, resolved in write_bindings.items():
                if not self.driver.pool.contract(resolved).writable:
                    raise StateOwnershipError(
                        f"Python process write field {exposed!r} resolves to "
                        f"read-only StatePool field {resolved!r}"
                    )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, "Python process preflight")

        hashes = self.driver.comm.allgather(spec.payload_hash)
        if len(set(hashes)) != 1:
            raise PythonProcessContractError(
                f"Python process payload differs across MPI ranks: {hashes}"
            )

        installed_scheme: SuiteScheme | None = None
        local_error = None
        try:
            scheme = SuiteScheme(
                name=spec.name,
                source_group=f"python:{spec.name}",
                occurrence=0,
                group=spec.group,
                category="plugin",
                description=f"Notebook Python process {spec.name}",
                implementation="python-runtime-process",
                required=False,
                enabled=spec.enabled,
            )
            installed_scheme = self.driver.scheme_plan.add(
                scheme,
                before=spec.before,
                after=spec.after,
                unsafe=True,
            )
            record = RegisteredPythonProcess(
                spec=spec,
                scheme_key=installed_scheme.key,
                read_bindings=read_bindings,
                write_bindings=write_bindings,
            )
            self.installed[spec.name] = record
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            self.installed.pop(spec.name, None)
            if installed_scheme is not None:
                self.driver.scheme_plan.remove(installed_scheme.key, unsafe=True)
            message = collective_error_message(
                "Python process installation rollback", errors
            )
            raise PythonProcessContractError(
                message or "Python process installation failed"
            )
        self.driver.comm.barrier()
        return self.installed[spec.name]

    def remove(self, name: str) -> dict[str, Any]:
        self._require_boundary()
        key = str(name).lower()
        record = self.installed.get(key)
        self._collective_error(
            None if record is not None else f"unknown Python process {key!r}",
            "Python process removal",
        )
        assert record is not None
        removed = self.driver.scheme_plan.remove(record.scheme_key, unsafe=True)
        self.installed.pop(key)
        self.driver.comm.barrier()
        return {
            "name": record.name,
            "scheme_key": removed.key,
            "payload_hash": record.spec.payload_hash,
        }

    def set_parameters(self, name: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        """Collectively update persistent keyword parameters for one callback."""

        self._require_boundary()
        key = str(name).lower()
        record = self.installed.get(key)
        local_error: str | None = None
        candidate: PythonProcessSpec | None = None
        try:
            if record is None:
                raise PythonProcessContractError(f"unknown Python process {key!r}")
            candidate = record.spec.with_parameters(updates)
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, "Python process parameter update")
        assert record is not None and candidate is not None
        hashes = self.driver.comm.allgather(_parameter_hash(candidate.parameters or {}))
        if len(set(hashes)) != 1:
            raise PythonProcessContractError(
                f"Python process parameters differ across MPI ranks: {hashes}"
            )
        record.spec = candidate
        self.driver.comm.barrier()
        return {
            "name": record.name,
            "parameters": dict(record.spec.parameters or {}),
            "payload_hash": record.spec.payload_hash,
        }

    def invoke(
        self,
        scheme: SuiteScheme,
        pool: Any,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            record = self.installed[scheme.name]
        except KeyError as exc:
            raise PythonProcessContractError(
                f"Python process {scheme.name!r} is not installed"
            ) from exc
        local_error: str | None = None
        effective_parameters: dict[str, Any] = {}
        try:
            effective_parameters = dict(record.spec.parameters or {})
            effective_parameters.update(
                normalize_python_parameters(
                    parameters,
                    max_parameter_bytes=record.spec.max_parameter_bytes,
                )
            )
            effective_parameters = normalize_python_parameters(
                effective_parameters,
                max_parameter_bytes=record.spec.max_parameter_bytes,
            )
            function = cloudpickle.loads(record.spec.payload)
            _validate_signature(function, effective_parameters)
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, "Python process parameter preflight")
        parameter_hashes = self.driver.comm.allgather(
            _parameter_hash(effective_parameters)
        )
        if len(set(parameter_hashes)) != 1:
            raise PythonProcessContractError(
                "Python process effective parameters differ across MPI ranks: "
                f"{parameter_hashes}"
            )

        snapshots = (
            {
                name: np.array(pool.get(resolved), copy=True, order="F")
                for name, resolved in record.write_bindings.items()
            }
            if record.spec.transactional
            else {}
        )
        before = pool.pointer_records()
        read_writeability = {
            resolved: bool(pool.get(resolved).flags.writeable)
            for resolved in set(record.read_bindings.values())
        }
        local_error = None
        try:
            for resolved in read_writeability:
                pool.get(resolved).flags.writeable = False
            function = record.function
            if function is None:
                function = cloudpickle.loads(record.spec.payload)
                record.function = function
            fields = PythonFieldView(
                pool,
                reads=record.read_bindings,
                writes=record.write_bindings,
            )
            clock = self.driver.clock
            native = NativeAccess(self.driver) if record.spec.native else None
            context = PythonProcessContext(
                process_name=record.name,
                group=str(scheme.group),
                rank=int(self.driver.comm.rank),
                size=int(self.driver.comm.size),
                step=int(clock.nstep),
                timestep_seconds=int(clock.dt_seconds),
                year=int(clock.year),
                month=int(clock.month),
                day=int(clock.day),
                seconds=int(clock.seconds),
                calendar=str(clock.calendar),
                native=native,
            )
            result = function(fields, context, **effective_parameters)
            if result is not None:
                raise PythonProcessContractError(
                    f"Python process {record.name!r} must return None, got "
                    f"{type(result).__name__}"
                )
            pool.assert_pointer_stability(before)
        except BaseException:
            local_error = traceback.format_exc()
        finally:
            for resolved, writable in read_writeability.items():
                pool.get(resolved).flags.writeable = writable
        errors = self.driver.comm.allgather(local_error)
        failure = collective_error_message(
            f"Python process {record.name!r}", errors
        )
        if failure is not None:
            if record.spec.transactional:
                for name, values in snapshots.items():
                    np.copyto(
                        pool.get(record.write_bindings[name]),
                        values,
                        casting="no",
                    )
                self.driver.comm.barrier()
                raise PythonProcessExecutionError(
                    f"Python process {record.name!r} failed and its declared "
                    f"write fields were restored:\n" + failure
                )
            raise PythonProcessTaintedError(
                f"Python process {record.name!r} failed without rollback; "
                f"the model state is tainted:\n" + failure
            )

    def inventory(self) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        for name in sorted(self.installed):
            record = self.installed[name]
            scheme = self.driver.scheme_plan.scheme(record.scheme_key)
            rows = self.driver.scheme_plan.describe(scheme.group)
            index = next(
                position
                for position, row in enumerate(rows)
                if row["key"] == record.scheme_key
            )
            values = record.as_dict()
            values["spec"] = {
                **values["spec"],
                "group": str(scheme.group),
                "enabled": bool(scheme.enabled),
            }
            values["placement"] = {
                "group": str(scheme.group),
                "previous": (None if index == 0 else str(rows[index - 1]["key"])),
                "next": (
                    None if index + 1 == len(rows) else str(rows[index + 1]["key"])
                ),
            }
            records.append(values)
        return tuple(records)

    def prune_to_plan(self) -> None:
        """Drop registry records whose runtime suite nodes were removed."""

        live_keys = {
            scheme.key
            for scheme in self.driver.scheme_plan.schemes
            if scheme.implementation == "python-runtime-process"
        }
        for name, record in tuple(self.installed.items()):
            if record.scheme_key not in live_keys:
                self.installed.pop(name)

    def restore_inventory(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Restore exact callback bytes without executing callback code."""

        local_identity = tuple(
            (
                str(values["spec"]["name"]),
                str(values["spec"]["payload_hash"]),
            )
            for values in records
        )
        identities = self.driver.comm.allgather(local_identity)
        if any(identity != local_identity for identity in identities):
            raise PythonProcessContractError(
                "checkpoint Python process inventory differs across MPI ranks"
            )
        restored_records: dict[str, RegisteredPythonProcess] = {}
        for values in records:
            spec = PythonProcessSpec.from_mapping(values["spec"])
            try:
                function = cloudpickle.loads(spec.payload)
                _validate_signature(function, spec.parameters)
            except BaseException as exc:
                raise PythonProcessContractError(
                    f"cannot restore Python process {spec.name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if spec.name in self.installed or spec.name in restored_records:
                raise PythonProcessContractError(
                    f"Python process {spec.name!r} is already restored"
                )
            scheme_key = str(values["scheme_key"])
            scheme = self.driver.scheme_plan.scheme(scheme_key)
            if scheme.implementation != "python-runtime-process":
                raise PythonProcessContractError(
                    f"scheme {scheme_key!r} is not a Python runtime process"
                )
            read_bindings = {name: self._resolve_field(name) for name in spec.reads}
            write_bindings = {name: self._resolve_field(name) for name in spec.writes}
            overlap = set(read_bindings.values()) & set(write_bindings.values())
            if overlap:
                raise PythonProcessContractError(
                    "restored Python process read/write fields overlap: "
                    f"{sorted(overlap)}"
                )
            for exposed, resolved in write_bindings.items():
                if not self.driver.pool.contract(resolved).writable:
                    raise StateOwnershipError(
                        f"restored Python process write field {exposed!r} "
                        f"resolves to read-only field {resolved!r}"
                    )
            restored_records[spec.name] = RegisteredPythonProcess(
                spec=spec,
                scheme_key=scheme_key,
                read_bindings=read_bindings,
                write_bindings=write_bindings,
            )
        self.installed.update(restored_records)

    def _resolve_field(self, name: str) -> str:
        token = str(name)
        if token.startswith("field:"):
            resolved = token.removeprefix("field:")
            self.driver.pool.get(resolved)
            return resolved
        if token.startswith("ccpp:"):
            return self.driver.pool.ccpp_field_name(token.removeprefix("ccpp:"))
        try:
            # Unqualified CCPP standard names take precedence because runtime
            # physics callbacks operate at CCPP scheme boundaries.  Use the
            # explicit ``field:`` prefix for a colliding canonical field.
            return self.driver.pool.ccpp_field_name(token)
        except KeyError:
            self.driver.pool.get(token)
            return token

    def _require_boundary(self) -> None:
        if self.driver.pool is None:
            raise StateOwnershipError(
                "initialize the driver before changing runtime state"
            )
        if getattr(self.driver, "_native_call_depth", 0):
            raise StateOwnershipError(
                "Python processes cannot change inside a process call"
            )
        cursor = self.driver.execution_cursor
        cursors = self.driver.comm.allgather(cursor)
        if any(item != cursor for item in cursors):
            raise StateOwnershipError(
                f"MPI ranks are at different execution boundaries: {cursors}"
            )

    def _collective_error(self, error: str | None, operation: str) -> None:
        errors = self.driver.comm.allgather(error)
        message = collective_error_message(operation, errors)
        if message is not None:
            raise PythonProcessContractError(message)
