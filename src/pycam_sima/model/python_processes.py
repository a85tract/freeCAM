"""Trusted Notebook-defined Python processes executed on rank-local state."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import base64
import hashlib
import inspect
import platform
import traceback
from typing import Any

import cloudpickle
import numpy as np

from .ccpp_suite import PHYSICS_BEFORE_COUPLER, SuiteScheme
from .errors import (
    PythonProcessContractError,
    PythonProcessExecutionError,
    PythonProcessTaintedError,
    StateOwnershipError,
)


PYTHON_PROCESS_SCHEMA_VERSION = 1
DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024


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


def _validate_signature(function: Callable[..., Any]) -> None:
    try:
        signature = inspect.signature(function)
        signature.bind(object(), object())
    except (TypeError, ValueError) as exc:
        raise PythonProcessContractError(
            "Python process must accept exactly (fields, context)"
        ) from exc
    parameters = tuple(signature.parameters.values())
    if any(
        parameter.kind
        in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in parameters
    ):
        raise PythonProcessContractError(
            "Python process may not use *args or **kwargs"
        )
    required_or_positional = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    )
    if len(parameters) != 2 or len(required_or_positional) != 2:
        raise PythonProcessContractError(
            "Python process must accept exactly (fields, context)"
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
    enabled: bool = True
    transactional: bool = True
    source: str | None = None
    python_version: str = platform.python_version()
    cloudpickle_version: str = cloudpickle.__version__

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
        enabled: bool = True,
        transactional: bool = True,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> "PythonProcessSpec":
        if not callable(function):
            raise TypeError("Python process must be callable")
        _validate_signature(function)
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
        _validate_signature(restored)
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
            enabled=enabled,
            transactional=transactional,
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
            "enabled": self.enabled,
            "transactional": self.transactional,
            "source": self.source,
            "python_version": self.python_version,
            "cloudpickle_version": self.cloudpickle_version,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PythonProcessSpec":
        version = int(
            values.get("schema_version", PYTHON_PROCESS_SCHEMA_VERSION)
        )
        if version != PYTHON_PROCESS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Python process schema version {version}"
            )
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
            before=(
                None if values.get("before") is None else str(values["before"])
            ),
            after=(
                None if values.get("after") is None else str(values["after"])
            ),
            reads=tuple(str(item) for item in values.get("reads", ())),
            writes=tuple(str(item) for item in values.get("writes", ())),
            enabled=bool(values.get("enabled", True)),
            transactional=bool(values.get("transactional", True)),
            source=(
                None if values.get("source") is None else str(values["source"])
            ),
            python_version=str(
                values.get("python_version", platform.python_version())
            ),
            cloudpickle_version=str(
                values.get("cloudpickle_version", cloudpickle.__version__)
            ),
        )


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


@dataclass(slots=True)
class RegisteredPythonProcess:
    """Rank-local registry record for one installed Python process."""

    spec: PythonProcessSpec
    scheme_key: str
    read_bindings: Mapping[str, str]
    write_bindings: Mapping[str, str]

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

    def has_process(
        self, process: str, *, source_group: str | None = None
    ) -> bool:
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
                raise PythonProcessContractError(
                    f"suite has no group {spec.group!r}"
                )
            if spec.before is not None:
                self.driver.scheme_plan.scheme(spec.before, group=spec.group)
            if spec.after is not None:
                self.driver.scheme_plan.scheme(spec.after, group=spec.group)
            function = cloudpickle.loads(spec.payload)
            _validate_signature(function)
            read_bindings = {
                name: self._resolve_field(name) for name in spec.reads
            }
            write_bindings = {
                name: self._resolve_field(name) for name in spec.writes
            }
            overlap = set(read_bindings.values()) & set(
                write_bindings.values()
            )
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
                self.driver.scheme_plan.remove(
                    installed_scheme.key, unsafe=True
                )
            raise PythonProcessContractError(
                f"Python process installation rolled back collectively: "
                f"{errors}"
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
        removed = self.driver.scheme_plan.remove(
            record.scheme_key, unsafe=True
        )
        self.installed.pop(key)
        self.driver.comm.barrier()
        return {
            "name": record.name,
            "scheme_key": removed.key,
            "payload_hash": record.spec.payload_hash,
        }

    def invoke(self, scheme: SuiteScheme, pool: Any) -> None:
        try:
            record = self.installed[scheme.name]
        except KeyError as exc:
            raise PythonProcessContractError(
                f"Python process {scheme.name!r} is not installed"
            ) from exc
        snapshots = {
            name: np.array(
                pool.get(resolved), copy=True, order="F"
            )
            for name, resolved in record.write_bindings.items()
        } if record.spec.transactional else {}
        before = pool.pointer_records()
        read_writeability = {
            resolved: bool(pool.get(resolved).flags.writeable)
            for resolved in set(record.read_bindings.values())
        }
        local_error: str | None = None
        try:
            for resolved in read_writeability:
                pool.get(resolved).flags.writeable = False
            function = cloudpickle.loads(record.spec.payload)
            fields = PythonFieldView(
                pool,
                reads=record.read_bindings,
                writes=record.write_bindings,
            )
            clock = self.driver.clock
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
            )
            result = function(fields, context)
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
        failures = [
            f"rank {rank}:\n{message}"
            for rank, message in enumerate(errors)
            if message is not None
        ]
        if failures:
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
                    f"write fields were restored:\n" + "\n".join(failures)
                )
            raise PythonProcessTaintedError(
                f"Python process {record.name!r} failed without rollback; "
                f"the model state is tainted:\n" + "\n".join(failures)
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
                "previous": (
                    None if index == 0 else str(rows[index - 1]["key"])
                ),
                "next": (
                    None
                    if index + 1 == len(rows)
                    else str(rows[index + 1]["key"])
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

    def restore_inventory(
        self, records: Sequence[Mapping[str, Any]]
    ) -> None:
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
            if (
                spec.name in self.installed
                or spec.name in restored_records
            ):
                raise PythonProcessContractError(
                    f"Python process {spec.name!r} is already restored"
                )
            scheme_key = str(values["scheme_key"])
            scheme = self.driver.scheme_plan.scheme(scheme_key)
            if scheme.implementation != "python-runtime-process":
                raise PythonProcessContractError(
                    f"scheme {scheme_key!r} is not a Python runtime process"
                )
            read_bindings = {
                name: self._resolve_field(name) for name in spec.reads
            }
            write_bindings = {
                name: self._resolve_field(name) for name in spec.writes
            }
            overlap = set(read_bindings.values()) & set(
                write_bindings.values()
            )
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
            return self.driver.pool.ccpp_field_name(
                token.removeprefix("ccpp:")
            )
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

    def _collective_error(
        self, error: str | None, operation: str
    ) -> None:
        errors = self.driver.comm.allgather(error)
        failures = [
            f"rank {rank}: {message}"
            for rank, message in enumerate(errors)
            if message is not None
        ]
        if failures:
            raise PythonProcessContractError(
                f"{operation} failed collectively: " + "; ".join(failures)
            )
