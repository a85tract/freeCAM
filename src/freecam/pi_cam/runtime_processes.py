"""Notebook-defined Python processes for the PI-CAM execution plan."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, replace
from typing import Any, Mapping

import cloudpickle
import numpy as np

from freecam.model.collective import collective_error_message
from freecam.model.errors import (
    PythonProcessContractError,
    PythonProcessExecutionError,
    PythonProcessTaintedError,
)
from freecam.model.python_processes import (
    PythonFieldView,
    NativeAccess,
    PythonProcessContext,
    PythonProcessSpec,
    PythonStateView,
    _parameter_hash,
    _validate_signature,
)

from .errors import PICAMConfigurationError, PICAMStateError
from .plan import PICAMAction


@dataclass(slots=True)
class RegisteredPICAMPythonProcess:
    spec: PythonProcessSpec
    action_name: str
    read_bindings: Mapping[str, str]
    write_bindings: Mapping[str, str]
    #: The callable, materialised from the payload once at install.  Loading
    #: it per invocation gave every step a fresh copy of whatever the callable
    #: closed over -- a bound method's object included -- so no state could
    #: survive from one step to the next, and a stage that caches its runtime
    #: rebuilt it every step.
    function: Any = None


class PICAMPythonProcessRegistry:
    """Collectively install callbacks into one live PI-CAM model."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self.installed: dict[str, RegisteredPICAMPythonProcess] = {}

    @property
    def process_names(self) -> frozenset[str]:
        return frozenset(self.installed)

    def install(
        self,
        spec: PythonProcessSpec | Mapping[str, Any],
        *,
        unsafe: bool = False,
    ) -> RegisteredPICAMPythonProcess:
        if not isinstance(spec, PythonProcessSpec):
            spec = PythonProcessSpec.from_mapping(spec)
        if not spec.transactional and not unsafe:
            raise PICAMConfigurationError(
                "transactional=False requires unsafe=True"
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
            if spec.group not in self.driver.step_plan.phases:
                raise PythonProcessContractError(
                    f"PI-CAM plan has no phase {spec.group!r}"
                )
            try:
                self.driver.step_plan.select(spec.name, phase=spec.group)
            except PICAMConfigurationError:
                pass
            else:
                raise PythonProcessContractError(
                    f"action name {spec.name!r} already exists in {spec.group!r}"
                )
            function = cloudpickle.loads(spec.payload)
            _validate_signature(function, spec.parameters)
            read_bindings = {
                name: self._resolve_field(name) for name in spec.reads
            }
            write_bindings = {
                name: self._resolve_field(name) for name in spec.writes
            }
            overlap = set(read_bindings.values()) & set(write_bindings.values())
            if overlap:
                raise PythonProcessContractError(
                    "fields resolving to the same storage must appear only in writes: "
                    + ", ".join(sorted(overlap))
                )
            for exposed, resolved in write_bindings.items():
                if not self.driver.pool.contract(resolved).writable:
                    raise PythonProcessContractError(
                        f"write field {exposed!r} resolves to read-only field "
                        f"{resolved!r}"
                    )
            if spec.before is not None:
                self.driver.step_plan.select(spec.before, phase=spec.group)
            if spec.after is not None:
                self.driver.step_plan.select(spec.after, phase=spec.group)
            if spec.before is None and spec.after is None:
                raise PythonProcessContractError(
                    "PI-CAM Python process placement requires before= or after="
                )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, "Python process preflight")
        hashes = self.driver.comm.allgather(spec.payload_hash)
        if len(set(hashes)) != 1:
            raise PythonProcessContractError(
                f"Python process payload differs across MPI ranks: {hashes}"
            )

        action = PICAMAction(
            name=spec.name,
            phase=spec.group,
            operation=spec.name,
            kind="python_process",
            native_id=None,
            enabled=spec.enabled,
        )
        local_error = None
        try:
            self.driver.step_plan.add(
                action,
                before=spec.before,
                after=spec.after,
                experimental=True,
            )
            self.installed[spec.name] = RegisteredPICAMPythonProcess(
                spec=spec,
                action_name=action.name,
                read_bindings=read_bindings,
                write_bindings=write_bindings,
                function=function,
            )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            self.installed.pop(spec.name, None)
            try:
                selected = self.driver.step_plan.select(spec.name, phase=spec.group)
            except PICAMConfigurationError:
                selected = None
            if selected is not None and selected.kind == "python_process":
                self.driver.step_plan.remove(
                    spec.name, phase=spec.group, experimental=True
                )
            message = collective_error_message(
                "Python process installation rollback", errors
            )
            raise PythonProcessContractError(
                message or "Python process installation failed"
            )
        self.driver.comm.barrier()
        return self.installed[spec.name]

    def reload(
        self,
        name: str,
        spec: PythonProcessSpec | Mapping[str, Any],
        *,
        preserve_parameters: bool = False,
        preserve_transactional: bool = False,
        unsafe: bool = False,
    ) -> dict[str, Any]:
        """Atomically replace one callback without moving its plan action."""

        if not isinstance(spec, PythonProcessSpec):
            spec = PythonProcessSpec.from_mapping(spec)
        self._require_boundary()
        key = str(name).strip().lower()
        record = self.installed.get(key)
        candidate = spec
        read_bindings: dict[str, str] = {}
        write_bindings: dict[str, str] = {}
        local_error: str | None = None
        try:
            if record is None:
                raise PythonProcessContractError(
                    f"unknown Python process {key!r}"
                )
            if candidate.name != record.spec.name:
                raise PythonProcessContractError(
                    "Python process reload cannot rename a process"
                )
            if candidate.group != record.spec.group:
                raise PythonProcessContractError(
                    "Python process reload cannot change its workflow phase; "
                    "use move() separately"
                )
            if candidate.before is not None or candidate.after is not None:
                raise PythonProcessContractError(
                    "Python process reload preserves placement; do not pass "
                    "before= or after="
                )
            action = self.driver.step_plan.select(
                record.action_name, phase=record.spec.group
            )
            candidate = replace(
                candidate,
                before=record.spec.before,
                after=record.spec.after,
                parameters=(
                    record.spec.parameters
                    if preserve_parameters
                    else candidate.parameters
                ),
                enabled=action.enabled,
                transactional=(
                    record.spec.transactional
                    if preserve_transactional
                    else candidate.transactional
                ),
            )
            if not candidate.transactional and not unsafe:
                raise PICAMConfigurationError(
                    "transactional=False requires unsafe=True"
                )
            function = cloudpickle.loads(candidate.payload)
            _validate_signature(function, candidate.parameters)
            read_bindings = {
                field: self._resolve_field(field) for field in candidate.reads
            }
            write_bindings = {
                field: self._resolve_field(field) for field in candidate.writes
            }
            overlap = set(read_bindings.values()) & set(write_bindings.values())
            if overlap:
                raise PythonProcessContractError(
                    "fields resolving to the same storage must appear only in "
                    "writes: " + ", ".join(sorted(overlap))
                )
            for exposed, resolved in write_bindings.items():
                if not self.driver.pool.contract(resolved).writable:
                    raise PythonProcessContractError(
                        f"write field {exposed!r} resolves to read-only field "
                        f"{resolved!r}"
                    )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, "Python process reload preflight")
        assert record is not None
        hashes = self.driver.comm.allgather(candidate.payload_hash)
        if len(set(hashes)) != 1:
            raise PythonProcessContractError(
                f"Python process payload differs across MPI ranks: {hashes}"
            )

        previous = (
            record.spec,
            record.read_bindings,
            record.write_bindings,
            record.function,
        )
        local_error = None
        try:
            record.spec = candidate
            record.read_bindings = read_bindings
            record.write_bindings = write_bindings
            # the materialised callable is part of the record, so a reload
            # that kept the old one would keep running the old code
            record.function = function
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            record.spec, record.read_bindings, record.write_bindings, record.function = previous
            self.driver.comm.barrier()
            message = collective_error_message(
                "Python process reload rollback", errors
            )
            raise PythonProcessContractError(
                message or "Python process reload failed"
            )
        self.driver.comm.barrier()
        return {
            "name": record.spec.name,
            "phase": record.spec.group,
            "old_payload_hash": previous[0].payload_hash,
            "payload_hash": record.spec.payload_hash,
            "reads": tuple(record.read_bindings),
            "writes": tuple(record.write_bindings),
            "enabled": bool(action.enabled),
        }

    def set_parameters(
        self, name: str, updates: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Collectively update one process's persistent keyword parameters.

        The worker deserializes the payload on every invocation, so the
        replaced spec takes effect at the process's next call with nothing
        to invalidate.
        """

        self._require_boundary()
        key = str(name).strip().lower()
        record = self.installed.get(key)
        candidate: PythonProcessSpec | None = None
        local_error: str | None = None
        try:
            if record is None:
                raise PythonProcessContractError(
                    f"unknown Python process {key!r}"
                )
            candidate = record.spec.with_parameters(updates)
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, "Python process parameter update")
        assert record is not None and candidate is not None
        hashes = self.driver.comm.allgather(
            _parameter_hash(candidate.parameters or {})
        )
        if len(set(hashes)) != 1:
            raise PythonProcessContractError(
                f"Python process parameters differ across MPI ranks: {hashes}"
            )
        record.spec = candidate
        self.driver.comm.barrier()
        return {
            "name": record.spec.name,
            "parameters": dict(record.spec.parameters or {}),
            "payload_hash": record.spec.payload_hash,
        }

    def parameters(self, name: str) -> dict[str, Any]:
        """Read one process's current persistent keyword parameters."""

        key = str(name).strip().lower()
        record = self.installed.get(key)
        if record is None:
            raise PythonProcessContractError(f"unknown Python process {key!r}")
        return dict(record.spec.parameters or {})

    def _resolve_field(self, name: str) -> str:
        """Resolve callback names without changing the name exposed to Python."""

        for candidate in PythonStateView.field_candidates(name):
            try:
                return self.driver.pool.canonical_name(candidate)
            except KeyError:
                try:
                    return self.driver.pool.ccpp_field_name(candidate)
                except KeyError:
                    continue
        raise KeyError(f"unknown PI-CAM field {name!r}")

    def remove(self, name: str) -> dict[str, Any]:
        self._require_boundary()
        key = str(name).strip().lower()
        record = self.installed.get(key)
        self._collective_error(
            None if record is not None else f"unknown Python process {key!r}",
            "Python process removal",
        )
        assert record is not None
        self.driver.step_plan.remove(
            record.action_name, phase=record.spec.group, experimental=True
        )
        self.installed.pop(key)
        self.driver.comm.barrier()
        return {"name": key, "payload_hash": record.spec.payload_hash}

    def dependencies(self, field: str) -> tuple[str, ...]:
        canonical = self.driver.pool.canonical_name(field)
        return tuple(
            name
            for name, record in self.installed.items()
            if canonical
            in {*record.read_bindings.values(), *record.write_bindings.values()}
        )

    def invoke(self, action: PICAMAction) -> None:
        try:
            record = self.installed[action.name]
        except KeyError as exc:
            raise PythonProcessContractError(
                f"Python process {action.name!r} is not installed"
            ) from exc
        snapshots = (
            {
                name: np.array(self.driver.pool[resolved], copy=True, order="F")
                for name, resolved in record.write_bindings.items()
            }
            if record.spec.transactional
            else {}
        )
        before = self.driver.pool.pointer_records()
        read_writeability = {
            resolved: bool(self.driver.pool[resolved].flags.writeable)
            for resolved in set(record.read_bindings.values())
        }
        local_error: str | None = None
        try:
            for resolved in read_writeability:
                self.driver.pool[resolved].flags.writeable = False
            fields = PythonFieldView(
                self.driver.pool,
                reads=record.read_bindings,
                writes=record.write_bindings,
            )
            context = PythonProcessContext(
                process_name=record.spec.name,
                group=record.spec.group,
                rank=self.driver.rank,
                size=self.driver.size,
                step=self.driver.clock.nstep,
                timestep_seconds=self.driver.clock.dt_seconds,
                year=self.driver.clock.year,
                month=self.driver.clock.month,
                day=self.driver.clock.day,
                seconds=self.driver.clock.seconds,
                calendar=self.driver.clock.calendar,
                native=NativeAccess(self.driver) if record.spec.native else None,
            )
            function = record.function
            if function is None:
                # a record restored without its callable (an older pickle of the
                # registry, say) materialises it once, here, and keeps it
                function = cloudpickle.loads(record.spec.payload)
                record.function = function
            result = function(fields, context, **dict(record.spec.parameters or {}))
            if result is not None:
                raise PythonProcessContractError(
                    f"Python process must return None, got {type(result).__name__}"
                )
            self.driver.pool.assert_pointer_stability(before)
        except BaseException:
            local_error = traceback.format_exc()
        finally:
            for resolved, writable in read_writeability.items():
                self.driver.pool[resolved].flags.writeable = writable
        errors = self.driver.comm.allgather(local_error)
        failure = collective_error_message(
            f"Python process {record.spec.name!r}", errors
        )
        if failure is not None:
            if record.spec.transactional:
                for exposed, values in snapshots.items():
                    np.copyto(
                        self.driver.pool[record.write_bindings[exposed]],
                        values,
                        casting="no",
                    )
                self.driver.comm.barrier()
                raise PythonProcessExecutionError(
                    f"Python process {record.spec.name!r} failed; declared writes "
                    "were restored:\n" + failure
                )
            raise PythonProcessTaintedError(
                f"Python process {record.spec.name!r} failed without rollback:\n"
                + failure
            )

    def inventory(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "spec": record.spec.as_dict(),
                "action": record.action_name,
                "read_bindings": dict(record.read_bindings),
                "write_bindings": dict(record.write_bindings),
            }
            for _, record in sorted(self.installed.items())
        )

    def _require_boundary(self) -> None:
        if self.driver.lifecycle.value not in {"initialized", "running"}:
            raise PICAMStateError(
                "runtime processes may change only on an initialized model"
            )
        if self.driver._native_call_depth:
            raise PICAMStateError("cannot change runtime processes inside a kernel")
        cursors = self.driver.comm.allgather(
            (self.driver.coupling_step, self.driver.trace_count)
        )
        if len(set(cursors)) != 1:
            raise PICAMStateError(f"MPI ranks are at different boundaries: {cursors}")

    def _collective_error(self, error: str | None, operation: str) -> None:
        errors = self.driver.comm.allgather(error)
        message = collective_error_message(operation, errors)
        if message is not None:
            raise PythonProcessContractError(message)
