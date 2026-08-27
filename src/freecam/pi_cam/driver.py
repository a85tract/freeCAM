"""Python control plane for one live PI-CAM model."""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from freecam.model.clock import ModelClock
from freecam.model.collective import collective_error_message
from freecam.model.python_processes import PythonProcessSpec

from .boundary import CAMBoundaryProvider
from .config import DEFAULT_TRACE_LIMIT, PICAMConfig, validate_trace_limit
from .errors import BoundaryReplayError, PICAMConfigurationError, PICAMStateError
from .history_output import PICAMHistoryStreamRegistry
from .native import CAMNumericalBackend
from .parameters import PICAMModuleParameterRegistry
from .physics_catalog import (
    PICAMPhysicsCatalog,
    merge_runtime_process_records,
)
from .plan import PICAMAction, PICAMStepPlan
from .process_context import (
    PICAMProcessContextRegistry,
    PICAMPromotedProcess,
)
from .runtime_fortran import (
    PICAMFortranProcessRegistry,
    PICAMFortranProcessSpec,
)
from .runtime_processes import PICAMPythonProcessRegistry
from .state import PICAMStatePool, PICAMStateSchema, PICAMVariableSpec
from .timing import CESMTimingContext, FreeCAMProfiler


class _SerialComm:
    rank = 0
    size = 1

    @staticmethod
    def allgather(value: Any) -> list[Any]:
        return [value]

    @staticmethod
    def gather(value: Any, root: int = 0) -> list[Any]:
        del root
        return [value]

    @staticmethod
    def barrier() -> None:
        return None

    @staticmethod
    def bcast(value: Any, root: int = 0) -> Any:
        del root
        return value


class PICAMLifecycle(str, Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PICAMActionTrace:
    sequence: int
    coupling_step: int
    model_step: int
    phase: str
    name: str
    operation: str
    native_id: int | None


class _ActionReference:
    def __init__(
        self,
        driver: "PICAMDriver",
        action: PICAMAction,
        record: Mapping[str, Any] | None = None,
    ) -> None:
        self.driver = driver
        self.action = action
        self._snapshot = None if record is None else dict(record)

    @property
    def name(self) -> str:
        return self.action.name

    @property
    def phase(self) -> str:
        return self.action.phase

    @property
    def operation(self) -> str:
        return self.action.operation

    @property
    def qualified_name(self) -> str:
        return self.action.qualified_name

    @property
    def granularity(self) -> str:
        return "leaf" if self.operation.startswith("leaf_") else "stage"

    @property
    def parent_stage(self) -> str | None:
        """Return the composite source stage that this leaf expands."""

        if self._snapshot is not None:
            value = self._snapshot.get("parent_stage")
            return None if value is None else str(value)
        return self.action.parent_stage

    @property
    def enabled(self) -> bool:
        return self.driver.step_plan.select(
            self.action.name, phase=self.action.phase
        ).enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.driver.step_plan.set_enabled(
            self.action.name,
            bool(value),
            phase=self.action.phase,
            experimental=True,
        )

    @property
    def runnable(self) -> bool:
        if self._snapshot is None:
            return True
        return bool(self._snapshot.get("native_available", True))

    @property
    def capability(self) -> str:
        return "runtime"

    @property
    def metadata(self) -> Mapping[str, Any]:
        if self._snapshot is not None:
            return dict(self._snapshot)
        result = {
            "name": self.name,
            "phase": self.phase,
            "operation": self.operation,
            "kind": self.action.kind,
            "native_id": self.action.native_id,
            "enabled": self.enabled,
            "runnable": True,
            "capability": "runtime",
            "parent_stage": self.parent_stage,
        }
        return result

    def run(self, *, experimental: bool = True) -> PICAMActionTrace:
        if not experimental:
            raise PICAMConfigurationError(
                "running an isolated CAM action requires experimental=True"
            )
        # A source workflow action may only be entered at its native sequence
        # cursor.  When the same operation has a generated StatePool adapter,
        # an isolated Python call must use that adapter instead of jumping into
        # the middle of CAM's timestep state machine.
        if self.operation in self.driver.kernels.names:
            return self.driver.run_kernel(self.operation, experimental=True)
        return self.driver.run_action(
            self.action.name, phase=self.action.phase, experimental=True
        )

    def disable(self, *, experimental: bool = True) -> None:
        self.driver.step_plan.set_enabled(
            self.action.name,
            False,
            phase=self.action.phase,
            experimental=experimental,
        )

    def enable(self, *, experimental: bool = True) -> None:
        self.driver.step_plan.set_enabled(
            self.action.name,
            True,
            phase=self.action.phase,
            experimental=experimental,
        )

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        experimental: bool = True,
    ) -> None:
        self.driver.step_plan.move(
            self.action.name,
            phase=self.action.phase,
            before=before,
            after=after,
            experimental=experimental,
        )

    def __repr__(self) -> str:
        return (
            f"PhysicsProcess(name={self.name!r}, operation={self.operation!r}, "
            f"enabled={self.enabled}, granularity={self.granularity!r})"
        )


class _PythonProcessReference(_ActionReference):
    def remove(self) -> Mapping[str, Any]:
        return self.driver.python_processes.remove(self.action.name)


class _FortranProcessReference(_ActionReference):
    def remove(self) -> Mapping[str, Any]:
        return self.driver.fortran_processes.remove(self.action.name)


class _LocalProcessCallResult(Mapping[str, np.ndarray]):
    """Outputs from one directly called original CAM process."""

    def __init__(
        self,
        driver: "PICAMDriver",
        record: PICAMPromotedProcess,
        trace: PICAMActionTrace,
    ) -> None:
        self.driver = driver
        self.name = record.name
        self.trace = trace
        self.bindings = record.bindings
        self._outputs = {
            binding.argument: binding.field
            for binding in record.bindings
            if binding.intent.lower() in {"out", "inout"}
        }

    @property
    def fields(self) -> Mapping[str, np.ndarray]:
        return {name: self.driver.pool[field] for name, field in self._outputs.items()}

    @property
    def process(self) -> Any:
        return self.driver.physics.process(self.name)

    def remove(self) -> PICAMPromotedProcess:
        return self.driver.remove_promoted_process(self.name)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.driver.pool[self._outputs[str(name)]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._outputs)

    def __len__(self) -> int:
        return len(self._outputs)

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        return f"ProcessResult(name={self.name!r}, outputs={tuple(self)!r})"


class _CatalogActionReference:
    def __init__(
        self,
        driver: "PICAMDriver",
        record: Mapping[str, Any],
    ) -> None:
        self.driver = driver
        self._record = dict(record)

    @property
    def name(self) -> str:
        return str(self._record["api_name"])

    @property
    def phase(self) -> str:
        return str(self._record["phase"])

    @property
    def operation(self) -> str:
        return str(self._record["operation"])

    @property
    def qualified_name(self) -> str:
        return str(self._record["qualified_name"])

    @property
    def source(self) -> str:
        return str(self._record["source"])

    @property
    def level(self) -> str:
        return str(self._record["level"])

    @property
    def granularity(self) -> str:
        return self.level

    @property
    def enabled(self) -> None:
        return None

    @property
    def runnable(self) -> bool:
        return bool(self._record.get("runnable", False))

    @property
    def capability(self) -> str:
        return str(self._record["capability"])

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(str(item) for item in self._record.get("blockers", ()))

    @property
    def parent_processes(self) -> tuple[str, ...]:
        return tuple(
            str(item) for item in self._record.get("parent_processes", ())
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return dict(self._record)

    def _unavailable(self, operation: str) -> None:
        reason = ", ".join(self.blockers) or self.capability
        parents = ", ".join(self.parent_processes) or "its enclosing CAM process"
        raise PICAMConfigurationError(
            f"{self.name!r} is a cataloged {self.level}, not an independently "
            f"runnable boundary; cannot {operation}. Run {parents}, or first "
            f"admit explicit StatePool/context bindings. Current blockers: {reason}"
        )

    def run(self, *, experimental: bool = True) -> PICAMActionTrace:
        del experimental
        if not self.runnable:
            self._unavailable("run it")
        return self().trace

    def __call__(self, **arguments: Any) -> _LocalProcessCallResult:
        """Bind inferred StatePool inputs, run once, and return named outputs."""

        bindings, initials = self.driver._process_call_arguments(arguments)
        if self.name not in self.driver.process_contexts:
            self.driver.promote_process(
                self.name,
                bindings=bindings,
                initials=initials,
            )
        elif arguments:
            raise PICAMConfigurationError(
                f"{self.name!r} is already bound; remove its previous result "
                "before changing call arguments"
            )
        record = self.driver.process_contexts.record(self.name)
        trace = self.driver.run_promoted_process(self.name, experimental=True)
        return _LocalProcessCallResult(self.driver, record, trace)

    def bind(self, **arguments: Any) -> "_PromotedProcessReference":
        """Bind normal Python values/StatePool fields without running the process."""

        bindings, initials = self.driver._process_call_arguments(arguments)
        return self.promote(bindings=bindings, initials=initials)

    def promote(
        self,
        *,
        bindings: Mapping[str, str] | None = None,
        initials: Mapping[str, Any] | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> "_PromotedProcessReference":
        """Make the routine's caller arguments explicit rank-local state."""

        self.driver.promote_process(
            self.name,
            bindings=bindings,
            initials=initials,
            dimensions=dimensions,
        )
        reference = self.driver.physics.process(self.name)
        assert isinstance(reference, _PromotedProcessReference)
        return reference

    def enable(self, *, experimental: bool = True) -> None:
        del experimental
        self._unavailable("enable it")

    def disable(self, *, experimental: bool = True) -> None:
        del experimental
        self._unavailable("disable it")

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        experimental: bool = True,
    ) -> None:
        del before, after, experimental
        self._unavailable("move it")

    def __repr__(self) -> str:
        return (
            f"PhysicsProcess(name={self.name!r}, source={self.qualified_name!r}, "
            f"level={self.level!r}, runnable=False, capability={self.capability!r})"
        )


class _PromotedProcessReference(_CatalogActionReference):
    """One catalog routine whose complete argument list now lives in StatePool."""

    @property
    def runnable(self) -> bool:
        return bool(self._record.get("native_available"))

    @property
    def capability(self) -> str:
        if self._runtime_action() is not None:
            return "runtime"
        return (
            "statepool_bound"
            if self.runnable
            else "statepool_bound_no_native"
        )

    @property
    def enabled(self) -> bool | None:
        action = self._runtime_action()
        return None if action is None else action.enabled

    @property
    def bindings(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._record.get("bindings", ()))

    @property
    def fields(self) -> Mapping[str, np.ndarray]:
        return {
            str(binding["argument"]): self.driver.pool[str(binding["field"])]
            for binding in self.bindings
            if str(binding.get("intent", "inout")).lower() in {"out", "inout"}
        }

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[str(name)]

    def __getattr__(self, name: str) -> np.ndarray:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def run(self, *, experimental: bool = True) -> PICAMActionTrace:
        if not experimental:
            raise PICAMConfigurationError(
                "isolated promoted CAM processes require experimental=True"
            )
        return self.driver.run_promoted_process(self.name, experimental=True)

    def __call__(self, **arguments: Any) -> _LocalProcessCallResult:
        if arguments:
            raise PICAMConfigurationError(
                f"{self.name!r} is already bound; call it without arguments or "
                "remove it before rebinding"
            )
        record = self.driver.process_contexts.record(self.name)
        trace = self.run(experimental=True)
        return _LocalProcessCallResult(self.driver, record, trace)

    def insert(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        enabled: bool = True,
    ) -> "_PromotedProcessReference":
        self.driver.install_promoted_process(
            self.name,
            before=before,
            after=after,
            enabled=enabled,
        )
        return self.driver.physics.process(self.name)

    def remove(self) -> PICAMPromotedProcess:
        return self.driver.remove_promoted_process(self.name)

    def enable(self, *, experimental: bool = True) -> None:
        action = self._require_runtime_action()
        self.driver.step_plan.set_enabled(
            action.name,
            True,
            phase=action.phase,
            experimental=experimental,
        )

    def disable(self, *, experimental: bool = True) -> None:
        action = self._require_runtime_action()
        self.driver.step_plan.set_enabled(
            action.name,
            False,
            phase=action.phase,
            experimental=experimental,
        )

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        experimental: bool = True,
    ) -> None:
        action = self._require_runtime_action()
        self.driver.step_plan.move(
            action.name,
            phase=action.phase,
            before=before,
            after=after,
            experimental=experimental,
        )

    def _runtime_action(self) -> PICAMAction | None:
        try:
            action = self.driver.step_plan.select(self.name)
        except PICAMConfigurationError:
            return None
        return action if action.kind == "runtime_catalog_process" else None

    def _require_runtime_action(self) -> PICAMAction:
        action = self._runtime_action()
        if action is None:
            raise PICAMConfigurationError(
                f"{self.name!r} is bound but not in the complete workflow; "
                "insert it with driver.cam.workflow.insert(process, before=... "
                "or after=...)"
            )
        return action


class _PhysicsCollection:
    def __init__(self, driver: "PICAMDriver") -> None:
        self.driver = driver
        self.catalog = PICAMPhysicsCatalog.load_default()

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        runtime = tuple(
            row
            for row in self.driver.step_plan.describe()
            if row["kind"] in {
                "scheme",
                "python_process",
                "runtime_fortran_process",
            }
        )
        records = [dict(row) for row in merge_runtime_process_records(runtime, self.catalog)]
        has_adapter = getattr(self.driver.backend, "has_process_adapter", None)
        for row in records:
            if row["kind"] == "catalog_process" and callable(has_adapter):
                available = bool(
                    has_adapter(str(row["qualified_name"]), str(row["source"]))
                )
                row.update(
                    {
                        "runnable": available,
                        "capability": (
                            "compiled_process_device"
                            if available
                            else row["capability"]
                        ),
                    }
                )
            if str(row["api_name"]) not in self.driver.process_contexts:
                continue
            promoted = self.driver.process_contexts.record(str(row["api_name"]))
            try:
                runtime_action = self.driver.step_plan.select(promoted.name)
            except PICAMConfigurationError:
                runtime_action = None
            if (
                runtime_action is not None
                and runtime_action.kind != "runtime_catalog_process"
            ):
                runtime_action = None
            row.update(
                {
                    "phase": (
                        "promoted_process"
                        if runtime_action is None
                        else runtime_action.phase
                    ),
                    "kind": (
                        "promoted_process"
                        if runtime_action is None
                        else "runtime_catalog_process"
                    ),
                    "runnable": promoted.native_available,
                    "enabled": (
                        None if runtime_action is None else runtime_action.enabled
                    ),
                    "capability": (
                        "runtime"
                        if runtime_action is not None
                        else "statepool_bound"
                        if promoted.native_available
                        else "statepool_bound_no_native"
                    ),
                    "bindings": tuple(
                        binding.to_payload() for binding in promoted.bindings
                    ),
                    "created_fields": promoted.created_fields,
                    "native_available": promoted.native_available,
                    "requires_binding": False,
                }
            )
        return tuple(records)

    @property
    def names(self) -> tuple[str, ...]:
        """Flat Python names for every case-reachable physics interface."""

        return tuple(str(row["api_name"]) for row in self.records)

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(str(row["operation"]) for row in self.records)

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(
            str(row["name"])
            for row in self.records
            if bool(row["runnable"])
        )

    @property
    def interfaces(self) -> tuple[Any, ...]:
        references: list[Any] = []
        for record in self.records:
            if record["kind"] in {
                "promoted_process",
                "runtime_catalog_process",
            }:
                references.append(_PromotedProcessReference(self.driver, record))
                continue
            if record["kind"] == "catalog_process":
                references.append(_CatalogActionReference(self.driver, record))
                continue
            action = self.driver.step_plan.select(
                str(record["name"]), phase=str(record["phase"])
            )
            if action.kind == "python_process":
                references.append(_PythonProcessReference(self.driver, action, record))
            elif action.kind == "runtime_fortran_process":
                references.append(_FortranProcessReference(self.driver, action, record))
            else:
                references.append(_ActionReference(self.driver, action, record))
        return tuple(references)

    @property
    def runnable(self) -> tuple[Any, ...]:
        return tuple(reference for reference in self.interfaces if reference.runnable)

    @property
    def catalog_only(self) -> tuple[Any, ...]:
        return tuple(
            reference for reference in self.interfaces if not reference.runnable
        )

    @property
    def coverage(self) -> Mapping[str, int]:
        records = self.records
        has_adapter = getattr(self.driver.backend, "has_process_adapter", None)
        source_only = tuple(
            row
            for row in merge_runtime_process_records(
                tuple(
                    row
                    for row in self.driver.step_plan.describe()
                    if row["kind"]
                    in {"scheme", "python_process", "runtime_fortran_process"}
                ),
                self.catalog,
            )
            if row["kind"] == "catalog_process"
        )
        runnable = tuple(row for row in records if bool(row["runnable"]))
        catalog_only = tuple(row for row in records if not bool(row["runnable"]))
        leaf = sum(
            str(row.get("granularity")) == "leaf"
            or str(row["operation"]).startswith("leaf_")
            for row in runnable
        )
        planned = tuple(row for row in runnable if row.get("enabled") is not None)
        enabled = sum(bool(row["enabled"]) for row in planned)
        result = {
            "interfaces": len(records),
            "runnable": len(runnable),
            "catalog_only": len(catalog_only),
            "source_reachable": self.catalog.reachable_procedures,
            "source_catalog": len(self.catalog.processes),
            "physical_processes": len(self.catalog.physics_processes),
            "compiled_process_adapters": sum(
                process.generated_adapter
                for process in self.catalog.physics_processes
            ),
            "formerly_catalog_only_interfaces": len(source_only),
            "catalog_adapters_compiled": sum(
                bool(row.get("generated_adapter")) for row in source_only
            ),
            "catalog_current_case_loadable": sum(
                bool(
                    callable(has_adapter)
                    and has_adapter(str(row["qualified_name"]), str(row["source"]))
                )
                for row in source_only
            ),
            "runtime_templates": sum(
                bool(row.get("generated_adapter")) for row in source_only
            ),
            "runtime_templates_loadable": sum(
                bool(
                    callable(has_adapter)
                    and has_adapter(str(row["qualified_name"]), str(row["source"]))
                )
                for row in source_only
            ),
            "runtime_bound": sum(
                str(row["kind"])
                in {"promoted_process", "runtime_catalog_process"}
                for row in records
            ),
            "runtime_inserted": sum(
                str(row["kind"]) == "runtime_catalog_process" for row in records
            ),
            "current_case_loadable": sum(
                bool(callable(has_adapter) and has_adapter(process.qualified_name, process.source))
                for process in self.catalog.physics_processes
            ),
            "helper_routines": len(self.catalog.helpers),
            "runtime_overlap": len(self.catalog.physics_processes) - len(source_only),
            "excluded_lifecycle": self.catalog.excluded_lifecycle,
            "enabled": enabled,
            "disabled": len(planned) - enabled,
            "leaf": leaf,
            "stage": len(runnable) - leaf,
        }
        result["configuration_specific"] = (
            result["physical_processes"] - result["current_case_loadable"]
        )
        return result

    def __len__(self) -> int:
        return len(self.interfaces)

    def __iter__(self) -> Iterator[_ActionReference]:
        return iter(self.interfaces)

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__())
            | {
                name
                for name in (*self.names, *self.operation_names, *self.action_names)
                if name.isidentifier()
            }
        )

    def __getattr__(self, name: str) -> _ActionReference:
        matches = [
            record
            for record in self.records
            if record["name"] == name
            or record["api_name"] == name
            or record["operation"] == name
            or name in record.get("aliases", ())
            or record.get("qualified_name") == name
        ]
        if len(matches) != 1:
            raise AttributeError(name)
        record = matches[0]
        if record["kind"] in {"promoted_process", "runtime_catalog_process"}:
            return _PromotedProcessReference(self.driver, record)
        if record["kind"] == "catalog_process":
            return _CatalogActionReference(self.driver, record)
        action = self.driver.step_plan.select(
            str(record["name"]), phase=str(record["phase"])
        )
        if action.kind == "python_process":
            return _PythonProcessReference(self.driver, action, record)
        if action.kind == "runtime_fortran_process":
            return _FortranProcessReference(self.driver, action, record)
        return _ActionReference(self.driver, action, record)

    def scheme(self, name: str) -> Any:
        matches = tuple(
            record
            for record in self.records
            if (
                record["name"] == name
                or record["api_name"] == name
                or record["operation"] == name
                or name in record.get("aliases", ())
                or record.get("qualified_name") == name
            )
        )
        if len(matches) != 1:
            raise PICAMConfigurationError(
                f"physics process {name!r} is unknown or ambiguous"
            )
        record = matches[0]
        if record["kind"] in {"promoted_process", "runtime_catalog_process"}:
            return _PromotedProcessReference(self.driver, record)
        if record["kind"] == "catalog_process":
            return _CatalogActionReference(self.driver, record)
        return getattr(self, str(record["api_name"]))

    def process(self, name: str) -> Any:
        """Resolve one flat physics process; ``scheme`` remains an alias."""

        return self.scheme(name)

    def install_python(
        self,
        function: Callable[..., None],
        *,
        name: str,
        before: str | None = None,
        after: str | None = None,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        enabled: bool = True,
        transactional: bool = True,
        unsafe: bool = False,
    ) -> _PythonProcessReference:
        phase, before, after = self._resolve_placement(
            phase=None, before=before, after=after
        )
        spec = PythonProcessSpec.from_callable(
            function,
            name=name,
            group=phase,
            before=before,
            after=after,
            reads=reads,
            writes=writes,
            parameters=parameters,
            enabled=enabled,
            transactional=transactional,
        )
        self.driver.python_processes.install(spec, unsafe=unsafe)
        action = self.driver.step_plan.select(spec.name, phase=spec.group)
        return _PythonProcessReference(self.driver, action)

    def remove_python(self, name: str) -> Mapping[str, Any]:
        return self.driver.python_processes.remove(name)

    def install_fortran(
        self,
        source: str | Path,
        *,
        process: str,
        before: str | None = None,
        after: str | None = None,
        project_root: str | Path | None = None,
        enabled: bool = True,
        unsafe: bool = False,
    ) -> _FortranProcessReference:
        phase, before, after = self._resolve_placement(
            phase=None, before=before, after=after
        )
        spec = PICAMFortranProcessSpec(
            source=str(source),
            process=process,
            phase=phase,
            before=before,
            after=after,
            project_root=None if project_root is None else str(project_root),
            enabled=enabled,
        )
        self.driver.fortran_processes.install(spec, unsafe=unsafe)
        action = self.driver.step_plan.select(process, phase=phase)
        return _FortranProcessReference(self.driver, action)

    def remove_fortran(self, name: str) -> Mapping[str, Any]:
        return self.driver.fortran_processes.remove(name)

    def _resolve_placement(
        self,
        *,
        phase: str | None,
        before: str | None,
        after: str | None,
    ) -> tuple[str, str | None, str | None]:
        """Translate a Pythonic neighbor placement into the internal plan key."""

        if before is None and after is None:
            before = "state_export"
        if before is not None and after is not None:
            raise PICAMConfigurationError("provide only one of before= or after=")
        target = self.driver.step_plan.select(before or after or "")
        if phase is not None and str(phase) != target.phase:
            raise PICAMConfigurationError(
                f"placement target {target.name!r} belongs to a different "
                "internal source region"
            )
        return target.phase, before, after


class _PhaseReference:
    def __init__(self, driver: "PICAMDriver", name: str) -> None:
        self.driver = driver
        self.name = name

    def run(self, *, experimental: bool = True) -> tuple[PICAMActionTrace, ...]:
        return self.driver.run_phase(self.name, experimental=experimental)

    def expand(self, *, experimental: bool = True) -> tuple[PICAMAction, ...]:
        expanders = {
            "cam_run1": self.driver.expand_cam_run1_leaves,
            "cam_run2": self.driver.expand_cam_run2_leaves,
            "cam_run4": self.driver.expand_cam_run4_leaves,
        }
        if self.name not in expanders:
            raise PICAMConfigurationError(
                f"phase {self.name!r} has no finer validated expansion"
            )
        return expanders[self.name](experimental=experimental)


class _PhaseCollection:
    def __init__(self, driver: "PICAMDriver") -> None:
        self.driver = driver

    def __getattr__(self, name: str) -> _PhaseReference:
        if name not in self.driver.step_plan.phases:
            raise AttributeError(name)
        return _PhaseReference(self.driver, name)

    def __getitem__(self, name: str) -> _PhaseReference:
        if name not in self.driver.step_plan.phases:
            raise KeyError(name)
        return _PhaseReference(self.driver, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.driver.step_plan.phases))


class _KernelReference:
    def __init__(self, driver: "PICAMDriver", name: str) -> None:
        self.driver = driver
        self.name = name

    def run(self, *, experimental: bool = False) -> PICAMActionTrace:
        return self.driver.run_kernel(self.name, experimental=experimental)


class _KernelCollection:
    """Original leaf routines with generated raw-array adapters."""

    def __init__(self, driver: "PICAMDriver") -> None:
        self.driver = driver

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(getattr(self.driver.backend, "direct_kernels", ()))

    def __getattr__(self, name: str) -> _KernelReference:
        if name not in self.names:
            raise AttributeError(name)
        return _KernelReference(self.driver, name)

    def __getitem__(self, name: str) -> _KernelReference:
        if name not in self.names:
            raise KeyError(name)
        return _KernelReference(self.driver, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.names))


class PICAMDriver:
    """Own CAM orchestration while delegating leaf numerics to native devices."""

    def __init__(
        self,
        config: PICAMConfig,
        boundary: CAMBoundaryProvider,
        backend: CAMNumericalBackend,
        *,
        rank: int,
        size: int,
        fcomm: int = 0,
        communicator: Any | None = None,
        step_plan: PICAMStepPlan | None = None,
        state_schema: PICAMStateSchema | None = None,
        history_callback: Callable[[PICAMAction, "PICAMDriver"], None] | None = None,
        run_dir: str | Path | None = None,
        trace_limit: int | None = DEFAULT_TRACE_LIMIT,
        default_history_stream: bool = True,
    ) -> None:
        if not 0 <= rank < size:
            raise PICAMConfigurationError("rank must be in the MPI communicator")
        self.config = config
        self.boundary = boundary
        self.backend = backend
        self.rank = int(rank)
        self.size = int(size)
        self.fcomm = int(fcomm)
        self.comm = communicator if communicator is not None else _SerialComm()
        if int(self.comm.rank) != self.rank or int(self.comm.size) != self.size:
            raise PICAMConfigurationError(
                "communicator rank/size do not match PICAMDriver rank/size"
            )
        self.profiler = FreeCAMProfiler(rank=self.rank, size=self.size)
        self.step_plan = step_plan or PICAMStepPlan.default()
        self.pool = (state_schema or PICAMStateSchema.core()).allocate(
            {
                "pver": config.pver,
                "pcols": config.pcols,
                "mpi_rank": rank,
                "mpi_size": size,
                "case_name_length": 256,
            }
        )
        year, month, day = config.start_parts
        self.clock = ModelClock(
            year=year,
            month=month,
            day=day,
            seconds=config.start_seconds,
            dt_seconds=config.timestep_seconds,
            calendar=config.calendar,
        )
        self.lifecycle = PICAMLifecycle.CREATED
        self.coupling_step = 0
        self._boundary_step = 0
        self._native_step = 0
        self._trace: deque[PICAMActionTrace] = deque(
            maxlen=validate_trace_limit(trace_limit)
        )
        self._trace_count = 0
        self._operation_counts: Counter[str] = Counter()
        self._trace_captures: list[list[PICAMActionTrace]] = []
        self.history_callback = history_callback
        self.history_every: int | None = 1
        self.restart_every: int | None = None
        self.restart_at_end = True
        self.run_dir = None if run_dir is None else Path(run_dir).resolve()
        self._previous_directory: Path | None = None
        self._advance_public_clock = True
        self._backend_initialized = False
        self._native_call_depth = 0
        self._python_initialized_addresses: dict[str, int] = {}
        self.python_processes = PICAMPythonProcessRegistry(self)
        self.module_parameters = PICAMModuleParameterRegistry(self)
        self.history_streams = PICAMHistoryStreamRegistry(self)
        self.default_history_stream = bool(default_history_stream)
        self.fortran_processes = PICAMFortranProcessRegistry(self)
        self.process_contexts = PICAMProcessContextRegistry(self)
        self.physics = _PhysicsCollection(self)
        self.phases = _PhaseCollection(self)
        self.kernels = _KernelCollection(self)

    @property
    def trace(self) -> tuple[PICAMActionTrace, ...]:
        """Retained tail of the action trace, newest ``trace_limit`` records.

        ``trace_count`` reports the exact lifetime total; construct the driver
        with ``trace_limit=None`` when a complete in-memory trace is needed.
        """

        return tuple(self._trace)

    @property
    def trace_limit(self) -> int | None:
        """Retention limit of the in-memory trace; ``None`` means unbounded."""

        return self._trace.maxlen

    @property
    def trace_count(self) -> int:
        """Exact number of actions ever recorded, immune to truncation."""

        return self._trace_count

    @property
    def trace_retained(self) -> int:
        """Number of records currently retained in memory."""

        return len(self._trace)

    @property
    def trace_first_sequence(self) -> int:
        """Sequence number of the oldest retained record."""

        return self._trace_count - len(self._trace)

    @property
    def operation_counts(self) -> Counter[str]:
        """Cumulative per-operation execution counts; missing keys read zero."""

        return Counter(self._operation_counts)

    def trace_since(self, since: int = 0) -> tuple[PICAMActionTrace, ...]:
        """Return retained records with ``sequence >= since``.

        Records evicted by ``trace_limit`` cannot be returned; callers detect
        the gap by comparing the first returned sequence against the cursor.
        """

        cursor = int(since)
        if not 0 <= cursor <= self._trace_count:
            raise ValueError(f"trace cursor must be in 0..{self._trace_count}")
        start = max(0, cursor - self.trace_first_sequence)
        return tuple(islice(self._trace, start, None))

    @property
    def native_step(self) -> int:
        """Python's mirror of CAM's internal time-manager step."""

        return self._native_step

    @property
    def python_initialized_addresses(self) -> dict[str, int]:
        """Addresses captured after Python filled state and before native finish."""

        return dict(self._python_initialized_addresses)

    def _set_scalar(self, name: str, value: int | float) -> None:
        self.pool[name][()] = value

    def _sync_clock_fields(self) -> None:
        self._set_scalar("model_step", self.clock.nstep)
        self._set_scalar("current_date", self.clock.yyyymmdd)
        self._set_scalar("current_seconds_of_day", self.clock.seconds)

    def initialize(self) -> None:
        if self.lifecycle != PICAMLifecycle.CREATED:
            raise PICAMStateError(f"initialize from {self.lifecycle.value}")
        self.profiler.start_total()
        with self.profiler.region("FREECAM:INITIALIZE"):
            self._initialize()

    def _initialize(self) -> None:
        if self.lifecycle != PICAMLifecycle.CREATED:
            raise PICAMStateError(f"initialize from {self.lifecycle.value}")
        if self.size != self.config.mpi_size:
            raise PICAMConfigurationError(
                f"PI-CAM config requires {self.config.mpi_size} ranks, got {self.size}"
            )
        try:
            if self.run_dir is not None:
                if not (self.run_dir / "atm_in").is_file():
                    raise PICAMConfigurationError(
                        f"PI-CAM run directory lacks atm_in: {self.run_dir}"
                    )
                self._previous_directory = Path.cwd()
                os.chdir(self.run_dir)
            self._set_scalar("model_timestep", self.config.timestep_seconds)
            self._set_scalar("configured_stop_n", self.config.stop_n)
            encoded_case_name = self.config.case_name.encode("utf-8")
            case_name = self.pool["case_name_utf8"]
            if len(encoded_case_name) >= case_name.size:
                raise PICAMConfigurationError(
                    f"case_name must contain fewer than {case_name.size} UTF-8 bytes"
                )
            case_name[...] = 0
            case_name[: len(encoded_case_name)] = np.frombuffer(
                encoded_case_name, dtype=np.uint8
            )
            self._set_scalar("orbital_year", self.config.orbital_year)
            self._set_scalar("mpi_rank", self.rank)
            self._set_scalar("mpi_size", self.size)
            self._sync_clock_fields()
            shared_cam = bool(self.boundary.shares_cam_instance)
            if not shared_cam:
                self._collective_boundary_call(
                    "boundary initialization",
                    lambda: self.boundary.initialize(
                        rank=self.rank,
                        size=self.size,
                        config_fingerprint=self.config.fingerprint,
                    ),
                )
            # The source initial-run lifecycle invokes atm_init_mct twice.  Its
            # second call imports the first surface state and executes CAM run1
            # before the first normal run2/run3/run4 timestep.  Keep that
            # orchestration in Python so native CAM never observes an
            # unprimed phys_state.
            if not shared_cam:
                self._collective_boundary_call(
                    "initial boundary import",
                    lambda: self.boundary.import_fields(0, self.rank, self.pool),
                )
            configure_shared_coupler = getattr(
                self.backend, "configure_shared_coupler", None
            )
            if callable(configure_shared_coupler):
                configure_shared_coupler(shared_cam)
            prepare_initialize = getattr(self.backend, "prepare_initialize", None)
            if callable(prepare_initialize):
                prepare_initialize(self.pool, fcomm=self.fcomm)
            prepare_state = getattr(self.backend, "prepare_state", None)
            if callable(prepare_state):
                prepare_state(
                    self.pool,
                    self.config,
                    rank=self.rank,
                    size=self.size,
                )
            self._python_initialized_addresses = {
                name: int(values.ctypes.data) for name, values in self.pool.items()
            }
            with self.profiler.region("FORTRAN:CAM_INITIALIZE"):
                self.backend.initialize(self.pool, fcomm=self.fcomm)
            changed = tuple(
                name
                for name, address in self._python_initialized_addresses.items()
                if int(self.pool[name].ctypes.data) != address
            )
            if changed:
                raise PICAMStateError(
                    "native initialization changed Python StatePool addresses: "
                    + ", ".join(changed[:8])
                )
            self._backend_initialized = True
            if shared_cam:
                initialize_before_cam = getattr(
                    self.boundary, "initialize_before_cam"
                )
                self._collective_boundary_call(
                    "coupler initialization before attachment",
                    lambda: initialize_before_cam(
                        rank=self.rank,
                        size=self.size,
                        config_fingerprint=self.config.fingerprint,
                    ),
                )
                initialize_after_cam = getattr(
                    self.boundary, "initialize_after_cam"
                )
                self._collective_boundary_call(
                    "coupler attachment after CAM",
                    initialize_after_cam,
                )
                self._collective_boundary_call(
                    "initial boundary import",
                    lambda: self.boundary.import_fields(0, self.rank, self.pool),
                )
            # Native initialization has read atm_in into module storage;
            # only now do the audited tunables hold values to verify
            # against, so this is the earliest safe binding point.
            self.module_parameters.bind()
            self._validate_initial_cam_export()
            self._prime_initial_cam_state()
            self._install_default_history_stream()
        except Exception:
            self.lifecycle = PICAMLifecycle.FAILED
            raise
        self.lifecycle = PICAMLifecycle.INITIALIZED
        # iCESM CAM uses a split timestep: startup performs one complete
        # control pass to prepare run1 state for the first public coupling
        # step.  Native CAM time advances, while the Python completed-step
        # clock intentionally remains at the initial date.
        for _ in range(self.config.initialization_lookahead_steps):
            self._advance_public_clock = False
            try:
                self.step()
            finally:
                self._advance_public_clock = True

    def _prime_initial_cam_state(self) -> None:
        """Reproduce the initial-only ``atm_import -> cam_run1 -> atm_export``."""

        boundary_import = self.step_plan.select("boundary_import")
        boundary_export = self.step_plan.select("boundary_export")
        initial_priming = PICAMAction(
            "initial_priming",
            "initialization",
            "initial_priming",
            "kernel",
            200,
        )
        if self.boundary.shares_cam_instance:
            self._collective_boundary_call(
                "begin coupled initial priming",
                getattr(self.boundary, "begin_initial_priming"),
            )
        self._collective_boundary_call(
            "priming boundary import",
            lambda: self.boundary.import_fields(1, self.rank, self.pool),
        )
        # The source second atm_init_mct call performs import, cam_run1 and
        # export in one Fortran stack frame.  ``initial_priming`` preserves
        # that startup-only numerical boundary while the trace still exposes
        # the three source actions to Python.
        self._record(boundary_import)
        with self.profiler.region("FORTRAN:INITIAL_PRIMING"):
            self.backend.execute(initial_priming, self.pool, fcomm=self.fcomm)
        self._record(initial_priming)
        self._record(boundary_export)
        self._collective_boundary_call(
            "priming boundary export",
            lambda: self.boundary.export_fields(1, self.rank, self.pool),
        )
        if self.boundary.shares_cam_instance:
            self._collective_boundary_call(
                "finish coupled initial priming",
                getattr(self.boundary, "finish_initial_priming"),
            )
        self._boundary_step = 2

    def _validate_initial_cam_export(self) -> None:
        """Reproduce the first atm_init_mct export immediately after cam_init."""

        boundary_export = self.step_plan.select("boundary_export")
        self._execute_boundary_kernel(boundary_export)
        self._record(boundary_export)
        self._collective_boundary_call(
            "initial boundary export",
            lambda: self.boundary.export_fields(0, self.rank, self.pool),
        )

    def _record(self, action: PICAMAction) -> PICAMActionTrace:
        trace = PICAMActionTrace(
            sequence=self._trace_count,
            coupling_step=self.coupling_step,
            model_step=self.clock.nstep,
            phase=action.phase,
            name=action.name,
            operation=action.operation,
            native_id=action.native_id,
        )
        self._trace_count += 1
        self._trace.append(trace)
        self._operation_counts[trace.operation] += 1
        for frame in self._trace_captures:
            frame.append(trace)
        return trace

    def _execute(self, action: PICAMAction) -> PICAMActionTrace:
        with self.profiler.region(f"CAM:{action.operation}"):
            return self._execute_action(action)

    def _execute_action(self, action: PICAMAction) -> PICAMActionTrace:
        if action.kind == "python_process":
            self.python_processes.invoke(action)
        elif action.kind == "runtime_fortran_process":
            self.fortran_processes.invoke(action)
        elif action.kind == "runtime_catalog_process":
            self.process_contexts.invoke(action.name)
        elif action.operation == "boundary_import":
            self._collective_boundary_call(
                f"step {self._boundary_step} boundary import",
                lambda: self.boundary.import_fields(
                    self._boundary_step, self.rank, self.pool
                ),
            )
            # A coupled atmosphere interval can contain more than one CAM
            # substep.  CESM calls atm_import only at the beginning of that
            # interval and holds cam_in for the remaining substeps.  The
            # replay manifest records those held boundaries explicitly.
            if self._collective_boundary_call(
                f"step {self._boundary_step} boundary import schedule",
                lambda: self.boundary.has_fresh_import(
                    self._boundary_step, self.rank
                ),
            ):
                self._execute_boundary_kernel(action)
        elif action.operation == "boundary_export":
            self._execute_boundary_kernel(action)
            self._collective_boundary_call(
                f"step {self._boundary_step} boundary export",
                lambda: self.boundary.export_fields(
                    self._boundary_step, self.rank, self.pool
                ),
            )
        elif action.operation == "advance_timestep":
            clock_state = (
                self.clock.year,
                self.clock.month,
                self.clock.day,
                self.clock.seconds,
                self.clock.nstep,
            )
            if self._advance_public_clock:
                self.clock.advance()
                self._sync_clock_fields()
            self._native_step += 1
            try:
                self._synchronize_clock(action)
            except BaseException:
                self._native_step -= 1
                (
                    self.clock.year,
                    self.clock.month,
                    self.clock.day,
                    self.clock.seconds,
                    self.clock.nstep,
                ) = clock_state
                self._sync_clock_fields()
                raise
        elif action.kind == "io":
            if self.history_callback is not None:
                self.history_callback(action, self)
            elif action.operation == "wshist" and self._history_due():
                self._execute_io_service(action)
            elif action.operation == "restart" and self._restart_due():
                self._execute_io_service(action)
            elif action.operation not in {"wshist", "restart"}:
                self._execute_io_service(action)
        elif action.kind == "python_history":
            self._execute_python_history(action)
        elif action.kind == "service":
            self._execute_state_service(action)
        else:
            self._execute_native(action)
        return self._record(action)

    def _execute_python_history(self, action: PICAMAction) -> None:
        """Extend CAM history output at this stream's workflow position."""

        with self.profiler.region(f"HISTORY:{action.operation}"):
            self.history_streams.step(action.operation)

    def _restart_due(self) -> bool:
        """Return the Python-owned restart alarm for the admitted nstep case."""

        if self.restart_every is not None:
            return (self._native_step + 1) % self.restart_every == 0
        return self.restart_at_end and self._native_step >= self.config.stop_n

    def _history_due(self) -> bool:
        """Return whether the current output boundary should write history."""

        return self.history_every is not None and (
            (self._native_step + 1) % self.history_every == 0
        )

    def configure_output(
        self,
        *,
        history_every: int | None = 1,
        restart_every: int | None = None,
        restart_at_end: bool = True,
    ) -> None:
        """Configure Python-owned history and restart alarms in model steps."""

        for name, value in (
            ("history_every", history_every),
            ("restart_every", restart_every),
        ):
            if value is not None and (
                isinstance(value, bool) or int(value) < 1
            ):
                raise PICAMConfigurationError(
                    f"{name} must be a positive integer or None"
                )
        self.history_every = (
            None if history_every is None else int(history_every)
        )
        self.restart_every = (
            None if restart_every is None else int(restart_every)
        )
        self.restart_at_end = bool(restart_at_end)

    def _execute_backend_primitive(
        self, method: str, action: PICAMAction
    ) -> None:
        primitive = getattr(self.backend, method, None)
        if not callable(primitive):
            raise PICAMStateError(
                f"backend does not implement required {method} primitive"
            )
        self._native_call_depth += 1
        try:
            with self.profiler.region(f"FORTRAN:{method.upper()}"):
                primitive(action, self.pool, fcomm=self.fcomm)
        finally:
            self._native_call_depth -= 1

    def _execute_boundary_kernel(self, action: PICAMAction) -> None:
        self._execute_backend_primitive("execute_boundary_kernel", action)

    def _execute_io_service(self, action: PICAMAction) -> None:
        self._execute_backend_primitive("execute_io_service", action)

    def _execute_state_service(self, action: PICAMAction) -> None:
        self._execute_backend_primitive("execute_state_service", action)

    def _synchronize_clock(self, action: PICAMAction) -> None:
        self._execute_backend_primitive("synchronize_clock", action)

    def _execute_native(self, action: PICAMAction) -> None:
        if action.kind not in {
            "scheme",
            "coupling",
            "dynamics",
            "kernel",
            "runtime_fortran_process",
        }:
            raise PICAMStateError(
                f"Python control cannot send {action.kind!r} action "
                f"{action.operation!r} through generic native execution"
            )
        self._native_call_depth += 1
        try:
            with self.profiler.region("FORTRAN:KERNEL"):
                self.backend.execute(action, self.pool, fcomm=self.fcomm)
        finally:
            self._native_call_depth -= 1

    def define_variable(
        self, spec: PICAMVariableSpec | Mapping[str, Any]
    ) -> np.ndarray:
        """Collectively allocate one rank-local Python-owned StatePool field."""

        if not isinstance(spec, PICAMVariableSpec):
            spec = PICAMVariableSpec.from_payload(spec)
        self._require_collective_boundary("define a variable")
        payload = json.dumps(spec.to_payload(), sort_keys=True, separators=(",", ":"))
        payloads = self.comm.allgather(payload)
        if len(set(payloads)) != 1:
            raise PICAMStateError("dynamic variable definition differs across MPI ranks")
        local_error: str | None = None
        try:
            spec.contract().shape(self.pool.dimensions)
            if spec.name in self.pool:
                raise PICAMStateError(f"field {spec.name!r} already exists")
            for alias in spec.aliases:
                try:
                    self.pool.canonical_name(alias)
                except KeyError:
                    continue
                raise PICAMStateError(f"field alias {alias!r} already exists")
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_state_error(local_error, "dynamic variable preflight")
        created = False
        local_error = None
        try:
            values = self.pool.create(spec.contract(), initial=spec.initial)
            created = True
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
            values = np.empty(0)
        errors = self.comm.allgather(local_error)
        if any(error is not None for error in errors):
            if created:
                self.pool.remove(spec.name)
            self.comm.barrier()
            message = collective_error_message(
                "dynamic variable creation rollback", errors
            )
            raise PICAMStateError(message or "dynamic variable creation failed")
        self.comm.barrier()
        return values

    def define_array(self, name: str, values: np.ndarray) -> np.ndarray:
        """Collectively copy one rank-independent array into every StatePool."""

        self._require_collective_boundary("define an array variable")
        array = np.asarray(values)
        signature = {
            "name": str(name),
            "shape": tuple(int(extent) for extent in array.shape),
            "dtype": array.dtype.str,
            "sha256": hashlib.sha256(array.tobytes(order="F")).hexdigest(),
        }
        signatures = self.comm.allgather(
            json.dumps(signature, sort_keys=True, separators=(",", ":"))
        )
        if len(set(signatures)) != 1:
            raise PICAMStateError(
                "direct NumPy state assignment differs across MPI ranks"
            )

        created = False
        local_error: str | None = None
        try:
            result = self.pool.create_from_array(str(name), array)
            created = True
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
            result = np.empty(0)
        errors = self.comm.allgather(local_error)
        if any(error is not None for error in errors):
            if created:
                self.pool.remove(str(name))
            self.comm.barrier()
            message = collective_error_message(
                "dynamic NumPy array creation rollback", errors
            )
            raise PICAMStateError(message or "dynamic NumPy array creation failed")
        self.comm.barrier()
        return result

    def delete_variable(self, name: str) -> None:
        """Collectively remove an unused dynamic field."""

        self._require_collective_boundary("delete a variable")
        local_error: str | None = None
        canonical = str(name)
        try:
            canonical = self.pool.canonical_name(name)
            if canonical not in self.pool.dynamic_fields:
                raise PICAMStateError(
                    f"field {canonical!r} is not a removable dynamic field"
                )
            dependencies = self.python_processes.dependencies(canonical)
            dependencies += self.fortran_processes.dependencies(canonical)
            dependencies += self.process_contexts.dependencies(canonical)
            if dependencies:
                raise PICAMStateError(
                    f"field {canonical!r} is used by runtime processes: "
                    + ", ".join(dependencies)
                )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_state_error(local_error, "dynamic variable deletion")
        self.pool.remove_dynamic(canonical)
        self.comm.barrier()

    @staticmethod
    def _initial_signature(value: Any) -> Mapping[str, Any]:
        array = np.asarray(value)
        return {
            "shape": tuple(int(extent) for extent in array.shape),
            "dtype": array.dtype.str,
        }

    def _process_call_arguments(
        self, arguments: Mapping[str, Any]
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Separate Python field objects from literal/array initial values."""

        bindings: dict[str, str] = {}
        initials: dict[str, Any] = {}
        for raw_name, value in arguments.items():
            name = str(raw_name)
            field: str | None = None
            if isinstance(value, str):
                field = value
            elif isinstance(value, np.ndarray):
                matches = tuple(
                    field_name
                    for field_name, array in self.pool.items()
                    if array is value
                )
                if len(matches) == 1:
                    field = matches[0]
            else:
                candidate = getattr(value, "name", None)
                if isinstance(candidate, str):
                    try:
                        field = self.pool.canonical_name(candidate)
                    except KeyError:
                        field = None
            if field is None:
                initials[name] = value
            else:
                bindings[name] = self.pool.canonical_name(field)
        return bindings, initials

    def promote_process(
        self,
        name: str,
        *,
        bindings: Mapping[str, str] | None = None,
        initials: Mapping[str, Any] | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> PICAMPromotedProcess:
        """Collectively turn one routine's caller arguments into StatePool fields."""

        self._require_collective_boundary("promote a physics process")
        requested = dict(bindings or {})
        values = dict(initials or {})
        extents = {str(key): int(value) for key, value in (dimensions or {}).items()}
        signature = json.dumps(
            {
                "name": str(name),
                "bindings": requested,
                "initials": {
                    str(key): self._initial_signature(value)
                    for key, value in values.items()
                },
                "dimensions": extents,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        signatures = self.comm.allgather(signature)
        if len(set(signatures)) != 1:
            raise PICAMStateError(
                "promoted process contract differs across MPI ranks"
            )
        record: PICAMPromotedProcess | None = None
        local_error: str | None = None
        try:
            record = self.process_contexts.promote(
                name,
                bindings=requested,
                initials=values,
                dimensions=extents,
            )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.comm.allgather(local_error)
        if any(error is not None for error in errors):
            if record is not None and record.name in self.process_contexts:
                self.process_contexts.remove(record.name)
            self.comm.barrier()
            message = collective_error_message(
                "physics process promotion rollback", errors
            )
            raise PICAMStateError(message or "physics process promotion failed")
        self.comm.barrier()
        assert record is not None
        return record

    def run_promoted_process(
        self, name: str, *, experimental: bool = False
    ) -> PICAMActionTrace:
        """Run one StatePool-bound original routine without advancing time."""

        self._require_collective_boundary("run a promoted physics process")
        if not experimental:
            raise PICAMConfigurationError(
                "isolated promoted CAM processes require experimental=True"
            )
        names = self.comm.allgather(str(name))
        if len(set(names)) != 1:
            raise PICAMStateError("promoted process name differs across MPI ranks")
        return self.process_contexts.run(name)

    def install_promoted_process(
        self,
        name: str,
        *,
        before: str | None = None,
        after: str | None = None,
        enabled: bool = True,
    ) -> PICAMAction:
        """Insert one StatePool-bound source routine into the complete workflow."""

        self._require_collective_boundary("insert a bound physics process")
        if (before is None) == (after is None):
            raise PICAMConfigurationError("provide exactly one of before or after")
        signature = (str(name), before, after, bool(enabled))
        if len(set(self.comm.allgather(signature))) != 1:
            raise PICAMStateError(
                "runtime process placement differs across MPI ranks"
            )

        action: PICAMAction | None = None
        local_error: str | None = None
        try:
            record = self.process_contexts.record(name)
            if not record.native_available:
                raise PICAMConfigurationError(
                    f"{record.name!r} is bound, but its adapter is not loaded "
                    "by the current PI-CAM executable"
                )
            try:
                existing = self.step_plan.select(record.name)
            except PICAMConfigurationError:
                existing = None
            if existing is not None:
                raise PICAMConfigurationError(
                    f"runtime process {record.name!r} is already in the workflow"
                )
            self.step_plan.select(before or after or "")
            action = PICAMAction(
                name=record.name,
                # Runtime processes are positioned in the one complete list;
                # they do not acquire a legacy cam_run1/cam_run2 identity from
                # whichever neighbor happened to be chosen at insertion time.
                phase="runtime",
                operation=record.name,
                kind="runtime_catalog_process",
                native_id=None,
                enabled=bool(enabled),
            )
            self.step_plan.add(
                action,
                before=before,
                after=after,
                experimental=True,
            )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.comm.allgather(local_error)
        if any(error is not None for error in errors):
            if action is not None:
                try:
                    selected = self.step_plan.select(action.name, phase=action.phase)
                except PICAMConfigurationError:
                    selected = None
                if selected is not None and selected.kind == "runtime_catalog_process":
                    self.step_plan.remove(
                        action.name,
                        phase=action.phase,
                        experimental=True,
                    )
            self.comm.barrier()
            message = collective_error_message(
                "runtime process insertion rollback", errors
            )
            raise PICAMStateError(message or "runtime process insertion failed")
        self.comm.barrier()
        assert action is not None
        return action

    def _install_default_history_stream(self) -> None:
        """Let Python-owned fields reach CAM's own history files by default.

        The stream mirrors the case's ``nhtfrq`` so its samples line up with
        the tape CAM writes.  With no Python-owned fields it accumulates
        nothing and adds nothing, so the run directory stays identical to the
        original model's.
        """

        if not self.default_history_stream:
            return
        if "phys_state.cid" not in self.pool:
            # Without CAM's column identifiers there is no way to place
            # Python-owned columns; leave output to the original writer.
            return
        name = "python_fields"
        if name in self.history_streams:
            return
        self.history_streams.install(
            name,
            fields=None,
            stream="h0",
            nhtfrq=self._history_nhtfrq(),
        )
        self.step_plan.add(
            PICAMAction(
                name=name,
                phase="runtime",
                operation=name,
                kind="python_history",
                native_id=None,
            ),
            after="wshist",
            experimental=True,
        )

    def _history_nhtfrq(self) -> int:
        """Read CAM's own output frequency from the run directory namelist."""

        if self.run_dir is None:
            return 0
        namelist = Path(self.run_dir) / "atm_in"
        if not namelist.is_file():
            return 0
        from .namelist import read_values

        try:
            raw = read_values(namelist).get("nhtfrq")
        except PICAMConfigurationError:
            return 0
        if raw is None:
            return 0
        try:
            return int(raw.split(",")[0].strip())
        except ValueError:
            return 0

    def set_module_parameter(self, name: str, value: object) -> dict:
        """Collectively change one audited CAM physics tunable in place."""

        return self.module_parameters.set(name, value)

    def install_history_stream(
        self,
        name: str = "python_fields",
        *,
        fields: Sequence[Any] | None = None,
        stream: str = "h0",
        nhtfrq: int = 0,
        before: str | None = None,
        after: str | None = "wshist",
        enabled: bool = True,
        time_period: str = "mean",
        precision: str = "float32",
    ) -> PICAMAction:
        """Add one Python-owned CAM-format history stream to the workflow."""

        self._require_collective_boundary("install a history stream")
        if (before is None) == (after is None):
            raise PICAMConfigurationError("provide exactly one of before or after")
        signature = (
            str(name),
            str(stream),
            None if fields is None else tuple(str(item) for item in fields),
            int(nhtfrq),
            before,
            after,
            bool(enabled),
        )
        if len(set(self.comm.allgather(signature))) != 1:
            raise PICAMStateError("history stream request differs across MPI ranks")

        action: PICAMAction | None = None
        installed = False
        local_error: str | None = None
        try:
            self.history_streams.install(
                name,
                fields=fields,
                stream=stream,
                nhtfrq=nhtfrq,
                precision=precision,
                time_period=time_period,
            )
            installed = True
            action = PICAMAction(
                name=str(name),
                phase="runtime",
                operation=str(name),
                kind="python_history",
                native_id=None,
                enabled=bool(enabled),
            )
            self.step_plan.add(
                action,
                before=before,
                after=after,
                experimental=True,
            )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.comm.allgather(local_error)
        if any(error is not None for error in errors):
            if action is not None:
                try:
                    selected = self.step_plan.select(action.name, phase=action.phase)
                except PICAMConfigurationError:
                    selected = None
                if selected is not None and selected.kind == "python_history":
                    self.step_plan.remove(
                        action.name, phase=action.phase, experimental=True
                    )
            if installed:
                self.history_streams.remove(str(name))
            self.comm.barrier()
            message = collective_error_message(
                "history stream installation rollback", errors
            )
            raise PICAMStateError(message or "history stream installation failed")
        self.comm.barrier()
        assert action is not None
        return action

    def remove_history_stream(self, name: str) -> None:
        """Collectively remove one Python-owned history stream."""

        self._require_collective_boundary("remove a history stream")
        if len(set(self.comm.allgather(str(name)))) != 1:
            raise PICAMStateError("history stream removal differs across MPI ranks")
        local_error: str | None = None
        try:
            self.step_plan.remove(str(name), phase="runtime", experimental=True)
            self.history_streams.remove(str(name))
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.comm.allgather(local_error)
        if any(error is not None for error in errors):
            self.comm.barrier()
            message = collective_error_message("history stream removal", errors)
            raise PICAMStateError(message or "history stream removal failed")
        self.comm.barrier()

    def _record_promoted_process(self, name: str) -> PICAMActionTrace:
        return self._record(
            PICAMAction(
                name=str(name),
                phase="promoted_process",
                operation=str(name),
                kind="promoted_process",
                native_id=None,
            )
        )

    def remove_promoted_process(self, name: str) -> PICAMPromotedProcess:
        """Collectively remove a promoted routine and fields it owns."""

        self._require_collective_boundary("remove a promoted physics process")
        local_error = None
        try:
            record = self.process_contexts.record(name)
            for field in record.created_fields:
                dependencies = self.python_processes.dependencies(field)
                dependencies += self.fortran_processes.dependencies(field)
                dependencies += tuple(
                    dependency
                    for dependency in self.process_contexts.dependencies(field)
                    if dependency != record.name
                )
                if dependencies:
                    raise PICAMStateError(
                        f"cannot remove promoted process {record.name!r}; field "
                        f"{field!r} is used by: " + ", ".join(dependencies)
                    )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_state_error(local_error, "physics process removal")
        try:
            action = self.step_plan.select(name)
        except PICAMConfigurationError:
            action = None
        if action is not None and action.kind == "runtime_catalog_process":
            self.step_plan.remove(
                action.name,
                phase=action.phase,
                experimental=True,
            )
        record = self.process_contexts.remove(name)
        self.comm.barrier()
        return record

    def _require_collective_boundary(self, operation: str) -> None:
        if self.lifecycle not in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            raise PICAMStateError(f"cannot {operation} from {self.lifecycle.value}")
        if self._native_call_depth:
            raise PICAMStateError(f"cannot {operation} inside a native call")
        cursors = self.comm.allgather((self.coupling_step, self._trace_count))
        if len(set(cursors)) != 1:
            raise PICAMStateError(f"MPI ranks are at different boundaries: {cursors}")

    def _collective_state_error(self, error: str | None, operation: str) -> None:
        errors = self.comm.allgather(error)
        message = collective_error_message(operation, errors)
        if message is not None:
            raise PICAMStateError(message)

    def run_action(
        self, name: str, *, phase: str | None = None, experimental: bool = False
    ) -> PICAMActionTrace:
        if self.lifecycle not in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            raise PICAMStateError(f"run action from {self.lifecycle.value}")
        if not experimental:
            raise PICAMConfigurationError(
                "isolated action execution requires experimental=True"
            )
        action = self.step_plan.select(name, phase=phase)
        # Coupler exchange and the model clock define a complete-step boundary
        # and remain explicit Python-driver responsibilities.  I/O actions are
        # Python-gated low-level services and may be invoked independently for
        # experimental workflow construction.
        if action.kind in {"boundary", "clock"}:
            raise PICAMConfigurationError(
                f"{action.operation!r} is controlled by a complete step"
            )
        return self._execute(action)

    def expand_cam_run1_leaves(
        self, *, experimental: bool = False
    ) -> tuple[PICAMAction, ...]:
        """Replace three composite stages with ordered native leaf actions."""

        self.step_plan.expand_cam_run1_leaves(experimental=experimental)
        return self.step_plan.in_phase("cam_run1")

    def expand_cam_run2_leaves(
        self, *, experimental: bool = False
    ) -> tuple[PICAMAction, ...]:
        """Replace admitted ``cam_run2`` composites with ordered leaves."""

        self.step_plan.expand_cam_run2_leaves(experimental=experimental)
        return self.step_plan.in_phase("cam_run2")

    def expand_cam_run4_leaves(
        self, *, experimental: bool = False
    ) -> tuple[PICAMAction, ...]:
        """Replace the ``cam_run4`` finish composite with ordered leaves."""

        self.step_plan.expand_cam_run4_leaves(experimental=experimental)
        return self.step_plan.in_phase("cam_run4")

    def expand_cam_run2_run4_leaves(
        self, *, experimental: bool = False
    ) -> tuple[PICAMAction, ...]:
        """Expand all admitted leaves from ``cam_run2`` through run4."""

        self.step_plan.expand_cam_run2_run4_leaves(
            experimental=experimental
        )
        return tuple(
            action
            for phase in ("cam_run2", "cam_run3", "cam_run4")
            for action in self.step_plan.in_phase(phase)
        )

    def run_phase(
        self, phase: str, *, experimental: bool = False
    ) -> tuple[PICAMActionTrace, ...]:
        if not experimental:
            raise PICAMConfigurationError(
                "isolated phase execution requires experimental=True"
            )
        actions = self.step_plan.in_phase(phase)
        if any(action.kind in {"boundary", "clock", "io"} for action in actions):
            raise PICAMConfigurationError(
                f"phase {phase!r} contains step-controlled boundary, clock, or I/O"
            )
        return tuple(self._execute(action) for action in actions)

    def run_kernel(
        self, name: str, *, experimental: bool = False, pool: Mapping[str, np.ndarray] | None = None
    ) -> PICAMActionTrace:
        """Run one raw-array numerical routine without its CAM host stage.

        ``pool`` defaults to this rank's StatePool; a native Python process
        passes its own mapping of arrays instead, one chunk per last axis.
        """

        if self.lifecycle not in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            raise PICAMStateError(f"run kernel from {self.lifecycle.value}")
        if not experimental:
            raise PICAMConfigurationError(
                "isolated raw CAM kernels require experimental=True"
            )
        execute = getattr(self.backend, "execute_kernel", None)
        if not callable(execute):
            raise PICAMConfigurationError("the selected backend has no direct kernels")
        with self.profiler.region(f"CAM:{name}"):
            with self.profiler.region("FORTRAN:DIRECT_KERNEL"):
                execute(name, self.pool if pool is None else pool, fcomm=self.fcomm)
        return self._record(
            PICAMAction(
                name=name,
                phase="direct_kernel",
                operation=name,
                kind="kernel",
                native_id=None,
            )
        )

    def step(self) -> tuple[PICAMActionTrace, ...]:
        with self.profiler.region("FREECAM:STEP"):
            return self._step()

    def _step(self) -> tuple[PICAMActionTrace, ...]:
        if self.lifecycle not in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            raise PICAMStateError(f"step from {self.lifecycle.value}")
        self.lifecycle = PICAMLifecycle.RUNNING
        actions = tuple(self.step_plan)
        imports = tuple(action for action in actions if action.operation == "boundary_import")
        exports = tuple(action for action in actions if action.operation == "boundary_export")
        body = tuple(action for action in actions if action.kind != "boundary")
        if len(imports) != 1 or len(exports) != 1:
            raise PICAMStateError("complete PI-CAM step requires one import and one export")
        capture: list[PICAMActionTrace] = []
        self._trace_captures.append(capture)
        try:
            source_step = getattr(self.backend, "execute_source_step", None)
            use_source_boundary = (
                self.config.execution_mode == "source_compat"
                and callable(source_step)
                and self.step_plan.is_source_default
                and self.config.substeps_per_coupling == 1
            )
            if use_source_boundary:
                # Boundary data remains Python-owned.  For the unchanged
                # scientific plan, pass both arrays through one original CAM
                # numerical call so hidden HOMME state sees exactly the same
                # call/stack boundary as the source executable.
                self._collective_boundary_call(
                    f"step {self._boundary_step} boundary import",
                    lambda: self.boundary.import_fields(
                        self._boundary_step, self.rank, self.pool
                    ),
                )
                self._record(imports[0])
                with self.profiler.region("FORTRAN:SOURCE_STEP"):
                    source_step(
                        self.pool,
                        fcomm=self.fcomm,
                        apply_import=self._collective_boundary_call(
                            f"step {self._boundary_step} boundary import schedule",
                            lambda: self.boundary.has_fresh_import(
                                self._boundary_step, self.rank
                            ),
                        ),
                    )
                for action in body:
                    if action.kind == "io" and self.history_callback is not None:
                        self.history_callback(action, self)
                    if action.operation == "advance_timestep":
                        self._native_step += 1
                        if self._advance_public_clock:
                            self.clock.advance()
                            self._sync_clock_fields()
                    self._record(action)
                self._record(exports[0])
                self._collective_boundary_call(
                    f"step {self._boundary_step} boundary export",
                    lambda: self.boundary.export_fields(
                        self._boundary_step, self.rank, self.pool
                    ),
                )
            else:
                self._execute(imports[0])
                for _ in range(self.config.substeps_per_coupling):
                    for action in body:
                        self._execute(action)
                self._execute(exports[0])
            self.coupling_step += 1
            self._boundary_step += 1
        except Exception:
            self.lifecycle = PICAMLifecycle.FAILED
            raise
        finally:
            self._trace_captures.pop()
        return tuple(capture)

    def advance(self, steps: int | None = None) -> None:
        count = self.config.stop_n if steps is None else int(steps)
        if count < 0:
            raise PICAMConfigurationError("steps cannot be negative")
        for _ in range(count):
            self.step()

    def save(self, directory: str | Path) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        rank_file = destination / f"rank-{self.rank:04d}.npz"
        np.savez(rank_file, **self.pool.snapshot(restart_only=True))
        metadata = {
            "schema_version": 1,
            "scope": "python-statepool-only",
            "native_restart_complete": False,
            "config_fingerprint": self.config.fingerprint,
            "rank": self.rank,
            "size": self.size,
            "coupling_step": self.coupling_step,
            "boundary_step": self._boundary_step,
            "native_step": self._native_step,
            "clock": {
                "year": self.clock.year,
                "month": self.clock.month,
                "day": self.clock.day,
                "seconds": self.clock.seconds,
                "nstep": self.clock.nstep,
            },
        }
        (destination / f"rank-{self.rank:04d}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        return rank_file

    def finalize(self) -> None:
        if self.lifecycle == PICAMLifecycle.FINALIZED:
            return
        try:
            with self.profiler.region("FREECAM:FINALIZE"):
                self._finalize()
        finally:
            self.profiler.stop_total()
            if self.run_dir is not None:
                self.profiler.write(
                    self.run_dir,
                    self.comm,
                    profile=self._cesm_timing_context(),
                )

    def _finalize(self) -> None:
        if self.lifecycle == PICAMLifecycle.FINALIZED:
            return
        if self.lifecycle in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            # CAM writes a tape sample one step before this stream's window
            # for that sample closes, so the run's last window is still open
            # here.  Closing it is collective; every rank reaches this line.
            # Whatever it queues for the tape CAM still holds is written
            # after cam_final.
            self.history_streams.flush()
        if self.lifecycle not in {
            PICAMLifecycle.INITIALIZED,
            PICAMLifecycle.RUNNING,
            PICAMLifecycle.FAILED,
        }:
            raise PICAMStateError(f"finalize from {self.lifecycle.value}")
        failed = self.lifecycle == PICAMLifecycle.FAILED
        try:
            # A failed native action can leave opaque CAM state between
            # lifecycle boundaries.  Calling cam_final from that state can
            # mask the original error (and, for Python-owned pointer shells,
            # can attempt to release memory that Fortran does not own).
            shared_cam = bool(self.boundary.shares_cam_instance)
            if shared_cam and not failed:
                with self.profiler.region("BOUNDARY:FINALIZE_BEGIN"):
                    getattr(self.boundary, "finalize_before_cam")()
            if self._backend_initialized and not failed:
                with self.profiler.region("FORTRAN:CAM_FINALIZE"):
                    self.backend.finalize(self.pool, fcomm=self.fcomm)
                self._backend_initialized = False
                # A run that stops on a history boundary closes its last
                # window while CAM still owns the file that window belongs
                # to.  cam_final writes and closes that tape, so this is the
                # first moment the sample can be appended -- and the last.
                self.history_streams.drain()
            with self.profiler.region("BOUNDARY:FINALIZE"):
                if shared_cam and not failed:
                    getattr(self.boundary, "finalize_after_cam")()
                elif not shared_cam:
                    self.boundary.finalize()
        finally:
            self.lifecycle = PICAMLifecycle.FINALIZED
            if self._previous_directory is not None:
                os.chdir(self._previous_directory)
                self._previous_directory = None

    def _cesm_timing_context(self) -> CESMTimingContext:
        """Assemble the metadata for the CIME-format performance profile.

        The wall clock and environment are read once here, at finalization, so
        the profile carries the run's real LID and date while the formatter it
        feeds stays pure and testable.
        """

        config = self.config
        online = config.boundary_mode == "online"
        components = (
            ("cpl", "cpl"),
            ("atm", "cam"),
            ("lnd", "clm" if online else "slnd"),
            ("ice", "cice" if online else "sice"),
            ("ocn", "docn" if online else "socn"),
            ("rof", "rtm" if online else "srof"),
            ("glc", "sglc"),
            ("wav", "swav"),
            ("esp", "sesp"),
        )
        surface = "_".join(model.upper() for _label, model in components[2:6])
        compset = (
            f"{config.orbital_year}_CAM%{config.physics_package.upper()}"
            f"_{surface}_SGLC_SWAV_SESP"
        )
        job = os.environ.get("PBS_JOBID") or os.uname().nodename
        return CESMTimingContext(
            case_name=config.case_name,
            lid=f"{job}.{time.strftime('%y%m%d-%H%M%S')}",
            machine=os.environ.get("NCAR_HOST") or "derecho",
            caseroot="" if self.run_dir is None else str(self.run_dir),
            user=os.environ.get("USER") or os.uname().nodename,
            curr_date=time.strftime("%a %b %e %H:%M:%S %Y"),
            driver="freeCAM",
            grid=(
                f"a%{config.resolution}np4 "
                f"{config.physics_package}-{config.dynamics}"
            ),
            compset=compset,
            run_type="startup, continue_run = FALSE (inittype = TRUE)",
            timestep_seconds=config.timestep_seconds,
            mpi_ranks=self.size,
            tasks_per_node=int(os.environ.get("FREECAM_TASKS_PER_NODE") or 128),
            components=components,
        )

    def __enter__(self) -> "PICAMDriver":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.finalize()

    def _collective_boundary_call(
        self,
        label: str,
        function: Callable[[], Any],
    ) -> Any:
        timer = self._boundary_timer_name(label)
        with self.profiler.region(timer):
            return self._collective_boundary_call_impl(label, function)

    @staticmethod
    def _boundary_timer_name(label: str) -> str:
        lowered = label.lower()
        if "schedule" in lowered:
            return "BOUNDARY:HAS_FRESH_IMPORT"
        if "initialization" in lowered:
            return "BOUNDARY:INITIALIZE"
        if "import" in lowered:
            return "BOUNDARY:IMPORT"
        if "export" in lowered:
            return "BOUNDARY:EXPORT"
        return "BOUNDARY:CALL"

    def _collective_boundary_call_impl(
        self,
        label: str,
        function: Callable[[], Any],
    ) -> Any:
        """Run one rank-local boundary operation and fail all ranks together.

        Boundary replay reads and comparisons are local filesystem/NumPy
        operations.  A failure may therefore occur on only a subset of ranks.
        No rank may enter the next native MPI kernel until every rank agrees
        that the boundary operation succeeded.
        """

        result: Any = None
        local_error: str | None = None
        try:
            result = function()
        except BaseException:
            local_error = traceback.format_exc()
        errors = self.comm.allgather(local_error)
        failure = collective_error_message(label, errors)
        if failure is not None:
            # Every rank raises the whole message, not a pointer to rank 0's:
            # the first rank to reach MPI_Abort kills the others where they
            # stand, so a message only rank 0 holds is lost exactly when it
            # is needed.  The text is already grouped by distinct traceback,
            # so this repeats nothing per rank.
            raise BoundaryReplayError(failure)
        return result
