"""Serializable edits and action plans for isolated model branches."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Union

import numpy as np


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FIELD_OPERATIONS = frozenset(("set", "add", "multiply"))
_OBSERVATION_STATISTICS = frozenset(("min", "max", "mean"))
SEGMENT_PLAN_SCHEMA_VERSION = 1


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item) for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class FieldEdit:
    name: str
    operation: str
    value: float
    unsafe: bool = False

    def __post_init__(self) -> None:
        if self.operation not in _FIELD_OPERATIONS:
            raise ValueError(
                f"field operation must be one of {sorted(_FIELD_OPERATIONS)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operation": self.operation,
            "value": self.value,
            "unsafe": self.unsafe,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FieldEdit":
        return cls(
            name=str(values["name"]),
            operation=str(values["operation"]),
            value=float(values["value"]),
            unsafe=bool(values.get("unsafe", False)),
        )


@dataclass(frozen=True, slots=True)
class SchemeMove:
    name: str
    before: str | None = None
    after: str | None = None
    to_group: str | None = None

    def __post_init__(self) -> None:
        if self.before is not None and self.after is not None:
            raise ValueError("scheme move accepts at most one of before or after")
        if self.before is None and self.after is None and self.to_group is None:
            raise ValueError("scheme move requires before, after, or to_group")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "before": self.before,
            "after": self.after,
            "to_group": self.to_group,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SchemeMove":
        return cls(
            name=str(values["name"]),
            before=(None if values.get("before") is None else str(values["before"])),
            after=(None if values.get("after") is None else str(values["after"])),
            to_group=(
                None
                if values.get("to_group") is None
                else str(values["to_group"])
            ),
        )


@dataclass(frozen=True, slots=True)
class PrepareInitialStep:
    """Run the driver's complete initial preparation boundary."""


@dataclass(frozen=True, slots=True)
class RunPhase:
    name: str


@dataclass(frozen=True, slots=True)
class RunScheme:
    name: str
    group: str | None = None


@dataclass(frozen=True, slots=True)
class RunSchemeGroup:
    group: str


@dataclass(frozen=True, slots=True)
class RunSteps:
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("step action count must be non-negative")


@dataclass(frozen=True, slots=True)
class SetSchemeEnabled:
    name: str
    enabled: bool
    group: str | None = None


@dataclass(frozen=True, slots=True)
class MoveScheme:
    name: str
    before: str | None = None
    after: str | None = None
    to_group: str | None = None

    def __post_init__(self) -> None:
        SchemeMove(
            self.name,
            before=self.before,
            after=self.after,
            to_group=self.to_group,
        )


@dataclass(frozen=True, slots=True)
class ObserveFields:
    fields: tuple[str, ...]
    statistics: tuple[str, ...] = ("min", "max", "mean")

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "statistics", tuple(self.statistics))
        if not self.fields:
            raise ValueError("observe action requires at least one field")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("observe action fields must be unique")
        if not self.statistics:
            raise ValueError("observe action requires at least one statistic")
        unknown = set(self.statistics) - _OBSERVATION_STATISTICS
        if unknown:
            raise ValueError(
                f"unknown observation statistics: {sorted(unknown)}"
            )
        if len(set(self.statistics)) != len(self.statistics):
            raise ValueError("observe action statistics must be unique")


@dataclass(frozen=True, slots=True)
class DefineVariable:
    spec: Any
    initial_value: Any = 0.0

    def __post_init__(self) -> None:
        from .plugins import VariableSpec

        if isinstance(self.spec, Mapping):
            object.__setattr__(
                self, "spec", VariableSpec.from_mapping(self.spec)
            )
        elif not isinstance(self.spec, VariableSpec):
            raise TypeError("define_variable requires a VariableSpec")


@dataclass(frozen=True, slots=True)
class InstallPhysics:
    plugin: Any
    initial_values: Mapping[str, Any] | None = None
    effective: str = "now"

    def __post_init__(self) -> None:
        from .plugins import PhysicsPluginSpec

        if isinstance(self.plugin, Mapping):
            object.__setattr__(
                self, "plugin", PhysicsPluginSpec.from_mapping(self.plugin)
            )
        elif not isinstance(self.plugin, PhysicsPluginSpec):
            raise TypeError("install_physics requires a PhysicsPluginSpec")
        if self.effective not in {"now", "next_step"}:
            raise ValueError("effective must be 'now' or 'next_step'")
        object.__setattr__(
            self, "initial_values", dict(self.initial_values or {})
        )


@dataclass(frozen=True, slots=True)
class ActivatePhysics:
    name: str


@dataclass(frozen=True, slots=True)
class DeactivatePhysics:
    name: str


Action = Union[
    PrepareInitialStep,
    RunPhase,
    RunScheme,
    RunSchemeGroup,
    RunSteps,
    SetSchemeEnabled,
    MoveScheme,
    FieldEdit,
    ObserveFields,
    DefineVariable,
    InstallPhysics,
    ActivatePhysics,
    DeactivatePhysics,
]


def _action_as_dict(action: Action) -> dict[str, Any]:
    if isinstance(action, PrepareInitialStep):
        return {"type": "prepare_initial_step"}
    if isinstance(action, RunPhase):
        return {"type": "run_phase", "name": action.name}
    if isinstance(action, RunScheme):
        return {
            "type": "run_scheme",
            "name": action.name,
            "group": action.group,
        }
    if isinstance(action, RunSchemeGroup):
        return {"type": "run_scheme_group", "group": action.group}
    if isinstance(action, RunSteps):
        return {"type": "run_steps", "count": action.count}
    if isinstance(action, SetSchemeEnabled):
        return {
            "type": "set_scheme_enabled",
            "name": action.name,
            "enabled": action.enabled,
            "group": action.group,
        }
    if isinstance(action, MoveScheme):
        return {
            "type": "move_scheme",
            "name": action.name,
            "before": action.before,
            "after": action.after,
            "to_group": action.to_group,
        }
    if isinstance(action, FieldEdit):
        return {"type": "field_edit", **action.as_dict()}
    if isinstance(action, ObserveFields):
        return {
            "type": "observe_fields",
            "fields": list(action.fields),
            "statistics": list(action.statistics),
        }
    if isinstance(action, DefineVariable):
        return {
            "type": "define_variable",
            "spec": action.spec.as_dict(),
            "initial_value": _json_value(action.initial_value),
        }
    if isinstance(action, InstallPhysics):
        return {
            "type": "install_physics",
            "plugin": action.plugin.as_dict(),
            "initial_values": _json_value(action.initial_values or {}),
            "effective": action.effective,
        }
    if isinstance(action, ActivatePhysics):
        return {"type": "activate_physics", "name": action.name}
    if isinstance(action, DeactivatePhysics):
        return {"type": "deactivate_physics", "name": action.name}
    raise TypeError(f"unsupported segment action {type(action).__name__}")


def _optional_string(values: Mapping[str, Any], name: str) -> str | None:
    value = values.get(name)
    return None if value is None else str(value)


def _action_from_mapping(values: Mapping[str, Any]) -> Action:
    action_type = str(values.get("type", ""))
    if action_type == "prepare_initial_step":
        return PrepareInitialStep()
    if action_type == "run_phase":
        return RunPhase(str(values["name"]))
    if action_type == "run_scheme":
        return RunScheme(
            str(values["name"]),
            group=_optional_string(values, "group"),
        )
    if action_type == "run_scheme_group":
        return RunSchemeGroup(str(values["group"]))
    if action_type == "run_steps":
        return RunSteps(int(values.get("count", 1)))
    if action_type == "set_scheme_enabled":
        enabled = values.get("enabled")
        if not isinstance(enabled, bool):
            raise TypeError("set_scheme_enabled requires a bool enabled value")
        return SetSchemeEnabled(
            str(values["name"]),
            enabled,
            group=_optional_string(values, "group"),
        )
    if action_type == "move_scheme":
        return MoveScheme(
            str(values["name"]),
            before=_optional_string(values, "before"),
            after=_optional_string(values, "after"),
            to_group=_optional_string(values, "to_group"),
        )
    if action_type == "field_edit":
        return FieldEdit.from_mapping(values)
    if action_type == "observe_fields":
        return ObserveFields(
            tuple(str(value) for value in values.get("fields", ())),
            tuple(
                str(value)
                for value in values.get(
                    "statistics", ("min", "max", "mean")
                )
            ),
        )
    if action_type == "define_variable":
        return DefineVariable(
            values["spec"],
            initial_value=values.get("initial_value", 0.0),
        )
    if action_type == "install_physics":
        return InstallPhysics(
            values["plugin"],
            initial_values=values.get("initial_values"),
            effective=str(values.get("effective", "now")),
        )
    if action_type == "activate_physics":
        return ActivatePhysics(str(values["name"]))
    if action_type == "deactivate_physics":
        return DeactivatePhysics(str(values["name"]))
    raise ValueError(f"unknown segment action type {action_type!r}")


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    """One serializable sequence executed inside a single MPI segment."""

    name: str
    actions: tuple[Action, ...] = ()
    unsafe: bool = False

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "plan name may contain only letters, digits, dot, dash, and underscore"
            )
        object.__setattr__(self, "actions", tuple(self.actions))
        for action in self.actions:
            _action_as_dict(action)

    @property
    def step_count(self) -> int:
        return sum(
            action.count
            for action in self.actions
            if isinstance(action, RunSteps)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEGMENT_PLAN_SCHEMA_VERSION,
            "name": self.name,
            "unsafe": self.unsafe,
            "actions": [_action_as_dict(action) for action in self.actions],
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SegmentPlan":
        version = values.get("schema_version")
        if version != SEGMENT_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported segment plan schema {version!r}; "
                f"expected {SEGMENT_PLAN_SCHEMA_VERSION}"
            )
        rows = values.get("actions")
        if not isinstance(rows, list):
            raise TypeError("segment plan requires an actions list")
        if not all(isinstance(row, Mapping) for row in rows):
            raise TypeError("each segment plan action must be a JSON object")
        return cls(
            name=str(values["name"]),
            actions=tuple(_action_from_mapping(row) for row in rows),
            unsafe=bool(values.get("unsafe", False)),
        )


@dataclass(frozen=True, slots=True)
class BranchSpec:
    """One branch from a common model snapshot."""

    name: str
    steps: int = 1
    disable_schemes: tuple[str, ...] = ()
    enable_schemes: tuple[str, ...] = ()
    scheme_moves: tuple[SchemeMove, ...] = ()
    field_edits: tuple[FieldEdit, ...] = ()

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "branch name may contain only letters, digits, dot, dash, "
                "and underscore"
            )
        if self.steps < 0:
            raise ValueError("branch steps must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "steps": self.steps,
            "disable_schemes": list(self.disable_schemes),
            "enable_schemes": list(self.enable_schemes),
            "scheme_moves": [move.as_dict() for move in self.scheme_moves],
            "field_edits": [edit.as_dict() for edit in self.field_edits],
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BranchSpec":
        return cls(
            name=str(values["name"]),
            steps=int(values.get("steps", 1)),
            disable_schemes=tuple(
                str(value) for value in values.get("disable_schemes", ())
            ),
            enable_schemes=tuple(
                str(value) for value in values.get("enable_schemes", ())
            ),
            scheme_moves=tuple(
                SchemeMove.from_mapping(value)
                for value in values.get("scheme_moves", ())
            ),
            field_edits=tuple(
                FieldEdit.from_mapping(value)
                for value in values.get("field_edits", ())
            ),
        )

    def apply(self, driver: Any) -> None:
        """Apply branch-local edits after restoring private arrays."""

        for name in self.disable_schemes:
            driver.scheme_plan.disable(name, unsafe=True)
        for name in self.enable_schemes:
            driver.scheme_plan.enable(name)
        for move in self.scheme_moves:
            driver.scheme_plan.move(
                move.name,
                before=move.before,
                after=move.after,
                to_group=move.to_group,
                unsafe=True,
            )
        for edit in self.field_edits:
            _apply_field_edit(driver, edit)

    def to_segment_plan(self) -> SegmentPlan:
        """Preserve the legacy edit-then-step ordering as an action plan."""

        actions: list[Action] = []
        actions.extend(
            SetSchemeEnabled(name, False) for name in self.disable_schemes
        )
        actions.extend(
            SetSchemeEnabled(name, True) for name in self.enable_schemes
        )
        actions.extend(
            MoveScheme(
                move.name,
                before=move.before,
                after=move.after,
                to_group=move.to_group,
            )
            for move in self.scheme_moves
        )
        actions.extend(self.field_edits)
        actions.append(RunSteps(self.steps))
        return SegmentPlan(
            self.name,
            tuple(actions),
            unsafe=bool(
                self.disable_schemes
                or self.scheme_moves
                or any(edit.unsafe for edit in self.field_edits)
            ),
        )


def _apply_field_edit(driver: Any, edit: FieldEdit) -> None:
    current = driver.pool.get(edit.name, unsafe=edit.unsafe)
    if edit.operation == "set":
        updated = np.full_like(current, edit.value)
    elif edit.operation == "add":
        updated = np.add(current, edit.value)
    else:
        updated = np.multiply(current, edit.value)
    driver.pool.set(edit.name, updated, unsafe=edit.unsafe)


def validate_segment_plan(driver: Any, plan: SegmentPlan) -> None:
    """Validate a complete plan before mutating the driver."""

    from .scheme_plan import SCHEME_GROUPS

    scheme_plan = driver.scheme_plan.copy()
    planned_plugins: set[str] = set(
        getattr(getattr(driver, "plugins", None), "installed", {})
    )
    planned_processes: set[str] = set()
    planned_fields: set[str] = (
        set(driver.pool.contracts)
        | set(getattr(driver.pool, "_aliases", {}))
    )
    for action in plan.actions:
        if isinstance(action, RunPhase):
            if action.name not in driver.phase_names:
                raise ValueError(
                    f"unknown model phase {action.name!r}; "
                    f"choose one of {driver.phase_names}"
                )
            if not plan.unsafe:
                raise ValueError("run_phase actions require SegmentPlan(unsafe=True)")
        elif isinstance(action, RunScheme):
            if action.name not in planned_processes:
                scheme_plan.scheme(action.name, group=action.group)
            if not plan.unsafe:
                raise ValueError("run_scheme actions require SegmentPlan(unsafe=True)")
        elif isinstance(action, RunSchemeGroup):
            if action.group not in SCHEME_GROUPS:
                raise ValueError(
                    f"unknown scheme group {action.group!r}; "
                    f"choose one of {SCHEME_GROUPS}"
                )
            if not plan.unsafe:
                raise ValueError(
                    "run_scheme_group actions require SegmentPlan(unsafe=True)"
                )
        elif isinstance(action, SetSchemeEnabled):
            scheme_plan.scheme(action.name, group=action.group)
            if not action.enabled and not plan.unsafe:
                raise ValueError(
                    "disabling a scheme requires SegmentPlan(unsafe=True)"
                )
            if action.enabled:
                scheme_plan.enable(action.name, group=action.group)
            else:
                scheme_plan.disable(
                    action.name, group=action.group, unsafe=True
                )
        elif isinstance(action, MoveScheme):
            scheme_plan.scheme(action.name)
            if action.before is not None:
                scheme_plan.scheme(action.before)
            if action.after is not None:
                scheme_plan.scheme(action.after)
            if action.to_group is not None and action.to_group not in SCHEME_GROUPS:
                raise ValueError(
                    f"unknown destination group {action.to_group!r}; "
                    f"choose one of {SCHEME_GROUPS}"
                )
            if not plan.unsafe:
                raise ValueError("moving a scheme requires SegmentPlan(unsafe=True)")
            scheme_plan.move(
                action.name,
                before=action.before,
                after=action.after,
                to_group=action.to_group,
                unsafe=True,
            )
        elif isinstance(action, FieldEdit):
            if action.name not in planned_fields:
                raise ValueError(f"unknown state field {action.name!r}")
            contract = (
                driver.pool.contract(action.name)
                if action.name in driver.pool.contracts
                or action.name in getattr(driver.pool, "_aliases", {})
                else None
            )
            if action.unsafe and not plan.unsafe:
                raise ValueError(
                    "unsafe field edits require SegmentPlan(unsafe=True)"
                )
            if (
                contract is not None
                and driver.pool.sealed
                and not contract.writable
                and not action.unsafe
            ):
                raise ValueError(
                    f"field {action.name!r} is read-only after initialization"
                )
        elif isinstance(action, ObserveFields):
            for name in action.fields:
                if name not in planned_fields:
                    driver.pool.contract(name)
        elif isinstance(action, DefineVariable):
            contract = action.spec.contract()
            contract.shape(driver.pool.dimensions)
            np.dtype(contract.dtype)
            if contract.standard_name in planned_fields:
                raise ValueError(
                    f"duplicate state field {contract.standard_name!r}"
                )
            planned_fields.add(contract.standard_name)
            planned_fields.update(contract.aliases)
        elif isinstance(action, InstallPhysics):
            if not plan.unsafe:
                raise ValueError(
                    "install_physics actions require SegmentPlan(unsafe=True)"
                )
            for variable in action.plugin.variables:
                variable.contract().shape(driver.pool.dimensions)
                np.dtype(variable.dtype)
                planned_fields.add(variable.name)
                planned_fields.update(variable.aliases)
            plugin_name = action.plugin.name
            if plugin_name is not None:
                if plugin_name in planned_plugins:
                    raise ValueError(
                        f"physics plugin {plugin_name!r} is already planned"
                    )
                planned_plugins.add(plugin_name)
            planned_processes.update(
                placement.process
                for placement in action.plugin.placements
            )
        elif isinstance(action, (ActivatePhysics, DeactivatePhysics)):
            if not plan.unsafe:
                raise ValueError(
                    f"{_action_as_dict(action)['type']} actions require "
                    "SegmentPlan(unsafe=True)"
                )
            if (
                action.name not in planned_plugins
                and action.name
                not in getattr(
                    getattr(driver, "plugins", None), "installed", {}
                )
            ):
                raise ValueError(
                    f"unknown planned physics plugin {action.name!r}"
                )


def _observe_fields(driver: Any, action: ObserveFields) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for name in action.fields:
        values = np.asarray(driver.pool.get(name))
        local: dict[str, Any] = {
            "rank": int(driver.comm.rank),
            "shape": list(values.shape),
            "dtype": values.dtype.str,
            "count": int(values.size),
            "sum": float(np.sum(values, dtype=np.float64)),
        }
        if "min" in action.statistics:
            local["min"] = float(np.min(values))
        if "max" in action.statistics:
            local["max"] = float(np.max(values))
        if "mean" in action.statistics:
            local["mean"] = float(np.mean(values, dtype=np.float64))
        gathered = driver.comm.gather(local, root=0)
        if driver.comm.rank != 0:
            continue
        assert gathered is not None
        total_count = sum(int(row["count"]) for row in gathered)
        global_statistics: dict[str, float] = {}
        if "min" in action.statistics:
            global_statistics["min"] = min(float(row["min"]) for row in gathered)
        if "max" in action.statistics:
            global_statistics["max"] = max(float(row["max"]) for row in gathered)
        if "mean" in action.statistics:
            global_statistics["mean"] = (
                sum(float(row["sum"]) for row in gathered) / total_count
            )
        for row in gathered:
            row.pop("sum", None)
        observations.append(
            {
                "field": name,
                "global": global_statistics,
                "ranks": gathered,
            }
        )
    return observations


def execute_segment_plan(
    driver: Any, plan: SegmentPlan
) -> tuple[dict[str, Any], ...]:
    """Execute one validated plan collectively on all model ranks."""

    validate_segment_plan(driver, plan)
    trace: list[dict[str, Any]] = []
    for index, action in enumerate(plan.actions):
        step_before = int(driver.clock.nstep)
        calls_before = int(driver.backend.call_count)
        observations: list[dict[str, Any]] = []
        if isinstance(action, PrepareInitialStep):
            driver.prepare_initial_step()
        elif isinstance(action, RunPhase):
            driver.run_phase(action.name)
        elif isinstance(action, RunScheme):
            driver.run_scheme(action.name, group=action.group)
        elif isinstance(action, RunSchemeGroup):
            driver.run_scheme_group(action.group)
        elif isinstance(action, RunSteps):
            for _ in range(action.count):
                driver.step()
        elif isinstance(action, SetSchemeEnabled):
            if action.enabled:
                driver.scheme_plan.enable(action.name, group=action.group)
            else:
                driver.scheme_plan.disable(
                    action.name, group=action.group, unsafe=True
                )
        elif isinstance(action, MoveScheme):
            driver.scheme_plan.move(
                action.name,
                before=action.before,
                after=action.after,
                to_group=action.to_group,
                unsafe=True,
            )
        elif isinstance(action, FieldEdit):
            _apply_field_edit(driver, action)
        elif isinstance(action, ObserveFields):
            observations = _observe_fields(driver, action)
        elif isinstance(action, DefineVariable):
            driver.define_variable(
                action.spec, initial=action.initial_value
            )
        elif isinstance(action, InstallPhysics):
            driver.install_physics(
                action.plugin,
                initial_values=action.initial_values,
                effective=action.effective,
                unsafe=True,
            )
        elif isinstance(action, ActivatePhysics):
            driver.activate_physics(action.name, unsafe=True)
        elif isinstance(action, DeactivatePhysics):
            driver.deactivate_physics(action.name, unsafe=True)
        else:  # pragma: no cover - SegmentPlan validates this at construction.
            raise TypeError(f"unsupported segment action {type(action).__name__}")
        driver.comm.barrier()
        record = {
            "index": index,
            "type": _action_as_dict(action)["type"],
            "action": _action_as_dict(action),
            "step_before": step_before,
            "step_after": int(driver.clock.nstep),
            "last_phase": driver._last_phase,
            "last_scheme": driver._last_scheme,
            "last_scheme_group": driver._last_scheme_group,
            "native_calls_delta": int(driver.backend.call_count) - calls_before,
        }
        if observations:
            record["observations"] = observations
        trace.append(record)
    return tuple(trace)
