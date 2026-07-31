"""Small, Pythonic façades over the serializable model control API."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .ccpp_suite import PHYSICS_AFTER_COUPLER, PHYSICS_BEFORE_COUPLER
from .experiment import (
    Action,
    DefineVariable,
    FieldEdit,
    InstallPhysics,
    InstallPythonProcess,
    MoveScheme,
    ObserveFields,
    PrepareInitialStep,
    RunPhase,
    RunScheme,
    RunSchemeGroup,
    RunSteps,
    RemovePythonProcess,
    SegmentPlan,
    SetSchemeEnabled,
)
from .python_processes import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    PythonProcessSpec,
)
from .plugins import (
    PhysicsPluginSpec,
    SchemePlacement,
    VariableSpec,
)


_DIMENSION_ALIASES = {
    "column": "nphys_local",
    "columns": "nphys_local",
    "level": "pver",
    "levels": "pver",
    "interface_level": "pverp",
    "interface_levels": "pverp",
    "constituent": "number_of_ccpp_constituents",
    "constituents": "number_of_ccpp_constituents",
}
_GROUP_ALIASES = {
    "before": PHYSICS_BEFORE_COUPLER,
    "before_coupler": PHYSICS_BEFORE_COUPLER,
    PHYSICS_BEFORE_COUPLER: PHYSICS_BEFORE_COUPLER,
    "after": PHYSICS_AFTER_COUPLER,
    "after_coupler": PHYSICS_AFTER_COUPLER,
    PHYSICS_AFTER_COUPLER: PHYSICS_AFTER_COUPLER,
}


def _dimensions(values: Sequence[str | int]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("dims must be a sequence, not one string")
    return tuple(
        _DIMENSION_ALIASES.get(str(value), str(value))
        for value in values
    )


def _group(value: str | None) -> str | None:
    if value is None:
        return None
    return _GROUP_ALIASES.get(str(value), str(value))


def _wait(value: Any) -> Any:
    result = getattr(value, "result", None)
    return result() if callable(result) else value


class FieldReference:
    """A named field bound to one local or remote model controller."""

    def __init__(self, fields: "FieldCollection", name: str) -> None:
        self._fields = fields
        self.name = str(name)

    def get(self, *, rank: int | str | None = None) -> Any:
        owner = self._fields.owner
        getter = getattr(owner, "field", None)
        if getter is None:
            getter = owner.get_field
        if rank is None:
            return getter(self.name)
        return getter(self.name, rank=rank)

    def set(
        self,
        value: Any,
        *,
        rank: int | str | None = None,
        unsafe: bool = False,
    ) -> Any:
        kwargs: dict[str, Any] = {"unsafe": bool(unsafe)}
        if rank is not None:
            kwargs["rank"] = rank
        return self._fields.owner.set_field(self.name, value, **kwargs)

    def stats(self, *, rank: int | str | None = None) -> Any:
        owner = self._fields.owner
        getter = getattr(owner, "field_stats", None)
        if getter is None:
            getter = getattr(owner, "get_field_stats", None)
        if getter is not None:
            if rank is None:
                return getter(self.name)
            return getter(self.name, rank=rank)
        values = np.asarray(self.get(rank=rank))
        return {
            "shape": values.shape,
            "dtype": values.dtype.str,
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    def info(self) -> Mapping[str, Any]:
        getter = getattr(self._fields.owner, "field_info", None)
        if getter is None:
            raise AttributeError(
                "this controller does not expose field metadata directly"
            )
        return getter(self.name)

    def _edit(self, operation: str, value: Any) -> "FieldReference":
        editor = getattr(self._fields.owner, "edit_field", None)
        if editor is None:
            raise TypeError(
                "this model does not support in-place field arithmetic"
            )
        editor(self.name, operation, value, unsafe=True)
        return self

    def __iadd__(self, value: Any) -> "FieldReference":
        return self._edit("add", value)

    def __isub__(self, value: Any) -> "FieldReference":
        return self._edit("add", -value)

    def __imul__(self, value: Any) -> "FieldReference":
        return self._edit("multiply", value)

    def __repr__(self) -> str:
        return f"FieldReference({self.name!r})"


class FieldCollection:
    """Dictionary-like access to Python-owned StatePool fields."""

    def __init__(self, owner: Any) -> None:
        object.__setattr__(self, "owner", owner)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "owner" or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        # ``fields.temperature += 1`` writes the FieldReference returned by
        # __iadd__ back to this attribute. The numerical edit already happened.
        if isinstance(value, FieldReference) and value.name == name:
            return
        self[name].set(value)

    def __getitem__(self, name: str) -> FieldReference:
        return FieldReference(self, name)

    def __getattr__(self, name: str) -> FieldReference:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def create(
        self,
        name: str,
        *,
        dims: Sequence[str | int] = (),
        dtype: str = "float64",
        units: str = "1",
        initial: Any = 0.0,
        standard_name: str | None = None,
        intent: str = "inout",
        category: str = "plugin_state",
        aliases: Sequence[str] = (),
        restart: bool = True,
        history: bool = False,
        writable: bool = True,
    ) -> Any:
        """Allocate and register a field collectively on the live model."""

        spec = VariableSpec(
            name=str(name),
            standard_name=standard_name or str(name),
            dtype=str(dtype),
            dimensions=_dimensions(dims),
            units=str(units),
            intent=str(intent),
            category=str(category),
            aliases=tuple(str(item) for item in aliases),
            restart=bool(restart),
            history=bool(history),
            writable=bool(writable),
        )
        return self.owner.define_variable(spec, initial=initial)

    define = create

    def delete(self, name: str) -> Any:
        """Delete an unused dynamic field collectively on every MPI rank."""

        return self.owner.delete_variable(str(name))

    remove = delete

    def get(
        self, name: str, *, rank: int | str | None = None
    ) -> Any:
        return self[name].get(rank=rank)

    def set(
        self,
        name: str,
        value: Any,
        *,
        rank: int | str | None = None,
        unsafe: bool = False,
    ) -> Any:
        return self[name].set(value, rank=rank, unsafe=unsafe)

    def stats(
        self, name: str, *, rank: int | str | None = None
    ) -> Any:
        return self[name].stats(rank=rank)


class PhaseReference:
    """One named model phase exposed as an explicit runnable object."""

    def __init__(self, phases: "PhaseCollection", name: str) -> None:
        self._phases = phases
        self.name = str(name)

    def run(self) -> Any:
        return self._phases.run(self.name)

    __call__ = run

    def __repr__(self) -> str:
        return f"PhaseReference({self.name!r})"


class PhaseCollection:
    """Dictionary-like access to model phases."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def __getitem__(self, name: str) -> PhaseReference:
        return PhaseReference(self, name)

    def __getattr__(self, name: str) -> PhaseReference:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def run(self, name: str) -> Any:
        return self.owner.run_phase(str(name))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.owner.phase_names)

    def prepare(self) -> Any:
        """Run the complete initial-step preparation boundary."""

        return self.owner.prepare_initial_step()


class SchemeReference:
    """Convenient controls for one scheme occurrence or process name."""

    def __init__(
        self,
        physics: "PhysicsCollection",
        name: str,
        *,
        group: str | None = None,
    ) -> None:
        self._physics = physics
        self.name = str(name)
        self.group = _group(group)

    def run(self) -> Any:
        return self._physics.run(self.name, group=self.group)

    __call__ = run

    def enable(self) -> Any:
        return self._physics._set_scheme_enabled(
            self.name, True, group=self.group
        )

    def disable(self) -> Any:
        return self._physics._set_scheme_enabled(
            self.name, False, group=self.group
        )

    @property
    def enabled(self) -> bool:
        rows = self._physics.describe(self.group)
        matches = [
            row for row in rows
            if row.get("name") == self.name or row.get("key") == self.name
        ]
        if len(matches) != 1:
            raise KeyError(
                f"scheme {self.name!r} is absent or ambiguous; "
                "select it with physics.scheme(name, group=...)"
            )
        return bool(matches[0]["enabled"])

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._physics._set_scheme_enabled(
            self.name, bool(value), group=self.group
        )

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        to_group: str | None = None,
    ) -> Any:
        return self._physics._move_scheme(
            self.name,
            before=before,
            after=after,
            group=self.group,
            to_group=_group(to_group),
        )

    def __repr__(self) -> str:
        return (
            f"SchemeReference({self.name!r}, group={self.group!r})"
        )


class InstalledPythonProcess(SchemeReference):
    """User-facing handle for one installed Notebook Python callback."""

    def __init__(
        self,
        physics: "PhysicsCollection",
        spec: PythonProcessSpec,
    ) -> None:
        super().__init__(physics, spec.name, group=spec.group)
        self.spec = spec

    @property
    def payload_hash(self) -> str:
        return self.spec.payload_hash

    @property
    def reads(self) -> tuple[str, ...]:
        return self.spec.reads

    @property
    def writes(self) -> tuple[str, ...]:
        return self.spec.writes

    @property
    def transactional(self) -> bool:
        return self.spec.transactional

    def remove(self) -> Any:
        return self._physics.remove_python(self.name)

    def __repr__(self) -> str:
        return (
            f"InstalledPythonProcess({self.name!r}, "
            f"group={self.group!r})"
        )


class PhysicsCollection:
    """Install, place, and control physics without exposing protocol objects."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def __getitem__(self, name: str) -> SchemeReference:
        return SchemeReference(self, name)

    def __getattr__(self, name: str) -> SchemeReference:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def scheme(
        self, name: str, *, group: str | None = None
    ) -> SchemeReference:
        return SchemeReference(self, name, group=group)

    def install(
        self,
        source: str | Path | PhysicsPluginSpec,
        *,
        process: str | None = None,
        name: str | None = None,
        group: str | None = None,
        before: str | None = None,
        after: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        variables: Sequence[VariableSpec | Mapping[str, Any]] = (),
        project_root: str | Path | None = None,
        effective: str = "now",
        enabled: bool = True,
    ) -> Any:
        """Build/load a device and insert it into the active suite plan.

        Calling this high-level method is itself the explicit opt-in to a
        scientific sequence change. The lower-level ``install_physics`` API
        retains its ``unsafe=True`` guard for serialized control paths.
        """

        spec = _physics_plugin_spec(
            source,
            process=process,
            name=name,
            group=group,
            before=before,
            after=after,
            variables=variables,
            project_root=project_root,
            enabled=enabled,
        )
        return self.owner.install_physics(
            spec,
            initial_values=dict(inputs or {}),
            effective=effective,
            unsafe=True,
        )

    add = install

    def install_python(
        self,
        function: Any,
        *,
        name: str | None = None,
        group: str = PHYSICS_BEFORE_COUPLER,
        before: str | None = None,
        after: str | None = None,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        enabled: bool = True,
        transactional: bool = True,
        unsafe: bool = False,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> InstalledPythonProcess:
        """Install a trusted Notebook callback into the live suite plan."""

        spec = PythonProcessSpec.from_callable(
            function,
            name=name,
            group=_group(group) or str(group),
            before=before,
            after=after,
            reads=reads,
            writes=writes,
            enabled=enabled,
            transactional=transactional,
            max_payload_bytes=max_payload_bytes,
        )
        self.owner.install_python_process(
            spec,
            unsafe=bool(unsafe),
        )
        return InstalledPythonProcess(self, spec)

    def remove_python(self, name: str) -> Any:
        """Remove a previously installed Notebook Python process."""

        return self.owner.remove_python_process(str(name))

    def run(
        self,
        name: str,
        *,
        group: str | None = None,
    ) -> Any:
        return self.owner.run_scheme(str(name), group=_group(group))

    def activate(self, name: str) -> Any:
        return self.owner.activate_physics(str(name), unsafe=True)

    def deactivate(self, name: str) -> Any:
        return self.owner.deactivate_physics(str(name), unsafe=True)

    def describe(self, group: str | None = None) -> Any:
        return self.owner.scheme_plan.describe(_group(group))

    @property
    def sequence_safe(self) -> bool:
        return bool(self.owner.scheme_plan.sequence_safe)

    def reset(self) -> Any:
        """Restore the suite XML's validated scheme order and enable state."""

        resetter = getattr(self.owner, "reset_scheme_plan", None)
        if resetter is not None:
            return resetter()
        return self.owner.scheme_plan.reset()

    def _set_scheme_enabled(
        self,
        name: str,
        enabled: bool,
        *,
        group: str | None,
    ) -> Any:
        setter = getattr(self.owner, "set_scheme_enabled", None)
        if setter is not None:
            return setter(
                str(name),
                bool(enabled),
                group=_group(group),
                unsafe=True,
            )
        plan = self.owner.scheme_plan
        if enabled:
            return plan.enable(str(name), group=_group(group))
        return plan.disable(
            str(name), group=_group(group), unsafe=True
        )

    def _move_scheme(
        self,
        name: str,
        *,
        before: str | None,
        after: str | None,
        group: str | None,
        to_group: str | None,
    ) -> Any:
        mover = getattr(self.owner, "move_scheme", None)
        if mover is not None:
            return mover(
                str(name),
                before=before,
                after=after,
                group=_group(group),
                to_group=_group(to_group),
                unsafe=True,
            )
        return self.owner.scheme_plan.move(
            str(name),
            before=before,
            after=after,
            group=_group(group),
            to_group=_group(to_group),
            unsafe=True,
        )


class PlanPhaseReference:
    """Append one phase call to a :class:`PlanBuilder`."""

    def __init__(self, plan: "PlanBuilder", name: str) -> None:
        self._plan = plan
        self.name = str(name)

    def run(self) -> "PlanBuilder":
        return self._plan._append(RunPhase(self.name))

    __call__ = run


class PlanPhaseCollection:
    def __init__(self, plan: "PlanBuilder") -> None:
        self._plan = plan

    def __getitem__(self, name: str) -> PlanPhaseReference:
        return PlanPhaseReference(self._plan, name)

    def __getattr__(self, name: str) -> PlanPhaseReference:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def run(self, name: str) -> "PlanBuilder":
        return self[name].run()

    def prepare(self) -> "PlanBuilder":
        return self._plan._append(PrepareInitialStep())


class PlanFieldCollection:
    def __init__(self, plan: "PlanBuilder") -> None:
        self._plan = plan

    def create(
        self,
        name: str,
        *,
        dims: Sequence[str | int] = (),
        dtype: str = "float64",
        units: str = "1",
        initial: Any = 0.0,
        standard_name: str | None = None,
        intent: str = "inout",
        category: str = "plugin_state",
        aliases: Sequence[str] = (),
        restart: bool = True,
        history: bool = False,
        writable: bool = True,
    ) -> "PlanBuilder":
        spec = VariableSpec(
            name=str(name),
            standard_name=standard_name or str(name),
            dtype=str(dtype),
            dimensions=_dimensions(dims),
            units=str(units),
            intent=str(intent),
            category=str(category),
            aliases=tuple(str(item) for item in aliases),
            restart=bool(restart),
            history=bool(history),
            writable=bool(writable),
        )
        return self._plan._append(
            DefineVariable(spec, initial_value=initial)
        )

    define = create

    def edit(
        self,
        name: str,
        operation: str,
        value: float,
        *,
        unsafe: bool = False,
    ) -> "PlanBuilder":
        return self._plan._append(
            FieldEdit(str(name), str(operation), value, unsafe=unsafe)
        )


class PlanSchemeReference:
    def __init__(
        self,
        plan: "PlanBuilder",
        name: str,
        *,
        group: str | None = None,
    ) -> None:
        self._plan = plan
        self.name = str(name)
        self.group = _group(group)

    def run(self) -> "PlanBuilder":
        return self._plan._append(RunScheme(self.name, self.group))

    __call__ = run

    def enable(self) -> "PlanBuilder":
        return self._plan._append(
            SetSchemeEnabled(self.name, True, self.group)
        )

    def disable(self) -> "PlanBuilder":
        return self._plan._append(
            SetSchemeEnabled(self.name, False, self.group)
        )

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        to_group: str | None = None,
    ) -> "PlanBuilder":
        return self._plan._append(
            MoveScheme(
                self.name,
                before=before,
                after=after,
                to_group=_group(to_group),
            )
        )


class PlanPhysicsCollection:
    def __init__(self, plan: "PlanBuilder") -> None:
        self._plan = plan

    def __getitem__(self, name: str) -> PlanSchemeReference:
        return PlanSchemeReference(self._plan, name)

    def __getattr__(self, name: str) -> PlanSchemeReference:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]

    def scheme(
        self, name: str, *, group: str | None = None
    ) -> PlanSchemeReference:
        return PlanSchemeReference(self._plan, name, group=group)

    def group(self, name: str) -> "PlanBuilder":
        return self._plan._append(RunSchemeGroup(_group(name) or str(name)))

    def install(
        self,
        source: str | Path | PhysicsPluginSpec,
        *,
        process: str | None = None,
        name: str | None = None,
        group: str | None = None,
        before: str | None = None,
        after: str | None = None,
        inputs: Mapping[str, Any] | None = None,
        variables: Sequence[VariableSpec | Mapping[str, Any]] = (),
        project_root: str | Path | None = None,
        effective: str = "now",
        enabled: bool = True,
    ) -> "PlanBuilder":
        spec = _physics_plugin_spec(
            source,
            process=process,
            name=name,
            group=group,
            before=before,
            after=after,
            variables=variables,
            project_root=project_root,
            enabled=enabled,
        )
        return self._plan._append(
            InstallPhysics(
                spec,
                initial_values=dict(inputs or {}),
                effective=effective,
            )
        )

    def install_python(
        self,
        function: Any,
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
    ) -> "PlanBuilder":
        spec = PythonProcessSpec.from_callable(
            function,
            name=name,
            group=_group(group) or str(group),
            before=before,
            after=after,
            reads=reads,
            writes=writes,
            enabled=enabled,
            transactional=transactional,
            max_payload_bytes=max_payload_bytes,
        )
        return self._plan._append(InstallPythonProcess(spec))

    def remove_python(self, name: str) -> "PlanBuilder":
        return self._plan._append(RemovePythonProcess(str(name)))


class PlanBuilder:
    """Pythonic builder for one serializable Dask action segment.

    ``experimental=True`` is the explicit opt-in required for standalone
    phase/scheme calls or order changes that bypass the validated full-step
    dependency sequence.
    """

    def __init__(self, name: str, *, experimental: bool = False) -> None:
        self.name = str(name)
        self.experimental = bool(experimental)
        self._actions: list[Action] = []
        self.fields = PlanFieldCollection(self)
        self.physics = PlanPhysicsCollection(self)
        self.phases = PlanPhaseCollection(self)

    def _append(self, action: Action) -> "PlanBuilder":
        self._actions.append(action)
        return self

    def prepare(self) -> "PlanBuilder":
        return self._append(PrepareInitialStep())

    def step(self, count: int = 1) -> "PlanBuilder":
        return self._append(RunSteps(count))

    steps = step

    def observe(
        self,
        *fields: str,
        statistics: Sequence[str] = ("min", "max", "mean"),
    ) -> "PlanBuilder":
        return self._append(
            ObserveFields(
                tuple(str(name) for name in fields),
                tuple(str(name) for name in statistics),
            )
        )

    def build(self) -> SegmentPlan:
        return SegmentPlan(
            self.name,
            actions=tuple(self._actions),
            unsafe=self.experimental,
        )

    def as_dict(self) -> dict[str, Any]:
        return self.build().as_dict()

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(self._actions)


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Typed snapshot of one live persistent model's control state."""

    name: str
    running: bool
    ranks: int
    step: int
    native_calls: int
    mpi_launch_count: int
    worker_host: str
    worker_pid: int
    launch_mode: str | None
    pbs_job_id: str | None
    outer_pbs_job_id: str | None
    field_count: int
    snapshot_transport: str
    run_dir: Path
    history_dir: Path
    log_path: Path
    details: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelStatus":
        payload = dict(values)
        return cls(
            name=str(payload["name"]),
            running=bool(payload["running"]),
            ranks=int(payload["ranks"]),
            step=int(payload["step"]),
            native_calls=int(payload["native_calls"]),
            mpi_launch_count=int(payload["mpi_launch_count"]),
            worker_host=str(payload["worker_host"]),
            worker_pid=int(payload["worker_pid"]),
            launch_mode=(
                None
                if payload.get("launch_mode") is None
                else str(payload["launch_mode"])
            ),
            pbs_job_id=(
                None
                if payload.get("pbs_job_id") is None
                else str(payload["pbs_job_id"])
            ),
            outer_pbs_job_id=(
                None
                if payload.get("outer_pbs_job_id") is None
                else str(payload["outer_pbs_job_id"])
            ),
            field_count=int(payload["field_count"]),
            snapshot_transport=str(payload["snapshot_transport"]),
            run_dir=Path(payload["run_dir"]),
            history_dir=Path(payload["history_dir"]),
            log_path=Path(payload["log_path"]),
            details=payload,
        )


@dataclass(frozen=True, slots=True)
class SavedCheckpoint:
    """Durable checkpoint created from a live persistent model."""

    path: Path
    step: int
    native_calls: int
    mpi_launch_count: int

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SavedCheckpoint":
        return cls(
            path=Path(values["checkpoint_dir"]),
            step=int(values["step"]),
            native_calls=int(values["native_calls"]),
            mpi_launch_count=int(values["mpi_launch_count"]),
        )


class BlockingModel:
    """Blocking view of a Future-returning persistent model controller."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.fields = FieldCollection(self)
        self.physics = PhysicsCollection(self)
        self.phases = PhaseCollection(self)

    @property
    def submit(self) -> Any:
        """Return the underlying asynchronous Dask model proxy."""

        return self._model

    @property
    def sync(self) -> "BlockingModel":
        return self

    @property
    def status(self) -> ModelStatus:
        return ModelStatus.from_mapping(_wait(self._model.describe()))

    @property
    def step_count(self) -> int:
        return self.status.step

    @property
    def mpi_launch_count(self) -> int:
        return self.status.mpi_launch_count

    def advance(self, steps: int = 1) -> "BlockingModel":
        """Advance complete model steps and keep the same live MPI model."""

        if steps < 0:
            raise ValueError("steps must be non-negative")
        _wait(self._model.step(int(steps)))
        return self

    def save(self, path: str | Path | None = None) -> SavedCheckpoint:
        """Write a durable checkpoint and return typed checkpoint metadata."""

        result = _wait(self._model.checkpoint(path))
        return SavedCheckpoint.from_mapping(result)

    def snapshot(self) -> Any:
        """Capture an immutable in-memory checkpoint without disk I/O."""

        return _wait(self._model.memory_checkpoint())

    def execute(
        self, plan: SegmentPlan | PlanBuilder
    ) -> Mapping[str, Any]:
        """Execute a Pythonic plan against the same live StatePool."""

        compiled = plan.build() if isinstance(plan, PlanBuilder) else plan
        if not isinstance(compiled, SegmentPlan):
            raise TypeError("plan must be SegmentPlan or PlanBuilder")
        return _wait(self._model.run_plan(compiled))

    def close(self) -> Any:
        return _wait(self._model.close())

    def __enter__(self) -> "BlockingModel":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not getattr(self._model, "_closed", False):
            self.close()

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._model, name)
        if not callable(value):
            return value

        def blocking_call(*args: Any, **kwargs: Any) -> Any:
            try:
                return _wait(value(*args, **kwargs))
            except BaseException:
                clear_failed_tail = getattr(
                    self._model, "clear_failed_tail", None
                )
                if callable(clear_failed_tail):
                    clear_failed_tail()
                raise

        blocking_call.__name__ = name
        return blocking_call

    def __repr__(self) -> str:
        return f"BlockingModel({self._model!r})"


class ModelGroup(Mapping[str, BlockingModel]):
    """Context-managed collection of independent persistent models."""

    def __init__(self, models: Mapping[str, BlockingModel]) -> None:
        self._models = dict(models)
        self._closed = False

    def __getitem__(self, name: str) -> BlockingModel:
        return self._models[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    @property
    def statuses(self) -> dict[str, ModelStatus]:
        return {name: model.status for name, model in self._models.items()}

    def advance(self, steps: int = 1) -> "ModelGroup":
        for model in self._models.values():
            model.advance(steps)
        return self

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        for model in self._models.values():
            try:
                model.close()
            except BaseException as exc:
                failures.append(exc)
        self._closed = True
        if failures:
            raise RuntimeError(
                f"failed to close {len(failures)} persistent model(s)"
            ) from failures[0]

    def __enter__(self) -> "ModelGroup":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _descriptor_path(source: str | Path) -> Path | None:
    path = Path(source).expanduser()
    if not path.exists():
        return None
    if path.is_dir():
        for name in ("device.json", "device.yaml"):
            candidate = path / name
            if candidate.is_file():
                return candidate
        return None
    return path


def _physics_plugin_spec(
    source: str | Path | PhysicsPluginSpec,
    *,
    process: str | None,
    name: str | None,
    group: str | None,
    before: str | None,
    after: str | None,
    variables: Sequence[VariableSpec | Mapping[str, Any]],
    project_root: str | Path | None,
    enabled: bool,
) -> PhysicsPluginSpec:
    if isinstance(source, PhysicsPluginSpec):
        if any(
            value is not None
            for value in (
                process,
                name,
                group,
                before,
                after,
                project_root,
            )
        ) or variables:
            raise ValueError(
                "a PhysicsPluginSpec cannot be combined with façade "
                "installation options"
            )
        return source

    normalized_group = _group(group)
    placement_requested = any(
        value is not None
        for value in (process, group, before, after)
    ) or not enabled
    placements: tuple[SchemePlacement, ...] = ()
    if placement_requested:
        selected_process = process or _infer_process(source, name)
        placements = (
            SchemePlacement(
                selected_process,
                group=normalized_group or PHYSICS_BEFORE_COUPLER,
                before=before,
                after=after,
                enabled=bool(enabled),
            ),
        )
    specs = tuple(
        item
        if isinstance(item, VariableSpec)
        else VariableSpec.from_mapping(item)
        for item in variables
    )
    return PhysicsPluginSpec(
        str(source),
        project_root=(
            None if project_root is None else str(Path(project_root))
        ),
        name=name,
        placements=placements,
        variables=specs,
    )


def _infer_process(
    source: str | Path,
    requested_name: str | None,
) -> str:
    path = _descriptor_path(source)
    if path is None:
        if requested_name is not None:
            return requested_name
        raise ValueError(
            "process= is required when the plugin descriptor is not "
            "locally readable"
        )
    payload = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else yaml.safe_load(path.read_text())
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"plugin descriptor must contain a mapping: {path}")
    processes = payload.get("processes", {})
    candidates = [
        str(process)
        for process, endpoint in processes.items()
        if ":" not in str(process) and str(endpoint) == "run"
    ]
    descriptor_name = str(payload.get("name", ""))
    for preferred in (requested_name, descriptor_name):
        if preferred and preferred in candidates:
            return preferred
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"cannot infer one run process from {path}; choose process= from "
        f"{candidates}"
    )


__all__ = [
    "BlockingModel",
    "FieldCollection",
    "FieldReference",
    "ModelGroup",
    "ModelStatus",
    "PhaseCollection",
    "PhaseReference",
    "PlanBuilder",
    "PlanFieldCollection",
    "PlanPhaseCollection",
    "PlanPhysicsCollection",
    "PlanSchemeReference",
    "PhysicsCollection",
    "SavedCheckpoint",
    "SchemeReference",
]
