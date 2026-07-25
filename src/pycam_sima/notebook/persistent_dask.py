"""Dask Actor control of one persistent configured-rank CAM model."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import re
import shutil
import socket
import threading
from typing import Any, Callable, Mapping

import numpy as np

from ..model import (
    ActivatePhysics,
    BlockingModel,
    CheckpointBundle,
    CCPPSuitePlan,
    DeactivatePhysics,
    DefineVariable,
    FieldEdit,
    FieldCollection,
    InstallPhysics,
    ModelOptions,
    MoveScheme,
    ObserveFields,
    PrepareInitialStep,
    RunPhase,
    RunScheme,
    RunSchemeGroup,
    RunSteps,
    SegmentPlan,
    SetSchemeEnabled,
    PhaseCollection,
    PhysicsPluginSpec,
    PhysicsCollection,
    VariableSpec,
)
from .session import NotebookSession


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True, slots=True)
class PersistentDaskRequest:
    """Serializable configuration used to construct a worker-local Actor."""

    name: str
    config: str
    initial_run_dir: str
    run_root: str
    library: str
    environment_script: str
    python_executable: str
    log_dir: str
    ranks: int
    launch_mode: str
    pbs_account: str
    pbs_queue: str
    pbs_walltime: str
    startup_timeout: float
    request_timeout: float
    options: Mapping[str, Any]
    scheme_plan: Mapping[str, Any]
    execution_mode: str
    parent_name: str | None = None
    startup_plan: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "persistent session name may contain only letters, digits, "
                "dot, dash, and underscore"
            )
        if self.ranks <= 0:
            raise ValueError("ranks must be positive")
        if self.launch_mode not in {"auto", "local", "pbs"}:
            raise ValueError("launch_mode must be auto, local, or pbs")
        if self.execution_mode not in {"pbs", "allocation"}:
            raise ValueError("execution_mode must be pbs or allocation")
        if self.execution_mode == "allocation" and self.launch_mode != "local":
            raise ValueError(
                "persistent allocation execution requires launch_mode='local'"
            )
        if self.parent_name is not None and not _SAFE_NAME.fullmatch(
            self.parent_name
        ):
            raise ValueError(
                "persistent parent name may contain only letters, digits, "
                "dot, dash, and underscore"
            )
        if self.startup_plan is not None:
            plan = SegmentPlan.from_mapping(self.startup_plan)
            if plan.name != self.name:
                raise ValueError(
                    "persistent startup plan name must match session name"
                )


class PersistentCAMActor:
    """Worker-side Dask Actor that owns one live :class:`NotebookSession`.

    Construction launches MPI exactly once. Every later method reuses the same
    MPI ranks, Python-owned StatePool, model clock, and loaded Fortran devices.
    """

    def __init__(
        self,
        request: PersistentDaskRequest,
        snapshot: CheckpointBundle | None = None,
        session_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._request = request
        self._guard = threading.RLock()
        self._closed = False
        self._mpi_launch_count = 0
        self._allocation_lock: Any = None
        self._session: Any = None
        self._restored_from_memory = False
        self._source_snapshot_nbytes = 0
        self._branch_root = Path(request.run_root) / request.name
        self._run_dir = self._branch_root / "run"
        self._history_dir = self._branch_root / "history"
        self._checkpoint_root = self._branch_root / "checkpoints"
        self._log_path = Path(request.log_dir) / (
            f"pycam_dask_persistent_{request.name}.log"
        )

        try:
            self._acquire_allocation()
            self._prepare_run_directory()
            factory = session_factory or NotebookSession
            options = ModelOptions(**dict(request.options))
            scheme_plan = CCPPSuitePlan.from_payload(request.scheme_plan)
            self._session = factory(
                request.config,
                run_dir=self._run_dir,
                library=request.library,
                history_dir=self._history_dir,
                ranks=request.ranks,
                env_script=request.environment_script,
                launch_mode=request.launch_mode,
                pbs_account=request.pbs_account,
                pbs_queue=request.pbs_queue,
                pbs_walltime=request.pbs_walltime,
                python_executable=request.python_executable,
                startup_timeout=request.startup_timeout,
                request_timeout=request.request_timeout,
                log_path=self._log_path,
                options=options,
                scheme_plan=scheme_plan,
            )
            self._session.start()
            self._mpi_launch_count = 1
            if snapshot is not None:
                if not isinstance(snapshot, CheckpointBundle):
                    raise TypeError("persistent parent snapshot must be CheckpointBundle")
                self._session.restore_memory_checkpoint(snapshot)
                self._restored_from_memory = True
                self._source_snapshot_nbytes = snapshot.nbytes
            if request.startup_plan is not None:
                self.run_plan(request.startup_plan)
        except BaseException:
            if self._session is not None:
                try:
                    self._session.close()
                except BaseException:
                    pass
            self._release_allocation()
            raise

    def describe(self) -> dict[str, Any]:
        with self._guard:
            self._ensure_open()
            return {
                "name": self._request.name,
                "parent_name": self._request.parent_name,
                "running": bool(self._session.running),
                "worker_host": socket.gethostname(),
                "worker_pid": os.getpid(),
                "ranks": int(self._session.ranks),
                "step": int(self._session.current_step),
                "native_calls": int(self._session.native_calls),
                "mpi_launch_count": self._mpi_launch_count,
                "snapshot_transport": (
                    "memory" if self._restored_from_memory else "initialization"
                ),
                "source_snapshot_nbytes": self._source_snapshot_nbytes,
                "startup_plan": (
                    None
                    if self._request.startup_plan is None
                    else dict(self._request.startup_plan)
                ),
                "launch_mode": self._session.launch_mode_used,
                "pbs_job_id": self._session.job_id,
                "outer_pbs_job_id": os.environ.get("PBS_JOBID"),
                "run_dir": str(self._run_dir),
                "history_dir": str(self._history_dir),
                "log_path": str(self._log_path),
                "field_count": len(self._session.field_names),
                "phase_names": tuple(self._session.phase_names),
                "phase_status": dict(self._session.phase_status),
                "scheme_names": tuple(self._session.scheme_names),
                "scheme_status": dict(self._session.scheme_status),
                "plugins": tuple(
                    getattr(self._session, "physics_plugins", ())
                ),
            }

    def prepare_initial_step(self) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._session.prepare_initial_step()
            return self._command_status()

    def step(self, count: int = 1) -> dict[str, Any]:
        with self._guard:
            self._ensure_open()
            current = self._session.step(count)
            status = self._command_status()
            status["step"] = int(current)
            return status

    def run_phase(self, name: str) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._session.run_phase(name)
            return self._command_status()

    def run_scheme(self, name: str, group: str | None = None) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._session.run_scheme(name, group=group)
            return self._command_status()

    def run_scheme_group(self, group: str) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._session.run_scheme_group(group)
            return self._command_status()

    def define_variable(
        self, payload: Mapping[str, Any], initial: Any = 0.0
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            spec = VariableSpec.from_mapping(payload)
            metadata = self._session.define_variable(spec, initial=initial)
            return {
                **self._command_status(),
                "variable": dict(metadata),
            }

    def install_physics(
        self,
        payload: Mapping[str, Any],
        initial_values: Mapping[str, Any] | None = None,
        effective: str = "now",
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            plugin = self._session.install_physics(
                PhysicsPluginSpec.from_mapping(payload),
                initial_values=initial_values,
                effective=effective,
                unsafe=unsafe,
            )
            return {
                **self._command_status(),
                "plugin": dict(plugin),
            }

    def activate_physics(
        self, name: str, unsafe: bool = False
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            plugin = self._session.activate_physics(name, unsafe=unsafe)
            return {**self._command_status(), "plugin": dict(plugin)}

    def deactivate_physics(
        self, name: str, unsafe: bool = False
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            plugin = self._session.deactivate_physics(name, unsafe=unsafe)
            return {**self._command_status(), "plugin": dict(plugin)}

    def get_field(self, name: str, rank: int | str = 0) -> Any:
        with self._guard:
            self._ensure_open()
            return self._session.get_field(name, rank=rank)

    def get_field_stats(self, name: str, rank: int | str = 0) -> Any:
        with self._guard:
            self._ensure_open()
            return self._session.get_field_stats(name, rank=rank)

    def set_field(
        self,
        name: str,
        value: Any,
        rank: int | str = 0,
        unsafe: bool = False,
    ) -> dict[str, Any]:
        with self._guard:
            self._ensure_open()
            self._session.set_field(
                name,
                value,
                rank=rank,
                unsafe=unsafe,
            )
            return {
                "field": name,
                "rank": rank,
                "step": int(self._session.current_step),
                "mpi_launch_count": self._mpi_launch_count,
            }

    def set_scheme_enabled(
        self,
        name: str,
        enabled: bool,
        group: str | None = None,
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            if enabled:
                self._session.scheme_plan.enable(name, group=group)
            else:
                self._session.scheme_plan.disable(
                    name,
                    group=group,
                    unsafe=unsafe,
                )
            return self._command_status()

    def move_scheme(
        self,
        name: str,
        before: str | None = None,
        after: str | None = None,
        group: str | None = None,
        to_group: str | None = None,
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._session.scheme_plan.move(
                name,
                before=before,
                after=after,
                group=group,
                to_group=to_group,
                unsafe=unsafe,
            )
            return self._command_status()

    def reset_scheme_plan(self) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._session.scheme_plan.reset()
            return self._command_status()

    def describe_scheme_plan(self, group: str | None = None) -> list[dict[str, object]]:
        with self._guard:
            self._ensure_open()
            return self._session.scheme_plan.describe(group)

    def run_plan(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Execute many granular actions without restarting MPI between them."""

        with self._guard:
            self._ensure_open()
            plan = SegmentPlan.from_mapping(payload)
            self._validate_plan(plan)
            trace: list[dict[str, Any]] = []
            encoded_actions = plan.as_dict()["actions"]
            for index, action in enumerate(plan.actions):
                step_before = int(self._session.current_step)
                calls_before = int(self._session.native_calls)
                observations: list[dict[str, Any]] = []
                if isinstance(action, PrepareInitialStep):
                    self._session.prepare_initial_step()
                elif isinstance(action, RunPhase):
                    self._session.run_phase(action.name)
                elif isinstance(action, RunScheme):
                    self._session.run_scheme(action.name, group=action.group)
                elif isinstance(action, RunSchemeGroup):
                    self._session.run_scheme_group(action.group)
                elif isinstance(action, RunSteps):
                    if action.count:
                        self._session.step(action.count)
                elif isinstance(action, SetSchemeEnabled):
                    if action.enabled:
                        self._session.scheme_plan.enable(
                            action.name, group=action.group
                        )
                    else:
                        self._session.scheme_plan.disable(
                            action.name,
                            group=action.group,
                            unsafe=True,
                        )
                elif isinstance(action, MoveScheme):
                    self._session.scheme_plan.move(
                        action.name,
                        before=action.before,
                        after=action.after,
                        to_group=action.to_group,
                        unsafe=True,
                    )
                elif isinstance(action, FieldEdit):
                    self._session.edit_field(
                        action.name,
                        action.operation,
                        action.value,
                        unsafe=action.unsafe,
                    )
                elif isinstance(action, ObserveFields):
                    observations = self._observe_fields(action)
                elif isinstance(action, DefineVariable):
                    self._session.define_variable(
                        action.spec, initial=action.initial_value
                    )
                elif isinstance(action, InstallPhysics):
                    self._session.install_physics(
                        action.plugin,
                        initial_values=action.initial_values,
                        effective=action.effective,
                        unsafe=True,
                    )
                elif isinstance(action, ActivatePhysics):
                    self._session.activate_physics(
                        action.name, unsafe=True
                    )
                elif isinstance(action, DeactivatePhysics):
                    self._session.deactivate_physics(
                        action.name, unsafe=True
                    )
                else:  # pragma: no cover - SegmentPlan validates construction.
                    raise TypeError(
                        f"unsupported persistent action {type(action).__name__}"
                    )

                record: dict[str, Any] = {
                    "index": index,
                    "type": encoded_actions[index]["type"],
                    "action": encoded_actions[index],
                    "step_before": step_before,
                    "step_after": int(self._session.current_step),
                    "last_phase": self._session.phase_status.get("last_phase"),
                    "last_scheme": self._session.scheme_status.get("last_scheme"),
                    "last_scheme_group": self._session.scheme_status.get(
                        "last_scheme_group"
                    ),
                    "native_calls_delta": (
                        int(self._session.native_calls) - calls_before
                    ),
                }
                if observations:
                    record["observations"] = observations
                trace.append(record)
            return {
                "name": plan.name,
                "step": int(self._session.current_step),
                "native_calls": int(self._session.native_calls),
                "mpi_launch_count": self._mpi_launch_count,
                "action_trace": trace,
                "segment_plan": plan.as_dict(),
            }

    def checkpoint(self, path: str | None = None) -> dict[str, Any]:
        """Persist the current live state without stopping the Actor."""

        with self._guard:
            self._ensure_open()
            if path is None:
                target = (
                    self._checkpoint_root / f"step-{self._session.current_step:06d}"
                )
            else:
                target = Path(path).resolve()
            checkpoint = self._session.write_checkpoint(target)
            return {
                "checkpoint_dir": str(checkpoint),
                "step": int(self._session.current_step),
                "native_calls": int(self._session.native_calls),
                "mpi_launch_count": self._mpi_launch_count,
            }

    def memory_checkpoint(self) -> CheckpointBundle:
        """Return a bit-preserving snapshot without writing checkpoint files."""

        with self._guard:
            self._ensure_open()
            return self._session.memory_checkpoint()

    def close(self) -> dict[str, Any]:
        with self._guard:
            if self._closed:
                return {
                    "closed": True,
                    "mpi_launch_count": self._mpi_launch_count,
                }
            error: BaseException | None = None
            try:
                if self._session is not None:
                    self._session.close()
            except BaseException as exc:
                error = exc
            finally:
                self._closed = True
                self._release_allocation()
            if error is not None:
                raise error
            return {
                "closed": True,
                "mpi_launch_count": self._mpi_launch_count,
                "run_dir": str(self._run_dir),
            }

    def _validate_plan(self, plan: SegmentPlan) -> None:
        candidate = CCPPSuitePlan.from_payload(self._session.scheme_status["plan"])
        scheme_groups = candidate.group_names
        planned_processes: set[str] = set()
        planned_plugins = {
            str(item["name"])
            for item in getattr(self._session, "physics_plugins", ())
        }
        planned_fields = set(self._session.field_names)
        for action in plan.actions:
            if isinstance(action, RunPhase):
                if action.name not in self._session.phase_names:
                    raise ValueError(
                        f"unknown model phase {action.name!r}; choose one of "
                        f"{self._session.phase_names}"
                    )
                if not plan.unsafe:
                    raise ValueError(
                        "run_phase actions require SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, RunScheme):
                if action.name not in planned_processes:
                    candidate.scheme(action.name, group=action.group)
                if not plan.unsafe:
                    raise ValueError(
                        "run_scheme actions require SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, RunSchemeGroup):
                if action.group not in scheme_groups:
                    raise ValueError(
                        f"unknown scheme group {action.group!r}; choose one of "
                        f"{scheme_groups}"
                    )
                if not plan.unsafe:
                    raise ValueError(
                        "run_scheme_group actions require SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, SetSchemeEnabled):
                candidate.scheme(action.name, group=action.group)
                if not action.enabled and not plan.unsafe:
                    raise ValueError(
                        "disabling a scheme requires SegmentPlan(unsafe=True)"
                    )
                if action.enabled:
                    candidate.enable(action.name, group=action.group)
                else:
                    candidate.disable(
                        action.name,
                        group=action.group,
                        unsafe=True,
                    )
            elif isinstance(action, MoveScheme):
                candidate.scheme(action.name)
                if action.before is not None:
                    candidate.scheme(action.before)
                if action.after is not None:
                    candidate.scheme(action.after)
                if (
                    action.to_group is not None
                    and action.to_group not in scheme_groups
                ):
                    raise ValueError(
                        f"unknown destination group {action.to_group!r}; "
                        f"choose one of {scheme_groups}"
                    )
                if not plan.unsafe:
                    raise ValueError(
                        "moving a scheme requires SegmentPlan(unsafe=True)"
                    )
                candidate.move(
                    action.name,
                    before=action.before,
                    after=action.after,
                    to_group=action.to_group,
                    unsafe=True,
                )
            elif isinstance(action, FieldEdit):
                if action.name not in planned_fields:
                    raise KeyError(f"unknown CAM-SIMA field: {action.name}")
                info = self._session.field_info(action.name)
                if action.unsafe and not plan.unsafe:
                    raise ValueError(
                        "unsafe field edits require SegmentPlan(unsafe=True)"
                    )
                if (
                    not bool(info.get("writable", True))
                    and not action.unsafe
                ):
                    raise ValueError(
                        f"field {action.name!r} is read-only after initialization"
                    )
            elif isinstance(action, DefineVariable):
                if action.spec.name in planned_fields:
                    raise ValueError(
                        f"duplicate state field {action.spec.name!r}"
                    )
                planned_fields.add(action.spec.name)
                planned_fields.update(action.spec.aliases)
            elif isinstance(action, InstallPhysics):
                if not plan.unsafe:
                    raise ValueError(
                        "install_physics actions require "
                        "SegmentPlan(unsafe=True)"
                    )
                for variable in action.plugin.variables:
                    planned_fields.add(variable.name)
                    planned_fields.update(variable.aliases)
                planned_processes.update(
                    item.process for item in action.plugin.placements
                )
                if action.plugin.name is not None:
                    planned_plugins.add(action.plugin.name)
            elif isinstance(action, (ActivatePhysics, DeactivatePhysics)):
                if not plan.unsafe:
                    raise ValueError(
                        f"{type(action).__name__} requires "
                        "SegmentPlan(unsafe=True)"
                    )
                if action.name not in planned_plugins:
                    raise ValueError(
                        f"unknown planned physics plugin {action.name!r}"
                    )
            elif isinstance(action, ObserveFields):
                for name in action.fields:
                    if name not in planned_fields:
                        self._session.field_info(name)

    def _observe_fields(self, action: ObserveFields) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for name in action.fields:
            ranks = [
                dict(row) for row in self._session.get_field_stats(name, rank="all")
            ]
            counts = [int(np.prod(row["shape"], dtype=np.int64)) for row in ranks]
            total = sum(counts)
            global_statistics: dict[str, float] = {}
            if "min" in action.statistics:
                global_statistics["min"] = min(float(row["min"]) for row in ranks)
            if "max" in action.statistics:
                global_statistics["max"] = max(float(row["max"]) for row in ranks)
            if "mean" in action.statistics:
                global_statistics["mean"] = (
                    sum(float(row["mean"]) * count for row, count in zip(ranks, counts))
                    / total
                )
            observations.append(
                {
                    "field": name,
                    "global": global_statistics,
                    "ranks": ranks,
                }
            )
        return observations

    def _prepare_run_directory(self) -> None:
        if self._branch_root.exists():
            raise FileExistsError(
                f"refusing to replace persistent Dask session: {self._branch_root}"
            )
        self._run_dir.mkdir(parents=True)
        shutil.copy2(
            Path(self._request.initial_run_dir) / "atm_in",
            self._run_dir / "atm_in",
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def _acquire_allocation(self) -> None:
        if self._request.execution_mode != "allocation":
            return
        if not os.environ.get("PBS_JOBID"):
            raise RuntimeError(
                "persistent allocation mode requires an active PBS allocation"
            )
        root = Path(self._request.run_root)
        root.mkdir(parents=True, exist_ok=True)
        handle = (root / ".allocation-mpi.lock").open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise RuntimeError(
                "the allocation already has an active full-node MPI segment "
                "or persistent session"
            ) from exc
        self._allocation_lock = handle

    def _release_allocation(self) -> None:
        if self._allocation_lock is None:
            return
        try:
            fcntl.flock(self._allocation_lock, fcntl.LOCK_UN)
        finally:
            self._allocation_lock.close()
            self._allocation_lock = None

    def _ensure_open(self) -> None:
        if self._closed or self._session is None:
            raise RuntimeError("persistent Dask CAM session is closed")

    def _command_status(self) -> dict[str, Any]:
        return {
            "step": int(self._session.current_step),
            "native_calls": int(self._session.native_calls),
            "mpi_launch_count": self._mpi_launch_count,
            "phase_status": dict(self._session.phase_status),
            "scheme_status": dict(self._session.scheme_status),
            "field_count": len(self._session.field_names),
            "plugins": tuple(
                getattr(self._session, "physics_plugins", ())
            ),
        }

    def __del__(self) -> None:
        """Best-effort cleanup if a Dask client drops its Actor reference."""

        try:
            if not self._closed and self._session is not None:
                self._session.close()
        except BaseException:
            pass
        finally:
            try:
                self._release_allocation()
            except BaseException:
                pass


class PersistentDaskSession:
    """Notebook-side proxy whose methods return Dask ``ActorFuture`` values."""

    def __init__(
        self,
        client: Any,
        actor: Any,
        actor_future: Any,
        *,
        worker: str,
        name: str,
    ) -> None:
        self.client = client
        self.actor = actor
        self.actor_future = actor_future
        self.worker = worker
        self.name = name
        self._closed = False
        self.fields = FieldCollection(self)
        self.phases = PhaseCollection(self)
        self.physics = PhysicsCollection(self)
        self._sync = BlockingModel(self)

    @property
    def sync(self) -> BlockingModel:
        """Blocking, Notebook-friendly view of this persistent model."""

        return self._sync

    @property
    def submit(self) -> "PersistentDaskSession":
        """Asynchronous Dask-native view whose calls return ActorFuture."""

        return self

    def describe(self) -> Any:
        return self._call("describe")

    status = describe

    def prepare_initial_step(self) -> Any:
        return self._call("prepare_initial_step")

    def step(self, count: int = 1) -> Any:
        return self._call("step", int(count))

    def run_phase(self, name: str) -> Any:
        return self._call("run_phase", str(name))

    def run_scheme(self, name: str, *, group: str | None = None) -> Any:
        return self._call("run_scheme", str(name), group)

    def run_scheme_group(self, group: str) -> Any:
        return self._call("run_scheme_group", str(group))

    def define_variable(
        self, spec: VariableSpec, *, initial: Any = 0.0
    ) -> Any:
        if not isinstance(spec, VariableSpec):
            raise TypeError("spec must be VariableSpec")
        return self._call("define_variable", spec.as_dict(), initial)

    def install_physics(
        self,
        spec: PhysicsPluginSpec,
        *,
        initial_values: Mapping[str, Any] | None = None,
        effective: str = "now",
        unsafe: bool = False,
    ) -> Any:
        if not isinstance(spec, PhysicsPluginSpec):
            raise TypeError("spec must be PhysicsPluginSpec")
        return self._call(
            "install_physics",
            spec.as_dict(),
            dict(initial_values or {}),
            effective,
            bool(unsafe),
        )

    def activate_physics(
        self, name: str, *, unsafe: bool = False
    ) -> Any:
        return self._call("activate_physics", str(name), bool(unsafe))

    def deactivate_physics(
        self, name: str, *, unsafe: bool = False
    ) -> Any:
        return self._call("deactivate_physics", str(name), bool(unsafe))

    def field(self, name: str, *, rank: int | str = 0) -> Any:
        return self._call("get_field", str(name), rank)

    get_field = field

    def field_stats(self, name: str, *, rank: int | str = 0) -> Any:
        return self._call("get_field_stats", str(name), rank)

    get_field_stats = field_stats

    def set_field(
        self,
        name: str,
        value: Any,
        *,
        rank: int | str = 0,
        unsafe: bool = False,
    ) -> Any:
        return self._call(
            "set_field",
            str(name),
            value,
            rank,
            bool(unsafe),
        )

    def set_scheme_enabled(
        self,
        name: str,
        enabled: bool,
        *,
        group: str | None = None,
        unsafe: bool = False,
    ) -> Any:
        return self._call(
            "set_scheme_enabled",
            str(name),
            bool(enabled),
            group,
            bool(unsafe),
        )

    def move_scheme(
        self,
        name: str,
        *,
        before: str | None = None,
        after: str | None = None,
        group: str | None = None,
        to_group: str | None = None,
        unsafe: bool = False,
    ) -> Any:
        return self._call(
            "move_scheme",
            str(name),
            before,
            after,
            group,
            to_group,
            bool(unsafe),
        )

    def reset_scheme_plan(self) -> Any:
        return self._call("reset_scheme_plan")

    def describe_scheme_plan(self, group: str | None = None) -> Any:
        return self._call("describe_scheme_plan", group)

    def run_plan(self, plan: SegmentPlan) -> Any:
        if not isinstance(plan, SegmentPlan):
            raise TypeError("plan must be SegmentPlan")
        return self._call("run_plan", plan.as_dict())

    def run_action(
        self,
        *,
        name: str,
        action: Any,
        unsafe: bool = True,
    ) -> Any:
        return self.run_plan(SegmentPlan(name=name, actions=(action,), unsafe=unsafe))

    def checkpoint(self, path: str | Path | None = None) -> Any:
        return self._call(
            "checkpoint",
            None if path is None else str(Path(path).resolve()),
        )

    def memory_checkpoint(self) -> Any:
        """Return an ActorFuture containing the immutable in-memory snapshot."""

        return self._call("memory_checkpoint")

    snapshot = memory_checkpoint

    def close(self) -> Any:
        if self._closed:
            raise RuntimeError("persistent Dask CAM session is already closed")
        self._closed = True
        return self.actor.close()

    def __enter__(self) -> "PersistentDaskSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if not self._closed:
            future = self.close()
            future.result()

    def _call(self, method: str, *args: Any) -> Any:
        if self._closed:
            raise RuntimeError("persistent Dask CAM session is closed")
        return getattr(self.actor, method)(*args)


__all__ = [
    "PersistentCAMActor",
    "PersistentDaskRequest",
    "PersistentDaskSession",
]
