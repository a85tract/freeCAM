"""Dask Actor and Pythonic controls for one persistent multi-model MPI pool."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import re
import socket
import threading
import time
from typing import Any, Callable

from ..model import (
    ActivatePhysics,
    BlockingModel,
    CCPPSuitePlan,
    DeactivatePhysics,
    DefineVariable,
    FieldEdit,
    InstallPhysics,
    ModelOptions,
    MoveScheme,
    ObserveFields,
    PhysicsPluginSpec,
    PrepareInitialStep,
    RunPhase,
    RunScheme,
    RunSchemeGroup,
    RunSteps,
    SegmentPlan,
    SetSchemeEnabled,
    VariableSpec,
)


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _wait(value: Any) -> Any:
    result = getattr(value, "result", None)
    return result() if callable(result) else value


def _action_kind(action: Any) -> str:
    words: list[str] = []
    current = ""
    for character in type(action).__name__:
        if character.isupper() and current:
            words.append(current.lower())
            current = character
        else:
            current += character
    if current:
        words.append(current.lower())
    return "_".join(words)


@dataclass(frozen=True, slots=True)
class PooledDaskRequest:
    """Serializable launch configuration for one pooled MPI world."""

    name: str
    config: str
    initial_run_dir: str
    run_root: str
    library: str
    environment_script: str
    python_executable: str
    log_dir: str
    resource_plan: Mapping[str, Any]
    launch_mode: str
    pbs_account: str
    pbs_queue: str
    pbs_walltime: str
    startup_timeout: float
    request_timeout: float
    options: Mapping[str, Any]
    scheme_plan: Mapping[str, Any]
    execution_mode: str

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "pool name may contain only letters, digits, dot, dash, "
                "and underscore"
            )
        ranks = int(self.resource_plan.get("ranks_per_model", 0))
        slots = int(self.resource_plan.get("model_slots", 0))
        world = int(self.resource_plan.get("world_size", 0))
        if ranks <= 0 or slots <= 0:
            raise ValueError("resource plan must contain positive ranks and slots")
        if world != ranks * slots:
            raise ValueError(
                "resource plan world_size must equal "
                "ranks_per_model * model_slots"
            )
        if self.launch_mode not in {"auto", "local", "pbs"}:
            raise ValueError("launch_mode must be auto, local, or pbs")
        if self.execution_mode not in {"pbs", "allocation"}:
            raise ValueError("execution_mode must be pbs or allocation")


PoolSessionFactory = Callable[..., Any]


def _default_pool_session_factory() -> PoolSessionFactory:
    # Imported lazily so the public planning API remains usable on systems
    # without mpi4py or a configured launcher.
    from .pool_session import PooledWorkerSession

    return PooledWorkerSession


class PoolLauncherActor:
    """Worker-pinned launcher for one MPI world split into reusable slots."""

    def __init__(
        self,
        request: PooledDaskRequest,
        session_factory: PoolSessionFactory | None = None,
    ) -> None:
        self._request = request
        self._guard = threading.RLock()
        self._closed = False
        self._mpi_launch_count = 0
        self._mpi_launch_id = (
            f"{request.name}-{socket.gethostname()}-{os.getpid()}-"
            f"{time.time_ns()}"
        )
        self._model_plans: dict[str, CCPPSuitePlan] = {}
        self._model_details: dict[str, dict[str, Any]] = {}
        self._command_condition = threading.Condition(self._guard)
        self._command_queue: deque[
            tuple[int, str, str, tuple[Any, ...], dict[str, Any]]
        ] = deque()
        self._command_results: dict[int, tuple[str, Any]] = {}
        self._next_command_id = 1
        plan = dict(request.resource_plan)
        options = ModelOptions(**dict(request.options))
        scheme_plan = CCPPSuitePlan.from_payload(request.scheme_plan)
        factory = session_factory or _default_pool_session_factory()
        self._session = factory(
            request.config,
            initial_run_dir=request.initial_run_dir,
            run_root=request.run_root,
            library=request.library,
            ranks_per_model=int(plan["ranks_per_model"]),
            model_slots=int(plan["model_slots"]),
            resource_plan=plan,
            env_script=request.environment_script,
            launch_mode=request.launch_mode,
            pbs_account=request.pbs_account,
            pbs_queue=request.pbs_queue,
            pbs_walltime=request.pbs_walltime,
            python_executable=request.python_executable,
            startup_timeout=request.startup_timeout,
            request_timeout=request.request_timeout,
            log_path=Path(request.log_dir) / f"pycam_pool_{request.name}.log",
            options=options,
            scheme_plan=scheme_plan,
            pool_name=request.name,
        )
        try:
            self._session.start()
            self._mpi_launch_count = 1
            self._dispatcher = threading.Thread(
                target=self._dispatch_commands,
                name=f"pycam-pool-{request.name}",
                daemon=True,
            )
            self._dispatcher.start()
        except BaseException:
            try:
                self._session.close()
            except BaseException:
                pass
            raise

    def describe(self) -> dict[str, Any]:
        with self._guard:
            self._ensure_open()
            result = dict(self._session.describe())
            result.setdefault("name", self._request.name)
            result.setdefault("running", True)
            result["mpi_launch_count"] = self._mpi_launch_count
            result["pool_mpi_launch_id"] = self._mpi_launch_id
            result["resource_plan"] = dict(self._request.resource_plan)
            return result

    def slots(self) -> tuple[Mapping[str, Any], ...]:
        with self._guard:
            self._ensure_open()
            values = self._session.slots
            if callable(values):
                values = values()
            return tuple(dict(value) for value in values)

    def create_model(self, name: str, slot: int | None = None) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._validate_model_name(name)
            root = Path(self._request.run_root) / self._request.name / name
            result = self._session.create_model(
                name,
                run_dir=root / "run",
                history_dir=root / "history",
                slot=slot,
            )
            self._model_plans[name] = CCPPSuitePlan.from_payload(
                self._request.scheme_plan
            )
            self._model_details[name] = {
                **dict(result),
                "run_dir": str(root / "run"),
                "history_dir": str(root / "history"),
            }
            return result

    def fork_model(
        self,
        parent: str,
        children: Sequence[str],
        *,
        require_concurrent: bool = False,
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._validate_model_name(parent)
            normalized = tuple(str(name) for name in children)
            if not normalized:
                raise ValueError("fork requires at least one child")
            if len(normalized) != len(set(normalized)):
                raise ValueError("fork child names must be unique")
            if parent in normalized:
                raise ValueError("fork child name must differ from parent")
            descriptors = []
            for name in normalized:
                self._validate_model_name(name)
                root = Path(self._request.run_root) / self._request.name / name
                descriptors.append(
                    {
                        "name": name,
                        "run_dir": str(root / "run"),
                        "history_dir": str(root / "history"),
                    }
                )
            result = self._session.fork_model(
                parent,
                tuple(descriptors),
                require_concurrent=require_concurrent,
            )
            for name in normalized:
                self._model_plans[name] = self._model_plans[parent].copy()
                root = Path(self._request.run_root) / self._request.name / name
                self._model_details[name] = {
                    **self._model_details.get(parent, {}),
                    "name": name,
                    "run_dir": str(root / "run"),
                    "history_dir": str(root / "history"),
                    "snapshot_transport": "mpi",
                }
            return result

    def model_call(
        self,
        name: str,
        operation: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        with self._guard:
            self._ensure_open()
            self._validate_model_name(name)
            if operation == "describe":
                return self._describe_model(name)
            if operation == "field_info":
                fields = self._model_details.get(name, {}).get("fields", {})
                try:
                    return dict(fields[str(args[0])])
                except KeyError as exc:
                    raise KeyError(str(args[0])) from exc
            if operation == "describe_scheme_plan":
                group = args[0] if args else None
                return self._model_plans[name].describe(group)
            if operation == "reset_scheme_plan":
                candidate = CCPPSuitePlan.from_payload(
                    self._request.scheme_plan
                )
                self._install_model_plan(name, candidate)
                return candidate.describe()
            if operation == "set_scheme_enabled":
                candidate = self._model_plans[name].copy()
                scheme, enabled = str(args[0]), bool(args[1])
                group = kwargs.get("group")
                if enabled:
                    candidate.enable(scheme, group=group)
                else:
                    candidate.disable(
                        scheme,
                        group=group,
                        unsafe=bool(kwargs.get("unsafe", False)),
                    )
                self._install_model_plan(name, candidate)
                return candidate.describe()
            if operation == "move_scheme":
                candidate = self._model_plans[name].copy()
                candidate.move(
                    str(args[0]),
                    before=kwargs.get("before"),
                    after=kwargs.get("after"),
                    group=kwargs.get("group"),
                    to_group=kwargs.get("to_group"),
                    unsafe=bool(kwargs.get("unsafe", False)),
                )
                self._install_model_plan(name, candidate)
                return candidate.describe()
            if operation == "run_plan":
                return self._run_plan(name, args[0])
            if operation == "checkpoint" and (not args or args[0] is None):
                target = (
                    Path(self._request.run_root)
                    / self._request.name
                    / name
                    / "checkpoints"
                    / f"checkpoint-{time.time_ns()}"
                )
                args = (target,)
            wire_operation, payload = self._wire_payload(
                operation, args, kwargs
            )
            result = self._session.call(name, wire_operation, **payload)
            if isinstance(result, Mapping):
                self._model_details.setdefault(name, {}).update(result)
            if operation == "install_physics":
                if not isinstance(result, Mapping):
                    raise TypeError(
                        "pooled install_physics returned a non-mapping result"
                    )
                try:
                    return dict(result["installed_plugin"])
                except (KeyError, TypeError) as exc:
                    raise RuntimeError(
                        "pooled install_physics response lacks "
                        "installed_plugin metadata"
                    ) from exc
            if operation == "delete_variable":
                if not isinstance(result, Mapping):
                    raise TypeError(
                        "pooled delete_variable returned a non-mapping result"
                    )
                try:
                    return dict(result["deleted_variable"])
                except (KeyError, TypeError) as exc:
                    raise RuntimeError(
                        "pooled delete_variable response lacks "
                        "deleted_variable metadata"
                    ) from exc
            return result

    def submit_model_call(
        self,
        name: str,
        operation: str,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> int:
        """Queue a model command without blocking the launcher's Actor thread."""

        with self._command_condition:
            self._ensure_open()
            self._validate_model_name(name)
            token = self._next_command_id
            self._next_command_id += 1
            self._command_results[token] = ("pending", None)
            self._command_queue.append(
                (
                    token,
                    str(name),
                    str(operation),
                    tuple(args),
                    dict(kwargs or {}),
                )
            )
            self._command_condition.notify()
            return token

    def poll_model_call(self, token: int) -> Mapping[str, Any]:
        """Return a non-blocking command result so Actor RPC stays re-entrant."""

        with self._guard:
            try:
                state, value = self._command_results[int(token)]
            except KeyError as exc:
                raise KeyError(f"unknown pooled command token {token}") from exc
            if state == "pending":
                return {"state": "pending"}
            self._command_results.pop(int(token), None)
            return {"state": state, "value": value}

    def _dispatch_commands(self) -> None:
        while True:
            with self._command_condition:
                while not self._command_queue and not self._closed:
                    self._command_condition.wait()
                if self._closed and not self._command_queue:
                    return
                batch = self._next_command_batch()
            self._execute_command_batch(batch)

    def _next_command_batch(
        self,
    ) -> list[tuple[int, str, str, tuple[Any, ...], dict[str, Any]]]:
        first = self._command_queue.popleft()
        if first[2] not in self._batchable_operations():
            return [first]
        selected = [first]
        names = {first[1]}
        retained: deque[
            tuple[int, str, str, tuple[Any, ...], dict[str, Any]]
        ] = deque()
        while self._command_queue:
            item = self._command_queue.popleft()
            if item[2] not in self._batchable_operations():
                retained.append(item)
                retained.extend(self._command_queue)
                self._command_queue.clear()
                break
            if item[1] in names:
                retained.append(item)
                continue
            names.add(item[1])
            selected.append(item)
        self._command_queue.extendleft(reversed(retained))
        return selected

    @staticmethod
    def _batchable_operations() -> frozenset[str]:
        return frozenset(
            {
                "step",
                "describe",
                "prepare_initial_step",
                "run_phase",
                "run_scheme",
                "run_scheme_group",
                "get_field",
                "get_field_stats",
                "set_field",
                "edit_field",
            }
        )

    def _execute_command_batch(
        self,
        batch: Sequence[
            tuple[int, str, str, tuple[Any, ...], dict[str, Any]]
        ],
    ) -> None:
        outcomes: dict[int, tuple[str, Any]] = {}
        try:
            with self._guard:
                calls = [
                    (name, operation, args, kwargs)
                    for _token, name, operation, args, kwargs in batch
                ]
                call_models = getattr(self._session, "call_models", None)
                if len(batch) > 1 and callable(call_models):
                    results = call_models(calls)
                    for token, name, _operation, _args, _kwargs in batch:
                        value = results[name]
                        if isinstance(value, Mapping):
                            self._model_details.setdefault(name, {}).update(
                                value
                            )
                        outcomes[token] = ("done", value)
                else:
                    for token, name, operation, args, kwargs in batch:
                        try:
                            outcomes[token] = (
                                "done",
                                self.model_call(
                                    name,
                                    operation,
                                    *args,
                                    **kwargs,
                                ),
                            )
                        except BaseException as exc:
                            outcomes[token] = (
                                "error",
                                f"{type(exc).__name__}: {exc}",
                            )
        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            outcomes.update(
                (token, ("error", message))
                for token, *_rest in batch
            )
        with self._command_condition:
            self._command_results.update(outcomes)
            self._command_condition.notify_all()

    def advance_models(
        self,
        names: Sequence[str],
        count: int = 1,
    ) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            normalized = tuple(str(name) for name in names)
            if count < 0:
                raise ValueError("steps must be non-negative")
            for name in normalized:
                self._validate_model_name(name)
            advance = getattr(self._session, "advance_models", None)
            if callable(advance):
                results = advance(normalized, int(count))
                for name, result in results.items():
                    if isinstance(result, Mapping):
                        self._model_details.setdefault(name, {}).update(result)
                return results
            # Fake and legacy adapters may not yet provide the collective
            # command. The real pooled worker does, so production slots run
            # concurrently while this deterministic fallback aids testing.
            return {
                name: self._session.call(name, "step", int(count))
                for name in normalized
            }

    def close_model(self, name: str) -> Mapping[str, Any]:
        with self._guard:
            self._ensure_open()
            self._validate_model_name(name)
            result = self._session.close_model(name)
            self._model_plans.pop(name, None)
            self._model_details.pop(name, None)
            return result

    def close(self) -> dict[str, Any]:
        with self._command_condition:
            if self._closed:
                return {
                    "closed": True,
                    "mpi_launch_count": self._mpi_launch_count,
                }
            self._closed = True
            self._command_condition.notify_all()
            pending = tuple(self._command_queue)
            self._command_queue.clear()
            for token, *_rest in pending:
                self._command_results[token] = (
                    "error",
                    "RuntimeError: persistent model pool is closed",
                )
            self._session.close()
            return {
                "closed": True,
                "mpi_launch_count": self._mpi_launch_count,
            }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("persistent model pool is closed")

    @staticmethod
    def _validate_model_name(name: str) -> None:
        if not _SAFE_NAME.fullmatch(str(name)):
            raise ValueError(
                "model name may contain only letters, digits, dot, dash, "
                "and underscore"
            )

    def _install_model_plan(
        self, name: str, candidate: CCPPSuitePlan
    ) -> None:
        self._session.call(
            name,
            "configure_scheme_plan",
            plan=candidate.to_payload(),
        )
        self._model_plans[name] = candidate

    def _describe_model(self, name: str) -> dict[str, Any]:
        pool_status = dict(self._session.describe())
        slots = tuple(pool_status.get("slots", self._session.slots))
        slot = next(
            (
                dict(item)
                for item in slots
                if item.get("model_name") == name
            ),
            None,
        )
        if slot is None:
            raise KeyError(f"unknown pooled model: {name!r}")
        details = dict(self._model_details.get(name, {}))
        fields = details.get("fields", {})
        return {
            **details,
            **slot,
            "name": name,
            "running": True,
            "ranks": int(self._request.resource_plan["ranks_per_model"]),
            "step": int(slot.get("step", details.get("step", 0))),
            "native_calls": int(
                slot.get("native_calls", details.get("native_calls", 0))
            ),
            "mpi_launch_count": self._mpi_launch_count,
            "pool_mpi_launch_id": self._mpi_launch_id,
            "worker_host": socket.gethostname(),
            "worker_pid": os.getpid(),
            "launch_mode": getattr(self._session, "launch_mode_used", None),
            "pbs_job_id": getattr(self._session, "job_id", None),
            "outer_pbs_job_id": os.environ.get("PBS_JOBID"),
            "field_count": len(fields),
            "snapshot_transport": details.get(
                "snapshot_transport", "initialization"
            ),
            "log_path": str(
                Path(self._request.log_dir)
                / f"pycam_pool_{self._request.name}.log"
            ),
            "phase_names": tuple(details.get("phase_names", ())),
            "scheme_names": tuple(details.get("scheme_names", ())),
            "scheme_status": details.get(
                "scheme_status",
                {
                    "sequence_safe": self._model_plans[
                        name
                    ].sequence_safe
                },
            ),
        }

    def _run_plan(
        self,
        name: str,
        value: SegmentPlan | Mapping[str, Any],
    ) -> dict[str, Any]:
        plan = (
            value
            if isinstance(value, SegmentPlan)
            else SegmentPlan.from_mapping(value)
        )
        self._validate_plan(name, plan)
        trace: list[dict[str, Any]] = []
        for index, action in enumerate(plan.actions):
            if isinstance(action, PrepareInitialStep):
                result = self.model_call(name, "prepare_initial_step")
            elif isinstance(action, RunPhase):
                result = self.model_call(name, "run_phase", action.name)
            elif isinstance(action, RunScheme):
                result = self.model_call(
                    name, "run_scheme", action.name, group=action.group
                )
            elif isinstance(action, RunSchemeGroup):
                result = self.model_call(name, "run_scheme_group", action.group)
            elif isinstance(action, RunSteps):
                result = self.model_call(name, "step", action.count)
            elif isinstance(action, SetSchemeEnabled):
                result = self.model_call(
                    name,
                    "set_scheme_enabled",
                    action.name,
                    action.enabled,
                    group=action.group,
                    unsafe=plan.unsafe,
                )
            elif isinstance(action, MoveScheme):
                result = self.model_call(
                    name,
                    "move_scheme",
                    action.name,
                    before=action.before,
                    after=action.after,
                    to_group=action.to_group,
                    unsafe=plan.unsafe,
                )
            elif isinstance(action, FieldEdit):
                result = self.model_call(
                    name,
                    "edit_field",
                    action.name,
                    action.operation,
                    action.value,
                    unsafe=action.unsafe,
                )
            elif isinstance(action, DefineVariable):
                result = self.model_call(
                    name,
                    "define_variable",
                    action.spec,
                    initial=action.initial_value,
                )
            elif isinstance(action, InstallPhysics):
                result = self.model_call(
                    name,
                    "install_physics",
                    action.plugin,
                    initial_values=action.initial_values,
                    effective=action.effective,
                    unsafe=plan.unsafe,
                )
            elif isinstance(action, ActivatePhysics):
                result = self.model_call(
                    name,
                    "activate_physics",
                    action.name,
                    unsafe=plan.unsafe,
                )
            elif isinstance(action, DeactivatePhysics):
                result = self.model_call(
                    name,
                    "deactivate_physics",
                    action.name,
                    unsafe=plan.unsafe,
                )
            elif isinstance(action, ObserveFields):
                result = {
                    field: self.model_call(
                        name, "get_field_stats", field, rank="all"
                    )
                    for field in action.fields
                }
            else:
                raise TypeError(
                    f"unsupported pooled action {type(action).__name__}"
                )
            trace.append(
                {
                    "index": index,
                    "type": _action_kind(action),
                    "result": result,
                }
            )
        return {
            "name": plan.name,
            "action_trace": trace,
            "action_count": len(trace),
        }

    def _validate_plan(self, name: str, plan: SegmentPlan) -> None:
        """Reject invalid names and unsafe calls before the first mutation."""

        scheme_plan = self._model_plans[name].copy()
        phases = set(self._model_details.get(name, {}).get("phase_names", ()))
        fields = set(self._model_details.get(name, {}).get("fields", ()))
        for action in plan.actions:
            if isinstance(action, RunPhase):
                if action.name not in phases:
                    raise ValueError(f"unknown model phase {action.name!r}")
                if not plan.unsafe:
                    raise ValueError(
                        "run_phase actions require SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, RunScheme):
                scheme_plan.scheme(action.name, group=action.group)
                if not plan.unsafe:
                    raise ValueError(
                        "run_scheme actions require SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, RunSchemeGroup):
                if action.group not in scheme_plan.group_names:
                    raise ValueError(f"unknown scheme group {action.group!r}")
                if not plan.unsafe:
                    raise ValueError(
                        "run_scheme_group actions require "
                        "SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, SetSchemeEnabled):
                scheme_plan.scheme(action.name, group=action.group)
                if not action.enabled and not plan.unsafe:
                    raise ValueError(
                        "disabling a scheme requires SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, MoveScheme):
                scheme_plan.scheme(action.name)
                if not plan.unsafe:
                    raise ValueError(
                        "moving a scheme requires SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, FieldEdit):
                if action.name not in fields:
                    raise ValueError(f"unknown state field {action.name!r}")
                if action.unsafe and not plan.unsafe:
                    raise ValueError(
                        "unsafe field edits require SegmentPlan(unsafe=True)"
                    )
            elif isinstance(action, ObserveFields):
                unknown = set(action.fields) - fields
                if unknown:
                    raise ValueError(
                        f"unknown state fields: {sorted(unknown)}"
                    )
            elif isinstance(action, DefineVariable):
                if action.spec.name in fields:
                    raise ValueError(
                        f"state field already exists: {action.spec.name!r}"
                    )
                fields.add(action.spec.name)

    @staticmethod
    def _wire_payload(
        operation: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        values = dict(kwargs)
        if operation == "step":
            return operation, {"count": int(args[0])}
        if operation == "run_phase":
            return operation, {"phase": str(args[0])}
        if operation == "run_scheme":
            return operation, {
                "scheme": str(args[0]),
                "group": values.get("group"),
            }
        if operation == "run_scheme_group":
            return operation, {"group": str(args[0])}
        if operation in {"get_field", "get_field_stats"}:
            return operation, {
                "field": str(args[0]),
                "rank": values.get("rank", 0),
            }
        if operation == "set_field":
            return operation, {
                "field": str(args[0]),
                "value": args[1],
                "rank": values.get("rank", 0),
                "unsafe": bool(values.get("unsafe", False)),
            }
        if operation == "edit_field":
            return operation, {
                "field": str(args[0]),
                "operation": str(args[1]),
                "value": args[2],
                "unsafe": bool(values.get("unsafe", False)),
            }
        if operation == "define_variable":
            spec = args[0]
            return operation, {
                "spec": (
                    spec.as_dict()
                    if isinstance(spec, VariableSpec)
                    else dict(spec)
                ),
                "initial_value": values.get("initial", 0.0),
            }
        if operation == "delete_variable":
            return operation, {"name": str(args[0])}
        if operation == "install_physics":
            spec = args[0]
            return operation, {
                "plugin": (
                    spec.as_dict()
                    if isinstance(spec, PhysicsPluginSpec)
                    else dict(spec)
                ),
                "initial_values": values.get("initial_values"),
                "effective": values.get("effective", "now"),
                "unsafe": bool(values.get("unsafe", False)),
            }
        if operation in {"activate_physics", "deactivate_physics"}:
            return operation, {
                "name": str(args[0]),
                "unsafe": bool(values.get("unsafe", False)),
            }
        if operation == "checkpoint":
            return "write_checkpoint", {"path": str(args[0])}
        if operation == "memory_checkpoint":
            return "capture_memory_checkpoint", {}
        if operation in {"prepare_initial_step"}:
            return operation, {}
        raise ValueError(f"unsupported pooled model operation: {operation!r}")


# Compatibility name for notebooks created before the model-per-worker layout.
PersistentPoolActor = PoolLauncherActor


class ModelActor:
    """Worker-resident controller for exactly one model in one MPI slot."""

    def __init__(
        self,
        launcher: Any,
        *,
        pool_name: str,
        model_name: str,
        slot_id: int,
    ) -> None:
        self._launcher = launcher
        self.pool_name = str(pool_name)
        self.model_name = str(model_name)
        self.slot_id = int(slot_id)
        self._closed = False
        self.worker = self._worker_address()

    @staticmethod
    def _worker_address() -> str:
        try:
            from distributed import get_worker

            return str(get_worker().address)
        except (ImportError, ValueError):
            return f"local://{socket.gethostname()}-{os.getpid()}"

    def describe(self) -> dict[str, Any]:
        self._ensure_open()
        return dict(self.execute("describe"))

    def execute(
        self,
        operation: str,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        self._ensure_open()
        submit = getattr(self._launcher, "submit_model_call", None)
        poll = getattr(self._launcher, "poll_model_call", None)
        if callable(submit) and callable(poll):
            token = int(
                _wait(
                    submit(
                        self.model_name,
                        str(operation),
                        tuple(args),
                        dict(kwargs or {}),
                    )
                )
            )
            while True:
                outcome = dict(_wait(poll(token)))
                if outcome["state"] == "pending":
                    time.sleep(0.001)
                    continue
                if outcome["state"] == "error":
                    raise RuntimeError(str(outcome["value"]))
                value = outcome.get("value")
                return (
                    self._decorate_status(value)
                    if str(operation) == "describe"
                    else value
                )
        value = _wait(
            self._launcher.model_call(
                self.model_name,
                str(operation),
                *tuple(args),
                **dict(kwargs or {}),
            )
        )
        return (
            self._decorate_status(value)
            if str(operation) == "describe"
            else value
        )

    def _decorate_status(self, value: Any) -> dict[str, Any]:
        result = dict(value)
        result.update(
            {
                "dask_worker": self.worker,
                "slot_id": self.slot_id,
                "pool_name": self.pool_name,
                "actor_layout": "model-per-worker",
            }
        )
        return result

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            return {
                "closed": True,
                "name": self.model_name,
                "slot_id": self.slot_id,
            }
        result = _wait(self._launcher.close_model(self.model_name))
        self._closed = True
        return result

    def detach(self) -> Mapping[str, Any]:
        """Close only this Dask Actor after the launcher released its slot."""

        self._closed = True
        return {
            "closed": True,
            "name": self.model_name,
            "slot_id": self.slot_id,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"model actor {self.model_name!r} is closed")


def execute_model_action(
    model_actor: Any,
    operation: str,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    dependency: Any = None,
) -> Any:
    """Dask task node for one ordered operation on a persistent ModelActor."""

    # Dask resolves Future dependencies before calling this function.  Keeping
    # the argument makes the dependency visible in the scheduler task graph.
    del dependency
    return _wait(
        model_actor.execute(
            str(operation),
            tuple(args),
            dict(kwargs or {}),
        )
    )


class _PooledSchemePlan:
    def __init__(self, model: "PooledModelSession") -> None:
        self._model = model

    @property
    def sequence_safe(self) -> bool:
        status = self._model.describe()
        result = _wait(status)
        return bool(result.get("scheme_status", {}).get("sequence_safe", False))

    def describe(self, group: str | None = None) -> Any:
        return _wait(self._model.describe_scheme_plan(group))

    def reset(self) -> Any:
        return _wait(self._model.reset_scheme_plan())


class _AsyncFieldReference:
    def __init__(self, model: "PooledModelSession", name: str) -> None:
        self._model = model
        self.name = str(name)

    def get(
        self,
        *,
        rank: int | str = 0,
        depends_on: Any = None,
    ) -> Any:
        return self._model._call(
            "get_field",
            self.name,
            rank=rank,
            depends_on=depends_on,
        )

    def stats(
        self,
        *,
        rank: int | str = 0,
        depends_on: Any = None,
    ) -> Any:
        return self._model._call(
            "get_field_stats",
            self.name,
            rank=rank,
            depends_on=depends_on,
        )


class _AsyncFieldCollection:
    def __init__(self, model: "PooledModelSession") -> None:
        self._model = model

    def __getitem__(self, name: str) -> _AsyncFieldReference:
        return _AsyncFieldReference(self._model, str(name))

    def __getattr__(self, name: str) -> _AsyncFieldReference:
        if name.startswith("_"):
            raise AttributeError(name)
        return self[name]


class PooledModelSession:
    """Future-returning proxy for one worker-resident ModelActor."""

    def __init__(
        self,
        pool: "PersistentModelPool",
        name: str,
        *,
        actor: Any | None = None,
        actor_future: Any | None = None,
        worker: str | None = None,
        slot_id: int | None = None,
    ) -> None:
        self.pool = pool
        self.actor = actor or pool.actor
        self.actor_future = actor_future
        self.worker = worker or pool.worker
        self.slot_id = slot_id
        self.name = str(name)
        self._closed = False
        self._tail: Any = None
        self.scheme_plan = _PooledSchemePlan(self)
        self.fields = _AsyncFieldCollection(self)

    @property
    def sync(self) -> "PooledModel":
        return PooledModel(self)

    @property
    def phase_names(self) -> tuple[str, ...]:
        return tuple(_wait(self.describe()).get("phase_names", ()))

    @property
    def scheme_names(self) -> tuple[str, ...]:
        return tuple(_wait(self.describe()).get("scheme_names", ()))

    def describe(self, *, depends_on: Any = None) -> Any:
        if self.pool.actor_layout == "legacy-single-worker":
            return self._call("describe", depends_on=depends_on)
        return self._submit("describe", depends_on=depends_on)

    def initialize(self, *, depends_on: Any = None) -> Any:
        return self.prepare_initial_step(depends_on=depends_on)

    def step(self, count: int = 1, *, depends_on: Any = None) -> Any:
        return self._call("step", int(count), depends_on=depends_on)

    def advance(self, steps: int = 1, *, depends_on: Any = None) -> Any:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return self.step(int(steps), depends_on=depends_on)

    def prepare_initial_step(self, *, depends_on: Any = None) -> Any:
        return self._call("prepare_initial_step", depends_on=depends_on)

    def run_phase(
        self,
        name: str,
        *,
        unsafe: bool = False,
        depends_on: Any = None,
    ) -> Any:
        del unsafe
        return self._call("run_phase", str(name), depends_on=depends_on)

    def run_scheme(
        self,
        name: str,
        *,
        group: str | None = None,
        unsafe: bool = False,
        depends_on: Any = None,
    ) -> Any:
        del unsafe
        return self._call(
            "run_scheme",
            str(name),
            group=group,
            depends_on=depends_on,
        )

    def run_scheme_group(
        self,
        group: str,
        *,
        unsafe: bool = False,
        depends_on: Any = None,
    ) -> Any:
        del unsafe
        return self._call(
            "run_scheme_group",
            str(group),
            depends_on=depends_on,
        )

    def get_field(self, name: str, *, rank: int | str = 0) -> Any:
        return self._call("get_field", str(name), rank=rank)

    field = get_field

    def get_field_stats(self, name: str, *, rank: int | str = 0) -> Any:
        return self._call("get_field_stats", str(name), rank=rank)

    field_stats = get_field_stats

    def field_info(self, name: str) -> Any:
        return self._call("field_info", str(name))

    def set_field(
        self,
        name: str,
        value: Any,
        *,
        rank: int | str = 0,
        unsafe: bool = False,
    ) -> Any:
        return self._call(
            "set_field", str(name), value, rank=rank, unsafe=bool(unsafe)
        )

    def edit_field(
        self,
        name: str,
        operation: str,
        value: Any,
        *,
        unsafe: bool = False,
    ) -> Any:
        return self._call(
            "edit_field",
            str(name),
            str(operation),
            value,
            unsafe=bool(unsafe),
        )

    def define_variable(
        self,
        spec: VariableSpec,
        *,
        initial: Any = 0.0,
    ) -> Any:
        return self._call("define_variable", spec, initial=initial)

    def delete_variable(self, name: str) -> Any:
        return self._call("delete_variable", str(name))

    def install_physics(
        self,
        spec: PhysicsPluginSpec,
        *,
        initial_values: Any = None,
        effective: str = "now",
        unsafe: bool = False,
    ) -> Any:
        return self._call(
            "install_physics",
            spec,
            initial_values=initial_values,
            effective=effective,
            unsafe=bool(unsafe),
        )

    def activate_physics(self, name: str, *, unsafe: bool = False) -> Any:
        return self._call("activate_physics", str(name), unsafe=bool(unsafe))

    def deactivate_physics(self, name: str, *, unsafe: bool = False) -> Any:
        return self._call("deactivate_physics", str(name), unsafe=bool(unsafe))

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
            group=group,
            unsafe=bool(unsafe),
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
            before=before,
            after=after,
            group=group,
            to_group=to_group,
            unsafe=bool(unsafe),
        )

    def reset_scheme_plan(self) -> Any:
        return self._call("reset_scheme_plan")

    def describe_scheme_plan(self, group: str | None = None) -> Any:
        return self._call("describe_scheme_plan", group)

    def run_plan(self, plan: SegmentPlan) -> Any:
        if not isinstance(plan, SegmentPlan):
            raise TypeError("plan must be SegmentPlan")
        return self._call("run_plan", plan)

    def checkpoint(self, path: str | Path | None = None) -> Any:
        return self._call(
            "checkpoint",
            None if path is None else str(Path(path).resolve()),
        )

    def memory_checkpoint(self) -> Any:
        return self._call("memory_checkpoint")

    snapshot = memory_checkpoint

    def fork(
        self,
        *names: str,
        require_concurrent: bool = False,
        depends_on: Any = None,
    ) -> "PooledModelGroup":
        dependencies = tuple(
            value
            for value in (self._tail, depends_on)
            if value is not None
        )
        return self.pool._fork(
            self.name,
            names,
            require_concurrent=require_concurrent,
            depends_on=dependencies,
        )

    def close(self) -> Any:
        if self._closed:
            raise RuntimeError("pooled model is already closed")
        if self._tail is not None:
            _wait(self._tail)
        self._closed = True
        if self.pool.actor_layout == "legacy-single-worker":
            return self.actor.close_model(self.name)
        result = self.actor.close()
        self.pool._release_model_actor(self.name)
        self.actor = None
        self.actor_future = None
        return result

    def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("pooled model is closed")
        depends_on = kwargs.pop("depends_on", None)
        if self.pool.actor_layout == "legacy-single-worker":
            if depends_on is not None:
                _wait(depends_on)
            return self.actor.model_call(self.name, operation, *args, **kwargs)
        return self._submit(
            operation,
            *args,
            depends_on=depends_on,
            **kwargs,
        )

    def _submit(
        self,
        operation: str,
        *args: Any,
        depends_on: Any = None,
        **kwargs: Any,
    ) -> Any:
        if self._closed:
            raise RuntimeError("pooled model is closed")
        dependencies = tuple(
            value
            for value in (self._tail, depends_on)
            if value is not None
        )
        dependency = (
            None
            if not dependencies
            else dependencies[0]
            if len(dependencies) == 1
            else dependencies
        )
        future = self.pool.client.submit(
            execute_model_action,
            self.actor,
            str(operation),
            tuple(args),
            dict(kwargs),
            dependency,
            pure=False,
            workers=[self.worker],
            allow_other_workers=False,
        )
        self._tail = future
        return future


class PooledModel(BlockingModel):
    """Blocking Pythonic model whose StatePool lives in one reusable slot."""

    @property
    def pool(self) -> "PersistentModelPool":
        return self.submit.pool

    @property
    def name(self) -> str:
        return self.submit.name

    @property
    def worker(self) -> str:
        return self.submit.worker

    @property
    def slot_id(self) -> int | None:
        return self.submit.slot_id

    def fork(
        self,
        *names: str,
        require_concurrent: bool = False,
        depends_on: Any = None,
    ) -> "PooledModelGroup":
        return self.submit.fork(
            *names,
            require_concurrent=require_concurrent,
            depends_on=depends_on,
        )


class PooledModelGroup(Mapping[str, PooledModel]):
    """Independent child slots produced from one in-memory parent state."""

    def __init__(
        self,
        pool: "PersistentModelPool",
        models: Mapping[str, PooledModel],
    ) -> None:
        self.pool = pool
        self._models = dict(models)
        self._closed = False

    def __getitem__(self, name: str) -> PooledModel:
        return self._models[name]

    def __getattr__(self, name: str) -> PooledModel:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._models[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)

    @property
    def statuses(self) -> dict[str, Any]:
        return {name: model.status for name, model in self._models.items()}

    def advance(self, steps: int = 1) -> "PooledModelGroup":
        if steps < 0:
            raise ValueError("steps must be non-negative")
        for model in self._models.values():
            if model.submit._tail is not None:
                _wait(model.submit._tail)
        _wait(
            self.pool.launcher.advance_models(
                tuple(self._models),
                int(steps),
            )
        )
        return self

    def close(self) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        for model in self._models.values():
            try:
                _wait(model.submit.close())
            except BaseException as exc:
                failures.append(exc)
        self._closed = True
        if failures:
            raise RuntimeError(
                f"failed to close {len(failures)} pooled model(s)"
            ) from failures[0]

    def __enter__(self) -> "PooledModelGroup":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


class PersistentModelPool:
    """Client-side coordinator for one launcher and per-model Dask Actors."""

    def __init__(
        self,
        client: Any,
        actor: Any,
        actor_future: Any,
        *,
        worker: str,
        name: str,
        resource_plan: Any,
        actor_layout: str = "legacy-single-worker",
        worker_policy: str = "shared",
        model_workers: Sequence[str] = (),
        model_actor_factory: Any = ModelActor,
    ) -> None:
        self.client = client
        self.launcher = actor
        self.actor = actor
        self.actor_future = actor_future
        self.worker = worker
        self.name = str(name)
        self.resource_plan = resource_plan
        self.actor_layout = str(actor_layout)
        self.worker_policy = str(worker_policy)
        self._model_workers = tuple(str(value) for value in model_workers)
        self._model_actor_factory = model_actor_factory
        self._models: dict[str, PooledModel] = {}
        self._actor_futures: dict[str, Any] = {}
        self._closed = False

    @property
    def status(self) -> Mapping[str, Any]:
        self._ensure_open()
        return _wait(self.actor.describe())

    @property
    def slots(self) -> tuple[Mapping[str, Any], ...]:
        self._ensure_open()
        return tuple(_wait(self.launcher.slots()))

    @property
    def models(self) -> Mapping[str, PooledModel]:
        return {
            name: model
            for name, model in self._models.items()
            if not model.submit._closed
        }

    @property
    def scheduler_status(self) -> Mapping[str, Any]:
        info = self.client.scheduler_info()
        return {
            "launcher_worker": self.worker,
            "model_workers": {
                name: model.worker for name, model in self.models.items()
            },
            "available_workers": tuple(sorted(info.get("workers", {}))),
            "actor_layout": self.actor_layout,
            "worker_policy": self.worker_policy,
        }

    def model(self, name: str, *, slot: int | None = None) -> PooledModel:
        self._ensure_open()
        if name in self._models and not self._models[name].submit._closed:
            raise ValueError(f"model {name!r} already exists in this pool")
        descriptor = dict(_wait(self.launcher.create_model(str(name), slot)))
        model = self._attach_model(
            str(name),
            int(descriptor["slot_id"]),
        )
        self._models[str(name)] = model
        return model

    def _fork(
        self,
        parent: str,
        names: Sequence[str],
        *,
        require_concurrent: bool,
        depends_on: Any = None,
    ) -> PooledModelGroup:
        self._ensure_open()
        if depends_on is not None:
            if isinstance(depends_on, tuple):
                self.client.gather(depends_on)
            else:
                _wait(depends_on)
        normalized = tuple(str(name) for name in names)
        descriptors = dict(
            _wait(
                self.launcher.fork_model(
                    str(parent),
                    normalized,
                    require_concurrent=require_concurrent,
                )
            )
        )
        models: dict[str, PooledModel] = {}
        try:
            for name in normalized:
                descriptor = dict(descriptors[name])
                models[name] = self._attach_model(
                    name,
                    int(descriptor["slot_id"]),
                )
        except BaseException:
            for model in models.values():
                try:
                    _wait(model.submit.close())
                except BaseException:
                    pass
            for name in normalized:
                if name in models:
                    continue
                try:
                    _wait(self.launcher.close_model(name))
                except BaseException:
                    pass
            raise
        self._models.update(models)
        return PooledModelGroup(self, models)

    def _attach_model(self, name: str, slot_id: int) -> PooledModel:
        if self.actor_layout == "legacy-single-worker":
            return PooledModel(
                PooledModelSession(
                    self,
                    name,
                    worker=self.worker,
                    slot_id=slot_id,
                )
            )
        worker = self._worker_for_slot(slot_id)
        actor_future = self.client.submit(
            self._model_actor_factory,
            self.launcher,
            pool_name=self.name,
            model_name=name,
            slot_id=int(slot_id),
            actor=True,
            workers=[worker],
            allow_other_workers=False,
            pure=False,
        )
        try:
            actor = actor_future.result()
        except BaseException:
            _wait(self.launcher.close_model(name))
            raise
        self._actor_futures[name] = actor_future
        return PooledModel(
            PooledModelSession(
                self,
                name,
                actor=actor,
                actor_future=actor_future,
                worker=worker,
                slot_id=int(slot_id),
            )
        )

    def _worker_for_slot(self, slot_id: int) -> str:
        if not self._model_workers:
            return self.worker
        if self.worker_policy == "exclusive":
            return self._model_workers[int(slot_id)]
        return self._model_workers[int(slot_id) % len(self._model_workers)]

    def _release_model_actor(self, name: str) -> None:
        self._actor_futures.pop(str(name), None)

    def close(self) -> Mapping[str, Any]:
        if self._closed:
            return {"closed": True}
        failures: list[BaseException] = []
        if self.actor_layout != "legacy-single-worker":
            for model in tuple(self._models.values()):
                if model.submit._closed:
                    continue
                try:
                    _wait(model.submit.close())
                except BaseException as exc:
                    failures.append(exc)
        try:
            result = _wait(self.launcher.close())
        finally:
            self._closed = True
            for model in self._models.values():
                model.submit._closed = True
        if failures:
            raise RuntimeError(
                f"failed to close {len(failures)} pooled model actor(s)"
            ) from failures[0]
        return result

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("persistent model pool is closed")

    def __enter__(self) -> "PersistentModelPool":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "PersistentModelPool",
    "PersistentPoolActor",
    "PooledDaskRequest",
    "PooledModel",
    "PooledModelGroup",
    "PooledModelSession",
]
