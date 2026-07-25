"""Dask task fan-out over restartable 24-rank CAM segments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
import fcntl
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
from typing import Any, Callable, Iterator, Literal, Mapping, Sequence

import numpy as np

from ..model import (
    BlockingModel,
    CCPPSuitePlan,
    ModelConfig,
    ModelGroup,
    ModelOptions,
    PlanBuilder,
)
from ..model.checkpoint import CheckpointBundle
from ..model.experiment import Action, BranchSpec, SegmentPlan
from .persistent_dask import (
    PersistentCAMActor,
    PersistentDaskRequest,
    PersistentDaskSession,
)


ExecutionMode = Literal["pbs", "allocation"]


@dataclass(frozen=True, slots=True)
class DaskPBSOptions:
    account: str = "UCUB0188"
    queue: str = "develop"
    walltime: str = "00:30:00"
    ranks: int = 24
    memory: str = "80GB"

    def __post_init__(self) -> None:
        if self.ranks != 24:
            raise ValueError("the validated pycam-sima target requires 24 MPI ranks")


@dataclass(frozen=True, slots=True)
class SegmentRequest:
    plan: SegmentPlan
    task_id: str
    project_root: str
    config: str
    initial_run_dir: str
    run_root: str
    library: str
    environment_script: str
    python_executable: str
    log_dir: str
    pbs: DaskPBSOptions
    execution_mode: ExecutionMode


@dataclass(frozen=True, slots=True)
class DaskRunResult:
    """Small metadata plus an immutable in-memory distributed snapshot."""

    branch: str
    parent_branch: str | None
    run_dir: str
    history_dir: str
    checkpoint_dir: str
    log_path: str
    execution_mode: ExecutionMode
    pbs_job_id: str | None
    stats: Mapping[str, Any]
    snapshot: CheckpointBundle
    action_trace: tuple[Mapping[str, Any], ...] = ()
    segment_plan: Mapping[str, Any] = dataclass_field(default_factory=dict)

    @property
    def snapshot_nbytes(self) -> int:
        return self.snapshot.nbytes

    def describe(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "parent_branch": self.parent_branch,
            "step": self.stats.get("step"),
            "history_samples": self.stats.get("history_samples"),
            "run_dir": self.run_dir,
            "history_dir": self.history_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "snapshot_nbytes": self.snapshot_nbytes,
            "execution_mode": self.execution_mode,
            "pbs_job_id": self.pbs_job_id,
            "log_path": self.log_path,
            "action_count": len(self.action_trace),
            "action_trace": [dict(record) for record in self.action_trace],
            "segment_plan": dict(self.segment_plan),
            "plugins": tuple(self.stats.get("plugins", ())),
        }


TaskRunner = Callable[[SegmentRequest, DaskRunResult | None], DaskRunResult]
PersistentActorFactory = Callable[..., Any]


def _coerce_plan(value: SegmentPlan | PlanBuilder) -> SegmentPlan:
    if isinstance(value, PlanBuilder):
        return value.build()
    if isinstance(value, SegmentPlan):
        return value
    raise TypeError("plan must be SegmentPlan or PlanBuilder")


class DaskExperimentClient:
    """Submit a common CAM state and independent restart branches to Dask."""

    def __init__(
        self,
        client: Any,
        *,
        config: str | Path,
        initial_run_dir: str | Path,
        run_root: str | Path,
        library: str | Path | None = None,
        environment_script: str | Path | None = None,
        python_executable: str | Path | None = None,
        log_dir: str | Path | None = None,
        pbs: DaskPBSOptions | None = None,
        execution_mode: ExecutionMode = "pbs",
        task_runner: TaskRunner | None = None,
        persistent_actor_factory: PersistentActorFactory | None = None,
    ) -> None:
        if not callable(getattr(client, "submit", None)):
            raise TypeError("client must provide dask.distributed.Client.submit")
        project = Path(__file__).resolve().parents[3]
        self.client = client
        self.config = Path(config).resolve()
        self.initial_run_dir = Path(initial_run_dir).resolve()
        self.run_root = Path(run_root).resolve()
        self.library = Path(
            library or project / "build" / "libpycam_sima_kernels.so"
        ).resolve()
        self.environment_script = Path(
            environment_script
            or project
            / "reference/cases/FKESSLER_ne3pg3_gnu_24x50/.env_mach_specific.sh"
        ).resolve()
        self.python_executable = Path(
            os.path.abspath(os.fspath(python_executable or sys.executable))
        )
        self.log_dir = Path(log_dir or project / "logs").resolve()
        self.pbs = pbs or DaskPBSOptions()
        if execution_mode not in ("pbs", "allocation"):
            raise ValueError("execution_mode must be 'pbs' or 'allocation'")
        self.execution_mode: ExecutionMode = execution_mode
        runners: dict[ExecutionMode, TaskRunner] = {
            "pbs": run_pbs_segment,
            "allocation": run_allocation_segment,
        }
        self.task_runner = task_runner or runners[execution_mode]
        self.persistent_actor_factory = persistent_actor_factory or PersistentCAMActor

        required = (
            self.config,
            self.initial_run_dir / "atm_in",
            self.library,
            self.environment_script,
            self.python_executable,
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Dask experiment inputs are absent: {missing}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def plan(
        self, name: str, *, experimental: bool = False
    ) -> PlanBuilder:
        """Create a Pythonic, serializable action-plan builder."""

        return PlanBuilder(name, experimental=experimental)

    def submit_base(
        self, branch: BranchSpec | SegmentPlan | PlanBuilder
    ) -> Any:
        """Submit the one task whose snapshot becomes the fan-out parent."""

        request = self._request(branch)
        return self.client.submit(
            self.task_runner,
            request,
            None,
            pure=False,
        )

    def submit_branch(
        self,
        parent: Any,
        branch: BranchSpec | SegmentPlan | PlanBuilder,
    ) -> Any:
        """Submit one branch that depends on a parent Dask Future."""

        request = self._request(branch)
        return self.client.submit(
            self.task_runner,
            request,
            parent,
            pure=False,
        )

    def submit_plan(
        self, parent: Any, plan: SegmentPlan | PlanBuilder
    ) -> Any:
        """Submit one multi-action MPI segment from a parent snapshot."""

        return self.submit_branch(parent, _coerce_plan(plan))

    def submit_action(
        self,
        parent: Any,
        *,
        name: str,
        action: Action,
        unsafe: bool = True,
    ) -> Any:
        """Submit one action as its own checkpointed MPI segment."""

        return self.submit_plan(
            parent,
            SegmentPlan(name=name, actions=(action,), unsafe=unsafe),
        )

    def fork(
        self,
        parent: Any,
        branches: Sequence[BranchSpec | SegmentPlan | PlanBuilder],
    ) -> dict[str, Any]:
        """Fan out independent tasks from one common snapshot Future."""

        normalized = tuple(
            branch
            if isinstance(branch, BranchSpec)
            else _coerce_plan(branch)
            for branch in branches
        )
        names = [branch.name for branch in normalized]
        if len(names) != len(set(names)):
            raise ValueError("branch names must be unique")
        return {
            branch.name: self.submit_branch(parent, branch)
            for branch in normalized
        }

    def gather(self, branches: Mapping[str, Any]) -> dict[str, DaskRunResult]:
        """Download complete results, including all in-memory snapshots."""

        values = self.client.gather(list(branches.values()))
        return dict(zip(branches, values))

    def summary(self, result: Any) -> Any:
        """Return a Future for small metadata without downloading the snapshot."""

        return self.client.submit(_describe_result, result, pure=False)

    def summaries(self, branches: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        futures = [self.summary(result) for result in branches.values()]
        values = self.client.gather(futures)
        return dict(zip(branches, values))

    def field(self, result: Any, name: str, *, rank: int = 0) -> Any:
        """Return a Future for one rank-local field, not the full snapshot."""

        if not 0 <= rank < self.pbs.ranks:
            raise ValueError(
                f"rank must be between 0 and {self.pbs.ranks - 1}, got {rank}"
            )
        return self.client.submit(
            _extract_checkpoint_field,
            result,
            str(name),
            int(rank),
            pure=False,
        )

    def start_persistent(
        self,
        name: str,
        *,
        worker: str | None = None,
        launch_mode: str | None = None,
        options: ModelOptions | None = None,
        scheme_plan: CCPPSuitePlan | None = None,
        startup_timeout: float = 900.0,
        request_timeout: float = 600.0,
    ) -> PersistentDaskSession:
        """Create one worker-pinned Actor and launch its MPI model once.

        The call waits only for Actor/MPI initialization. Methods on the
        returned proxy are asynchronous and return Dask ``ActorFuture`` values.
        """

        request, selected_worker = self._persistent_request(
            name,
            worker=worker,
            launch_mode=launch_mode,
            options=options,
            scheme_plan=scheme_plan,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
        )
        actor_future = self._submit_persistent_actor(
            request,
            selected_worker,
        )
        actor = actor_future.result()
        return PersistentDaskSession(
            self.client,
            actor,
            actor_future,
            worker=selected_worker,
            name=name,
        )

    def submit_persistent(
        self,
        name: str,
        **kwargs: Any,
    ) -> PersistentDaskSession:
        """Alias for :meth:`start_persistent` matching Dask submit vocabulary."""

        return self.start_persistent(name, **kwargs)

    def open_persistent(
        self,
        name: str,
        **kwargs: Any,
    ) -> BlockingModel:
        """Open a blocking, context-manager-friendly persistent model.

        Use :meth:`start_persistent` or ``model.submit`` when ActorFuture
        control is wanted explicitly.
        """

        return self.start_persistent(name, **kwargs).sync

    def model(self, name: str, **kwargs: Any) -> BlockingModel:
        """Open one persistent MPI model as a blocking context manager."""

        return self.open_persistent(name, **kwargs)

    def fork_persistent(
        self,
        parent: PersistentDaskSession | BlockingModel,
        branches: Sequence[BranchSpec | SegmentPlan | PlanBuilder],
        *,
        workers: Mapping[str, str] | None = None,
        launch_mode: str | None = None,
        startup_timeout: float = 900.0,
        request_timeout: float = 600.0,
        close_parent: bool = False,
    ) -> dict[str, PersistentDaskSession]:
        """Fork independent live MPI models from one immutable memory snapshot.

        The parent snapshot remains in Dask distributed memory. Each child
        receives the same bytes, restores private rank-local arrays, applies
        its branch plan, and keeps its MPI ranks alive for later Actor calls.
        ``close_parent=True`` releases the parent MPI job after Dask has
        retained the snapshot and before child Actors are launched.
        """

        if isinstance(parent, BlockingModel):
            parent = parent.submit
        if not isinstance(parent, PersistentDaskSession):
            raise TypeError("persistent parent must be PersistentDaskSession")
        if parent.client is not self.client:
            raise ValueError("persistent parent belongs to a different Dask client")
        if self.execution_mode != "pbs":
            raise RuntimeError(
                "persistent in-memory fork currently requires execution_mode='pbs'; "
                "allocation mode reserves one full-node MPI world"
            )
        plans = tuple(
            branch.to_segment_plan()
            if isinstance(branch, BranchSpec)
            else _coerce_plan(branch)
            for branch in branches
        )
        if not plans:
            raise ValueError("persistent fork requires at least one branch")
        if not all(isinstance(plan, SegmentPlan) for plan in plans):
            raise TypeError("persistent branches must be BranchSpec or SegmentPlan")
        names = [plan.name for plan in plans]
        if len(names) != len(set(names)):
            raise ValueError("persistent branch names must be unique")
        if parent.name in names:
            raise ValueError("persistent child name must differ from parent name")

        available_workers = self._persistent_workers()
        if workers is not None:
            unknown = set(workers) - set(names)
            if unknown:
                raise ValueError(
                    f"worker assignments contain unknown branches: {sorted(unknown)}"
                )
            missing = set(names) - set(workers)
            if missing:
                raise ValueError(
                    f"worker assignments omit branches: {sorted(missing)}"
                )
            selected_workers = {
                name: self._persistent_worker(workers[name]) for name in names
            }
        else:
            if len(available_workers) < len(names):
                raise RuntimeError(
                    "persistent in-memory fork requires at least one Dask worker "
                    f"per child; need {len(names)}, found {len(available_workers)}"
                )
            selected_workers = dict(zip(names, available_workers))
        if len(set(selected_workers.values())) != len(names):
            raise ValueError(
                "persistent children require distinct Dask workers so their "
                "blocking Actor calls can run concurrently"
            )

        snapshot_future = self.client.submit(
            _capture_persistent_checkpoint,
            parent.actor,
            pure=False,
            workers=[parent.worker],
            allow_other_workers=False,
        )
        if close_parent:
            snapshot_size = self.client.submit(
                _checkpoint_nbytes,
                snapshot_future,
                pure=False,
                workers=[parent.worker],
                allow_other_workers=False,
            )
            snapshot_size.result()
            parent.close().result()
        actor_futures: dict[str, Any] = {}
        for plan in plans:
            request, _selected = self._persistent_request(
                plan.name,
                worker=selected_workers[plan.name],
                launch_mode=launch_mode,
                options=None,
                scheme_plan=None,
                startup_timeout=startup_timeout,
                request_timeout=request_timeout,
                parent_name=parent.name,
                startup_plan=plan,
            )
            actor_futures[plan.name] = self._submit_persistent_actor(
                request,
                selected_workers[plan.name],
                snapshot=snapshot_future,
            )

        children: dict[str, PersistentDaskSession] = {}
        try:
            for name in names:
                actor_future = actor_futures[name]
                actor = actor_future.result()
                children[name] = PersistentDaskSession(
                    self.client,
                    actor,
                    actor_future,
                    worker=selected_workers[name],
                    name=name,
                )
        except BaseException:
            for child in children.values():
                try:
                    child.close().result()
                except BaseException:
                    pass
            for name, actor_future in actor_futures.items():
                if name in children:
                    continue
                try:
                    actor = actor_future.result()
                    actor.close().result()
                except BaseException:
                    pass
            raise
        return children

    def fork_models(
        self,
        parent: PersistentDaskSession | BlockingModel,
        branches: Sequence[BranchSpec | SegmentPlan | PlanBuilder],
        **kwargs: Any,
    ) -> ModelGroup:
        """Fork live models and return blocking Pythonic controllers."""

        children = self.fork_persistent(parent, branches, **kwargs)
        return ModelGroup(
            {name: child.sync for name, child in children.items()}
        )

    def _persistent_worker(self, worker: str | None) -> str:
        workers = self._persistent_workers()
        if worker is not None:
            if worker not in workers:
                raise ValueError(
                    f"Dask worker {worker!r} is unavailable; choose one of "
                    f"{workers}"
                )
            return worker
        return workers[0]

    def _persistent_workers(self) -> tuple[str, ...]:
        scheduler_info = getattr(self.client, "scheduler_info", None)
        if not callable(scheduler_info):
            raise TypeError("persistent Dask execution requires Client.scheduler_info")
        workers = tuple(sorted(scheduler_info().get("workers", {})))
        if not workers:
            raise RuntimeError("Dask has no workers for the persistent CAM Actor")
        return workers

    def _persistent_request(
        self,
        name: str,
        *,
        worker: str | None,
        launch_mode: str | None,
        options: ModelOptions | None,
        scheme_plan: CCPPSuitePlan | None,
        startup_timeout: float,
        request_timeout: float,
        parent_name: str | None = None,
        startup_plan: SegmentPlan | None = None,
    ) -> tuple[PersistentDaskRequest, str]:
        selected_worker = self._persistent_worker(worker)
        selected_launch_mode = launch_mode or (
            "local" if self.execution_mode == "allocation" else "pbs"
        )
        config = ModelConfig.from_yaml(self.config)
        selected_options = options or ModelOptions.from_config(config)
        selected_options.validate(config)
        selected_scheme_plan = (
            scheme_plan
            or CCPPSuitePlan.from_xml(config.verify_suite())
        )
        request = PersistentDaskRequest(
            name=name,
            config=str(self.config),
            initial_run_dir=str(self.initial_run_dir),
            run_root=str(self.run_root),
            library=str(self.library),
            environment_script=str(self.environment_script),
            python_executable=str(self.python_executable),
            log_dir=str(self.log_dir),
            ranks=self.pbs.ranks,
            launch_mode=selected_launch_mode,
            pbs_account=self.pbs.account,
            pbs_queue=self.pbs.queue,
            pbs_walltime=self.pbs.walltime,
            startup_timeout=float(startup_timeout),
            request_timeout=float(request_timeout),
            options={
                "timestep_seconds": selected_options.timestep_seconds,
                "physics_profile": selected_options.physics_profile,
                "mediator_present": selected_options.mediator_present,
            },
            scheme_plan=selected_scheme_plan.to_payload(),
            execution_mode=self.execution_mode,
            parent_name=parent_name,
            startup_plan=(
                None if startup_plan is None else startup_plan.as_dict()
            ),
        )
        return request, selected_worker

    def _submit_persistent_actor(
        self,
        request: PersistentDaskRequest,
        worker: str,
        *,
        snapshot: Any | None = None,
    ) -> Any:
        arguments = (
            (self.persistent_actor_factory, request)
            if snapshot is None
            else (self.persistent_actor_factory, request, snapshot)
        )
        return self.client.submit(
            *arguments,
            actor=True,
            workers=[worker],
            allow_other_workers=False,
            pure=False,
        )

    def _request(
        self, branch: BranchSpec | SegmentPlan | PlanBuilder
    ) -> SegmentRequest:
        plan = (
            branch.to_segment_plan()
            if isinstance(branch, BranchSpec)
            else _coerce_plan(branch)
        )
        if not isinstance(plan, SegmentPlan):
            raise TypeError("Dask branches must be BranchSpec or SegmentPlan")
        return SegmentRequest(
            plan=plan,
            task_id=secrets.token_hex(6),
            project_root=str(Path(__file__).resolve().parents[3]),
            config=str(self.config),
            initial_run_dir=str(self.initial_run_dir),
            run_root=str(self.run_root),
            library=str(self.library),
            environment_script=str(self.environment_script),
            python_executable=str(self.python_executable),
            log_dir=str(self.log_dir),
            pbs=self.pbs,
            execution_mode=self.execution_mode,
        )


@dataclass(frozen=True, slots=True)
class _SegmentPaths:
    branch_root: Path
    run_dir: Path
    history_dir: Path
    checkpoint_dir: Path
    result_json: Path
    plan_json: Path
    input_checkpoint: Path | None
    log_path: Path


def run_pbs_segment(
    request: SegmentRequest,
    parent: DaskRunResult | None,
) -> DaskRunResult:
    """Run one PBS segment from a materialized Future snapshot."""

    if request.execution_mode != "pbs":
        raise ValueError("run_pbs_segment requires execution_mode='pbs'")
    paths = _prepare_segment(request, parent)
    script = paths.branch_root / "job.pbs"
    script.write_text(
        _pbs_script(
            request,
            run_dir=paths.run_dir,
            history_dir=paths.history_dir,
            input_checkpoint=paths.input_checkpoint,
            checkpoint_dir=paths.checkpoint_dir,
            plan_json=paths.plan_json,
            result_json=paths.result_json,
            log_path=paths.log_path,
        )
    )
    try:
        completed = subprocess.run(
            ("qsub", "-W", "block=true", str(script)),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Dask CAM PBS segment failed to submit or complete: "
            f"{exc.stderr.strip()}{_log_tail(paths.log_path)}"
        ) from exc
    stats = _read_segment_stats(paths, label="PBS")
    job_id = stats.get("pbs_job_id")
    if not job_id:
        lines = completed.stdout.strip().splitlines()
        job_id = lines[-1] if lines else None
    return _segment_result(request, parent, paths, stats, job_id=job_id)


def run_allocation_segment(
    request: SegmentRequest,
    parent: DaskRunResult | None,
) -> DaskRunResult:
    """Run one MPI segment directly inside the Dask controller's PBS allocation."""

    if request.execution_mode != "allocation":
        raise ValueError("run_allocation_segment requires execution_mode='allocation'")
    outer_job_id = os.environ.get("PBS_JOBID")
    if not outer_job_id:
        raise RuntimeError(
            "allocation execution requires an active PBS allocation "
            "(PBS_JOBID is unset)"
        )

    paths = _prepare_segment(request, parent)
    environment = _sourced_environment(Path(request.environment_script))
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
        }
    )
    command = [
        *_allocation_launcher(environment, request.pbs.ranks),
        *_segment_arguments(request, paths),
    ]
    try:
        with _allocation_mpi_lock(Path(request.run_root)):
            with paths.log_path.open("w") as log:
                subprocess.run(
                    command,
                    check=True,
                    cwd=request.project_root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Dask CAM allocation segment failed: "
            f"command exited with {exc.returncode}{_log_tail(paths.log_path)}"
        ) from exc

    stats = _read_segment_stats(paths, label="allocation")
    job_id = stats.get("pbs_job_id") or outer_job_id
    if job_id != outer_job_id:
        raise RuntimeError(
            f"segment reported PBS job {job_id}, expected outer job {outer_job_id}"
        )
    return _segment_result(request, parent, paths, stats, job_id=job_id)


def _prepare_segment(
    request: SegmentRequest,
    parent: DaskRunResult | None,
) -> _SegmentPaths:
    branch_root = Path(request.run_root) / request.plan.name
    if branch_root.exists():
        raise FileExistsError(f"refusing to replace Dask branch: {branch_root}")
    run_dir = branch_root / "run"
    history_dir = branch_root / "history"
    checkpoint_dir = branch_root / "checkpoint"
    result_json = branch_root / "result.json"
    plan_json = branch_root / "segment-plan.json"
    input_checkpoint: Path | None = None
    run_dir.mkdir(parents=True)
    shutil.copy2(Path(request.initial_run_dir) / "atm_in", run_dir / "atm_in")

    if parent is not None:
        input_checkpoint = parent.snapshot.materialize(branch_root / "input-checkpoint")
        _inherit_history(Path(parent.history_dir), history_dir)
    plan_json.write_text(json.dumps(request.plan.as_dict(), indent=2, sort_keys=True))
    log_path = Path(request.log_dir) / (
        f"pycam_dask_{request.execution_mode}_{request.plan.name}_"
        f"{request.task_id}.log"
    )
    return _SegmentPaths(
        branch_root=branch_root,
        run_dir=run_dir,
        history_dir=history_dir,
        checkpoint_dir=checkpoint_dir,
        result_json=result_json,
        plan_json=plan_json,
        input_checkpoint=input_checkpoint,
        log_path=log_path,
    )


def _segment_arguments(
    request: SegmentRequest,
    paths: _SegmentPaths,
) -> list[str]:
    arguments = [
        request.python_executable,
        "-m",
        "pycam_sima.cli",
        "run-segment",
        request.config,
        "--run-dir",
        str(paths.run_dir),
        "--history-dir",
        str(paths.history_dir),
        "--library",
        request.library,
        "--output-checkpoint",
        str(paths.checkpoint_dir),
        "--segment-plan",
        str(paths.plan_json),
        "--result-json",
        str(paths.result_json),
    ]
    if paths.input_checkpoint is not None:
        arguments.extend(("--input-checkpoint", str(paths.input_checkpoint)))
    return arguments


def _read_segment_stats(paths: _SegmentPaths, *, label: str) -> dict[str, Any]:
    if not paths.result_json.is_file():
        raise RuntimeError(
            f"{label} segment produced no result file: "
            f"{paths.result_json}{_log_tail(paths.log_path)}"
        )
    return json.loads(paths.result_json.read_text())


def _segment_result(
    request: SegmentRequest,
    parent: DaskRunResult | None,
    paths: _SegmentPaths,
    stats: Mapping[str, Any],
    *,
    job_id: str | None,
) -> DaskRunResult:
    return DaskRunResult(
        branch=request.plan.name,
        parent_branch=None if parent is None else parent.branch,
        run_dir=str(paths.run_dir),
        history_dir=str(paths.history_dir),
        checkpoint_dir=str(paths.checkpoint_dir),
        log_path=str(paths.log_path),
        execution_mode=request.execution_mode,
        pbs_job_id=job_id,
        stats=stats,
        snapshot=CheckpointBundle.from_directory(paths.checkpoint_dir),
        action_trace=tuple(stats.get("action_trace", ())),
        segment_plan=dict(stats.get("segment_plan", request.plan.as_dict())),
    )


def _sourced_environment(script: Path) -> dict[str, str]:
    try:
        completed = subprocess.run(
            (
                "bash",
                "-c",
                'source "$1" >/dev/null 2>&1 && env -0',
                "pycam-sima-environment",
                str(script),
            ),
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        message = os.fsdecode(exc.stderr).strip()
        raise RuntimeError(
            f"failed to source machine environment {script}: {message}"
        ) from exc
    environment: dict[str, str] = {}
    for entry in completed.stdout.split(b"\0"):
        if not entry:
            continue
        key, separator, value = entry.partition(b"=")
        if separator:
            environment[os.fsdecode(key)] = os.fsdecode(value)
    return environment


def _allocation_launcher(environment: Mapping[str, str], ranks: int) -> list[str]:
    if environment.get("PBS_NODEFILE"):
        return ["mpiexec", "-n", str(ranks)]
    return [
        "mpiexec",
        "--hosts",
        socket.gethostname(),
        "--no-vni",
        "-n",
        str(ranks),
    ]


@contextmanager
def _allocation_mpi_lock(run_root: Path) -> Iterator[None]:
    """Serialize full-node MPI launches within one allocation."""

    run_root.mkdir(parents=True, exist_ok=True)
    with (run_root / ".allocation-mpi.lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _pbs_script(
    request: SegmentRequest,
    *,
    run_dir: Path,
    history_dir: Path,
    input_checkpoint: Path | None,
    checkpoint_dir: Path,
    plan_json: Path,
    result_json: Path,
    log_path: Path,
) -> str:
    paths = _SegmentPaths(
        branch_root=run_dir.parent,
        run_dir=run_dir,
        history_dir=history_dir,
        checkpoint_dir=checkpoint_dir,
        result_json=result_json,
        plan_json=plan_json,
        input_checkpoint=input_checkpoint,
        log_path=log_path,
    )
    arguments = _segment_arguments(request, paths)
    command = shlex.join(arguments)
    return (
        "#!/bin/bash\n"
        "#PBS -N pycam_dask\n"
        f"#PBS -A {request.pbs.account}\n"
        f"#PBS -q {request.pbs.queue}\n"
        f"#PBS -l select=1:ncpus={request.pbs.ranks}:"
        f"mpiprocs={request.pbs.ranks}:ompthreads=1:mem={request.pbs.memory}\n"
        f"#PBS -l walltime={request.pbs.walltime}\n"
        "#PBS -j oe\n"
        f"#PBS -o {log_path}\n"
        "set -euo pipefail\n"
        "export OMP_NUM_THREADS=1\n"
        "export PYTHONUNBUFFERED=1\n"
        "export PYTHONFAULTHANDLER=1\n"
        f"cd {shlex.quote(request.project_root)}\n"
        f"source {shlex.quote(request.environment_script)} >/dev/null 2>&1\n"
        f"exec mpiexec -n {request.pbs.ranks} {command}\n"
    )


def _inherit_history(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    if not source.exists():
        return
    for original in source.glob("*.nc"):
        target = destination / original.name
        try:
            os.link(original, target)
        except OSError:
            shutil.copy2(original, target)


def _log_tail(path: Path, lines: int = 60) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(errors="replace").splitlines()
    return f"\nSegment log ({path}):\n" + "\n".join(content[-lines:])


def _describe_result(result: DaskRunResult) -> dict[str, Any]:
    return result.describe()


def _capture_persistent_checkpoint(actor: Any) -> CheckpointBundle:
    """Keep an Actor snapshot in Dask memory instead of the notebook process."""

    snapshot = actor.memory_checkpoint().result()
    if not isinstance(snapshot, CheckpointBundle):
        raise TypeError("persistent actor returned an invalid memory checkpoint")
    return snapshot


def _checkpoint_nbytes(snapshot: CheckpointBundle) -> int:
    """Wait for a distributed snapshot without downloading its contents."""

    return snapshot.nbytes


def _extract_checkpoint_field(
    result: DaskRunResult, name: str, rank: int
) -> np.ndarray:
    """Extract one canonical field or zero-copy alias from a Future snapshot."""

    from ..model.contracts import default_alias_rules, default_contracts

    filename = f"rank-{rank:03d}.npz"
    try:
        content = next(
            content
            for stored_name, content in result.snapshot.files
            if stored_name == filename
        )
    except StopIteration as exc:
        raise KeyError(
            f"checkpoint for rank {rank} is absent from branch {result.branch!r}"
        ) from exc

    direct_aliases = {
        alias: contract.standard_name
        for contract in default_contracts()
        for alias in contract.aliases
    }
    try:
        manifest_bytes = next(
            content
            for stored_name, content in result.snapshot.files
            if stored_name == "manifest.json"
        )
        manifest = json.loads(manifest_bytes)
        rank_record = next(
            item for item in manifest["ranks"]
            if int(item["rank"]) == rank
        )
        for contract in rank_record.get("contracts", ()):
            for alias in contract.get("aliases", ()):
                direct_aliases[str(alias)] = str(
                    contract["standard_name"]
                )
    except (StopIteration, KeyError, TypeError, ValueError):
        # Schema-v1 snapshots have only the built-in alias catalog.
        pass
    alias_rules = {rule.alias: rule for rule in default_alias_rules()}
    with np.load(BytesIO(content), allow_pickle=False) as stored:
        if name in stored.files:
            value = stored[name]
        elif name in direct_aliases:
            value = stored[direct_aliases[name]]
        elif name in alias_rules:
            rule = alias_rules[name]
            value = stored[rule.target]
            if rule.index is not None:
                selector = [slice(None)] * value.ndim
                selector[rule.axis] = rule.index
                value = value[tuple(selector)]
        else:
            raise KeyError(f"unknown checkpoint field {name!r}")
        return np.array(value, dtype=value.dtype, order="F", copy=True)
