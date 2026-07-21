"""Dask task fan-out over restartable 24-rank CAM segments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

from ..model.checkpoint import CheckpointBundle
from ..model.experiment import BranchSpec


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
    branch: BranchSpec
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


@dataclass(frozen=True, slots=True)
class DaskRunResult:
    """Small metadata plus an immutable in-memory distributed snapshot."""

    branch: str
    parent_branch: str | None
    run_dir: str
    history_dir: str
    checkpoint_dir: str
    log_path: str
    pbs_job_id: str | None
    stats: Mapping[str, Any]
    snapshot: CheckpointBundle

    @property
    def snapshot_nbytes(self) -> int:
        return self.snapshot.nbytes

    def describe(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "parent_branch": self.parent_branch,
            "step": self.stats.get("step"),
            "history_samples": self.stats.get("history_samples"),
            "checkpoint_dir": self.checkpoint_dir,
            "snapshot_nbytes": self.snapshot_nbytes,
            "pbs_job_id": self.pbs_job_id,
            "log_path": self.log_path,
        }


TaskRunner = Callable[[SegmentRequest, DaskRunResult | None], DaskRunResult]


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
        task_runner: TaskRunner | None = None,
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
        self.task_runner = task_runner or run_pbs_segment

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

    def submit_base(self, branch: BranchSpec) -> Any:
        """Submit the one task whose snapshot becomes the fan-out parent."""

        request = self._request(branch)
        return self.client.submit(
            self.task_runner,
            request,
            None,
            pure=False,
        )

    def submit_branch(self, parent: Any, branch: BranchSpec) -> Any:
        """Submit one branch that depends on a parent Dask Future."""

        request = self._request(branch)
        return self.client.submit(
            self.task_runner,
            request,
            parent,
            pure=False,
        )

    def fork(
        self,
        parent: Any,
        branches: Sequence[BranchSpec],
    ) -> dict[str, Any]:
        """Fan out independent tasks from one common snapshot Future."""

        names = [branch.name for branch in branches]
        if len(names) != len(set(names)):
            raise ValueError("branch names must be unique")
        return {
            branch.name: self.submit_branch(parent, branch)
            for branch in branches
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

    def _request(self, branch: BranchSpec) -> SegmentRequest:
        return SegmentRequest(
            branch=branch,
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
        )


def run_pbs_segment(
    request: SegmentRequest,
    parent: DaskRunResult | None,
) -> DaskRunResult:
    """Materialize a Future snapshot, run one PBS segment, and return a Future snapshot."""

    branch_root = Path(request.run_root) / request.branch.name
    if branch_root.exists():
        raise FileExistsError(f"refusing to replace Dask branch: {branch_root}")
    run_dir = branch_root / "run"
    history_dir = branch_root / "history"
    checkpoint_dir = branch_root / "checkpoint"
    result_json = branch_root / "result.json"
    branch_json = branch_root / "branch.json"
    input_checkpoint: Path | None = None
    run_dir.mkdir(parents=True)
    shutil.copy2(Path(request.initial_run_dir) / "atm_in", run_dir / "atm_in")

    if parent is not None:
        input_checkpoint = parent.snapshot.materialize(
            branch_root / "input-checkpoint"
        )
        _inherit_history(Path(parent.history_dir), history_dir)
    branch_json.write_text(
        json.dumps(request.branch.as_dict(), indent=2, sort_keys=True)
    )

    log_path = (
        Path(request.log_dir)
        / f"pycam_dask_{request.branch.name}_{request.task_id}.log"
    )
    script = branch_root / "job.pbs"
    script.write_text(
        _pbs_script(
            request,
            run_dir=run_dir,
            history_dir=history_dir,
            input_checkpoint=input_checkpoint,
            checkpoint_dir=checkpoint_dir,
            branch_json=branch_json,
            result_json=result_json,
            log_path=log_path,
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
            f"{exc.stderr.strip()}{_log_tail(log_path)}"
        ) from exc
    if not result_json.is_file():
        raise RuntimeError(
            f"PBS segment produced no result file: {result_json}{_log_tail(log_path)}"
        )
    stats = json.loads(result_json.read_text())
    job_id = stats.get("pbs_job_id")
    if not job_id:
        lines = completed.stdout.strip().splitlines()
        job_id = lines[-1] if lines else None
    snapshot = CheckpointBundle.from_directory(checkpoint_dir)
    return DaskRunResult(
        branch=request.branch.name,
        parent_branch=None if parent is None else parent.branch,
        run_dir=str(run_dir),
        history_dir=str(history_dir),
        checkpoint_dir=str(checkpoint_dir),
        log_path=str(log_path),
        pbs_job_id=job_id,
        stats=stats,
        snapshot=snapshot,
    )


def _pbs_script(
    request: SegmentRequest,
    *,
    run_dir: Path,
    history_dir: Path,
    input_checkpoint: Path | None,
    checkpoint_dir: Path,
    branch_json: Path,
    result_json: Path,
    log_path: Path,
) -> str:
    arguments = [
        request.python_executable,
        "-m",
        "pycam_sima.cli",
        "run-segment",
        request.config,
        "--run-dir",
        str(run_dir),
        "--history-dir",
        str(history_dir),
        "--library",
        request.library,
        "--output-checkpoint",
        str(checkpoint_dir),
        "--branch-spec",
        str(branch_json),
        "--result-json",
        str(result_json),
    ]
    if input_checkpoint is not None:
        arguments.extend(("--input-checkpoint", str(input_checkpoint)))
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
    return f"\nPBS log ({path}):\n" + "\n".join(content[-lines:])


def _describe_result(result: DaskRunResult) -> dict[str, Any]:
    return result.describe()
