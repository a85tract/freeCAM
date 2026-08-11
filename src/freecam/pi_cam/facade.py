"""FreeCESM-style user interface for the real persistent PI-CAM runtime.

This module intentionally keeps machine setup out of notebooks.  ``Driver``
prepares an isolated CAM run directory and lazily starts one persistent MPI
session when the user first touches live model state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import uuid4
from xml.etree import ElementTree

from .config import PICAMConfig
from .session import PICAMNotebookSession


def _case_pbs_account(case_root: Path) -> str | None:
    """Read the allocation recorded by a configured CESM reference case."""

    batch_config = case_root / "env_batch.xml"
    if not batch_config.is_file():
        return None
    try:
        root = ElementTree.parse(batch_config).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"invalid CESM batch configuration: {batch_config}") from exc
    values: dict[str, str] = {}
    for entry in root.iter("entry"):
        key = entry.get("id")
        value = entry.get("value")
        if key and value and not value.startswith("$"):
            values[key] = value
    return values.get("CHARGE_ACCOUNT") or values.get("PROJECT")


def _resolve_pbs_account(explicit: str | None, case_root: Path) -> str | None:
    """Resolve an allocation without embedding a user or project in source."""

    if explicit:
        return explicit
    machine_specific = os.environ.get("PBS_ACCOUNT_DERECHO")
    if machine_specific not in {None, "", "N/A"}:
        return machine_specific
    case_account = _case_pbs_account(case_root)
    if case_account:
        return case_account
    # A generic PBS_ACCOUNT is reliable only inside an existing allocation.
    # Login-shell profiles may export an account for another NCAR machine.
    if os.environ.get("PBS_JOBID") or os.environ.get("PBS_NODEFILE"):
        allocation_account = os.environ.get("PBS_ACCOUNT")
        if allocation_account not in {None, "", "N/A"}:
            return allocation_account
    return None


@dataclass(frozen=True, slots=True)
class Variable:
    """Definition assigned to ``driver.cam.state.<name>``.

    A real MPI field needs named model dimensions rather than one rank-0
    ``numpy.zeros`` shape.  Every rank resolves these names against its own
    local StatePool dimensions and allocates its own Fortran-contiguous array.
    """

    dims: tuple[str, ...]
    units: str = "1"
    initial: float | int = 0.0
    dtype: str = "float64"
    writable: bool = True
    restart: bool = True
    aliases: tuple[str, ...] = ()
    standard_name: str | None = None

    def __init__(
        self,
        dims: Sequence[str] = (),
        *,
        units: str = "1",
        initial: float | int = 0.0,
        dtype: str = "float64",
        writable: bool = True,
        restart: bool = True,
        aliases: Sequence[str] = (),
        standard_name: str | None = None,
    ) -> None:
        object.__setattr__(self, "dims", tuple(str(item) for item in dims))
        object.__setattr__(self, "units", str(units))
        object.__setattr__(self, "initial", initial)
        object.__setattr__(self, "dtype", str(dtype))
        object.__setattr__(self, "writable", bool(writable))
        object.__setattr__(self, "restart", bool(restart))
        object.__setattr__(self, "aliases", tuple(str(item) for item in aliases))
        object.__setattr__(self, "standard_name", standard_name)


class Physics:
    """Base class for one Notebook-defined rank-local Python process."""

    name: str | None = None
    phase: str = "cam_run1"
    before: str | None = None
    after: str | None = None
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    enabled: bool = True
    transactional: bool = True

    def tendency(self, fields: Any, context: Any) -> None:
        raise NotImplementedError

    def _install(
        self,
        session: PICAMNotebookSession,
        *,
        phase: str | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> Any:
        process_name = self.name or type(self).__name__.lower()
        if before is not None or after is not None:
            placement_before = before
            placement_after = after
        else:
            placement_before = self.before
            placement_after = self.after
        return session.physics.install_python(
            self.tendency,
            name=process_name,
            phase=phase or self.phase,
            before=placement_before,
            after=placement_after,
            reads=self.reads,
            writes=self.writes,
            enabled=self.enabled,
            transactional=self.transactional,
        )


@dataclass(frozen=True, slots=True)
class PICAMCaseInfo:
    """Compact case description displayed by the high-level driver."""

    key: str
    config: PICAMConfig

    def __str__(self) -> str:
        return (
            f"{self.key}: {self.config.resolution} {self.config.physics_package.upper()} "
            f"with {self.config.dynamics.upper()} dynamics, "
            f"{self.config.mpi_size} MPI ranks, {self.config.calendar} calendar"
        )


class _CAMFacade:
    """Lazy FreeCAM handle exposed as ``driver.cam``."""

    def __init__(self, driver: "Driver") -> None:
        self._driver = driver

    @property
    def state(self) -> Any:
        return self._driver._live_session().state

    @property
    def workflow(self) -> Any:
        return self._driver._live_session().workflow

    @property
    def fields(self) -> Any:
        return self._driver._live_session().fields

    @property
    def physics(self) -> Any:
        return self._driver._live_session().physics

    @property
    def phases(self) -> Any:
        return self._driver._live_session().phases

    @property
    def kernels(self) -> Any:
        return self._driver._live_session().kernels

    @property
    def status(self) -> Mapping[str, Any]:
        return self._driver.status

    def advance(self, steps: int = 1) -> Mapping[str, Any]:
        return self._driver.advance(steps)


class Driver:
    """High-level PI-CAM interface modelled after ``freecesm.Driver``.

    ``Driver`` is intentionally lazy: constructing it performs no PBS or MPI
    work.  The first live operation prepares a private run directory and starts
    one persistent session; every later operation reuses those MPI ranks.
    """

    _CASE_CONFIGS = {"PI-atm": "configs/pi_cam_icesm131.yaml"}

    def __init__(
        self,
        case: str = "PI-atm",
        nsteps: int = 10,
        *,
        repo: str | Path | None = None,
        config: str | Path | None = None,
        scratch: str | Path | None = None,
        reference_case: str | Path | None = None,
        reference_run: str | Path | None = None,
        boundary: str | Path | None = None,
        run_dir: str | Path | None = None,
        launch_mode: str = "auto",
        account: str | None = None,
        queue: str = "develop",
        walltime: str = "02:00:00",
        python_executable: str | Path | None = None,
        session_factory: Any = PICAMNotebookSession,
    ) -> None:
        if int(nsteps) < 1:
            raise ValueError("nsteps must be positive")
        self.repo = Path(repo or Path(__file__).resolve().parents[3]).resolve()
        if config is None:
            try:
                config = self.repo / self._CASE_CONFIGS[case]
            except KeyError as exc:
                raise ValueError(
                    f"unsupported case {case!r}; available cases: "
                    + ", ".join(self._CASE_CONFIGS)
                ) from exc
        self.config_path = Path(config).expanduser().resolve()
        self.config = PICAMConfig.from_yaml(self.config_path)
        if int(nsteps) > self.config.stop_n:
            raise ValueError(
                f"nsteps={nsteps} exceeds the {self.config.stop_n}-step replay boundary"
            )
        self.case = PICAMCaseInfo(case, self.config)
        self.nsteps = int(nsteps)
        self.scratch = Path(
            scratch
            or os.environ.get("SCRATCH")
            or f"/glade/derecho/scratch/{os.environ.get('USER', 'unknown')}"
        ).expanduser().resolve()
        case_name = self.config.case_name
        self.reference_case = Path(
            reference_case
            or self.repo.parent / "CESM_cases" / case_name
        ).expanduser().resolve()
        self.reference_run = Path(
            reference_run
            or self.scratch / "pyCAM" / "PI-cam" / case_name / "run"
        ).expanduser().resolve()
        self.boundary = Path(
            boundary
            or self.scratch
            / "pyCAM"
            / "PI-cam"
            / "nonpic-boundary-capture-50step"
            / "boundary"
            / "replay"
        ).expanduser().resolve()
        self._requested_run_dir = (
            None if run_dir is None else Path(run_dir).expanduser().resolve()
        )
        self.launch_mode = launch_mode
        self.account = _resolve_pbs_account(account, self.reference_case)
        self.queue = queue
        self.walltime = walltime
        # Do not resolve the final ``.venv/bin/python`` symlink: Python uses
        # that invocation path to select the virtual environment's site-packages.
        self.python_executable = Path(
            python_executable or self.repo / ".venv" / "bin" / "python"
        ).expanduser().absolute()
        self._session_factory = session_factory
        self._session: PICAMNotebookSession | None = None
        self._run_dir: Path | None = None
        self.cam = _CAMFacade(self)

    @property
    def running(self) -> bool:
        return self._session is not None and bool(self._session.running)

    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    @property
    def status(self) -> Mapping[str, Any]:
        return self._live_session().status

    @property
    def validation(self) -> Mapping[str, Any]:
        """Return the committed 50-step BFB evidence without submitting a job."""

        path = (
            self.repo
            / "validation"
            / "pi_cam_python_zero_copy_state_vs_oracle_50step_bfb.json"
        )
        if not path.is_file():
            return {"available": False, "path": str(path)}
        return {"available": True, "path": str(path), **json.loads(path.read_text())}

    def initialize(self) -> "Driver":
        self._live_session()
        return self

    def advance(self, steps: int = 1) -> Mapping[str, Any]:
        if int(steps) < 1:
            raise ValueError("steps must be positive")
        return self._live_session().advance(steps=int(steps))

    def execute(
        self,
        steps: int | None = None,
        *,
        verbose: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        """Run complete CAM steps and return the actual worker action trace."""

        session = self._live_session()
        first = int(session.status.get("actions", 0))
        session.advance(steps=self.nsteps if steps is None else int(steps))
        trace = session.trace(since=first)
        if verbose:
            for action in trace:
                print(
                    f"step {action['model_step']:>3}  "
                    f"{action['phase']}.{action['name']}"
                )
        return trace

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "Driver":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _live_session(self) -> PICAMNotebookSession:
        if self._session is None:
            run_dir = self._prepare_run_dir()
            session = self._session_factory(
                self.config_path,
                boundary=self.boundary,
                run_dir=run_dir,
                env_script=self.reference_case / ".env_mach_specific.sh",
                python_executable=self.python_executable,
                launch_mode=self.launch_mode,
                pbs_account=self.account,
                pbs_queue=self.queue,
                pbs_walltime=self.walltime,
            )
            session.start()
            self._session = session
        return self._session

    def _prepare_run_dir(self) -> Path:
        if self._run_dir is not None:
            return self._run_dir
        if not (self.reference_run / "atm_in").is_file():
            raise FileNotFoundError(
                f"PI-CAM reference run lacks atm_in: {self.reference_run}"
            )
        if not (self.reference_case / ".env_mach_specific.sh").is_file():
            raise FileNotFoundError(
                f"PI-CAM reference case lacks .env_mach_specific.sh: "
                f"{self.reference_case}"
            )
        if not (self.boundary / "manifest.json").is_file():
            raise FileNotFoundError(
                f"PI-CAM replay boundary lacks manifest.json: {self.boundary}"
            )
        if self._requested_run_dir is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            destination = (
                self.scratch
                / "freeCAM"
                / "PI-cam"
                / f"notebook-{stamp}-{uuid4().hex[:8]}"
                / "run"
            )
        else:
            destination = self._requested_run_dir
        if destination.exists():
            if any(destination.iterdir()):
                raise FileExistsError(
                    f"PI-CAM run directory is not empty: {destination}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            self.reference_run,
            destination,
            dirs_exist_ok=True,
            ignore=self._ignore_reference_output,
        )
        (destination / "timing" / "checkpoints").mkdir(parents=True, exist_ok=True)
        self._run_dir = destination.resolve()
        return self._run_dir

    @staticmethod
    def _ignore_reference_output(directory: str, names: list[str]) -> set[str]:
        del directory
        ignored: set[str] = set()
        for name in names:
            if (
                name == "timing"
                or name.startswith("rpointer.")
                or fnmatch(name, "*.cam.*.nc")
                or fnmatch(name, "*.log.*")
            ):
                ignored.add(name)
        return ignored


__all__ = ["Driver", "Physics", "PICAMCaseInfo", "Variable"]
