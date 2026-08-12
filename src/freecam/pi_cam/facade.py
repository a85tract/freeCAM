"""FreeCESM-style user interface for the real persistent PI-CAM runtime.

This module intentionally keeps machine setup out of notebooks.  ``Driver``
prepares an isolated CAM run directory and lazily starts one persistent MPI
session when the user first touches live model state.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatch
import inspect
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterator, Mapping, Sequence
from uuid import uuid4
from xml.etree import ElementTree

from freecam.model.python_processes import PythonStateView

from .config import PICAMConfig
from .history import PICAMOutputView
from .plan import PICAMStepPlan
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
    before: str | None = None
    after: str | None = None
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    enabled: bool = True
    transactional: bool = True

    def run(self, state: PythonStateView, context: Any) -> None:
        """Run against friendly rank-local StatePool attributes."""

        raise NotImplementedError

    def tendency(self, fields: Any, context: Any) -> None:
        """Compatibility callback for the original mapping-style API."""

        if type(self).run is Physics.run:
            raise NotImplementedError(
                f"{type(self).__name__} must implement run(state, context) "
                "or tendency(fields, context)"
            )
        return self.run(PythonStateView(fields), context)

    def _install(
        self,
        session: PICAMNotebookSession,
        *,
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
            before=placement_before,
            after=placement_after,
            reads=self.reads,
            writes=self.writes,
            enabled=self.enabled,
            transactional=self.transactional,
        )


class WorkflowTemplate(list[Any]):
    """Mutable declaration of one complete, live PI-CAM process order.

    The initial contents are the validated default workflow.  A case-level
    factory receives a private copy and can use normal list operations or the
    named helpers below.  Items may be existing process handles, their names,
    or :class:`Physics` instances that should be installed at startup.
    """

    def copy(self) -> "WorkflowTemplate":
        return WorkflowTemplate(self)

    def process(self, name: str) -> Any:
        matches = [item for item in self if _workflow_item_matches(item, name)]
        if len(matches) != 1:
            raise KeyError(f"workflow process {name!r} is unknown or ambiguous")
        return matches[0]

    def insert_before(self, anchor: str, process: Any) -> Any:
        self.insert(self.index(self.process(anchor)), process)
        return process

    def insert_after(self, anchor: str, process: Any) -> Any:
        self.insert(self.index(self.process(anchor)) + 1, process)
        return process


WorkflowFactory = Callable[[WorkflowTemplate], Sequence[Any] | None]


@dataclass(frozen=True, slots=True)
class WorkflowPreviewAction:
    """One process in a declarative workflow preview."""

    name: str
    operation: str
    phase: str
    kind: str
    implementation: str
    enabled: bool = True

    @property
    def qualified_name(self) -> str:
        return f"{self.phase}.{self.name}"

    def __str__(self) -> str:
        return self.name


class WorkflowPreview(Sequence[WorkflowPreviewAction]):
    """Read-only workflow description that never launches PBS or MPI."""

    def __init__(self, actions: Sequence[WorkflowPreviewAction]) -> None:
        self._actions = tuple(actions)

    def __getitem__(self, index: int | slice) -> Any:
        return self._actions[index]

    def __len__(self) -> int:
        return len(self._actions)

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "index": index,
                "name": action.name,
                "operation": action.operation,
                "kind": action.kind,
                "implementation": action.implementation,
            }
            for index, action in enumerate(self._actions)
        )

    def __repr__(self) -> str:
        return "WorkflowPreview([" + ", ".join(
            action.name for action in self._actions
        ) + "])"


class FreeCAM:
    """Declarative atmosphere configuration used by :class:`CaseConfig`.

    This object is intentionally lightweight: constructing it starts no MPI
    work.  Its workflow is compiled against the real live process handles only
    after the persistent CAM session has initialized.
    """

    def __init__(
        self,
        workflow: WorkflowFactory | Sequence[Any] | None = None,
    ) -> None:
        if isinstance(workflow, (str, bytes)) or (
            workflow is not None
            and not callable(workflow)
            and not isinstance(workflow, Sequence)
        ):
            raise TypeError("FreeCAM workflow must be a factory or sequence")
        self.workflow = workflow

    def preview(self) -> WorkflowPreview:
        """Compile the declared workflow without starting the model."""

        original = tuple(PICAMStepPlan.default())
        if self.workflow is None:
            resolved = original
            names: Mapping[int, str] = {}
        else:
            resolved, names, _ = _compile_case_workflow(self.workflow, original)
        preview: list[WorkflowPreviewAction] = []
        for index, item in enumerate(resolved):
            if isinstance(item, Physics):
                anchor = next(
                    later
                    for later in resolved[index + 1 :]
                    if not isinstance(later, Physics)
                )
                preview.append(
                    WorkflowPreviewAction(
                        name=names[index],
                        operation=names[index],
                        phase=str(anchor.phase),
                        kind="python_process",
                        implementation="python",
                    )
                )
            else:
                preview.append(
                    WorkflowPreviewAction(
                        name=str(item.name),
                        operation=str(item.operation),
                        phase=str(item.phase),
                        kind=str(item.kind),
                        implementation=str(item.implementation),
                        enabled=bool(item.enabled),
                    )
                )
        return WorkflowPreview(preview)


@dataclass(frozen=True, slots=True)
class CaseConfig:
    """User-defined PI-CAM case and its declarative atmosphere workflow."""

    name: str
    description: str
    forcing: str
    make_atm: Callable[[], FreeCAM] = FreeCAM
    base: str = "PI-atm"
    config: str | Path | None = None

    def build_atmosphere(self) -> FreeCAM:
        atmosphere = self.make_atm()
        if not isinstance(atmosphere, FreeCAM):
            raise TypeError("CaseConfig.make_atm must return freecam.FreeCAM")
        return atmosphere

    @property
    def workflow(self) -> WorkflowPreview:
        """Preview this case's complete CAM order without launching MPI."""

        return self.build_atmosphere().preview()

    def preview(self) -> WorkflowPreview:
        return self.workflow

    @property
    def key(self) -> str:
        return self.name

    def __str__(self) -> str:
        return f"{self.name}: {self.description} [{self.forcing}]"


class CaseRegistry(Mapping[str, CaseConfig]):
    """Small public registry for reusable Notebook case declarations."""

    def __init__(self, cases: Sequence[CaseConfig] = ()) -> None:
        self._cases: dict[str, CaseConfig] = {}
        for case in cases:
            self.register(case)

    def __getitem__(self, name: str) -> CaseConfig:
        return self._cases[str(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cases)

    def __len__(self) -> int:
        return len(self._cases)

    def register(
        self,
        case: CaseConfig,
        *,
        replace: bool = False,
    ) -> CaseConfig:
        if not isinstance(case, CaseConfig):
            raise TypeError("CASES.register expects a freecam.CaseConfig")
        if case.name in self._cases and not replace:
            raise KeyError(f"case {case.name!r} is already registered")
        self._cases[case.name] = case
        return case

    def unregister(self, name: str) -> CaseConfig:
        return self._cases.pop(str(name))

    def __repr__(self) -> str:
        return "CaseRegistry(" + ", ".join(self._cases) + ")"


CASES = CaseRegistry(
    (
        CaseConfig(
            name="PI-atm",
            description="iCESM1.3.1 preindustrial CAM atmosphere",
            forcing="1850 fixed preindustrial with replayed coupler boundaries",
        ),
    )
)


def _workflow_item_matches(item: Any, token: str) -> bool:
    name = str(token)
    return name in {
        str(getattr(item, "name", "")),
        str(getattr(item, "operation", "")),
        str(getattr(item, "qualified_name", "")),
    }


def _workflow_process_name(process: Physics) -> str:
    raw = process.name or type(process).__name__
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(raw)
    )
    normalized = "_".join(part for part in normalized.split("_") if part)
    if not normalized:
        raise ValueError("Physics process name cannot be empty")
    if not normalized[0].isalpha():
        normalized = "physics_" + normalized
    return normalized


def _materialize_workflow(
    declaration: WorkflowFactory | Sequence[Any],
    default: WorkflowTemplate,
) -> WorkflowTemplate:
    if not callable(declaration):
        return WorkflowTemplate(declaration)
    candidate = default.copy()
    signature = inspect.signature(declaration)
    try:
        signature.bind(candidate)
    except TypeError as one_argument_error:
        try:
            signature.bind()
        except TypeError:
            raise TypeError(
                "a real freeCAM workflow factory must accept one default "
                "WorkflowTemplate argument (or no arguments); the FreeCESM "
                "toy (dynamics, history) signature omits required CAM control "
                "actions"
            ) from one_argument_error
        result = declaration()
    else:
        result = declaration(candidate)
    return candidate if result is None else WorkflowTemplate(result)


def _resolve_declared_workflow_item(
    item: Any,
    original: Sequence[Any],
) -> Any:
    if isinstance(item, Physics):
        return item
    if isinstance(item, str):
        matches = tuple(
            process for process in original if _workflow_item_matches(process, item)
        )
    else:
        qualified = getattr(item, "qualified_name", None)
        matches = tuple(
            process
            for process in original
            if process is item
            or (
                qualified is not None
                and process.qualified_name == str(qualified)
            )
        )
    if len(matches) != 1:
        raise ValueError(
            f"declared workflow item {item!r} is unknown or ambiguous"
        )
    return matches[0]


def _compile_case_workflow(
    declaration: WorkflowFactory | Sequence[Any],
    original: Sequence[Any],
) -> tuple[tuple[Any, ...], Mapping[int, str], tuple[Any, ...]]:
    requested = _materialize_workflow(
        declaration,
        WorkflowTemplate(original),
    )
    resolved = tuple(
        _resolve_declared_workflow_item(item, original) for item in requested
    )
    if not resolved:
        raise ValueError("case workflow cannot be empty")
    original_by_key = {item.qualified_name: item for item in original}
    requested_original_keys = tuple(
        item.qualified_name for item in resolved if not isinstance(item, Physics)
    )
    duplicates = tuple(
        key
        for key, count in Counter(requested_original_keys).items()
        if count != 1
    )
    if duplicates:
        raise ValueError(
            "an existing CAM process may appear only once in a case workflow: "
            + ", ".join(duplicates)
        )
    required_operations = {"boundary_import", "advance_timestep", "boundary_export"}
    requested_operations = {
        item.operation for item in resolved if not isinstance(item, Physics)
    }
    missing_required = sorted(required_operations - requested_operations)
    if missing_required:
        raise ValueError(
            "case workflow cannot remove required CAM control actions: "
            + ", ".join(missing_required)
        )
    if isinstance(resolved[0], Physics) or (
        resolved[0].operation != "boundary_import"
    ):
        raise ValueError("case workflow must start with boundary_import")
    if isinstance(resolved[-1], Physics) or (
        resolved[-1].operation != "boundary_export"
    ):
        raise ValueError("case workflow must end with boundary_export")

    custom = tuple(item for item in resolved if isinstance(item, Physics))
    if any(not item.enabled for item in custom):
        raise ValueError(
            "a Physics object listed in a case workflow must be enabled; omit "
            "it from the declaration instead"
        )
    base_counts = Counter(_workflow_process_name(item) for item in custom)
    base_seen: defaultdict[str, int] = defaultdict(int)
    occupied_names = {str(item.name).lower() for item in original}
    runtime_names: dict[int, str] = {}
    for index, item in enumerate(resolved):
        if not isinstance(item, Physics):
            continue
        base = _workflow_process_name(item)
        base_seen[base] += 1
        candidate = base if base_counts[base] == 1 else f"{base}_{base_seen[base]}"
        suffix = 2
        while candidate.lower() in occupied_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        occupied_names.add(candidate.lower())
        runtime_names[index] = candidate
    requested_keys = set(requested_original_keys)
    omitted = tuple(
        item for key, item in original_by_key.items() if key not in requested_keys
    )
    return resolved, runtime_names, omitted


def _apply_case_workflow(
    session: PICAMNotebookSession,
    declaration: WorkflowFactory | Sequence[Any],
) -> tuple[Any, ...]:
    """Install a case workflow atomically after the live model initializes."""

    original = tuple(session.workflow[:])
    resolved, runtime_names, omitted = _compile_case_workflow(
        declaration, original
    )

    installed: list[Any] = []
    final: list[Any] = []
    try:
        for index, item in enumerate(resolved):
            if not isinstance(item, Physics):
                final.append(item)
                continue
            anchor = next(
                (
                    later
                    for later in resolved[index + 1 :]
                    if not isinstance(later, Physics)
                ),
                None,
            )
            if anchor is None:
                raise ValueError(
                    "custom Physics cannot be placed after boundary_export"
                )
            handle = session.physics.install_python(
                item.tendency,
                name=runtime_names[index],
                before=anchor.qualified_name,
                reads=item.reads,
                writes=item.writes,
                enabled=True,
                transactional=item.transactional,
            )
            installed.append(handle)
            final.append(handle)
        for item in omitted:
            item.disable()
        session.workflow.replace(final)
    except BaseException:
        for item in reversed(installed):
            try:
                item.remove()
            except BaseException:
                pass
        for item in omitted:
            try:
                item.enable()
            except BaseException:
                pass
        try:
            session.workflow.replace(original)
        except BaseException:
            pass
        raise
    return tuple(installed)


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
        self.history = PICAMOutputView(driver, "history")
        self.restart = PICAMOutputView(driver, "restart")

    @property
    def state(self) -> Any:
        return self._driver._live_session().state

    @property
    def workflow(self) -> Any:
        return self._driver._live_session().workflow

    @workflow.setter
    def workflow(self, processes: Sequence[Any]) -> None:
        """Replace CAM's enabled process order with a Python sequence."""

        self._driver._live_session().workflow.replace(processes)

    @property
    def fields(self) -> Any:
        return self._driver._live_session().fields

    @property
    def physics(self) -> Any:
        return self._driver._live_session().physics

    @property
    def kernels(self) -> Any:
        return self._driver._live_session().kernels

    @property
    def status(self) -> Mapping[str, Any]:
        return self._driver.status

    @property
    def configured_processes(self) -> tuple[Any, ...]:
        """Python processes installed by the declarative case workflow."""

        self._driver._live_session()
        return self._driver._configured_processes

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
        case: str | CaseConfig = "PI-atm",
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
        history_every: int | None = 1,
        restart_every: int | None | str = "end",
        verify_boundary_exports: bool = False,
        python_executable: str | Path | None = None,
        session_factory: Any = PICAMNotebookSession,
    ) -> None:
        if int(nsteps) < 1:
            raise ValueError("nsteps must be positive")
        self.repo = Path(repo or Path(__file__).resolve().parents[3]).resolve()
        declared_case: CaseConfig | None
        if isinstance(case, CaseConfig):
            declared_case = case
        elif isinstance(case, str):
            declared_case = CASES.get(case)
        else:
            raise TypeError("case must be a case name or freecam.CaseConfig")
        if declared_case is not None:
            case_key = declared_case.base
            atmosphere = declared_case.build_atmosphere()
            if config is None and declared_case.config is not None:
                config = declared_case.config
        else:
            case_key = str(case)
            atmosphere = FreeCAM()
        if config is None:
            try:
                config = self.repo / self._CASE_CONFIGS[case_key]
            except KeyError as exc:
                raise ValueError(
                    f"unsupported base case {case_key!r}; available cases: "
                    + ", ".join(self._CASE_CONFIGS)
                ) from exc
        self.config_path = Path(config).expanduser().resolve()
        self.config = PICAMConfig.from_yaml(self.config_path)
        if int(nsteps) > self.config.stop_n:
            raise ValueError(
                f"nsteps={nsteps} exceeds the {self.config.stop_n}-step replay boundary"
            )
        self.case = declared_case or PICAMCaseInfo(case_key, self.config)
        self._atmosphere = atmosphere
        self._configured_processes: tuple[Any, ...] = ()
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
        if history_every is not None and (
            isinstance(history_every, bool) or int(history_every) < 1
        ):
            raise ValueError("history_every must be a positive integer or None")
        if restart_every == "end":
            self.restart_every: int | None | str = "end"
        elif restart_every is None:
            self.restart_every = None
        elif isinstance(restart_every, bool) or int(restart_every) < 1:
            raise ValueError(
                "restart_every must be a positive integer, 'end', or None"
            )
        else:
            self.restart_every = int(restart_every)
        self.history_every = (
            None if history_every is None else int(history_every)
        )
        self.verify_boundary_exports = bool(verify_boundary_exports)
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

    def preview(self) -> WorkflowPreview:
        """Return the configured workflow without submitting PBS or MPI."""

        return self._atmosphere.preview()

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

    def run(
        self,
        steps: int | None = None,
        *,
        verbose: bool = False,
    ) -> tuple[Mapping[str, Any], ...]:
        """FreeCESM-style alias for :meth:`execute`."""

        return self.execute(steps=steps, verbose=verbose)

    @property
    def trace(self) -> tuple[Mapping[str, Any], ...]:
        """Return the live action trace accumulated by this model."""

        if self._session is None:
            return ()
        return self._session.trace(since=0)

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
                verify_boundary_exports=self.verify_boundary_exports,
            )
            try:
                session.start()
                configure_output = getattr(session, "configure_output", None)
                if callable(configure_output):
                    configure_output(
                        history_every=self.history_every,
                        restart_every=self.restart_every,
                    )
                if self._atmosphere.workflow is not None:
                    self._configured_processes = _apply_case_workflow(
                        session,
                        self._atmosphere.workflow,
                    )
            except BaseException:
                try:
                    session.close()
                finally:
                    self._configured_processes = ()
                raise
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


__all__ = [
    "CASES",
    "CaseConfig",
    "CaseRegistry",
    "Driver",
    "FreeCAM",
    "Physics",
    "PICAMCaseInfo",
    "Variable",
    "WorkflowFactory",
    "WorkflowPreview",
    "WorkflowPreviewAction",
    "WorkflowTemplate",
]
