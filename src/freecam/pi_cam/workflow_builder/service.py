"""The local service behind the page: it holds the Driver and the draft.

The browser edits; the service stores the draft so a refresh comes back to
it, checks a document at the local level, saves what the browser generated,
and runs the model: the first Run initializes the Driver and applies the
document, later Runs apply only the difference and continue from the
current step.  Every Run is bound to the document's hash, its starting step
and its target.  Nothing here runs user code or loads a model file before
an explicit Run.

Security: the service listens on loopback, expects the session token on
every API request, and refuses cross-origin requests.  Reach a remote one
through an SSH tunnel.
"""

from __future__ import annotations

import secrets
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .bridge import AppliedState, RestartRequired, apply_document, model_calls
from .catalog import load_catalog
from .codegen import write_artifacts
from .document import WorkflowDocument, WorkflowEditError, WorkflowEditSession
from .templates import python_process_template
from .validate import validate_document

VERSION = "0.1"


class ServiceRefused(RuntimeError):
    """The request cannot be honoured in the model's current state (HTTP 409)."""


@dataclass(slots=True)
class LogEvent:
    sequence: int
    time: str
    level: str
    message: str

    def to_payload(self) -> dict[str, Any]:
        return {"sequence": self.sequence, "time": self.time, "level": self.level, "message": self.message}


@dataclass(slots=True)
class RunStatus:
    state: str = "idle"
    step: int | None = None
    target_step: int | None = None
    job_id: str | None = None
    run_dir: str | None = None
    workflow_hash: str | None = None
    applied_hash: str | None = None
    message: str | None = None
    model_calls: dict[str, int] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "state": self.state, "step": self.step, "target_step": self.target_step, "job_id": self.job_id,
            "run_dir": self.run_dir, "workflow_hash": self.workflow_hash, "applied_hash": self.applied_hash,
            "message": self.message, "model_calls": dict(self.model_calls), "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkflowService:
    """Everything the HTTP layer delegates to; testable without a server."""

    def __init__(self, driver: Any, *, root: Path | None = None, token: str | None = None,
                 generated_dir: Path | None = None) -> None:
        self.driver = driver
        self.token = token or secrets.token_urlsafe(24)
        document, entries, snapshot = load_catalog(root=root)
        case = getattr(driver, "case", None)
        case_name = getattr(case, "key", None) or getattr(case, "name", None) or getattr(driver, "case_name", None)
        if isinstance(case_name, str) and case_name in snapshot["cases"]:
            document = WorkflowDocument.from_payload({**document.to_payload(), "case": case_name})
        nsteps = getattr(driver, "nsteps", None)
        if isinstance(nsteps, int) and nsteps >= 1:
            document = WorkflowDocument.from_payload({**document.to_payload(), "nsteps": nsteps})
        self.snapshot = snapshot
        self.session = WorkflowEditSession(document, entries, python_template=python_process_template)
        self.default = self.session.default_document
        self.entries = entries
        self._draft: WorkflowDocument | None = None
        self._applied: AppliedState | None = None
        self._run = RunStatus()
        self._handle: Any = None
        self._events: deque[LogEvent] = deque(maxlen=2000)
        self._sequence = 0
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._generated_dir = generated_dir
        self.log("info", "workflow builder ready")

    # -- log ------------------------------------------------------------------

    def log(self, level: str, message: str) -> None:
        with self._lock:
            self._sequence += 1
            self._events.append(LogEvent(self._sequence, _now(), level, message))

    def events(self, since: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [event.to_payload() for event in self._events if event.sequence >= since]

    # -- state ----------------------------------------------------------------

    @property
    def driver_initialized(self) -> bool:
        return getattr(self.driver, "_session", None) is not None

    def resources(self) -> dict[str, Any]:
        config = getattr(self.driver, "config", None)
        ranks = int(getattr(config, "mpi_size", 0) or 0)
        return {
            "ranks": ranks,
            "nodes": max(1, ranks // 128) if ranks else 0,
            "queue": getattr(self.driver, "queue", None),
            "walltime": getattr(self.driver, "walltime", None),
            "account_set": getattr(self.driver, "account", None) is not None,
        }

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "local",
                "snapshot": self.snapshot,
                "draft": None if self._draft is None else self._draft.to_payload(),
                "run": self._run_payload(),
                "case": self.default.case,
                "nsteps": self.default.nsteps,
                "resources": self.resources(),
                "driver_initialized": self.driver_initialized,
                "version": VERSION,
            }

    def _run_payload(self) -> dict[str, Any]:
        self._refresh_run()
        return self._run.to_payload()

    def run_payload(self) -> dict[str, Any]:
        with self._lock:
            return self._run_payload()

    # -- draft ----------------------------------------------------------------

    def save_draft(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        document = self._parse(payload)
        with self._lock:
            self._draft = document
        return {"workflow_hash": document.workflow_hash, "revision": document.revision}

    def _parse(self, payload: Mapping[str, Any]) -> WorkflowDocument:
        try:
            return WorkflowDocument.from_payload(payload)
        except (WorkflowEditError, KeyError, TypeError, ValueError) as error:
            raise WorkflowEditError(f"not a workflow document: {error}") from error

    # -- checks ---------------------------------------------------------------

    def validate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        document = self._parse(payload)
        report = validate_document(
            document, default=self.default, catalog=self.entries, level="local",
            catalog_version=self.snapshot["catalog_hash"], root=None,
        )
        return report.to_payload()

    # -- generation -----------------------------------------------------------

    def generate(self, payload: Mapping[str, Any], artifacts: Mapping[str, str]) -> dict[str, Any]:
        document = self._parse(payload)
        directory = self._generated_dir
        if directory is None:
            run_dir = getattr(self.driver, "run_dir", None)
            directory = Path(run_dir) / "generated" if run_dir else Path.cwd() / "freecam-generated"
        written = write_artifacts(Path(directory), document.workflow_hash, artifacts)
        self.log("info", f"generated workflow {document.workflow_hash[:12]} into {written.directory}")
        return written.to_payload()

    # -- running --------------------------------------------------------------

    def start_run(self, payload: Mapping[str, Any], steps: int, confirm_resources: bool) -> dict[str, Any]:
        document = self._parse(payload)
        if int(steps) < 1:
            raise WorkflowEditError("steps must be positive")
        report = validate_document(document, default=self.default, catalog=self.entries, level="local",
                                   catalog_version=self.snapshot["catalog_hash"])
        if not report.ok:
            raise ServiceRefused("the document has errors: " + "; ".join(i.message for i in report.errors[:3]))
        with self._lock:
            self._refresh_run()
            if self._run.state in {"initializing", "queued", "running", "stopping"}:
                raise ServiceRefused("a run is in progress; wait for it or stop it")
            if not self.driver_initialized and not confirm_resources:
                raise ServiceRefused("the first Run starts the model; confirm the resources to proceed")
            self._draft = document
            self._run = RunStatus(
                state="initializing" if not self.driver_initialized else "running",
                step=self._current_step(), target_step=None, workflow_hash=document.workflow_hash,
                applied_hash=None if self._applied is None else self._applied.document.workflow_hash,
                started_at=_now(), model_calls=model_calls(self._applied),
            )
            self._worker = threading.Thread(target=self._run_worker, args=(document, int(steps)),
                                            name="freecam-ui-run", daemon=True)
            self._worker.start()
            return self._run.to_payload()

    def _current_step(self) -> int | None:
        if not self.driver_initialized:
            return None
        try:
            return int(self.driver.status.get("step", 0))
        except Exception:
            return None

    def _run_worker(self, document: WorkflowDocument, steps: int) -> None:
        try:
            if not self.driver_initialized:
                self.log("info", "initializing the model (PBS + MPI)")
                self.driver.initialize()
                self.log("info", "model initialized")
            with self._lock:
                self._run.job_id = self._job_id()
                self._run.run_dir = None if getattr(self.driver, "run_dir", None) is None else str(self.driver.run_dir)
                self._run.state = "running"
            if self._applied is None or self._applied.document.workflow_hash != document.workflow_hash:
                self.log("info", f"applying workflow {document.workflow_hash[:12]}")
                try:
                    applied = apply_document(self.driver, document, self._applied, default=self.default)
                except RestartRequired as error:
                    self.log("error", f"restart required: {error}")
                    with self._lock:
                        self._run.state = "error"
                        self._run.message = f"restart required: {error}"
                        self._run.finished_at = _now()
                    return
                for line in applied.log:
                    self.log("info", line)
                with self._lock:
                    self._applied = applied
                    self._run.applied_hash = document.workflow_hash
            start = self._current_step() or 0
            with self._lock:
                self._run.step = start
                self._run.target_step = start + steps
            self.log("info", f"running {steps} step(s) from step {start}")
            handle = self.driver.run_async(steps, progress=self._progress)
            with self._lock:
                self._handle = handle
            result = handle.result()
            with self._lock:
                self._run.step = self._current_step()
                self._run.model_calls = model_calls(self._applied)
                self._run.state = "completed" if not handle.cancelled() else "idle"
                self._run.message = "stopped at a step boundary" if handle.cancelled() else None
                self._run.finished_at = _now()
                self._handle = None
            self.log("info", f"run finished: {result!r}"[:300])
        except BaseException as error:  # the page must learn of every failure
            self.log("error", "".join(traceback.format_exception_only(type(error), error)).strip())
            with self._lock:
                self._run.state = "error"
                self._run.message = str(error)[:500]
                self._run.finished_at = _now()
                self._handle = None

    def _progress(self, progress: Any) -> None:
        with self._lock:
            completed = getattr(progress, "completed_steps", None)
            model_step = getattr(progress, "model_step", None)
            if model_step is not None:
                self._run.step = int(model_step)
            elif completed is not None and self._run.target_step is not None:
                self._run.step = int(self._run.target_step) - int(getattr(progress, "requested_steps", 0)) + int(completed)

    def _refresh_run(self) -> None:
        if self._run.state == "running" and self._handle is None and (self._worker is None or not self._worker.is_alive()):
            self._run.state = "error"
            self._run.message = self._run.message or "the run thread ended without a result"

    def _job_id(self) -> str | None:
        session = getattr(self.driver, "_session", None)
        return getattr(session, "job_id", None) if session is not None else None

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self._run.state != "running" or self._handle is None:
                raise ServiceRefused("nothing is running")
            self._run.state = "stopping"
            self._handle.cancel()
        self.log("info", "stop requested; the run ends at the next complete step")
        return self.run_payload()

    def close_model(self) -> dict[str, Any]:
        with self._lock:
            if self._run.state in {"running", "initializing", "queued", "stopping"}:
                raise ServiceRefused("stop the run before closing the model")
            self.driver.close()
            self._applied = None
            self._run = RunStatus(state="closed", message="the model is closed; the next Run starts a new one")
        self.log("info", "model closed")
        return self.run_payload()

    def shutdown(self) -> None:
        try:
            if self.driver_initialized:
                self.driver.close()
        finally:
            self.log("info", "service shutting down")


def create_app(service: WorkflowService, *, static_dir: Path | None = None) -> Any:
    """The FastAPI application for ``service``; FastAPI is imported only here."""

    from .http import build_app

    return build_app(service, static_dir=static_dir)


__all__ = ["LogEvent", "RunStatus", "ServiceRefused", "VERSION", "WorkflowService", "create_app"]
