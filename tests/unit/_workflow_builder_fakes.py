"""A Driver double for the Workflow Builder tests: records what is asked of it.

It offers what the bridge and the generated code call -- the workflow list
with its process handles, the state's ``create``, the parameters mapping,
the physics catalog's ``process(...).insert`` -- and what the service reads:
initialization, status, a background run with progress and cancel.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any


class FakeHandle:
    def __init__(self, workflow: "FakeWorkflow", name: str) -> None:
        self.workflow = workflow
        self.name = name
        self.phase = "cam_run1"
        self.properties: dict[str, Any] = _RecordingProperties(workflow, name)

    @property
    def qualified_name(self) -> str:
        return f"{self.phase}.{self.name}"

    def _row(self) -> dict[str, Any]:
        return next(row for row in self.workflow.rows if row["name"] == self.name)

    @property
    def enabled(self) -> bool:
        return bool(self._row()["enabled"])

    @property
    def operation(self) -> str:
        return str(self._row().get("operation", self.name))

    @property
    def kind(self) -> str:
        return str(self._row()["kind"])

    def enable(self) -> None:
        self.workflow.calls.append(("enable", self.name))
        self._row()["enabled"] = True

    def disable(self) -> None:
        self.workflow.calls.append(("disable", self.name))
        self._row()["enabled"] = False

    def remove(self) -> None:
        self.workflow.calls.append(("remove", self.name))
        self.workflow.rows = [row for row in self.workflow.rows if row["name"] != self.name]

    def reload(self, process: Any) -> "FakeHandle":
        self.workflow.calls.append(("reload", self.name, type(process).__name__))
        return self

    def move(self, *, before: str | None = None, after: str | None = None) -> None:
        self.workflow.calls.append(("move", self.name, before, after))


class _RecordingProperties(dict):
    def __init__(self, workflow: "FakeWorkflow", name: str) -> None:
        super().__init__()
        self._workflow = workflow
        self._name = name

    def __setitem__(self, key: str, value: Any) -> None:
        self._workflow.calls.append(("property", self._name, key, value))
        super().__setitem__(key, value)


class FakeWorkflow:
    """The visible scientific list of a PI-atm step, by process name."""

    def __init__(self, names: list[str], disabled: set[str] = frozenset()) -> None:
        self.rows: list[dict[str, Any]] = [
            {"name": name, "enabled": name not in disabled, "kind": "scheme", "operation": name} for name in names
        ]
        self.calls: list[tuple[Any, ...]] = []

    def describe(self, *, include_disabled: bool = False) -> list[dict[str, Any]]:
        return [dict(row) for row in self.rows if include_disabled or row["enabled"]]

    def _resolve_name(self, key: str) -> str:
        name = key.split(".", 1)[1] if "." in key else key
        if not any(row["name"] == name for row in self.rows):
            raise KeyError(f"workflow action {key!r} is unknown or ambiguous")
        return name

    def __getitem__(self, key: str) -> FakeHandle:
        return FakeHandle(self, self._resolve_name(key))

    def process(self, key: str) -> FakeHandle:
        return self[key]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len([row for row in self.rows if row["enabled"]])

    def install(self, process: Any, *, before: str | None = None, after: str | None = None) -> FakeHandle:
        name = process.name or type(process).__name__.lower()
        self.calls.append(("insert", name, before, after))
        row = {"name": name, "enabled": getattr(process, "enabled", True), "kind": "python_process", "operation": name,
               "process": process}
        if before is not None:
            self.rows.insert(self._index(before), row)
        elif after is not None:
            self.rows.insert(self._index(after) + 1, row)
        else:
            self.rows.append(row)
        return FakeHandle(self, name)

    def insert(self, process: Any, *, before: str | None = None, after: str | None = None) -> None:
        self.install(process, before=before, after=after)

    def _index(self, key: str) -> int:
        name = self._resolve_name(key)
        return next(i for i, row in enumerate(self.rows) if row["name"] == name)

    def replace(self, processes: Any) -> dict[str, Any]:
        names = [p if isinstance(p, str) else p.name for p in processes]
        self.calls.append(("replace", tuple(names)))
        listed = set(names)
        by_name = {row["name"]: row for row in self.rows}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise KeyError(f"workflow action {missing[0]!r} is unknown or ambiguous")
        for row in self.rows:
            row["enabled"] = row["name"] in listed
        # the listed processes run in the given order; anything else stops running and trails behind
        self.rows = [by_name[name] for name in names] + [row for row in self.rows if row["name"] not in listed]
        return {"plan": self.describe(include_disabled=True)}


class FakeState:
    def __init__(self, workflow: FakeWorkflow) -> None:
        self.workflow = workflow
        self.created: dict[str, dict[str, Any]] = {}

    def create(self, name: str, **kwargs: Any) -> None:
        self.workflow.calls.append(("create", name, tuple(sorted(kwargs.items()))))
        self.created[name] = kwargs


class FakeParameters(dict):
    def __init__(self, workflow: FakeWorkflow) -> None:
        super().__init__()
        self.workflow = workflow

    def __setitem__(self, key: str, value: Any) -> None:
        self.workflow.calls.append(("parameter", key, value))
        super().__setitem__(key, value)


class FakePhysics:
    def __init__(self, workflow: FakeWorkflow) -> None:
        self.workflow = workflow

    def process(self, name: str) -> Any:
        workflow = self.workflow

        class Reference:
            def insert(self, *, before: str | None = None, after: str | None = None, enabled: bool = True) -> None:
                workflow.calls.append(("catalog_insert", name, before, after))
                workflow.rows.append({"name": name, "enabled": enabled, "kind": "runtime_catalog_process", "operation": name})

        return Reference()


class FakeRunHandle:
    def __init__(self, driver: "FakeDriver", steps: int, progress: Any) -> None:
        self.driver = driver
        self.steps = steps
        self.progress = progress
        self._cancelled = threading.Event()
        self._done = threading.Event()

    def start(self) -> "FakeRunHandle":
        def run() -> None:
            for _ in range(self.steps):
                if self._cancelled.is_set():
                    break
                self.driver.step += 1
                if callable(self.progress):
                    self.progress(SimpleNamespace(requested_steps=self.steps, completed_steps=self.driver.step, model_step=self.driver.step))
            self._done.set()

        threading.Thread(target=run, daemon=True).start()
        return self

    def result(self, timeout: float | None = None) -> str:
        self._done.wait(timeout)
        return f"ran to step {self.driver.step}"

    def cancel(self) -> bool:
        self._cancelled.set()
        return True

    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def done(self) -> bool:
        return self._done.is_set()


def _default_order() -> list[str]:
    from freecam.pi_cam.plan import PICAMStepPlan
    from freecam.pi_cam.workflow_builder.document import SCIENTIFIC_KINDS

    return [action.name for action in PICAMStepPlan.default().actions
            if action.kind in SCIENTIFIC_KINDS and action.enabled]


DEFAULT_ORDER = _default_order()


class FakeDriver:
    """Enough of ``freecam.Driver`` for the builder: nothing starts until initialize()."""

    def __init__(self, case: str = "PI-atm", nsteps: int = 2, run_dir: Any = None) -> None:
        self.case = SimpleNamespace(key=case, name=case)
        self.nsteps = nsteps
        self.config = SimpleNamespace(mpi_size=512)
        self.queue = "develop"
        self.walltime = "02:00:00"
        self.account = "AN-ALLOCATION"
        self.run_dir = run_dir
        self.workflow = FakeWorkflow(list(DEFAULT_ORDER))
        self.cam = SimpleNamespace(
            workflow=self.workflow,
            state=FakeState(self.workflow),
            parameters=FakeParameters(self.workflow),
            physics=FakePhysics(self.workflow),
            history=SimpleNamespace(latest=lambda: "case.cam.h0.nc"),
        )
        self._session = None
        self.step = 0
        self.closed = False
        self.initialized = 0

    def initialize(self) -> "FakeDriver":
        self.initialized += 1
        self._session = SimpleNamespace(job_id="12345.fake")
        return self

    @property
    def status(self) -> dict[str, Any]:
        return {"step": self.step}

    def run_async(self, steps: int | None = None, *, progress: Any = None, verbose: bool = False) -> FakeRunHandle:
        return FakeRunHandle(self, int(steps or self.nsteps), progress).start()

    def run(self, steps: int | None = None, *, progress: Any = None, verbose: bool = False) -> str:
        return self.run_async(steps, progress=progress).result()

    def close(self) -> None:
        self.closed = True
        self._session = None

    def __enter__(self) -> "FakeDriver":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
