"""Python control plane for one live PI-CAM model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from pycam_sima.model.clock import ModelClock

from .boundary import CAMBoundaryProvider
from .config import PICAMConfig
from .errors import PICAMConfigurationError, PICAMStateError
from .native import CAMNumericalBackend
from .plan import PICAMAction, PICAMStepPlan
from .state import PICAMStatePool, PICAMStateSchema


class PICAMLifecycle(str, Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINALIZED = "finalized"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PICAMActionTrace:
    sequence: int
    coupling_step: int
    model_step: int
    phase: str
    name: str
    operation: str
    native_id: int | None


class _ActionReference:
    def __init__(self, driver: "PICAMDriver", action: PICAMAction) -> None:
        self.driver = driver
        self.action = action

    @property
    def enabled(self) -> bool:
        return self.driver.step_plan.select(
            self.action.name, phase=self.action.phase
        ).enabled

    def run(self, *, experimental: bool = True) -> PICAMActionTrace:
        if not experimental:
            raise PICAMConfigurationError(
                "running an isolated CAM action requires experimental=True"
            )
        return self.driver.run_action(
            self.action.name, phase=self.action.phase, experimental=True
        )

    def disable(self, *, experimental: bool = True) -> None:
        self.driver.step_plan.set_enabled(
            self.action.name,
            False,
            phase=self.action.phase,
            experimental=experimental,
        )

    def enable(self, *, experimental: bool = True) -> None:
        self.driver.step_plan.set_enabled(
            self.action.name,
            True,
            phase=self.action.phase,
            experimental=experimental,
        )

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
        experimental: bool = True,
    ) -> None:
        self.driver.step_plan.move(
            self.action.name,
            phase=self.action.phase,
            before=before,
            after=after,
            experimental=experimental,
        )


class _PhysicsCollection:
    def __init__(self, driver: "PICAMDriver") -> None:
        self.driver = driver

    def __getattr__(self, name: str) -> _ActionReference:
        matches = [
            action
            for action in self.driver.step_plan.actions
            if action.kind == "scheme"
            and (action.name == name or action.operation == name)
        ]
        if len(matches) != 1:
            raise AttributeError(name)
        return _ActionReference(self.driver, matches[0])

    def scheme(self, name: str, *, phase: str | None = None) -> _ActionReference:
        action = self.driver.step_plan.select(name, phase=phase)
        if action.kind != "scheme":
            raise PICAMConfigurationError(f"{name!r} is not a physics scheme")
        return _ActionReference(self.driver, action)


class _PhaseReference:
    def __init__(self, driver: "PICAMDriver", name: str) -> None:
        self.driver = driver
        self.name = name

    def run(self, *, experimental: bool = True) -> tuple[PICAMActionTrace, ...]:
        return self.driver.run_phase(self.name, experimental=experimental)


class _PhaseCollection:
    def __init__(self, driver: "PICAMDriver") -> None:
        self.driver = driver

    def __getattr__(self, name: str) -> _PhaseReference:
        if name not in self.driver.step_plan.phases:
            raise AttributeError(name)
        return _PhaseReference(self.driver, name)

    def __getitem__(self, name: str) -> _PhaseReference:
        if name not in self.driver.step_plan.phases:
            raise KeyError(name)
        return _PhaseReference(self.driver, name)


class _KernelReference:
    def __init__(self, driver: "PICAMDriver", name: str) -> None:
        self.driver = driver
        self.name = name

    def run(self, *, experimental: bool = False) -> PICAMActionTrace:
        return self.driver.run_kernel(self.name, experimental=experimental)


class _KernelCollection:
    """Original leaf routines with generated raw-array adapters."""

    def __init__(self, driver: "PICAMDriver") -> None:
        self.driver = driver

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(getattr(self.driver.backend, "direct_kernels", ()))

    def __getattr__(self, name: str) -> _KernelReference:
        if name not in self.names:
            raise AttributeError(name)
        return _KernelReference(self.driver, name)

    def __getitem__(self, name: str) -> _KernelReference:
        if name not in self.names:
            raise KeyError(name)
        return _KernelReference(self.driver, name)


class PICAMDriver:
    """Own CAM orchestration while delegating leaf numerics to native devices."""

    def __init__(
        self,
        config: PICAMConfig,
        boundary: CAMBoundaryProvider,
        backend: CAMNumericalBackend,
        *,
        rank: int,
        size: int,
        fcomm: int = 0,
        step_plan: PICAMStepPlan | None = None,
        state_schema: PICAMStateSchema | None = None,
        history_callback: Callable[[PICAMAction, "PICAMDriver"], None] | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        if not 0 <= rank < size:
            raise PICAMConfigurationError("rank must be in the MPI communicator")
        self.config = config
        self.boundary = boundary
        self.backend = backend
        self.rank = int(rank)
        self.size = int(size)
        self.fcomm = int(fcomm)
        self.step_plan = step_plan or PICAMStepPlan.default()
        self.pool = (state_schema or PICAMStateSchema.core()).allocate(
            {
                "pver": config.pver,
                "pcols": config.pcols,
                "mpi_rank": rank,
                "mpi_size": size,
                "case_name_length": 256,
            }
        )
        year, month, day = config.start_parts
        self.clock = ModelClock(
            year=year,
            month=month,
            day=day,
            seconds=config.start_seconds,
            dt_seconds=config.timestep_seconds,
            calendar=config.calendar,
        )
        self.lifecycle = PICAMLifecycle.CREATED
        self.coupling_step = 0
        self._boundary_step = 0
        self._trace: list[PICAMActionTrace] = []
        self.history_callback = history_callback
        self.run_dir = None if run_dir is None else Path(run_dir).resolve()
        self._previous_directory: Path | None = None
        self._advance_public_clock = True
        self.physics = _PhysicsCollection(self)
        self.phases = _PhaseCollection(self)
        self.kernels = _KernelCollection(self)

    @property
    def trace(self) -> tuple[PICAMActionTrace, ...]:
        return tuple(self._trace)

    def _set_scalar(self, name: str, value: int | float) -> None:
        self.pool[name][()] = value

    def _sync_clock_fields(self) -> None:
        self._set_scalar("model_step", self.clock.nstep)
        self._set_scalar("current_date", self.clock.yyyymmdd)
        self._set_scalar("current_seconds_of_day", self.clock.seconds)

    def initialize(self) -> None:
        if self.lifecycle != PICAMLifecycle.CREATED:
            raise PICAMStateError(f"initialize from {self.lifecycle.value}")
        if self.size != self.config.mpi_size:
            raise PICAMConfigurationError(
                f"PI-CAM config requires {self.config.mpi_size} ranks, got {self.size}"
            )
        try:
            if self.run_dir is not None:
                if not (self.run_dir / "atm_in").is_file():
                    raise PICAMConfigurationError(
                        f"PI-CAM run directory lacks atm_in: {self.run_dir}"
                    )
                self._previous_directory = Path.cwd()
                os.chdir(self.run_dir)
            self._set_scalar("model_timestep", self.config.timestep_seconds)
            self._set_scalar("configured_stop_n", self.config.stop_n)
            encoded_case_name = self.config.case_name.encode("utf-8")
            case_name = self.pool["case_name_utf8"]
            if len(encoded_case_name) >= case_name.size:
                raise PICAMConfigurationError(
                    f"case_name must contain fewer than {case_name.size} UTF-8 bytes"
                )
            case_name[...] = 0
            case_name[: len(encoded_case_name)] = np.frombuffer(
                encoded_case_name, dtype=np.uint8
            )
            self._set_scalar("orbital_year", self.config.orbital_year)
            self._set_scalar("mpi_rank", self.rank)
            self._set_scalar("mpi_size", self.size)
            self._sync_clock_fields()
            self.boundary.initialize(
                rank=self.rank,
                size=self.size,
                config_fingerprint=self.config.fingerprint,
            )
            # The source initial-run lifecycle invokes atm_init_mct twice.  Its
            # second call imports the first surface state and executes CAM run1
            # before the first normal run2/run3/run4 timestep.  Keep that
            # orchestration in Python so native CAM never observes an
            # unprimed phys_state.
            self.boundary.import_fields(0, self.rank, self.pool)
            self.backend.initialize(self.pool, fcomm=self.fcomm)
            self._validate_initial_cam_export()
            self._prime_initial_cam_state()
        except Exception:
            self.lifecycle = PICAMLifecycle.FAILED
            raise
        self.lifecycle = PICAMLifecycle.INITIALIZED
        # iCESM CAM uses a split timestep: startup performs one complete
        # control pass to prepare run1 state for the first public coupling
        # step.  Native CAM time advances, while the Python completed-step
        # clock intentionally remains at the initial date.
        for _ in range(self.config.initialization_lookahead_steps):
            self._advance_public_clock = False
            try:
                self.step()
            finally:
                self._advance_public_clock = True

    def _prime_initial_cam_state(self) -> None:
        """Reproduce the initial-only ``atm_import -> cam_run1 -> atm_export``."""

        boundary_import = self.step_plan.select("boundary_import")
        boundary_export = self.step_plan.select("boundary_export")
        initial_priming = PICAMAction(
            "initial_priming",
            "initialization",
            "initial_priming",
            "control",
            200,
        )
        self.boundary.import_fields(1, self.rank, self.pool)
        # The source second atm_init_mct call performs import, cam_run1 and
        # export in one Fortran stack frame.  ``initial_priming`` preserves
        # that startup-only numerical boundary while the trace still exposes
        # the three source actions to Python.
        self._record(boundary_import)
        self.backend.execute(initial_priming, self.pool, fcomm=self.fcomm)
        self._record(initial_priming)
        self._record(boundary_export)
        self.boundary.export_fields(1, self.rank, self.pool)
        self._boundary_step = 2

    def _validate_initial_cam_export(self) -> None:
        """Reproduce the first atm_init_mct export immediately after cam_init."""

        boundary_export = self.step_plan.select("boundary_export")
        self.backend.execute(boundary_export, self.pool, fcomm=self.fcomm)
        self._record(boundary_export)
        self.boundary.export_fields(0, self.rank, self.pool)

    def _record(self, action: PICAMAction) -> PICAMActionTrace:
        trace = PICAMActionTrace(
            sequence=len(self._trace),
            coupling_step=self.coupling_step,
            model_step=self.clock.nstep,
            phase=action.phase,
            name=action.name,
            operation=action.operation,
            native_id=action.native_id,
        )
        self._trace.append(trace)
        return trace

    def _execute(self, action: PICAMAction) -> PICAMActionTrace:
        if action.operation == "boundary_import":
            self.boundary.import_fields(self._boundary_step, self.rank, self.pool)
            self.backend.execute(action, self.pool, fcomm=self.fcomm)
        elif action.operation == "boundary_export":
            self.backend.execute(action, self.pool, fcomm=self.fcomm)
            self.boundary.export_fields(self._boundary_step, self.rank, self.pool)
        elif action.operation == "advance_timestep":
            self.backend.execute(action, self.pool, fcomm=self.fcomm)
            if self._advance_public_clock:
                self.clock.advance()
                self._sync_clock_fields()
        elif action.kind == "io" and self.history_callback is not None:
            self.history_callback(action, self)
        else:
            self.backend.execute(action, self.pool, fcomm=self.fcomm)
        return self._record(action)

    def run_action(
        self, name: str, *, phase: str | None = None, experimental: bool = False
    ) -> PICAMActionTrace:
        if self.lifecycle not in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            raise PICAMStateError(f"run action from {self.lifecycle.value}")
        if not experimental:
            raise PICAMConfigurationError(
                "isolated action execution requires experimental=True"
            )
        action = self.step_plan.select(name, phase=phase)
        if action.kind in {"boundary", "clock", "io"}:
            raise PICAMConfigurationError(
                f"{action.operation!r} is controlled by a complete step"
            )
        return self._execute(action)

    def run_phase(
        self, phase: str, *, experimental: bool = False
    ) -> tuple[PICAMActionTrace, ...]:
        if not experimental:
            raise PICAMConfigurationError(
                "isolated phase execution requires experimental=True"
            )
        actions = self.step_plan.in_phase(phase)
        if any(action.kind in {"boundary", "clock", "io"} for action in actions):
            raise PICAMConfigurationError(
                f"phase {phase!r} contains step-controlled boundary, clock, or I/O"
            )
        return tuple(self._execute(action) for action in actions)

    def run_kernel(
        self, name: str, *, experimental: bool = False
    ) -> PICAMActionTrace:
        """Run one raw-array numerical routine without its CAM host stage."""

        if self.lifecycle not in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            raise PICAMStateError(f"run kernel from {self.lifecycle.value}")
        if not experimental:
            raise PICAMConfigurationError(
                "isolated raw CAM kernels require experimental=True"
            )
        execute = getattr(self.backend, "execute_kernel", None)
        if not callable(execute):
            raise PICAMConfigurationError("the selected backend has no direct kernels")
        execute(name, self.pool, fcomm=self.fcomm)
        return self._record(
            PICAMAction(
                name=name,
                phase="direct_kernel",
                operation=name,
                kind="kernel",
                native_id=None,
            )
        )

    def step(self) -> tuple[PICAMActionTrace, ...]:
        if self.lifecycle not in {PICAMLifecycle.INITIALIZED, PICAMLifecycle.RUNNING}:
            raise PICAMStateError(f"step from {self.lifecycle.value}")
        self.lifecycle = PICAMLifecycle.RUNNING
        first_trace = len(self._trace)
        actions = tuple(self.step_plan)
        imports = tuple(action for action in actions if action.operation == "boundary_import")
        exports = tuple(action for action in actions if action.operation == "boundary_export")
        body = tuple(action for action in actions if action.kind != "boundary")
        if len(imports) != 1 or len(exports) != 1:
            raise PICAMStateError("complete PI-CAM step requires one import and one export")
        try:
            source_step = getattr(self.backend, "execute_source_step", None)
            use_source_boundary = (
                callable(source_step)
                and self.step_plan.is_source_default
                and self.config.substeps_per_coupling == 1
            )
            if use_source_boundary:
                # Boundary data remains Python-owned.  For the unchanged
                # scientific plan, pass both arrays through one original CAM
                # numerical call so hidden HOMME state sees exactly the same
                # call/stack boundary as the source executable.
                self.boundary.import_fields(self._boundary_step, self.rank, self.pool)
                self._record(imports[0])
                source_step(
                    self.pool,
                    fcomm=self.fcomm,
                    apply_import=self.boundary.has_fresh_import(
                        self._boundary_step, self.rank
                    ),
                )
                for action in body:
                    if action.kind == "io" and self.history_callback is not None:
                        self.history_callback(action, self)
                    if action.operation == "advance_timestep" and self._advance_public_clock:
                        self.clock.advance()
                        self._sync_clock_fields()
                    self._record(action)
                self._record(exports[0])
                self.boundary.export_fields(self._boundary_step, self.rank, self.pool)
            else:
                self._execute(imports[0])
                for _ in range(self.config.substeps_per_coupling):
                    for action in body:
                        self._execute(action)
                self._execute(exports[0])
            self.coupling_step += 1
            self._boundary_step += 1
        except Exception:
            self.lifecycle = PICAMLifecycle.FAILED
            raise
        return tuple(self._trace[first_trace:])

    def advance(self, steps: int | None = None) -> None:
        count = self.config.stop_n if steps is None else int(steps)
        if count < 0:
            raise PICAMConfigurationError("steps cannot be negative")
        for _ in range(count):
            self.step()

    def save(self, directory: str | Path) -> Path:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        rank_file = destination / f"rank-{self.rank:04d}.npz"
        np.savez(rank_file, **self.pool.snapshot(restart_only=True))
        metadata = {
            "schema_version": 1,
            "config_fingerprint": self.config.fingerprint,
            "rank": self.rank,
            "size": self.size,
            "coupling_step": self.coupling_step,
            "boundary_step": self._boundary_step,
            "clock": {
                "year": self.clock.year,
                "month": self.clock.month,
                "day": self.clock.day,
                "seconds": self.clock.seconds,
                "nstep": self.clock.nstep,
            },
        }
        (destination / f"rank-{self.rank:04d}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        return rank_file

    def finalize(self) -> None:
        if self.lifecycle == PICAMLifecycle.FINALIZED:
            return
        if self.lifecycle not in {
            PICAMLifecycle.INITIALIZED,
            PICAMLifecycle.RUNNING,
            PICAMLifecycle.FAILED,
        }:
            raise PICAMStateError(f"finalize from {self.lifecycle.value}")
        try:
            self.backend.finalize(self.pool, fcomm=self.fcomm)
            self.boundary.finalize()
        finally:
            self.lifecycle = PICAMLifecycle.FINALIZED
            if self._previous_directory is not None:
                os.chdir(self._previous_directory)
                self._previous_directory = None

    def __enter__(self) -> "PICAMDriver":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.finalize()
