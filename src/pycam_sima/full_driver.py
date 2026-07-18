from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .clock import ModelClock
from .config import CaseConfig
from .full_native import FullNativeBackend
from .observer import ObserverContext, ObserverRegistry
from .state_pool import StatePool
from .task_graph import run_linear


class FullCAMDriver:
    """Python control layer for the complete FKESSLER + SE CAM-SIMA path."""

    def __init__(
        self,
        config: CaseConfig,
        comm: Any,
        *,
        library: str | Path,
        run_dir: str | Path,
    ) -> None:
        self.config = config
        self.comm = comm
        self.backend = FullNativeBackend(library)
        self.run_dir = Path(run_dir).resolve()
        self.pool = StatePool()
        self.clock = ModelClock(config.dt_seconds)
        self.observers = ObserverRegistry(mode=config.mode)
        self._initialized = False
        self._previous_cwd: Path | None = None

    def observe(
        self, event: str, callback: Callable[[ObserverContext], None], *, access: str = "readwrite"
    ) -> None:
        self.observers.observe(event, callback, access=access)

    def initialize(self) -> None:
        if self._initialized:
            raise RuntimeError("driver is already initialized")
        if not (self.run_dir / "atm_in").is_file():
            raise FileNotFoundError(f"full CAM run directory lacks atm_in: {self.run_dir}")
        self._previous_cwd = Path.cwd()
        os.chdir(self.run_dir)
        try:
            self._emit("initialize_begin", "cam_init")
            self.backend.initialize(self.comm)
            self._emit("after:cam_init", "cam_init")
            self.backend.timestep_init()
            self.backend.attach_state(self.pool)
            self._emit("after:dynamics_to_physics", "cam_timestep_init")
            self.backend.run1()
            self._emit("after:kessler_before_coupler", "cam_run1")
        except BaseException:
            os.chdir(self._previous_cwd)
            self._previous_cwd = None
            raise
        self._initialized = True
        self._emit("initialize_end", "DataInitialize")

    def run(self, steps: int | None = None) -> None:
        if not self._initialized:
            raise RuntimeError("initialize the driver before run")
        count = self.config.steps if steps is None else steps
        # NUOPC's first ModelAdvance call executes the nstep=0 cycle before it
        # starts returning one coupled state per requested step.  Reproduce
        # that initial send cycle without incrementing the user-visible clock.
        if self.backend.nstep == 0:
            self._emit("initial_send_cycle_begin", "ModelAdvance")
            self._run_advance_cycle(count_clock=False)
            self._emit("initial_send_cycle_end", "ModelAdvance")
        for _ in range(count):
            self._emit("step_begin", "ModelAdvance")
            self._run_advance_cycle(count_clock=True)
            self._emit("step_end", "ModelAdvance")

    def _run_advance_cycle(self, *, count_clock: bool) -> None:
        run_linear(
            "FullModelAdvanceFlow",
            (
                ("cam_run2", self._run2),
                ("cam_run3", self._run3),
                ("cam_timestep_final", self._timestep_final),
                ("advance_timestep", lambda: self._advance(count_clock=count_clock)),
                ("cam_timestep_init", self._timestep_init),
                ("cam_run1", self._run1),
            ),
        )

    def finalize(self) -> None:
        if not self._initialized:
            raise RuntimeError("driver is not initialized")
        self._emit("finalize_begin", "ModelFinalize")
        self.backend.finalize()
        self._initialized = False
        for name in list(self.pool):
            del self.pool[name]
        # cam_final deallocates the native CAM state.  Remove every zero-copy
        # view before observers are called again so Python cannot access a
        # dangling pointer.
        self._emit("finalize_end", "ModelFinalize")
        assert self._previous_cwd is not None
        os.chdir(self._previous_cwd)
        self._previous_cwd = None

    def _run2(self) -> None:
        self._emit("before:kessler_after_coupler", "cam_run2")
        self.backend.run2()
        self._emit("after:physics_to_dynamics", "cam_run2")

    def _run3(self) -> None:
        self._emit("before:se_dynamics", "cam_run3")
        self.backend.run3()
        self._emit("after:se_dynamics", "cam_run3")

    def _timestep_final(self) -> None:
        self.backend.timestep_final()
        self._emit("after:timestep_final", "cam_timestep_final")

    def _advance(self, *, count_clock: bool) -> None:
        self.backend.advance_timestep()
        if count_clock:
            self.clock.advance()

    def _timestep_init(self) -> None:
        self.backend.timestep_init()
        self.backend.attach_state(self.pool)
        self._emit("after:dynamics_to_physics", "cam_timestep_init")

    def _run1(self) -> None:
        self.backend.run1()
        self._emit("after:kessler_before_coupler", "cam_run1")

    def _emit(self, event: str, task_name: str) -> None:
        self.observers.emit(
            event,
            ObserverContext(
                step=self.clock.step,
                rank=int(self.comm.rank),
                size=int(self.comm.size),
                phase=event.split(":", 1)[0],
                task_name=task_name,
                state=self.pool,
                clock_seconds=self.clock.elapsed_seconds,
                comm=self.comm,
            ),
        )
