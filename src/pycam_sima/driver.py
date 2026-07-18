from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .clock import ModelClock
from .config import CaseConfig
from .dynamics import IdentityDynamics
from .mpi_runtime import SerialComm
from .native import RecordingBackend
from .observer import ObserverContext, ObserverRegistry
from .state_layout import allocate_fkessler_kernel_state
from .state_pool import StatePool
from .suites.kessler import KesslerSuite
from .task_graph import run_linear


class FKesslerDriver:
    def __init__(
        self,
        config: CaseConfig,
        comm: Any | None = None,
        *,
        backend: Any | None = None,
        dynamics: Any | None = None,
    ) -> None:
        self.config = config
        self.comm = comm if comm is not None else SerialComm()
        self.backend = backend if backend is not None else RecordingBackend()
        self.dynamics = dynamics if dynamics is not None else IdentityDynamics()
        self.pool = StatePool()
        self.clock = ModelClock(config.dt_seconds)
        self.observers = ObserverRegistry(mode=config.mode)
        self.suite = KesslerSuite(self.backend)
        self._initialized = False

    def observe(
        self, event: str, callback: Callable[[ObserverContext], None], *, access: str = "readwrite"
    ) -> None:
        self.observers.observe(event, callback, access=access)

    def allocate_minimal_state(self, ncol: int = 1) -> None:
        allocate_fkessler_kernel_state(
            self.pool,
            ncol=ncol,
            pver=self.config.pver,
            dt_seconds=self.config.dt_seconds,
        )

    def initialize(self) -> None:
        if self._initialized:
            raise RuntimeError("driver is already initialized")
        if not len(self.pool):
            self.allocate_minimal_state()
        self._emit("initialize_begin", "initialize")
        run_linear(
            "BootstrapFlow",
            (
                ("dynamics.initialize", lambda: self.dynamics.initialize(self.pool)),
                ("kessler.register", lambda: self.suite.register(self.pool)),
                ("kessler.initialize", lambda: self.suite.initialize(self.pool)),
            ),
        )
        run_linear(
            "DataInitializeFlow",
            (
                ("dynamics_to_physics", lambda: self.dynamics.dynamics_to_physics(self.pool)),
                ("physics_timestep_initial", self._timestep_initial),
                ("kessler_before_coupler", self._run_before),
            ),
        )
        self._initialized = True
        self._emit("initialize_end", "initialize")

    def run(self, steps: int | None = None) -> None:
        if not self._initialized:
            raise RuntimeError("initialize the driver before run")
        count = self.config.steps if steps is None else steps
        for _ in range(count):
            self._emit("step_begin", "step")
            run_linear(
                "ModelAdvanceFlow",
                (
                    ("kessler_after_coupler", self._run_after),
                    ("physics_to_dynamics", lambda: self.dynamics.physics_to_dynamics(self.pool)),
                    ("se_dynamics", self._run_dynamics),
                    ("physics_timestep_final", self._timestep_final),
                    ("advance_clock", self.clock.advance),
                    ("dynamics_to_physics", lambda: self.dynamics.dynamics_to_physics(self.pool)),
                    ("physics_timestep_initial", self._timestep_initial),
                    ("kessler_before_coupler", self._run_before),
                ),
            )
            self._emit("step_end", "step")

    def finalize(self) -> None:
        if not self._initialized:
            raise RuntimeError("driver is not initialized")
        self._emit("finalize_begin", "finalize")
        # CAM leaves a partially prepared next step; close that suite step first.
        self.suite.timestep_final(self.pool)
        self.suite.finalize(self.pool)
        self.dynamics.finalize(self.pool)
        self._initialized = False
        self._emit("finalize_end", "finalize")

    def _run_before(self) -> None:
        self.suite.run_before(self.pool, self._invoke_scheme)

    def _run_after(self) -> None:
        self.suite.run_after(self.pool, self._invoke_scheme)

    def _invoke_scheme(self, name: str, pool: StatePool) -> None:
        self._emit(f"before:{name}", name)
        pointers = {field: pool.pointer(field) for field in pool}
        self.backend.call(name, pool)
        pool.validate()
        for field, pointer in pointers.items():
            if pool.pointer(field) != pointer:
                raise RuntimeError(f"native call {name} replaced Python-owned field {field}")
        self._emit(f"after:{name}", name)

    def _timestep_initial(self) -> None:
        self._emit("timestep_initial_begin", "timestep_initial")
        self.suite.timestep_initial(self.pool)
        self._emit("timestep_initial_end", "timestep_initial")

    def _timestep_final(self) -> None:
        self._emit("timestep_final_begin", "timestep_final")
        self.suite.timestep_final(self.pool)
        self._emit("timestep_final_end", "timestep_final")

    def _run_dynamics(self) -> None:
        self._emit("dynamics_begin", "dynamics")
        self.dynamics.run(self.pool)
        self._emit("dynamics_end", "dynamics")

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
