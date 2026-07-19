from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from .clock import ModelClock
from .config import CaseConfig
from .dynamics import IdentityDynamics
from .mpi_runtime import SerialComm
from .native import RecordingBackend
from .observer import ObserverContext, ObserverRegistry
from .runtime_control import KesslerParameters, RuntimeOptions, StepPhase, StepPlan
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
        options: RuntimeOptions | None = None,
        step_plan: StepPlan | None = None,
    ) -> None:
        self.config = config
        self.comm = comm if comm is not None else SerialComm()
        self.backend = backend if backend is not None else RecordingBackend()
        self.dynamics = dynamics if dynamics is not None else IdentityDynamics()
        self.pool = StatePool()
        self.options = (
            options
            if options is not None
            else RuntimeOptions(timestep_seconds=config.dt_seconds)
        )
        self.options.validate()
        self.step_plan = step_plan if step_plan is not None else StepPlan.default()
        self.clock = ModelClock(self.options.timestep_seconds)
        self.parameters = KesslerParameters(self.pool, self.options, self.clock)
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
            dt_seconds=self.options.timestep_seconds,
        )
        self.parameters.sync_runtime_options()

    def initialize(self) -> None:
        if self._initialized:
            raise RuntimeError("driver is already initialized")
        if not len(self.pool):
            self.allocate_minimal_state()
        self.parameters.sync_runtime_options()
        self._emit("initialize_begin", "initialize")
        run_linear(
            "BootstrapFlow",
            (
                ("dynamics.initialize", self._initialize_dynamics),
                ("kessler.register", self._register_suite),
                ("kessler.initialize", self._initialize_suite),
            ),
        )
        data_initialize = [
            ("dynamics_to_physics", self._dynamics_to_physics),
            ("physics_timestep_initial", self._timestep_initial),
        ]
        if self.options.physics_before:
            data_initialize.append(("kessler_before_coupler", self._run_before))
        run_linear(
            "DataInitializeFlow",
            data_initialize,
        )
        self._initialized = True
        self._emit("initialize_end", "initialize")

    def run(self, steps: int | None = None) -> None:
        if not self._initialized:
            raise RuntimeError("initialize the driver before run")
        count = self.config.steps if steps is None else steps
        for _ in range(count):
            self.step()

    def step(self) -> None:
        """Execute one editable ModelAdvance plan and stop at its boundary."""

        if not self._initialized:
            raise RuntimeError("initialize the driver before step")
        self.parameters.sync_runtime_options()
        self._emit("step_begin", "step")
        calls = [
            (phase.name, partial(self._execute_phase, phase))
            for phase in self.step_plan.phases
            if self.step_plan.is_enabled(phase, self.options)
        ]
        run_linear("ModelAdvanceFlow", calls)
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

    def _execute_phase(self, phase: StepPhase) -> None:
        self._emit(f"phase_begin:{phase.name}", phase.name)
        getattr(self, phase.driver_method)()
        self._emit(f"phase_end:{phase.name}", phase.name)

    def _initialize_dynamics(self) -> None:
        self.dynamics.initialize(self.pool)

    def _register_suite(self) -> None:
        self.suite.register(self.pool)

    def _initialize_suite(self) -> None:
        self.suite.initialize(self.pool)

    def _physics_to_dynamics(self) -> None:
        self.dynamics.physics_to_dynamics(self.pool)

    def _dynamics_to_physics(self) -> None:
        self.dynamics.dynamics_to_physics(self.pool)

    def _advance_clock(self) -> None:
        self.clock.advance()

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
