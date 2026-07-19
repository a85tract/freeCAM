from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .clock import ModelClock
from .config import CaseConfig
from .full_native import FullNativeBackend
from .observer import ObserverContext, ObserverRegistry
from .state_pool import StatePool
from .task_graph import run_linear


FULL_CAM_PHASES = (
    "cam_run2",
    "cam_run3",
    "cam_run4",
    "cam_timestep_final",
    "advance_timestep",
    "cam_timestep_init",
    "cam_run1",
)


@dataclass(frozen=True)
class PhaseStatus:
    last_phase: str | None
    next_phase: str | None
    sequence_safe: bool
    cycle_kind: str
    cycle_complete: bool
    step: int
    native_nstep: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FullCAMDriver:
    """Python control layer for a complete no-mediator CAM-SIMA + SE path."""

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
        self._next_phase_index = 0
        self._last_phase: str | None = None
        self._sequence_safe = True
        self._active_cycle_kind: str | None = None

    @property
    def phase_names(self) -> tuple[str, ...]:
        return FULL_CAM_PHASES

    @property
    def next_phase(self) -> str | None:
        if not self._sequence_safe:
            return None
        return FULL_CAM_PHASES[self._next_phase_index]

    @property
    def phase_status(self) -> PhaseStatus:
        if self._active_cycle_kind is not None:
            cycle_kind = self._active_cycle_kind
        elif not self._sequence_safe:
            cycle_kind = "unsafe"
        elif self.backend.nstep == 0:
            cycle_kind = "initial_send"
        else:
            cycle_kind = "requested_step"
        return PhaseStatus(
            last_phase=self._last_phase,
            next_phase=self.next_phase,
            sequence_safe=self._sequence_safe,
            cycle_kind=cycle_kind,
            cycle_complete=self.next_phase == FULL_CAM_PHASES[0],
            step=self.clock.step,
            native_nstep=self.backend.nstep,
        )

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
            self._emit("after:physics_before_coupler", "cam_run1")
        except BaseException:
            os.chdir(self._previous_cwd)
            self._previous_cwd = None
            raise
        self._initialized = True
        self._next_phase_index = 0
        self._last_phase = "cam_run1"
        self._sequence_safe = True
        self._active_cycle_kind = None
        self._emit("initialize_end", "DataInitialize")

    def run(self, steps: int | None = None) -> None:
        if not self._initialized:
            raise RuntimeError("initialize the driver before run")
        if not self._sequence_safe:
            raise RuntimeError("cannot use step() after an unsafe phase ordering")
        if self.next_phase != FULL_CAM_PHASES[0]:
            raise RuntimeError(
                f"cannot start a complete step while a phase cycle is in progress; "
                f"next phase is {self.next_phase}"
            )
        count = self.config.steps if steps is None else steps
        # NUOPC's first ModelAdvance call executes the nstep=0 cycle before it
        # starts returning one coupled state per requested step.  Reproduce
        # that initial send cycle without incrementing the user-visible clock.
        if self.backend.nstep == 0:
            self._run_advance_cycle(count_clock=False)
        for _ in range(count):
            self._run_advance_cycle(count_clock=True)

    def _run_advance_cycle(self, *, count_clock: bool) -> None:
        expected_kind = "requested_step" if count_clock else "initial_send"
        actual_kind = "requested_step" if self.backend.nstep > 0 else "initial_send"
        if expected_kind != actual_kind:
            raise RuntimeError(
                f"phase clock mismatch: requested {expected_kind}, native state requires {actual_kind}"
            )
        run_linear(
            "FullModelAdvanceFlow",
            tuple(
                (phase, lambda phase=phase: self.run_phase(phase))
                for phase in FULL_CAM_PHASES
            ),
        )

    def run_phase(
        self,
        phase: str | None = None,
        *,
        allow_unsafe_order: bool = False,
    ) -> PhaseStatus:
        if not self._initialized:
            raise RuntimeError("initialize the driver before running a phase")
        selected = self.next_phase if phase is None else str(phase)
        if selected not in FULL_CAM_PHASES:
            raise ValueError(
                f"unknown CAM phase {selected!r}; choose one of {FULL_CAM_PHASES}"
            )

        expected = self.next_phase
        follows_sequence = self._sequence_safe and selected == expected
        if not follows_sequence and not allow_unsafe_order:
            if self._sequence_safe:
                raise RuntimeError(
                    f"invalid phase order: expected {expected}, got {selected}; "
                    "pass allow_unsafe_order=True only for an explicit experiment"
                )
            raise RuntimeError(
                "the driver is already in unsafe-order mode; every further phase "
                "must explicitly pass allow_unsafe_order=True"
            )

        if follows_sequence and self._next_phase_index == 0:
            self._active_cycle_kind = (
                "initial_send" if self.backend.nstep == 0 else "requested_step"
            )
            event = (
                "initial_send_cycle_begin"
                if self._active_cycle_kind == "initial_send"
                else "step_begin"
            )
            self._emit(event, "ModelAdvance")

        self._execute_phase(selected)
        self._last_phase = selected

        if follows_sequence:
            self._next_phase_index = (self._next_phase_index + 1) % len(FULL_CAM_PHASES)
            if self._next_phase_index == 0:
                event = (
                    "initial_send_cycle_end"
                    if self._active_cycle_kind == "initial_send"
                    else "step_end"
                )
                self._emit(event, "ModelAdvance")
                self._active_cycle_kind = None
        else:
            self._sequence_safe = False
            self._active_cycle_kind = "unsafe"

        return self.phase_status

    def run_sequence(
        self,
        phases: tuple[str, ...] | list[str],
        *,
        allow_unsafe_order: bool = False,
    ) -> tuple[PhaseStatus, ...]:
        return tuple(
            self.run_phase(phase, allow_unsafe_order=allow_unsafe_order)
            for phase in phases
        )

    def _execute_phase(self, phase: str) -> None:
        if phase == "cam_run2":
            self._run2()
        elif phase == "cam_run3":
            self._run3()
        elif phase == "cam_run4":
            self._run4()
        elif phase == "cam_timestep_final":
            self._timestep_final()
        elif phase == "advance_timestep":
            count_clock = (
                self._active_cycle_kind == "requested_step"
                if self._sequence_safe
                else self.backend.nstep > 0
            )
            self._advance(count_clock=count_clock)
        elif phase == "cam_timestep_init":
            self._timestep_init()
        elif phase == "cam_run1":
            self._run1()
        else:  # pragma: no cover - validated before dispatch
            raise AssertionError(phase)

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
        self._emit("before:physics_after_coupler", "cam_run2")
        self.backend.run2()
        self._emit("after:physics_to_dynamics", "cam_run2")

    def _run3(self) -> None:
        self._emit("before:se_dynamics", "cam_run3")
        self.backend.run3()
        self._emit("after:se_dynamics", "cam_run3")

    def _run4(self) -> None:
        self.backend.run4()
        self._emit("after:cam_run4", "cam_run4")

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
        self._emit("after:physics_before_coupler", "cam_run1")

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
