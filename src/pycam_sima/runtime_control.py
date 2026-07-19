from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np

from .clock import ModelClock
from .state_pool import StatePool


@dataclass(slots=True)
class RuntimeOptions:
    """Mutable controls that are safe to change between complete model steps."""

    timestep_seconds: int
    physics_before: bool = True
    physics_after: bool = True
    dynamics: bool = True

    def validate(self) -> None:
        if isinstance(self.timestep_seconds, bool) or not isinstance(
            self.timestep_seconds, int
        ):
            raise TypeError("timestep_seconds must be an integer")
        if self.timestep_seconds <= 0:
            raise ValueError("timestep_seconds must be positive")
        for name in ("physics_before", "physics_after", "dynamics"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")


@dataclass(frozen=True, slots=True)
class StepPhase:
    """One visible operation in a Kessler model step."""

    name: str
    driver_method: str
    required: bool = False
    controlled_by: str | None = None
    enabled: bool = True


DEFAULT_STEP_PHASES = (
    StepPhase(
        "kessler_after_coupler",
        "_run_after",
        controlled_by="physics_after",
    ),
    StepPhase("physics_to_dynamics", "_physics_to_dynamics", required=True),
    StepPhase("se_dynamics", "_run_dynamics", controlled_by="dynamics"),
    StepPhase("physics_timestep_final", "_timestep_final", required=True),
    StepPhase("advance_clock", "_advance_clock", required=True),
    StepPhase("dynamics_to_physics", "_dynamics_to_physics", required=True),
    StepPhase("physics_timestep_initial", "_timestep_initial", required=True),
    StepPhase(
        "kessler_before_coupler",
        "_run_before",
        controlled_by="physics_before",
    ),
)


class StepPlan:
    """Editable, declarative ordering for one Kessler model step.

    Optional phases may be enabled or disabled directly.  Disabling a required
    lifecycle phase or changing CAM's validated ordering requires an explicit
    ``unsafe=True`` acknowledgement.
    """

    def __init__(self, phases: Iterable[StepPhase] = DEFAULT_STEP_PHASES) -> None:
        self._phases = list(phases)
        self._sequence_safe = True
        self._validate_unique_names()

    @classmethod
    def default(cls) -> StepPlan:
        return cls(DEFAULT_STEP_PHASES)

    @property
    def phases(self) -> tuple[StepPhase, ...]:
        return tuple(self._phases)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(phase.name for phase in self._phases)

    @property
    def sequence_safe(self) -> bool:
        return self._sequence_safe

    def copy(self) -> StepPlan:
        result = StepPlan(self._phases)
        result._sequence_safe = self._sequence_safe
        return result

    def reset(self) -> None:
        self._phases = list(DEFAULT_STEP_PHASES)
        self._sequence_safe = True

    def phase(self, name: str) -> StepPhase:
        try:
            return next(phase for phase in self._phases if phase.name == name)
        except StopIteration as exc:
            raise ValueError(
                f"unknown step phase {name!r}; choose from {self.names}"
            ) from exc

    def is_enabled(self, phase: StepPhase | str, options: RuntimeOptions) -> bool:
        selected = self.phase(phase) if isinstance(phase, str) else phase
        option_enabled = (
            True
            if selected.controlled_by is None
            else bool(getattr(options, selected.controlled_by))
        )
        return selected.enabled and option_enabled

    def enable(self, name: str) -> None:
        self._replace(name, enabled=True)

    def disable(self, name: str, *, unsafe: bool = False) -> None:
        phase = self.phase(name)
        if phase.required and not unsafe:
            raise ValueError(
                f"{name!r} is required by the validated model-step lifecycle; "
                "pass unsafe=True for an intentional experiment"
            )
        self._replace(name, enabled=False)
        if phase.required:
            self._sequence_safe = False

    def move(
        self,
        name: str,
        *,
        before: str | None = None,
        after: str | None = None,
        unsafe: bool = False,
    ) -> None:
        if (before is None) == (after is None):
            raise ValueError("provide exactly one of before= or after=")
        target = before if before is not None else after
        assert target is not None
        self.phase(name)
        self.phase(target)
        if name == target:
            raise ValueError("a phase cannot be moved relative to itself")
        if not unsafe:
            raise ValueError(
                "changing the validated CAM order requires unsafe=True"
            )

        moving = self.phase(name)
        self._phases.remove(moving)
        target_index = next(
            index for index, phase in enumerate(self._phases) if phase.name == target
        )
        insert_at = target_index if before is not None else target_index + 1
        self._phases.insert(insert_at, moving)
        self._sequence_safe = False

    def describe(self, options: RuntimeOptions | None = None) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index, phase in enumerate(self._phases, start=1):
            row: dict[str, Any] = {
                "order": index,
                "name": phase.name,
                "required": phase.required,
                "controlled_by": phase.controlled_by,
                "plan_enabled": phase.enabled,
            }
            if options is not None:
                row["enabled"] = self.is_enabled(phase, options)
            result.append(row)
        return result

    def _replace(self, name: str, **changes: Any) -> None:
        current = self.phase(name)
        index = self._phases.index(current)
        self._phases[index] = replace(current, **changes)

    def _validate_unique_names(self) -> None:
        if len(self.names) != len(set(self.names)):
            raise ValueError("step phase names must be unique")


class KesslerParameters:
    """Typed live view of frequently changed Kessler state parameters."""

    def __init__(
        self,
        pool: StatePool,
        options: RuntimeOptions,
        clock: ModelClock,
    ) -> None:
        self._pool = pool
        self._options = options
        self._clock = clock

    @property
    def timestep_seconds(self) -> int:
        return self._options.timestep_seconds

    @timestep_seconds.setter
    def timestep_seconds(self, value: int) -> None:
        previous = self._options.timestep_seconds
        self._options.timestep_seconds = value
        try:
            self._options.validate()
        except (TypeError, ValueError):
            self._options.timestep_seconds = previous
            raise
        self.sync_runtime_options()

    @property
    def surface_reference_pressure(self) -> float:
        return float(self._scalar("surface_reference_pressure")[0])

    @surface_reference_pressure.setter
    def surface_reference_pressure(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("surface_reference_pressure must be finite and positive")
        self._scalar("surface_reference_pressure")[0] = value

    @property
    def dycore_energy_adjustment(self) -> bool:
        return bool(
            self._scalar("flag_for_dycore_energy_consistency_adjustment")[0]
        )

    @dycore_energy_adjustment.setter
    def dycore_energy_adjustment(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("dycore_energy_adjustment must be bool")
        self._scalar("flag_for_dycore_energy_consistency_adjustment")[0] = int(value)

    @property
    def is_first_timestep(self) -> bool:
        return bool(self._scalar("is_first_timestep")[0])

    @is_first_timestep.setter
    def is_first_timestep(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("is_first_timestep must be bool")
        self._scalar("is_first_timestep")[0] = int(value)

    @property
    def constituent_minimum_values(self) -> np.ndarray[Any, Any]:
        return self._pool.require("ccpp_constituent_minimum_values")

    @constituent_minimum_values.setter
    def constituent_minimum_values(self, value: Any) -> None:
        self.constituent_minimum_values[...] = value

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {"timestep_seconds": self.timestep_seconds}
        if len(self._pool):
            result.update(
                surface_reference_pressure=self.surface_reference_pressure,
                dycore_energy_adjustment=self.dycore_energy_adjustment,
                is_first_timestep=self.is_first_timestep,
                constituent_minimum_values=self.constituent_minimum_values.tolist(),
            )
        return result

    def sync_runtime_options(self) -> None:
        """Apply mutable options to the clock and allocated native argument."""

        self._options.validate()
        self._clock.dt_seconds = self._options.timestep_seconds
        if len(self._pool):
            self._scalar("timestep_for_physics")[0] = self._options.timestep_seconds

    def _scalar(self, name: str) -> np.ndarray[Any, Any]:
        if not len(self._pool):
            raise RuntimeError(
                "allocate_minimal_state() or initialize() before accessing "
                f"parameter {name!r}"
            )
        return self._pool.require(name)
