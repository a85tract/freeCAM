from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping


@dataclass(slots=True)
class FullCAMRuntimeOptions:
    """Complete-CAM settings that must be fixed before ``cam_init``."""

    timestep_seconds: int
    physics_profile: str
    mediator_present: bool = False

    @classmethod
    def from_config(cls, config: Any) -> FullCAMRuntimeOptions:
        return cls(
            timestep_seconds=int(config.dt_seconds),
            physics_profile=str(getattr(config, "physics_suite", "kessler")),
            mediator_present=bool(getattr(config, "mediator_present", False)),
        )

    @property
    def physics_enabled(self) -> bool:
        return self.physics_profile != "adiabatic"

    @property
    def dynamics_enabled(self) -> bool:
        # The complete supported configurations both use the real SE dycore.
        return True

    def validate(self, config: Any | None = None) -> None:
        if isinstance(self.timestep_seconds, bool) or not isinstance(
            self.timestep_seconds, int
        ):
            raise TypeError("timestep_seconds must be an integer")
        if self.timestep_seconds <= 0:
            raise ValueError("timestep_seconds must be positive")
        if self.physics_profile not in {"kessler", "adiabatic"}:
            raise ValueError("physics_profile must be 'kessler' or 'adiabatic'")
        if self.mediator_present:
            raise ValueError("the complete Python driver supports ATM-only operation")
        if config is None:
            return
        configured_profile = getattr(config, "physics_suite", self.physics_profile)
        if self.physics_profile != configured_profile:
            raise ValueError(
                f"physics_profile={self.physics_profile!r} does not match the selected "
                f"configuration ({configured_profile!r}); select the matching YAML "
                "before starting CAM"
            )
        configured_mediator = bool(getattr(config, "mediator_present", False))
        if self.mediator_present != configured_mediator:
            raise ValueError("runtime mediator setting does not match the configuration")

    def describe(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            physics_enabled=self.physics_enabled,
            dynamics_enabled=self.dynamics_enabled,
            mutable_until="cam_init",
        )
        return result

    def fingerprint(self) -> tuple[int, str, bool]:
        return (
            self.timestep_seconds,
            self.physics_profile,
            self.mediator_present,
        )


@dataclass(frozen=True, slots=True)
class FullCAMPhase:
    name: str
    category: str
    description: str
    required: bool = True
    enabled: bool = True


DEFAULT_FULL_CAM_STEP_PHASES = (
    FullCAMPhase(
        "cam_run2",
        "physics_and_mapping",
        "physics after coupler, then physics-to-dynamics mapping",
    ),
    FullCAMPhase("cam_run3", "dynamics", "SE dynamics"),
    FullCAMPhase("cam_run4", "post_dynamics", "post-dynamics CAM work"),
    FullCAMPhase(
        "cam_timestep_final",
        "lifecycle",
        "finish the current CAM timestep",
    ),
    FullCAMPhase("advance_timestep", "clock", "advance the native CAM clock"),
    FullCAMPhase(
        "cam_timestep_init",
        "lifecycle_and_mapping",
        "initialize the next timestep and map dynamics to physics",
    ),
    FullCAMPhase(
        "cam_run1",
        "physics",
        "physics before coupler for the prepared timestep",
    ),
)

FULL_CAM_PHASE_NAMES = tuple(phase.name for phase in DEFAULT_FULL_CAM_STEP_PHASES)
_FULL_CAM_PHASE_BY_NAME = {
    phase.name: phase for phase in DEFAULT_FULL_CAM_STEP_PHASES
}


class FullCAMStepPlan:
    """Editable top-level phase plan for one complete CAM advance cycle."""

    def __init__(
        self,
        phases: Iterable[FullCAMPhase] = DEFAULT_FULL_CAM_STEP_PHASES,
        *,
        sequence_safe: bool = True,
    ) -> None:
        self._phases = list(phases)
        self._sequence_safe = bool(sequence_safe)
        self._validate_phases()
        if self._phases != list(DEFAULT_FULL_CAM_STEP_PHASES):
            self._sequence_safe = False

    @classmethod
    def default(cls) -> FullCAMStepPlan:
        return cls(DEFAULT_FULL_CAM_STEP_PHASES)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> FullCAMStepPlan:
        raw_phases = payload.get("phases")
        if not isinstance(raw_phases, list):
            raise ValueError("step plan payload requires a phases list")
        phases: list[FullCAMPhase] = []
        for row in raw_phases:
            if not isinstance(row, Mapping):
                raise TypeError("each step plan phase must be a mapping")
            name = str(row.get("name"))
            try:
                template = _FULL_CAM_PHASE_BY_NAME[name]
            except KeyError as exc:
                raise ValueError(f"unknown complete-CAM phase {name!r}") from exc
            enabled = row.get("enabled")
            if not isinstance(enabled, bool):
                raise TypeError(f"enabled for {name!r} must be bool")
            phases.append(replace(template, enabled=enabled))
        claimed_safe = payload.get("sequence_safe", False)
        if not isinstance(claimed_safe, bool):
            raise TypeError("step plan sequence_safe must be bool")
        return cls(phases, sequence_safe=claimed_safe)

    @property
    def phases(self) -> tuple[FullCAMPhase, ...]:
        return tuple(self._phases)

    @property
    def active_phases(self) -> tuple[FullCAMPhase, ...]:
        return tuple(phase for phase in self._phases if phase.enabled)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(phase.name for phase in self._phases)

    @property
    def sequence_safe(self) -> bool:
        return self._sequence_safe

    def copy(self) -> FullCAMStepPlan:
        return FullCAMStepPlan(self._phases, sequence_safe=self._sequence_safe)

    def phase(self, name: str) -> FullCAMPhase:
        try:
            return next(phase for phase in self._phases if phase.name == name)
        except StopIteration as exc:
            raise ValueError(
                f"unknown complete-CAM phase {name!r}; choose from {self.names}"
            ) from exc

    def enable(self, name: str) -> None:
        self._replace(name, enabled=True)

    def disable(self, name: str, *, unsafe: bool = False) -> None:
        phase = self.phase(name)
        if phase.required and not unsafe:
            raise ValueError(
                f"{name!r} is required by the validated complete-CAM sequence; "
                "pass unsafe=True for an intentional experiment"
            )
        self._replace(name, enabled=False)
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
        moving = self.phase(name)
        self.phase(target)
        if name == target:
            raise ValueError("a phase cannot be moved relative to itself")
        if not unsafe:
            raise ValueError(
                "changing the validated complete-CAM order requires unsafe=True"
            )
        self._phases.remove(moving)
        target_index = next(
            index for index, phase in enumerate(self._phases) if phase.name == target
        )
        insert_at = target_index if before is not None else target_index + 1
        self._phases.insert(insert_at, moving)
        self._sequence_safe = False

    def reset(self) -> None:
        self._phases = list(DEFAULT_FULL_CAM_STEP_PHASES)
        self._sequence_safe = True

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "order": index,
                "name": phase.name,
                "category": phase.category,
                "description": phase.description,
                "required": phase.required,
                "enabled": phase.enabled,
            }
            for index, phase in enumerate(self._phases, start=1)
        ]

    def to_payload(self) -> dict[str, object]:
        return {
            "sequence_safe": self._sequence_safe,
            "phases": [
                {"name": phase.name, "enabled": phase.enabled}
                for phase in self._phases
            ],
        }

    def _replace(self, name: str, **changes: Any) -> None:
        current = self.phase(name)
        index = self._phases.index(current)
        self._phases[index] = replace(current, **changes)

    def _validate_phases(self) -> None:
        if len(self.names) != len(set(self.names)):
            raise ValueError("complete-CAM phase names must be unique")
        if set(self.names) != set(FULL_CAM_PHASE_NAMES):
            raise ValueError(
                "a complete-CAM step plan must contain every known phase exactly once"
            )


class RemoteCAMField:
    """Notebook-side handle for a live field stored on the MPI workers."""

    def __init__(self, session: Any, name: str) -> None:
        self._session = session
        self.name = name

    @property
    def info(self) -> Mapping[str, Any]:
        return self._session.field_info(self.name)

    def get(self, *, rank: int | str = 0) -> Any:
        return self._session.get_field(self.name, rank=rank)

    def stats(self, *, rank: int | str = 0) -> Any:
        return self._session.get_field_stats(self.name, rank=rank)

    def set(self, value: Any, *, rank: int | str = 0) -> None:
        self._session.set_field(self.name, value, rank=rank)


class FullCAMParameters:
    """Typed Notebook facade for key writable complete-CAM fields."""

    KEY_FIELDS = (
        "air_temperature",
        "eastward_wind",
        "northward_wind",
        "surface_air_pressure",
        "air_pressure_thickness",
        "ccpp_constituents",
        "tendency_of_air_temperature_due_to_model_physics",
        "tendency_of_eastward_wind_due_to_model_physics",
        "tendency_of_northward_wind_due_to_model_physics",
    )

    def __init__(self, session: Any) -> None:
        self._session = session

    def field(self, name: str) -> RemoteCAMField:
        if name not in self._session.field_names:
            raise KeyError(f"unknown complete-CAM field {name!r}")
        return RemoteCAMField(self._session, name)

    def describe(self) -> dict[str, object]:
        return {
            "runtime": self._session.options.describe(),
            "key_fields": {
                name: dict(self._session.field_info(name))
                for name in self.KEY_FIELDS
                if name in self._session.field_names
            },
            "all_fields": self._session.field_names,
        }

    @property
    def air_temperature(self) -> RemoteCAMField:
        return self.field("air_temperature")

    @property
    def eastward_wind(self) -> RemoteCAMField:
        return self.field("eastward_wind")

    @property
    def northward_wind(self) -> RemoteCAMField:
        return self.field("northward_wind")

    @property
    def surface_air_pressure(self) -> RemoteCAMField:
        return self.field("surface_air_pressure")

    @property
    def air_pressure_thickness(self) -> RemoteCAMField:
        return self.field("air_pressure_thickness")

    @property
    def ccpp_constituents(self) -> RemoteCAMField:
        return self.field("ccpp_constituents")

    @property
    def temperature_physics_tendency(self) -> RemoteCAMField:
        return self.field("tendency_of_air_temperature_due_to_model_physics")

    @property
    def eastward_wind_physics_tendency(self) -> RemoteCAMField:
        return self.field("tendency_of_eastward_wind_due_to_model_physics")

    @property
    def northward_wind_physics_tendency(self) -> RemoteCAMField:
        return self.field("tendency_of_northward_wind_due_to_model_physics")
