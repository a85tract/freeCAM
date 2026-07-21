"""Editable CCPP scheme-level plan for the fixed FKESSLER suite."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping


PHYSICS_BEFORE_COUPLER = "physics_before_coupler"
PHYSICS_AFTER_COUPLER = "physics_after_coupler"
SCHEME_GROUPS = (PHYSICS_BEFORE_COUPLER, PHYSICS_AFTER_COUPLER)


@dataclass(frozen=True, slots=True)
class PhysicsScheme:
    name: str
    # ``group`` is the current execution group and may be edited.
    group: str
    category: str
    description: str
    implementation: str
    required: bool = True
    enabled: bool = True
    # The source group is immutable identity from suite_kessler.xml.
    source_group: str = ""

    def __post_init__(self) -> None:
        if not self.source_group:
            object.__setattr__(self, "source_group", self.group)

    @property
    def key(self) -> str:
        """Stable identity, independent of the current execution group."""

        return f"{self.source_group}.{self.name}"


def _scheme(
    name: str,
    group: str,
    category: str,
    description: str,
    implementation: str = "python",
) -> PhysicsScheme:
    return PhysicsScheme(
        name=name,
        group=group,
        category=category,
        description=description,
        implementation=implementation,
        source_group=group,
    )


DEFAULT_KESSLER_SCHEMES = (
    _scheme("calc_exner", PHYSICS_BEFORE_COUPLER, "thermodynamics", "calculate the Exner function"),
    _scheme("temp_to_potential_temp", PHYSICS_BEFORE_COUPLER, "conversion", "convert temperature to potential temperature"),
    _scheme("calc_dry_air_ideal_gas_density", PHYSICS_BEFORE_COUPLER, "thermodynamics", "calculate dry-air density"),
    _scheme("wet_to_dry_water_vapor", PHYSICS_BEFORE_COUPLER, "conversion", "convert water vapor to dry mixing ratio"),
    _scheme("wet_to_dry_cloud_liquid_water", PHYSICS_BEFORE_COUPLER, "conversion", "convert cloud liquid to dry mixing ratio"),
    _scheme("wet_to_dry_rain", PHYSICS_BEFORE_COUPLER, "conversion", "convert rain to dry mixing ratio"),
    _scheme("kessler", PHYSICS_BEFORE_COUPLER, "microphysics", "run Kessler warm-rain microphysics", "fortran-kernel"),
    _scheme("potential_temp_to_temp", PHYSICS_BEFORE_COUPLER, "conversion", "convert potential temperature back to temperature"),
    _scheme("dry_to_wet_water_vapor", PHYSICS_BEFORE_COUPLER, "conversion", "convert water vapor to moist mixing ratio"),
    _scheme("dry_to_wet_cloud_liquid_water", PHYSICS_BEFORE_COUPLER, "conversion", "convert cloud liquid to moist mixing ratio"),
    _scheme("dry_to_wet_rain", PHYSICS_BEFORE_COUPLER, "conversion", "convert rain to moist mixing ratio"),
    _scheme("kessler_update", PHYSICS_BEFORE_COUPLER, "microphysics", "form the Kessler temperature tendency", "fortran-kernel"),
    _scheme("qneg", PHYSICS_BEFORE_COUPLER, "constraint", "apply constituent lower bounds"),
    _scheme("geopotential_temp", PHYSICS_BEFORE_COUPLER, "diagnostic", "refresh geopotential heights", "python+fortran-kernel"),
    _scheme("check_energy_zero_fluxes", PHYSICS_BEFORE_COUPLER, "conservation", "set closed-case energy fluxes to zero", "python-diagnostic"),
    _scheme("check_energy_scaling", PHYSICS_BEFORE_COUPLER, "conservation", "expose the pre-coupler energy-check boundary", "python-diagnostic"),
    _scheme("check_energy_chng", PHYSICS_BEFORE_COUPLER, "conservation", "check message-only energy bookkeeping", "python-diagnostic"),
    _scheme("sima_state_diagnostics", PHYSICS_BEFORE_COUPLER, "diagnostic", "capture the state-history boundary", "python-history"),
    _scheme("kessler_diagnostics", PHYSICS_BEFORE_COUPLER, "diagnostic", "capture Kessler precipitation diagnostics", "python-history"),
    _scheme("thermo_water_update", PHYSICS_AFTER_COUPLER, "thermodynamics", "update water-dependent SE heat capacity"),
    _scheme("check_energy_scaling", PHYSICS_AFTER_COUPLER, "conservation", "calculate SE temperature-increment scaling"),
    _scheme("dycore_energy_consistency_adjust", PHYSICS_AFTER_COUPLER, "conservation", "form the SE energy-consistency tendency"),
    _scheme("apply_tendency_of_air_temperature", PHYSICS_AFTER_COUPLER, "tendency", "apply and accumulate the temperature tendency"),
    _scheme("sima_tend_diagnostics", PHYSICS_AFTER_COUPLER, "diagnostic", "capture the physics-tendency boundary", "python-history"),
)

_DEFAULT_BY_KEY = {scheme.key: scheme for scheme in DEFAULT_KESSLER_SCHEMES}


class KesslerSchemePlan:
    """User-editable scheme sequence with a safe, BFB-validated default."""

    def __init__(
        self,
        schemes: Iterable[PhysicsScheme] = DEFAULT_KESSLER_SCHEMES,
        *,
        sequence_safe: bool = True,
    ) -> None:
        self._schemes = list(schemes)
        self._validate()
        self._sequence_safe = bool(sequence_safe)
        self._refresh_safety()

    @classmethod
    def default(cls) -> "KesslerSchemePlan":
        return cls(DEFAULT_KESSLER_SCHEMES)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "KesslerSchemePlan":
        rows = payload.get("schemes")
        if not isinstance(rows, list):
            raise ValueError("scheme plan payload requires a schemes list")
        schemes: list[PhysicsScheme] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError("each scheme plan row must be a mapping")
            group, name = str(row.get("group")), str(row.get("name"))
            source_group = str(row.get("source_group", group))
            cls._validate_group(group)
            key = f"{source_group}.{name}"
            try:
                template = _DEFAULT_BY_KEY[key]
            except KeyError as exc:
                raise ValueError(f"unknown FKESSLER scheme {key!r}") from exc
            enabled = row.get("enabled")
            if not isinstance(enabled, bool):
                raise TypeError(f"enabled for {key!r} must be bool")
            schemes.append(
                replace(template, group=group, enabled=enabled)
            )
        claimed_safe = payload.get("sequence_safe", False)
        if not isinstance(claimed_safe, bool):
            raise TypeError("scheme plan sequence_safe must be bool")
        return cls(schemes, sequence_safe=claimed_safe)

    @property
    def schemes(self) -> tuple[PhysicsScheme, ...]:
        return tuple(self._schemes)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(scheme.key for scheme in self._schemes)

    @property
    def sequence_safe(self) -> bool:
        return self._sequence_safe

    def copy(self) -> "KesslerSchemePlan":
        return KesslerSchemePlan(self._schemes, sequence_safe=self._sequence_safe)

    def active(self, group: str) -> tuple[PhysicsScheme, ...]:
        self._validate_group(group)
        return tuple(
            scheme for scheme in self._schemes
            if scheme.group == group and scheme.enabled
        )

    def scheme(
        self, name: str, *, group: str | None = None
    ) -> PhysicsScheme:
        if group is not None:
            self._validate_group(group)
        if name in _DEFAULT_BY_KEY:
            matches = [
                scheme for scheme in self._schemes
                if scheme.key == name
                and (group is None or scheme.group == group)
            ]
        elif group is not None:
            matches = [
                scheme for scheme in self._schemes
                if scheme.group == group and scheme.name == name
            ]
        else:
            matches = [scheme for scheme in self._schemes if scheme.name == name]
        if not matches:
            raise ValueError(f"unknown FKESSLER scheme {name!r}")
        if len(matches) > 1:
            identities = tuple(scheme.key for scheme in matches)
            raise ValueError(
                f"scheme {name!r} is ambiguous; use one of {identities}"
            )
        return matches[0]

    def enable(self, name: str, *, group: str | None = None) -> None:
        self._replace(self.scheme(name, group=group), enabled=True)
        self._refresh_safety()

    def disable(
        self,
        name: str,
        *,
        group: str | None = None,
        unsafe: bool = False,
    ) -> None:
        scheme = self.scheme(name, group=group)
        if scheme.required and not unsafe:
            raise ValueError(
                f"{scheme.key!r} is required by the validated FKESSLER suite; "
                "pass unsafe=True for an intentional experiment"
            )
        self._replace(scheme, enabled=False)
        self._sequence_safe = False

    def move(
        self,
        name: str,
        *,
        before: str | None = None,
        after: str | None = None,
        group: str | None = None,
        to_group: str | None = None,
        unsafe: bool = False,
    ) -> None:
        """Move a scheme within or across groups.

        An anchor determines the destination group. With only ``to_group``,
        the scheme is appended to that group. ``group`` selects a scheme by
        its current location; a source-qualified key is always unambiguous.
        """

        if before is not None and after is not None:
            raise ValueError("provide at most one of before= or after=")
        if before is None and after is None and to_group is None:
            raise ValueError("provide before=, after=, or to_group=")
        if to_group is not None:
            self._validate_group(to_group)
        moving = self.scheme(name, group=group)
        target_name = before if before is not None else after
        target = None
        if target_name is not None:
            target = self.scheme(target_name, group=to_group)
            if moving.key == target.key:
                raise ValueError("a scheme cannot be moved relative to itself")
            destination = target.group
        else:
            assert to_group is not None
            destination = to_group
        if not unsafe:
            raise ValueError("changing the validated scheme order requires unsafe=True")
        self._schemes.remove(moving)
        moved = replace(moving, group=destination)
        if target is None:
            group_indices = [
                index for index, scheme in enumerate(self._schemes)
                if scheme.group == destination
            ]
            insert_at = (
                group_indices[-1] + 1 if group_indices else len(self._schemes)
            )
        else:
            target_index = self._schemes.index(target)
            insert_at = target_index if before is not None else target_index + 1
        self._schemes.insert(insert_at, moved)
        self._sequence_safe = False

    def reset(self) -> None:
        self._schemes = list(DEFAULT_KESSLER_SCHEMES)
        self._sequence_safe = True

    def describe(self, group: str | None = None) -> list[dict[str, object]]:
        if group is not None:
            self._validate_group(group)
        selected = [
            scheme for scheme in self._schemes
            if group is None or scheme.group == group
        ]
        group_orders = {name: 0 for name in SCHEME_GROUPS}
        rows: list[dict[str, object]] = []
        for scheme in selected:
            group_orders[scheme.group] += 1
            rows.append(
                {
                    "order": group_orders[scheme.group],
                    "key": scheme.key,
                    "name": scheme.name,
                    "group": scheme.group,
                    "execution_group": scheme.group,
                    "source_group": scheme.source_group,
                    "category": scheme.category,
                    "description": scheme.description,
                    "implementation": scheme.implementation,
                    "required": scheme.required,
                    "enabled": scheme.enabled,
                }
            )
        return rows

    def to_payload(self) -> dict[str, object]:
        return {
            "sequence_safe": self._sequence_safe,
            "schemes": [
                {
                    "group": scheme.group,
                    "source_group": scheme.source_group,
                    "name": scheme.name,
                    "enabled": scheme.enabled,
                }
                for scheme in self._schemes
            ],
        }

    def _replace(self, current: PhysicsScheme, **changes: Any) -> None:
        index = self._schemes.index(current)
        self._schemes[index] = replace(current, **changes)

    def _refresh_safety(self) -> None:
        if self._schemes != list(DEFAULT_KESSLER_SCHEMES):
            self._sequence_safe = False
        elif all(scheme.enabled for scheme in self._schemes):
            self._sequence_safe = True

    def _validate(self) -> None:
        if len(self.keys) != len(set(self.keys)):
            raise ValueError("scheme source identities must be unique")
        if set(self.keys) != set(_DEFAULT_BY_KEY):
            raise ValueError("a scheme plan must contain every FKESSLER scheme exactly once")
        for scheme in self._schemes:
            self._validate_group(scheme.group)

    @staticmethod
    def _validate_group(group: str) -> None:
        if group not in SCHEME_GROUPS:
            raise ValueError(f"unknown scheme group {group!r}; choose from {SCHEME_GROUPS}")
