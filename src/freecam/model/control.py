"""Notebook-side model options and field metadata."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from ..core.remote import RemoteCAMField


@dataclass(slots=True)
class ModelOptions:
    timestep_seconds: int = 1800
    physics_profile: str = "kessler"
    mediator_present: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "ModelOptions":
        return cls(
            timestep_seconds=int(config.dt_seconds),
            physics_profile=str(config.physics_suite),
            mediator_present=False,
        )

    def validate(self, config: Any | None = None) -> None:
        if self.timestep_seconds <= 0:
            raise ValueError("the model timestep must be positive")
        if not self.physics_profile.strip():
            raise ValueError("the physics suite name must be non-empty")
        if self.mediator_present:
            raise ValueError(
                "the current CAM component runtime is ATM-only and has no mediator"
            )
        if config is not None:
            if int(config.dt_seconds) != self.timestep_seconds:
                raise ValueError("runtime timestep differs from the model config")
            if str(config.physics_suite) != self.physics_profile:
                raise ValueError("runtime suite differs from the model config")

    def describe(self) -> dict[str, object]:
        result = asdict(self)
        result.update(
            runtime="model",
            state_owner="python",
            initialization="pure-python",
            mutable_until="phase boundary",
        )
        return result

    def fingerprint(self) -> tuple[int, str, bool]:
        return self.timestep_seconds, self.physics_profile, self.mediator_present


class ModelParameters:
    """Typed handles plus generic access to all Python-owned fields."""

    KEY_FIELDS = (
        "air_temperature",
        "zonal_wind",
        "meridional_wind",
        "surface_pressure",
        "layer_pressure_thickness",
        "constituent_mixing_ratio",
        "physics_air_temperature",
        "physics_water_vapor",
        "physics_cloud_liquid_water",
        "physics_rain_water",
    )

    def __init__(self, session: Any) -> None:
        self._session = session

    def field(self, name: str) -> RemoteCAMField:
        if name not in self._session.field_names:
            raise KeyError(f"unknown model field {name!r}")
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
    def zonal_wind(self) -> RemoteCAMField:
        return self.field("zonal_wind")

    @property
    def meridional_wind(self) -> RemoteCAMField:
        return self.field("meridional_wind")

    @property
    def surface_pressure(self) -> RemoteCAMField:
        return self.field("surface_pressure")

    @property
    def constituents(self) -> RemoteCAMField:
        return self.field("constituent_mixing_ratio")

    @property
    def physics_air_temperature(self) -> RemoteCAMField:
        return self.field("physics_air_temperature")

    @property
    def physics_water_vapor(self) -> RemoteCAMField:
        return self.field("physics_water_vapor")


def model_field_metadata(pool: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    names = tuple(pool.contracts) + tuple(pool._aliases)
    for name in names:
        array = pool.get(name)
        contract = pool.contract(name)
        result[name] = {
            "standard_name": contract.standard_name,
            "ccpp_standard_name": contract.ccpp_standard_name,
            "shape": tuple(array.shape),
            "dtype": array.dtype.str,
            "dimensions": contract.dimensions,
            "intent": contract.intent,
            "owner": contract.owner,
            "lifetime": contract.lifetime,
            "category": contract.category,
            "units": contract.units,
            "writable": contract.writable,
            "alias": name != contract.standard_name,
        }
    return result
