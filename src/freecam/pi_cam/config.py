"""Configuration for the admitted iCESM1.3.1 PI-CAM-only case."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .errors import PICAMConfigurationError


@dataclass(frozen=True, slots=True)
class PICAMConfig:
    """Fail-closed configuration for one PI-CAM MPI rank."""

    case_name: str
    source_root: Path
    resolution: str = "ne16"
    mpi_size: int = 512
    calendar: str = "NO_LEAP"
    start_date: str = "0001-01-01"
    start_seconds: int = 0
    orbital_year: int = 1850
    timestep_seconds: int = 1800
    coupling_seconds: int = 1800
    stop_n: int = 50
    initialization_lookahead_steps: int = 0
    pver: int = 30
    pcols: int = 16
    physics_package: str = "cam5"
    dynamics: str = "se"
    boundary_mode: str = "replay"
    execution_mode: str = "fine_grained"
    native_manifest: Path | None = None
    initial_conditions: Path | None = None
    namelist: Path | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        for name in ("native_manifest", "initial_conditions", "namelist"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if self.schema_version != 1:
            raise PICAMConfigurationError(
                f"unsupported PI-CAM schema version {self.schema_version}"
            )
        if self.resolution != "ne16":
            raise PICAMConfigurationError(
                "the first PI-CAM implementation admits only ne16"
            )
        if self.physics_package.lower() != "cam5" or self.dynamics.lower() != "se":
            raise PICAMConfigurationError(
                "the admitted case requires CAM5 physics and SE dynamics"
            )
        if self.mpi_size < 1 or self.stop_n < 1:
            raise PICAMConfigurationError("mpi_size and stop_n must be positive")
        if not -1_000_000 <= self.orbital_year <= 1_000_000:
            raise PICAMConfigurationError("orbital_year is outside the CAM orbit range")
        if self.initialization_lookahead_steps < 0:
            raise PICAMConfigurationError(
                "initialization_lookahead_steps cannot be negative"
            )
        if self.pver < 1 or self.pcols < 1:
            raise PICAMConfigurationError("pver and pcols must be positive")
        if self.timestep_seconds < 1 or self.coupling_seconds < 1:
            raise PICAMConfigurationError("timesteps must be positive")
        if self.coupling_seconds % self.timestep_seconds:
            raise PICAMConfigurationError(
                "coupling_seconds must be an integer multiple of timestep_seconds"
            )
        if self.calendar.upper() not in {"NO_LEAP", "NOLEAP", "365_DAY"}:
            raise PICAMConfigurationError("the PI-CAM oracle uses a NO_LEAP calendar")
        if self.boundary_mode not in {"replay", "memory", "online"}:
            raise PICAMConfigurationError(
                "boundary_mode must be 'replay', 'memory', or 'online'"
            )
        if self.execution_mode not in {
            "fine_grained",
            "source_compat",
        }:
            raise PICAMConfigurationError(
                "execution_mode must be 'fine_grained' or 'source_compat'"
            )
        try:
            year, month, day = (int(item) for item in self.start_date.split("-"))
        except (TypeError, ValueError) as exc:
            raise PICAMConfigurationError(
                "start_date must have YYYY-MM-DD form"
            ) from exc
        if year < 1 or not 1 <= month <= 12 or not 1 <= day <= 31:
            raise PICAMConfigurationError("start_date is outside the supported range")

    @property
    def substeps_per_coupling(self) -> int:
        return self.coupling_seconds // self.timestep_seconds

    @property
    def start_parts(self) -> tuple[int, int, int]:
        return tuple(int(item) for item in self.start_date.split("-"))  # type: ignore[return-value]

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("source_root", "native_manifest", "initial_conditions", "namelist"):
            value = payload[name]
            payload[name] = None if value is None else str(value)
        return payload

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return scientific settings without checkout- or build-local paths."""

        payload = self.to_payload()
        payload.pop("source_root")
        payload.pop("native_manifest")
        # Runtime orchestration does not change the captured atmosphere
        # boundary contract.  Keeping it out lets the same oracle payload
        # compare the fine-grained and source-compatible control paths.
        payload.pop("execution_mode")
        return payload

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.fingerprint_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PICAMConfig":
        admitted = {
            "schema_version",
            "case_name",
            "source_root",
            "resolution",
            "mpi_size",
            "calendar",
            "start_date",
            "start_seconds",
            "orbital_year",
            "timestep_seconds",
            "coupling_seconds",
            "stop_n",
            "initialization_lookahead_steps",
            "pver",
            "pcols",
            "physics_package",
            "dynamics",
            "boundary_mode",
            "execution_mode",
            "native_manifest",
            "initial_conditions",
            "namelist",
        }
        unknown = sorted(set(values) - admitted)
        if unknown:
            raise PICAMConfigurationError(
                "unsupported PI-CAM configuration keys: " + ", ".join(unknown)
            )
        return cls(**{name: values[name] for name in admitted if name in values})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PICAMConfig":
        source = Path(path).resolve()
        values = yaml.safe_load(source.read_text())
        if not isinstance(values, Mapping):
            raise PICAMConfigurationError("PI-CAM YAML root must be a mapping")
        values = dict(values)
        base = source.parent.parent if source.parent.name == "configs" else source.parent
        for name in ("source_root", "native_manifest", "initial_conditions", "namelist"):
            value = values.get(name)
            if value is not None and not Path(value).is_absolute():
                values[name] = base / value
        return cls.from_mapping(values)


DEFAULT_TRACE_LIMIT: int = 4096


def validate_trace_limit(value: int | None) -> int | None:
    """Validate a per-rank action-trace retention limit; ``None`` is unbounded."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PICAMConfigurationError(
            "trace_limit must be a positive integer or None"
        )
    return value
