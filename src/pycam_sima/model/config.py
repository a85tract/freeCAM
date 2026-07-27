"""Serializable case configuration independent of a particular physics suite."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError
from .clock import month_lengths, normalize_calendar


REFERENCE_SOURCE_REVISION = "f8daa568eae2696b7c4ebff7768f02f5d097d9df"
SUPPORTED_CALENDARS = frozenset(
    {"NO_LEAP", "GREGORIAN", "PROLEPTIC_GREGORIAN", "JULIAN", "ALL_LEAP", "360_DAY"}
)
SUPPORTED_ANALYTIC_INITIAL_STATES = frozenset(
    {"moist_baroclinic_wave_dcmip2016", "resting_isothermal"}
)


def _default_constituent_metadata(
    count: int,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    base_names = ("cloud_liquid_water", "rain_water", "water_vapor")
    names = tuple(
        base_names[index] if index < len(base_names) else f"tracer_{index + 1}"
        for index in range(count)
    )
    minima = tuple(1.0e-12 if index == 2 else 0.0 for index in range(count))
    molecular_weights = tuple(18.016 for _ in range(count))
    return names, minima, molecular_weights


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Describe one model case without deciding which runtime can execute it.

    Generic checks live here. Grid/dycore/initial-condition limitations belong
    to the selected runtime capability provider and are checked by CAMDriver.
    """

    source_revision: str = REFERENCE_SOURCE_REVISION
    source_root: str = str(
        Path(__file__).resolve().parents[3] / "external" / "CAM-SIMA"
    )
    physics_suite: str = "kessler"
    suite_xml: str | None = None
    grid: str = "ne3np4.pg3"
    ne: int = 3
    np: int = 4
    fv_nphys: int = 3
    pver: int = 30
    constituent_count: int = 3
    mpi_size: int = 24
    threads_per_rank: int = 1
    dt_seconds: int = 1800
    stop_n: int = 50
    calendar: str = "NO_LEAP"
    run_type: str = "startup"
    analytic_ic_type: str = "moist_baroclinic_wave_dcmip2016"
    pertlim: float = 0.0
    start_date: str = "0001-01-01"
    start_seconds: int = 0
    restart_path: str | None = None
    constituent_names: tuple[str, ...] = (
        "cloud_liquid_water",
        "rain_water",
        "water_vapor",
    )
    constituent_minima: tuple[float, ...] = (0.0, 0.0, 1.0e-12)
    constituent_molecular_weights: tuple[float, ...] = (
        18.016,
        18.016,
        18.016,
    )
    initial_temperature: float = 300.0
    initial_surface_pressure: float = 100000.0
    case_name: str = "test_ne3_baseline_verify"
    atm_in: str = "atm_in"
    history_enabled: bool = True

    def __post_init__(self) -> None:
        # Direct dataclass construction should be as useful as from_mapping().
        # If only the count changed, expand the untouched reference defaults.
        reference_metadata = (
            ("cloud_liquid_water", "rain_water", "water_vapor"),
            (0.0, 0.0, 1.0e-12),
            (18.016, 18.016, 18.016),
        )
        current_metadata = (
            tuple(self.constituent_names),
            tuple(self.constituent_minima),
            tuple(self.constituent_molecular_weights),
        )
        object.__setattr__(self, "constituent_names", current_metadata[0])
        object.__setattr__(self, "constituent_minima", current_metadata[1])
        object.__setattr__(
            self,
            "constituent_molecular_weights",
            current_metadata[2],
        )
        if self.constituent_count != 3 and current_metadata == reference_metadata:
            names, minima, weights = _default_constituent_metadata(
                self.constituent_count
            )
            object.__setattr__(self, "constituent_names", names)
            object.__setattr__(self, "constituent_minima", minima)
            object.__setattr__(
                self,
                "constituent_molecular_weights",
                weights,
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigurationError(f"unknown configuration keys: {unknown}")
        normalized = dict(values)
        if isinstance(normalized.get("start_date"), date):
            normalized["start_date"] = normalized["start_date"].isoformat()
        if "constituent_count" in normalized:
            count = int(normalized["constituent_count"])
            names, minima, weights = _default_constituent_metadata(count)
            normalized.setdefault("constituent_names", names)
            normalized.setdefault("constituent_minima", minima)
            normalized.setdefault("constituent_molecular_weights", weights)
        for name in (
            "constituent_names",
            "constituent_minima",
            "constituent_molecular_weights",
        ):
            if name in normalized:
                normalized[name] = tuple(normalized[name])
        config = cls(**normalized)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream) or {}
        if not isinstance(values, Mapping):
            raise ConfigurationError("configuration YAML must contain a mapping")
        return cls.from_mapping(values)

    def with_overrides(self, **values: Any) -> "ModelConfig":
        if "constituent_count" in values:
            count = int(values["constituent_count"])
            metadata_names = {
                "constituent_names",
                "constituent_minima",
                "constituent_molecular_weights",
            }
            if not metadata_names.intersection(values):
                names, minima, weights = _default_constituent_metadata(count)
                values.update(
                    constituent_names=names,
                    constituent_minima=minima,
                    constituent_molecular_weights=weights,
                )
        result = replace(self, **values)
        result.validate()
        return result

    def validate(self) -> None:
        errors: list[str] = []
        for name in (
            "source_revision",
            "source_root",
            "physics_suite",
            "grid",
            "calendar",
            "run_type",
            "analytic_ic_type",
            "case_name",
            "atm_in",
        ):
            if not str(getattr(self, name)).strip():
                errors.append(f"{name} must be non-empty")
        for name in (
            "ne",
            "np",
            "fv_nphys",
            "pver",
            "constituent_count",
            "mpi_size",
            "threads_per_rank",
            "dt_seconds",
            "stop_n",
        ):
            if int(getattr(self, name)) <= 0:
                errors.append(f"{name} must be positive")
        if self.np < 2:
            errors.append("np must be at least 2")
        grid_match = re.fullmatch(
            r"ne([0-9]+)np([0-9]+)\.pg([0-9]+)",
            self.grid.lower(),
        )
        if grid_match is None:
            errors.append("grid must use ne<N>np<N>.pg<N> syntax")
        else:
            encoded = tuple(int(value) for value in grid_match.groups())
            configured = (self.ne, self.np, self.fv_nphys)
            if encoded != configured:
                errors.append(
                    f"grid={self.grid!r} encodes {encoded}, but "
                    f"(ne, np, fv_nphys)={configured}"
                )
        run_type = self.run_type.lower()
        if run_type not in {"startup", "continue", "branch"}:
            errors.append(
                "run_type must be startup, continue, or branch"
            )
        calendar = normalize_calendar(self.calendar)
        if calendar not in SUPPORTED_CALENDARS:
            errors.append(
                f"calendar must be one of {sorted(SUPPORTED_CALENDARS)}"
            )
        try:
            date_parts = tuple(int(part) for part in self.start_date.split("-"))
            if len(date_parts) != 3:
                raise ValueError
            year, month, day = date_parts
            if year < 1 or not 1 <= month <= 12:
                raise ValueError
            if not 1 <= day <= month_lengths(year, calendar)[month - 1]:
                raise ValueError
        except (AttributeError, ConfigurationError, TypeError, ValueError):
            errors.append("start_date must use YYYY-MM-DD")
        if not 0 <= int(self.start_seconds) < 86400:
            errors.append("start_seconds must be between 0 and 86399")
        if run_type == "startup":
            if self.analytic_ic_type.lower() not in SUPPORTED_ANALYTIC_INITIAL_STATES:
                errors.append(
                    "analytic_ic_type must be one of "
                    f"{sorted(SUPPORTED_ANALYTIC_INITIAL_STATES)} for startup"
                )
            if (
                self.analytic_ic_type.lower()
                == "moist_baroclinic_wave_dcmip2016"
                and self.constituent_count < 3
            ):
                errors.append(
                    "moist_baroclinic_wave_dcmip2016 requires at least "
                    "three constituents"
                )
        elif not self.restart_path:
            errors.append(f"restart_path is required for run_type={run_type}")
        metadata_lengths = {
            "constituent_names": len(self.constituent_names),
            "constituent_minima": len(self.constituent_minima),
            "constituent_molecular_weights": len(
                self.constituent_molecular_weights
            ),
        }
        for name, length in metadata_lengths.items():
            if length != self.constituent_count:
                errors.append(
                    f"{name} has {length} values, expected "
                    f"constituent_count={self.constituent_count}"
                )
        normalized_names = tuple(
            str(name).strip().lower() for name in self.constituent_names
        )
        if any(not name for name in normalized_names):
            errors.append("constituent_names cannot contain empty names")
        if len(set(normalized_names)) != len(normalized_names):
            errors.append("constituent_names must be unique")
        if any(float(value) < 0.0 for value in self.constituent_minima):
            errors.append("constituent_minima cannot be negative")
        if any(
            float(value) <= 0.0
            for value in self.constituent_molecular_weights
        ):
            errors.append("constituent_molecular_weights must be positive")
        if float(self.initial_temperature) <= 0.0:
            errors.append("initial_temperature must be positive")
        if float(self.initial_surface_pressure) <= 0.0:
            errors.append("initial_surface_pressure must be positive")
        if errors:
            raise ConfigurationError(
                "invalid model configuration: " + "; ".join(errors)
            )

    def resolve_suite_xml(self) -> Path:
        """Return the configured CCPP suite XML without assuming FKESSLER."""

        if self.suite_xml is not None:
            path = Path(self.suite_xml).expanduser()
            if not path.is_absolute():
                path = Path(self.source_root) / path
        else:
            path = (
                Path(self.source_root)
                / "src"
                / "physics"
                / "ncar_ccpp"
                / "suites"
                / f"suite_{self.physics_suite.lower()}.xml"
            )
        return path.resolve()

    def resolve_atm_in(self, run_dir: str | Path) -> Path:
        value = Path(self.atm_in)
        return value if value.is_absolute() else Path(run_dir) / value

    def resolve_restart_path(self, run_dir: str | Path) -> Path | None:
        if self.restart_path is None:
            return None
        value = Path(self.restart_path).expanduser()
        return value.resolve() if value.is_absolute() else (
            Path(run_dir) / value
        ).resolve()

    @property
    def kernel_specialization(self) -> dict[str, int]:
        """Compile-time dimensions of one stateless native-kernel build."""

        return {
            "np": int(self.np),
            "fv_nphys": int(self.fv_nphys),
            "pver": int(self.pver),
            "constituent_count": int(self.constituent_count),
        }

    @property
    def kernel_specialization_id(self) -> str:
        encoded = json.dumps(
            self.kernel_specialization,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]

    def default_kernel_library(self, project_root: str | Path) -> Path:
        """Return the default or cached library for this specialization."""

        root = Path(project_root)
        reference = ModelConfig()
        if self.kernel_specialization == reference.kernel_specialization:
            return root / "build" / "libpycam_sima_kernels.so"
        return (
            root
            / "build"
            / "kernels"
            / self.kernel_specialization_id
            / "libpycam_sima_kernels.so"
        )

    def verify_source_revision(self) -> None:
        root = Path(self.source_root)
        if not (root / ".git").exists():
            raise ConfigurationError(f"CAM-SIMA source_root is not a git checkout: {root}")
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()
        if result != self.source_revision:
            raise ConfigurationError(
                f"CAM-SIMA checkout is {result}, required {self.source_revision}: {root}"
            )

    def verify_suite(self) -> Path:
        path = self.resolve_suite_xml()
        if not path.is_file():
            raise ConfigurationError(
                f"CCPP suite XML does not exist for "
                f"{self.physics_suite!r}: {path}"
            )
        return path

    def as_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}
