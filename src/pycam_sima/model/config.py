"""Serializable case configuration independent of a particular physics suite."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError


REFERENCE_SOURCE_REVISION = "f8daa568eae2696b7c4ebff7768f02f5d097d9df"


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
    case_name: str = "test_ne3_baseline_verify"
    atm_in: str = "atm_in"
    history_enabled: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigurationError(f"unknown configuration keys: {unknown}")
        config = cls(**dict(values))
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
        if self.fv_nphys > self.np:
            errors.append("fv_nphys cannot exceed np")
        if self.run_type.lower() not in {"startup", "continue", "branch"}:
            errors.append(
                "run_type must be startup, continue, or branch"
            )
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
