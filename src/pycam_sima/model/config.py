"""Configuration and input validation for the fixed FKESSLER model."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
import subprocess
from typing import Any, Mapping

import yaml

from .errors import ConfigurationError


SUPPORTED_REVISION = "f8daa568eae2696b7c4ebff7768f02f5d097d9df"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    source_revision: str = SUPPORTED_REVISION
    source_root: str = str(
        Path(__file__).resolve().parents[3] / "external" / "CAM-SIMA"
    )
    physics_suite: str = "kessler"
    grid: str = "ne3np4.pg3"
    ne: int = 3
    np: int = 4
    fv_nphys: int = 3
    pver: int = 30
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
        expected = {
            "source_revision": SUPPORTED_REVISION,
            "physics_suite": "kessler",
            "grid": "ne3np4.pg3",
            "ne": 3,
            "np": 4,
            "fv_nphys": 3,
            "pver": 30,
            "mpi_size": 24,
            "threads_per_rank": 1,
            "dt_seconds": 1800,
            "stop_n": 50,
            "calendar": "NO_LEAP",
            "run_type": "startup",
            "analytic_ic_type": "moist_baroclinic_wave_dcmip2016",
            "pertlim": 0.0,
        }
        errors = []
        for name, wanted in expected.items():
            actual = getattr(self, name)
            if actual != wanted:
                errors.append(f"{name}={actual!r}, required {wanted!r}")
        if errors:
            raise ConfigurationError(
                "the model supports one fixed case: " + "; ".join(errors)
            )

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

    def as_dict(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}
