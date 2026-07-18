from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PINNED_CAM_SIMA_COMMIT = "f8daa568eae2696b7c4ebff7768f02f5d097d9df"


@dataclass(frozen=True)
class NativeConfig:
    kessler_library: Path
    se_library: Path


@dataclass(frozen=True)
class CaseConfig:
    name: str
    compset: str
    resolution: str
    steps: int
    dt_seconds: int
    mpi_ranks: int
    threads_per_rank: int
    pver: int
    se_ne: int
    physics_suite: str
    analytic_ic_type: str
    mediator_present: bool
    mode: str
    cam_sima: Path
    suite_xml: Path
    commit: str
    native: NativeConfig
    observers: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CaseConfig":
        config_path = Path(path).resolve()
        root = config_path.parent.parent
        raw = yaml.safe_load(config_path.read_text())
        case = raw["case"]
        source = raw["source"]
        native = raw["native"]

        def rooted(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else (root / candidate).resolve()

        result = cls(
            **case,
            cam_sima=rooted(source["cam_sima"]),
            suite_xml=rooted(str(Path(source["cam_sima"]) / source["suite_xml"])),
            commit=source["commit"],
            native=NativeConfig(
                kessler_library=rooted(native["kessler_library"]),
                se_library=rooted(native["se_library"]),
            ),
            observers=tuple(raw.get("observers", ())),
            config_path=config_path,
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.compset != "FKESSLER":
            raise ValueError(f"Only FKESSLER is supported, got {self.compset!r}")
        if self.physics_suite != "kessler":
            raise ValueError("physics_suite must be 'kessler'")
        if self.mediator_present:
            raise ValueError("v1 implements the ATM-only, no-mediator run sequence")
        if self.pver != 30 or self.se_ne != 3:
            raise ValueError("the supported v1 kernel layout is pver=30 and se_ne=3")
        if self.dt_seconds != 1800:
            raise ValueError("the validated FKESSLER step is 1800 seconds")
        if self.commit != PINNED_CAM_SIMA_COMMIT:
            raise ValueError(
                f"CAM-SIMA must be pinned to {PINNED_CAM_SIMA_COMMIT}, got {self.commit}"
            )
        if self.mode not in {"interactive", "validation"}:
            raise ValueError("mode must be 'interactive' or 'validation'")
