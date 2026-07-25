"""Runtime capability checks kept separate from generic case configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import ModelConfig, REFERENCE_SOURCE_REVISION
from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Machine-readable constraints of one concrete model implementation."""

    name: str
    source_revisions: tuple[str, ...]
    constraints: Mapping[str, Any]

    def validate(self, config: ModelConfig) -> None:
        errors: list[str] = []
        if (
            self.source_revisions
            and config.source_revision not in self.source_revisions
        ):
            errors.append(
                f"source_revision={config.source_revision!r}, available "
                f"{self.source_revisions}"
            )
        for name, wanted in self.constraints.items():
            actual = getattr(config, name)
            if actual != wanted:
                errors.append(f"{name}={actual!r}, available {wanted!r}")
        if errors:
            raise ConfigurationError(
                f"runtime {self.name!r} cannot execute this case: "
                + "; ".join(errors)
            )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_revisions": self.source_revisions,
            "constraints": dict(self.constraints),
        }


CAM_SE_FVM_V1 = RuntimeCapabilities(
    name="cam-se-fvm-v1",
    source_revisions=(REFERENCE_SOURCE_REVISION,),
    constraints={
        "grid": "ne3np4.pg3",
        "ne": 3,
        "np": 4,
        "fv_nphys": 3,
        "pver": 30,
        "constituent_count": 3,
        "mpi_size": 24,
        "threads_per_rank": 1,
        "calendar": "NO_LEAP",
        "run_type": "startup",
        "analytic_ic_type": "moist_baroclinic_wave_dcmip2016",
    },
)
