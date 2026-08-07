"""Runtime capability checks kept separate from generic case configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .config import ModelConfig, REFERENCE_SOURCE_REVISION
from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Machine-readable constraints of one concrete model implementation."""

    name: str
    source_revisions: tuple[str, ...]
    constraints: Mapping[str, Any]
    bounds: Mapping[str, tuple[int | None, int | None]] = field(
        default_factory=dict
    )

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
        for name, (minimum, maximum) in self.bounds.items():
            actual = getattr(config, name)
            if minimum is not None and actual < minimum:
                errors.append(
                    f"{name}={actual!r}, minimum available {minimum!r}"
                )
            if maximum is not None and actual > maximum:
                errors.append(
                    f"{name}={actual!r}, maximum available {maximum!r}"
                )
        element_count = 6 * int(config.ne) * int(config.ne)
        if config.mpi_size > element_count:
            errors.append(
                f"mpi_size={config.mpi_size!r}, maximum available "
                f"{element_count} nonempty SE partitions for ne={config.ne}"
            )
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
            "bounds": {
                name: {"minimum": values[0], "maximum": values[1]}
                for name, values in self.bounds.items()
            },
        }


CAM_SE_FVM_V1 = RuntimeCapabilities(
    name="cam-se-fvm-v2-configurable",
    source_revisions=(REFERENCE_SOURCE_REVISION,),
    constraints={
        "threads_per_rank": 1,
    },
    bounds={
        "ne": (1, None),
        "np": (2, None),
        "fv_nphys": (1, None),
        "pver": (1, None),
        "constituent_count": (1, None),
        "mpi_size": (1, None),
    },
)
