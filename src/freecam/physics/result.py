"""What one call of a physics function returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

STATUSES = ("ok", "invalid_input", "fortran_abort", "worker_crash", "internal_error", "stub_called")


@dataclass(frozen=True)
class FunctionResult:
    """Outputs, updated in/out values, and how the call ended.

    ``status`` is ``ok`` when the routine returned; every other status means
    ``outputs`` is empty and ``message`` says why.  A sample that aborted in
    Fortran is ``fortran_abort`` -- data the routine refused, not a failure
    of the harness -- while ``stub_called`` marks an implementation error.
    """

    outputs: Mapping[str, np.ndarray]
    updated_inputs: Mapping[str, np.ndarray]
    status: str = "ok"
    message: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown result status {self.status!r}")

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def __getitem__(self, name: str) -> np.ndarray:
        key = name.lower()
        for table in (self.outputs, self.updated_inputs):
            for item, value in table.items():
                if item.lower() == key:
                    return value
        raise KeyError(name)

    def __repr__(self) -> str:
        return (
            f"FunctionResult(status={self.status!r}, outputs={tuple(self.outputs)!r}, "
            f"updated_inputs={tuple(self.updated_inputs)!r})"
        )


__all__ = ["STATUSES", "FunctionResult"]
