"""High-level PI-CAM-only case factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .boundary import CAMBoundaryProvider, ReplayBoundaryProvider
from .config import DEFAULT_TRACE_LIMIT, PICAMConfig
from .driver import PICAMDriver
from .errors import PICAMConfigurationError
from .native import CAMNumericalBackend, NativeCAMDevice
from .plan import PICAMStepPlan


class PICAMCase:
    def __init__(self, config: PICAMConfig) -> None:
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PICAMCase":
        return cls(PICAMConfig.from_yaml(path))

    def runtime(
        self,
        *,
        boundary: CAMBoundaryProvider | str | Path,
        backend: CAMNumericalBackend | None = None,
        communicator: Any | None = None,
        step_plan: PICAMStepPlan | None = None,
        run_dir: str | Path | None = None,
        trace_limit: int | None = DEFAULT_TRACE_LIMIT,
    ) -> PICAMDriver:
        if isinstance(boundary, (str, Path)):
            boundary = ReplayBoundaryProvider(boundary)
        if communicator is None:
            try:
                from mpi4py import MPI
            except ImportError as exc:
                raise PICAMConfigurationError(
                    "mpi4py is required unless communicator is provided"
                ) from exc
            communicator = MPI.COMM_WORLD
        rank = int(communicator.Get_rank())
        size = int(communicator.Get_size())
        try:
            fcomm = int(communicator.py2f())
        except AttributeError:
            fcomm = 0
        if backend is None:
            if self.config.native_manifest is None:
                raise PICAMConfigurationError(
                    "native_manifest or an explicit numerical backend is required"
                )
            backend = NativeCAMDevice(self.config.native_manifest)
        return PICAMDriver(
            self.config,
            boundary,
            backend,
            rank=rank,
            size=size,
            fcomm=fcomm,
            communicator=communicator,
            step_plan=step_plan,
            run_dir=run_dir,
            trace_limit=trace_limit,
        )
