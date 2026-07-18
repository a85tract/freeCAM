from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SerialComm:
    rank: int = 0
    size: int = 1

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value

    def gather(self, value: Any, root: int = 0) -> list[Any] | None:
        return [value] if root == 0 else None

    def Barrier(self) -> None:
        return None


def world_comm() -> Any:
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError(
            "mpi4py is installed but its MPI runtime is unavailable; run through "
            "the pycam-sima CLI or configure LD_LIBRARY_PATH for the site MPI"
        ) from exc
    return MPI.COMM_WORLD
