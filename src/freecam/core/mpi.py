"""MPI communicator access."""

from __future__ import annotations

from typing import Any


def world_comm() -> Any:
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError(
            "mpi4py is installed but its MPI runtime is unavailable; run through "
            "the freecam CLI or configure LD_LIBRARY_PATH for the site MPI"
        ) from exc
    return MPI.COMM_WORLD
