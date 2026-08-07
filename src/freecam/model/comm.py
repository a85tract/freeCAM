"""Communicator protocol used by model initialization and history code."""

from __future__ import annotations

from typing import Any, Protocol


class Comm(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def size(self) -> int: ...

    def bcast(self, value: Any, root: int = 0) -> Any: ...

    def gather(self, value: Any, root: int = 0) -> Any: ...

    def allgather(self, value: Any) -> list[Any]: ...

    def barrier(self) -> None: ...


class SerialComm:
    rank = 0
    size = 1

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value

    def gather(self, value: Any, root: int = 0) -> list[Any]:
        return [value]

    def allgather(self, value: Any) -> list[Any]:
        return [value]

    def barrier(self) -> None:
        return None


def world_comm() -> Comm:
    try:
        from mpi4py import MPI
    except ImportError:
        return SerialComm()
    return MPI.COMM_WORLD
