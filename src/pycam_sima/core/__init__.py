"""Shared Python infrastructure."""

from .mpi import world_comm
from .remote import RemoteCAMField

__all__ = [
    "RemoteCAMField",
    "world_comm",
]
