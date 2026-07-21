"""Client-side handles for state stored on MPI workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RemoteCAMField:
    """Typed handle for reading or updating one live worker field."""

    def __init__(self, session: Any, name: str) -> None:
        self._session = session
        self.name = name

    @property
    def info(self) -> Mapping[str, Any]:
        return self._session.field_info(self.name)

    def get(self, *, rank: int | str = 0) -> Any:
        return self._session.get_field(self.name, rank=rank)

    def stats(self, *, rank: int | str = 0) -> Any:
        return self._session.get_field_stats(self.name, rank=rank)

    def set(
        self, value: Any, *, rank: int | str = 0, unsafe: bool = False
    ) -> None:
        self._session.set_field(self.name, value, rank=rank, unsafe=unsafe)
