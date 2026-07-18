from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Callable, Protocol

import numpy as np

from .state_pool import StatePool


class CommLike(Protocol):
    rank: int
    size: int

    def gather(self, value: object, root: int = 0) -> list[object] | None: ...


@dataclass
class ObserverContext:
    step: int
    rank: int
    size: int
    phase: str
    task_name: str
    state: StatePool
    clock_seconds: int
    comm: CommLike

    def gather(
        self, field: str, *, root: int = 0, axis: int = 0
    ) -> np.ndarray | None:
        pieces = self.comm.gather(np.array(self.state[field], copy=True, order="F"), root=root)
        if self.rank != root:
            return None
        assert pieces is not None
        arrays = [np.asarray(piece) for piece in pieces]
        return np.concatenate(arrays, axis=axis) if arrays[0].ndim else np.asarray(arrays)


Callback = Callable[[ObserverContext], None]


@dataclass(frozen=True)
class Subscription:
    pattern: str
    callback: Callback
    access: str


class ObserverRegistry:
    def __init__(self, *, mode: str = "interactive") -> None:
        if mode not in {"interactive", "validation"}:
            raise ValueError("mode must be interactive or validation")
        self.mode = mode
        self._subscriptions: list[Subscription] = []

    def observe(self, event: str, callback: Callback, *, access: str = "readwrite") -> None:
        if access not in {"readonly", "readwrite"}:
            raise ValueError("access must be readonly or readwrite")
        if self.mode == "validation" and access == "readwrite":
            raise ValueError("validation observers must be readonly")
        self._subscriptions.append(Subscription(event, callback, access))

    def emit(self, event: str, context: ObserverContext) -> None:
        for subscription in self._subscriptions:
            if not fnmatch(event, subscription.pattern):
                continue
            writable = self.mode == "interactive" and subscription.access == "readwrite"
            with context.state.callback_access(writable=writable):
                subscription.callback(context)
