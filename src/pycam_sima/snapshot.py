from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .observer import ObserverContext


class NpzSnapshotWriter:
    """Observer callback that writes selected Python-pool arrays without conversion."""

    def __init__(self, directory: str | Path, fields: tuple[str, ...] = ()) -> None:
        self.directory = Path(directory)
        self.fields = fields

    def __call__(self, context: ObserverContext) -> None:
        names = self.fields or tuple(context.state)
        missing = [name for name in names if name not in context.state]
        if missing:
            raise KeyError(f"snapshot fields are missing: {', '.join(missing)}")
        event = re.sub(r"[^A-Za-z0-9_.-]+", "_", context.phase + "_" + context.task_name)
        target = self.directory / event / f"step_{context.step:04d}" / f"rank_{context.rank:05d}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **{name: context.state.require(name) for name in names})
