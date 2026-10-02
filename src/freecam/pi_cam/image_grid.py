"""The grid a native image was compiled for: CAM's ppgrid extents, read from its manifest.

``pcols`` and ``pver`` are compile-time parameters of CAM (``-DPCOLS``, ``-DPLEV``):
the image has no symbol to ask.  The build records them in the manifest's
``dimensions``; a manifest written before it did carries them only inside its
compile commands, where every command must agree.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .errors import NativeCAMError

#: the manifest's dimension names and the CAM macros they are compiled from
GRID_MACROS = {"pcols": "PCOLS", "pver": "PLEV", "pcnst": "PCNST", "psubcols": "PSUBCOLS"}


def grid_from_commands(commands: Mapping[str, Any] | Iterable[Any]) -> dict[str, int]:
    """The grid every compile command defines; refused when two commands define it differently."""

    items = commands.values() if isinstance(commands, Mapping) else commands
    grid: dict[str, int] = {}
    for command in items:
        text = command if isinstance(command, str) else " ".join(str(part) for part in command)
        for name, macro in GRID_MACROS.items():
            for value in re.findall(rf"-D{macro}=(\d+)", text):
                if grid.setdefault(name, int(value)) != int(value):
                    raise NativeCAMError(f"the image's compile commands disagree on {macro}: "
                                         f"{grid[name]} and {value}")
    return grid


def image_grid(manifest: Mapping[str, Any]) -> dict[str, int]:
    """The image's grid: the manifest's ``dimensions``, else what its compile commands define."""

    recorded = manifest.get("dimensions")
    if isinstance(recorded, Mapping) and recorded:
        return {str(name): int(value) for name, value in recorded.items()}
    return grid_from_commands(manifest.get("compile_commands") or {})


def check_config_grid(config: Any, grid: Mapping[str, int]) -> None:
    """A configuration runs only on an image compiled for its columns and levels."""

    for name in ("pcols", "pver"):
        if name in grid and int(getattr(config, name)) != int(grid[name]):
            raise NativeCAMError(
                f"the native image was compiled with {name}={grid[name]} ({GRID_MACROS[name]}); "
                f"the configuration says {name}={getattr(config, name)}")


__all__ = ["GRID_MACROS", "check_config_grid", "grid_from_commands", "image_grid"]
