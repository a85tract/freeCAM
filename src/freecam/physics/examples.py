"""Reviewed example columns shipped with the package.

A function is easiest to explore from a real column.  Each example is one
column captured from a real CAM call -- its inputs exactly as the model
passed them, plus the surface pressure and the hybrid coordinate of the run
it came from -- stored as package data so a notebook never reaches into the
repository.
"""

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .distributions import HybridCoordinate
from .errors import PhysicsError


class ExampleColumn(dict):
    """One column's inputs (a mapping) with where it came from.

    Behaves as a plain ``{name: array}`` mapping of the function's inputs, so
    it can be passed to ``run`` or merged with ``{**column, ...}``; the extra
    attributes describe the column rather than feed the routine.
    """

    def __init__(self, inputs: dict[str, np.ndarray], *, function: str, name: str, surface_pressure: float,
                 hybrid: HybridCoordinate | None, source: dict[str, Any]) -> None:
        super().__init__(inputs)
        self.function = function
        self.name = name
        self.surface_pressure = float(surface_pressure)
        self.hybrid = hybrid
        self.source = dict(source)

    def __repr__(self) -> str:
        return (f"ExampleColumn({self.name!r} for {self.function}: {len(self)} inputs, "
                f"surface_pressure={self.surface_pressure:.1f} Pa, {self.source.get('description', 'captured from a model run')})")


def available_examples(function: str) -> tuple[str, ...]:
    package = resources.files("freecam.physics.data") / function
    if not package.is_dir():
        return ()
    return tuple(sorted(path.name[: -len(".json")] for path in package.iterdir() if path.name.endswith(".json")))


def load_example_column(function: str, name: str = "captured-anchor") -> ExampleColumn:
    """A shipped example column for ``function``."""

    resource = resources.files("freecam.physics.data") / function / f"{name}.json"
    if not resource.is_file():
        raise PhysicsError(
            f"no example {name!r} for {function!r}; available: {list(available_examples(function))}"
        )
    with resource.open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    hybrid = None
    if record.get("hybrid_coordinate"):
        coordinate = record["hybrid_coordinate"]
        hybrid = HybridCoordinate(np.asarray(coordinate["hyai"]), np.asarray(coordinate["hybi"]), float(coordinate["p0"]))
    return ExampleColumn(
        {key: np.asarray(value, dtype=np.float64) for key, value in record["inputs"].items()},
        function=str(record["function"]),
        name=name,
        surface_pressure=float(record["surface_pressure"]),
        hybrid=hybrid,
        source=dict(record.get("source", {})),
    )


__all__ = ["ExampleColumn", "available_examples", "load_example_column"]
