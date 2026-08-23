"""One public column in, one native chunk out, and back again.

The user sees a routine's arguments at column granularity: a level profile is
``(pver,)``, a surface value is a scalar.  The routine sees ``(pcols, pver)``
chunks with ``ncol`` active lanes.  This module is the only place that knows
the packing: lane 0 receives the column, ``ncol`` is 1, structural arguments
take their declared values, outputs and workspace are zeroed, and after the
call lane 0 is read back.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .errors import PhysicsError
from .spec import ArgumentSpec, FunctionSpec


class InvalidInput(PhysicsError):
    """The supplied inputs do not fit the function's declared boundary."""


def _canonical(spec: FunctionSpec, name: str) -> ArgumentSpec:
    try:
        return spec.argument(name)
    except KeyError as error:
        raise InvalidInput(f"{spec.function} has no argument {name!r}") from error


def coerce_inputs(spec: FunctionSpec, inputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Resolve names, check roles, shapes, dtypes, and finiteness."""

    resolved: dict[str, np.ndarray] = {}
    for name, value in inputs.items():
        item = _canonical(spec, name)
        if not item.user_visible:
            raise InvalidInput(f"{item.name} is {item.role}; it is not a user input")
        array = np.asarray(value, dtype=np.dtype(item.dtype))
        expected = item.public_extent(spec.dimensions)
        if array.shape != expected:
            raise InvalidInput(
                f"{item.name} has shape {array.shape}, expected {expected} "
                f"({'scalar' if not expected else '[' + ', '.join(spec.public_axis(a) for a in item.public_shape) + ']'})"
            )
        if array.dtype.kind == "f" and not np.all(np.isfinite(array)):
            raise InvalidInput(f"{item.name} contains non-finite values")
        resolved[item.name] = array
    missing = [
        item.name for item in spec.user_arguments
        if item.name not in resolved and item.default is None
    ]
    if missing:
        raise InvalidInput("missing inputs without defaults: " + ", ".join(missing))
    return resolved


def empty_pool(spec: FunctionSpec, nchunks: int = 1) -> dict[str, np.ndarray]:
    pool: dict[str, np.ndarray] = {}
    for item in spec.arguments:
        shape = (*item.native_extent(spec.dimensions), nchunks)
        pool[f"{spec.function}.{item.name}"] = np.zeros(shape, dtype=np.dtype(item.dtype), order="F")
    return pool


def pack_column(spec: FunctionSpec, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """A one-chunk pool with the column in lane 0 and ``ncol = 1``."""

    pool = empty_pool(spec, 1)
    for item in spec.arguments:
        target = pool[f"{spec.function}.{item.name}"]
        if item.role == "structural":
            target[...] = item.value
            continue
        if item.role in ("output", "workspace"):
            continue
        value = inputs.get(item.name)
        if value is None:
            value = item.default
        if item.rank == 0:
            target[0] = value
        else:
            target[0, ..., 0] = value
    return pool


def unpack_column(spec: FunctionSpec, pool: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Lane 0 of every returned argument: (outputs, updated in/outs)."""

    outputs: dict[str, np.ndarray] = {}
    updated: dict[str, np.ndarray] = {}
    for item in spec.arguments:
        if not item.returned:
            continue
        array = pool[f"{spec.function}.{item.name}"]
        value = array[0].copy() if item.rank == 0 else np.array(array[0, ..., 0], copy=True)
        (updated if item.role == "inout" else outputs)[item.name] = value
    return outputs, updated


__all__ = ["InvalidInput", "coerce_inputs", "empty_pool", "pack_column", "unpack_column"]
