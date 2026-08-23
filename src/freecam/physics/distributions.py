"""How to draw a function's inputs and parameters.

Each user-visible argument and each parameter gets a distribution.  Plain
ones (``Uniform``, ``Normal``, ``LogUniform``, ``Constant``) draw every
element independently over a declared range -- the supervisor's request,
and the right tool for boundary and sensitivity studies.  Real atmospheres
are not boxes, though: ``HybridPressure`` draws a whole pressure profile
from one surface pressure through the hybrid coordinate so ``p`` and ``dp``
never contradict each other; ``Derived`` computes one input from others
already drawn (humidity from relative humidity, temperature and pressure);
``Anchored`` perturbs a real captured column within bounds.  A
``SamplingSpace`` resolves the dependency order and draws one complete
joint sample at a time from a single seeded generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .errors import PhysicsError
from .spec import FunctionSpec


class Distribution:
    """Draws one argument; ``depends`` names the arguments it needs first."""

    depends: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()

    def sample(self, rng: np.random.Generator, shape: tuple[int, ...], drawn: Mapping[str, np.ndarray]) -> Any:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class Constant(Distribution):
    value: Any

    def sample(self, rng, shape, drawn):
        return np.broadcast_to(np.asarray(self.value, dtype=np.float64), shape).copy()

    def describe(self) -> str:
        return f"Constant({self.value!r})"


@dataclass(frozen=True)
class Uniform(Distribution):
    low: Any
    high: Any

    def sample(self, rng, shape, drawn):
        low = np.broadcast_to(np.asarray(self.low, dtype=np.float64), shape)
        high = np.broadcast_to(np.asarray(self.high, dtype=np.float64), shape)
        return low + rng.uniform(0.0, 1.0, shape) * (high - low)

    def describe(self) -> str:
        return f"Uniform({self.low!r}, {self.high!r})"


@dataclass(frozen=True)
class LogUniform(Distribution):
    low: Any
    high: Any

    def __post_init__(self) -> None:
        if np.any(np.asarray(self.low) <= 0) or np.any(np.asarray(self.high) <= 0):
            raise PhysicsError("LogUniform bounds must be positive")

    def sample(self, rng, shape, drawn):
        low = np.log(np.broadcast_to(np.asarray(self.low, dtype=np.float64), shape))
        high = np.log(np.broadcast_to(np.asarray(self.high, dtype=np.float64), shape))
        return np.exp(low + rng.uniform(0.0, 1.0, shape) * (high - low))

    def describe(self) -> str:
        return f"LogUniform({self.low!r}, {self.high!r})"


@dataclass(frozen=True)
class Normal(Distribution):
    mean: Any
    std: Any
    clip: tuple[Any, Any] | None = None

    def sample(self, rng, shape, drawn):
        mean = np.broadcast_to(np.asarray(self.mean, dtype=np.float64), shape)
        std = np.broadcast_to(np.asarray(self.std, dtype=np.float64), shape)
        value = mean + rng.standard_normal(shape) * std
        if self.clip is not None:
            value = np.clip(value, self.clip[0], self.clip[1])
        return value

    def describe(self) -> str:
        return f"Normal({self.mean!r}, {self.std!r}{', clip=' + repr(self.clip) if self.clip else ''})"


@dataclass(frozen=True)
class Anchored(Distribution):
    """A real column plus bounded noise: ``anchor + Normal(0, scale)``.

    With ``relative`` the noise scales with the anchor's own magnitude, which
    is what mixing ratios spanning orders of magnitude need.  ``clip`` keeps
    the draw physical (a species cannot go negative).
    """

    anchor: Any
    scale: float
    relative: bool = False
    clip: tuple[Any, Any] | None = None

    def sample(self, rng, shape, drawn):
        anchor = np.broadcast_to(np.asarray(self.anchor, dtype=np.float64), shape)
        noise = rng.standard_normal(shape) * self.scale
        value = anchor * (1.0 + noise) if self.relative else anchor + noise
        if self.clip is not None:
            value = np.clip(value, self.clip[0], self.clip[1])
        return value

    def describe(self) -> str:
        return f"Anchored(scale={self.scale!r}{', relative' if self.relative else ''})"


@dataclass(frozen=True)
class Derived(Distribution):
    """One argument computed from others: ``fn(rng, **drawn_values)``."""

    fn: Callable[..., Any]
    depends: tuple[str, ...] = ()

    def sample(self, rng, shape, drawn):
        value = self.fn(rng, **{name: drawn[name] for name in self.depends})
        return np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()

    def describe(self) -> str:
        return f"Derived({getattr(self.fn, '__name__', 'fn')}, depends={list(self.depends)})"


@dataclass(frozen=True)
class HybridPressure(Distribution):
    """The whole pressure profile from one surface pressure.

    CAM's interfaces are ``hyai*p0 + hybi*ps``; midpoints are their means and
    thicknesses their differences.  Drawing ``ps`` once and deriving the rest
    keeps every sampled column structurally consistent -- no thickness ever
    contradicts its interfaces -- which independent per-level draws cannot.
    """

    hyai: Any
    hybi: Any
    p0: float
    surface: Distribution
    # The argument names receiving (midpoint, thickness, interface) pressure.
    produces: tuple[str, ...] = ("p", "dp", "pint")

    @classmethod
    def from_history(cls, path: str | Path, surface: Distribution, produces: tuple[str, ...] = ("p", "dp", "pint")) -> "HybridPressure":
        """Coefficients from any CAM history/initial file carrying hyai/hybi/P0."""

        from netCDF4 import Dataset

        with Dataset(str(path)) as handle:
            return cls(
                np.asarray(handle.variables["hyai"][...], dtype=np.float64),
                np.asarray(handle.variables["hybi"][...], dtype=np.float64),
                float(np.asarray(handle.variables["P0"][...])),
                surface,
                produces,
            )

    def sample(self, rng, shape, drawn):
        ps = float(np.asarray(self.surface.sample(rng, (), drawn)))
        interfaces = np.asarray(self.hyai) * self.p0 + np.asarray(self.hybi) * ps
        midpoint, thickness, interface = self.produces
        return {
            midpoint: 0.5 * (interfaces[:-1] + interfaces[1:]),
            thickness: np.diff(interfaces),
            interface: interfaces,
            "ps": np.asarray(ps),
        }

    def describe(self) -> str:
        return f"HybridPressure(surface={self.surface.describe()})"


class SamplingSpace:
    """Distributions for a function's arguments and parameters, drawn jointly."""

    def __init__(self, spec: FunctionSpec, distributions: Mapping[str, Distribution]) -> None:
        self.spec = spec
        self.distributions: dict[str, Distribution] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.kinds: dict[str, str] = {}
        for name, distribution in distributions.items():
            key = self._resolve(name)
            self.distributions[key] = distribution
        self.order = self._order()

    def _resolve(self, name: str) -> str:
        if name in self.spec.parameters:
            self.kinds[name] = "parameter"
            self.shapes[name] = ()
            return name
        try:
            item = self.spec.argument(name)
        except KeyError as error:
            raise PhysicsError(f"{self.spec.function} has no argument or parameter {name!r}") from error
        if not item.user_visible:
            raise PhysicsError(f"{item.name} is {item.role}; it cannot be sampled")
        self.kinds[item.name] = "input"
        self.shapes[item.name] = item.public_extent(self.spec.dimensions)
        return item.name

    def _order(self) -> list[str]:
        produced: dict[str, str] = {}
        for key, distribution in self.distributions.items():
            for name in distribution.produces or ():
                if name != key and (name in self.spec.parameters or self._is_argument(name)):
                    produced[name] = key
        remaining = dict(self.distributions)
        done: set[str] = set()
        order: list[str] = []
        while remaining:
            progressed = False
            for key in list(remaining):
                needs = {self._resolve_dependency(dep, produced) for dep in remaining[key].depends}
                if needs <= done:
                    order.append(key)
                    done.add(key)
                    done.update(name for name in remaining[key].produces if name in produced or name == key)
                    del remaining[key]
                    progressed = True
            if not progressed:
                raise PhysicsError("sampling dependencies form a cycle or reference nothing drawn: " + ", ".join(remaining))
        return order

    def _is_argument(self, name: str) -> bool:
        try:
            self.spec.argument(name)
        except KeyError:
            return False
        return True

    def _resolve_dependency(self, name: str, produced: Mapping[str, str]) -> str:
        if name in self.distributions:
            return name
        if name in produced:
            return name
        for key in self.distributions:
            if key.lower() == name.lower():
                return key
        raise PhysicsError(f"dependency {name!r} is not drawn by any distribution")

    def draw(self, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """One joint sample: (inputs, parameters)."""

        drawn: dict[str, Any] = {}
        for key in self.order:
            distribution = self.distributions[key]
            value = distribution.sample(rng, self.shapes.get(key, ()), drawn)
            if isinstance(value, Mapping):
                for name, item in value.items():
                    drawn[name] = np.asarray(item)
            else:
                drawn[key] = np.asarray(value)
        inputs: dict[str, np.ndarray] = {}
        parameters: dict[str, Any] = {}
        for name, value in drawn.items():
            if name in self.spec.parameters:
                parameter = self.spec.parameters[name]
                parameters[name] = int(np.round(value)) if parameter.dtype in ("int32", "int64") else float(value)
            elif self._is_argument(name) and self.spec.argument(name).user_visible:
                inputs[self.spec.argument(name).name] = value
        return inputs, parameters

    def describe(self) -> str:
        return "\n".join(f"  {key:<24s} {self.distributions[key].describe()}" for key in self.order)


__all__ = [
    "Anchored", "Constant", "Derived", "Distribution", "HybridPressure",
    "LogUniform", "Normal", "SamplingSpace", "Uniform",
]
