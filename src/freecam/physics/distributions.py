"""How to draw a function's inputs and parameters.

Each drawn argument and each drawn parameter gets a distribution.  Plain ones
(``Uniform``, ``Normal``, ``LogUniform``, ``Constant``) draw every element
independently over a declared range -- the right tool for boundary and
sensitivity studies -- and ``Choice`` does the same for a knob whose spec
declares a fixed set of values rather than a range.  Real atmospheres are not boxes, though:
``HybridPressure`` draws a whole pressure profile from one surface pressure
through the hybrid coordinate so midpoints, thicknesses and interfaces never
contradict each other; ``Derived`` computes one input from others already
drawn; ``Anchored`` perturbs a real column within bounds, and
``CapturedColumns`` draws the column itself from a capture so a sample is a
state the model actually visited.  A
``SamplingSpace`` ties distributions to a function, takes every undrawn
input from a base column, resolves the dependency order, and draws one
complete joint sample at a time from a single seeded generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .errors import PhysicsError
from .spec import FunctionSpec

_PRESSURE_ROLES = ("pressure_midpoint", "pressure_thickness", "pressure_interface")


class Distribution:
    """Draws one argument; ``depends`` names the arguments it needs first."""

    depends: tuple[str, ...] = ()
    produces: tuple[str, ...] | None = None

    def sample(self, rng: np.random.Generator, shape: tuple[int, ...], drawn: Mapping[str, np.ndarray]) -> Any:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return self.describe()


@dataclass(frozen=True, repr=False)
class Constant(Distribution):
    value: Any

    def sample(self, rng, shape, drawn):
        return np.broadcast_to(np.asarray(self.value, dtype=np.float64), shape).copy()

    def describe(self) -> str:
        return f"Constant({np.asarray(self.value).tolist()!r})" if np.ndim(self.value) == 0 else "Constant(array)"


@dataclass(frozen=True, repr=False)
class Uniform(Distribution):
    low: Any
    high: Any

    def sample(self, rng, shape, drawn):
        low = np.broadcast_to(np.asarray(self.low, dtype=np.float64), shape)
        high = np.broadcast_to(np.asarray(self.high, dtype=np.float64), shape)
        return low + rng.uniform(0.0, 1.0, shape) * (high - low)

    def describe(self) -> str:
        return f"Uniform({_fmt(self.low)}, {_fmt(self.high)})"


@dataclass(frozen=True, repr=False)
class Choice(Distribution):
    """One of a fixed set of values, each equally likely.

    A categorical knob -- a scheme selector whose spec declares ``values``
    instead of a ``range`` -- has no continuous draw to round: rounding
    distorts the frequencies at both ends of the set and can land outside
    it.  This draws from the set itself.
    """

    values: Any

    def __post_init__(self) -> None:
        if np.asarray(self.values).size == 0:
            raise PhysicsError("Choice needs at least one value")

    def sample(self, rng, shape, drawn):
        table = np.asarray(self.values, dtype=np.float64).reshape(-1)
        return table[rng.integers(0, table.size, shape)]

    def describe(self) -> str:
        return f"Choice({np.asarray(self.values).tolist()!r})"


@dataclass(frozen=True, repr=False)
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
        return f"LogUniform({_fmt(self.low)}, {_fmt(self.high)})"


@dataclass(frozen=True, repr=False)
class Normal(Distribution):
    mean: Any
    std: Any
    clip: tuple[Any, Any] | None = None

    def sample(self, rng, shape, drawn):
        mean = np.broadcast_to(np.asarray(self.mean, dtype=np.float64), shape)
        std = np.broadcast_to(np.asarray(self.std, dtype=np.float64), shape)
        value = mean + rng.standard_normal(shape) * std
        return _clip(value, self.clip)

    def describe(self) -> str:
        return f"Normal({_fmt(self.mean)}, {_fmt(self.std)}{_fmt_clip(self.clip)})"


@dataclass(frozen=True, repr=False)
class Anchored(Distribution):
    """A real column plus bounded noise.

    ``anchor * (1 + N(0, relative_scale)) + N(0, absolute_scale)``: the
    relative term scales the noise with the anchor's own magnitude, as mixing
    ratios spanning orders of magnitude need; the absolute term keeps a level
    whose anchor is exactly zero -- a clear layer -- from staying frozen at
    zero, so the space can reach a little cloud there too.  ``clip`` keeps
    the draw physical.
    """

    anchor: Any
    relative_scale: float = 0.0
    absolute_scale: float = 0.0
    clip: tuple[Any, Any] | None = None

    def __post_init__(self) -> None:
        if self.relative_scale < 0 or self.absolute_scale < 0:
            raise PhysicsError("Anchored scales must be non-negative")
        if self.relative_scale == 0 and self.absolute_scale == 0:
            raise PhysicsError("Anchored needs a relative_scale or an absolute_scale")

    def sample(self, rng, shape, drawn):
        anchor = np.broadcast_to(np.asarray(self.anchor, dtype=np.float64), shape)
        value = anchor.copy()
        if self.relative_scale:
            value = value * (1.0 + rng.standard_normal(shape) * self.relative_scale)
        if self.absolute_scale:
            value = value + rng.standard_normal(shape) * self.absolute_scale
        return _clip(value, self.clip)

    def describe(self) -> str:
        parts = []
        if self.relative_scale:
            parts.append(f"relative_scale={self.relative_scale!r}")
        if self.absolute_scale:
            parts.append(f"absolute_scale={self.absolute_scale!r}")
        return f"Anchored({', '.join(parts)}{_fmt_clip(self.clip)})"


@dataclass(frozen=True, repr=False, eq=False)
class CapturedColumns(Distribution):
    """Real columns from a capture, drawn whole and perturbed.

    ``Anchored`` perturbs one column, so every sample is that column's shape
    scaled up and down: the model's own cloud water needs twenty-four
    principal components over thirty levels, a single anchor supplies one, and
    no retuning of the noise supplies the rest.  This draws the anchor itself,
    one captured column per sample, and answers with every argument taken from
    **that same column** -- a sample stays one coherent atmospheric state
    rather than six unrelated ones spliced together.

    Perturbation is what keeps the space from being the capture replayed.  It
    is deliberately two-sided:

    * ``relative_scale`` multiplies each level by ``1 + N(0, scale)``, which
      leaves an exact zero exactly zero -- the model's clear layers stay
      clear, and the fraction of levels holding no condensate is the model's
      own rather than an artefact of the sampler;
    * ``absolute_scale`` adds ``N(0, scale)``, which is the only way a level
      the model left empty can take on a value.  ``absolute_probability``
      gates it per argument: unlisted arguments are ungated (temperature wants
      half a degree at every level), while condensate is listed at a small
      rate so a clear layer takes on cloud at a stated frequency instead of
      everywhere.  A rate may be one number or one per level, which is how a
      level the model never clouds -- the top of the column holds no liquid
      water at all -- is left alone rather than spending the budget there.  This is the off-manifold budget -- a surrogate run inside
      the model is pushed off the manifold by its own error, and a training
      set with no such states has nothing to say when it happens.

    ``columns`` maps an argument name to its captured values, leading axis the
    column.  ``produces`` names the arguments answered together.
    """

    columns: Mapping[str, np.ndarray]
    produces: tuple[str, ...] = ()
    relative_scale: Mapping[str, float] = field(default_factory=dict)
    absolute_scale: Mapping[str, float] = field(default_factory=dict)
    absolute_probability: Mapping[str, float] = field(default_factory=dict)
    clip: Mapping[str, tuple[Any, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.produces:
            raise PhysicsError("CapturedColumns needs the arguments it produces")
        missing = [name for name in self.produces if name not in self.columns]
        if missing:
            raise PhysicsError("CapturedColumns has no captured values for: " + ", ".join(missing))
        lengths = {int(np.asarray(self.columns[name]).shape[0]) for name in self.produces}
        if len(lengths) != 1:
            raise PhysicsError(f"captured arguments disagree on how many columns they hold: {sorted(lengths)}")
        for name, rate in self.absolute_probability.items():
            values = np.asarray(rate, dtype=np.float64)
            if values.size and (values.min() < 0.0 or values.max() > 1.0):
                raise PhysicsError(f"{name}: absolute_probability is a probability, got {rate!r}")
        for name, scale in list(self.relative_scale.items()) + list(self.absolute_scale.items()):
            if scale < 0:
                raise PhysicsError(f"{name}: perturbation scales must be non-negative")

    @property
    def length(self) -> int:
        return int(np.asarray(self.columns[self.produces[0]]).shape[0])

    def sample(self, rng, shape, drawn):
        index = int(rng.integers(0, self.length))
        answer: dict[str, np.ndarray] = {}
        for name in self.produces:
            value = np.array(self.columns[name][index], dtype=np.float64)
            relative = float(self.relative_scale.get(name, 0.0))
            if relative:
                value = value * (1.0 + rng.standard_normal(value.shape) * relative)
            absolute = float(self.absolute_scale.get(name, 0.0))
            if absolute:
                noise = rng.standard_normal(value.shape) * absolute
                rate = np.asarray(self.absolute_probability.get(name, 1.0), dtype=np.float64)
                if rate.ndim or rate < 1.0:
                    noise = noise * (rng.random(value.shape) < rate)
                value = value + noise
            answer[name] = _clip(value, self.clip.get(name))
        return answer

    def describe(self) -> str:
        perturbed = sum(1 for name in self.produces
                        if self.relative_scale.get(name) or self.absolute_scale.get(name))
        def rate_of(value):
            value = np.asarray(value, dtype=np.float64)
            return f"{float(value):g}" if not value.ndim else \
                f"{float(value.max()):g} on {int(np.count_nonzero(value))}/{value.size} levels"

        gated = ", ".join(f"{name}@{rate_of(rate)}"
                          for name, rate in sorted(self.absolute_probability.items()))
        return (f"CapturedColumns({self.length} columns, {len(self.produces)} arguments, "
                f"{perturbed} perturbed" + (f", gated {gated}" if gated else "") + ")")


@dataclass(frozen=True, repr=False)
class Derived(Distribution):
    """One argument computed from others: ``fn(rng, **drawn_values)``."""

    fn: Callable[..., Any]
    depends: tuple[str, ...] = ()

    def sample(self, rng, shape, drawn):
        value = self.fn(rng, **{name: drawn[name] for name in self.depends})
        return np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()

    def describe(self) -> str:
        return f"Derived({getattr(self.fn, '__name__', 'fn')}, depends={list(self.depends)})"


@dataclass(frozen=True, repr=False)
class HybridCoordinate:
    """CAM's hybrid sigma-pressure coefficients: ``pint = hyai*p0 + hybi*ps``."""

    hyai: np.ndarray
    hybi: np.ndarray
    p0: float

    def interfaces(self, surface_pressure: float) -> np.ndarray:
        return np.asarray(self.hyai, dtype=np.float64) * self.p0 + np.asarray(self.hybi, dtype=np.float64) * float(surface_pressure)

    @classmethod
    def from_history(cls, path: str | Path) -> "HybridCoordinate":
        """Coefficients from any CAM history or initial file carrying hyai/hybi/P0."""

        from netCDF4 import Dataset

        with Dataset(str(path)) as handle:
            return cls(
                np.asarray(handle.variables["hyai"][...], dtype=np.float64),
                np.asarray(handle.variables["hybi"][...], dtype=np.float64),
                float(np.asarray(handle.variables["P0"][...])),
            )

    def __repr__(self) -> str:
        return f"HybridCoordinate(levels={len(self.hyai) - 1}, p0={self.p0!r})"


@dataclass(frozen=True, repr=False)
class HybridPressure(Distribution):
    """The whole pressure profile from one drawn surface pressure.

    Drawing ``ps`` once and deriving interfaces, midpoints and thicknesses
    keeps every sampled column structurally consistent, which independent
    per-level draws cannot.  Which arguments receive which profile comes from
    the function spec (``pressure_midpoint`` / ``pressure_thickness`` /
    ``pressure_interface`` constraints) unless ``produces`` names them.
    """

    coordinate: HybridCoordinate
    surface_pressure: Distribution
    produces: tuple[str, ...] | None = None

    @classmethod
    def from_column(cls, column: Any, surface_pressure: Distribution | None = None) -> "HybridPressure":
        """Use an example column's own coordinate; fix its surface pressure unless told otherwise."""

        coordinate = getattr(column, "hybrid", None)
        if coordinate is None:
            raise PhysicsError("the column carries no hybrid coordinate; use HybridPressure(coordinate, ...)")
        if surface_pressure is None:
            surface_pressure = Constant(float(column.surface_pressure))
        return cls(coordinate, surface_pressure)

    def sample(self, rng, shape, drawn):
        ps = float(np.asarray(self.surface_pressure.sample(rng, (), drawn)))
        interfaces = self.coordinate.interfaces(ps)
        assert self.produces is not None, "SamplingSpace resolves the produced names"
        midpoint, thickness, interface = self.produces
        values = {"surface_pressure": np.asarray(ps)}
        if midpoint:
            values[midpoint] = 0.5 * (interfaces[:-1] + interfaces[1:])
        if thickness:
            values[thickness] = np.diff(interfaces)
        if interface:
            values[interface] = interfaces
        return values

    def describe(self) -> str:
        return f"HybridPressure(surface_pressure={self.surface_pressure.describe()})"


class SamplingSpace:
    """Distributions for a function's inputs and parameters, drawn jointly.

    ``inputs`` and ``parameters`` map names to distributions; ``base`` is a
    column whose values are used for every input that is not drawn, so the
    space always produces a complete call.  Inputs a distribution produces
    (the three pressure profiles) are never also taken from the base.
    """

    def __init__(
        self,
        spec: FunctionSpec,
        *,
        inputs: Mapping[str, Distribution] | None = None,
        parameters: Mapping[str, Distribution] | None = None,
        base: Mapping[str, Any] | None = None,
        fixed_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.spec = spec
        self.distributions: dict[str, Distribution] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.produced: dict[str, str] = {}
        for name, distribution in (inputs or {}).items():
            item = self._argument(name)
            if not item.user_visible:
                raise PhysicsError(f"{item.name} is {item.role}; it cannot be sampled")
            distribution = self._resolve_produces(item.name, distribution)
            self.distributions[item.name] = distribution
            self.shapes[item.name] = item.public_extent(spec.dimensions)
        for name, distribution in (parameters or {}).items():
            if name not in spec.parameters:
                raise PhysicsError(f"{spec.function} has no parameter {name!r}")
            if name in self.distributions:
                raise PhysicsError(f"{name} is both an input and a parameter distribution")
            self.distributions[name] = distribution
            self.shapes[name] = ()
        self.fixed_parameters = dict(fixed_parameters or {})
        self.base: dict[str, np.ndarray] = {}
        for name, value in (base or {}).items():
            try:
                item = spec.argument(name)
            except KeyError:
                continue  # an example column may carry attributes that are not inputs
            if item.user_visible and item.name not in self.distributions and item.name not in self.produced:
                self.base[item.name] = np.asarray(value)
        self.order = self._order()

    # -- construction ---------------------------------------------------------

    def _argument(self, name: str):
        try:
            return self.spec.argument(name)
        except KeyError as error:
            raise PhysicsError(f"{self.spec.function} has no argument {name!r}") from error

    def _resolve_produces(self, key: str, distribution: Distribution) -> Distribution:
        """Register a distribution that answers with several arguments at once.

        ``HybridPressure`` names its arguments by their declared roles when it
        was built without them; every other multi-argument distribution states
        ``produces`` itself.
        """

        if isinstance(distribution, HybridPressure) and distribution.produces is None:
            by_role = {}
            for item in self.spec.user_arguments:
                for role in _PRESSURE_ROLES:
                    if role in item.constraints:
                        by_role[role] = item.name
            produces = tuple(by_role.get(role) for role in _PRESSURE_ROLES)
            if not any(produces):
                raise PhysicsError(f"{self.spec.function} declares no pressure arguments for HybridPressure")
            distribution = HybridPressure(distribution.coordinate, distribution.surface_pressure, produces)
        produces = distribution.produces
        if produces is None:
            return distribution
        for name in produces:
            if not name:
                continue
            item = self._argument(name)
            if not item.user_visible:
                raise PhysicsError(f"{item.name} is {item.role}; it cannot be sampled")
            self.produced[name] = key
        if key not in produces:
            raise PhysicsError(f"the distribution under {key!r} produces {[n for n in produces if n]}, not {key!r}")
        return distribution

    def _order(self) -> list[str]:
        remaining = dict(self.distributions)
        done: set[str] = set(self.base)
        order: list[str] = []
        while remaining:
            progressed = False
            for key in list(remaining):
                needs = set()
                for dependency in remaining[key].depends:
                    if dependency in self.distributions or dependency in self.base:
                        needs.add(dependency)
                    elif dependency in self.produced:
                        needs.add(self.produced[dependency])
                    else:
                        raise PhysicsError(f"dependency {dependency!r} of {key!r} is neither drawn nor in the base column")
                if needs <= done:
                    order.append(key)
                    done.add(key)
                    done.update(name for name, producer in self.produced.items() if producer == key)
                    del remaining[key]
                    progressed = True
            if not progressed:
                raise PhysicsError("sampling dependencies form a cycle: " + ", ".join(remaining))
        return order

    # -- drawing -----------------------------------------------------------------

    @property
    def drawn_inputs(self) -> tuple[str, ...]:
        return tuple(name for name in self.distributions if name not in self.spec.parameters)

    @property
    def drawn_parameters(self) -> tuple[str, ...]:
        return tuple(name for name in self.distributions if name in self.spec.parameters)

    def draw(self, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """One joint sample: (inputs, parameters), complete for a call."""

        drawn: dict[str, Any] = dict(self.base)
        for key in self.order:
            value = self.distributions[key].sample(rng, self.shapes[key], drawn)
            if isinstance(value, Mapping):
                drawn.update({name: np.asarray(item) for name, item in value.items()})
            else:
                drawn[key] = np.asarray(value)
        inputs: dict[str, np.ndarray] = {}
        parameters: dict[str, Any] = dict(self.fixed_parameters)
        for name, value in drawn.items():
            if name in self.spec.parameters:
                parameter = self.spec.parameters[name]
                parameters[name] = int(np.round(value)) if parameter.dtype in ("int32", "int64") else float(value)
            else:
                try:
                    item = self.spec.argument(name)
                except KeyError:
                    continue
                if item.user_visible:
                    inputs[item.name] = value
        return inputs, parameters

    def describe(self) -> str:
        lines = [f"SamplingSpace for {self.spec.function}: {len(self.drawn_inputs)} drawn inputs, {len(self.drawn_parameters)} drawn parameters, {len(self.base)} inputs from the base column"]
        for key in self.order:
            kind = "parameter" if key in self.spec.parameters else "input"
            lines.append(f"  {key:<24s} {kind:<10s} {self.distributions[key].describe()}")
        if self.fixed_parameters:
            lines.append("  fixed parameters: " + ", ".join(f"{k}={v!r}" for k, v in self.fixed_parameters.items()))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.describe()


def _clip(value: np.ndarray, clip: tuple[Any, Any] | None) -> np.ndarray:
    if clip is None:
        return value
    low, high = clip
    return np.clip(value, -np.inf if low is None else low, np.inf if high is None else high)


def _fmt(value: Any) -> str:
    array = np.asarray(value)
    return f"{float(array):.6g}" if array.ndim == 0 else "array"


def _fmt_clip(clip: tuple[Any, Any] | None) -> str:
    return "" if clip is None else f", clip={clip!r}"


__all__ = [
    "Anchored", "Choice", "Constant", "Derived", "Distribution", "HybridCoordinate", "HybridPressure",
    "LogUniform", "Normal", "SamplingSpace", "Uniform",
]
