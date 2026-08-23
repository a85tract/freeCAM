"""Reviewed single-column function specifications.

A physics routine becomes a plain function ``y = f(x, p)`` only once its
boundary is written down: which dummy arguments are user inputs, which are
updated in place, which are outputs the runtime allocates, which are
structural (chunk bookkeeping the user never sees), and which module-level
state and tunables the routine reads besides its arguments.  That boundary
lives in one reviewed YAML file per function under
``native/pi_cam/functions``; this module loads it and fails closed on anything
inconsistent.  The YAML is the runtime's only authority -- the kernel
inventory and the Fortran source are consulted at build time by
``tools/verify_pi_cam_function_spec.py``, never here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from .errors import PhysicsSpecError

ROLES = ("structural", "input", "inout", "output", "workspace")
USER_ROLES = ("input", "inout")
INTENTS = ("in", "out", "inout")
DTYPES = ("float64", "int32", "int64")
MODULE_STATE_WRITES = ("verify", "declared", "snapshot", "initializer", "parameter")
STUB_CLASSES = ("inert", "fail_closed", "abort")
_NATIVE_ONLY_AXES = ("pcols",)


def default_functions_dir() -> Path:
    """The checked-in function specifications, relative to the project root."""

    return Path(__file__).resolve().parents[3] / "native" / "pi_cam" / "functions"


@dataclass(frozen=True, slots=True)
class ArgumentSpec:
    """One dummy argument of the original routine, in Fortran order."""

    name: str
    role: str
    fortran_type: str
    dtype: str
    rank: int
    intent: str
    native_shape: tuple[str, ...]
    public_shape: tuple[str, ...] | None = None
    units: str | None = None
    range: tuple[float, float] | tuple[int, int] | None = None
    constraints: tuple[str, ...] = ()
    default: Any = None
    value: Any = None
    description: str = ""
    pointer: bool = False
    carrier: str | None = None

    @property
    def user_visible(self) -> bool:
        return self.role in USER_ROLES

    @property
    def returned(self) -> bool:
        return self.role in ("inout", "output")

    def native_extent(self, dimensions: Mapping[str, int]) -> tuple[int, ...]:
        return tuple(int(dimensions[axis]) for axis in self.native_shape)

    def public_extent(self, dimensions: Mapping[str, int]) -> tuple[int, ...]:
        if self.public_shape is None:
            return ()
        return tuple(int(dimensions[axis]) for axis in self.public_shape)


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One module-level tunable, named as in the namelist."""

    name: str
    dtype: str
    symbols: tuple[str, ...]
    default: Any
    range: tuple[float, float] | tuple[int, int] | None = None
    values: tuple[Any, ...] | None = None
    units: str | None = None
    constraints: tuple[str, ...] = ()
    description: str = ""


@dataclass(frozen=True, slots=True)
class ModuleStateEntry:
    """One Fortran module variable the routine reads besides its arguments."""

    symbol: str
    dtype: str
    shape: tuple[int, ...]
    write: str
    value: Any = None
    expected: Any = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class ImageSpec:
    """How the function's standalone image is linked."""

    archive_members: tuple[str, ...]
    stubs: Mapping[str, tuple[str, ...]]
    base_address: int

    @property
    def stub_symbols(self) -> frozenset[str]:
        return frozenset(symbol for group in self.stubs.values() for symbol in group)


@dataclass(frozen=True, slots=True)
class FunctionSpec:
    """The complete reviewed boundary of one routine."""

    function: str
    qualified_name: str
    routine: str
    source: str
    module: str | None
    dimensions: Mapping[str, int]
    public_axes: Mapping[str, str]
    arguments: tuple[ArgumentSpec, ...]
    parameters: Mapping[str, ParameterSpec]
    module_state: tuple[ModuleStateEntry, ...]
    initializers: tuple[str, ...]
    image: ImageSpec
    path: Path | None = None
    schema_version: int = 1

    def __iter__(self) -> Iterator[ArgumentSpec]:
        return iter(self.arguments)

    def argument(self, name: str) -> ArgumentSpec:
        key = name.lower()
        for item in self.arguments:
            if item.name.lower() == key:
                return item
        raise KeyError(f"{self.function} has no argument {name!r}")

    def _by_role(self, *roles: str) -> tuple[ArgumentSpec, ...]:
        return tuple(item for item in self.arguments if item.role in roles)

    @property
    def inputs(self) -> tuple[ArgumentSpec, ...]:
        return self._by_role("input")

    @property
    def inouts(self) -> tuple[ArgumentSpec, ...]:
        return self._by_role("inout")

    @property
    def outputs(self) -> tuple[ArgumentSpec, ...]:
        return self._by_role("output")

    @property
    def workspace(self) -> tuple[ArgumentSpec, ...]:
        return self._by_role("workspace")

    @property
    def structural(self) -> tuple[ArgumentSpec, ...]:
        return self._by_role("structural")

    @property
    def user_arguments(self) -> tuple[ArgumentSpec, ...]:
        return self._by_role(*USER_ROLES)

    def public_axis(self, axis: str) -> str:
        return str(self.public_axes.get(axis, axis))

    def describe(self) -> str:
        """A readable signature: inputs, inouts, outputs, parameters."""

        def shape(item: ArgumentSpec) -> str:
            names = item.public_shape if item.public_shape is not None else ()
            return "[" + ", ".join(self.public_axis(axis) for axis in names) + "]" if names else "scalar"

        lines = [f"{self.qualified_name}  ({self.source})", ""]
        for title, items in (
            ("Inputs", self.inputs),
            ("In/out (initial value supplied, updated on return)", self.inouts),
            ("Outputs", self.outputs),
        ):
            lines.append(f"{title}:")
            for item in items:
                unit = f"  {item.units}" if item.units else ""
                default = f"  default={item.default!r}" if item.default is not None else ""
                lines.append(f"  {item.name:<14s} {shape(item):<12s}{unit}{default}")
            lines.append("")
        lines.append("Parameters:")
        for item in self.parameters.values():
            span = (
                f"  values={list(item.values)}"
                if item.values is not None
                else (f"  range={list(item.range)}" if item.range else "")
            )
            lines.append(f"  {item.name:<24s} default={item.default!r}{span}")
        return "\n".join(lines)


def _tuple_of_str(values: Any, where: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise PhysicsSpecError(f"{where} must be a list")
    return tuple(str(item) for item in values)


def _range(values: Any, where: str) -> tuple[float, float] | tuple[int, int] | None:
    if values is None:
        return None
    if not isinstance(values, Sequence) or len(values) != 2:
        raise PhysicsSpecError(f"{where} range must be [min, max]")
    if all(isinstance(item, int) and not isinstance(item, bool) for item in values):
        low, high = int(values[0]), int(values[1])
    else:
        low, high = float(values[0]), float(values[1])
    if not low <= high:
        raise PhysicsSpecError(f"{where} range is inverted: {values}")
    return (low, high)


def _argument(entry: Mapping[str, Any], dimensions: Mapping[str, int]) -> ArgumentSpec:
    name = str(entry.get("name", "")).strip()
    if not name:
        raise PhysicsSpecError("every argument needs a name")
    where = f"argument {name!r}"
    role = str(entry.get("role", ""))
    if role not in ROLES:
        raise PhysicsSpecError(f"{where} has unsupported role {role!r}")
    dtype = str(entry.get("dtype", ""))
    if dtype not in DTYPES:
        raise PhysicsSpecError(f"{where} has unsupported dtype {dtype!r}")
    intent = str(entry.get("intent", ""))
    if intent not in INTENTS:
        raise PhysicsSpecError(f"{where} has unsupported intent {intent!r}")
    rank = int(entry.get("rank", -1))
    native_shape = _tuple_of_str(entry.get("native_shape"), f"{where} native_shape")
    if len(native_shape) != rank:
        raise PhysicsSpecError(
            f"{where} declares rank {rank} but native_shape {list(native_shape)}"
        )
    for axis in native_shape:
        if axis not in dimensions:
            raise PhysicsSpecError(f"{where} uses unknown dimension {axis!r}")
    public_shape: tuple[str, ...] | None = None
    if "public_shape" in entry:
        public_shape = _tuple_of_str(entry["public_shape"], f"{where} public_shape")
        expected = tuple(axis for axis in native_shape if axis not in _NATIVE_ONLY_AXES)
        if public_shape != expected:
            raise PhysicsSpecError(
                f"{where} public_shape {list(public_shape)} must be native_shape "
                f"without the column axis, {list(expected)}"
            )
    if role == "structural":
        if "value" not in entry:
            raise PhysicsSpecError(f"{where} is structural and needs a value")
        if public_shape is not None:
            raise PhysicsSpecError(f"{where} is structural and has no public shape")
    elif public_shape is None:
        raise PhysicsSpecError(f"{where} needs a public_shape")
    role_intents = {
        "input": ("in",),
        "inout": ("inout",),
        "output": ("out",),
        "workspace": ("in", "inout"),
        "structural": ("in",),
    }
    if intent not in role_intents[role]:
        raise PhysicsSpecError(f"{where} role {role!r} does not admit intent {intent!r}")
    return ArgumentSpec(
        name=name,
        role=role,
        fortran_type=str(entry.get("fortran_type", "")),
        dtype=dtype,
        rank=rank,
        intent=intent,
        native_shape=native_shape,
        public_shape=public_shape,
        units=None if entry.get("units") is None else str(entry["units"]),
        range=_range(entry.get("range"), where),
        constraints=_tuple_of_str(entry.get("constraints"), f"{where} constraints"),
        default=entry.get("default"),
        value=entry.get("value"),
        description=str(entry.get("description", "")).strip(),
        pointer=bool(entry.get("pointer", False)),
        carrier=None if entry.get("carrier") is None else str(entry["carrier"]),
    )


def _parameter(name: str, entry: Mapping[str, Any]) -> ParameterSpec:
    where = f"parameter {name!r}"
    dtype = str(entry.get("dtype", ""))
    if dtype not in DTYPES:
        raise PhysicsSpecError(f"{where} has unsupported dtype {dtype!r}")
    symbols = _tuple_of_str(entry.get("symbols"), f"{where} symbols")
    if not symbols:
        raise PhysicsSpecError(f"{where} names no symbols")
    if "default" not in entry:
        raise PhysicsSpecError(f"{where} needs a default")
    values = entry.get("values")
    if values is not None:
        values = tuple(values)
    span = _range(entry.get("range"), where)
    if (values is None) == (span is None):
        raise PhysicsSpecError(f"{where} needs exactly one of range or values")
    return ParameterSpec(
        name=name,
        dtype=dtype,
        symbols=symbols,
        default=entry["default"],
        range=span,
        values=values,
        units=None if entry.get("units") is None else str(entry["units"]),
        constraints=_tuple_of_str(entry.get("constraints"), f"{where} constraints"),
        description=str(entry.get("description", "")).strip(),
    )


def _module_state(entry: Mapping[str, Any]) -> ModuleStateEntry:
    symbol = str(entry.get("symbol", "")).strip()
    if not symbol:
        raise PhysicsSpecError("every module_state entry needs a symbol")
    where = f"module state {symbol!r}"
    write = str(entry.get("write", ""))
    if write not in MODULE_STATE_WRITES:
        raise PhysicsSpecError(f"{where} has unsupported write mode {write!r}")
    if write == "declared" and "value" not in entry:
        raise PhysicsSpecError(f"{where} is declared but has no value")
    if write != "declared" and "value" in entry:
        raise PhysicsSpecError(f"{where} carries a value but is not declared")
    if write == "declared" and "expected" in entry:
        raise PhysicsSpecError(f"{where} is declared; state the value, not an expectation")
    shape = entry.get("shape", [])
    if not isinstance(shape, Sequence):
        raise PhysicsSpecError(f"{where} shape must be a list")
    return ModuleStateEntry(
        symbol=symbol,
        dtype=str(entry.get("dtype", "")),
        shape=tuple(int(item) for item in shape),
        write=write,
        value=entry.get("value"),
        expected=entry.get("expected"),
        note=str(entry.get("note", "")).strip(),
    )


def _image(entry: Mapping[str, Any]) -> ImageSpec:
    members = _tuple_of_str(entry.get("archive_members"), "image.archive_members")
    if not members:
        raise PhysicsSpecError("image.archive_members is empty")
    raw = entry.get("stubs") or {}
    unknown = sorted(set(raw) - set(STUB_CLASSES))
    if unknown:
        raise PhysicsSpecError("image.stubs has unsupported classes: " + ", ".join(unknown))
    stubs = {name: _tuple_of_str(raw.get(name), f"image.stubs.{name}") for name in STUB_CLASSES}
    seen: set[str] = set()
    for group in stubs.values():
        for symbol in group:
            if symbol in seen:
                raise PhysicsSpecError(f"stub {symbol!r} appears in more than one class")
            seen.add(symbol)
    base = entry.get("base_address")
    if base is None:
        raise PhysicsSpecError("image.base_address is required")
    return ImageSpec(archive_members=members, stubs=stubs, base_address=int(base))


def parse_function_spec(document: Mapping[str, Any], *, path: Path | None = None) -> FunctionSpec:
    """Validate one loaded YAML document and build the spec, failing closed."""

    if not isinstance(document, Mapping) or int(document.get("schema_version", 0)) != 1:
        raise PhysicsSpecError("function spec must declare schema_version: 1")
    required = ("function", "qualified_name", "routine", "source", "dimensions", "arguments", "image")
    missing = [key for key in required if key not in document]
    if missing:
        raise PhysicsSpecError("function spec is missing: " + ", ".join(missing))
    dimensions = {str(key): int(value) for key, value in dict(document["dimensions"]).items()}
    arguments = tuple(_argument(entry, dimensions) for entry in document["arguments"])
    if not arguments:
        raise PhysicsSpecError("function spec declares no arguments")
    names = [item.name.lower() for item in arguments]
    if len(set(names)) != len(names):
        raise PhysicsSpecError("argument names repeat")
    parameters = {
        str(name): _parameter(str(name), entry)
        for name, entry in dict(document.get("parameters") or {}).items()
    }
    module_state = tuple(_module_state(entry) for entry in document.get("module_state") or ())
    symbols = [item.symbol for item in module_state]
    if len(set(symbols)) != len(symbols):
        raise PhysicsSpecError("module_state symbols repeat")
    by_symbol = {item.symbol: item for item in module_state}
    for parameter in parameters.values():
        for symbol in parameter.symbols:
            entry = by_symbol.get(symbol)
            if entry is None or entry.write != "parameter":
                raise PhysicsSpecError(
                    f"parameter {parameter.name!r} symbol {symbol!r} is not a "
                    "module_state entry with write: parameter"
                )
            if entry.dtype != parameter.dtype:
                raise PhysicsSpecError(
                    f"parameter {parameter.name!r} dtype {parameter.dtype} differs "
                    f"from module_state {symbol!r} dtype {entry.dtype}"
                )
    owned = {symbol for parameter in parameters.values() for symbol in parameter.symbols}
    orphans = [item.symbol for item in module_state if item.write == "parameter" and item.symbol not in owned]
    if orphans:
        raise PhysicsSpecError("module_state parameter symbols without a parameter: " + ", ".join(orphans))
    image = _image(document["image"])
    return FunctionSpec(
        function=str(document["function"]),
        qualified_name=str(document["qualified_name"]),
        routine=str(document["routine"]),
        source=str(document["source"]),
        module=None if document.get("module") is None else str(document["module"]),
        dimensions=dimensions,
        public_axes={str(key): str(value) for key, value in dict(document.get("public_axes") or {}).items()},
        arguments=arguments,
        parameters=parameters,
        module_state=module_state,
        initializers=_tuple_of_str(document.get("initializers"), "initializers"),
        image=image,
        path=path,
    )


def load_function_spec(name_or_path: str | Path, *, functions_dir: Path | None = None) -> FunctionSpec:
    """Load ``<name>.yaml`` from the functions directory, or an explicit path."""

    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.is_file():
        path = candidate
    else:
        path = (functions_dir or default_functions_dir()) / f"{name_or_path}.yaml"
    if not path.is_file():
        raise PhysicsSpecError(f"function spec not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = parse_function_spec(document, path=path)
    if spec.function != path.stem:
        raise PhysicsSpecError(f"{path} declares function {spec.function!r}, expected {path.stem!r}")
    return spec


__all__ = [
    "ArgumentSpec",
    "FunctionSpec",
    "ImageSpec",
    "ModuleStateEntry",
    "ParameterSpec",
    "default_functions_dir",
    "load_function_spec",
    "parse_function_spec",
]
