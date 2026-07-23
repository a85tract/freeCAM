"""Generate standalone C ABI devices around unmodified CCPP Fortran schemes."""

from __future__ import annotations

from dataclasses import dataclass, replace
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from functools import lru_cache
from typing import Any, Iterable, Mapping

import yaml

if __package__:
    from .errors import DeviceBuildError
else:  # Support clean build environments without importing pycam_sima.
    class DeviceBuildError(RuntimeError):
        """A source scheme cannot be converted into a standalone device."""


DEVICE_DESCRIPTION_SCHEMA_VERSION = 1
DEVICE_ABI_VERSION = 1
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PROCESS_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_USE_STATEMENT = re.compile(
    r"^\s*use\b\s*(?:,\s*(?:non_)?intrinsic\s*)?"
    r"(?:::\s*)?([A-Za-z][A-Za-z0-9_]*)\b",
    re.IGNORECASE,
)
_INTERNAL_STANDARD_NAMES = {
    "ccpp_error_code",
    "ccpp_error_message",
    "scheme_name",
}
_INTRINSIC_MODULES = {
    "iso_c_binding",
    "iso_fortran_env",
    "ieee_arithmetic",
    "ieee_exceptions",
    "ieee_features",
}
_FORBIDDEN_MODULE_PATTERNS = (
    re.compile(r"^mpi(?:_|$)", re.IGNORECASE),
    re.compile(r"^esmf(?:_|$)", re.IGNORECASE),
    re.compile(r"^pio(?:_|$)", re.IGNORECASE),
    re.compile(r"^cam_history(?:_|$)", re.IGNORECASE),
    re.compile(r"^ccpp_io_reader$", re.IGNORECASE),
)
_FORBIDDEN_ELF_DEPENDENCIES = (
    "mpi",
    "pmi",
    "pals",
    "esmf",
    "pio",
    "netcdf",
    "hdf5",
    "libsci",
)
_PHYS_CONST_BINDINGS: Mapping[str, tuple[str, str]] = {
    "epsilo": ("water_to_dry_molecular_weight_ratio", "1"),
    "latvap": ("latent_heat_of_vaporization", "J kg-1"),
    "latice": ("latent_heat_of_fusion", "J kg-1"),
    "rh2o": ("water_vapor_gas_constant", "J kg-1 K-1"),
    "cpair": ("dry_air_specific_heat", "J kg-1 K-1"),
    "tmelt": ("water_freezing_temperature", "K"),
    "h2otrip": ("water_triple_point_temperature", "K"),
    "rearth": ("earth_radius", "m"),
    "r_universal": ("universal_gas_constant", "J K-1 kmol-1"),
    "avogad": ("avogadro_constant", "molecule kmol-1"),
    "boltz": ("boltzmann_constant", "J K-1 molecule-1"),
    "pi": ("circle_constant", "1"),
    "gravit": ("gravitational_acceleration", "m s-2"),
    "rair": ("dry_air_gas_constant", "J kg-1 K-1"),
    "rga": ("reciprocal_gravitational_acceleration", "s2 m-1"),
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeviceBuildError(f"{label} must be a mapping")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeviceBuildError(f"{label} must be a non-empty string")
    return value.strip()


def _resolve_path(project_root: Path, value: Any, label: str) -> Path:
    text = _require_string(value, label)
    path = Path(text)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise DeviceBuildError(f"{label} does not exist: {path}")
    return path


def _safe_name(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise DeviceBuildError(
            f"{label} must be a Fortran-compatible identifier, got {value!r}"
        )
    return value.lower()


def _safe_process_name(value: str, label: str) -> str:
    if not _PROCESS_IDENTIFIER.fullmatch(value):
        raise DeviceBuildError(
            f"{label} must be a process identifier, got {value!r}"
        )
    return value.lower()


def _fortran_identifier(value: str) -> str:
    """Shorten generated local names to Fortran's 63-character limit."""

    if len(value) <= 63:
        return value
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"{value[:50]}_{digest}"


@dataclass(frozen=True, slots=True)
class EntrypointDescription:
    name: str
    table: str
    bindings: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class DeviceDescription:
    path: Path
    project_root: Path
    name: str
    module: str
    sources: tuple[Path, ...]
    metadata: tuple[Path, ...]
    providers: Mapping[str, Path]
    source_modules: frozenset[str]
    state_policy: str
    initialize_entrypoint: str | None
    dimension_bindings: Mapping[str, str]
    global_bindings: Mapping[str, Mapping[str, Any]]
    entrypoints: tuple[EntrypointDescription, ...]
    processes: Mapping[str, str]
    auto_dependencies: bool

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, project_root: str | Path
    ) -> "DeviceDescription":
        descriptor_path = Path(path).resolve()
        root = Path(project_root).resolve()
        payload = yaml.safe_load(descriptor_path.read_text())
        data = _require_mapping(payload, str(descriptor_path))
        if data.get("schema_version") != DEVICE_DESCRIPTION_SCHEMA_VERSION:
            raise DeviceBuildError(
                f"{descriptor_path}: unsupported schema_version "
                f"{data.get('schema_version')!r}"
            )
        name = _safe_name(_require_string(data.get("name"), "name"), "name")
        module = _safe_name(
            _require_string(data.get("fortran_module"), "fortran_module"),
            "fortran_module",
        )

        source_values = data.get("sources")
        metadata_values = data.get("metadata")
        if not isinstance(source_values, list) or not source_values:
            raise DeviceBuildError("sources must be a non-empty list")
        if not isinstance(metadata_values, list) or not metadata_values:
            raise DeviceBuildError("metadata must be a non-empty list")
        sources = tuple(
            _resolve_path(root, value, f"sources[{index}]")
            for index, value in enumerate(source_values)
        )
        metadata = tuple(
            _resolve_path(root, value, f"metadata[{index}]")
            for index, value in enumerate(metadata_values)
        )

        providers_data = _require_mapping(
            data.get("providers", {}), "providers"
        )
        providers = {
            _safe_name(str(module_name), "provider module"): _resolve_path(
                root, source, f"provider {module_name}"
            )
            for module_name, source in providers_data.items()
        }
        source_modules_data = data.get("source_modules", [module])
        if not isinstance(source_modules_data, list) or not source_modules_data:
            raise DeviceBuildError("source_modules must be a non-empty list")
        source_modules = frozenset(
            _safe_name(str(value), "source module")
            for value in source_modules_data
        )
        if module not in source_modules:
            raise DeviceBuildError(
                f"fortran_module {module!r} is missing from source_modules"
            )

        state_policy = _require_string(
            data.get("state_policy", "stateless"), "state_policy"
        )
        if state_policy not in {
            "stateless",
            "reinitialize_each_run",
            "initialize_once",
        }:
            raise DeviceBuildError(
                f"unsupported state_policy {state_policy!r}"
            )
        initialize_entrypoint = data.get("initialize_entrypoint")
        if initialize_entrypoint is not None:
            initialize_entrypoint = _safe_name(
                _require_string(
                    initialize_entrypoint, "initialize_entrypoint"
                ),
                "initialize_entrypoint",
            )

        dimension_data = _require_mapping(
            data.get("dimension_bindings", {}), "dimension_bindings"
        )
        dimension_bindings = {
            _require_string(key, "dimension standard name").lower(): (
                _require_string(value, f"dimension binding {key}")
            )
            for key, value in dimension_data.items()
        }
        global_bindings = _parse_bindings(data.get("bindings", {}), "bindings")

        entrypoint_data = _require_mapping(
            data.get("entrypoints"), "entrypoints"
        )
        entrypoints: list[EntrypointDescription] = []
        for raw_name, raw_value in entrypoint_data.items():
            entry_name = _safe_name(str(raw_name), "entrypoint name")
            entry = _require_mapping(
                raw_value, f"entrypoints.{entry_name}"
            )
            table = _safe_name(
                _require_string(
                    entry.get("table"), f"entrypoints.{entry_name}.table"
                ),
                f"entrypoints.{entry_name}.table",
            )
            bindings = _parse_bindings(
                entry.get("bindings", {}),
                f"entrypoints.{entry_name}.bindings",
            )
            entrypoints.append(
                EntrypointDescription(entry_name, table, bindings)
            )
        entrypoint_names = {item.name for item in entrypoints}
        if initialize_entrypoint not in entrypoint_names | {None}:
            raise DeviceBuildError(
                f"initialize_entrypoint {initialize_entrypoint!r} is unknown"
            )

        process_data = _require_mapping(data.get("processes"), "processes")
        processes: dict[str, str] = {}
        for raw_process, raw_entrypoint in process_data.items():
            process = _safe_process_name(str(raw_process), "process name")
            entrypoint = _safe_name(
                _require_string(
                    raw_entrypoint, f"processes.{process}"
                ),
                f"processes.{process}",
            )
            if entrypoint not in entrypoint_names:
                raise DeviceBuildError(
                    f"process {process!r} references unknown entrypoint "
                    f"{entrypoint!r}"
                )
            processes[process] = entrypoint
        if not processes:
            raise DeviceBuildError("processes must not be empty")
        auto_dependencies = data.get("auto_dependencies", False)
        if not isinstance(auto_dependencies, bool):
            raise DeviceBuildError("auto_dependencies must be bool")

        return cls(
            path=descriptor_path,
            project_root=root,
            name=name,
            module=module,
            sources=sources,
            metadata=metadata,
            providers=providers,
            source_modules=source_modules,
            state_policy=state_policy,
            initialize_entrypoint=initialize_entrypoint,
            dimension_bindings=dimension_bindings,
            global_bindings=global_bindings,
            entrypoints=tuple(entrypoints),
            processes=processes,
            auto_dependencies=auto_dependencies,
        )


def _parse_bindings(
    value: Any, label: str
) -> Mapping[str, Mapping[str, Any]]:
    data = _require_mapping(value, label)
    parsed: dict[str, Mapping[str, Any]] = {}
    for raw_name, raw_binding in data.items():
        name = _require_string(raw_name, f"{label} key").lower()
        if isinstance(raw_binding, str):
            binding: Mapping[str, Any] = {
                "source": "field",
                "name": raw_binding,
            }
        else:
            binding = dict(
                _require_mapping(raw_binding, f"{label}.{name}")
            )
        source = binding.get("source")
        if source not in {"field", "standard_name", "dimension", "literal"}:
            raise DeviceBuildError(
                f"{label}.{name}.source must be field, standard_name, "
                "dimension, or literal"
            )
        if source == "literal":
            if "value" not in binding:
                raise DeviceBuildError(
                    f"{label}.{name} literal binding requires value"
                )
        else:
            _require_string(binding.get("name"), f"{label}.{name}.name")
        parsed[name] = binding
    return parsed


@dataclass(frozen=True, slots=True)
class MetadataArgument:
    local_name: str
    standard_name: str
    fortran_type: str
    kind: str
    dimensions: tuple[str, ...]
    intent: str
    units: str
    optional: bool
    allocatable: bool

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    @property
    def caller_owned_allocatable(self) -> bool:
        """Whether Python can satisfy CCPP allocation with a shaped array.

        CCPP metadata may mark an array allocatable even when the scheme's
        Fortran dummy argument is an assumed-shape array (for example
        ``rrtmgp_lw_rte_run/lw_Ds``).  A primitive, explicitly dimensioned
        field can remain Python-owned and cross ABI v1 as the usual array.
        Derived objects and deferred-shape scalars still require a dedicated
        ownership adapter.
        """

        return (
            self.allocatable
            and self.rank > 0
            and self.dtype in {
                "float64",
                "int32",
                "bool",
                "character",
            }
        )

    @property
    def dtype(self) -> str:
        if self.fortran_type == "real" and self.kind.lower() in {
            "kind_phys",
            "real64",
            "c_double",
        }:
            return "float64"
        if self.fortran_type == "integer" and self.kind in {"", "c_int"}:
            return "int32"
        if self.fortran_type == "logical" and self.kind.lower() in {
            "",
            "c_bool",
        }:
            return "bool"
        if self.fortran_type == "character":
            return "character"
        return "opaque"


@dataclass(frozen=True, slots=True)
class MetadataEntrypoint:
    table: str
    module: str
    arguments: tuple[MetadataArgument, ...]


def _load_ccpp_entrypoints(
    description: DeviceDescription,
) -> Mapping[str, MetadataEntrypoint]:
    scripts = (
        description.project_root
        / "external"
        / "CAM-SIMA"
        / "ccpp_framework"
        / "scripts"
    )
    if not scripts.is_dir():
        raise DeviceBuildError(
            f"CCPP Framework parser is missing: {scripts}"
        )
    scripts_text = str(scripts)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    try:
        from ccpp_capgen import parse_scheme_files
        from framework_env import CCPPFrameworkEnv
        from parse_tools import init_log, set_log_to_null
    except ImportError as exc:  # pragma: no cover - environment corruption.
        raise DeviceBuildError(
            f"cannot import CCPP Framework parser from {scripts}"
        ) from exc

    logger = init_log(f"pycam-device-{description.name}")
    set_log_to_null(logger)
    run_env = CCPPFrameworkEnv(
        logger,
        host_files="",
        scheme_files="",
        suites="",
        kind_types=["kind_phys=REAL64"],
        output_root=str(
            description.project_root / "build" / ".ccpp-device-parse"
        ),
    )
    try:
        headers, _ = parse_scheme_files(
            [str(path) for path in description.metadata],
            run_env,
            skip_ddt_check=True,
        )
    except Exception as exc:
        raise DeviceBuildError(
            f"CCPP metadata/source verification failed for "
            f"{description.name}: {exc}"
        ) from exc

    result: dict[str, MetadataEntrypoint] = {}
    for header in headers:
        if header.module.lower() not in description.source_modules:
            continue
        arguments: list[MetadataArgument] = []
        for variable in header.variable_list():
            optional = variable.get_prop_value("optional")
            if isinstance(optional, str):
                optional = optional.lower() in {".true.", "true", "t"}
            arguments.append(
                MetadataArgument(
                    local_name=variable.get_prop_value("local_name").lower(),
                    standard_name=variable.get_prop_value(
                        "standard_name"
                    ).lower(),
                    fortran_type=variable.get_prop_value("type").lower(),
                    kind=(variable.get_prop_value("kind") or "").lower(),
                    dimensions=tuple(
                        str(item).lower()
                        for item in variable.get_prop_value("dimensions")
                    ),
                    intent=variable.get_prop_value("intent").lower(),
                    units=variable.get_prop_value("units"),
                    optional=bool(optional),
                    allocatable=bool(
                        str(
                            variable.get_prop_value("allocatable") or ""
                        ).lower()
                        in {"true", ".true.", "t", "1", "yes"}
                    ),
                )
            )
        table = header.title.lower()
        result[table] = MetadataEntrypoint(
            table=table,
            module=header.module.lower(),
            arguments=tuple(arguments),
        )
    return result


def _logical_fortran_lines(source: Path) -> Iterable[str]:
    pending = ""
    for raw in source.read_text().splitlines():
        code = raw.split("!", 1)[0].strip()
        if not code:
            continue
        if pending:
            code = code.lstrip("&").strip()
            pending = f"{pending} {code}"
        else:
            pending = code
        if pending.endswith("&"):
            pending = pending[:-1].rstrip()
            continue
        yield pending
        pending = ""
    if pending:
        yield pending


def _validate_dependencies(description: DeviceDescription) -> tuple[str, ...]:
    scripts = (
        description.project_root
        / "external"
        / "CAM-SIMA"
        / "ccpp_framework"
        / "scripts"
    )
    scripts_text = str(scripts)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)
    try:
        from fortran_tools.parse_fortran import UseStatement
    except ImportError as exc:  # pragma: no cover
        raise DeviceBuildError("cannot import CCPP UseStatement parser") from exc

    dependencies: set[str] = set()
    allowed = (
        set(description.providers)
        | set(description.source_modules)
        | _INTRINSIC_MODULES
    )
    for source in description.sources:
        for line in _logical_fortran_lines(source):
            match = _USE_STATEMENT.match(line)
            if match is None:
                continue
            statement = UseStatement(line)
            if statement.valid:
                module = statement.module.lower()
            else:
                module = match.group(1).lower()
            for pattern in _FORBIDDEN_MODULE_PATTERNS:
                if pattern.search(module):
                    raise DeviceBuildError(
                        f"{source}: host/framework dependency {module!r} "
                        "cannot be packaged as a numerical device"
                    )
            dependencies.add(module)
            if module not in allowed:
                raise DeviceBuildError(
                    f"{source}: unresolved Fortran module {module!r}; add a "
                    "portable provider/source module or keep this scheme out "
                    "of the standalone device boundary"
                )
    return tuple(sorted(dependencies))


@lru_cache(maxsize=4)
def _project_module_index(project_root: Path) -> Mapping[str, Path]:
    """Index module providers once for recursive, source-only packaging."""

    candidates: list[Path] = []
    cam_root = project_root / "external/CAM-SIMA"
    for pattern in ("*.F90", "*.f90", "*.F", "*.f"):
        candidates.extend(cam_root.rglob(pattern))
    result: dict[str, Path] = {}
    for source in sorted(set(candidates)):
        try:
            for line in _logical_fortran_lines(source):
                match = re.match(
                    r"^\s*module\s+(?!procedure\b|subroutine\b|function\b)"
                    r"([A-Za-z][A-Za-z0-9_]*)",
                    line,
                    flags=re.IGNORECASE,
                )
                if match:
                    result.setdefault(
                        match.group(1).lower(), source.resolve()
                    )
        except UnicodeDecodeError:
            continue
    return result


@lru_cache(maxsize=4)
def _project_type_index(project_root: Path) -> Mapping[str, str]:
    """Map public/native derived-type names to their defining modules."""

    result: dict[str, str] = {}
    for module, source in _project_module_index(project_root).items():
        try:
            for line in _logical_fortran_lines(source):
                match = re.match(
                    r"^\s*type\s*(?:,\s*[^:]*)?::\s*"
                    r"([A-Za-z][A-Za-z0-9_]*)\b",
                    line,
                    flags=re.IGNORECASE,
                )
                if match:
                    result.setdefault(match.group(1).lower(), module)
        except UnicodeDecodeError:
            continue
    return result


def _source_type_index(source: Path) -> Mapping[str, str]:
    """Return derived types declared in one source and their host module."""

    result: dict[str, str] = {}
    current_module: str | None = None
    for line in _logical_fortran_lines(source):
        module_match = re.match(
            r"^\s*module\s+(?!procedure\b|subroutine\b|function\b)"
            r"([A-Za-z][A-Za-z0-9_]*)",
            line,
            flags=re.IGNORECASE,
        )
        if module_match:
            current_module = module_match.group(1).lower()
            continue
        if re.match(r"^\s*end\s+module\b", line, flags=re.IGNORECASE):
            current_module = None
            continue
        type_match = re.match(
            r"^\s*type\s*(?:,\s*[^:]*)?::\s*"
            r"([A-Za-z][A-Za-z0-9_]*)\b",
            line,
            flags=re.IGNORECASE,
        )
        if type_match and current_module is not None:
            result[type_match.group(1).lower()] = current_module
    return result


def _defined_modules(source: Path) -> frozenset[str]:
    result: set[str] = set()
    for line in _logical_fortran_lines(source):
        match = re.match(
            r"^\s*module\s+(?!procedure\b|subroutine\b|function\b)"
            r"([A-Za-z][A-Za-z0-9_]*)",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            result.add(match.group(1).lower())
    return frozenset(result)


def _source_uses(source: Path) -> frozenset[str]:
    return frozenset(
        match.group(1).lower()
        for line in _logical_fortran_lines(source)
        if (match := _USE_STATEMENT.match(line)) is not None
    )


def _module_only_names(sources: Iterable[Path], module: str) -> tuple[str, ...]:
    """Collect local names imported from ``use module, only: ...``."""

    result: set[str] = set()
    pattern = re.compile(
        rf"^\s*use\s+{re.escape(module)}\s*,\s*only\s*:\s*(.*)$",
        flags=re.IGNORECASE,
    )
    for source in sources:
        for line in _logical_fortran_lines(source):
            match = pattern.match(line)
            if not match:
                continue
            for token in match.group(1).split(","):
                local = token.strip().split("=>", 1)[0].strip().lower()
                if _IDENTIFIER.fullmatch(local):
                    result.add(local)
    return tuple(sorted(result))


def resolve_source_closure(
    description: DeviceDescription,
) -> DeviceDescription:
    """Recursively package module sources without linking the CAM runtime."""

    if not description.auto_dependencies:
        return description
    index = dict(_project_module_index(description.project_root))
    providers = set(description.providers)
    initial = tuple(path.resolve() for path in description.sources)
    for source in initial:
        for module in _defined_modules(source):
            index[module] = source

    visiting: set[Path] = set()
    visited: set[Path] = set()
    ordered: list[Path] = []
    unresolved: dict[str, Path] = {}
    forbidden: dict[str, Path] = {}

    def visit(source: Path) -> None:
        source = source.resolve()
        if source in visited:
            return
        if source in visiting:
            return
        visiting.add(source)
        local_modules = _defined_modules(source)
        for module in sorted(_source_uses(source)):
            if (
                module in _INTRINSIC_MODULES
                or module in providers
                or module in local_modules
            ):
                continue
            if any(
                pattern.search(module)
                for pattern in _FORBIDDEN_MODULE_PATTERNS
            ):
                forbidden[module] = source
                continue
            dependency = index.get(module)
            if dependency is None:
                unresolved[module] = source
                continue
            visit(dependency)
        # RTE/RRTMGP ships an interface-only module under api/ and a
        # source-equivalent CPU implementation one directory above.  A
        # module-use graph sees only the interface, so include its paired
        # implementation explicitly to satisfy the bind(C) symbols without
        # linking an accelerator or host framework runtime.
        if source.parent.name == "api":
            implementation = source.parent.parent / source.name
            if implementation.is_file():
                visit(implementation)
        visiting.remove(source)
        visited.add(source)
        ordered.append(source)

    for source in initial:
        visit(source)
    if forbidden or unresolved:
        details = [
            *(
                f"host module {module!r} used by {source}"
                for module, source in sorted(forbidden.items())
            ),
            *(
                f"unresolved module {module!r} used by {source}"
                for module, source in sorted(unresolved.items())
            ),
        ]
        raise DeviceBuildError(
            f"device {description.name!r} recursive source closure is not "
            f"standalone: {'; '.join(details)}"
        )
    source_modules = frozenset(
        module
        for source in ordered
        for module in _defined_modules(source)
    )
    return replace(
        description,
        sources=tuple(ordered),
        source_modules=source_modules,
    )


def _dimension_standard(expression: str) -> str:
    tokens = [token.strip().lower() for token in expression.split(":")]
    upper = tokens[-1]
    if upper.isdigit():
        return upper
    if not _IDENTIFIER.fullmatch(upper):
        raise DeviceBuildError(
            f"unsupported CCPP dimension expression {expression!r}"
        )
    return upper


def _binding_for(
    description: DeviceDescription,
    entrypoint: EntrypointDescription,
    argument: MetadataArgument,
) -> Mapping[str, Any]:
    explicit = dict(description.global_bindings)
    explicit.update(entrypoint.bindings)
    if argument.standard_name in explicit:
        return dict(explicit[argument.standard_name])
    if argument.standard_name in description.dimension_bindings:
        return {
            "source": "dimension",
            "name": description.dimension_bindings[argument.standard_name],
        }
    return {
        "source": "standard_name",
        "name": argument.standard_name,
    }


def _manifest_argument(
    argument: MetadataArgument,
    binding: Mapping[str, Any],
    *,
    injected: bool = False,
    opaque: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "abi_name": argument.local_name,
        "standard_name": argument.standard_name,
        "fortran_type": argument.fortran_type,
        "kind": argument.kind,
        "dtype": argument.dtype,
        "rank": argument.rank,
        "dimensions": [
            _dimension_standard(item) for item in argument.dimensions
        ],
        "intent": argument.intent,
        "units": argument.units,
        "passing": (
            "value"
            if argument.rank == 0 and argument.intent == "in"
            else "reference"
        ),
        "binding": dict(binding),
        "injected": injected,
        "metadata_allocatable": argument.allocatable,
        "allocation_owner": (
            "python"
            if argument.caller_owned_allocatable
            else "fortran"
            if argument.dtype == "opaque"
            else None
        ),
    }
    if argument.dtype == "opaque":
        if opaque is None:
            raise DeviceBuildError(
                f"{argument.local_name}: opaque ABI contract is missing"
            )
        result["opaque"] = dict(opaque)
    if argument.fortran_type == "character":
        match = re.fullmatch(r"len\s*=\s*(\d+)", argument.kind)
        result["character_length"] = (
            int(match.group(1)) if match else 512
        )
    return result


def _fortran_declaration(
    argument: MetadataArgument,
    dimension_symbols: Mapping[str, str],
    *,
    local_name: str | None = None,
) -> str:
    if argument.dtype == "float64":
        declaration = "real(c_double)"
    elif argument.dtype == "int32":
        declaration = "integer(c_int)"
    elif argument.dtype == "bool":
        declaration = "logical(c_bool)"
    elif argument.dtype == "character":
        abi_name = local_name or argument.local_name
        length_name = f"{abi_name}_length"
        resolved = [
            dimension_symbols.get(
                _dimension_standard(item), _dimension_standard(item)
            )
            for item in argument.dimensions
        ]
        shape = ",".join([length_name, *resolved])
        return (
            f"    integer(c_int), value, intent(in) :: {length_name}\n"
            f"    character(kind=c_char), intent({argument.intent}) :: "
            f"{abi_name}({shape})"
        )
    elif argument.dtype == "opaque":
        return (
            f"    type(c_ptr), value, intent(in) :: "
            f"{local_name or argument.local_name}"
        )
    else:
        raise DeviceBuildError(
            f"cannot expose {argument.local_name} as a C ABI argument"
        )
    attributes = [f"intent({argument.intent})"]
    if argument.rank == 0 and argument.intent == "in":
        attributes.insert(0, "value")
    dimensions = ""
    if argument.dimensions:
        resolved: list[str] = []
        for item in argument.dimensions:
            standard = _dimension_standard(item)
            resolved.append(dimension_symbols.get(standard, standard))
        dimensions = "(" + ",".join(resolved) + ")"
    return (
        f"    {declaration}, {', '.join(attributes)} :: "
        f"{local_name or argument.local_name}{dimensions}"
    )


def _native_logical_declaration(
    argument: MetadataArgument,
    dimension_symbols: Mapping[str, str],
) -> str:
    dimensions = ""
    if argument.dimensions:
        resolved = [
            dimension_symbols.get(
                _dimension_standard(item), _dimension_standard(item)
            )
            for item in argument.dimensions
        ]
        dimensions = "(" + ",".join(resolved) + ")"
    return f"    logical :: {argument.local_name}{dimensions}"


def _native_character_declaration(
    argument: MetadataArgument,
    dimension_symbols: Mapping[str, str],
) -> str:
    abi_name = f"abi_{argument.local_name}"
    dimensions = ""
    if argument.dimensions:
        resolved = [
            dimension_symbols.get(
                _dimension_standard(item), _dimension_standard(item)
            )
            for item in argument.dimensions
        ]
        dimensions = "(" + ",".join(resolved) + ")"
    return (
        f"    character(len={abi_name}_length) :: "
        f"{argument.local_name}{dimensions}"
    )


def _native_opaque_declaration(argument: MetadataArgument) -> str:
    dimensions = ""
    if argument.rank:
        dimensions = "(" + ",".join(":" for _ in argument.dimensions) + ")"
    return (
        f"    type({argument.fortran_type}), pointer :: "
        f"{argument.local_name}{dimensions}"
    )


def _opaque_pointer_line(
    argument: MetadataArgument,
    dimension_symbols: Mapping[str, str],
) -> str:
    abi_name = f"abi_{argument.local_name}"
    if not argument.rank:
        return f"    call c_f_pointer({abi_name},{argument.local_name})"
    shape = ",".join(
        dimension_symbols.get(
            _dimension_standard(item), _dimension_standard(item)
        )
        for item in argument.dimensions
    )
    return (
        f"    call c_f_pointer({abi_name},{argument.local_name},"
        f"[{shape}])"
    )


def _opaque_factory_functions(
    *,
    symbol_stem: str,
    fortran_type: str,
    rank: int,
) -> tuple[str, str, str]:
    """Return factory symbol, destroy symbol, and Fortran implementations."""

    factory_symbol = f"{symbol_stem}_create_v{DEVICE_ABI_VERSION}"
    destroy_symbol = f"{symbol_stem}_destroy_v{DEVICE_ABI_VERSION}"
    factory_local = _fortran_identifier(factory_symbol)
    destroy_local = _fortran_identifier(destroy_symbol)
    dimensions = [f"extent_{index + 1}" for index in range(rank)]
    dimension_signature = (
        ("," + ",".join(dimensions)) if dimensions else ""
    )
    pointer_shape = (
        "(" + ",".join(":" for _ in dimensions) + ")"
        if dimensions
        else ""
    )
    allocate_shape = (
        "(" + ",".join(dimensions) + ")" if dimensions else ""
    )
    address_target = (
        "object(" + ",".join("1" for _ in dimensions) + ")"
        if dimensions
        else "object"
    )
    shape_vector = (
        ",[" + ",".join(dimensions) + "]" if dimensions else ""
    )
    lines = [
        f"  function {factory_local}("
        + ",".join(dimensions)
        + f') result(address) bind(C,name="{factory_symbol}")',
        "    type(c_ptr) :: address",
    ]
    lines.extend(
        f"    integer(c_int), value, intent(in) :: {name}"
        for name in dimensions
    )
    lines.extend(
        [
            f"    type({fortran_type}), pointer :: object{pointer_shape}",
            f"    allocate(object{allocate_shape})",
            f"    address = c_loc({address_target})",
            f"  end function {factory_local}",
            "",
            f"  subroutine {destroy_local}(address{dimension_signature}) "
            f'bind(C,name="{destroy_symbol}")',
            "    type(c_ptr), value, intent(in) :: address",
        ]
    )
    lines.extend(
        f"    integer(c_int), value, intent(in) :: {name}"
        for name in dimensions
    )
    lines.extend(
        [
            f"    type({fortran_type}), pointer :: object{pointer_shape}",
            f"    call c_f_pointer(address,object{shape_vector})",
            "    if (associated(object)) deallocate(object)",
            f"  end subroutine {destroy_local}",
        ]
    )
    return factory_symbol, destroy_symbol, "\n".join(lines)


def _character_copy_lines(
    argument: MetadataArgument,
    dimension_symbols: Mapping[str, str],
    *,
    to_native: bool,
) -> list[str]:
    """Copy scalar/1-D fixed-width C bytes to/from native CHARACTER."""

    name = argument.local_name
    abi = f"abi_{name}"
    length = f"{abi}_length"
    if argument.rank == 0:
        if to_native:
            assignment = (
                f"if ({abi}(abi_char) /= c_null_char) "
                f"{name}(abi_char:abi_char)=achar("
                f"iachar({abi}(abi_char)),kind=kind({name}))"
            )
        else:
            assignment = (
                f"{abi}(abi_char)=achar(iachar("
                f"{name}(abi_char:abi_char)),kind=c_char)"
            )
        return [
            f"    do abi_char=1,int({length})",
            f"      {assignment}",
            "    end do",
        ]
    if argument.rank == 1:
        standard = _dimension_standard(argument.dimensions[0])
        extent = dimension_symbols.get(standard, standard)
        if to_native:
            assignment = (
                f"if ({abi}(abi_char,abi_item) /= c_null_char) "
                f"{name}(abi_item)(abi_char:abi_char)=achar("
                f"iachar({abi}(abi_char,abi_item)),kind=kind({name}))"
            )
        else:
            assignment = (
                f"{abi}(abi_char,abi_item)=achar(iachar("
                f"{name}(abi_item)(abi_char:abi_char)),kind=c_char)"
            )
        return [
            f"    do abi_item=1,int({extent})",
            f"      do abi_char=1,int({length})",
            f"        {assignment}",
            "      end do",
            "    end do",
        ]
    raise DeviceBuildError(
        f"{argument.local_name}: character rank {argument.rank} is unsupported"
    )


def _internal_declaration(argument: MetadataArgument) -> str:
    if argument.standard_name == "ccpp_error_code":
        return f"    integer :: {argument.local_name}"
    if argument.fortran_type == "character":
        length = "512"
        match = re.fullmatch(r"len\s*=\s*(\d+)", argument.kind)
        if match:
            length = match.group(1)
        return f"    character(len={length}) :: {argument.local_name}"
    raise DeviceBuildError(
        f"unsupported internal argument {argument.standard_name!r}"
    )


def _generate_adapter_and_manifest(
    description: DeviceDescription,
    metadata_entrypoints: Mapping[str, MetadataEntrypoint],
    dependencies: tuple[str, ...],
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested: list[
        tuple[EntrypointDescription, MetadataEntrypoint]
    ] = []
    for entrypoint in description.entrypoints:
        try:
            metadata = metadata_entrypoints[entrypoint.table]
        except KeyError as exc:
            raise DeviceBuildError(
                f"entrypoint {entrypoint.name!r} references metadata table "
                f"{entrypoint.table!r}, which was not found"
            ) from exc
        if metadata.module != description.module:
            raise DeviceBuildError(
                f"metadata table {metadata.table!r} belongs to module "
                f"{metadata.module!r}, expected {description.module!r}"
            )
        requested.append((entrypoint, metadata))

    adapter_path = output_dir / f"{description.name}_adapter.F90"
    version_map_path = output_dir / f"{description.name}.map"
    module_name = _fortran_identifier(
        f"pycam_device_{description.name}_adapter"
    )
    symbols: list[str] = ["pycam_device_abi_version"]
    manifest_entrypoints: dict[str, Any] = {}
    functions: list[str] = []
    type_index = dict(_project_type_index(description.project_root))
    for source in (*description.providers.values(), *description.sources):
        type_index.update(_source_type_index(source))
    opaque_modules: dict[str, str] = {}
    opaque_contracts: dict[tuple[str, int], dict[str, Any]] = {}
    opaque_factory_bodies: list[str] = []
    physconst_names = _module_only_names(description.sources, "physconst")
    unsupported_physconst = sorted(
        set(physconst_names) - set(_PHYS_CONST_BINDINGS)
    )
    if unsupported_physconst:
        raise DeviceBuildError(
            f"device {description.name!r} imports unsupported Python-owned "
            f"physical constants {unsupported_physconst}"
        )

    for entrypoint, metadata in requested:
        public = [
            item
            for item in metadata.arguments
            if item.standard_name not in _INTERNAL_STANDARD_NAMES
        ]
        internal = [
            item
            for item in metadata.arguments
            if item.standard_name in _INTERNAL_STANDARD_NAMES
        ]
        for item in metadata.arguments:
            if item.optional:
                raise DeviceBuildError(
                    f"{metadata.table}.{item.local_name}: optional arguments "
                    "are not supported by device ABI v1"
                )
            if item.allocatable and not item.caller_owned_allocatable:
                raise DeviceBuildError(
                    f"{metadata.table}.{item.local_name}: allocatable "
                    "arguments require a type-specific ownership adapter"
                )
            item.dtype
            if item.dtype == "opaque":
                try:
                    type_module = type_index[item.fortran_type]
                except KeyError as exc:
                    raise DeviceBuildError(
                        f"{metadata.table}.{item.local_name}: cannot locate "
                        f"the module defining derived type "
                        f"{item.fortran_type!r}"
                    ) from exc
                existing = opaque_modules.get(item.fortran_type)
                if existing not in {None, type_module}:
                    raise DeviceBuildError(
                        f"derived type {item.fortran_type!r} is ambiguous "
                        f"between modules {existing!r} and {type_module!r}"
                    )
                opaque_modules[item.fortran_type] = type_module
                signature = (item.fortran_type, item.rank)
                if signature not in opaque_contracts:
                    stem = (
                        f"pycam_device_{description.name}_"
                        f"{item.fortran_type}_rank{item.rank}"
                    )
                    factory, destroy, body = _opaque_factory_functions(
                        symbol_stem=stem,
                        fortran_type=item.fortran_type,
                        rank=item.rank,
                    )
                    opaque_contracts[signature] = {
                        "type": item.fortran_type,
                        "module": type_module,
                        "factory_symbol": factory,
                        "destroy_symbol": destroy,
                    }
                    opaque_factory_bodies.append(body)
                    symbols.extend((factory, destroy))

        scalar_dimensions = {
            item.standard_name: item.local_name
            for item in public
            if item.rank == 0
            and item.fortran_type == "integer"
            and item.standard_name in description.dimension_bindings
        }
        required_dimensions: list[str] = []
        for item in public:
            for expression in item.dimensions:
                dimension = _dimension_standard(expression)
                if dimension.isdigit():
                    continue
                if dimension not in required_dimensions:
                    required_dimensions.append(dimension)
        injected: list[MetadataArgument] = []
        for dimension in required_dimensions:
            if dimension in scalar_dimensions:
                continue
            if dimension not in description.dimension_bindings:
                raise DeviceBuildError(
                    f"{metadata.table}: dimension {dimension!r} has no "
                    "scalar scheme argument or dimension_bindings entry"
                )
            local_name = f"abi_dim_{dimension}"
            scalar_dimensions[dimension] = local_name
            injected.append(
                MetadataArgument(
                    local_name=local_name,
                    standard_name=dimension,
                    fortran_type="integer",
                    kind="c_int",
                    dimensions=(),
                    intent="in",
                    units="count",
                    optional=False,
                    allocatable=False,
                )
            )
        physconst_injected: dict[str, str] = {}
        for constant in physconst_names:
            field_name, units = _PHYS_CONST_BINDINGS[constant]
            local_name = f"abi_const_{constant}"
            physconst_injected[local_name] = constant
            injected.append(
                MetadataArgument(
                    local_name=local_name,
                    standard_name=field_name,
                    fortran_type="real",
                    kind="c_double",
                    dimensions=(),
                    intent="in",
                    units=units,
                    optional=False,
                    allocatable=False,
                )
            )

        symbol = (
            f"pycam_device_{description.name}_{entrypoint.name}_v"
            f"{DEVICE_ABI_VERSION}"
        )
        local_symbol = _fortran_identifier(symbol)
        symbols.append(symbol)
        exposed = [*injected, *public]
        logical_bridges = {
            item.local_name
            for item in public
            if item.fortran_type == "logical"
            and item.kind.lower() != "c_bool"
        }
        character_bridges = {
            item.local_name
            for item in public
            if item.fortran_type == "character"
        }
        opaque_bridges = {
            item.local_name for item in public if item.dtype == "opaque"
        }
        bridges = logical_bridges | character_bridges | opaque_bridges
        argument_names: list[str] = []
        for item in exposed:
            name = (
                f"abi_{item.local_name}"
                if item.local_name in bridges
                else item.local_name
            )
            argument_names.append(name)
            if item.local_name in character_bridges:
                argument_names.append(f"{name}_length")
        signature_names = [
            *argument_names,
            "error_message",
            "error_capacity",
        ]
        lines = [
            f"  integer(c_int) function {local_symbol}("
            + ",".join(signature_names)
            + f') result(status) bind(C,name="{symbol}")',
        ]
        for item in injected:
            lines.append(_fortran_declaration(item, scalar_dimensions))
        for item in public:
            abi_name = (
                f"abi_{item.local_name}"
                if item.local_name in bridges
                else item.local_name
            )
            lines.append(
                _fortran_declaration(
                    item, scalar_dimensions, local_name=abi_name
                )
            )
        lines.extend(
            [
                "    character(kind=c_char), intent(out) :: error_message(*)",
                "    integer(c_int), value, intent(in) :: error_capacity",
            ]
        )
        for item in internal:
            lines.append(_internal_declaration(item))
        for item in public:
            if item.local_name in logical_bridges:
                lines.append(
                    _native_logical_declaration(item, scalar_dimensions)
                )
            elif item.local_name in character_bridges:
                lines.append(
                    _native_character_declaration(item, scalar_dimensions)
                )
            elif item.local_name in opaque_bridges:
                lines.append(_native_opaque_declaration(item))
        if character_bridges:
            lines.append("    integer :: abi_char,abi_item")
        error_code = next(
            (
                item.local_name
                for item in internal
                if item.standard_name == "ccpp_error_code"
            ),
            None,
        )
        error_message = next(
            (
                item.local_name
                for item in internal
                if item.standard_name == "ccpp_error_message"
            ),
            None,
        )
        for item in internal:
            if item.fortran_type == "character":
                lines.append(f"    {item.local_name} = ''")
            elif item.standard_name == "ccpp_error_code":
                lines.append(f"    {item.local_name} = 0")
        for local_name, constant in physconst_injected.items():
            lines.append(
                f"    call pycam_physconst_set_{constant}({local_name})"
            )
        for item in public:
            if item.local_name in opaque_bridges:
                lines.append(_opaque_pointer_line(item, scalar_dimensions))
            elif (
                item.local_name in logical_bridges
                and item.intent in {"in", "inout"}
            ):
                lines.append(
                    f"    {item.local_name} = abi_{item.local_name}"
                )
            elif (
                item.local_name in character_bridges
                and item.intent in {"in", "inout"}
            ):
                lines.append(f"    {item.local_name} = ' '")
                lines.extend(
                    _character_copy_lines(
                        item, scalar_dimensions, to_native=True
                    )
                )
        call_arguments = ",".join(
            item.local_name for item in metadata.arguments
        )
        lines.append(f"    call {metadata.table}({call_arguments})")
        for item in public:
            if (
                item.local_name in logical_bridges
                and item.intent in {"out", "inout"}
            ):
                lines.append(
                    f"    abi_{item.local_name} = {item.local_name}"
                )
            elif (
                item.local_name in character_bridges
                and item.intent in {"out", "inout"}
            ):
                lines.extend(
                    _character_copy_lines(
                        item, scalar_dimensions, to_native=False
                    )
                )
        if error_code is None:
            lines.append("    status = 0_c_int")
        else:
            lines.append(f"    status = int({error_code},c_int)")
        if error_message is None:
            lines.append(
                "    call copy_error_to_c('',error_message,error_capacity)"
            )
        else:
            lines.append(
                f"    call copy_error_to_c({error_message},"
                "error_message,error_capacity)"
            )
        lines.append(f"  end function {local_symbol}")
        functions.append("\n".join(lines))

        manifest_arguments = []
        for item in injected:
            if item.local_name in physconst_injected:
                binding = {
                    "source": "field",
                    "name": item.standard_name,
                }
            else:
                binding = {
                    "source": "dimension",
                    "name": description.dimension_bindings[
                        item.standard_name
                    ],
                }
            manifest_arguments.append(
                _manifest_argument(item, binding, injected=True)
            )
        manifest_arguments.extend(
            _manifest_argument(
                item,
                _binding_for(description, entrypoint, item),
                opaque=(
                    opaque_contracts[(item.fortran_type, item.rank)]
                    if item.dtype == "opaque"
                    else None
                ),
            )
            for item in public
        )
        manifest_entrypoints[entrypoint.name] = {
            "metadata_table": metadata.table,
            "symbol": symbol,
            "arguments": manifest_arguments,
        }

    functions.extend(opaque_factory_bodies)
    type_use_lines = [
        f"  use {module}, only: {','.join(sorted(types))}"
        for module, types in sorted(
            (
                (module, {name for name, owner in opaque_modules.items()
                          if owner == module})
                for module in set(opaque_modules.values())
            ),
            key=lambda item: item[0],
        )
    ]
    physconst_use_lines = (
        [
            "  use physconst, only: "
            + ",".join(
                f"pycam_physconst_set_{name}"
                for name in physconst_names
            )
        ]
        if physconst_names
        else []
    )
    adapter = [
        "! Generated by pycam_sima.model.device_codegen; do not edit.",
        f"module {module_name}",
        "  use iso_c_binding, only: c_bool,c_char,c_double,c_int,c_null_char,"
        "c_ptr,c_f_pointer,c_loc",
        f"  use {description.module}, only: "
        + ",".join(metadata.table for _, metadata in requested),
        *type_use_lines,
        *physconst_use_lines,
        "  implicit none",
        "  private",
        "  public :: pycam_device_abi_version",
    ]
    adapter.extend(
        f"  public :: {_fortran_identifier(symbol)}"
        for symbol in symbols[1:]
    )
    adapter.extend(
        [
            "contains",
            '  integer(c_int) function pycam_device_abi_version() '
            'result(version) bind(C,name="pycam_device_abi_version")',
            f"    version = {DEVICE_ABI_VERSION}_c_int",
            "  end function pycam_device_abi_version",
            "",
            "  subroutine copy_error_to_c(message,buffer,capacity)",
            "    character(len=*), intent(in) :: message",
            "    character(kind=c_char), intent(out) :: buffer(*)",
            "    integer(c_int), value, intent(in) :: capacity",
            "    integer :: index,count",
            "    if (capacity <= 0_c_int) return",
            "    count = min(len_trim(message),int(capacity)-1)",
            "    do index=1,count",
            "      buffer(index)=achar(iachar(message(index:index)),"
            "kind=c_char)",
            "    end do",
            "    buffer(count+1)=c_null_char",
            "  end subroutine copy_error_to_c",
            "",
            "\n\n".join(functions),
            f"end module {module_name}",
            "",
        ]
    )
    adapter_path.write_text("\n".join(adapter))

    version_map = (
        f"PYCAM_DEVICE_{DEVICE_ABI_VERSION}.0 {{\n"
        "  global:\n"
        + "".join(f"    {symbol};\n" for symbol in symbols)
        + "  local: *;\n};\n"
    )
    version_map_path.write_text(version_map)

    def portable_path(path: Path) -> str:
        try:
            return str(path.relative_to(description.project_root))
        except ValueError:
            return str(path)

    digest = hashlib.sha256()
    hashed_files = [
        description.path,
        *description.sources,
        *description.metadata,
        *description.providers.values(),
    ]
    for path in hashed_files:
        digest.update(portable_path(path).encode())
        digest.update(path.read_bytes())
    manifest = {
        "schema_version": 1,
        "abi_version": DEVICE_ABI_VERSION,
        "name": description.name,
        "fortran_module": description.module,
        "library": f"libpycam_device_{description.name}.so",
        "state_policy": description.state_policy,
        "initialize_entrypoint": description.initialize_entrypoint,
        "dimension_bindings": dict(description.dimension_bindings),
        "entrypoints": manifest_entrypoints,
        "processes": dict(description.processes),
        "source": {
            "descriptor": portable_path(description.path),
            "files": [portable_path(path) for path in description.sources],
            "metadata": [portable_path(path) for path in description.metadata],
            "sha256": digest.hexdigest(),
        },
        "fortran_dependencies": list(dependencies),
        "persistent_native_state": (
            description.state_policy == "initialize_once"
        ),
    }
    return adapter_path, version_map_path, manifest


def _validate_elf(library: Path) -> None:
    dynamic = subprocess.run(
        ("/usr/bin/readelf", "-d", str(library)),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.lower()
    for forbidden in _FORBIDDEN_ELF_DEPENDENCIES:
        if forbidden in dynamic:
            raise DeviceBuildError(
                f"{library} has forbidden runtime dependency {forbidden!r}"
            )
    if "rpath" in dynamic or "runpath" in dynamic:
        raise DeviceBuildError(f"{library} contains RPATH/RUNPATH")
    undefined = subprocess.run(
        ("/usr/bin/nm", "-u", str(library)),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.lower()
    for forbidden in (*_FORBIDDEN_ELF_DEPENDENCIES, "cam_init", "cam_run"):
        if forbidden in undefined:
            raise DeviceBuildError(
                f"{library} references forbidden symbol {forbidden!r}"
            )


def build_device(
    descriptor: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path,
    compiler: str,
    fflags: Iterable[str],
    ldflags: Iterable[str] = (),
) -> Path:
    """Generate, compile, and validate one source-preserving device."""

    description = DeviceDescription.from_yaml(
        descriptor, project_root=project_root
    )
    description = resolve_source_closure(description)
    dependencies = _validate_dependencies(description)
    entrypoints = _load_ccpp_entrypoints(description)
    output_dir = Path(output_root).resolve() / description.name
    generated_dir = output_dir / "generated"
    adapter, version_map, manifest = _generate_adapter_and_manifest(
        description, entrypoints, dependencies, generated_dir
    )
    module_dir = output_dir / "mod"
    module_dir.mkdir(parents=True, exist_ok=True)
    library = output_dir / manifest["library"]
    sources = [
        *description.providers.values(),
        *description.sources,
        adapter,
    ]
    command = [
        str(Path(compiler).absolute()),
        *fflags,
        "-shared",
        *ldflags,
        "-J",
        str(module_dir),
        "-I",
        str(module_dir),
        f"-Wl,--version-script={version_map}",
        "-o",
        str(library),
        *(str(path) for path in sources),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", ""),
        "LC_ALL": "C",
    }
    try:
        subprocess.run(
            command,
            check=True,
            env=environment,
            cwd=description.project_root,
        )
        _validate_elf(library)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DeviceBuildError(
            f"failed to compile device {description.name}: {exc}"
        ) from exc
    manifest_path = output_dir / "device.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build one or more source-preserving CCPP devices"
    )
    parser.add_argument("descriptors", nargs="+")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument(
        "--fflags",
        default=(
            "-O2 -march=znver3 -fPIC -ffp-contract=off -fno-fast-math "
            "-ffree-line-length-none -cpp "
            "-DUSE_CONTIGUOUS=contiguous,"
        ),
    )
    parser.add_argument(
        "--ldflags", default="-Wl,--as-needed -Wl,--no-undefined"
    )
    args = parser.parse_args(argv)
    for descriptor in args.descriptors:
        manifest = build_device(
            descriptor,
            project_root=args.project_root,
            output_root=args.output_root,
            compiler=args.compiler,
            fflags=shlex.split(args.fflags),
            ldflags=shlex.split(args.ldflags),
        )
        print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
