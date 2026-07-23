"""Discover every CCPP scheme used by the pinned CAM-SIMA suites."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

import yaml

from .contracts import default_alias_rules, default_contracts
from .device_codegen import (
    _FORBIDDEN_MODULE_PATTERNS,
    _INTERNAL_STANDARD_NAMES,
    _INTRINSIC_MODULES,
    _dimension_standard,
    _logical_fortran_lines,
)
from .errors import DeviceBuildError


CATALOG_SCHEMA_VERSION = 1
_MODULE_DEFINITION = re.compile(
    r"^\s*module\s+(?!procedure\b|subroutine\b|function\b)"
    r"([A-Za-z][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
_USE_MODULE = re.compile(
    r"^\s*use\s*(?:,\s*(?:non_)?intrinsic\s*)?"
    r"(?:::\s*)?([A-Za-z][A-Za-z0-9_]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SuiteOccurrence:
    suite: str
    suite_file: str
    group: str
    path: tuple[str, ...]
    order: int


@dataclass(frozen=True, slots=True)
class CatalogArgument:
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
    def caller_owned_allocatable(self) -> bool:
        kind = self.kind.lower()
        ftype = self.fortran_type.lower()
        primitive = (
            (ftype == "real" and kind in {"kind_phys", "real64", "c_double"})
            or (ftype == "integer" and kind in {"", "c_int"})
            or (ftype == "logical" and kind in {"", "c_bool"})
            or ftype == "character"
        )
        return self.allocatable and bool(self.dimensions) and primitive


@dataclass(frozen=True, slots=True)
class CatalogEntrypoint:
    table: str
    phase: str
    arguments: tuple[CatalogArgument, ...]


@dataclass(frozen=True, slots=True)
class SchemeCatalogEntry:
    name: str
    module: str
    metadata: str
    source: str
    metadata_dependencies: tuple[str, ...]
    lifecycle: tuple[str, ...]
    entrypoints: tuple[CatalogEntrypoint, ...]
    occurrences: tuple[SuiteOccurrence, ...]
    use_modules: tuple[str, ...]
    source_module_dependencies: tuple[str, ...]
    external_cam_dependencies: tuple[str, ...]
    unresolved_modules: tuple[str, ...]
    missing_statepool_fields: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def device_abi_v1_compatible(self) -> bool:
        return not self.blockers

    def machine_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["device_abi_v1_compatible"] = self.device_abi_v1_compatible
        return record


class DeviceCatalog:
    """Complete suite-to-source inventory for the pinned CAM-SIMA tree."""

    def __init__(
        self,
        *,
        project_root: Path,
        cam_root: Path,
        suites: tuple[Path, ...],
        entries: Mapping[str, SchemeCatalogEntry],
        descriptor_overrides: Mapping[str, Mapping[str, Any]],
        descriptor_override_source: Path | None,
    ):
        self.project_root = project_root
        self.cam_root = cam_root
        self.suites = suites
        self.entries = dict(entries)
        self.descriptor_overrides = {
            name: dict(payload)
            for name, payload in descriptor_overrides.items()
        }
        self.descriptor_override_source = descriptor_override_source

    @classmethod
    def discover(
        cls,
        project_root: str | Path,
        *,
        cam_root: str | Path | None = None,
    ) -> "DeviceCatalog":
        root = Path(project_root).resolve()
        cam = (
            Path(cam_root).resolve()
            if cam_root is not None
            else root / "external/CAM-SIMA"
        )
        physics = cam / "src/physics/ncar_ccpp"
        suite_files = tuple(sorted((physics / "suites").glob("suite_*.xml")))
        if not suite_files:
            raise DeviceBuildError(f"no CCPP suite XML files under {physics}")

        occurrences = _suite_occurrences(suite_files, root)
        active_names = frozenset(occurrences)
        override_source = root / "devices/overrides.yaml"
        descriptor_overrides = _load_descriptor_overrides(
            override_source, active_names
        )
        scheme_root = physics / "schemes"
        metadata_map = _metadata_map(scheme_root)
        missing = sorted(active_names - set(metadata_map))
        if missing:
            raise DeviceBuildError(
                f"suite schemes have no metadata/source: {missing}"
            )

        headers = _parse_all_metadata(
            tuple(sorted(scheme_root.rglob("*.meta"))), root, cam
        )
        entrypoints: dict[str, list[CatalogEntrypoint]] = defaultdict(list)
        modules: dict[str, str] = {}
        for header in headers:
            if getattr(header, "header_type", None) != "scheme":
                continue
            base, _, phase = _function_match(header.title)
            if not base:
                continue
            modules[base] = header.module.lower()
            arguments = tuple(
                _catalog_argument(variable)
                for variable in header.variable_list()
            )
            entrypoints[base].append(
                CatalogEntrypoint(
                    table=header.title.lower(),
                    phase=phase,
                    arguments=arguments,
                )
            )

        module_sources = _module_source_index(cam)
        statepool_names = {
            contract.ccpp_standard_name.lower()
            for contract in default_contracts()
            if contract.ccpp_standard_name
        }
        statepool_names.update(
            rule.ccpp_standard_name.lower()
            for rule in default_alias_rules()
            if rule.ccpp_standard_name
        )

        entries: dict[str, SchemeCatalogEntry] = {}
        for name in sorted(active_names):
            metadata, source = metadata_map[name]
            eps = tuple(
                sorted(
                    entrypoints.get(name, ()),
                    key=lambda item: _phase_order(item.phase),
                )
            )
            if not eps:
                raise DeviceBuildError(
                    f"scheme {name!r} has no parsed lifecycle entrypoint"
                )
            dependencies, missing_dependency_files = (
                _metadata_dependencies(metadata)
            )
            source_files = (
                source,
                *(
                    path
                    for path in dependencies
                    if not _source_is_portable_provider(path)
                ),
            )
            use_modules = _use_modules(source_files)
            source_modules: list[str] = []
            external_cam: list[str] = []
            unresolved: list[str] = []
            for module in use_modules:
                if module in _INTRINSIC_MODULES:
                    continue
                if module in _PORTABLE_PROVIDERS:
                    external_cam.append(module)
                    continue
                module_source = module_sources.get(module)
                if module_source is None:
                    unresolved.append(module)
                elif _is_within(module_source, scheme_root):
                    source_modules.append(module)
                else:
                    external_cam.append(module)

            standard_names = {
                argument.standard_name
                for entrypoint in eps
                for argument in entrypoint.arguments
                if argument.standard_name not in _INTERNAL_STANDARD_NAMES
                and argument.standard_name not in _DIMENSION_STANDARD_NAMES
            }
            missing_fields = sorted(standard_names - statepool_names)
            blockers = _abi_blockers(
                eps,
                use_modules,
                tuple(missing_dependency_files),
            )
            entries[name] = SchemeCatalogEntry(
                name=name,
                module=modules.get(name, ""),
                metadata=_relative(metadata, root),
                source=_relative(source, root),
                metadata_dependencies=tuple(
                    _relative(path, root) for path in dependencies
                ),
                lifecycle=tuple(item.phase for item in eps),
                entrypoints=eps,
                occurrences=tuple(occurrences[name]),
                use_modules=use_modules,
                source_module_dependencies=tuple(sorted(source_modules)),
                external_cam_dependencies=tuple(sorted(external_cam)),
                unresolved_modules=tuple(sorted(unresolved)),
                missing_statepool_fields=tuple(missing_fields),
                blockers=tuple(sorted(blockers)),
            )
        return cls(
            project_root=root,
            cam_root=cam,
            suites=suite_files,
            entries=entries,
            descriptor_overrides=descriptor_overrides,
            descriptor_override_source=(
                override_source if override_source.is_file() else None
            ),
        )

    def summary(self) -> dict[str, Any]:
        blocker_counts: Counter[str] = Counter()
        for entry in self.entries.values():
            blocker_counts.update(
                blocker.split(":", 1)[0] for blocker in entry.blockers
            )
        return {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "suite_count": len(self.suites),
            "suite_names": tuple(
                path.stem.removeprefix("suite_") for path in self.suites
            ),
            "active_scheme_count": len(self.entries),
            "descriptor_override_count": len(self.descriptor_overrides),
            "descriptor_override_source": (
                None
                if self.descriptor_override_source is None
                else _relative(
                    self.descriptor_override_source, self.project_root
                )
            ),
            "occurrence_count": sum(
                len(entry.occurrences) for entry in self.entries.values()
            ),
            "device_abi_v1_compatible": sum(
                entry.device_abi_v1_compatible
                for entry in self.entries.values()
            ),
            "schemes_with_external_cam_dependencies": sum(
                bool(entry.external_cam_dependencies)
                for entry in self.entries.values()
            ),
            "schemes_with_unresolved_modules": sum(
                bool(entry.unresolved_modules)
                for entry in self.entries.values()
            ),
            "unique_missing_statepool_fields": len(
                {
                    name
                    for entry in self.entries.values()
                    for name in entry.missing_statepool_fields
                }
            ),
            "blocker_counts": dict(sorted(blocker_counts.items())),
        }

    def machine_record(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "cam_root": str(self.cam_root),
            "suites": [
                _relative(path, self.project_root) for path in self.suites
            ],
            "schemes": [
                self.entries[name].machine_record()
                for name in sorted(self.entries)
            ],
        }

    @property
    def source_revision(self) -> str:
        return _source_revision(self.cam_root)

    def descriptor_payload(self, name: str) -> dict[str, Any]:
        """Return a generated device description for one suite scheme.

        The description is deliberately a source-level contract.  It is
        generated even when the audit reports a host-service or ABI blocker,
        so every suite scheme has one concrete connector specification and a
        blocked build can explain the exact missing service instead of falling
        back to hand-written glue.
        """

        try:
            entry = self.entries[name.lower()]
        except KeyError as exc:
            raise KeyError(f"unknown active CCPP scheme {name!r}") from exc
        source_paths = [
            entry.source,
            *(
                path
                for path in entry.metadata_dependencies
                if Path(path).suffix.lower() in {".f90", ".f", ".for"}
                and not _source_is_portable_provider(
                    (
                        Path(path)
                        if Path(path).is_absolute()
                        else self.project_root / path
                    )
                )
            ),
        ]
        source_modules: set[str] = {entry.module}
        for relative in source_paths:
            path = Path(relative)
            if not path.is_absolute():
                path = self.project_root / path
            for line in _logical_fortran_lines(path):
                match = _MODULE_DEFINITION.match(line)
                if match:
                    source_modules.add(match.group(1).lower())

        phases = {item.phase: item.table for item in entry.entrypoints}
        processes: dict[str, str] = {}
        if "run" in phases:
            processes[entry.name] = "run"
        for phase in phases:
            processes[f"{entry.name}:{phase}"] = phase

        dimensions = {
            argument.standard_name
            for endpoint in entry.entrypoints
            for argument in endpoint.arguments
            if argument.standard_name in _DIMENSION_STANDARD_NAMES
        }
        for endpoint in entry.entrypoints:
            for argument in endpoint.arguments:
                for expression in argument.dimensions:
                    try:
                        dimension = _dimension_standard(expression)
                    except DeviceBuildError:
                        continue
                    if not dimension.isdigit():
                        dimensions.add(dimension)

        # Providers are intentionally uniform across generated descriptors.
        # Recursive dependencies may introduce one of these modules even when
        # the primary scheme source does not import it directly.
        providers = dict(_PORTABLE_PROVIDERS)
        payload = {
            "schema_version": 1,
            "generated": {
                "catalog_schema_version": CATALOG_SCHEMA_VERSION,
                "source_revision": self.source_revision,
                "audit_blockers": list(entry.blockers),
                "external_cam_dependencies": list(
                    entry.external_cam_dependencies
                ),
                "unresolved_modules": list(entry.unresolved_modules),
            },
            "name": entry.name,
            "fortran_module": entry.module,
            "sources": source_paths,
            "metadata": [entry.metadata],
            "source_modules": sorted(source_modules),
            "providers": providers,
            "auto_dependencies": True,
            # Lifecycle actions are explicit processes.  The host, not an
            # implicit native side effect, decides when initialize/finalize
            # entrypoints run.
            "state_policy": "stateless",
            "dimension_bindings": {
                dimension: _pool_dimension(dimension)
                for dimension in sorted(dimensions)
            },
            "entrypoints": {
                phase: {"table": table}
                for phase, table in sorted(
                    phases.items(), key=lambda item: _phase_order(item[0])
                )
            },
            "processes": processes,
        }
        override = self.descriptor_overrides.get(entry.name)
        if override is None:
            return payload
        assert self.descriptor_override_source is not None
        payload["generated"]["override_source"] = _relative(
            self.descriptor_override_source, self.project_root
        )
        return _apply_descriptor_override(payload, override)

    def write_descriptors(
        self, output_root: str | Path, *, clean: bool = False
    ) -> tuple[Path, ...]:
        """Materialize one reproducible ``device.yaml`` per active scheme."""

        root = Path(output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        expected = set(self.entries)
        if clean:
            for child in root.iterdir():
                if child.is_dir() and child.name not in expected:
                    descriptor = child / "device.yaml"
                    if descriptor.is_file():
                        descriptor.unlink()
                    try:
                        child.rmdir()
                    except OSError:
                        pass
        written: list[Path] = []
        for name in sorted(self.entries):
            directory = root / name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "device.yaml"
            header = (
                "# Generated from pinned CAM-SIMA CCPP metadata. "
                "Do not edit by hand.\n"
            )
            path.write_text(
                header
                + yaml.safe_dump(
                    self.descriptor_payload(name),
                    sort_keys=False,
                    width=100,
                )
            )
            written.append(path)
        index = {
            "schema_version": CATALOG_SCHEMA_VERSION,
            "source_revision": self.source_revision,
            "scheme_count": len(written),
            "descriptors": [
                _relative(path, self.project_root) for path in written
            ],
        }
        (root / "index.json").write_text(
            __import__("json").dumps(index, indent=2, sort_keys=True) + "\n"
        )
        return tuple(written)


_DIMENSION_STANDARD_NAMES = frozenset(
    {
        "horizontal_dimension",
        "horizontal_loop_begin",
        "horizontal_loop_end",
        "horizontal_loop_extent",
        "vertical_layer_dimension",
        "vertical_interface_dimension",
    }
)

_POOL_DIMENSIONS = {
    "horizontal_dimension": "nphys_local",
    "horizontal_loop_extent": "nphys_local",
    "vertical_layer_dimension": "pver",
    "vertical_interface_dimension": "pverp",
}

_DESCRIPTOR_OVERRIDE_KEYS = frozenset(
    {
        "state_policy",
        "initialize_entrypoint",
        "bindings",
        "dimension_bindings",
    }
)
_DESCRIPTOR_OVERRIDE_MAPPING_KEYS = frozenset(
    {"bindings", "dimension_bindings"}
)


def _load_descriptor_overrides(
    path: Path, active_names: frozenset[str]
) -> dict[str, dict[str, Any]]:
    """Load small host-policy exceptions that metadata cannot represent."""

    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, Mapping):
        raise DeviceBuildError(f"{path}: override document must be a mapping")
    if payload.get("schema_version") != 1:
        raise DeviceBuildError(
            f"{path}: unsupported override schema_version "
            f"{payload.get('schema_version')!r}"
        )
    schemes = payload.get("schemes", {})
    if not isinstance(schemes, Mapping):
        raise DeviceBuildError(f"{path}: schemes must be a mapping")

    unknown_schemes = sorted(
        str(name).lower()
        for name in schemes
        if str(name).lower() not in active_names
    )
    if unknown_schemes:
        raise DeviceBuildError(
            f"{path}: overrides reference inactive schemes {unknown_schemes}"
        )

    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_override in schemes.items():
        name = str(raw_name).lower()
        if not isinstance(raw_override, Mapping):
            raise DeviceBuildError(
                f"{path}: override for {name!r} must be a mapping"
            )
        unknown_keys = sorted(
            str(key)
            for key in raw_override
            if str(key) not in _DESCRIPTOR_OVERRIDE_KEYS
        )
        if unknown_keys:
            raise DeviceBuildError(
                f"{path}: override for {name!r} has unsupported keys "
                f"{unknown_keys}"
            )
        override = dict(raw_override)
        for key in _DESCRIPTOR_OVERRIDE_MAPPING_KEYS:
            if key in override and not isinstance(override[key], Mapping):
                raise DeviceBuildError(
                    f"{path}: {name}.{key} must be a mapping"
                )
        result[name] = override
    return result


def _apply_descriptor_override(
    payload: dict[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge validated policy fields without replacing source-derived ABI."""

    result = dict(payload)
    for key, value in override.items():
        if key in _DESCRIPTOR_OVERRIDE_MAPPING_KEYS:
            merged = dict(result.get(key, {}))
            merged.update(dict(value))
            result[key] = merged
        else:
            result[key] = value
    return result


_PORTABLE_PROVIDERS = {
    "cam_abortutils": "native/devices/support/cam_abortutils.F90",
    "cam_logfile": "native/devices/support/cam_logfile.F90",
    "ccpp_kinds": "native/devices/support/ccpp_kinds.F90",
    "error_messages": "native/devices/support/error_messages.F90",
    "physconst": "native/devices/support/physconst.F90",
    "ref_pres": "native/devices/support/ref_pres.F90",
    "shr_kind_mod": "native/devices/support/shr_kind_mod.F90",
    "shr_log_mod": "native/devices/support/shr_log_mod.F90",
    "shr_assert_mod": "native/devices/support/shr_assert_mod.F90",
    "shr_sys_mod": "native/devices/support/shr_sys_mod.F90",
    "spmd_utils": "native/devices/support/spmd_utils.F90",
}


def _pool_dimension(standard_name: str) -> str:
    return _POOL_DIMENSIONS.get(standard_name, standard_name)


def _source_revision(cam_root: Path) -> str:
    head = cam_root / ".git"
    try:
        import subprocess

        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=cam_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ccpp_scripts(cam_root: Path) -> Path:
    return cam_root / "ccpp_framework/scripts"


def _function_match(name: str) -> tuple[str | None, str | None, str]:
    from ccpp_state_machine import CCPP_STATE_MACH

    base, transition, phase = CCPP_STATE_MACH.function_match(name)
    return base, transition, phase


def _parse_all_metadata(
    metadata: tuple[Path, ...], project_root: Path, cam_root: Path
) -> tuple[Any, ...]:
    scripts = _ccpp_scripts(cam_root)
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from ccpp_capgen import parse_scheme_files
    from framework_env import CCPPFrameworkEnv
    from parse_tools import init_log, set_log_to_null

    logger = init_log("pycam-device-catalog")
    set_log_to_null(logger)
    run_env = CCPPFrameworkEnv(
        logger,
        host_files="",
        scheme_files="",
        suites="",
        kind_types=["kind_phys=REAL64"],
        output_root=str(project_root / "build/.ccpp-device-catalog"),
    )
    headers, _ = parse_scheme_files(
        [str(path) for path in metadata],
        run_env,
        skip_ddt_check=True,
    )
    return tuple(headers)


def _metadata_map(scheme_root: Path) -> dict[str, tuple[Path, Path]]:
    scripts = _ccpp_scripts(
        scheme_root.parents[3]
    )
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from ccpp_capgen import find_associated_fortran_file
    from metadata_table import find_scheme_names

    result: dict[str, tuple[Path, Path]] = {}
    for metadata in sorted(scheme_root.rglob("*.meta")):
        source = Path(
            find_associated_fortran_file(
                str(metadata.resolve()), str(metadata.parent.resolve())
            )
        ).resolve()
        for name in find_scheme_names(str(metadata)):
            key = name.lower()
            if key in result:
                raise DeviceBuildError(
                    f"duplicate metadata definition for scheme {key!r}"
                )
            result[key] = (metadata.resolve(), source)
    return result


def _suite_occurrences(
    suite_files: Iterable[Path], project_root: Path
) -> dict[str, list[SuiteOccurrence]]:
    result: dict[str, list[SuiteOccurrence]] = defaultdict(list)
    for suite_file in suite_files:
        suite = ET.parse(suite_file).getroot()
        suite_name = suite.attrib.get(
            "name", suite_file.stem.removeprefix("suite_")
        )
        order = 0

        def walk(
            node: ET.Element, group: str, path: tuple[str, ...]
        ) -> None:
            nonlocal order
            for child in node:
                tag = child.tag.lower()
                if tag == "scheme":
                    name = (child.text or "").strip().lower()
                    if not name:
                        continue
                    order += 1
                    result[name].append(
                        SuiteOccurrence(
                            suite=suite_name,
                            suite_file=_relative(suite_file, project_root),
                            group=group,
                            path=path,
                            order=order,
                        )
                    )
                    continue
                child_group = (
                    child.attrib.get("name", group)
                    if tag == "group"
                    else group
                )
                label = child.attrib.get(
                    "name", child.attrib.get("loop", tag)
                )
                walk(child, child_group, (*path, f"{tag}:{label}"))

        walk(suite, "", ())
    return result


def _catalog_argument(variable: Any) -> CatalogArgument:
    optional = variable.get_prop_value("optional")
    allocatable = variable.get_prop_value("allocatable")
    return CatalogArgument(
        local_name=variable.get_prop_value("local_name").lower(),
        standard_name=variable.get_prop_value("standard_name").lower(),
        fortran_type=variable.get_prop_value("type").lower(),
        kind=(variable.get_prop_value("kind") or "").lower(),
        dimensions=tuple(
            str(item).lower()
            for item in variable.get_prop_value("dimensions")
        ),
        intent=variable.get_prop_value("intent").lower(),
        units=variable.get_prop_value("units") or "",
        optional=_truthy(optional),
        allocatable=_truthy(allocatable),
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            ".true.",
            "t",
            "yes",
            "1",
        }
    return bool(value)


def _phase_order(phase: str) -> tuple[int, str]:
    order = {
        "register": 0,
        "initialize": 1,
        "timestep_initial": 2,
        "run": 3,
        "timestep_final": 4,
        "finalize": 5,
    }
    return order.get(phase, 99), phase


def _metadata_dependencies(metadata: Path) -> tuple[tuple[Path, ...], list[str]]:
    result: list[Path] = []
    missing: list[str] = []
    for raw in metadata.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line.lower().startswith("dependencies"):
            continue
        _, _, values = line.partition("=")
        for value in values.split(","):
            text = value.strip()
            if not text:
                continue
            path = (metadata.parent / text).resolve()
            if path.is_file():
                if path not in result:
                    result.append(path)
            else:
                missing.append(str(path))
    return tuple(result), missing


def _module_source_index(cam_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for source in sorted(cam_root.rglob("*.F90")):
        try:
            lines = _logical_fortran_lines(source)
            for line in lines:
                match = _MODULE_DEFINITION.match(line)
                if match:
                    result.setdefault(match.group(1).lower(), source.resolve())
        except UnicodeDecodeError:
            continue
    return result


def _use_modules(sources: Iterable[Path]) -> tuple[str, ...]:
    result: set[str] = set()
    for source in sources:
        for line in _logical_fortran_lines(source):
            match = _USE_MODULE.match(line)
            if match:
                result.add(match.group(1).lower())
    return tuple(sorted(result))


def _source_is_portable_provider(source: Path) -> bool:
    """Return whether every module in a dependency has a local provider."""

    defined = {
        match.group(1).lower()
        for line in _logical_fortran_lines(source)
        if (match := _MODULE_DEFINITION.match(line))
    }
    return bool(defined) and defined <= frozenset(_PORTABLE_PROVIDERS)


def _abi_blockers(
    entrypoints: tuple[CatalogEntrypoint, ...],
    use_modules: tuple[str, ...],
    missing_dependency_files: tuple[str, ...],
) -> set[str]:
    blockers: set[str] = set()
    for module in use_modules:
        if any(pattern.search(module) for pattern in _FORBIDDEN_MODULE_PATTERNS):
            blockers.add(f"host_framework_module:{module}")
    for path in missing_dependency_files:
        blockers.add(f"missing_metadata_dependency:{path}")
    for entrypoint in entrypoints:
        for argument in entrypoint.arguments:
            label = f"{entrypoint.table}.{argument.local_name}"
            if argument.optional:
                blockers.add(f"optional_argument:{label}")
            if argument.allocatable and not argument.caller_owned_allocatable:
                blockers.add(f"allocatable_argument:{label}")
            if argument.standard_name in _INTERNAL_STANDARD_NAMES:
                continue
            kind = argument.kind.lower()
            ftype = argument.fortran_type.lower()
            if ftype == "real":
                if kind not in {"kind_phys", "real64", "c_double"}:
                    blockers.add(f"unsupported_kind:{label}:{ftype}/{kind}")
            elif ftype == "integer":
                if kind not in {"", "c_int"}:
                    blockers.add(f"unsupported_kind:{label}:{ftype}/{kind}")
            elif ftype == "logical":
                if kind not in {"", "c_bool"}:
                    blockers.add(
                        f"unsupported_kind:{label}:{ftype}/{kind}"
                    )
            elif ftype == "character":
                if len(argument.dimensions) > 1:
                    blockers.add(
                        f"character_rank:{label}:{len(argument.dimensions)}"
                    )
            else:
                # Non-allocatable derived types cross ABI v1 as opaque
                # process-state handles.  Python owns the lifetime record;
                # generated Fortran factories own allocation/deallocation.
                pass
            for dimension in argument.dimensions:
                try:
                    _dimension_standard(dimension)
                except DeviceBuildError:
                    blockers.add(f"dimension_expression:{label}:{dimension}")
    return blockers


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())
