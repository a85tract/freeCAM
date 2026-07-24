"""Runtime-discovered physics devices and dynamically registered variables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from .contracts import FieldContract
from .device_codegen import DeviceDescription, _validate_elf, build_device
from .devices import FortranDevice
from .errors import DeviceContractError, StateOwnershipError
from .scheme_plan import PHYSICS_BEFORE_COUPLER, PhysicsScheme, SCHEME_GROUPS


PLUGIN_SCHEMA_VERSION = 1
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_DEFAULT_COMPILER = "/opt/cray/pe/gcc/12.2.0/bin/gfortran"
_DEFAULT_FFLAGS = (
    "-O2",
    "-fPIC",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-fno-reciprocal-math",
    "-fno-associative-math",
    "-fno-unsafe-math-optimizations",
    "-ffree-line-length-none",
    "-cpp",
    "-DUSE_CONTIGUOUS=contiguous,",
)
_DEFAULT_LDFLAGS = ("-Wl,--as-needed", "-Wl,--no-undefined")
_UNSET = object()


def _safe_name(value: str, label: str) -> str:
    text = str(value)
    if not _NAME.fullmatch(text):
        raise ValueError(
            f"{label} must start with a letter and contain only letters, "
            "digits, dot, dash, and underscore"
        )
    return text


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Public, serializable description of one Python-owned state variable."""

    name: str
    dtype: str
    dimensions: tuple[str, ...]
    standard_name: str | None = None
    units: str = "1"
    intent: str = "inout"
    category: str = "plugin_state"
    aliases: tuple[str, ...] = ()
    restart: bool = True
    history: bool = False
    writable: bool = True

    def __post_init__(self) -> None:
        _safe_name(self.name, "variable name")
        object.__setattr__(
            self, "dimensions", tuple(str(item) for item in self.dimensions)
        )
        object.__setattr__(
            self, "aliases", tuple(str(item) for item in self.aliases)
        )
        if self.standard_name is not None:
            object.__setattr__(
                self, "standard_name", str(self.standard_name).lower()
            )
        if self.intent not in {"in", "out", "inout"}:
            raise ValueError("variable intent must be in, out, or inout")
        if self.history and self.dimensions not in {
            ("nphys_local",),
            ("nphys_local", "pver"),
            ("nphys_local", "pverp"),
        }:
            raise ValueError(
                "history variables must use (nphys_local), "
                "(nphys_local, pver), or (nphys_local, pverp)"
            )
        np.dtype(self.dtype)

    def contract(self) -> FieldContract:
        return FieldContract(
            standard_name=self.name,
            ccpp_standard_name=self.standard_name,
            dtype=self.dtype,
            dimensions=self.dimensions,
            intent=self.intent,
            category=self.category,
            units=self.units,
            aliases=self.aliases,
            owner="python",
            lifetime="persistent",
            history=self.history,
            restart=self.restart,
            writable=self.writable,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = list(self.dimensions)
        payload["aliases"] = list(self.aliases)
        return payload

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "VariableSpec":
        return cls(
            name=str(values["name"]),
            standard_name=(
                None
                if values.get("standard_name") is None
                else str(values["standard_name"])
            ),
            dtype=str(values["dtype"]),
            dimensions=tuple(
                str(item) for item in values.get("dimensions", ())
            ),
            units=str(values.get("units", "1")),
            intent=str(values.get("intent", "inout")),
            category=str(values.get("category", "plugin_state")),
            aliases=tuple(str(item) for item in values.get("aliases", ())),
            restart=bool(values.get("restart", True)),
            history=bool(values.get("history", False)),
            writable=bool(values.get("writable", True)),
        )


@dataclass(frozen=True, slots=True)
class SchemePlacement:
    process: str
    group: str = PHYSICS_BEFORE_COUPLER
    before: str | None = None
    after: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        _safe_name(self.process.replace(":", "."), "plugin process")
        if self.group not in SCHEME_GROUPS:
            raise ValueError(
                f"unknown scheme group {self.group!r}; choose from "
                f"{SCHEME_GROUPS}"
            )
        if self.before is not None and self.after is not None:
            raise ValueError("placement accepts at most one of before or after")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SchemePlacement":
        return cls(
            process=str(values["process"]),
            group=str(values.get("group", PHYSICS_BEFORE_COUPLER)),
            before=(
                None if values.get("before") is None else str(values["before"])
            ),
            after=(
                None if values.get("after") is None else str(values["after"])
            ),
            enabled=bool(values.get("enabled", True)),
        )


@dataclass(frozen=True, slots=True)
class PhysicsPluginSpec:
    """A source descriptor or prebuilt manifest plus plan placements."""

    source: str
    placements: tuple[SchemePlacement, ...] = ()
    variables: tuple[VariableSpec, ...] = ()
    project_root: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(
            self,
            "placements",
            tuple(
                item
                if isinstance(item, SchemePlacement)
                else SchemePlacement.from_mapping(item)
                for item in self.placements
            ),
        )
        object.__setattr__(
            self,
            "variables",
            tuple(
                item
                if isinstance(item, VariableSpec)
                else VariableSpec.from_mapping(item)
                for item in self.variables
            ),
        )
        if self.name is not None:
            _safe_name(self.name, "plugin name")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLUGIN_SCHEMA_VERSION,
            "source": self.source,
            "project_root": self.project_root,
            "name": self.name,
            "placements": [item.as_dict() for item in self.placements],
            "variables": [item.as_dict() for item in self.variables],
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PhysicsPluginSpec":
        version = values.get("schema_version", PLUGIN_SCHEMA_VERSION)
        if version != PLUGIN_SCHEMA_VERSION:
            raise ValueError(f"unsupported physics plugin schema {version!r}")
        return cls(
            source=str(values["source"]),
            project_root=(
                None
                if values.get("project_root") is None
                else str(values["project_root"])
            ),
            name=(
                None if values.get("name") is None else str(values["name"])
            ),
            placements=tuple(
                SchemePlacement.from_mapping(item)
                for item in values.get("placements", ())
            ),
            variables=tuple(
                VariableSpec.from_mapping(item)
                for item in values.get("variables", ())
            ),
        )


@dataclass(slots=True)
class InstalledPhysicsPlugin:
    name: str
    manifest_path: str
    manifest_hash: str
    library_hash: str
    source_hash: str
    state_policy: str
    placements: tuple[SchemePlacement, ...]
    variables: tuple[VariableSpec, ...]
    active: bool = True
    pending: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manifest_path": self.manifest_path,
            "manifest_hash": self.manifest_hash,
            "library_hash": self.library_hash,
            "source_hash": self.source_hash,
            "state_policy": self.state_policy,
            "placements": [item.as_dict() for item in self.placements],
            "variables": [item.as_dict() for item in self.variables],
            "active": self.active,
            "pending": self.pending,
        }

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "InstalledPhysicsPlugin":
        return cls(
            name=str(values["name"]),
            manifest_path=str(values["manifest_path"]),
            manifest_hash=str(values["manifest_hash"]),
            library_hash=str(values["library_hash"]),
            source_hash=str(values["source_hash"]),
            state_policy=str(values["state_policy"]),
            placements=tuple(
                SchemePlacement.from_mapping(item)
                for item in values.get("placements", ())
            ),
            variables=tuple(
                VariableSpec.from_mapping(item)
                for item in values.get("variables", ())
            ),
            active=bool(values.get("active", True)),
            pending=bool(values.get("pending", False)),
        )


class PhysicsPluginManager:
    """Collectively extend one live CAMDriver with generated devices."""

    def __init__(
        self,
        driver: Any,
        *,
        cache_dir: str | Path | None = None,
        compiler: str | Path | None = None,
        fflags: Iterable[str] = _DEFAULT_FFLAGS,
        ldflags: Iterable[str] = _DEFAULT_LDFLAGS,
    ) -> None:
        self.driver = driver
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "PYCAM_SIMA_PLUGIN_CACHE",
                Path.home() / ".cache/pycam-sima/plugins",
            )
        ).expanduser().resolve()
        self.compiler = str(
            compiler or os.environ.get("PYCAM_SIMA_PLUGIN_COMPILER", _DEFAULT_COMPILER)
        )
        self.fflags = tuple(fflags)
        self.ldflags = tuple(ldflags)
        self.installed: dict[str, InstalledPhysicsPlugin] = {}

    def discover(
        self, roots: Iterable[str | Path] = ()
    ) -> tuple[dict[str, str], ...]:
        """List source and prebuilt devices visible through configured roots."""

        candidates = [Path(item).expanduser() for item in roots]
        candidates.extend(
            Path(item)
            for item in os.environ.get("PYCAM_SIMA_PLUGIN_PATH", "").split(
                os.pathsep
            )
            if item.strip()
        )
        records: dict[str, dict[str, str]] = {}
        for root in candidates:
            paths = (
                (root,)
                if root.is_file()
                else tuple(root.glob("*/device.json"))
                + tuple(root.glob("*/device.yaml"))
            )
            for path in paths:
                try:
                    payload = (
                        json.loads(path.read_text())
                        if path.suffix == ".json"
                        else yaml.safe_load(path.read_text())
                    )
                    name = str(payload["name"])
                except (OSError, KeyError, TypeError, ValueError):
                    continue
                records[name] = {
                    "name": name,
                    "source": str(path.resolve()),
                    "kind": (
                        "prebuilt" if path.suffix == ".json" else "source"
                    ),
                }
        for entrypoint in importlib_metadata.entry_points(
            group="pycam_sima.physics"
        ):
            records.setdefault(
                entrypoint.name,
                {
                    "name": entrypoint.name,
                    "source": entrypoint.name,
                    "kind": "entry_point",
                },
            )
        return tuple(records[name] for name in sorted(records))

    def define_variable(
        self,
        spec: VariableSpec,
        *,
        initial: Any = _UNSET,
    ) -> np.ndarray:
        self._require_boundary()
        error: str | None = None
        try:
            spec.contract().shape(self.driver.pool.dimensions)
            np.dtype(spec.dtype)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error, "dynamic variable validation")
        value: np.ndarray | None = None
        local_error: str | None = None
        try:
            if initial is _UNSET:
                value = self.driver.pool.register_field(
                    spec.contract(), initialized=False
                )
            else:
                value = self.driver.pool.register_field(
                    spec.contract(), initial=initial, initialized=True
                )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            if value is not None:
                self.driver.pool.unregister_field(spec.name)
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            ]
            raise StateOwnershipError(
                "dynamic variable definition rolled back collectively: "
                + "; ".join(failures)
            )
        assert value is not None
        self._assert_schema_consistent()
        return value

    def install(
        self,
        spec: PhysicsPluginSpec | str | Path,
        *,
        initial_values: Mapping[str, Any] | None = None,
        effective: str = "now",
        unsafe: bool = False,
    ) -> InstalledPhysicsPlugin:
        if not unsafe:
            raise ValueError("installing new physics requires unsafe=True")
        if effective not in {"now", "next_step"}:
            raise ValueError("effective must be 'now' or 'next_step'")
        if not isinstance(spec, PhysicsPluginSpec):
            spec = PhysicsPluginSpec(str(spec))
        self._require_boundary()
        manifest_path = self._resolve_manifest_collectively(spec)
        device: FortranDevice | None = None
        load_error: str | None = None
        try:
            device = FortranDevice(manifest_path)
        except BaseException as exc:
            load_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(load_error, "plugin load")
        assert device is not None
        plugin_name = spec.name or device.name
        identity_error: str | None = None
        if plugin_name != device.name:
            identity_error = (
                "plugin name must match the generated device name; "
                f"got {plugin_name!r} and {device.name!r}"
            )
        elif plugin_name in self.installed:
            identity_error = (
                f"physics plugin {plugin_name!r} is already installed"
            )
        self._collective_error(identity_error, "plugin identity")
        placements = spec.placements or (
            SchemePlacement(self._default_process(device)),
        )
        self._preflight_device(device, placements)
        initial_values = dict(initial_values or {})
        variables: tuple[VariableSpec, ...] = ()
        new_specs: list[VariableSpec] = []
        schema_error: str | None = None
        try:
            inferred = self._infer_variables(device)
            variables = self._merge_variables(spec.variables, inferred)
            new_specs = [
                item for item in variables
                if not self._variable_is_provided(item)
            ]
        except BaseException as exc:
            schema_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(schema_error, "plugin schema resolution")
        self._preflight_variables(new_specs, initial_values)

        manifest_hash = ""
        library_hash = ""
        hash_error: str | None = None
        try:
            manifest_hash = _file_hash(device.manifest_path)
            library_hash = _file_hash(device.library_path)
        except BaseException as exc:
            hash_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(hash_error, "plugin hash")
        hashes = self.driver.comm.allgather(
            (manifest_hash, library_hash, device.source_hash)
        )
        if len(set(hashes)) != 1:
            raise DeviceContractError(
                f"plugin bytes differ across MPI ranks: {hashes}"
            )

        arrays_before = self.driver.pool.snapshot_arrays(readonly=False)
        added_fields: list[str] = []
        added_schemes: list[str] = []
        registered_device = False
        record: InstalledPhysicsPlugin | None = None
        local_error: str | None = None
        try:
            for variable in new_specs:
                initial = _initial_value(initial_values, variable)
                if initial is _UNSET:
                    values = self.driver.pool.register_field(
                        variable.contract(), initialized=False
                    )
                else:
                    values = self.driver.pool.register_field(
                        variable.contract(),
                        initial=initial,
                        initialized=True,
                    )
                if self.driver.pool.sealed and (
                    not variable.writable
                    or variable.category
                    in {
                        "configuration",
                        "constants",
                        "vertical_coordinate",
                        "grid",
                        "topology",
                        "communication",
                    }
                ):
                    values.flags.writeable = False
                added_fields.append(variable.name)
            self.driver.backend.devices.register(device)
            registered_device = True
            self._run_lifecycle(device, placements, "register")
            self._run_lifecycle(device, placements, "initialize")
            enabled = effective == "now"
            for placement in placements:
                scheme = PhysicsScheme(
                    name=placement.process,
                    group=placement.group,
                    source_group=f"plugin:{plugin_name}",
                    category="plugin",
                    description=f"runtime physics plugin {plugin_name}",
                    implementation="fortran-device",
                    required=False,
                    enabled=enabled and placement.enabled,
                )
                self.driver.scheme_plan.add(
                    scheme,
                    before=placement.before,
                    after=placement.after,
                    unsafe=True,
                )
                added_schemes.append(scheme.key)
            record = InstalledPhysicsPlugin(
                name=plugin_name,
                manifest_path=str(device.manifest_path),
                manifest_hash=manifest_hash,
                library_hash=library_hash,
                source_hash=device.source_hash,
                state_policy=device.state_policy,
                placements=placements,
                variables=variables,
                active=enabled,
                pending=effective == "next_step",
            )
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            for key in reversed(added_schemes):
                self.driver.scheme_plan.remove(key, unsafe=True)
            if registered_device:
                self.driver.backend.devices.unregister(device.name)
            for name in reversed(added_fields):
                self.driver.pool.unregister_field(name)
            self.driver.pool.restore_arrays(arrays_before)
            failures = [
                f"rank {rank}: {error}"
                for rank, error in enumerate(errors)
                if error is not None
            ]
            raise DeviceContractError(
                "plugin installation rolled back collectively: "
                + "; ".join(failures)
            )
        assert record is not None
        self.installed[plugin_name] = record
        self._assert_schema_consistent()
        self.driver.comm.barrier()
        return record

    def activate_pending(self) -> None:
        for record in self.installed.values():
            if not record.pending:
                continue
            for placement in record.placements:
                key = f"plugin:{record.name}.{placement.process}"
                if placement.enabled:
                    self.driver.scheme_plan.enable(key)
            record.pending = False
            record.active = True

    def activate(self, name: str, *, unsafe: bool = False) -> None:
        if not unsafe:
            raise ValueError("activating new physics requires unsafe=True")
        self._require_boundary()
        record = self.installed.get(name)
        self._collective_error(
            None if record is not None else f"unknown physics plugin {name!r}",
            "plugin activation",
        )
        assert record is not None
        if record.active and not record.pending:
            self.driver.comm.barrier()
            return
        device = self.driver.backend.devices.devices.get(name)
        self._collective_error(
            None if device is not None else f"missing device {name!r}",
            "plugin activation",
        )
        assert device is not None
        arrays_before = self.driver.pool.snapshot_arrays(readonly=False)
        changed: list[str] = []
        local_error: str | None = None
        try:
            if not record.pending:
                self._run_lifecycle(
                    device, record.placements, "initialize"
                )
            for placement in record.placements:
                key = f"plugin:{record.name}.{placement.process}"
                if placement.enabled:
                    self.driver.scheme_plan.enable(key)
                    changed.append(key)
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            for key in changed:
                self.driver.scheme_plan.disable(key, unsafe=True)
            self.driver.pool.restore_arrays(arrays_before)
            raise DeviceContractError(
                f"plugin activation rolled back collectively: {errors}"
            )
        record.pending = False
        record.active = True
        self.driver.comm.barrier()

    def deactivate(self, name: str, *, unsafe: bool = False) -> None:
        if not unsafe:
            raise ValueError("deactivating physics requires unsafe=True")
        self._require_boundary()
        record = self.installed.get(name)
        device = self.driver.backend.devices.devices.get(name)
        lookup_error = (
            None
            if record is not None and device is not None
            else f"unknown physics plugin {name!r}"
        )
        self._collective_error(lookup_error, "plugin deactivation")
        assert record is not None and device is not None
        if not record.active and not record.pending:
            self.driver.comm.barrier()
            return
        arrays_before = self.driver.pool.snapshot_arrays(readonly=False)
        local_error: str | None = None
        try:
            self._run_lifecycle(device, record.placements, "finalize")
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            self.driver.pool.restore_arrays(arrays_before)
            raise DeviceContractError(
                f"plugin deactivation rolled back collectively: {errors}"
            )
        for placement in record.placements:
            key = f"plugin:{record.name}.{placement.process}"
            self.driver.scheme_plan.disable(key, unsafe=True)
        record.active = False
        record.pending = False
        self.driver.comm.barrier()

    def finalize_all(self) -> None:
        """Run each still-active plugin finalizer before driver teardown."""

        for record in self.installed.values():
            if not record.active and not record.pending:
                continue
            device = self.driver.backend.devices.devices.get(record.name)
            local_error: str | None = None
            try:
                if device is None:
                    raise DeviceContractError(
                        f"missing installed device {record.name!r}"
                    )
                self._run_lifecycle(
                    device, record.placements, "finalize"
                )
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
            self._collective_error(
                local_error, f"plugin {record.name!r} finalization"
            )
            record.active = False
            record.pending = False

    def inventory(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            self.installed[name].as_dict()
            for name in sorted(self.installed)
        )

    def assert_checkpointable(self) -> None:
        unsupported = [
            record.name
            for record in self.installed.values()
            if record.state_policy == "initialize_once"
        ]
        if unsupported:
            raise StateOwnershipError(
                "checkpoint requires a serializer for initialize_once "
                f"plugin state: {unsupported}"
            )

    def restore_inventory(
        self, records: Iterable[Mapping[str, Any]]
    ) -> None:
        """Reload exact artifacts without replaying numerical lifecycle code."""

        for payload in records:
            record = InstalledPhysicsPlugin.from_mapping(payload)
            path = Path(record.manifest_path)
            if _file_hash(path) != record.manifest_hash:
                raise DeviceContractError(
                    f"checkpoint plugin manifest changed: {path}"
                )
            device = FortranDevice(path)
            if _file_hash(device.library_path) != record.library_hash:
                raise DeviceContractError(
                    f"checkpoint plugin library changed: "
                    f"{device.library_path}"
                )
            if device.source_hash != record.source_hash:
                raise DeviceContractError(
                    f"checkpoint plugin source hash changed: {record.name}"
                )
            if record.state_policy == "initialize_once":
                raise DeviceContractError(
                    f"plugin {record.name!r} has non-restartable "
                    "initialize_once native state"
                )
            self.driver.backend.devices.register(device)
            self.installed[record.name] = record

    def _resolve_manifest_collectively(
        self, spec: PhysicsPluginSpec
    ) -> Path:
        payload: dict[str, Any] | None = None
        if self.driver.comm.rank == 0:
            try:
                path = self._resolve_source(spec.source)
                if path.name == "device.yaml":
                    root = (
                        Path(spec.project_root).resolve()
                        if spec.project_root
                        else _infer_project_root(path)
                    )
                    description = DeviceDescription.from_yaml(
                        path, project_root=root
                    )
                    cache_key = _description_hash(
                        description,
                        self.compiler,
                        self.fflags,
                        self.ldflags,
                    )
                    output_root = self.cache_dir / cache_key
                    manifest = output_root / description.name / "device.json"
                    if not manifest.is_file():
                        self.cache_dir.mkdir(parents=True, exist_ok=True)
                        lock_path = self.cache_dir / f"{cache_key}.lock"
                        with lock_path.open("w") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            if not manifest.is_file():
                                output_root.mkdir(
                                    parents=True, exist_ok=True
                                )
                                manifest = build_device(
                                    path,
                                    project_root=root,
                                    output_root=output_root,
                                    compiler=self.compiler,
                                    fflags=self.fflags,
                                    ldflags=self.ldflags,
                                )
                    path = manifest
                payload = {
                    "path": str(path.resolve()),
                    "hash": _file_hash(path),
                    "error": None,
                }
            except BaseException as exc:
                payload = {
                    "path": None,
                    "hash": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        payload = self.driver.comm.bcast(payload, root=0)
        assert payload is not None
        if payload["error"]:
            raise DeviceContractError(
                f"cannot prepare physics plugin: {payload['error']}"
            )
        path = Path(payload["path"])
        local_error = None
        if not path.is_file():
            local_error = f"manifest is not visible: {path}"
        elif _file_hash(path) != payload["hash"]:
            local_error = f"manifest hash differs: {path}"
        self._collective_error(local_error, "plugin distribution")
        return path

    @staticmethod
    def _resolve_source(source: str) -> Path:
        candidate = Path(source).expanduser()
        if candidate.exists():
            if candidate.is_dir():
                for name in ("device.json", "device.yaml"):
                    path = candidate / name
                    if path.is_file():
                        return path.resolve()
                raise FileNotFoundError(
                    f"plugin directory contains no device.json/device.yaml: "
                    f"{candidate}"
                )
            return candidate.resolve()
        roots = [
            Path(item)
            for value in os.environ.get("PYCAM_SIMA_PLUGIN_PATH", "").split(
                os.pathsep
            )
            if (item := value.strip())
        ]
        for root in roots:
            for path in (
                root / source / "device.json",
                root / source / "device.yaml",
                root / source,
            ):
                if path.is_file():
                    return path.resolve()
        for entrypoint in importlib_metadata.entry_points(
            group="pycam_sima.physics"
        ):
            if entrypoint.name != source:
                continue
            provider = entrypoint.load()
            value = provider() if callable(provider) else provider
            path = Path(value).expanduser()
            if path.is_dir():
                return PhysicsPluginManager._resolve_source(str(path))
            if path.is_file():
                return path.resolve()
            raise FileNotFoundError(
                f"physics entry point {source!r} returned missing path "
                f"{path}"
            )
        raise FileNotFoundError(f"cannot find physics plugin {source!r}")

    @staticmethod
    def _default_process(device: FortranDevice) -> str:
        candidates = [
            name for name, endpoint in device.processes.items()
            if ":" not in name and endpoint == "run"
        ]
        if device.name in candidates:
            return device.name
        if len(candidates) == 1:
            return candidates[0]
        raise DeviceContractError(
            f"device {device.name!r} needs explicit placements; "
            f"run processes are {candidates}"
        )

    def _preflight_device(
        self,
        device: FortranDevice,
        placements: tuple[SchemePlacement, ...],
    ) -> None:
        error: str | None = None
        try:
            if device.name in self.driver.backend.devices.devices:
                raise DeviceContractError(
                    f"device name {device.name!r} is already registered"
                )
            duplicates = (
                set(device.processes)
                & self.driver.backend.devices.process_names
            )
            if duplicates:
                raise DeviceContractError(
                    f"device processes already exist: {sorted(duplicates)}"
                )
            for placement in placements:
                if placement.process not in device.processes:
                    raise DeviceContractError(
                        f"device {device.name!r} has no process "
                        f"{placement.process!r}"
                    )
                if placement.before is not None:
                    self.driver.scheme_plan.scheme(
                        placement.before, group=placement.group
                    )
                if placement.after is not None:
                    self.driver.scheme_plan.scheme(
                        placement.after, group=placement.group
                    )
            device._ensure_abi()
            _validate_elf(device.library_path)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error, "plugin preflight")

    def _infer_variables(
        self, device: FortranDevice
    ) -> tuple[VariableSpec, ...]:
        rows: dict[str, VariableSpec] = {}
        intents: dict[str, set[str]] = {}
        for endpoint in device.entrypoints.values():
            for argument in endpoint["arguments"]:
                binding = argument["binding"]
                source = binding["source"]
                if source not in {"standard_name", "field"}:
                    continue
                if argument["dtype"] == "opaque":
                    continue
                standard_name = (
                    str(binding["name"]).lower()
                    if source == "standard_name"
                    else None
                )
                name = (
                    f"ccpp_{_field_token(standard_name)}"
                    if standard_name is not None
                    else str(binding["name"])
                )
                dimensions = tuple(
                    str(item)
                    if str(item).isdigit()
                    else str(device.dimension_bindings[item])
                    for item in argument["dimensions"]
                )
                dtype = str(argument["dtype"])
                if dtype == "character":
                    dtype = "S512"
                current = VariableSpec(
                    name=name,
                    standard_name=standard_name,
                    dtype=dtype,
                    dimensions=dimensions,
                    units=str(argument.get("units", "1")),
                    intent=str(argument["intent"]),
                )
                key = standard_name or name
                if key in rows:
                    previous = rows[key]
                    signature = (
                        previous.dtype,
                        previous.dimensions,
                        previous.units.lower(),
                    )
                    candidate = (
                        current.dtype,
                        current.dimensions,
                        current.units.lower(),
                    )
                    if signature != candidate:
                        raise DeviceContractError(
                            f"plugin field {key!r} has incompatible ABI "
                            f"variants {signature} and {candidate}"
                        )
                else:
                    rows[key] = current
                intents.setdefault(key, set()).add(str(argument["intent"]))
        result: list[VariableSpec] = []
        for key, item in rows.items():
            seen = intents[key]
            intent = (
                "inout"
                if "inout" in seen or {"in", "out"} <= seen
                else ("out" if "out" in seen else "in")
            )
            result.append(
                VariableSpec(
                    name=item.name,
                    standard_name=item.standard_name,
                    dtype=item.dtype,
                    dimensions=item.dimensions,
                    units=item.units,
                    intent=intent,
                )
            )
        return tuple(result)

    @staticmethod
    def _merge_variables(
        explicit: tuple[VariableSpec, ...],
        inferred: tuple[VariableSpec, ...],
    ) -> tuple[VariableSpec, ...]:
        merged = {
            item.standard_name or item.name: item for item in inferred
        }
        for item in explicit:
            merged[item.standard_name or item.name] = item
        return tuple(merged[key] for key in sorted(merged))

    def _variable_is_provided(self, spec: VariableSpec) -> bool:
        try:
            if spec.standard_name is not None:
                name = self.driver.pool.ccpp_field_name(spec.standard_name)
            else:
                name = spec.name
            contract = self.driver.pool.contract(name)
            values = self.driver.pool.get(name)
        except KeyError:
            return False
        expected = spec.contract()
        if (
            values.dtype != np.dtype(expected.dtype)
            or values.shape
            != expected.shape(self.driver.pool.dimensions)
            or _normalized_units(contract.units)
            != _normalized_units(expected.units)
        ):
            raise DeviceContractError(
                f"existing field {name!r} is incompatible with plugin "
                f"standard name {spec.standard_name or spec.name!r}"
            )
        return True

    def _preflight_variables(
        self,
        variables: Iterable[VariableSpec],
        initial_values: Mapping[str, Any],
    ) -> None:
        error: str | None = None
        try:
            for item in variables:
                item.contract().shape(self.driver.pool.dimensions)
                np.dtype(item.dtype)
                initial = _initial_value(initial_values, item)
                if item.intent in {"in", "inout"} and initial is _UNSET:
                    raise DeviceContractError(
                        f"new input field {item.standard_name or item.name!r} "
                        "requires an initial value"
                    )
                if initial is not _UNSET:
                    probe = np.zeros(
                        item.contract().shape(self.driver.pool.dimensions),
                        dtype=item.dtype,
                        order="F",
                    )
                    np.copyto(probe, initial, casting="same_kind")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._collective_error(error, "plugin variable preflight")

    def _run_lifecycle(
        self,
        device: FortranDevice,
        placements: tuple[SchemePlacement, ...],
        phase: str,
    ) -> None:
        seen: set[str] = set()
        for placement in placements:
            process = f"{placement.process}:{phase}"
            if process in device.processes and process not in seen:
                device.invoke_process(process, self.driver.pool)
                self.driver.backend.call_count += 1
                seen.add(process)

    def _require_boundary(self) -> None:
        if self.driver.pool is None:
            raise StateOwnershipError(
                "initialize the driver before changing runtime state"
            )
        if getattr(self.driver, "_native_call_depth", 0):
            raise StateOwnershipError(
                "physics and variables cannot change inside a native call"
            )
        cursor = self.driver.execution_cursor
        cursors = self.driver.comm.allgather(cursor)
        if any(item != cursor for item in cursors):
            raise StateOwnershipError(
                f"MPI ranks are at different execution boundaries: {cursors}"
            )

    def _assert_schema_consistent(self) -> None:
        payload = [
            self.driver.pool.contracts[name].machine_record()
            for name in sorted(self.driver.pool.contracts)
        ]
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        digests = self.driver.comm.allgather(digest)
        if len(set(digests)) != 1:
            raise StateOwnershipError(
                f"StatePool schema differs across MPI ranks: {digests}"
            )

    def _collective_error(
        self, error: str | None, operation: str
    ) -> None:
        errors = self.driver.comm.allgather(error)
        failures = [
            f"rank {rank}: {message}"
            for rank, message in enumerate(errors)
            if message is not None
        ]
        if failures:
            raise DeviceContractError(
                f"{operation} failed collectively: " + "; ".join(failures)
            )


def _field_token(standard_name: str | None) -> str:
    assert standard_name is not None
    return re.sub(r"[^a-z0-9_]+", "_", standard_name.lower()).strip("_")


def _normalized_units(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _initial_value(
    values: Mapping[str, Any], variable: VariableSpec
) -> Any:
    for key in (variable.name, variable.standard_name):
        if key is not None and key in values:
            return values[key]
    return _UNSET


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _description_hash(
    description: DeviceDescription,
    compiler: str,
    fflags: tuple[str, ...],
    ldflags: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(PLUGIN_SCHEMA_VERSION).encode())
    digest.update(str(Path(compiler).resolve()).encode())
    digest.update("\0".join(fflags).encode())
    digest.update("\0".join(ldflags).encode())
    paths = (
        description.path,
        *description.sources,
        *description.metadata,
        *description.providers.values(),
    )
    for path in sorted(set(paths), key=str):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _infer_project_root(descriptor: Path) -> Path:
    for parent in (descriptor.parent, *descriptor.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return descriptor.parent
