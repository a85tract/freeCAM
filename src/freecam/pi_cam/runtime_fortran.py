"""Runtime CCPP-device installation for the PI-CAM execution plan."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping

from freecam.model.device_codegen import DeviceDescription, build_device
from freecam.model.devices import FortranDevice
from freecam.model.errors import DeviceContractError
from freecam.model.plugins import (
    _DEFAULT_COMPILER,
    _DEFAULT_FFLAGS,
    _DEFAULT_LDFLAGS,
)

from .errors import PICAMConfigurationError, PICAMStateError
from .plan import PICAMAction


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PICAMFortranProcessSpec:
    source: str
    process: str
    phase: str
    before: str | None = None
    after: str | None = None
    project_root: str | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if (self.before is None) == (self.after is None):
            raise ValueError("provide exactly one of before= or after=")
        for name in ("source", "process", "phase"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")

    def to_payload(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}

    @classmethod
    def from_payload(cls, values: Mapping[str, Any]) -> "PICAMFortranProcessSpec":
        if int(values.get("schema_version", 1)) != 1:
            raise ValueError("unsupported PI-CAM Fortran process schema")
        return cls(
            source=str(values["source"]),
            process=str(values["process"]),
            phase=str(values["phase"]),
            before=None if values.get("before") is None else str(values["before"]),
            after=None if values.get("after") is None else str(values["after"]),
            project_root=(
                None
                if values.get("project_root") is None
                else str(values["project_root"])
            ),
            enabled=bool(values.get("enabled", True)),
        )


@dataclass(slots=True)
class RegisteredPICAMFortranProcess:
    spec: PICAMFortranProcessSpec
    device: FortranDevice
    manifest_hash: str
    library_hash: str


class PICAMFortranProcessRegistry:
    """Build on local-rank 0, then dlopen the same device on every rank."""

    def __init__(self, driver: Any, *, cache_dir: str | Path | None = None) -> None:
        self.driver = driver
        self.cache_dir = Path(
            cache_dir
            or os.environ.get(
                "FREECAM_PLUGIN_CACHE", Path.home() / ".cache/freecam/plugins"
            )
        ).expanduser().resolve()
        self.installed: dict[str, RegisteredPICAMFortranProcess] = {}

    def install(
        self,
        spec: PICAMFortranProcessSpec | Mapping[str, Any],
        *,
        unsafe: bool = False,
    ) -> RegisteredPICAMFortranProcess:
        if not unsafe:
            raise PICAMConfigurationError(
                "installing runtime Fortran physics requires unsafe=True"
            )
        if not isinstance(spec, PICAMFortranProcessSpec):
            spec = PICAMFortranProcessSpec.from_payload(spec)
        self.driver._require_collective_boundary("install a Fortran process")
        manifest_path = self._prepare_manifest(spec)
        local_error: str | None = None
        device: FortranDevice | None = None
        try:
            if spec.process in self.installed:
                raise DeviceContractError(
                    f"Fortran process {spec.process!r} is already installed"
                )
            device = FortranDevice(manifest_path)
            if spec.process not in device.processes:
                raise DeviceContractError(
                    f"device {device.name!r} does not provide process "
                    f"{spec.process!r}"
                )
            if spec.phase not in self.driver.step_plan.phases:
                raise DeviceContractError(
                    f"PI-CAM plan has no phase {spec.phase!r}"
                )
            self.driver.step_plan.select(
                spec.before or spec.after or "", phase=spec.phase
            )
            self._validate_device_fields(device, spec.process)
            # Force dlopen and ABI verification before mutating SuitePlan.
            device._ensure_abi()
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, "Fortran process preflight")
        assert device is not None
        identity = (
            _file_hash(manifest_path),
            _file_hash(device.library_path),
            device.source_hash,
        )
        identities = self.driver.comm.allgather(identity)
        if len(set(identities)) != 1:
            raise DeviceContractError(
                f"Fortran device bytes differ across MPI ranks: {identities}"
            )

        action = PICAMAction(
            name=spec.process,
            phase=spec.phase,
            operation=spec.process,
            kind="runtime_fortran_process",
            native_id=None,
            enabled=spec.enabled,
        )
        local_error = None
        try:
            self.driver.step_plan.add(
                action,
                before=spec.before,
                after=spec.after,
                experimental=True,
            )
            record = RegisteredPICAMFortranProcess(
                spec=spec,
                device=device,
                manifest_hash=identity[0],
                library_hash=identity[1],
            )
            self.installed[spec.process] = record
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = self.driver.comm.allgather(local_error)
        if any(error is not None for error in errors):
            self.installed.pop(spec.process, None)
            try:
                selected = self.driver.step_plan.select(
                    spec.process, phase=spec.phase
                )
            except PICAMConfigurationError:
                selected = None
            if selected is not None and selected.kind == "runtime_fortran_process":
                self.driver.step_plan.remove(
                    spec.process, phase=spec.phase, experimental=True
                )
            raise DeviceContractError(
                f"Fortran process installation rolled back collectively: {errors}"
            )
        self.driver.comm.barrier()
        return self.installed[spec.process]

    def invoke(self, action: PICAMAction) -> None:
        try:
            record = self.installed[action.name]
        except KeyError as exc:
            raise DeviceContractError(
                f"Fortran process {action.name!r} is not installed"
            ) from exc
        before = self.driver.pool.pointer_records()
        local_error: str | None = None
        try:
            record.device.invoke_process(record.spec.process, self.driver.pool)
            self.driver.pool.assert_pointer_stability(before)
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._collective_error(local_error, f"Fortran process {action.name!r}")

    def remove(self, name: str) -> dict[str, Any]:
        self.driver._require_collective_boundary("remove a Fortran process")
        record = self.installed.get(str(name))
        self._collective_error(
            None if record is not None else f"unknown Fortran process {name!r}",
            "Fortran process removal",
        )
        assert record is not None
        self.driver.step_plan.remove(
            record.spec.process, phase=record.spec.phase, experimental=True
        )
        self.installed.pop(record.spec.process)
        self.driver.comm.barrier()
        return {"name": record.spec.process, "removed": True}

    def dependencies(self, field: str) -> tuple[str, ...]:
        canonical = self.driver.pool.canonical_name(field)
        contract = self.driver.pool.contract(canonical)
        tokens = {canonical.lower(), *(alias.lower() for alias in contract.aliases)}
        if contract.standard_name is not None:
            tokens.add(contract.standard_name.lower())
        dependencies: list[str] = []
        for name, record in self.installed.items():
            entrypoint = record.device.processes[name]
            for argument in record.device.entrypoints[entrypoint]["arguments"]:
                binding = argument.get("binding", {})
                if str(binding.get("name", "")).lower() in tokens:
                    dependencies.append(name)
                    break
        return tuple(dependencies)

    def _prepare_manifest(self, spec: PICAMFortranProcessSpec) -> Path:
        payload: dict[str, Any] | None = None
        if self.driver.rank == 0:
            try:
                source = Path(spec.source).expanduser().resolve()
                if source.is_dir():
                    source = next(
                        path
                        for path in (source / "device.json", source / "device.yaml")
                        if path.is_file()
                    )
                if source.name == "device.yaml":
                    root = (
                        Path(spec.project_root).resolve()
                        if spec.project_root is not None
                        else source.parent
                    )
                    description = DeviceDescription.from_yaml(
                        source, project_root=root
                    )
                    identity = sha256()
                    identity.update(source.read_bytes())
                    for path in (*description.sources, *description.metadata):
                        identity.update(path.read_bytes())
                    cache = self.cache_dir / identity.hexdigest()
                    manifest = cache / description.name / "device.json"
                    if not manifest.is_file():
                        cache.mkdir(parents=True, exist_ok=True)
                        with (cache / ".build.lock").open("w") as lock:
                            fcntl.flock(lock, fcntl.LOCK_EX)
                            if not manifest.is_file():
                                manifest = build_device(
                                    source,
                                    project_root=root,
                                    output_root=cache,
                                    compiler=os.environ.get(
                                        "FREECAM_PLUGIN_COMPILER", _DEFAULT_COMPILER
                                    ),
                                    fflags=_DEFAULT_FFLAGS,
                                    ldflags=_DEFAULT_LDFLAGS,
                                )
                    source = manifest
                if not source.is_file() or source.name != "device.json":
                    raise FileNotFoundError(
                        "runtime Fortran source must resolve to device.yaml or device.json"
                    )
                payload = {
                    "path": str(source),
                    "hash": _file_hash(source),
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
        if payload["error"] is not None:
            raise DeviceContractError(str(payload["error"]))
        path = Path(str(payload["path"]))
        local_error = (
            None
            if path.is_file() and _file_hash(path) == payload["hash"]
            else f"device manifest is missing or differs: {path}"
        )
        self._collective_error(local_error, "Fortran device distribution")
        return path

    def _validate_device_fields(self, device: FortranDevice, process: str) -> None:
        entrypoint = device.processes[process]
        for argument in device.entrypoints[entrypoint]["arguments"]:
            if argument["dtype"] in {"character", "opaque"}:
                raise DeviceContractError(
                    "PI-CAM runtime devices currently admit numeric fields and "
                    "dimensions only"
                )
            binding = argument.get("binding", {})
            source = str(binding.get("source"))
            name = str(binding.get("name"))
            if source == "field":
                self.driver.pool.canonical_name(name)
            elif source == "standard_name":
                if name not in self.driver.pool.dimensions:
                    self.driver.pool.ccpp_field_name(name)
            elif source == "dimension":
                if name not in self.driver.pool.dimensions:
                    raise DeviceContractError(f"unknown PI-CAM dimension {name!r}")

    def _collective_error(self, error: str | None, operation: str) -> None:
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
