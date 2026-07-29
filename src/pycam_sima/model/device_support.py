"""End-to-end support matrix for every active CAM-SIMA CCPP scheme."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .device_catalog import DeviceCatalog
from .device_codegen import DeviceDescription, resolve_source_closure
from .errors import DeviceBuildError
from .host_services import is_python_host_service_scheme
from .processes import CAM_SE_FVM_HOST_PROCESS_KEYS


SUPPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class RuntimeOccurrenceSupport:
    suite: str
    group: str
    order: int
    provider: str | None
    ready: bool

    def machine_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SchemeSupport:
    name: str
    suites: tuple[str, ...]
    descriptor: str
    connector_generated: bool
    status: str
    manifest: str | None
    blockers: tuple[str, ...]
    runtime_ready: bool
    runtime_provider_kinds: tuple[str, ...]
    runtime_occurrences: tuple[RuntimeOccurrenceSupport, ...]

    def machine_record(self) -> dict[str, Any]:
        return asdict(self)


class DeviceSupportMatrix:
    """Classify generated, buildable, and host-service-dependent schemes."""

    def __init__(
        self,
        catalog: DeviceCatalog,
        records: tuple[SchemeSupport, ...],
    ) -> None:
        self.catalog = catalog
        self.records = records

    @classmethod
    def discover(
        cls,
        project_root: str | Path,
        *,
        descriptor_root: str | Path | None = None,
        build_root: str | Path | None = None,
    ) -> "DeviceSupportMatrix":
        root = Path(project_root).resolve()
        catalog = DeviceCatalog.discover(root)
        descriptions = Path(
            descriptor_root or root / "devices/generated"
        ).resolve()
        builds = Path(
            build_root or root / "build/catalog_devices"
        ).resolve()
        records: list[SchemeSupport] = []
        for name, entry in sorted(catalog.entries.items()):
            descriptor = descriptions / name / "device.yaml"
            manifest = builds / name / "device.json"
            blockers = list(entry.blockers)
            description: DeviceDescription | None = None
            description_error: str | None = None
            if descriptor.is_file():
                try:
                    description = DeviceDescription.from_yaml(
                        descriptor, project_root=root
                    )
                except DeviceBuildError as exc:
                    description_error = str(exc)
            if description is not None and description.host_entrypoints:
                host_tables = {
                    endpoint.table
                    for endpoint in entry.entrypoints
                    if endpoint.phase in description.host_entrypoints
                }
                blockers = [
                    blocker
                    for blocker in blockers
                    if not any(
                        f"{table}." in blocker for table in host_tables
                    )
                ]
            remaining_unresolved = set(entry.unresolved_modules)
            if description is not None:
                remaining_unresolved -= set(description.external_modules)
            if not descriptor.is_file():
                status = "connector_missing"
            elif description_error is not None:
                status = "dependency_provider_required"
                blockers.append(f"descriptor:{description_error}")
            elif is_python_host_service_scheme(entry):
                status = "python_host_service_ready"
                blockers = []
            elif blockers:
                status = "abi_or_host_service_required"
            elif remaining_unresolved:
                status = "external_source_required"
                blockers.extend(
                    f"unresolved_module:{module}"
                    for module in sorted(remaining_unresolved)
                )
            else:
                try:
                    resolve_source_closure(
                        description
                    )
                except DeviceBuildError as exc:
                    status = "dependency_provider_required"
                    blockers.append(f"dependency_closure:{exc}")
                else:
                    status = (
                        "ready"
                        if manifest.is_file()
                        else "build_required"
                    )
            runtime_occurrences: list[RuntimeOccurrenceSupport] = []
            for occurrence in entry.occurrences:
                qualified = f"{occurrence.group}.{entry.name}"
                if qualified in CAM_SE_FVM_HOST_PROCESS_KEYS:
                    provider = "python-host-process"
                elif is_python_host_service_scheme(entry):
                    provider = "python-host-service"
                elif manifest.is_file():
                    provider = "fortran-device"
                else:
                    provider = None
                runtime_occurrences.append(
                    RuntimeOccurrenceSupport(
                        suite=occurrence.suite,
                        group=occurrence.group,
                        order=occurrence.order,
                        provider=provider,
                        ready=provider is not None,
                    )
                )
            records.append(
                SchemeSupport(
                    name=name,
                    suites=tuple(
                        sorted(
                            {
                                occurrence.suite
                                for occurrence in entry.occurrences
                            }
                        )
                    ),
                    descriptor=str(
                        descriptor.relative_to(root)
                        if descriptor.is_relative_to(root)
                        else descriptor
                    ),
                    connector_generated=descriptor.is_file(),
                    status=status,
                    manifest=(
                        str(
                            manifest.relative_to(root)
                            if manifest.is_relative_to(root)
                            else manifest
                        )
                        if manifest.is_file()
                        else None
                    ),
                    blockers=tuple(blockers),
                    runtime_ready=all(
                        item.ready for item in runtime_occurrences
                    ),
                    runtime_provider_kinds=tuple(
                        sorted(
                            {
                                item.provider
                                for item in runtime_occurrences
                                if item.provider is not None
                            }
                        )
                    ),
                    runtime_occurrences=tuple(runtime_occurrences),
                )
            )
        return cls(catalog, tuple(records))

    def summary(self) -> dict[str, Any]:
        statuses = Counter(item.status for item in self.records)
        native_ready = statuses["ready"]
        python_service_ready = statuses["python_host_service_ready"]
        runtime_occurrences = tuple(
            occurrence
            for record in self.records
            for occurrence in record.runtime_occurrences
        )
        runtime_provider_counts = Counter(
            occurrence.provider or "unresolved"
            for occurrence in runtime_occurrences
        )
        suite_runtime: dict[str, dict[str, Any]] = {}
        for suite in sorted(
            {
                occurrence.suite
                for occurrence in runtime_occurrences
            }
        ):
            selected = tuple(
                occurrence
                for occurrence in runtime_occurrences
                if occurrence.suite == suite
            )
            provider_counts = Counter(
                occurrence.provider or "unresolved"
                for occurrence in selected
            )
            suite_runtime[suite] = {
                "occurrence_count": len(selected),
                "ready_occurrence_count": sum(
                    occurrence.ready for occurrence in selected
                ),
                "unresolved_occurrence_count": sum(
                    not occurrence.ready for occurrence in selected
                ),
                "provider_counts": dict(sorted(provider_counts.items())),
                "ready": all(occurrence.ready for occurrence in selected),
            }
        return {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "source_revision": self.catalog.source_revision,
            "suite_count": len(self.catalog.suites),
            "scheme_count": len(self.records),
            "connectors_generated": sum(
                item.connector_generated for item in self.records
            ),
            "native_device_ready": native_ready,
            "python_host_service_ready": python_service_ready,
            "executable_connector_count": (
                native_ready + python_service_ready
            ),
            "external_service_required": (
                len(self.records) - native_ready - python_service_ready
            ),
            "status_counts": dict(sorted(statuses.items())),
            "runtime_executable_scheme_count": sum(
                record.runtime_ready for record in self.records
            ),
            "runtime_unresolved_scheme_count": sum(
                not record.runtime_ready for record in self.records
            ),
            "runtime_occurrence_count": len(runtime_occurrences),
            "runtime_ready_occurrence_count": sum(
                occurrence.ready for occurrence in runtime_occurrences
            ),
            "runtime_unresolved_occurrence_count": sum(
                not occurrence.ready for occurrence in runtime_occurrences
            ),
            "runtime_provider_counts": dict(
                sorted(runtime_provider_counts.items())
            ),
            "suite_runtime": suite_runtime,
        }

    def machine_record(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "schemes": [
                item.machine_record() for item in self.records
            ],
        }
