"""End-to-end support matrix for every active CAM-SIMA CCPP scheme."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .device_catalog import DeviceCatalog
from .device_codegen import DeviceDescription, resolve_source_closure
from .errors import DeviceBuildError
from .host_services import is_python_history_scheme


SUPPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SchemeSupport:
    name: str
    suites: tuple[str, ...]
    descriptor: str
    connector_generated: bool
    status: str
    manifest: str | None
    blockers: tuple[str, ...]

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
            if not descriptor.is_file():
                status = "connector_missing"
            elif is_python_history_scheme(entry):
                status = "python_host_service_ready"
                blockers = []
            elif blockers:
                status = "abi_or_host_service_required"
            elif entry.unresolved_modules:
                status = "external_source_required"
                blockers.extend(
                    f"unresolved_module:{module}"
                    for module in entry.unresolved_modules
                )
            else:
                try:
                    resolve_source_closure(
                        DeviceDescription.from_yaml(
                            descriptor, project_root=root
                        )
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
                )
            )
        return cls(catalog, tuple(records))

    def summary(self) -> dict[str, Any]:
        statuses = Counter(item.status for item in self.records)
        native_ready = statuses["ready"]
        python_service_ready = statuses["python_host_service_ready"]
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
        }

    def machine_record(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "schemes": [
                item.machine_record() for item in self.records
            ],
        }
