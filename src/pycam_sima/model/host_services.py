"""Python implementations of CCPP services that are not numerical kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .device_catalog import DeviceCatalog, SchemeCatalogEntry
from .errors import DeviceContractError, MissingKernelError


_INTERNAL = {"ccpp_error_code", "ccpp_error_message", "scheme_name"}
_HISTORY_MODULES = {"cam_history", "cam_history_support"}


def is_python_history_scheme(entry: SchemeCatalogEntry) -> bool:
    """Return whether this is a pure history sink with no state outputs."""

    blocker_modules = {
        item.split(":", 1)[1]
        for item in entry.blockers
        if item.startswith("host_framework_module:")
    }
    if not blocker_modules or not blocker_modules <= _HISTORY_MODULES:
        return False
    if any(
        not item.startswith("host_framework_module:")
        for item in entry.blockers
    ):
        return False
    return all(
        argument.intent == "in"
        for endpoint in entry.entrypoints
        for argument in endpoint.arguments
        if argument.standard_name not in _INTERNAL
    )


@dataclass(frozen=True, slots=True)
class HistoryObservation:
    standard_name: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | int | None
    maximum: float | int | None
    mean: float | None


@dataclass(frozen=True, slots=True)
class HostServiceEvent:
    scheme: str
    phase: str
    observations: tuple[HistoryObservation, ...]
    unavailable_fields: tuple[str, ...]

    def machine_record(self) -> dict[str, Any]:
        return asdict(self)


class PythonHistoryService:
    """Replace CAM registration/outfld calls with Python observations."""

    def __init__(self, entry: SchemeCatalogEntry):
        if not is_python_history_scheme(entry):
            raise DeviceContractError(
                f"{entry.name!r} is not a pure history service scheme"
            )
        self.entry = entry
        self.events: list[HostServiceEvent] = []
        self._process_phases = {
            **({entry.name: "run"} if "run" in entry.lifecycle else {}),
            **{
                f"{entry.name}:{phase}": phase
                for phase in entry.lifecycle
            },
        }

    @property
    def process_names(self) -> frozenset[str]:
        return frozenset(self._process_phases)

    def invoke(self, process: str, pool: Any) -> None:
        try:
            phase = self._process_phases[process]
        except KeyError as exc:
            raise MissingKernelError(
                f"history service {self.entry.name!r} does not provide "
                f"{process!r}"
            ) from exc
        endpoint = next(
            item for item in self.entry.entrypoints if item.phase == phase
        )
        observations: list[HistoryObservation] = []
        unavailable: list[str] = []
        seen: set[str] = set()
        for argument in endpoint.arguments:
            standard_name = argument.standard_name
            if standard_name in _INTERNAL or standard_name in seen:
                continue
            seen.add(standard_name)
            try:
                values = np.asarray(pool.get_ccpp(standard_name))
            except KeyError:
                # Dimension controls and opaque metadata are recorded as
                # unavailable rather than silently invented.
                unavailable.append(standard_name)
                continue
            if values.dtype.kind in {"b", "i", "u", "f"} and values.size:
                minimum: float | int | None = values.min().item()
                maximum: float | int | None = values.max().item()
                mean: float | None = float(values.mean())
            else:
                minimum = maximum = mean = None
            observations.append(
                HistoryObservation(
                    standard_name,
                    tuple(values.shape),
                    values.dtype.str,
                    minimum,
                    maximum,
                    mean,
                )
            )
        self.events.append(
            HostServiceEvent(
                self.entry.name,
                phase,
                tuple(observations),
                tuple(unavailable),
            )
        )


class HostServiceRegistry:
    """Route non-numerical suite processes to explicit Python services."""

    def __init__(self, services: Iterable[Any] = ()):
        self.services: list[Any] = []
        self._processes: dict[str, Any] = {}
        for service in services:
            self.register(service)

    @classmethod
    def from_catalog(
        cls,
        catalog: DeviceCatalog,
        *,
        suite: str | None = None,
    ) -> "HostServiceRegistry":
        services = []
        for entry in catalog.entries.values():
            if suite is not None and not any(
                item.suite == suite for item in entry.occurrences
            ):
                continue
            if is_python_history_scheme(entry):
                services.append(PythonHistoryService(entry))
        return cls(services)

    def register(self, service: Any) -> None:
        duplicates = set(service.process_names) & set(self._processes)
        if duplicates:
            raise DeviceContractError(
                f"duplicate host-service processes: {sorted(duplicates)}"
            )
        self.services.append(service)
        for process in service.process_names:
            self._processes[process] = service

    @property
    def process_names(self) -> frozenset[str]:
        return frozenset(self._processes)

    def invoke(self, process: str, pool: Any) -> None:
        try:
            service = self._processes[process]
        except KeyError as exc:
            raise MissingKernelError(
                f"no Python host service provides {process!r}"
            ) from exc
        service.invoke(process, pool)

    def events(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            event.machine_record()
            for service in self.services
            for event in getattr(service, "events", ())
        )
