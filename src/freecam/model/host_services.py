"""Python implementations of CCPP services that are not numerical kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .device_catalog import DeviceCatalog, SchemeCatalogEntry
from .errors import DeviceContractError, MissingKernelError


_INTERNAL = {"ccpp_error_code", "ccpp_error_message", "scheme_name"}
_HISTORY_MODULES = {"cam_history", "cam_history_support"}
_PYTHON_REGISTRY_SCHEMES = {"rrtmgp_constituents"}


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


def is_python_host_service_scheme(entry: SchemeCatalogEntry) -> bool:
    """Return whether Python intentionally owns this non-kernel service."""

    return (
        is_python_history_scheme(entry)
        or entry.name in _PYTHON_REGISTRY_SCHEMES
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
                field_name = pool.ccpp_field_name(standard_name)
                if not pool.is_initialized(field_name):
                    unavailable.append(standard_name)
                    continue
                values = np.asarray(pool.get(field_name))
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


class PythonRRTMGPConstituentService:
    """Implement CCPP constituent registration/index routing in Python."""

    process_names = frozenset(
        {
            "rrtmgp_constituents",
            "rrtmgp_constituents:register",
            "rrtmgp_constituents:run",
        }
    )

    @staticmethod
    def _strings(pool: Any, standard_name: str) -> tuple[str, ...]:
        field = pool.ccpp_field_name(standard_name)
        values = np.asarray(pool.get(field))
        return tuple(
            bytes(value).decode("utf-8", errors="strict").strip().strip("\0")
            if isinstance(value, (bytes, np.bytes_))
            else str(value).strip()
            for value in values.reshape(-1)
        )

    @staticmethod
    def _configured_constituents(pool: Any) -> dict[str, int]:
        return {
            str(name).strip().lower(): index
            for index, name in enumerate(pool.constituent_names)
        }

    def _register(self, pool: Any) -> None:
        entries = self._strings(
            pool,
            "sources_of_radiatively_active_gases_for_climate_calculation",
        )
        configured = self._configured_constituents(pool)
        for text in entries:
            if not text:
                continue
            parts = tuple(part.strip() for part in text.split(":"))
            if len(parts) != 3 or parts[0] not in {"A", "N", "Z"}:
                raise DeviceContractError(
                    "rad_climate entries must use "
                    "'flag:long_name:standard_name'; got "
                    f"{text!r}"
                )
            standard_name = parts[2].lower()
            if standard_name not in configured:
                raise DeviceContractError(
                    f"rad_climate registers {parts[2]!r}, but ModelConfig."
                    "constituent_names does not contain it; constituents "
                    "must be known before StatePool allocation"
                )

    def _run(self, pool: Any) -> None:
        gases = self._strings(
            pool,
            "list_of_active_gases_for_RRTMGP",
        )
        configured = self._configured_constituents(pool)
        source = np.asarray(
            pool.get(pool.ccpp_field_name("ccpp_constituents"))
        )
        target = pool.get(
            pool.ccpp_field_name(
                "radiatively_active_gas_mass_mixing_ratios_wrt_dry_air"
            )
        )
        target[...] = np.float64(0.0)
        for gas_index, gas in enumerate(gases):
            key = (
                "water_vapor"
                if gas.strip().upper() == "H2O"
                else gas.strip().lower()
            )
            constituent_index = configured.get(key)
            if constituent_index is not None:
                target[:, :, gas_index] = source[:, :, constituent_index]

    def invoke(self, process: str, pool: Any) -> None:
        if process == "rrtmgp_constituents:register":
            self._register(pool)
            return
        if process in {
            "rrtmgp_constituents",
            "rrtmgp_constituents:run",
        }:
            self._run(pool)
            return
        raise MissingKernelError(
            f"RRTMGP constituent service does not provide {process!r}"
        )


class PythonMUSICARegisterService:
    """Orchestrate MUSICA's dynamic registry into StatePool process state."""

    process_names = frozenset({"musica_ccpp:register"})

    def __init__(self, devices: Any):
        self.devices = devices

    def invoke(self, process: str, pool: Any) -> None:
        if process != "musica_ccpp:register":
            raise MissingKernelError(
                f"MUSICA register service does not provide {process!r}"
            )
        self.devices.invoke_host_entrypoint(
            "musica_ccpp", "register", pool
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
        processes: Iterable[str] | None = None,
        devices: Any | None = None,
    ) -> "HostServiceRegistry":
        selected_processes = (
            None
            if processes is None
            else {str(name).lower() for name in processes}
        )
        services = []
        for entry in catalog.entries.values():
            if (
                selected_processes is not None
                and entry.name not in selected_processes
            ):
                continue
            if suite is not None and not any(
                item.suite == suite for item in entry.occurrences
            ):
                continue
            if is_python_history_scheme(entry):
                services.append(PythonHistoryService(entry))
            elif entry.name == "rrtmgp_constituents":
                services.append(PythonRRTMGPConstituentService())
            elif entry.name == "musica_ccpp":
                if devices is None:
                    raise DeviceContractError(
                        "MUSICA host registration requires DeviceRegistry"
                    )
                services.append(PythonMUSICARegisterService(devices))
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
