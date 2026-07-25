"""Composable routing for generated devices and Python host processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .ccpp_suite import SuiteScheme
from .errors import MissingKernelError
from .phases import (
    apply_tendency_of_air_temperature,
    calc_dry_air_ideal_gas_density,
    calc_exner,
    check_energy_chng,
    check_energy_scaling,
    check_energy_scaling_before_coupler,
    check_energy_zero_fluxes,
    dry_to_wet_cloud_liquid_water,
    dry_to_wet_rain,
    dry_to_wet_water_vapor,
    dycore_energy_consistency_adjust,
    geopotential_temp,
    kessler_diagnostics,
    potential_temperature_to_temperature,
    qneg,
    sima_state_diagnostics,
    sima_tend_diagnostics,
    temp_to_potential_temp,
    thermo_water_update,
    wet_to_dry_cloud_liquid_water,
    wet_to_dry_rain,
    wet_to_dry_water_vapor,
)
from .ccpp_suite import PHYSICS_AFTER_COUPLER, PHYSICS_BEFORE_COUPLER


class ProcessRouter:
    """Route one suite occurrence without embedding suite order in the Driver."""

    def __init__(
        self,
        *,
        devices: Any,
        native_invoke: Callable[[str, Any], None],
        host_services: Any | None = None,
        host_handlers: Mapping[str, Callable[[Any], None]] | None = None,
    ) -> None:
        self.devices = devices
        self.native_invoke = native_invoke
        self.host_services = host_services
        self.host_handlers = dict(host_handlers or {})

    @property
    def process_names(self) -> frozenset[str]:
        names = set(self.devices.process_names)
        names.update(
            name.rsplit(".", 1)[-1] for name in self.host_handlers
        )
        if self.host_services is not None:
            names.update(self.host_services.process_names)
        return frozenset(names)

    def invoke(self, scheme: SuiteScheme, pool: Any) -> str:
        """Invoke the best declared provider and report its provider kind."""

        qualified = f"{scheme.source_group}.{scheme.name}"
        handler = self.host_handlers.get(qualified)
        if handler is None:
            handler = self.host_handlers.get(scheme.name)
        if handler is not None:
            handler(pool)
            return "python-host-process"
        if self.devices.has_process(scheme.name):
            self.native_invoke(scheme.name, pool)
            return "fortran-device"
        if (
            self.host_services is not None
            and scheme.name in self.host_services.process_names
        ):
            self.host_services.invoke(scheme.name, pool)
            return "python-host-service"
        raise MissingKernelError(
            f"suite process {scheme.name!r} has no generated device or "
            f"Python host provider; available process count="
            f"{len(self.process_names)}"
        )

    def provider_for(self, scheme: SuiteScheme) -> str | None:
        """Return the provider selected for one source-qualified occurrence."""

        qualified = f"{scheme.source_group}.{scheme.name}"
        if qualified in self.host_handlers or scheme.name in self.host_handlers:
            return "python-host-process"
        if self.devices.has_process(scheme.name):
            return "fortran-device"
        if (
            self.host_services is not None
            and scheme.name in self.host_services.process_names
        ):
            return "python-host-service"
        return None

    def describe(self, schemes: Any) -> tuple[dict[str, Any], ...]:
        """Describe provider coverage without executing numerical code."""

        return tuple(
            {
                "key": scheme.key,
                "name": scheme.name,
                "provider": self.provider_for(scheme),
            }
            for scheme in schemes
        )


def cam_se_fvm_host_processes(
    backend: Any,
) -> dict[str, Callable[[Any], None]]:
    """Return the Python providers required by the CAM SE/FVM v1 component.

    These are component services, not a Kessler execution plan. Any suite may
    use them by standard CCPP process name. Explicit component providers take
    precedence over a catalog device when they implement CAM host semantics.
    """

    before = PHYSICS_BEFORE_COUPLER
    after = PHYSICS_AFTER_COUPLER
    return {
        f"{before}.calc_exner": calc_exner,
        f"{before}.temp_to_potential_temp": temp_to_potential_temp,
        f"{before}.calc_dry_air_ideal_gas_density": (
            calc_dry_air_ideal_gas_density
        ),
        f"{before}.wet_to_dry_water_vapor": wet_to_dry_water_vapor,
        f"{before}.wet_to_dry_cloud_liquid_water": (
            wet_to_dry_cloud_liquid_water
        ),
        f"{before}.wet_to_dry_rain": wet_to_dry_rain,
        f"{before}.potential_temp_to_temp": (
            potential_temperature_to_temperature
        ),
        f"{before}.dry_to_wet_water_vapor": dry_to_wet_water_vapor,
        f"{before}.dry_to_wet_cloud_liquid_water": (
            dry_to_wet_cloud_liquid_water
        ),
        f"{before}.dry_to_wet_rain": dry_to_wet_rain,
        f"{before}.qneg": qneg,
        f"{before}.geopotential_temp": lambda pool: geopotential_temp(
            pool, backend
        ),
        f"{before}.check_energy_zero_fluxes": check_energy_zero_fluxes,
        f"{before}.check_energy_scaling": (
            check_energy_scaling_before_coupler
        ),
        f"{before}.check_energy_chng": check_energy_chng,
        f"{before}.sima_state_diagnostics": sima_state_diagnostics,
        f"{before}.kessler_diagnostics": kessler_diagnostics,
        f"{after}.thermo_water_update": thermo_water_update,
        f"{after}.check_energy_scaling": check_energy_scaling,
        f"{after}.dycore_energy_consistency_adjust": (
            dycore_energy_consistency_adjust
        ),
        f"{after}.apply_tendency_of_air_temperature": (
            apply_tendency_of_air_temperature
        ),
        f"{after}.sima_tend_diagnostics": sima_tend_diagnostics,
    }
