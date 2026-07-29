"""Composable routing for generated devices and Python host processes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any

from .ccpp_suite import SuiteScheme
from .errors import MissingKernelError
from .phases import (
    apply_tendency_of_air_temperature,
    calc_dry_air_ideal_gas_density,
    calc_exner,
    check_energy_chng,
    check_energy_gmean,
    check_energy_timestep_initial,
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
from .scientific_data import (
    solar_irradiance_data_finalize,
    solar_irradiance_data_initialize,
    solar_irradiance_data_register,
    solar_irradiance_data_timestep_initial,
)


CAM_SE_FVM_HOST_PROCESS_KEYS = frozenset(
    {
        f"{PHYSICS_BEFORE_COUPLER}.qneg",
        f"{PHYSICS_BEFORE_COUPLER}.check_energy_chng",
        f"{PHYSICS_BEFORE_COUPLER}.check_energy_gmean",
        f"{PHYSICS_BEFORE_COUPLER}.sima_state_diagnostics",
        f"{PHYSICS_BEFORE_COUPLER}.kessler_diagnostics",
        f"{PHYSICS_AFTER_COUPLER}.thermo_water_update",
        f"{PHYSICS_AFTER_COUPLER}.qneg",
        f"{PHYSICS_AFTER_COUPLER}.check_energy_chng",
        f"{PHYSICS_AFTER_COUPLER}.sima_tend_diagnostics",
        # Lifecycle for Python-owned component services.  The qneg provider
        # does not retain native warning buffers; its lifecycle is therefore
        # an explicit numerical no-op.  check_energy lifecycle state is
        # represented by Python-owned StatePool fields.
        f"{PHYSICS_BEFORE_COUPLER}.qneg:initialize",
        f"{PHYSICS_BEFORE_COUPLER}.qneg:timestep_final",
        f"{PHYSICS_BEFORE_COUPLER}.qneg:finalize",
        f"{PHYSICS_AFTER_COUPLER}.qneg:initialize",
        f"{PHYSICS_AFTER_COUPLER}.qneg:timestep_final",
        f"{PHYSICS_AFTER_COUPLER}.qneg:finalize",
        f"{PHYSICS_BEFORE_COUPLER}.check_energy_chng:initialize",
        f"{PHYSICS_BEFORE_COUPLER}.check_energy_chng:timestep_initial",
        f"{PHYSICS_AFTER_COUPLER}.check_energy_chng:initialize",
        f"{PHYSICS_AFTER_COUPLER}.check_energy_chng:timestep_initial",
        f"{PHYSICS_BEFORE_COUPLER}.solar_irradiance_data:register",
        f"{PHYSICS_BEFORE_COUPLER}.solar_irradiance_data:initialize",
        f"{PHYSICS_BEFORE_COUPLER}.solar_irradiance_data:timestep_initial",
        f"{PHYSICS_BEFORE_COUPLER}.solar_irradiance_data:finalize",
        f"{PHYSICS_BEFORE_COUPLER}.rrtmgp_sw_solar_var_setup:initialize",
    }
)


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

        return self.invoke_process(
            scheme.name,
            pool,
            source_group=scheme.source_group,
        )

    def invoke_process(
        self,
        process: str,
        pool: Any,
        *,
        source_group: str | None = None,
    ) -> str:
        """Invoke one run or lifecycle process by its exact CCPP name."""

        qualified = (
            process
            if source_group is None
            else f"{source_group}.{process}"
        )
        debug = bool(os.environ.get("PYCAM_DEBUG_PROCESS_TRACE"))
        debug_rank = (
            int(pool.get("mpi_rank", unsafe=True)) if debug else -1
        )
        if debug:
            print(
                f"PYCAM_PROCESS_BEGIN rank={debug_rank} {qualified}",
                flush=True,
            )

        def completed(provider: str) -> str:
            if debug:
                print(
                    f"PYCAM_PROCESS_DONE rank={debug_rank} {qualified} "
                    f"provider={provider}",
                    flush=True,
                )
            return provider

        handler = self.host_handlers.get(qualified)
        if handler is None:
            handler = self.host_handlers.get(process)
        if handler is not None:
            handler(pool)
            return completed("python-host-process")
        if self.devices.has_process(process):
            self.native_invoke(process, pool)
            return completed("fortran-device")
        if (
            self.host_services is not None
            and process in self.host_services.process_names
        ):
            self.host_services.invoke(process, pool)
            return completed("python-host-service")
        # CCPP suite XML can place an initialization-only scheme in a run
        # group.  The generated CCPP cap invokes its ``*_init`` routine during
        # suite initialization and emits no call for that XML node during the
        # run phase.  Preserve that behavior explicitly; this is not a fallback
        # for a missing numerical kernel because a lifecycle provider must
        # exist for the exact process name.
        lifecycle_prefix = f"{process}:"
        lifecycle_processes = set(self.devices.process_names)
        if self.host_services is not None:
            lifecycle_processes.update(self.host_services.process_names)
        if any(
            name.startswith(lifecycle_prefix)
            for name in lifecycle_processes
        ):
            return completed("lifecycle-only-noop")
        raise MissingKernelError(
            f"suite process {process!r} has no generated device or "
            f"Python host provider; available process count="
            f"{len(self.process_names)}"
        )

    def provider_for_process(
        self,
        process: str,
        *,
        source_group: str | None = None,
    ) -> str | None:
        """Return the provider for an exact run or lifecycle process."""

        qualified = (
            process
            if source_group is None
            else f"{source_group}.{process}"
        )
        if qualified in self.host_handlers or process in self.host_handlers:
            return "python-host-process"
        if self.devices.has_process(process):
            return "fortran-device"
        if (
            self.host_services is not None
            and process in self.host_services.process_names
        ):
            return "python-host-service"
        lifecycle_prefix = f"{process}:"
        lifecycle_processes = set(self.devices.process_names)
        if self.host_services is not None:
            lifecycle_processes.update(self.host_services.process_names)
        if any(
            name.startswith(lifecycle_prefix)
            for name in lifecycle_processes
        ):
            return "lifecycle-only-noop"
        return None

    def provider_for(self, scheme: SuiteScheme) -> str | None:
        """Return the provider selected for one source-qualified occurrence."""

        return self.provider_for_process(
            scheme.name,
            source_group=scheme.source_group,
        )

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
    comm: Any | None = None,
) -> dict[str, Callable[[Any], None]]:
    """Return the Python providers required by the CAM SE/FVM v1 component.

    These are component services, not a Kessler execution plan. Any suite may
    use them by standard CCPP process name. Explicit component providers take
    precedence over a catalog device when they implement CAM host semantics.
    """

    before = PHYSICS_BEFORE_COUPLER
    after = PHYSICS_AFTER_COUPLER
    if comm is None:
        from .comm import SerialComm

        comm = SerialComm()
    no_op = lambda pool: None

    def initialize_solar_variability(pool):
        scaling = bool(
            pool.get_ccpp(
                "do_spectral_scaling_of_solar_irradiance_data"
            ).item()
        )
        if scaling:
            backend.run_phase(
                "rrtmgp_sw_solar_var_setup:initialize", pool
            )

    handlers = {
        f"{before}.qneg": qneg,
        f"{before}.check_energy_chng": (
            lambda pool: check_energy_chng(pool, backend=backend)
        ),
        f"{before}.check_energy_gmean": (
            lambda pool: check_energy_gmean(pool, comm)
        ),
        f"{before}.sima_state_diagnostics": sima_state_diagnostics,
        f"{before}.kessler_diagnostics": kessler_diagnostics,
        f"{after}.thermo_water_update": thermo_water_update,
        f"{after}.qneg": qneg,
        f"{after}.check_energy_chng": (
            lambda pool: check_energy_chng(pool, backend=backend)
        ),
        f"{after}.sima_tend_diagnostics": sima_tend_diagnostics,
        f"{before}.qneg:initialize": no_op,
        f"{before}.qneg:timestep_final": no_op,
        f"{before}.qneg:finalize": no_op,
        f"{after}.qneg:initialize": no_op,
        f"{after}.qneg:timestep_final": no_op,
        f"{after}.qneg:finalize": no_op,
        f"{before}.check_energy_chng:initialize": no_op,
        f"{before}.check_energy_chng:timestep_initial": (
            lambda pool: check_energy_timestep_initial(pool, backend=backend)
        ),
        f"{after}.check_energy_chng:initialize": no_op,
        f"{after}.check_energy_chng:timestep_initial": (
            lambda pool: check_energy_timestep_initial(pool, backend=backend)
        ),
        f"{before}.solar_irradiance_data:register": (
            solar_irradiance_data_register
        ),
        f"{before}.solar_irradiance_data:initialize": (
            solar_irradiance_data_initialize
        ),
        f"{before}.solar_irradiance_data:timestep_initial": (
            solar_irradiance_data_timestep_initial
        ),
        f"{before}.solar_irradiance_data:finalize": (
            solar_irradiance_data_finalize
        ),
        f"{before}.rrtmgp_sw_solar_var_setup:initialize": (
            initialize_solar_variability
        ),
    }
    if frozenset(handlers) != CAM_SE_FVM_HOST_PROCESS_KEYS:
        raise RuntimeError("CAM SE/FVM host-process declaration is inconsistent")
    return handlers
