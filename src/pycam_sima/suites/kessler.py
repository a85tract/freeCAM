from __future__ import annotations

from enum import Enum
from typing import Protocol

from ..state_pool import StatePool


BEFORE_SCHEMES = (
    "calc_exner",
    "temp_to_potential_temp",
    "calc_dry_air_ideal_gas_density",
    "wet_to_dry_water_vapor",
    "wet_to_dry_cloud_liquid_water",
    "wet_to_dry_rain",
    "kessler",
    "potential_temp_to_temp",
    "dry_to_wet_water_vapor",
    "dry_to_wet_cloud_liquid_water",
    "dry_to_wet_rain",
    "kessler_update",
    "qneg",
    "geopotential_temp",
    "check_energy_zero_fluxes",
    "check_energy_scaling",
    "check_energy_chng",
    "sima_state_diagnostics",
    "kessler_diagnostics",
)

AFTER_SCHEMES = (
    "thermo_water_update",
    "check_energy_scaling",
    "dycore_energy_consistency_adjust",
    "apply_tendency_of_air_temperature",
    "sima_tend_diagnostics",
)


class SchemeBackend(Protocol):
    def lifecycle(self, name: str, pool: StatePool) -> None: ...

    def call(self, name: str, pool: StatePool) -> None: ...


class SuiteState(str, Enum):
    UNINITIALIZED = "uninitialized"
    REGISTERED = "registered"
    INITIALIZED = "initialized"
    IN_TIME_STEP = "in_time_step"
    FINALIZED = "finalized"


class KesslerSuite:
    def __init__(self, backend: SchemeBackend) -> None:
        self.backend = backend
        self.state = SuiteState.UNINITIALIZED

    def register(self, pool: StatePool) -> None:
        self._expect(SuiteState.UNINITIALIZED)
        self.backend.lifecycle("register", pool)
        self.state = SuiteState.REGISTERED

    def initialize(self, pool: StatePool) -> None:
        self._expect(SuiteState.REGISTERED)
        self.backend.lifecycle("initialize", pool)
        self.state = SuiteState.INITIALIZED

    def timestep_initial(self, pool: StatePool) -> None:
        self._expect(SuiteState.INITIALIZED)
        self.backend.lifecycle("timestep_initial", pool)
        self.state = SuiteState.IN_TIME_STEP

    def run_before(self, pool: StatePool, invoke: callable) -> None:
        self._expect(SuiteState.IN_TIME_STEP)
        for scheme in BEFORE_SCHEMES:
            invoke(scheme, pool)

    def run_after(self, pool: StatePool, invoke: callable) -> None:
        self._expect(SuiteState.IN_TIME_STEP)
        for scheme in AFTER_SCHEMES:
            invoke(scheme, pool)

    def timestep_final(self, pool: StatePool) -> None:
        self._expect(SuiteState.IN_TIME_STEP)
        self.backend.lifecycle("timestep_final", pool)
        self.state = SuiteState.INITIALIZED

    def finalize(self, pool: StatePool) -> None:
        self._expect(SuiteState.INITIALIZED)
        self.backend.lifecycle("finalize", pool)
        self.state = SuiteState.FINALIZED

    def _expect(self, expected: SuiteState) -> None:
        if self.state != expected:
            raise RuntimeError(f"Kessler suite state is {self.state}, expected {expected}")
