"""Lifecycle and phase orchestration for the CAM model."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from .backend import KernelBackend
from .comm import world_comm
from .config import ModelConfig
from .errors import RemoteRankAccessError, StateTransitionError
from .fvm_mapping import physics_to_dynamics_forcing
from .history import HistoryWriter
from .initialization import InitializationPlan
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
    dynamics_to_physics,
    dycore_energy_consistency_adjust,
    geopotential_temp,
    kessler_diagnostics,
    physics_timestep_final,
    physics_timestep_initial,
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
from .scheme_plan import (
    KesslerSchemePlan,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
    SCHEME_GROUPS,
)
from .se_runtime import (
    advance_fvm_tracers,
    advance_hyperviscosity,
    advance_se_tracers,
    apply_cam_forcing,
    compute_final_omega,
    initialize_prim_step,
    prim_advance_first_rhs,
    prim_advance_type4_rk,
    scale_physics_forcing,
    update_surface_dry_air_pressure,
    update_time_levels,
    vertical_remap_fvm,
    vertical_remap_se,
)


class DriverState(str, Enum):
    CREATED = "CREATED"
    INITIALIZED = "INITIALIZED"
    PRIMED = "PRIMED"
    RUNNING = "RUNNING"
    FINALIZED = "FINALIZED"


MODEL_PHASES = (
    "dynamics_to_physics",
    "physics_timestep_initial",
    "physics_to_dynamics",
    "scale_physics_forcing",
    "apply_cam_forcing",
    "apply_cam_forcing_substep_2",
    "initialize_prim_step",
    "se_first_rhs",
    "se_type4_rk",
    "advance_hyperviscosity",
    "update_surface_dry_air_pressure",
    "advance_se_tracers",
    "advance_fvm_tracers",
    "vertical_remap_se",
    "vertical_remap_fvm",
    "compute_final_omega",
    "update_time_levels",
    "physics_timestep_final",
)

INITIAL_PREP_PHASES = (
    "dynamics_to_physics",
    "physics_timestep_initial",
)


class CAMDriver:
    """Own one rank-local Python state pool and its validated phase sequence."""

    def __init__(
        self,
        config: ModelConfig | str | Path,
        *,
        run_dir: str | Path,
        comm: Any | None = None,
        kernel_library: str | Path | None = None,
        history_dir: str | Path | None = None,
        scheme_plan: KesslerSchemePlan | None = None,
    ) -> None:
        self.config = (
            ModelConfig.from_yaml(config)
            if isinstance(config, (str, Path))
            else config
        )
        self.config.validate()
        self.run_dir = Path(run_dir).resolve()
        self.runtime = "model"
        self.comm = comm or world_comm()
        self.state = DriverState.CREATED
        self.pool = None
        self.clock = None
        self._last_phase: str | None = None
        self._last_scheme: str | None = None
        self._last_scheme_group: str | None = None
        self.scheme_plan = (
            KesslerSchemePlan.default()
            if scheme_plan is None
            else scheme_plan.copy()
        )

        default_library = (
            Path(__file__).resolve().parents[3]
            / "build"
            / "libpycam_sima_kernels.so"
        )
        self.backend = KernelBackend(kernel_library or default_library)
        self.history = HistoryWriter(
            history_dir or self.run_dir / "history",
            self.config.case_name,
            self.comm,
        )
        self.initialization_plan = InitializationPlan(
            self.config, self.run_dir, self.comm
        )

    def initialize(self) -> CAMDriver:
        if self.state != DriverState.CREATED:
            raise StateTransitionError(f"initialize() from {self.state.value}")
        calls_before = self.backend.call_count
        context = self.initialization_plan.run()
        if self.backend.call_count != calls_before:
            raise RuntimeError("native call occurred during Python initialization")
        self.pool = context.pool
        self.clock = context.clock
        self.state = DriverState.INITIALIZED
        return self

    def start(self) -> CAMDriver:
        return self.initialize()

    @property
    def phase_names(self) -> tuple[str, ...]:
        return MODEL_PHASES

    @property
    def scheme_names(self) -> tuple[str, ...]:
        """Return unambiguous, group-qualified scheme identifiers."""

        return self.scheme_plan.keys

    @property
    def scheme_status(self) -> dict[str, object]:
        return {
            "last_scheme": self._last_scheme,
            "last_scheme_group": self._last_scheme_group,
            "sequence_safe": self.scheme_plan.sequence_safe,
            "groups": SCHEME_GROUPS,
            "plan": self.scheme_plan.to_payload(),
        }

    @property
    def phase_status(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "state": self.state.value,
            "last_phase": self._last_phase,
            "last_scheme": self._last_scheme,
            "last_scheme_group": self._last_scheme_group,
            "next_phase": None,
            "sequence_safe": self.scheme_plan.sequence_safe,
            "step": None if self.clock is None else self.clock.nstep,
            "native_nstep": None if self.clock is None else self.clock.nstep,
        }

    def get_field(
        self, name: str, *, rank: int | None = None, unsafe: bool = False
    ) -> Any:
        if self.pool is None:
            raise StateTransitionError("model has not been initialized")
        if rank is not None and rank != self.comm.rank:
            raise RemoteRankAccessError(
                f"rank {self.comm.rank} cannot return rank {rank} storage"
            )
        return self.pool.get(name, unsafe=unsafe)

    def set_field(
        self,
        name: str,
        value: Any,
        *,
        rank: int | None = None,
        unsafe: bool = False,
    ) -> None:
        if rank is not None and rank != self.comm.rank:
            raise RemoteRankAccessError(
                f"rank {self.comm.rank} cannot modify rank {rank} storage"
            )
        self.pool.set(name, value, unsafe=unsafe)

    def run_phase(self, name: str) -> CAMDriver:
        valid_states = (
            DriverState.INITIALIZED,
            DriverState.PRIMED,
            DriverState.RUNNING,
        )
        if self.state not in valid_states:
            raise StateTransitionError(f"run_phase() from {self.state.value}")
        if name not in self.phase_names:
            raise ValueError(
                f"unknown model phase {name!r}; choose one of {self.phase_names}"
            )

        before = self.pool.pointer_records()
        handler = self._phase_handlers().get(name)
        if handler is None:
            self.backend.run_phase(name, self.pool)
        else:
            handler(self.pool)
        self.pool.assert_pointer_stability(before)
        self._last_phase = name
        return self

    def run_scheme(
        self, name: str, *, group: str | None = None
    ) -> CAMDriver:
        """Run one CCPP scheme boundary against this rank's Python state."""

        valid_states = (
            DriverState.INITIALIZED,
            DriverState.PRIMED,
            DriverState.RUNNING,
        )
        if self.state not in valid_states:
            raise StateTransitionError(f"run_scheme() from {self.state.value}")
        scheme = self.scheme_plan.scheme(name, group=group)
        before = self.pool.pointer_records()
        self._scheme_handlers()[scheme.key](self.pool)
        self.pool.assert_pointer_stability(before)
        self._last_scheme = scheme.key
        self._last_scheme_group = scheme.group
        return self

    def run_scheme_group(
        self,
        group: str,
        *,
        callback: Callable[[str, CAMDriver], None] | None = None,
    ) -> CAMDriver:
        """Run all enabled schemes in one coupler group in plan order."""

        for scheme in self.scheme_plan.active(group):
            # Use the stable source identity: a scheme may execute in a group
            # different from the one where suite_kessler.xml defined it.
            self.run_scheme(scheme.key)
            if callback is not None:
                callback(scheme.key, self)
        return self

    def prepare_initial_step(
        self,
        phase_callback: Callable[[str, CAMDriver], None] | None = None,
        scheme_callback: Callable[[str, CAMDriver], None] | None = None,
    ) -> CAMDriver:
        if self.state == DriverState.CREATED:
            self.start()
        if self.state != DriverState.INITIALIZED:
            raise StateTransitionError(
                f"prepare_initial_step() from {self.state.value}"
            )
        self._run_phases(INITIAL_PREP_PHASES, callback=phase_callback)
        self.run_scheme_group(
            PHYSICS_BEFORE_COUPLER, callback=scheme_callback
        )
        if self.config.history_enabled:
            self.history.write(self.pool, self.clock)
        self.state = DriverState.PRIMED
        return self

    def step(
        self,
        *,
        phase_callback: Callable[[str, CAMDriver], None] | None = None,
        scheme_callback: Callable[[str, CAMDriver], None] | None = None,
    ) -> CAMDriver:
        if self.state == DriverState.CREATED:
            self.start()
        if self.state == DriverState.INITIALIZED:
            self.prepare_initial_step(
                phase_callback=phase_callback,
                scheme_callback=scheme_callback,
            )
        if self.state not in (
            DriverState.PRIMED,
            DriverState.RUNNING,
        ):
            raise StateTransitionError(f"step() from {self.state.value}")

        self.run_scheme_group(
            PHYSICS_AFTER_COUPLER, callback=scheme_callback
        )
        self._run_phases(
            ("physics_to_dynamics", "scale_physics_forcing"),
            callback=phase_callback,
        )
        for nsubstep in (1, 2):
            forcing_phase = (
                "apply_cam_forcing_substep_2"
                if nsubstep == 2
                else "apply_cam_forcing"
            )
            self.run_phase(forcing_phase)
            if phase_callback is not None:
                phase_callback(forcing_phase, self)
            for rstep in range(1, 4):
                if rstep > 1:
                    self.run_phase("update_time_levels")
                    if phase_callback is not None:
                        phase_callback("update_time_levels", self)
                self._run_phases(
                    (
                        "initialize_prim_step",
                        "se_type4_rk",
                        "advance_hyperviscosity",
                        "update_surface_dry_air_pressure",
                        "advance_se_tracers",
                    ),
                    callback=phase_callback,
                )
            self._run_phases(
                ("advance_fvm_tracers", "vertical_remap_se", "vertical_remap_fvm"),
                callback=phase_callback,
            )
            if nsubstep == 2:
                self.run_phase("compute_final_omega")
                if phase_callback is not None:
                    phase_callback("compute_final_omega", self)
            self.run_phase("update_time_levels")
            if phase_callback is not None:
                phase_callback("update_time_levels", self)

        self.run_phase("physics_timestep_final")
        if phase_callback is not None:
            phase_callback("physics_timestep_final", self)
        self.clock.advance()
        self.pool.set("model_step", self.clock.nstep)
        self.pool.set("current_date", self.clock.yyyymmdd)
        self.pool.set("current_seconds_of_day", self.clock.seconds)
        self._run_phases(INITIAL_PREP_PHASES, callback=phase_callback)
        self.run_scheme_group(
            PHYSICS_BEFORE_COUPLER, callback=scheme_callback
        )
        if self.config.history_enabled:
            self.history.write(self.pool, self.clock)
        self.state = DriverState.RUNNING
        return self

    def stats(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "state": self.state.value,
            "rank": self.comm.rank,
            "size": self.comm.size,
            "native_calls": self.backend.call_count,
            "step": None if self.clock is None else self.clock.nstep,
            "history_samples": (
                None
                if self.pool is None
                else int(self.pool.get("history_sample_count"))
            ),
            "last_phase": self._last_phase,
            "last_scheme": self._last_scheme,
            "last_scheme_group": self._last_scheme_group,
            "scheme_sequence_safe": self.scheme_plan.sequence_safe,
            "devices": self.backend.devices.describe(),
        }

    def finalize(self) -> None:
        if self.state == DriverState.FINALIZED:
            return
        self.comm.barrier()
        if self.pool is not None:
            self.backend.devices.release_pool(self.pool)
        self.state = DriverState.FINALIZED

    def snapshot(self):
        """Capture this rank's immutable state for a branch or checkpoint."""

        from .checkpoint import ModelSnapshot

        return ModelSnapshot.capture(self)

    def write_checkpoint(self, path: str | Path) -> Path:
        """Collectively persist all MPI ranks without finalizing the driver."""

        from .checkpoint import write_checkpoint

        return write_checkpoint(self, path)

    def _run_phases(
        self,
        phases: tuple[str, ...],
        *,
        callback: Callable[[str, CAMDriver], None] | None = None,
    ) -> None:
        for phase in phases:
            self.run_phase(phase)
            if callback is not None:
                callback(phase, self)

    def _phase_handlers(self) -> dict[str, Callable[[Any], None]]:
        return {
            "dynamics_to_physics": dynamics_to_physics,
            "physics_timestep_initial": lambda pool: physics_timestep_initial(
                pool, self.backend
            ),
            "physics_timestep_final": physics_timestep_final,
            "physics_to_dynamics": lambda pool: physics_to_dynamics_forcing(
                pool, self.comm
            ),
            "scale_physics_forcing": scale_physics_forcing,
            "apply_cam_forcing": apply_cam_forcing,
            "apply_cam_forcing_substep_2": lambda pool: apply_cam_forcing(
                pool, nsubstep=2
            ),
            "compute_final_omega": lambda pool: compute_final_omega(pool, self.comm),
            "initialize_prim_step": initialize_prim_step,
            "se_type4_rk": lambda pool: prim_advance_type4_rk(
                pool, self.comm, self.backend
            ),
            "advance_hyperviscosity": lambda pool: advance_hyperviscosity(
                pool, self.comm, self.backend
            ),
            "update_surface_dry_air_pressure": update_surface_dry_air_pressure,
            "update_time_levels": update_time_levels,
            "vertical_remap_se": lambda pool: vertical_remap_se(pool, self.backend),
            "vertical_remap_fvm": lambda pool: vertical_remap_fvm(pool, self.backend),
            "advance_se_tracers": lambda pool: advance_se_tracers(
                pool, self.comm, self.backend
            ),
            "advance_fvm_tracers": lambda pool: advance_fvm_tracers(
                pool, self.comm, self.backend
            ),
            "se_first_rhs": lambda pool: prim_advance_first_rhs(
                pool, self.comm, self.backend
            ),
        }

    def _scheme_handlers(self) -> dict[str, Callable[[Any], None]]:
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
            f"{before}.kessler": lambda pool: self.backend.run_phase(
                "kessler", pool
            ),
            f"{before}.potential_temp_to_temp": (
                potential_temperature_to_temperature
            ),
            f"{before}.dry_to_wet_water_vapor": dry_to_wet_water_vapor,
            f"{before}.dry_to_wet_cloud_liquid_water": (
                dry_to_wet_cloud_liquid_water
            ),
            f"{before}.dry_to_wet_rain": dry_to_wet_rain,
            f"{before}.kessler_update": lambda pool: self.backend.run_phase(
                "kessler_update", pool
            ),
            f"{before}.qneg": qneg,
            f"{before}.geopotential_temp": lambda pool: geopotential_temp(
                pool, self.backend
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
