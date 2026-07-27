"""Lifecycle and phase orchestration for the CAM model."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

from .backend import KernelBackend
from .capabilities import CAM_SE_FVM_V1, RuntimeCapabilities
from .ccpp_suite import (
    CCPPSuitePlan,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
)
from .ccpp_state import CCPPStateSchema
from .comm import world_comm
from .config import ModelConfig
from .device_catalog import DeviceCatalog
from .errors import RemoteRankAccessError, StateTransitionError
from .fvm_mapping import physics_to_dynamics_forcing
from .history import HistoryWriter
from .initialization import InitializationPlan
from .host_services import HostServiceRegistry
from .phases import (
    dynamics_to_physics,
    physics_timestep_final,
    physics_timestep_initial,
)
from .processes import ProcessRouter, cam_se_fvm_host_processes
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
from .plugins import (
    PhysicsPluginManager,
    PhysicsPluginSpec,
    VariableSpec,
    _UNSET as _PLUGIN_UNSET,
)
from .user_api import FieldCollection, PhaseCollection, PhysicsCollection


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
        scheme_plan: CCPPSuitePlan | None = None,
        capabilities: RuntimeCapabilities = CAM_SE_FVM_V1,
    ) -> None:
        self.config = (
            ModelConfig.from_yaml(config)
            if isinstance(config, (str, Path))
            else config
        )
        self.config.validate()
        self.capabilities = capabilities
        self.capabilities.validate(self.config)
        suite_xml = self.config.verify_suite()
        self.run_dir = Path(run_dir).resolve()
        self.runtime = "model"
        self.comm = comm or world_comm()
        self.state = DriverState.CREATED
        self.pool = None
        self.clock = None
        self._last_phase: str | None = None
        self._last_scheme: str | None = None
        self._last_scheme_group: str | None = None
        self._native_call_depth = 0
        self._boundary_index = 0
        self.scheme_plan = (
            CCPPSuitePlan.from_xml(suite_xml)
            if scheme_plan is None
            else scheme_plan.copy()
        )
        if self.scheme_plan.name.lower() != self.config.physics_suite.lower():
            raise ValueError(
                f"suite plan {self.scheme_plan.name!r} does not match "
                f"physics_suite={self.config.physics_suite!r}"
            )

        default_library = (
            Path(__file__).resolve().parents[3]
            / "build"
            / "libpycam_sima_kernels.so"
        )
        self.backend = KernelBackend(kernel_library or default_library)
        project_root = Path(__file__).resolve().parents[3]
        self.device_catalog = DeviceCatalog.discover(project_root)
        suite_processes = {
            scheme.name for scheme in self.scheme_plan.schemes
        }
        self.state_schema = CCPPStateSchema.from_scheme_names(
            self.device_catalog,
            self.config.physics_suite,
            suite_processes,
        )
        self.host_services = HostServiceRegistry.from_catalog(
            self.device_catalog,
            processes=suite_processes,
        )
        self.processes = ProcessRouter(
            devices=self.backend.devices,
            native_invoke=self.backend.run_phase,
            host_services=self.host_services,
            host_handlers=cam_se_fvm_host_processes(self.backend),
        )
        initialized_contracts, generated_contracts = (
            self.state_schema.pool_contract_groups()
        )
        self.plugins = PhysicsPluginManager(self)
        self.fields = FieldCollection(self)
        self.phases = PhaseCollection(self)
        self.physics = PhysicsCollection(self)
        self.history = HistoryWriter(
            history_dir or self.run_dir / "history",
            self.config.case_name,
            self.comm,
        )
        self.initialization_plan = InitializationPlan(
            self.config,
            self.run_dir,
            self.comm,
            contracts=initialized_contracts,
            generated_contracts=generated_contracts,
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
            "groups": self.scheme_plan.group_names,
            "plan": self.scheme_plan.to_payload(),
            "suite": self.scheme_plan.name,
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

    @property
    def execution_cursor(self) -> tuple[object, ...]:
        """Collectively comparable boundary occupied by this MPI rank."""

        return (
            self.state.value,
            None if self.clock is None else int(self.clock.nstep),
            self._last_phase,
            self._last_scheme,
            self._last_scheme_group,
            int(self._boundary_index),
            int(self._native_call_depth),
        )

    def define_variable(
        self,
        spec: VariableSpec,
        *,
        initial: Any = _PLUGIN_UNSET,
    ) -> Any:
        if initial is _PLUGIN_UNSET:
            return self.plugins.define_variable(spec)
        return self.plugins.define_variable(spec, initial=initial)

    def delete_variable(self, name: str) -> dict[str, Any]:
        """Collectively delete an unused dynamic StatePool variable."""

        return self.plugins.delete_variable(name)

    def install_physics(
        self,
        spec: PhysicsPluginSpec | str | Path,
        *,
        initial_values: dict[str, Any] | None = None,
        effective: str = "now",
        unsafe: bool = False,
    ) -> Any:
        return self.plugins.install(
            spec,
            initial_values=initial_values,
            effective=effective,
            unsafe=unsafe,
        )

    def deactivate_physics(
        self, name: str, *, unsafe: bool = False
    ) -> None:
        self.plugins.deactivate(name, unsafe=unsafe)

    def activate_physics(
        self, name: str, *, unsafe: bool = False
    ) -> None:
        self.plugins.activate(name, unsafe=unsafe)

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
        self._native_call_depth += 1
        try:
            handler = self._phase_handlers().get(name)
            if handler is None:
                self.backend.run_phase(name, self.pool)
            else:
                handler(self.pool)
            self.pool.assert_pointer_stability(before)
        finally:
            self._native_call_depth -= 1
        self._last_phase = name
        self._boundary_index += 1
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
        execution_group = self.scheme_plan.execution_group(scheme.key)
        before = self.pool.pointer_records()
        self._native_call_depth += 1
        try:
            self.processes.invoke(scheme, self.pool)
            self.pool.assert_pointer_stability(before)
        finally:
            self._native_call_depth -= 1
        self._last_scheme = scheme.key
        self._last_scheme_group = execution_group
        self._boundary_index += 1
        return self

    def run_scheme_group(
        self,
        group: str,
        *,
        callback: Callable[[str, CAMDriver], None] | None = None,
    ) -> CAMDriver:
        """Run all enabled schemes in one coupler group in plan order."""

        for scheme in self.scheme_plan.expanded(group, self.pool.dimensions):
            # Use stable source identity: a scheme may execute in a group
            # different from the one where its suite XML defined it.
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
        self._run_optional_scheme_group(
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
        self.plugins.activate_pending()
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

        self._run_optional_scheme_group(
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
        self._run_optional_scheme_group(
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
            "suite": self.scheme_plan.name,
            "capabilities": self.capabilities.describe(),
            "state_schema": self.state_schema.report(),
            "process_coverage": self.processes.describe(
                self.scheme_plan.schemes
            ),
            "devices": self.backend.devices.describe(),
            "host_service_events": self.host_services.events(),
            "plugins": self.plugins.inventory(),
            "execution_cursor": self.execution_cursor,
        }

    def finalize(self) -> None:
        if self.state == DriverState.FINALIZED:
            return
        self.comm.barrier()
        if self.pool is not None:
            self.plugins.finalize_all()
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

    def _run_optional_scheme_group(
        self,
        group: str,
        *,
        callback: Callable[[str, CAMDriver], None] | None = None,
    ) -> None:
        """Execute a CAM coupling group when the selected suite defines it."""

        if group in self.scheme_plan.group_names:
            self.run_scheme_group(group, callback=callback)

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
