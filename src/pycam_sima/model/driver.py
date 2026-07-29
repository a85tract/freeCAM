"""Lifecycle and phase orchestration for the CAM model."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import os
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
from .contracts import model_alias_rules, model_ccpp_field_aliases
from .constituents import (
    constituent_lookup_keys,
    constituent_standard_name,
    is_water_constituent,
)
from .device_catalog import DeviceCatalog
from .errors import MissingKernelError, RemoteRankAccessError, StateTransitionError
from .fvm_mapping import physics_to_dynamics_forcing
from .history import HISTORY_FIELDS, HistoryWriter
from .initialization import InitializationPlan
from .host_services import HostServiceRegistry
from .orbital_service import OrbitalHostService
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
from .scientific_data import read_musica_initial_concentrations
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
        self._suite_lifecycle_initialized = False
        # A complete CAM history sample is taken after the after-coupler
        # physics group, but before that group's tendencies are advanced by
        # the dycore.  Retain that boundary across public step() calls so the
        # terminal sample is available without running an extra model step.
        self._after_coupler_prepared = False
        # CAM snapshots the physics state after dynamics_to_physics and
        # physics_timestep_initial, before the next before-coupler group can
        # modify its working arrays.  Tendencies are added later, after the
        # after-coupler group has run.
        self._history_core_captured = False
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
        if self.config.history_core_boundary == "after_scheme":
            history_scheme = str(self.config.history_core_scheme).lower()
            if not any(
                scheme.name == history_scheme
                and scheme.source_group == PHYSICS_BEFORE_COUPLER
                for scheme in self.scheme_plan.schemes
            ):
                raise ValueError(
                    f"history_core_scheme {history_scheme!r} is not in "
                    f"{PHYSICS_BEFORE_COUPLER!r}"
                )

        project_root = Path(__file__).resolve().parents[3]
        default_library = self.config.default_kernel_library(project_root)
        self.backend = KernelBackend(kernel_library or default_library)
        self.backend.validate_specialization(self.config)
        self.device_catalog = DeviceCatalog.discover(project_root)
        suite_processes = {
            scheme.name for scheme in self.scheme_plan.schemes
        }
        self.alias_rules = model_alias_rules(self.config.constituent_names)
        self.ccpp_aliases = model_ccpp_field_aliases(
            self.config.constituent_names
        )
        self.state_schema = CCPPStateSchema.from_scheme_names(
            self.device_catalog,
            self.config.physics_suite,
            suite_processes,
            provided_standard_names=self.ccpp_aliases,
        )
        self.host_services = HostServiceRegistry.from_catalog(
            self.device_catalog,
            processes=suite_processes,
            devices=self.backend.devices,
        )
        self.processes = ProcessRouter(
            devices=self.backend.devices,
            native_invoke=self.backend.run_phase,
            host_services=self.host_services,
            host_handlers=cam_se_fvm_host_processes(
                self.backend, self.comm
            ),
        )
        initialized_contracts, generated_contracts = (
            self.state_schema.pool_contract_groups(
                provided_standard_names=self.ccpp_aliases
            )
        )
        self.plugins = PhysicsPluginManager(self)
        self.fields = FieldCollection(self)
        self.phases = PhaseCollection(self)
        self.physics = PhysicsCollection(self)
        self.history = HistoryWriter(
            history_dir or self.run_dir / "history",
            self.config.case_name,
            self.comm,
            config=self.config,
        )
        self.orbital_service = OrbitalHostService(project_root)
        self.initialization_plan = InitializationPlan(
            self.config,
            self.run_dir,
            self.comm,
            contracts=initialized_contracts,
            generated_contracts=generated_contracts,
            alias_rules=self.alias_rules,
            ccpp_aliases=self.ccpp_aliases,
            namelist_bindings=self.state_schema.namelist_bindings,
            required_dimensions=self.state_schema.dimension_names,
            fixed_dimensions=self.state_schema.fixed_dimensions,
        )

    def initialize(self) -> CAMDriver:
        if self.state != DriverState.CREATED:
            raise StateTransitionError(f"initialize() from {self.state.value}")
        if self.config.run_type.lower() != "startup":
            return self._initialize_from_restart()
        calls_before = self.backend.call_count
        context = self.initialization_plan.run()
        if self.backend.call_count != calls_before:
            raise RuntimeError("native call occurred during Python initialization")
        self.pool = context.pool
        self.clock = context.clock
        self.state = DriverState.INITIALIZED
        return self

    def _initialize_from_restart(self) -> CAMDriver:
        """Restore continue/branch cases from a Python-owned checkpoint."""

        from .checkpoint import read_checkpoint, restore_driver

        requested_config = self.config
        restart_path = requested_config.resolve_restart_path(self.run_dir)
        if restart_path is None:
            raise StateTransitionError(
                f"run_type={requested_config.run_type!r} requires restart_path"
            )
        snapshot = read_checkpoint(restart_path, self.comm)
        restored_config = ModelConfig.from_mapping(snapshot.config)
        ignored = {
            "run_type",
            "restart_path",
            "stop_n",
            "case_name",
            "history_enabled",
        }
        requested = requested_config.as_dict()
        previous = restored_config.as_dict()
        differences = {
            name: (previous[name], requested[name])
            for name in sorted(set(previous) | set(requested))
            if name not in ignored and previous.get(name) != requested.get(name)
        }
        if differences:
            raise StateTransitionError(
                "restart configuration changes model-defining values: "
                f"{differences}"
            )
        restored = restore_driver(
            snapshot,
            run_dir=self.run_dir,
            comm=self.comm,
            kernel_library=self.backend.path,
            history_dir=self.history.output_dir,
        )
        self.__dict__.update(restored.__dict__)
        self.config = ModelConfig.from_mapping(
            {
                **restored_config.as_dict(),
                "run_type": requested_config.run_type,
                "restart_path": requested_config.restart_path,
                "stop_n": requested_config.stop_n,
                "case_name": requested_config.case_name,
                "history_enabled": requested_config.history_enabled,
            }
        )
        self.history.config = self.config
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
        debug = bool(os.environ.get("PYCAM_DEBUG_PROCESS_TRACE"))
        if debug:
            print(
                f"PYCAM_PHASE_BEGIN rank={self.comm.rank} {name}",
                flush=True,
            )
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
        if debug:
            print(
                f"PYCAM_PHASE_DONE rank={self.comm.rank} {name}",
                flush=True,
            )
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
        if (
            self.config.history_core_boundary == "after_scheme"
            and execution_group == PHYSICS_BEFORE_COUPLER
            and scheme.name
            == str(self.config.history_core_scheme).strip().lower()
        ):
            self._capture_history_core()
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

    def run_suite_lifecycle(self, phase: str) -> tuple[str, ...]:
        """Run one CCPP lifecycle boundary once per unique suite scheme."""

        valid = {
            "register",
            "initialize",
            "timestep_initial",
            "timestep_final",
            "finalize",
        }
        if phase not in valid:
            raise ValueError(
                f"unknown CCPP lifecycle {phase!r}; choose from "
                f"{sorted(valid)}"
            )
        if self.pool is None:
            raise StateTransitionError("model has not been initialized")
        executed: list[str] = []
        seen: set[str] = set()
        before = self.pool.pointer_records()
        for scheme in self.scheme_plan.schemes:
            if scheme.name in seen:
                continue
            seen.add(scheme.name)
            entry = self.device_catalog.entries.get(scheme.name)
            if entry is None or phase not in entry.lifecycle:
                continue
            process = f"{scheme.name}:{phase}"
            provider = self.processes.provider_for_process(
                process,
                source_group=scheme.source_group,
            )
            if provider is None:
                raise MissingKernelError(
                    f"suite {self.scheme_plan.name!r} requires lifecycle "
                    f"process {process!r}, but no Fortran device or Python "
                    "host provider supplies it"
                )
            self._native_call_depth += 1
            try:
                self.processes.invoke_process(
                    process,
                    self.pool,
                    source_group=scheme.source_group,
                )
            finally:
                self._native_call_depth -= 1
            executed.append(process)
        self.pool.assert_pointer_stability(before)
        if executed:
            self._boundary_index += 1
        return tuple(executed)

    def _ensure_suite_lifecycle_initialized(self) -> None:
        if self._suite_lifecycle_initialized:
            return
        self.run_suite_lifecycle("register")
        configured_constituents = {
            constituent_standard_name(name).lower()
            for name in self.pool.constituent_names
        }
        suite_constituents = tuple(
            name
            for name in self.device_catalog.suite_constituent_standard_names(
                scheme.name for scheme in self.scheme_plan.schemes
            )
            if name.lower() in configured_constituents
        )
        self.backend.devices.initialize_constituent_registry(
            self.pool,
            device_names=(
                scheme.name for scheme in self.scheme_plan.schemes
            ),
            constituent_standard_names=suite_constituents,
        )
        self.run_suite_lifecycle("initialize")
        self._suite_lifecycle_initialized = True

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
        self._ensure_suite_lifecycle_initialized()
        self._run_phases(INITIAL_PREP_PHASES, callback=phase_callback)
        if self.config.history_core_boundary == "before_before_coupler":
            self._capture_history_core()
        self._run_optional_scheme_group(
            PHYSICS_BEFORE_COUPLER, callback=scheme_callback
        )
        self._after_coupler_prepared = False
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

        self._prepare_after_coupler_boundary(
            scheme_callback=scheme_callback
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
        if self.config.history_core_boundary == "before_before_coupler":
            self._capture_history_core()
        self._run_optional_scheme_group(
            PHYSICS_BEFORE_COUPLER, callback=scheme_callback
        )
        self._after_coupler_prepared = False
        self._prepare_after_coupler_boundary(
            scheme_callback=scheme_callback
        )
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
            if self._suite_lifecycle_initialized:
                self.run_suite_lifecycle("finalize")
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

    def _prepare_after_coupler_boundary(
        self,
        *,
        scheme_callback: Callable[[str, CAMDriver], None] | None = None,
    ) -> None:
        """Run after-coupler physics and sample the current CAM time once.

        CAM writes history from ``cam_timestep_final`` before advancing its
        clock.  At that point ``physics_after_coupler`` has produced the
        tendencies for the current state, while the dycore has not yet made
        the next state visible to physics.  Keeping this boundary prepared
        makes N public steps produce the CAM-compatible N+1 samples without
        shifting tendencies to the following timestamp.
        """

        if self._after_coupler_prepared:
            return
        self._capture_history_core()
        self._run_optional_scheme_group(
            PHYSICS_AFTER_COUPLER, callback=scheme_callback
        )
        if self.config.history_enabled:
            self.history.capture(
                self.pool,
                (
                    ("TTEND", "physics_air_temperature_tendency"),
                    ("UTEND", "physics_zonal_wind_tendency"),
                    ("VTEND", "physics_meridional_wind_tendency"),
                ),
            )
            self.history.write(self.pool, self.clock)
            self._history_core_captured = False
        self._after_coupler_prepared = True

    def _capture_history_core(self) -> None:
        """Preserve CAM core fields at their pre-physics history boundary."""

        if not self.config.history_enabled or self._history_core_captured:
            return
        initial_musica = (
            self.config.physics_suite.lower() == "musica"
            and self.clock.nstep == 0
        )
        self.history.capture(
            self.pool,
            tuple(
                (
                    output_name,
                    (
                        "physics_surface_dry_air_pressure"
                        if initial_musica and output_name == "PS"
                        else state_name
                    ),
                )
                for output_name, state_name in HISTORY_FIELDS
                if output_name not in {"TTEND", "UTEND", "VTEND"}
            ),
            reset=True,
        )
        self._history_core_captured = True

    def _phase_handlers(self) -> dict[str, Callable[[Any], None]]:
        return {
            "dynamics_to_physics": lambda pool: dynamics_to_physics(
                pool,
                comm=self.comm,
                backend=self.backend,
                canonicalize_resting_wind_zero=(
                    self.config.canonicalize_resting_wind_zero
                ),
            ),
            "physics_timestep_initial": self._physics_timestep_initial,
            "physics_timestep_final": self._physics_timestep_final,
            "physics_to_dynamics": lambda pool: physics_to_dynamics_forcing(
                pool, self.comm, self.backend
            ),
            "scale_physics_forcing": lambda pool: scale_physics_forcing(
                pool, self.backend
            ),
            "apply_cam_forcing": lambda pool: apply_cam_forcing(
                pool, self.backend
            ),
            "apply_cam_forcing_substep_2": lambda pool: apply_cam_forcing(
                pool, self.backend, nsubstep=2
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

    def _physics_timestep_initial(self, pool: Any) -> None:
        try:
            pool.set(
                pool.ccpp_field_name("is_first_timestep"),
                self.clock.nstep == 0,
            )
        except KeyError:
            pass
        current_calday = self.clock.fractional_calendar_day()
        next_calday = self.clock.fractional_calendar_day(
            self.clock.dt_seconds
        )
        for standard_name, value in (
            ("current_timestep_number", self.clock.nstep),
            (
                "fractional_calendar_days_on_end_of_current_timestep",
                current_calday,
            ),
            (
                "fractional_calendar_days_on_end_of_next_timestep",
                next_calday,
            ),
            (
                "next_calendar_day_to_perform_shortwave_radiation_for_"
                "surface_models",
                next_calday,
            ),
            (
                "number_of_seconds_until_next_shortwave_radiation_timestep",
                0,
            ),
        ):
            try:
                pool.set(pool.ccpp_field_name(standard_name), value)
            except KeyError:
                pass
        if (
            self.config.physics_suite.lower() == "musica"
            and int(pool.get("model_step")) == 0
        ):
            project_root = Path(__file__).resolve().parents[3]
            raw_configuration = self.config.namelist_overrides.get(
                "musica_ccpp", {}
            ).get("filename_of_micm_configuration")
            if raw_configuration is None:
                raise ValueError(
                    "MUSICA requires filename_of_micm_configuration"
                )
            configuration = (
                str(raw_configuration)
                .replace("${PROJECT_ROOT}", str(project_root))
                .replace("${SOURCE_ROOT}", str(self.config.source_root))
                .replace("${RUNDIR}", str(self.run_dir))
            )
            concentrations = read_musica_initial_concentrations(
                self.config.source_root,
                configuration,
            )
            constituents = pool.get("physics_constituent_mixing_ratio")
            for index, name in enumerate(self.config.constituent_names):
                value = next(
                    (
                        concentrations[key]
                        for key in constituent_lookup_keys(name)
                        if key in concentrations
                    ),
                    None,
                )
                if value is None:
                    if is_water_constituent(name):
                        # CAM's ordinary analytic-IC path owns water species;
                        # set_initial_musica_concentrations only fills the
                        # chemistry constituents registered by MUSICA.
                        continue
                    raise ValueError(
                        f"MUSICA startup concentration is missing {name!r}"
                    )
                constituents[:, :, index] = value
            pool.mark_initialized("physics_constituent_mixing_ratio")
        physics_timestep_initial(pool, self.backend)
        self.orbital_service.update(
            pool,
            self.clock,
            orbital_year=self.config.orbital_year,
        )
        self.run_suite_lifecycle("timestep_initial")
        # rrtmgp_pre_timestep_init writes radiation_offset.  CAM computes
        # nextsw_cday from that output only after phys_timestep_init returns;
        # it is not necessarily the calendar day of the immediately next
        # model step when radiation is subcycled.
        try:
            radiation_offset = int(
                pool.get(
                    pool.ccpp_field_name(
                        "number_of_seconds_until_next_shortwave_radiation_"
                        "timestep"
                    )
                ).item()
            )
            pool.set(
                pool.ccpp_field_name(
                    "next_calendar_day_to_perform_shortwave_radiation_for_"
                    "surface_models"
                ),
                self.clock.fractional_calendar_day(radiation_offset),
            )
        except KeyError:
            pass

    def _physics_timestep_final(self, pool: Any) -> None:
        self.run_suite_lifecycle("timestep_final")
        physics_timestep_final(pool)
