"""Source-ordered PI-CAM phase and scheme plan."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Iterator

from .errors import PICAMConfigurationError


@dataclass(frozen=True, slots=True)
class PICAMAction:
    name: str
    phase: str
    operation: str
    kind: str
    native_id: int | None = None
    enabled: bool = True

    @property
    def qualified_name(self) -> str:
        return f"{self.phase}.{self.name}"


_DEFAULT_ACTIONS = (
    PICAMAction("boundary_import", "coupling", "boundary_import", "boundary", 202),
    PICAMAction("prepare", "cam_run2", "prepare", "control", 401),
    PICAMAction("surface_fluxes_and_emissions", "cam_run2", "chem_emissions", "scheme", 402),
    PICAMAction("tracers_and_chemistry", "cam_run2", "tracers_chemistry", "scheme", 403),
    PICAMAction("tracer_tendencies_leaf", "cam_run2", "leaf_tracers_timestep_tend", "scheme", 460, False),
    PICAMAction("age_of_air_tendencies_leaf", "cam_run2", "leaf_aoa_tracers_timestep_tend", "scheme", 461, False),
    PICAMAction("chemistry_tendencies_leaf", "cam_run2", "leaf_chem_timestep_tend", "scheme", 462, False),
    PICAMAction("vertical_diffusion", "cam_run2", "vertical_diffusion_tend", "scheme", 404),
    PICAMAction("rayleigh_friction", "cam_run2", "rayleigh_friction_tend", "scheme", 405),
    PICAMAction("aerosol_dry_deposition", "cam_run2", "aero_model_drydep", "scheme", 406),
    PICAMAction("aerosol_dry_deposition_leaf", "cam_run2", "leaf_aero_model_drydep", "scheme", 463, False),
    PICAMAction("carma_aerosol_tendencies_leaf", "cam_run2", "leaf_carma_timestep_tend", "scheme", 464, False),
    PICAMAction("charge_neutrality", "cam_run2", "charge_fix", "scheme", 407),
    PICAMAction("gravity_wave_drag", "cam_run2", "gw_tend", "scheme", 408),
    PICAMAction("qbo_relaxation", "cam_run2", "qbo_relax", "scheme", 409),
    PICAMAction("ion_drag", "cam_run2", "iondrag_calc", "scheme", 410),
    PICAMAction("state_finalize", "cam_run2", "physics_dme_adjust", "scheme", 411),
    PICAMAction("finish", "cam_run2", "finish", "control", 412),
    PICAMAction("carma_statistics_leaf", "cam_run2", "leaf_carma_accumulate_stats", "scheme", 465, False),
    PICAMAction("physics_buffer_deallocate_leaf", "cam_run2", "leaf_pbuf_deallocate", "control", 466, False),
    PICAMAction("physics_buffer_time_advance_leaf", "cam_run2", "leaf_pbuf_update_tim_idx", "control", 467, False),
    PICAMAction("diagnostics_deallocate_leaf", "cam_run2", "leaf_diag_deallocate", "control", 468, False),
    PICAMAction("dynamics", "cam_run2", "stepon_run2", "dynamics", 413),
    PICAMAction("dynamics", "cam_run3", "stepon_run3", "dynamics", 414),
    PICAMAction("history", "cam_run4", "wshist", "io", 415),
    PICAMAction("restart", "cam_run4", "restart", "io", 416),
    PICAMAction("finish", "cam_run4", "wrapup", "control", 417),
    PICAMAction("wrapup_leaf", "cam_run4", "leaf_cam_run4_wrapup", "io", 470, False),
    PICAMAction("step_cost_leaf", "cam_run4", "leaf_cam_run4_step_cost", "control", 471, False),
    PICAMAction("flush_leaf", "cam_run4", "leaf_cam_run4_flush", "io", 472, False),
    PICAMAction("advance_timestep", "clock", "advance_timestep", "clock", 418),
    PICAMAction("dynamics", "cam_run1", "stepon_run1", "dynamics", 419),
    PICAMAction("prepare", "cam_run1", "prepare_cam_run1", "control", 420),
    PICAMAction("state_initialize", "cam_run1", "bc_init", "scheme", 421),
    PICAMAction("energy_fixer", "cam_run1", "check_energy_fix", "scheme", 422),
    PICAMAction("dry_adjustment", "cam_run1", "dadadj", "scheme", 423),
    PICAMAction("deep_convection", "cam_run1", "convect_deep_tend", "scheme", 424),
    PICAMAction("shallow_convection", "cam_run1", "convect_shallow_tend", "scheme", 425),
    PICAMAction("sea_salt_rebin", "cam_run1", "sslt_rebin_adv", "scheme", 426),
    PICAMAction("cloud_macro_microphysics", "cam_run1", "macro_microphysics", "scheme", 427),
    PICAMAction("wet_deposition", "cam_run1", "aero_model_wetdep", "scheme", 428),
    PICAMAction("modal_aerosol_preparation_leaf", "cam_run1", "leaf_modal_aero_prepare", "scheme", 450, False),
    PICAMAction("aerosol_wet_deposition_leaf", "cam_run1", "leaf_aero_model_wetdep", "scheme", 451, False),
    PICAMAction("carma_wet_deposition_leaf", "cam_run1", "leaf_carma_wetdep_tend", "scheme", 452, False),
    PICAMAction("convective_tracer_transport_leaf", "cam_run1", "leaf_convect_deep_tend_2", "scheme", 453, False),
    PICAMAction("diagnostics", "cam_run1", "physics_diagnostics", "scheme", 429),
    PICAMAction("state_and_convection_diagnostics_leaf", "cam_run1", "leaf_diag_phys_writeout", "scheme", 454, False),
    PICAMAction("cloud_diagnostics_leaf", "cam_run1", "leaf_cloud_diagnostics_calc", "scheme", 455, False),
    PICAMAction("radiation", "cam_run1", "radiation_tend", "scheme", 430),
    PICAMAction("state_export", "cam_run1", "cam_export", "scheme", 431),
    PICAMAction("tropopause_leaf", "cam_run1", "leaf_tropopause_output", "scheme", 456, False),
    PICAMAction("state_export_leaf", "cam_run1", "leaf_cam_export", "scheme", 457, False),
    PICAMAction("export_diagnostics_leaf", "cam_run1", "leaf_diag_export", "scheme", 458, False),
    PICAMAction("boundary_export", "coupling", "boundary_export", "boundary", 432),
)

class PICAMStepPlan:
    """Mutable ordering facade with a source-faithful safe default."""

    def __init__(self, actions: Iterable[PICAMAction] = _DEFAULT_ACTIONS) -> None:
        self._actions = list(actions)
        self._validate()

    @classmethod
    def default(cls) -> "PICAMStepPlan":
        return cls()

    def _validate(self) -> None:
        names = [action.qualified_name for action in self._actions]
        if len(names) != len(set(names)):
            raise PICAMConfigurationError("PI-CAM action names must be unique per phase")
        if not self._actions or self._actions[0].operation != "boundary_import":
            raise PICAMConfigurationError("PI-CAM step must begin with boundary import")
        if self._actions[-1].operation != "boundary_export":
            raise PICAMConfigurationError("PI-CAM step must end with boundary export")

    def __iter__(self) -> Iterator[PICAMAction]:
        return (action for action in self._actions if action.enabled)

    @property
    def actions(self) -> tuple[PICAMAction, ...]:
        return tuple(self._actions)

    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(action.phase for action in self._actions))

    @property
    def is_source_default(self) -> bool:
        """Whether the safe source ordering is still completely unchanged."""

        return tuple(self._actions) == _DEFAULT_ACTIONS

    def select(self, name: str, *, phase: str | None = None) -> PICAMAction:
        matches = [
            action
            for action in self._actions
            if (action.name == name or action.operation == name or action.qualified_name == name)
            and (phase is None or action.phase == phase)
        ]
        if len(matches) != 1:
            raise PICAMConfigurationError(
                f"action {name!r} is unknown or ambiguous; specify phase"
            )
        return matches[0]

    def in_phase(self, phase: str) -> tuple[PICAMAction, ...]:
        actions = tuple(action for action in self._actions if action.phase == phase and action.enabled)
        if not actions:
            raise PICAMConfigurationError(f"unknown or empty phase {phase!r}")
        return actions

    def set_enabled(
        self, name: str, enabled: bool, *, phase: str | None = None, experimental: bool = False
    ) -> None:
        if not experimental:
            raise PICAMConfigurationError(
                "changing the scientific PI-CAM plan requires experimental=True"
            )
        selected = self.select(name, phase=phase)
        index = self._actions.index(selected)
        self._actions[index] = replace(selected, enabled=bool(enabled))

    def expand_cam_run1_leaves(self, *, experimental: bool = False) -> None:
        """Replace three composite ``cam_run1`` stages with leaf actions."""

        if not experimental:
            raise PICAMConfigurationError(
                "expanding cam_run1 leaf routines requires experimental=True"
            )
        for name in ("wet_deposition", "diagnostics", "state_export"):
            self.set_enabled(
                name,
                False,
                phase="cam_run1",
                experimental=True,
            )
        for name in (
            "modal_aerosol_preparation_leaf",
            "aerosol_wet_deposition_leaf",
            "carma_wet_deposition_leaf",
            "convective_tracer_transport_leaf",
            "state_and_convection_diagnostics_leaf",
            "cloud_diagnostics_leaf",
            "tropopause_leaf",
            "state_export_leaf",
            "export_diagnostics_leaf",
        ):
            self.set_enabled(
                name,
                True,
                phase="cam_run1",
                experimental=True,
            )

    def expand_cam_run2_leaves(self, *, experimental: bool = False) -> None:
        """Replace three composite ``cam_run2`` stages with leaf actions."""

        if not experimental:
            raise PICAMConfigurationError(
                "expanding cam_run2 leaf routines requires experimental=True"
            )
        for name in (
            "tracers_and_chemistry",
            "aerosol_dry_deposition",
            "finish",
        ):
            self.set_enabled(
                name,
                False,
                phase="cam_run2",
                experimental=True,
            )
        for name in (
            "tracer_tendencies_leaf",
            "age_of_air_tendencies_leaf",
            "chemistry_tendencies_leaf",
            "aerosol_dry_deposition_leaf",
            "carma_aerosol_tendencies_leaf",
            "carma_statistics_leaf",
            "physics_buffer_deallocate_leaf",
            "physics_buffer_time_advance_leaf",
            "diagnostics_deallocate_leaf",
        ):
            self.set_enabled(
                name,
                True,
                phase="cam_run2",
                experimental=True,
            )

    def expand_cam_run4_leaves(self, *, experimental: bool = False) -> None:
        """Replace the composite ``cam_run4`` finish with leaf actions."""

        if not experimental:
            raise PICAMConfigurationError(
                "expanding cam_run4 leaf routines requires experimental=True"
            )
        self.set_enabled(
            "finish",
            False,
            phase="cam_run4",
            experimental=True,
        )
        for name in ("wrapup_leaf", "step_cost_leaf", "flush_leaf"):
            self.set_enabled(
                name,
                True,
                phase="cam_run4",
                experimental=True,
            )

    def expand_cam_run2_run4_leaves(
        self, *, experimental: bool = False
    ) -> None:
        """Expand every admitted leaf boundary from ``cam_run2`` to run4."""

        if not experimental:
            raise PICAMConfigurationError(
                "expanding cam_run2-run4 leaves requires experimental=True"
            )
        self.expand_cam_run2_leaves(experimental=True)
        self.expand_cam_run4_leaves(experimental=True)

    def move(
        self,
        name: str,
        *,
        phase: str | None = None,
        before: str | None = None,
        after: str | None = None,
        experimental: bool = False,
    ) -> None:
        if not experimental:
            raise PICAMConfigurationError(
                "changing the scientific PI-CAM order requires experimental=True"
            )
        if (before is None) == (after is None):
            raise PICAMConfigurationError("provide exactly one of before or after")
        selected = self.select(name, phase=phase)
        target = self.select(before or after or "", phase=phase)
        self._actions.remove(selected)
        target_index = self._actions.index(target)
        self._actions.insert(target_index + (after is not None), selected)
        self._validate()

    def add(
        self,
        action: PICAMAction,
        *,
        before: str | None = None,
        after: str | None = None,
        experimental: bool = False,
    ) -> PICAMAction:
        if not experimental:
            raise PICAMConfigurationError(
                "adding a PI-CAM process requires experimental=True"
            )
        if (before is None) == (after is None):
            raise PICAMConfigurationError("provide exactly one of before or after")
        if any(item.qualified_name == action.qualified_name for item in self._actions):
            raise PICAMConfigurationError(
                f"action {action.qualified_name!r} already exists"
            )
        target = self.select(before or after or "", phase=action.phase)
        target_index = self._actions.index(target)
        self._actions.insert(target_index + (after is not None), action)
        try:
            self._validate()
        except BaseException:
            self._actions.remove(action)
            raise
        return action

    def remove(
        self,
        name: str,
        *,
        phase: str | None = None,
        experimental: bool = False,
    ) -> PICAMAction:
        if not experimental:
            raise PICAMConfigurationError(
                "removing a PI-CAM process requires experimental=True"
            )
        selected = self.select(name, phase=phase)
        if selected.kind not in {"python_process", "runtime_fortran_process"}:
            raise PICAMConfigurationError(
                f"source action {selected.qualified_name!r} cannot be removed"
            )
        self._actions.remove(selected)
        self._validate()
        return selected

    def describe(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "index": index,
                "phase": action.phase,
                "name": action.name,
                "operation": action.operation,
                "kind": action.kind,
                "granularity": (
                    "leaf" if action.operation.startswith("leaf_") else "stage"
                ),
                "native_id": action.native_id,
                "enabled": action.enabled,
            }
            for index, action in enumerate(self._actions)
        )
