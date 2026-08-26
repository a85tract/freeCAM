"""Python-owned, source-ordered PI-CAM process plan."""

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
    parent_stage: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.phase}.{self.name}"

    @property
    def control_owner(self) -> str:
        """The complete workflow is always ordered and gated by Python."""

        return "python"

    @property
    def implementation(self) -> str:
        """Describe the primitive used after Python has selected this action."""

        return {
            "boundary": "fortran-numerical-kernel",
            "scheme": "fortran-numerical-kernel",
            "coupling": "fortran-numerical-kernel",
            "dynamics": "fortran-numerical-kernel",
            "kernel": "fortran-numerical-kernel",
            "runtime_fortran_process": "fortran-numerical-kernel",
            "runtime_catalog_process": "fortran-numerical-kernel",
            "python_process": "python",
            "io": "fortran-io-service",
            "service": "fortran-state-service",
            "clock": "fortran-clock-mirror",
        }.get(self.kind, "python")


_DEFAULT_ACTIONS = (
    PICAMAction("boundary_import", "coupling", "boundary_import", "boundary", 202),
    PICAMAction("prepare", "cam_run2", "prepare", "kernel", 401),
    PICAMAction("surface_fluxes_and_emissions", "cam_run2", "chem_emissions", "scheme", 402),
    PICAMAction("tracers_and_chemistry", "cam_run2", "tracers_chemistry", "scheme", 403, False),
    PICAMAction("tracer_tendencies_leaf", "cam_run2", "leaf_tracers_timestep_tend", "scheme", 460, parent_stage="cam_run2.tracers_and_chemistry"),
    PICAMAction("age_of_air_tendencies_leaf", "cam_run2", "leaf_aoa_tracers_timestep_tend", "scheme", 461, parent_stage="cam_run2.tracers_and_chemistry"),
    PICAMAction("chemistry_tendencies_leaf", "cam_run2", "leaf_chem_timestep_tend", "scheme", 462, parent_stage="cam_run2.tracers_and_chemistry"),
    PICAMAction("vertical_diffusion", "cam_run2", "vertical_diffusion_tend", "scheme", 404),
    PICAMAction("rayleigh_friction", "cam_run2", "rayleigh_friction_tend", "scheme", 405),
    PICAMAction("aerosol_dry_deposition", "cam_run2", "aero_model_drydep", "scheme", 406, False),
    PICAMAction("aerosol_dry_deposition_leaf", "cam_run2", "leaf_aero_model_drydep", "scheme", 463, parent_stage="cam_run2.aerosol_dry_deposition"),
    PICAMAction("carma_aerosol_tendencies_leaf", "cam_run2", "leaf_carma_timestep_tend", "scheme", 464, parent_stage="cam_run2.aerosol_dry_deposition"),
    PICAMAction("charge_neutrality", "cam_run2", "charge_fix", "scheme", 407),
    PICAMAction("gravity_wave_drag", "cam_run2", "gw_tend", "scheme", 408),
    PICAMAction("qbo_relaxation", "cam_run2", "qbo_relax", "scheme", 409),
    PICAMAction("ion_drag", "cam_run2", "iondrag_calc", "scheme", 410),
    PICAMAction("state_finalize", "cam_run2", "physics_dme_adjust", "scheme", 411),
    PICAMAction("finish", "cam_run2", "finish", "kernel", 412, False),
    PICAMAction("carma_statistics_leaf", "cam_run2", "leaf_carma_accumulate_stats", "scheme", 465, parent_stage="cam_run2.finish"),
    PICAMAction("physics_buffer_deallocate_leaf", "cam_run2", "leaf_pbuf_deallocate", "service", 466, parent_stage="cam_run2.finish"),
    PICAMAction("physics_buffer_time_advance_leaf", "cam_run2", "leaf_pbuf_update_tim_idx", "service", 467, parent_stage="cam_run2.finish"),
    PICAMAction("diagnostics_deallocate_leaf", "cam_run2", "leaf_diag_deallocate", "service", 468, parent_stage="cam_run2.finish"),
    PICAMAction("physics_to_dynamics", "cam_run2", "stepon_run2", "coupling", 413),
    PICAMAction("dynamics", "cam_run3", "stepon_run3", "dynamics", 414),
    PICAMAction("history", "cam_run4", "wshist", "io", 415),
    PICAMAction("restart", "cam_run4", "restart", "io", 416),
    PICAMAction("finish", "cam_run4", "wrapup", "io", 417, False),
    PICAMAction("wrapup_leaf", "cam_run4", "leaf_cam_run4_wrapup", "io", 470, parent_stage="cam_run4.finish"),
    PICAMAction("step_cost_leaf", "cam_run4", "leaf_cam_run4_step_cost", "io", 471, parent_stage="cam_run4.finish"),
    PICAMAction("flush_leaf", "cam_run4", "leaf_cam_run4_flush", "io", 472, parent_stage="cam_run4.finish"),
    PICAMAction("advance_timestep", "clock", "advance_timestep", "clock", 418),
    PICAMAction("dynamics_to_physics", "cam_run1", "stepon_run1", "coupling", 419),
    PICAMAction("prepare", "cam_run1", "prepare_cam_run1", "kernel", 420),
    PICAMAction("state_initialize", "cam_run1", "bc_init", "scheme", 421),
    PICAMAction("energy_fixer", "cam_run1", "check_energy_fix", "scheme", 422),
    PICAMAction("dry_adjustment", "cam_run1", "dadadj", "scheme", 423),
    PICAMAction("deep_convection", "cam_run1", "convect_deep_tend", "scheme", 424),
    PICAMAction("shallow_convection", "cam_run1", "convect_shallow_tend", "scheme", 425),
    PICAMAction("sea_salt_rebin", "cam_run1", "sslt_rebin_adv", "scheme", 426),
    PICAMAction("cloud_macro_microphysics", "cam_run1", "macro_microphysics", "scheme", 427),
    # The macrophysics stage stopped before its driver and resumed after it,
    # so a Python transliteration of the driver can run in between.  Off by
    # default: Macrophysics.attach enables these and disables their parent.
    PICAMAction("macro_tend_pre_leaf", "cam_run1", "leaf_macro_tend_pre", "scheme", 480, False, parent_stage="cam_run1.cloud_macro_microphysics"),
    PICAMAction("macro_tend_post_leaf", "cam_run1", "leaf_macro_tend_post", "scheme", 481, False, parent_stage="cam_run1.cloud_macro_microphysics"),
    PICAMAction("wet_deposition", "cam_run1", "aero_model_wetdep", "scheme", 428, False),
    PICAMAction("modal_aerosol_preparation_leaf", "cam_run1", "leaf_modal_aero_prepare", "scheme", 450, parent_stage="cam_run1.wet_deposition"),
    PICAMAction("aerosol_wet_deposition_leaf", "cam_run1", "leaf_aero_model_wetdep", "scheme", 451, parent_stage="cam_run1.wet_deposition"),
    PICAMAction("carma_wet_deposition_leaf", "cam_run1", "leaf_carma_wetdep_tend", "scheme", 452, parent_stage="cam_run1.wet_deposition"),
    PICAMAction("convective_tracer_transport_leaf", "cam_run1", "leaf_convect_deep_tend_2", "scheme", 453, parent_stage="cam_run1.wet_deposition"),
    PICAMAction("diagnostics", "cam_run1", "physics_diagnostics", "scheme", 429, False),
    PICAMAction("state_and_convection_diagnostics_leaf", "cam_run1", "leaf_diag_phys_writeout", "scheme", 454, parent_stage="cam_run1.diagnostics"),
    PICAMAction("cloud_diagnostics_leaf", "cam_run1", "leaf_cloud_diagnostics_calc", "scheme", 455, parent_stage="cam_run1.diagnostics"),
    PICAMAction("radiation", "cam_run1", "radiation_tend", "scheme", 430),
    PICAMAction("state_export", "cam_run1", "cam_export", "scheme", 431, False),
    PICAMAction("tropopause_leaf", "cam_run1", "leaf_tropopause_output", "scheme", 456, parent_stage="cam_run1.state_export"),
    PICAMAction("state_export_leaf", "cam_run1", "leaf_cam_export", "scheme", 457, parent_stage="cam_run1.state_export"),
    PICAMAction("export_diagnostics_leaf", "cam_run1", "leaf_diag_export", "scheme", 458, parent_stage="cam_run1.state_export"),
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

    def split_macrophysics(self, *, experimental: bool = False) -> None:
        """Stop the macrophysics stage before its driver and resume after it.

        The whole stage and its two halves are alternatives, never both:
        leaving the stage enabled beside its halves would run the
        macrophysics twice a step, and disabling all three would not run it
        at all.  Doing the swap in one call is the only way to be sure of
        neither.
        """

        if not experimental:
            raise PICAMConfigurationError(
                "splitting the macrophysics stage requires experimental=True"
            )
        self.set_enabled("cloud_macro_microphysics", False, phase="cam_run1", experimental=True)
        for name in ("macro_tend_pre_leaf", "macro_tend_post_leaf"):
            self.set_enabled(name, True, phase="cam_run1", experimental=True)

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
        # ``phase`` identifies the action being moved.  It must not also
        # constrain the target: the Python workflow is one ordered sequence,
        # and a process may intentionally move across a CAM source phase.
        target = self.select(before or after or "")
        if selected is target:
            raise PICAMConfigurationError(
                "an action cannot be moved relative to itself"
            )
        original = list(self._actions)
        try:
            self._actions.remove(selected)
            target_index = self._actions.index(target)
            self._actions.insert(target_index + (after is not None), selected)
            self._validate()
        except BaseException:
            self._actions[:] = original
            raise

    def replace_enabled_order(
        self,
        order: Iterable[str],
        *,
        experimental: bool = False,
    ) -> None:
        """Atomically replace the order of the enabled workflow actions.

        Disabled catalog entries are deliberately kept in their existing
        slots.  The caller must provide every currently enabled action exactly
        once: list assignment changes ordering, while ``enable``/``disable``
        remain the explicit APIs for changing membership.
        """

        if not experimental:
            raise PICAMConfigurationError(
                "changing the scientific PI-CAM order requires experimental=True"
            )
        requested = tuple(self.select(str(name)) for name in order)
        enabled = tuple(action for action in self._actions if action.enabled)
        requested_names = tuple(action.qualified_name for action in requested)
        enabled_names = tuple(action.qualified_name for action in enabled)
        if len(requested_names) != len(set(requested_names)):
            raise PICAMConfigurationError(
                "workflow order contains the same enabled action more than once"
            )
        if set(requested_names) != set(enabled_names):
            missing = sorted(set(enabled_names) - set(requested_names))
            extra = sorted(set(requested_names) - set(enabled_names))
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if extra:
                details.append("not enabled " + ", ".join(extra))
            raise PICAMConfigurationError(
                "workflow list must contain every enabled action exactly once"
                + (": " + "; ".join(details) if details else "")
            )

        original = list(self._actions)
        replacements = iter(requested)
        try:
            self._actions[:] = [
                next(replacements) if action.enabled else action
                for action in self._actions
            ]
            self._validate()
        except BaseException:
            self._actions[:] = original
            raise

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
        # Placement is defined by the one complete Python workflow.  ``phase``
        # remains source provenance for diagnostics; it must not prevent a
        # runtime process from being inserted beside an action that originated
        # in another legacy CAM phase.
        target = self.select(before or after or "")
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
        if selected.kind not in {
            "python_process",
            "python_history",
            "runtime_fortran_process",
            "runtime_catalog_process",
        }:
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
                "control_owner": action.control_owner,
                "implementation": action.implementation,
                "granularity": (
                    "leaf" if action.operation.startswith("leaf_") else "stage"
                ),
                "parent_stage": action.parent_stage,
                "native_id": action.native_id,
                "enabled": action.enabled,
            }
            for index, action in enumerate(self._actions)
        )
