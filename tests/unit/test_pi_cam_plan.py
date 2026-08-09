import pytest

from freecam.pi_cam import PICAMConfigurationError, PICAMStepPlan


def test_pi_cam_default_plan_matches_cesm_cam_source_order() -> None:
    plan = PICAMStepPlan.default()

    assert [item.operation for item in plan.actions] == [
        "boundary_import",
        "prepare",
        "chem_emissions",
        "tracers_chemistry",
        "vertical_diffusion_tend",
        "rayleigh_friction_tend",
        "aero_model_drydep",
        "charge_fix",
        "gw_tend",
        "qbo_relax",
        "iondrag_calc",
        "physics_dme_adjust",
        "finish",
        "stepon_run2",
        "stepon_run3",
        "wshist",
        "restart",
        "wrapup",
        "advance_timestep",
        "stepon_run1",
        "prepare_cam_run1",
        "bc_init",
        "check_energy_fix",
        "dadadj",
        "convect_deep_tend",
        "convect_shallow_tend",
        "sslt_rebin_adv",
        "macro_microphysics",
        "aero_model_wetdep",
        "leaf_modal_aero_prepare",
        "leaf_aero_model_wetdep",
        "leaf_carma_wetdep_tend",
        "leaf_convect_deep_tend_2",
        "physics_diagnostics",
        "leaf_diag_phys_writeout",
        "leaf_cloud_diagnostics_calc",
        "radiation_tend",
        "cam_export",
        "leaf_tropopause_output",
        "leaf_cam_export",
        "leaf_diag_export",
        "boundary_export",
    ]
    assert [item.native_id for item in plan.actions] == [
        202,
        *range(401, 429),
        *range(450, 454),
        429,
        454,
        455,
        430,
        431,
        *range(456, 459),
        432,
    ]
    assert len(plan.actions) == 42
    assert len(tuple(plan)) == 33
    assert len(plan.in_phase("cam_run1")) == 13
    assert sum(action.phase == "cam_run1" for action in plan.actions) == 22


def test_pi_cam_plan_changes_are_explicitly_experimental() -> None:
    plan = PICAMStepPlan.default()

    with pytest.raises(PICAMConfigurationError, match="experimental=True"):
        plan.set_enabled("dadadj", False, phase="cam_run1")

    plan.set_enabled("dadadj", False, phase="cam_run1", experimental=True)
    assert not plan.select("dadadj", phase="cam_run1").enabled


def test_pi_cam_leaf_processes_can_be_reordered_only_experimentally() -> None:
    plan = PICAMStepPlan.default()

    with pytest.raises(PICAMConfigurationError, match="experimental=True"):
        plan.move(
            "cloud_diagnostics_leaf", phase="cam_run1", before="radiation"
        )
    assert not plan.select("cloud_diagnostics_leaf", phase="cam_run1").enabled
    plan.set_enabled(
        "cloud_diagnostics_leaf", True, phase="cam_run1", experimental=True
    )

    plan.move(
        "cloud_diagnostics_leaf",
        phase="cam_run1",
        before="radiation",
        experimental=True,
    )
    names = [action.name for action in plan.in_phase("cam_run1")]
    assert names.index("cloud_diagnostics_leaf") < names.index("radiation")


def test_cam_run1_leaf_expansion_replaces_composites_in_source_order() -> None:
    plan = PICAMStepPlan.default()

    plan.expand_cam_run1_leaves(experimental=True)

    names = tuple(action.name for action in plan.in_phase("cam_run1"))
    assert "wet_deposition" not in names
    assert "diagnostics" not in names
    assert "state_export" not in names
    assert names[
        names.index("cloud_macro_microphysics") + 1 : names.index("radiation")
    ] == (
        "modal_aerosol_preparation_leaf",
        "aerosol_wet_deposition_leaf",
        "carma_wet_deposition_leaf",
        "convective_tracer_transport_leaf",
        "state_and_convection_diagnostics_leaf",
        "cloud_diagnostics_leaf",
    )
    assert names[-3:] == (
        "tropopause_leaf",
        "state_export_leaf",
        "export_diagnostics_leaf",
    )


def test_cam_run1_leaf_expansion_requires_explicit_experimental_flag() -> None:
    plan = PICAMStepPlan.default()

    with pytest.raises(PICAMConfigurationError, match="experimental=True"):
        plan.expand_cam_run1_leaves()
