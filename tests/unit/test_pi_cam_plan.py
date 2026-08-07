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
        "physics_diagnostics",
        "radiation_tend",
        "cam_export",
        "boundary_export",
    ]
    assert [item.native_id for item in plan.actions] == [
        202,
        *range(401, 433),
    ]


def test_pi_cam_plan_changes_are_explicitly_experimental() -> None:
    plan = PICAMStepPlan.default()

    with pytest.raises(PICAMConfigurationError, match="experimental=True"):
        plan.set_enabled("dadadj", False, phase="cam_run1")

    plan.set_enabled("dadadj", False, phase="cam_run1", experimental=True)
    assert not plan.select("dadadj", phase="cam_run1").enabled
