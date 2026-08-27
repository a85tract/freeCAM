"""The radiation boundary: tphysbc stops before its driver and resumes after.

A Python-driven radiation step walks radiation_tend's statements itself, so
tphysbc has to pause either side of the call it no longer makes.  These tests
hold the two generated patches to what makes that safe: they edit only a
control object, the original call survives verbatim as the fallback, the
configuration the transliteration cannot carry is refused, and a step Python
claims but does not answer aborts rather than passing a zero tendency on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import generate_pi_cam_rad_boundary as boundary  # noqa: E402
from apply_pi_cam_source_patches import PATCHES  # noqa: E402
from build_pi_cam_devices import (  # noqa: E402
    LEAF_OPERATION_IDS, LEAF_OPERATION_NAMES, LEAF_PATCHES,
)
from freecam.pi_cam.plan import PICAMStepPlan  # noqa: E402

PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/physpkg.F90"
pinned = pytest.mark.skipif(not PINNED.is_file(),
                            reason="the pinned iCESM submodule is not checked out")


def _added(path: Path) -> str:
    return "\n".join(l[1:] for l in path.read_text().splitlines() if l.startswith("+"))


def _removed(path: Path) -> str:
    return "\n".join(l[1:] for l in path.read_text().splitlines() if l.startswith("-"))


# -- what the patches are allowed to touch ---------------------------------------


@pinned
def test_the_committed_patches_are_what_the_generator_writes() -> None:
    for path, text in boundary.render().items():
        assert path.read_text() == text, path.name


def test_neither_patch_edits_a_numerical_object() -> None:
    for path in (boundary.BOUNDARY, boundary.DISPATCH):
        text = path.read_text()
        assert text.startswith("--- a/src/physics/cam/physpkg.F90"), path.name
        for numerical in ("radiation.F90", "radsw.F90", "radlw.F90", "rrtmg_state.F90",
                          "macrop_driver.F90"):
            assert numerical not in text, f"{path.name} touches {numerical}"


def test_each_patch_ships_in_the_set_that_has_to_prove_it() -> None:
    production = [Path(name).name for name in PATCHES]
    add_on = [path.name for path in LEAF_PATCHES]
    # the boundary edits tphysbc, which every run executes, so it answers to
    # the bit-for-bit gate as part of the production set
    assert "0041-rad-tend-boundary.patch" in production
    assert "0041-rad-tend-boundary.patch" not in add_on
    # the dispatcher only widens the leaf entry point Python drives
    assert "0042-rad-tend-leaf-dispatch.patch" in add_on
    assert "0042-rad-tend-leaf-dispatch.patch" not in production


# -- the call itself -------------------------------------------------------------


@pinned
def test_the_original_driver_call_is_kept_and_only_wrapped() -> None:
    added, removed = _added(boundary.BOUNDARY), _removed(boundary.BOUNDARY)
    # the call moves by indentation only: every argument line is re-added
    for fragment in ("call radiation_tend(state,ptend, pbuf, &",
                     "cam_in%landfrac, cam_in%icefrac, cam_in%snowhland,",
                     "fsds, net_flx)"):
        assert fragment in removed and fragment in added, fragment
    assert added.count("call radiation_tend(state,ptend, pbuf, &") == 1


@pinned
def test_stage_one_binds_the_chunk_then_leaves_and_stage_two_takes_the_answer() -> None:
    added = _added(boundary.BOUNDARY)
    # cam_in and cam_out are dummies of tphysbc, so their addresses can only
    # be taken while the chunk's own objects are in scope
    assert "call pycam_rad_bind_chunk(lchnk, cam_in, cam_out)" in added
    assert "if (rad_stage_local == 1) then" in added
    assert "call t_stopf('radiation')" in added
    # the resume takes Python's answer only when Python claimed the step
    assert "if (rad_stage_local == 2 .and. python_owns_rad) then" in added
    assert "ptend   = rad_ptend(lchnk)" in added
    assert "net_flx = rad_net_flx(:,lchnk)" in added


@pinned
def test_a_claimed_step_with_no_tendency_aborts_rather_than_passing_a_zero() -> None:
    added = _added(boundary.BOUNDARY)
    assert "if (.not. allocated(rad_ptend(lchnk)%s)) call endrun &" in added
    assert "Python claimed the radiation step but produced no tendency" in added


@pinned
def test_everything_after_the_call_is_left_exactly_where_it_was() -> None:
    """flx_net, physics_update and check_energy_chng are the reason for a
    resume rather than a Python reimplementation, so the patch must not move
    them."""

    removed = _removed(boundary.BOUNDARY)
    for untouched in ("tend%flx_net(i) = net_flx(i)",
                      "call physics_update(state, ptend, ztodt, tend)",
                      "check_energy_chng(state, tend, \"radheat\""):
        assert untouched not in removed, untouched


@pinned
def test_the_configuration_the_boundary_cannot_carry_is_refused() -> None:
    added = _added(boundary.BOUNDARY)
    assert "if (single_column .or. scm_crm_mode) call endrun &" in added
    assert "the radiation boundary does not carry the single-column path" in added
    # the rest need module state physpkg cannot see; Python refuses those at
    # attach, and says so here so the division of labour is not a surprise
    assert "made by Python at" in added


# -- the leaves ------------------------------------------------------------------


@pinned
def test_the_leaves_re_enter_stage_ten_with_the_stage_number() -> None:
    added = _added(boundary.DISPATCH)
    assert "cam_out(c), cam_in(c), 10, &" in added
    assert "rad_stage=action_id - 22" in added        # 23 -> 1, 24 -> 2
    assert "if (action_id >= 23) then" in added
    assert "else if (action_id >= 21) then" in added  # macrophysics keeps its branch
    assert "if (action_id < 12 .or. action_id > 24) then" in added


def test_the_two_leaves_are_wired_end_to_end_and_off_by_default() -> None:
    ids = dict(zip(LEAF_OPERATION_NAMES, LEAF_OPERATION_IDS))
    assert ids["leaf_rad_tend_pre"] == 482
    assert ids["leaf_rad_tend_post"] == 483
    assert len(set(LEAF_OPERATION_IDS)) == len(LEAF_OPERATION_IDS) == len(LEAF_OPERATION_NAMES)
    adapter = (REPO / "native/pi_cam/pi_cam_leaf_adapter.F90").read_text()
    assert "case (480:483)" in adapter
    # 482 -> 23, 483 -> 24, the ids the dispatcher branches on
    assert "cam_phys_run1_leaf_action(action_id - 459" in adapter

    plan = PICAMStepPlan.default()
    halves = [a for a in plan.actions
              if a.operation in ("leaf_rad_tend_pre", "leaf_rad_tend_post")]
    assert [a.native_id for a in halves] == [482, 483]
    assert not any(a.enabled for a in halves)
    assert all(a.parent_stage == "cam_run1.radiation" for a in halves)
    stage = next(a for a in plan.actions if a.operation == "radiation_tend")
    assert stage.enabled


def test_split_radiation_swaps_the_stage_for_its_halves_in_one_call() -> None:
    from freecam.pi_cam.errors import PICAMConfigurationError

    plan = PICAMStepPlan.default()
    with pytest.raises(PICAMConfigurationError):
        plan.split_radiation()
    plan.split_radiation(experimental=True)
    state = {a.operation: a.enabled for a in plan.actions
             if a.operation in ("radiation_tend", "leaf_rad_tend_pre", "leaf_rad_tend_post")}
    assert state == {"radiation_tend": False,
                     "leaf_rad_tend_pre": True, "leaf_rad_tend_post": True}
    # the halves run where the stage ran, in order
    order = [a.operation for a in plan.actions if a.enabled and a.phase == "cam_run1"]
    assert order.index("leaf_rad_tend_pre") < order.index("leaf_rad_tend_post")
    assert order.index("leaf_cloud_diagnostics_calc") < order.index("leaf_rad_tend_pre")


def test_the_two_stages_can_be_split_together_without_colliding() -> None:
    plan = PICAMStepPlan.default()
    plan.split_macrophysics(experimental=True)
    plan.split_radiation(experimental=True)
    enabled = [a.operation for a in plan.actions if a.enabled]
    for half in ("leaf_macro_tend_pre", "leaf_macro_tend_post",
                 "leaf_rad_tend_pre", "leaf_rad_tend_post"):
        assert half in enabled, half
    for whole in ("macro_microphysics", "radiation_tend"):
        assert whole not in enabled, whole
    assert enabled.index("leaf_macro_tend_post") < enabled.index("leaf_rad_tend_pre")


# -- the support modules ---------------------------------------------------------


def test_the_radiation_support_modules_are_additions_the_image_links() -> None:
    from build_pi_cam_devices import SUPPORT_MODULES

    for name in ("pycam_rad_kernels.F90", "pycam_rad_handles.F90"):
        assert name in SUPPORT_MODULES, name
        assert (REPO / "native/pi_cam/support" / name).is_file(), name
    installed = dict((Path(a).name, Path(b).name)
                     for a, b in __import__("apply_pi_cam_source_patches").SUPPORT_SOURCES)
    assert installed["pycam_rad_kernels.F90"] == "pycam_rad_kernels.F90"
    assert installed["pycam_rad_handles.F90"] == "pycam_rad_handles.F90"
