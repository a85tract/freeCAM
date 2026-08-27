"""tphysbc's stop/resume around macrop_driver_tend, and the leaves that drive it."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import generate_pi_cam_macro_boundary as boundary  # noqa: E402
from apply_pi_cam_source_patches import PATCHES  # noqa: E402
from build_pi_cam_devices import (  # noqa: E402
    LEAF_OPERATION_IDS, LEAF_OPERATION_NAMES, LEAF_PATCHES, SUPPORT_MODULES,
)

from freecam.pi_cam.plan import PICAMStepPlan  # noqa: E402

pinned = pytest.mark.skipif(
    not (boundary.PINNED / boundary.RELATIVE).is_file(),
    reason="the pinned iCESM submodule is not checked out",
)


@pinned
def test_the_patches_are_what_the_generator_produces_on_their_own_bases() -> None:
    """Each patch is diffed against its set's own predecessor state.

    A zero-context hunk that no longer matches its line is *searched for* by
    git apply, so a stale offset surfaces as an edit in the wrong place, not
    an error.  Regenerating on the true base and comparing is the guard.
    """

    rendered = boundary.render()
    assert boundary.BOUNDARY.read_text() == rendered[boundary.BOUNDARY]
    assert boundary.DISPATCH.read_text() == rendered[boundary.DISPATCH]


def test_each_patch_ships_in_the_set_that_has_to_prove_it() -> None:
    production = [Path(name).name for name in PATCHES]
    add_on = [path.name for path in LEAF_PATCHES]
    # The boundary edits tphysbc, which every run executes: production set,
    # answering to the bit-for-bit gate.
    assert "0039-macro-tend-boundary.patch" in production
    # Radiation's boundary is the same kind of edit and ships the same way.
    assert production[-1] == "0041-rad-tend-boundary.patch"
    # The dispatcher only widens the leaf entry point Python drives: add-on
    # set, last, since it edits what 0031 leaves behind.
    assert "0040-macro-tend-leaf-dispatch.patch" in add_on
    assert add_on[-1] == "0042-rad-tend-leaf-dispatch.patch"
    for name in ("0040-macro-tend-leaf-dispatch.patch", "0042-rad-tend-leaf-dispatch.patch"):
        assert name not in production
    # Neither touches a numerical object.
    for name in (boundary.BOUNDARY, boundary.DISPATCH):
        text = name.read_text()
        assert text.startswith("--- a/src/physics/cam/physpkg.F90")
        assert "macrop_driver.F90" not in text
        assert "radiation.F90" not in text


@pinned
def test_the_original_driver_call_is_kept_and_only_wrapped() -> None:
    patch = boundary.BOUNDARY.read_text()
    removed = "\n".join(l[1:] for l in patch.splitlines() if l.startswith("-"))
    added = "\n".join(l[1:] for l in patch.splitlines() if l.startswith("+"))
    # the call moves by indentation only: every argument line is re-added
    for fragment in ("call macrop_driver_tend( &", "pbuf,            det_s,          det_ice)",
                     "cam_in%landfrac, cam_in%ocnfrac, cam_in%snowhland"):
        assert fragment in removed and fragment in added
    # stage 1 leaves before the call; stage 2 takes Python's answer only when
    # Python claimed the step, and otherwise runs the original driver
    assert "if (macro_stage_local == 1) then" in added
    assert "if (macro_stage_local == 2 .and. python_owns_tend) then" in added
    assert "ptend   = macro_ptend(lchnk)" in added
    assert "det_ice = macro_det_ice(:,lchnk)" in added
    assert added.count("call macrop_driver_tend( &") == 1
    # a claimed step with nothing produced is an abort, not a silent zero
    assert "Python claimed the macrophysics step but produced no tendencies" in added


@pinned
def test_the_configurations_the_boundary_cannot_carry_are_refused() -> None:
    added = "\n".join(l[1:] for l in boundary.BOUNDARY.read_text().splitlines() if l.startswith("+"))
    for guard in ("cld_macmic_num_steps /= 1", "carma_do_cldice .or. carma_do_cldliq",
                  "micro_do_icesupersat", "microp_scheme /= 'MG'", "macrop_scheme == 'CLUBB_SGS'"):
        assert guard in added, guard
        assert "call endrun" in added


@pinned
def test_the_leaves_re_enter_stage_seven_with_the_stage_number() -> None:
    added = "\n".join(l[1:] for l in boundary.DISPATCH.read_text().splitlines() if l.startswith("+"))
    assert "cam_out(c), cam_in(c), 7, action_id - 20)" in added
    assert "if (action_id >= 21) then" in added
    assert "action_id > 22" in added
    # the plain leaves keep their own dispatcher
    assert "call tphysbc_leaf_action(action_id, ztodt, phys_state(c), &" in added


def test_the_two_leaves_are_wired_end_to_end_and_off_by_default() -> None:
    ids = dict(zip(LEAF_OPERATION_NAMES, LEAF_OPERATION_IDS))
    assert ids["leaf_macro_tend_pre"] == 480
    assert ids["leaf_macro_tend_post"] == 481
    assert len(set(LEAF_OPERATION_IDS)) == len(LEAF_OPERATION_IDS) == len(LEAF_OPERATION_NAMES)
    adapter = (REPO / "native/pi_cam/pi_cam_leaf_adapter.F90").read_text()
    # 480/481 share their dispatch block with radiation's 482/483
    assert "case (480:483)" in adapter
    assert "cam_phys_run1_leaf_action(action_id - 459" in adapter   # 480 -> 21, 481 -> 22

    plan = PICAMStepPlan.default()
    halves = [a for a in plan.actions if a.operation in ("leaf_macro_tend_pre", "leaf_macro_tend_post")]
    assert [a.native_id for a in halves] == [480, 481]
    assert not any(a.enabled for a in halves)
    assert all(a.parent_stage == "cam_run1.cloud_macro_microphysics" for a in halves)
    stage = next(a for a in plan.actions if a.operation == "macro_microphysics")
    assert stage.enabled


def test_the_support_modules_are_additions_the_image_links() -> None:
    assert SUPPORT_MODULES == ("pycam_macro_kernels.F90", "pycam_macro_handles.F90",
                               "pycam_rad_kernels.F90", "pycam_rad_handles.F90",
                               "pycam_micro_kernels.F90", "pycam_mm_kernels.F90",
                               "pycam_aero_kernels.F90",
                               "pycam_micro_handles.F90", "pycam_aero_handles.F90",
                               "pycam_mm_handles.F90")
    for name in SUPPORT_MODULES:
        assert (REPO / "native/pi_cam/support" / name).is_file()
    builder = (REPO / "tools/build_pi_cam_devices.py").read_text()
    # compiled before the control objects that `use` them, linked into the
    # fixed image as explicit objects, never as archive replacements
    assert builder.index("for source_name in SUPPORT_MODULES:") < builder.index(
        'for source_name in (\n        "physpkg.F90", "cam_comp.F90", "atm_comp_mct.F90",\n    ):')
    assert "*(str(path) for path in support_objects)," in builder


def test_split_macrophysics_swaps_the_stage_for_its_halves_in_one_call() -> None:
    from freecam.pi_cam.errors import PICAMConfigurationError

    plan = PICAMStepPlan.default()
    with pytest.raises(PICAMConfigurationError):
        plan.split_macrophysics()
    plan.split_macrophysics(experimental=True)
    state = {a.operation: a.enabled for a in plan.actions
             if a.operation in ("macro_microphysics", "leaf_macro_tend_pre", "leaf_macro_tend_post")}
    assert state == {"macro_microphysics": False, "leaf_macro_tend_pre": True, "leaf_macro_tend_post": True}
    # the halves run where the stage ran, in order
    order = [a.operation for a in plan]
    assert order.index("leaf_macro_tend_pre") + 1 == order.index("leaf_macro_tend_post")
    assert order.index("sslt_rebin_adv") < order.index("leaf_macro_tend_pre") < order.index("leaf_modal_aero_prepare") or \
        order.index("sslt_rebin_adv") < order.index("leaf_macro_tend_pre") < order.index("aero_model_wetdep")


@pinned
def test_the_boundary_hands_out_the_forcing_the_driver_was_called_with() -> None:
    """dlf, dlf2, cmfmc, cmfmc2, zdu, rliq, wtdlf live in physpkg's private
    buffers; a transliteration needs their addresses, nothing more."""

    added = "\n".join(l[1:] for l in boundary.BOUNDARY.read_text().splitlines() if l.startswith("+"))
    assert "bind(C, name='pycam_macro_forcing_v1')" in added
    for name in ("pycesm_bc_zdu", "pycesm_bc_cmfmc", "pycesm_bc_cmfmc2", "pycesm_bc_dlf",
                 "pycesm_bc_dlf2", "pycesm_bc_rliq", "pycesm_bc_wtdlf"):
        assert f"{name}(" in added, name
    # the buffers are not TARGET and their declarations are a shared anchor, so
    # the address goes through a TARGET dummy and no `public` line is added
    assert "real(r8), target, intent(in) :: array(:,:)" in added
    assert "public pycam_macro_forcing_v1" not in added
