#!/usr/bin/env python3
"""Let tphysbc stop before macrop_driver_tend and resume after it.

A Python-driven macrophysics step never calls the Fortran driver; it walks
the driver's statements itself.  Its caller therefore has to pause just
before ``call macrop_driver_tend`` and pick up just after, taking the
tendencies and the detrainment integrals from where Python left them.
Two patches do that, both on physpkg.F90 -- a control object the device
builder already replaces, so no numerical object changes:

``0039-macro-tend-boundary.patch`` (production set) gives ``tphysbc`` an
optional ``macro_stage``: 1 returns before the call, 2 resumes after it.
When Python has not claimed the step, stage 2 calls the original driver
itself, which is what makes a run with the boundary enabled but no Python
process a bit-for-bit gate of the boundary alone.

``0040-macro-tend-leaf-dispatch.patch`` (leaf add-on set) teaches
``phys_run1_leaf_action`` two ids that re-enter tphysbc's stage 7 with those
stages, so Python can ask for each half by number.

Each patch is generated against the source as its own set leaves it -- the
pinned file with every earlier patch of that set applied -- so a stale
offset is a --check failure here rather than a wrong-place edit at build
time.

    tools/generate_pi_cam_macro_boundary.py            # write both patches
    tools/generate_pi_cam_macro_boundary.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from apply_pi_cam_source_patches import PATCHES  # noqa: E402
from build_pi_cam_devices import LEAF_PATCHES  # noqa: E402

PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam"
BOUNDARY = REPO / "native/pi_cam/control_patches/0039-macro-tend-boundary.patch"
DISPATCH = REPO / "native/pi_cam/control_patches/0040-macro-tend-leaf-dispatch.patch"
RELATIVE = "src/physics/cam/physpkg.F90"

# The other files the production patches touch; they have to be present for
# the earlier patches to apply, even though only physpkg.F90 is edited here.
COMPANIONS = ("src/cpl/atm_comp_mct.F90", "src/control/cam_comp.F90")


def _apply(patches, tree: Path) -> None:
    for patch in patches:
        subprocess.run(
            ["git", "apply", "--unidiff-zero", str(patch)],
            cwd=tree, check=True, capture_output=True, text=True,
        )


def _base(tree: Path, patches) -> Path:
    """The pinned physpkg.F90 with ``patches`` applied, inside ``tree``."""

    for relative in (RELATIVE, *COMPANIONS):
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PINNED / relative, target)
    _apply(patches, tree)
    return tree / RELATIVE


def _diff(before: Path, after: Path) -> str:
    diff = subprocess.run(
        ["git", "diff", "--no-index", "--unified=0", "--no-prefix", str(before), str(after)],
        capture_output=True, text=True,
    ).stdout.splitlines()
    body = [line for line in diff if not line.startswith(("diff --git", "index ", "--- ", "+++ "))]
    return "\n".join([f"--- a/{RELATIVE}", f"+++ b/{RELATIVE}"] + body) + "\n"


def _insert_after(lines: list[str], anchor: str, added: list[str], start: int = 0) -> None:
    index = lines.index(anchor, start)
    lines[index + 1:index + 1] = added


# -- 0039: the boundary inside tphysbc -----------------------------------------


def edit_boundary(lines: list[str]) -> list[str]:
    out = list(lines)
    tphysbc = out.index("subroutine tphysbc (ztodt,               &")

    out[out.index("       sgh30, cam_out, cam_in, action_id)", tphysbc)] = \
        "       sgh30, cam_out, cam_in, action_id, macro_stage)"

    _insert_after(out, "    use macrop_driver,   only: macrop_driver_tend", [
        "    use pycam_macro_handles, only: macro_ptend, macro_det_s, macro_det_ice, &",
        "         python_owns_tend",
    ], tphysbc)

    _insert_after(out,
        "    integer, intent(in), optional :: action_id             ! Python-dispatched process boundary", [
        "    ! Where to pause around the macrophysics driver.  Absent or 0 runs",
        "    ! the stage whole.  1 returns just before call macrop_driver_tend;",
        "    ! 2 resumes just after it, taking the tendencies from Python if",
        "    ! Python claimed the step and calling the driver itself otherwise.",
        "    integer, intent(in), optional :: macro_stage",
    ], tphysbc)

    _insert_after(out, "    integer :: stage                          ! 0 is the original monolithic path", [
        "    integer :: macro_stage_local              ! 0 whole, 1 stop before, 2 resume after",
    ], tphysbc)

    _insert_after(out, "    if (present(action_id)) stage = action_id", [
        "    macro_stage_local = 0",
        "    if (present(macro_stage)) macro_stage_local = macro_stage",
        "    if (macro_stage_local /= 0) then",
        "       ! The resume re-enters this routine and reruns everything in the",
        "       ! stage that precedes the driver call.  That is only safe where",
        "       ! the preceding work is idempotent, so the paths that are not",
        "       ! are refused rather than silently run twice.",
        "       if (cld_macmic_num_steps /= 1) call endrun &",
        "            ('TPHYSBC: the macrophysics boundary needs cld_macmic_num_steps = 1')",
        "       if (carma_do_cldice .or. carma_do_cldliq) call endrun &",
        "            ('TPHYSBC: the macrophysics boundary does not carry CARMA cloud precipitation')",
        "       if (micro_do_icesupersat) call endrun &",
        "            ('TPHYSBC: the macrophysics boundary does not carry the ice-supersaturation aerosol step')",
        "    end if",
    ], tphysbc)

    _insert_after(out, "            state_debug_checks_out=state_debug_checks)", [
        "       if (macro_stage_local /= 0) then",
        "          if (microp_scheme /= 'MG') call endrun &",
        "               ('TPHYSBC: the macrophysics boundary is for the MG microphysics branch')",
        "          if (macrop_scheme == 'CLUBB_SGS') call endrun &",
        "               ('TPHYSBC: the macrophysics boundary is for the Park macrophysics, not CLUBB')",
        "       end if",
    ], tphysbc)

    # The call itself stays where it is, character for character.  A stop
    # goes in front of it, and the resume becomes the other branch of an
    # if/else around it.
    first = out.index("             call macrop_driver_tend( &", tphysbc)
    out[first:first] = [
        "             if (macro_stage_local == 1) then",
        "                ! Python runs here and calls back in with macro_stage = 2.",
        "                call t_stopf('macrop_tend')",
        "                return",
        "             end if",
        "",
        "             if (macro_stage_local == 2 .and. python_owns_tend) then",
        "                ! The driver's outputs, as Python's transliteration left them.",
        "                if (macro_ptend(lchnk)%psetcols < 1) call endrun &",
        "                     ('TPHYSBC: Python claimed the macrophysics step but produced no tendencies')",
        "                ptend   = macro_ptend(lchnk)",
        "                det_s   = macro_det_s(:,lchnk)",
        "                det_ice = macro_det_ice(:,lchnk)",
        "             else",
    ]
    start = out.index("             call macrop_driver_tend( &", first)
    last = out.index("                  pbuf,            det_s,          det_ice)", start)
    for index in range(start, last + 1):
        out[index] = "   " + out[index]
    out[last + 1:last + 1] = ["             end if"]

    # The driver's forcing arguments -- dlf, dlf2, cmfmc, cmfmc2, zdu, wtdlf,
    # rliq -- are tphysbc locals restored from physpkg's private pycesm_bc_*
    # buffers at every stage.  A Python transliteration needs the same values,
    # so physpkg gets one accessor that hands out their addresses.  Private
    # storage stays private; only the address of one chunk's slice leaves.
    # No `public` statement: a bind(C) procedure is reachable by its C name
    # whatever its Fortran accessibility, and the module's public list is an
    # anchor the leaf patches share.
    end = out.index("end module physpkg")
    out[end:end] = [
        "",
        "integer(c_int) function pycam_macro_forcing_v1(lchnk, code, ptr, ndims, extents) &",
        "     bind(C, name='pycam_macro_forcing_v1') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr, c_null_ptr",
        "  integer(c_int), value, intent(in) :: lchnk, code",
        "  type(c_ptr), intent(out) :: ptr",
        "  integer(c_int), intent(out) :: ndims",
        "  integer(c_int64_t), intent(out) :: extents(4)",
        "  ptr = c_null_ptr",
        "  ndims = 0_c_int",
        "  extents = 0_c_int64_t",
        "  status = 1_c_int",
        "  if (.not. allocated(pycesm_bc_dlf)) return",
        "  if (lchnk < begchunk .or. lchnk > endchunk) then",
        "    status = 2_c_int",
        "    return",
        "  endif",
        "  ! The buffers are not declared TARGET (their declarations are an anchor",
        "  ! the leaf patches share), so the address is taken through a TARGET",
        "  ! dummy: no copy is made for a contiguous chunk slice.",
        "  select case (code)",
        "  case (1)",
        "    call pycam_macro_address2(pycesm_bc_zdu(:,:,lchnk), ptr, ndims, extents)",
        "  case (2)",
        "    call pycam_macro_address2(pycesm_bc_cmfmc(:,:,lchnk), ptr, ndims, extents)",
        "  case (3)",
        "    call pycam_macro_address2(pycesm_bc_cmfmc2(:,:,lchnk), ptr, ndims, extents)",
        "  case (4)",
        "    call pycam_macro_address2(pycesm_bc_dlf(:,:,lchnk), ptr, ndims, extents)",
        "  case (5)",
        "    call pycam_macro_address2(pycesm_bc_dlf2(:,:,lchnk), ptr, ndims, extents)",
        "  case (6)",
        "    call pycam_macro_address1(pycesm_bc_rliq(:,lchnk), ptr, ndims, extents)",
        "  case (7)",
        "    call pycam_macro_address3(pycesm_bc_wtdlf(:,:,:,lchnk), ptr, ndims, extents)",
        "  case default",
        "    status = 3_c_int",
        "    return",
        "  end select",
        "  status = 0_c_int",
        "end function pycam_macro_forcing_v1",
        "",
        "subroutine pycam_macro_address1(array, ptr, ndims, extents)",
        "  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr, c_loc",
        "  real(r8), target, intent(in) :: array(:)",
        "  type(c_ptr), intent(out) :: ptr",
        "  integer(c_int), intent(out) :: ndims",
        "  integer(c_int64_t), intent(out) :: extents(4)",
        "  ptr = c_loc(array(1)); ndims = 1_c_int",
        "  extents(1) = int(size(array,1), c_int64_t)",
        "end subroutine pycam_macro_address1",
        "",
        "subroutine pycam_macro_address2(array, ptr, ndims, extents)",
        "  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr, c_loc",
        "  real(r8), target, intent(in) :: array(:,:)",
        "  type(c_ptr), intent(out) :: ptr",
        "  integer(c_int), intent(out) :: ndims",
        "  integer(c_int64_t), intent(out) :: extents(4)",
        "  ptr = c_loc(array(1,1)); ndims = 2_c_int",
        "  extents(1) = int(size(array,1), c_int64_t)",
        "  extents(2) = int(size(array,2), c_int64_t)",
        "end subroutine pycam_macro_address2",
        "",
        "subroutine pycam_macro_address3(array, ptr, ndims, extents)",
        "  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr, c_loc",
        "  real(r8), target, intent(in) :: array(:,:,:)",
        "  type(c_ptr), intent(out) :: ptr",
        "  integer(c_int), intent(out) :: ndims",
        "  integer(c_int64_t), intent(out) :: extents(4)",
        "  ptr = c_loc(array(1,1,1)); ndims = 3_c_int",
        "  extents(1) = int(size(array,1), c_int64_t)",
        "  extents(2) = int(size(array,2), c_int64_t)",
        "  extents(3) = int(size(array,3), c_int64_t)",
        "end subroutine pycam_macro_address3",
    ]
    return out


# -- 0040: the leaf dispatch -------------------------------------------------------


def edit_dispatch(lines: list[str]) -> list[str]:
    out = list(lines)
    start = out.index("subroutine phys_run1_leaf_action(action_id, phys_state, ztodt, phys_tend, &")
    out[out.index("    use water_tracer_vars, only: wtrc_nwset", start)] = (
        "    use water_tracer_vars, only: wtrc_nwset\n"
        "    use comsrf, only: fsns, fsnt, flns, flnt, fsds, landm, sgh30"
    )
    out[out.index("    if (action_id < 12 .or. action_id > 20) then", start)] = \
        "    if (action_id < 12 .or. action_id > 22) then"
    loop = out.index("!$OMP PARALLEL DO PRIVATE (C, phys_buffer_chunk)", start)
    end = out.index("    end do", loop)
    out[loop:end + 1] = [
        "    if (action_id >= 21) then",
        "       ! The two halves of the macrophysics stage.  Unlike the other",
        "       ! leaves these are not separate routines: they re-enter tphysbc's",
        "       ! own stage 7, which stops before the macrophysics driver and",
        "       ! resumes after it, so the statement order the bit-for-bit gate",
        "       ! depends on is the original one.",
        "!$OMP PARALLEL DO PRIVATE (C, phys_buffer_chunk)",
        "       do c=begchunk,endchunk",
        "          phys_buffer_chunk => pbuf_get_chunk(pbuf2d, c)",
        "          call tphysbc(ztodt, fsns(1,c), fsnt(1,c), flns(1,c), flnt(1,c), &",
        "               phys_state(c), phys_tend(c), phys_buffer_chunk, fsds(1,c), &",
        "               landm(1,c), sgh30(1,c), cam_out(c), cam_in(c), 7, action_id - 20)",
        "       end do",
        "    else",
        "!$OMP PARALLEL DO PRIVATE (C, phys_buffer_chunk)",
        "       do c=begchunk,endchunk",
        "          phys_buffer_chunk => pbuf_get_chunk(pbuf2d, c)",
        "          call tphysbc_leaf_action(action_id, ztodt, phys_state(c), &",
        "               phys_tend(c), phys_buffer_chunk, cam_out(c), cam_in(c))",
        "       end do",
        "    end if",
    ]
    return out


def render() -> dict[Path, str]:
    production = [REPO / name for name in PATCHES]
    boundary_index = next(i for i, p in enumerate(production) if p == BOUNDARY)
    leaf_index = next(i for i, p in enumerate(LEAF_PATCHES) if p == DISPATCH)
    rendered: dict[Path, str] = {}
    with tempfile.TemporaryDirectory(prefix="pycam-macro-boundary-") as temporary:
        root = Path(temporary)
        base = _base(root / "boundary", production[:boundary_index])
        before = root / "boundary-before.F90"
        shutil.copy2(base, before)
        base.write_text("\n".join(edit_boundary(base.read_text().splitlines())) + "\n")
        rendered[BOUNDARY] = _diff(before, base)

        # the leaf base is the full production set (including 0039) plus the
        # earlier leaf patches
        base = _base(root / "dispatch", production[:boundary_index])
        base.write_text("\n".join(edit_boundary(base.read_text().splitlines())) + "\n")
        _apply(production[boundary_index + 1:], root / "dispatch")
        _apply(LEAF_PATCHES[:leaf_index], root / "dispatch")
        before = root / "dispatch-before.F90"
        shutil.copy2(base, before)
        base.write_text("\n".join(edit_dispatch(base.read_text().splitlines())) + "\n")
        rendered[DISPATCH] = _diff(before, base)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stale = []
    for path, text in render().items():
        if arguments.check:
            current = path.read_text() if path.is_file() else ""
            if current != text:
                stale.append(path)
                sys.stderr.write("".join(difflib.unified_diff(
                    current.splitlines(keepends=True), text.splitlines(keepends=True),
                    fromfile=f"{path.name} (committed)", tofile=f"{path.name} (generated)",
                ))[:4000])
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(REPO)}")
    if stale:
        sys.stderr.write("\nstale: " + ", ".join(str(p.relative_to(REPO)) for p in stale) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
