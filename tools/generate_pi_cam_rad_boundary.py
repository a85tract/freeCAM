#!/usr/bin/env python3
"""Emit the patches that let Python stand where radiation_tend was called.

A Python-driven radiation step never calls the Fortran driver; it walks the
driver's statements itself.  For that, ``tphysbc`` has to stop just before
``call radiation_tend`` and resume just after, so everything around the call
-- the net-flux copy into ``tend%flx_net``, ``physics_update``,
``check_energy_chng`` -- stays exactly where the oracle put it.

Two patches do that:

``0041-rad-tend-boundary.patch`` edits ``physpkg.F90`` only, which the
builder already replaces.  ``tphysbc`` gains an optional ``rad_stage``:
1 binds the chunk's ``cam_in``/``cam_out`` and returns before the call,
2 resumes after it, taking the tendency and the net flux from the handles
module when Python claimed the step and calling the driver itself when it
did not.  No numerical object is touched.

``0042-rad-tend-leaf-dispatch.patch`` is a leaf add-on: action ids 23 and 24
re-enter ``tphysbc``'s own stage 10 with ``rad_stage`` 1 and 2.

    tools/generate_pi_cam_rad_boundary.py            # write both patches
    tools/generate_pi_cam_rad_boundary.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from apply_pi_cam_source_patches import PATCHES  # noqa: E402
from build_pi_cam_devices import LEAF_PATCHES  # noqa: E402

PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam"
BOUNDARY = REPO / "native/pi_cam/control_patches/0041-rad-tend-boundary.patch"
DISPATCH = REPO / "native/pi_cam/control_patches/0042-rad-tend-leaf-dispatch.patch"
RELATIVE = "src/physics/cam/physpkg.F90"
COMPANIONS = ("src/cpl/atm_comp_mct.F90", "src/control/cam_comp.F90")


def _apply(patches, tree: Path) -> None:
    for patch in patches:
        subprocess.run(
            ["git", "apply", "--unidiff-zero", str(patch)],
            cwd=tree, check=True, capture_output=True, text=True,
        )


def _base(tree: Path, patches) -> Path:
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


# -- 0041: the boundary inside tphysbc -----------------------------------------


def edit_boundary(lines: list[str]) -> list[str]:
    out = list(lines)
    tphysbc = out.index("subroutine tphysbc (ztodt,               &")

    out[out.index("       sgh30, cam_out, cam_in, action_id, macro_stage)", tphysbc)] = \
        "       sgh30, cam_out, cam_in, action_id, macro_stage, rad_stage)"

    _insert_after(out, "    use radiation,       only: radiation_tend", [
        "    use pycam_rad_handles, only: rad_ptend, rad_net_flx, python_owns_rad, &",
        "         pycam_rad_bind_chunk",
    ], tphysbc)

    _insert_after(out, "    integer, intent(in), optional :: macro_stage", [
        "    ! Where to pause around the radiation driver.  Absent or 0 runs the",
        "    ! stage whole.  1 binds this chunk's cam_in and cam_out and returns",
        "    ! just before call radiation_tend; 2 resumes just after it, taking",
        "    ! the tendency and the net flux from Python if Python claimed the",
        "    ! step and calling the driver itself otherwise.",
        "    integer, intent(in), optional :: rad_stage",
    ], tphysbc)

    _insert_after(out, "    integer :: macro_stage_local              ! 0 whole, 1 stop before, 2 resume after", [
        "    integer :: rad_stage_local                ! 0 whole, 1 stop before, 2 resume after",
    ], tphysbc)

    # After the macrophysics refusals, which end at their own `end if`.
    macro_refusals = out.index(
        "            ('TPHYSBC: the macrophysics boundary does not carry the "
        "ice-supersaturation aerosol step')", tphysbc)
    closing = out.index("    end if", macro_refusals)
    out[closing + 1:closing + 1] = [
        "    rad_stage_local = 0",
        "    if (present(rad_stage)) rad_stage_local = rad_stage",
        "    if (rad_stage_local /= 0) then",
        "       ! The resume re-enters this routine at stage 10, which begins at",
        "       ! the radiation timer, so nothing before the driver call is rerun.",
        "       ! What cannot be reproduced statement for statement is refused",
        "       ! here rather than run differently; the rest of the refusals need",
        "       ! module state physpkg cannot see and are made by Python at",
        "       ! attach, against the image, before a timestep runs.",
        "       if (single_column .or. scm_crm_mode) call endrun &",
        "            ('TPHYSBC: the radiation boundary does not carry the single-column path')",
        "    end if",
    ]

    # The call itself stays where it is, character for character.  A stop goes
    # in front of it, and the resume becomes the other branch of an if/else.
    first = out.index("    call radiation_tend(state,ptend, pbuf, &", tphysbc)
    out[first:first] = [
        "    if (rad_stage_local == 1) then",
        "       ! Python runs here and calls back in with rad_stage = 2.  cam_in",
        "       ! and cam_out are dummies of this routine, so their addresses are",
        "       ! handed over now, while the chunk's own objects are in scope.",
        "       call pycam_rad_bind_chunk(lchnk, cam_in, cam_out)",
        "       call t_stopf('radiation')",
        "       return",
        "    end if",
        "",
        "    if (rad_stage_local == 2 .and. python_owns_rad) then",
        "       ! The driver's outputs, as Python's transliteration left them.",
        "       if (.not. allocated(rad_ptend(lchnk)%s)) call endrun &",
        "            ('TPHYSBC: Python claimed the radiation step but produced no tendency')",
        "       ptend   = rad_ptend(lchnk)",
        "       net_flx = rad_net_flx(:,lchnk)",
        "    else",
    ]
    start = out.index("    call radiation_tend(state,ptend, pbuf, &", first)
    last = out.index("         fsds, net_flx)", start)
    for index in range(start, last + 1):
        out[index] = "   " + out[index]
    out[last + 1:last + 1] = ["    end if"]
    return out


# -- 0042: the leaf dispatch -------------------------------------------------------


def edit_dispatch(lines: list[str]) -> list[str]:
    out = list(lines)
    start = out.index("subroutine phys_run1_leaf_action(action_id, phys_state, ztodt, phys_tend, &")
    out[out.index("    if (action_id < 12 .or. action_id > 22) then", start)] = \
        "    if (action_id < 12 .or. action_id > 24) then"
    anchor = out.index("    if (action_id >= 21) then", start)
    out[anchor:anchor + 1] = [
        "    if (action_id >= 23) then",
        "       ! The two halves of the radiation stage, the same shape as the",
        "       ! macrophysics pair below: tphysbc's own stage 10, stopped before",
        "       ! the radiation driver and resumed after it, so the statement",
        "       ! order the bit-for-bit gate depends on is the original one.",
        "!$OMP PARALLEL DO PRIVATE (C, phys_buffer_chunk)",
        "       do c=begchunk,endchunk",
        "          phys_buffer_chunk => pbuf_get_chunk(pbuf2d, c)",
        "          call tphysbc(ztodt, fsns(1,c), fsnt(1,c), flns(1,c), flnt(1,c), &",
        "               phys_state(c), phys_tend(c), phys_buffer_chunk, fsds(1,c), &",
        "               landm(1,c), sgh30(1,c), cam_out(c), cam_in(c), 10, &",
        "               rad_stage=action_id - 22)",
        "       end do",
        "    else if (action_id >= 21) then",
    ]
    return out


def render() -> dict[Path, str]:
    production = [REPO / name for name in PATCHES]
    boundary_index = next(i for i, p in enumerate(production) if p == BOUNDARY)
    leaf_index = next(i for i, p in enumerate(LEAF_PATCHES) if p == DISPATCH)
    rendered: dict[Path, str] = {}
    with tempfile.TemporaryDirectory(prefix="pycam-rad-boundary-") as temporary:
        root = Path(temporary)
        base = _base(root / "boundary", production[:boundary_index])
        before = root / "boundary-before.F90"
        shutil.copy2(base, before)
        base.write_text("\n".join(edit_boundary(base.read_text().splitlines())) + "\n")
        rendered[BOUNDARY] = _diff(before, base)

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
