#!/usr/bin/env python3
"""Give physpkg address accessors for the carries the tphysac blocks read.

A Python-replaced tphysac stage still needs what the surrounding native
stages saved for it: the friction velocity and Obukhov length the vertical
diffusion stage wrote (``pycesm_ac_surfric``/``pycesm_ac_obklen``), and the
tracer totals stage 1 recorded (``pycesm_ac_tracerint``, and tphysbc's ``pycesm_bc_tracerint``).  Those arrays are
private to physpkg on purpose.  ``0043-stage-carry-boundary.patch`` adds
``bind(C)`` accessors that hand out one chunk's address -- the idiom of
0039's ``pycam_macro_forcing_v1``, whose helper subroutines it reuses --
and changes no executable statement of any routine.

The patch is generated against the pinned physpkg.F90 with every earlier
production patch applied, so a stale offset is a --check failure here
rather than a wrong-place edit at build time.

    tools/generate_pi_cam_stage_carry_boundary.py            # write the patch
    tools/generate_pi_cam_stage_carry_boundary.py --check    # fail if stale
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

PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam"
BOUNDARY = REPO / "native/pi_cam/control_patches/0043-stage-carry-boundary.patch"
RELATIVE = "src/physics/cam/physpkg.F90"

# The other files the earlier production patches touch; they have to be
# present for those patches to apply, though only physpkg.F90 is edited here.
COMPANIONS = ("src/cpl/atm_comp_mct.F90", "src/control/cam_comp.F90")

ACCESSORS = [
    "",
    "integer(c_int) function pycam_ac_carry_v1(lchnk, code, ptr, ndims, extents) &",
    "     bind(C, name='pycam_ac_carry_v1') result(status)",
    "  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_ptr, c_null_ptr",
    "  integer(c_int), value, intent(in) :: lchnk, code",
    "  type(c_ptr), intent(out) :: ptr",
    "  integer(c_int), intent(out) :: ndims",
    "  integer(c_int64_t), intent(out) :: extents(4)",
    "  ptr = c_null_ptr",
    "  ndims = 0_c_int",
    "  extents = 0_c_int64_t",
    "  status = 1_c_int",
    "  if (.not. allocated(pycesm_ac_surfric)) return",
    "  if (lchnk < begchunk .or. lchnk > endchunk) then",
    "    status = 2_c_int",
    "    return",
    "  endif",
    "  select case (code)",
    "  case (1)",
    "    call pycam_macro_address1(pycesm_ac_surfric(:,lchnk), ptr, ndims, extents)",
    "  case (2)",
    "    call pycam_macro_address1(pycesm_ac_obklen(:,lchnk), ptr, ndims, extents)",
    "  case default",
    "    status = 3_c_int",
    "    return",
    "  end select",
    "  status = 0_c_int",
    "end function pycam_ac_carry_v1",
    "",
    "integer(c_int) function pycam_ac_tracerint_addr_v1(lchnk, ptr) &",
    "     bind(C, name='pycam_ac_tracerint_addr_v1') result(status)",
    "  use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_null_ptr",
    "  integer(c_int), value, intent(in) :: lchnk",
    "  type(c_ptr), intent(out) :: ptr",
    "  ptr = c_null_ptr",
    "  status = 1_c_int",
    "  if (.not. allocated(pycesm_ac_tracerint)) return",
    "  if (lchnk < begchunk .or. lchnk > endchunk) then",
    "    status = 2_c_int",
    "    return",
    "  endif",
    "  call pycam_stage_record_addr(pycesm_ac_tracerint(lchnk), ptr)",
    "  status = 0_c_int",
    "end function pycam_ac_tracerint_addr_v1",
    "",
    "integer(c_int) function pycam_bc_tracerint_addr_v1(lchnk, ptr) &",
    "     bind(C, name='pycam_bc_tracerint_addr_v1') result(status)",
    "  use, intrinsic :: iso_c_binding, only: c_int, c_ptr, c_null_ptr",
    "  integer(c_int), value, intent(in) :: lchnk",
    "  type(c_ptr), intent(out) :: ptr",
    "  ptr = c_null_ptr",
    "  status = 1_c_int",
    "  if (.not. allocated(pycesm_bc_tracerint)) return",
    "  if (lchnk < begchunk .or. lchnk > endchunk) then",
    "    status = 2_c_int",
    "    return",
    "  endif",
    "  call pycam_stage_record_addr(pycesm_bc_tracerint(lchnk), ptr)",
    "  status = 0_c_int",
    "end function pycam_bc_tracerint_addr_v1",
    "",
    "subroutine pycam_stage_record_addr(record, ptr)",
    "  use, intrinsic :: iso_c_binding, only: c_ptr, c_loc",
    "  type(check_tracers_data), target, intent(in) :: record",
    "  type(c_ptr), intent(out) :: ptr",
    "  ptr = c_loc(record)",
    "end subroutine pycam_stage_record_addr",
]


def edit_carry(lines: list[str]) -> list[str]:
    out = list(lines)
    end = out.index("end module physpkg")
    out[end:end] = ACCESSORS
    return out


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


def render() -> dict[Path, str]:
    production = [REPO / name for name in PATCHES]
    boundary_index = next(i for i, p in enumerate(production) if p == BOUNDARY)
    with tempfile.TemporaryDirectory(prefix="pycam-stage-carry-") as temporary:
        root = Path(temporary)
        base = _base(root / "carry", production[:boundary_index])
        before = root / "carry-before.F90"
        shutil.copy2(base, before)
        base.write_text("\n".join(edit_carry(base.read_text().splitlines())) + "\n")
        return {BOUNDARY: _diff(before, base)}


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
