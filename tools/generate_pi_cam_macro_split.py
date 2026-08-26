#!/usr/bin/env python3
"""Generate the mmacro_pcond kernel boundary: one module and one patch.

`macrop_driver_tend` is cut at its call to `mmacro_pcond`
(macrop_driver.F90:1028, the only call site in the code base) so Python can
stand between the two halves and put a trained surrogate where the kernel was.
Both artefacts come from the reviewed specification
(``native/pi_cam/functions/mmacro_pcond.yaml``) rather than being maintained
by hand, so the Fortran record, the patch's argument list and the function
boundary a surrogate is trained against cannot drift apart.

    tools/generate_pi_cam_macro_split.py            # write both artefacts
    tools/generate_pi_cam_macro_split.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "native/pi_cam/functions/mmacro_pcond.yaml"
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/macrop_driver.F90"
MODULE = REPO / "native/pi_cam/support/pycam_macro_split.F90"
PATCH = REPO / "native/pi_cam/control_patches/0035-macro-split-actions.patch"

# Which local `macrop_driver_tend` passes for each specified argument.  Read
# off the call site itself; native/pi_cam/patches/0002-capture-mmacro-pcond.patch
# carries the same mapping for the capture hook and must agree with this one.
ACTUAL = {
    "lchnk": "lchnk", "ncol": "ncol", "dt": "dtime",
    "p": "state_loc%pmid", "dp": "state_loc%pdel",
    "t0": "t_inout", "qv0": "qv_inout", "ql0": "ql_inout", "qi0": "qi_inout",
    "nl0": "nl_inout", "ni0": "ni_inout",
    "a_t": "ttend", "a_qv": "qtend", "a_ql": "lmitend", "a_qi": "itend",
    "a_nl": "nltend", "a_ni": "nitend",
    "c_t": "CC_T", "c_qv": "CC_qv", "c_ql": "CC_ql", "c_qi": "CC_qi",
    "c_nl": "CC_nl", "c_ni": "CC_ni", "c_qlst": "CC_qlst",
    "d_t": "dlf_T", "d_qv": "dlf_qv", "d_ql": "dlf_ql", "d_qi": "dlf_qi",
    "d_nl": "dlf_nl", "d_ni": "dlf_ni",
    "a_cud": "concld_old", "a_cu0": "concld",
    "clrw_old": "clrw_old", "clri_old": "clri_old",
    "landfrac": "landfrac", "snowh": "snowh",
    "tke": "tke", "qtl_flx": "qtl_flx", "qti_flx": "qti_flx",
    "cmfr_det": "cmfr_det", "qlr_det": "qlr_det", "qir_det": "qir_det",
    "s_tendout": "tlat", "qv_tendout": "qvlat", "ql_tendout": "qcten",
    "qi_tendout": "qiten", "nl_tendout": "ncten", "ni_tendout": "niten",
    "qme": "cmeliq", "qvadj": "qvadj", "qladj": "qladj", "qiadj": "qiadj",
    "qllim": "qllim", "qilim": "qilim",
    "cld": "cld", "al_st_star": "alst", "ai_st_star": "aist",
    "ql_st_star": "qlst", "qi_st_star": "qist", "do_cldice": "do_cldice",
}

# The values the second half reads that the kernel interface does not carry.
# Measured, not guessed: every local declared in the routine that appears
# after the call site, minus the physics-buffer pointers (which persist on
# their own) and minus the kernel's own arguments.
CARRY = (
    ("cldsice", "pcols,pver"), ("cldst", "pcols,pver"), ("fice", "pcols,pver"),
    ("icecldf", "pcols,pver"), ("liqcldf", "pcols,pver"),
    ("mr_ccice", "pcols,pver"), ("mr_ccliq", "pcols,pver"),
    ("mr_lsice", "pcols,pver"), ("mr_lsliq", "pcols,pver"),
    ("nqctn", "pcols,pver"), ("nqitn", "pcols,pver"),
    ("pqctn", "pcols,pver"), ("pqitn", "pcols,pver"),
    ("process_rates", "pcols,pver,pwtype,pwtype,pwtype"),
    # Not locals but `intent(out)` arguments, and the first half is where they
    # are set (macrop_driver.F90:706-808).  The caller reads them after the
    # kernel, so they have to survive the gap like everything else.
    ("det_s", "pcols"), ("det_ice", "pcols"),
)

# Kernel inputs the second half still reads once the kernel has run, so the
# record has to hand them back after the first half's locals have died.
LATE_INPUTS = ("clrw_old", "clri_old", "d_ql", "d_qi")


def _arguments() -> list[dict]:
    return yaml.safe_load(SPEC.read_text())["arguments"]


def _dims(item: dict) -> str:
    return f"({','.join(item['native_shape'])})" if item["native_shape"] else ""


def _wrap(items, indent: str, *, close: bool = False, width: int = 78) -> list[str]:
    lines, current = [], indent
    for index, name in enumerate(items):
        piece = name + ("," if index < len(items) - 1 else "")
        if len(current) + len(piece) + 2 > width:
            lines.append(current + " &")
            current = indent
        current += ("" if current == indent else " ") + piece
    lines.append(current + (")" if close else ""))
    return lines


# -- the module ------------------------------------------------------------


def render_module() -> str:
    args = _arguments()
    spec = {a["name"]: a for a in args}
    inputs = [a for a in args if a["role"] in ("structural", "input", "inout")]
    returned = [a for a in args if a["role"] == "output"] + [a for a in args if a["role"] == "inout"]
    names = [a["name"] for a in args]

    def component(item, prefix):
        kind = "integer " if item["dtype"] in ("int32", "int64") else "real(r8)"
        return f"     {kind} :: {prefix}{item['name']}{_dims(item)}"

    def dummy(item):
        if item["role"] == "structural":
            return f"    integer, intent(in) :: {item['name']}"
        if item.get("pointer"):
            return f"    real(r8), pointer :: {item['name']}(:,:)"
        if item.get("carrier") == "logical":
            return f"    logical, intent(in) :: {item['name']}"
        # `input` arguments reach the hook from read-only locals of the caller
        # (dtime, landfrac, snowh); only what the kernel returns may be written.
        intent = "in" if item["role"] == "input" else "inout"
        return f"    real(r8), intent({intent}) :: {item['name']}{_dims(item)}"

    def carried(intent):
        return "\n".join(
            [f"    real(r8), intent({intent}) :: {n}({d})" for n, d in CARRY]
            + [f"    type(physics_state), intent({intent}) :: state_loc",
               f"    type(physics_ptend), intent({intent}) :: ptend_loc"]
        )

    held = [n for n, _ in CARRY] + ["state_loc", "ptend_loc"]
    before_args = ["stage"] + names + held
    after_args = ["stage", "lchnk"] + [a["name"] for a in returned] + list(LATE_INPUTS) + held
    after_decls = "\n".join(
        ["    integer, intent(in) :: stage", "    integer, intent(in) :: lchnk"]
        + [f"    real(r8), intent(inout) :: {n}{_dims(spec[n])}"
           for n in [a["name"] for a in returned] + list(LATE_INPUTS)]
    ) + "\n" + carried("inout")
    call = _wrap(names, "            ", close=True)
    nl = "\n"

    return f'''! One Python-visible boundary at CAM5's liquid macrophysics kernel.
!
! `mmacro_pcond` is the only pure-numerical leaf inside `macrop_driver_tend`
! (macrop_driver.F90:1028, the sole call site in the code base).  This module
! turns that call site into a boundary Python can stand at, so a trained
! surrogate can take the kernel's place without the layer above it leaving
! Fortran, and without Fortran ever calling into Python: the routine is cut in
! two, Python is the caller of both halves, and this module carries across the
! gap what the second half needs.
!
! GENERATED by tools/generate_pi_cam_macro_split.py from the reviewed
! specification native/pi_cam/functions/mmacro_pcond.yaml.  Do not edit by
! hand; edit the specification or the generator.
!
! The exposed record is registered as a state-bridge owner, so every component
! reaches Python as a zero-copy `macro_split.<name>` view.  `in_*` are the
! {len(inputs)} arguments the kernel reads -- including lchnk and ncol, without which
! Python could not tell which lanes of a chunk are live -- and `out_*` and
! `ref_*` the {len(returned)} it returns --
! 17 outputs and the 6 in/out values as they leave.  Those are exactly the
! `input__*` and `output__*`/`updated__*` variables of a training set built by
! examples/generate_mmacro_pcond_dataset.py, so a surrogate keeps one layout
! from training to substitution.
!
! `kernel_mode`, written by Python, selects who computes those {len(returned)}:
!
!   0  the original mmacro_pcond.  Its answer is published in `out_*` and read
!      back from `out_*`, so a Python process that only copies them proves the
!      round trip is exact without changing a single value.
!   1  Python.  The kernel is not called; `out_*` is whatever Python wrote.
!   2  the original, published in `ref_*` and read back from `ref_*`.  Python
!      sees the model's answer beside its own inputs and can score itself
!      inside a real integration without being able to perturb it.
!
! Nothing here is reached on the monolithic path (`stage == 0`): the original
! call runs untouched, which is what keeps the bit-for-bit gate meaningful.
!
! Not thread safe by construction: the admitted PI-atm build runs
! OMP_NUM_THREADS=1 and one chunk at a time per rank.

module pycam_macro_split

  use shr_kind_mod,   only: r8 => shr_kind_r8
  use ppgrid,         only: pcols, pver, begchunk, endchunk
  use physics_types,  only: physics_state, physics_ptend
  use water_types,    only: pwtype
  use cldwat2m_macro, only: mmacro_pcond
  use shr_sys_mod,    only: shr_sys_abort

  implicit none
  private

  public :: pycam_macro_record_t, pycam_macro_record
  public :: pycam_macro_before, pycam_macro_after

  integer, parameter, public :: pycam_macro_stage_first  = 1
  integer, parameter, public :: pycam_macro_stage_second = 2

  ! The kernel boundary, one record per chunk of this rank.  Registered in
  ! native/pi_cam/state_bridge.yaml; tests/unit/test_pi_cam_macro_split.py
  ! checks it component for component against the specification.
  type pycam_macro_record_t
     integer  :: kernel_mode
     integer  :: stage_seen
{nl.join(component(a, "in_") for a in inputs)}
{nl.join(component(a, "out_") for a in returned)}
{nl.join(component(a, "ref_") for a in returned)}
  end type pycam_macro_record_t

  type(pycam_macro_record_t), allocatable, target, save :: pycam_macro_record(:)

  ! Not a user boundary: continuity for the second half only.  These are the
  ! {len(CARRY)} values it reads that the kernel interface does not carry.
  type pycam_macro_carry_t
{nl.join(f"     real(r8) :: {n}({d})" for n, d in CARRY)}
  end type pycam_macro_carry_t

  type(pycam_macro_carry_t), allocatable, target, save :: carry(:)
  type(physics_state), allocatable, save :: carried_state(:)
  type(physics_ptend), allocatable, save :: carried_ptend(:)

contains

  subroutine ensure_allocated()
    if (allocated(pycam_macro_record)) return
    allocate(pycam_macro_record(begchunk:endchunk))
    allocate(carry(begchunk:endchunk))
    allocate(carried_state(begchunk:endchunk))
    allocate(carried_ptend(begchunk:endchunk))
    pycam_macro_record(:)%kernel_mode = 0
    pycam_macro_record(:)%stage_seen  = 0
  end subroutine ensure_allocated

  ! ------------------------------------------------------------------- !
  ! Tail of the first half: publish the boundary, run the kernel unless
  ! Python owns it, and park what the second half will need.
  ! ------------------------------------------------------------------- !
  subroutine pycam_macro_before( &
{nl.join(_wrap(before_args, "       ", close=True))}

    integer, intent(in) :: stage
{nl.join(dummy(a) for a in args)}
{carried("in")}

    type(pycam_macro_record_t), pointer :: slot
    type(pycam_macro_carry_t),  pointer :: hold
    integer :: mode

    if (stage == 0) then
       ! Monolithic path: the original call, and nothing else.
       call mmacro_pcond( &
{nl.join(call)}
       return
    end if

    call ensure_allocated()
    slot => pycam_macro_record(lchnk)
    hold => carry(lchnk)
    mode = slot%kernel_mode
    slot%stage_seen = pycam_macro_stage_first

{nl.join("    slot%in_" + a["name"] + " = "
         + (f"merge(1, 0, {a['name']})" if a.get("carrier") == "logical" else a["name"])
         for a in inputs)}

    if (mode /= 1) then
       call mmacro_pcond( &
{nl.join(call)}
    end if

    select case (mode)
    case (0)
{nl.join(f"       slot%out_{a['name']} = {a['name']}" for a in returned)}
    case (2)
{nl.join(f"       slot%ref_{a['name']} = {a['name']}" for a in returned)}
    end select

{nl.join(f"    hold%{n} = {n}" for n, _ in CARRY)}
    carried_state(lchnk) = state_loc
    carried_ptend(lchnk) = ptend_loc
  end subroutine pycam_macro_before

  ! ------------------------------------------------------------------- !
  ! Head of the second half: restore what died with the first half, then
  ! take the {len(returned)} returned values from whoever owns them.
  ! ------------------------------------------------------------------- !
  subroutine pycam_macro_after( &
{nl.join(_wrap(after_args, "       ", close=True))}

{after_decls}

    type(pycam_macro_record_t), pointer :: slot
    type(pycam_macro_carry_t),  pointer :: hold
    integer :: mode

    if (stage == 0) return

    if (.not. allocated(pycam_macro_record)) then
       call shr_sys_abort('pycam_macro_after: the first half has never run')
    end if
    slot => pycam_macro_record(lchnk)
    hold => carry(lchnk)
    mode = slot%kernel_mode
    if (slot%stage_seen /= pycam_macro_stage_first) then
       call shr_sys_abort('pycam_macro_after: out of order for this chunk')
    end if
    slot%stage_seen = pycam_macro_stage_second

{nl.join(f"    {n} = hold%{n}" for n, _ in CARRY)}
    state_loc = carried_state(lchnk)
    ptend_loc = carried_ptend(lchnk)

{nl.join(f"    {n} = slot%in_{n}" for n in LATE_INPUTS)}

    if (mode == 2) then
{nl.join(f"       {a['name']} = slot%ref_{a['name']}" for a in returned)}
    else
{nl.join(f"       {a['name']} = slot%out_{a['name']}" for a in returned)}
    end if
  end subroutine pycam_macro_after

end module pycam_macro_split
'''


# -- the patch -------------------------------------------------------------


def render_patch(source: Path | None = None) -> str:
    args = _arguments()
    returned = [a["name"] for a in args if a["role"] == "output"] + \
               [a["name"] for a in args if a["role"] == "inout"]
    held = [n for n, _ in CARRY] + ["state_loc", "ptend_loc"]
    before_actuals = [ACTUAL[a["name"]] for a in args] + held
    after_actuals = ["lchnk"] + [ACTUAL[n] for n in returned] + \
        [ACTUAL[n] if n in ACTUAL else n for n in LATE_INPUTS] + held

    lines = (source or PINNED).read_text().splitlines()

    def before(anchor: str, added: list[str]) -> None:
        index = lines.index(anchor)
        lines[index + 1:index + 1] = added

    lines[lines.index("             det_s, det_ice)")] = "             det_s, det_ice, macro_stage)"
    before("  use water_types,      only: pwtype, iwtvap, iwtliq, iwtice",
           ["  use pycam_macro_split, only: pycam_macro_before, pycam_macro_after"])
    before("  real(r8), intent(out) :: det_ice(pcols)           ! Integral of detrained ice for energy check", [
        "",
        "  ! Which half of the routine to run.  Absent or 0 is the original,",
        "  ! monolithic path: no boundary is published and the kernel call below is",
        "  ! the untouched one.  1 runs everything up to the kernel and parks what",
        "  ! the other half needs; 2 resumes from there.  See pycam_macro_split.",
        "  integer, intent(in), optional :: macro_stage",
    ])
    before("  integer :: ncol                                   ! Number of atmospheric columns",
           ["  integer :: macro_stage_local                      ! 0 monolithic, 1 first half, 2 second half"])
    before("  ncol  = state%ncol", [
        "",
        "  macro_stage_local = 0",
        "  if (present(macro_stage)) macro_stage_local = macro_stage",
        "  if (macro_stage_local == 2) go to 1000",
    ])
    first = lines.index("   call mmacro_pcond( lchnk, ncol, dtime, state_loc%pmid, state_loc%pdel,        &")
    last = lines.index("                      cld, alst, aist, qlst, qist, do_cldice ) ")
    lines[first:last + 1] = (
        ["   call pycam_macro_before(macro_stage_local, &"]
        + _wrap(before_actuals, "        ", close=True)
        + ["",
           "   if (macro_stage_local == 1) then",
           "      ! The kernel boundary is published; Python runs between the halves.",
           "      call t_stopf('mmacro_pcond')",
           "      return",
           "   end if",
           "",
           "1000 continue",
           "   if (macro_stage_local == 2) call t_startf('mmacro_pcond')",
           "   call pycam_macro_after(macro_stage_local, &"]
        + _wrap(after_actuals, "        ", close=True)
    )

    with tempfile.TemporaryDirectory(prefix="pycam-macro-split-") as temporary:
        work = Path(temporary)
        original, patched = work / "before.F90", work / "after.F90"
        shutil.copy2(source or PINNED, original)
        patched.write_text("\n".join(lines) + "\n")
        diff = subprocess.run(
            ["git", "diff", "--no-index", "--unified=0", "--no-prefix",
             str(original), str(patched)],
            capture_output=True, text=True,
        ).stdout.splitlines()
    body = [line for line in diff if not line.startswith(("diff --git", "index ", "--- ", "+++ "))]
    header = ["--- a/src/physics/cam/macrop_driver.F90",
              "+++ b/src/physics/cam/macrop_driver.F90"]
    return "\n".join(header + body) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if either artefact differs from what the spec generates")
    arguments = parser.parse_args()
    stale = []
    for path, rendered in ((MODULE, render_module()), (PATCH, render_patch())):
        if arguments.check:
            current = path.read_text() if path.is_file() else ""
            if current != rendered:
                stale.append(path)
                sys.stderr.write("".join(difflib.unified_diff(
                    current.splitlines(keepends=True), rendered.splitlines(keepends=True),
                    fromfile=f"{path.name} (committed)", tofile=f"{path.name} (generated)",
                ))[:4000])
        else:
            path.write_text(rendered)
            print(f"wrote {path.relative_to(REPO)}")
    if stale:
        sys.stderr.write("\nstale: " + ", ".join(str(p.relative_to(REPO)) for p in stale) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
