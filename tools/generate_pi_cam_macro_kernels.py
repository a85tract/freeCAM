#!/usr/bin/env python3
"""Carve macrop_driver_tend's inline arithmetic into callable Fortran kernels.

The driver layer of CAM5's macrophysics is mostly calls, copies and control
flow, but seven blocks of it compute.  For Python to own that layer without
owning any floating point, those blocks have to become routines the image can
be asked to run.  They are not rewritten here: each body is lifted verbatim
from the pinned source and only *renamed* -- a derived-type component becomes
the plain array it already is, a module flag becomes an argument -- so the
expressions the compiler sees are character for character the originals.

The module is an addition to the CAM source tree, never a replacement.
macrop_driver.F90 is not patched to call it: the oracle's macrop_driver.o
stays byte for byte what the bit-for-bit gate validated, and only a
Python-driven timestep reaches these routines.  The price is that the
arithmetic exists twice -- once in the driver, once here -- which is why a
test compares every body here against the pinned text, and why the
Python-driven run is itself gated bit-for-bit against the oracle.

    tools/generate_pi_cam_macro_kernels.py            # write the module
    tools/generate_pi_cam_macro_kernels.py --check    # fail if it is stale

The bodies come from ``external/iCESM1.3.1_fzhu``, so a change to the pinned
source shows up as a --check failure rather than as a silent divergence.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/macrop_driver.F90"
MODULE = REPO / "native/pi_cam/support/pycam_macro_kernels.F90"

# Renames applied to every carved body.  Each one replaces a name the routine
# could only see as a host association with a name it receives as an argument.
COMMON = {
    "state_loc%ncol": "ncol",
    "state_loc%t": "t",
    "state_loc%pdel": "pdel",
    "state_loc%q": "q",
    "ptend_loc%s": "ptend_s",
    "ptend_loc%q": "ptend_q",
    "wtrc_iatype(m,iwtliq)": "iatype_liq(m)",
    "wtrc_iatype(m,iwtice)": "iatype_ice(m)",
    "wtrc_nwset": "nwset",
    "(trace_water) .and. (wtrc_detrain_in_macrop)": "do_wtrc_detrain",
    "get_nstep() .le. 1": "nstep .le. 1",
}


class Block:
    """One carved routine: where its body is, and what it receives."""

    def __init__(self, name, first, last, signature, declarations, locals_, *,
                 renames=None):
        self.name = name
        self.first = first          # 1-based, inclusive
        self.last = last
        self.signature = signature
        self.declarations = declarations
        self.locals = locals_
        self.renames = dict(COMMON, **(renames or {}))

    @property
    def arguments(self) -> tuple[str, ...]:
        return tuple(", ".join(self.signature).split(", "))

    def body(self, lines: list[str]) -> list[str]:
        out = []
        for line in lines[self.first - 1:self.last]:
            for old, new in self.renames.items():
                line = line.replace(old, new)
            out.append("   " + line if line.strip() else line)
        return out


BLOCKS = (
    Block(
        "macrop_detrain_partition", 706, 797,
        ["ncol, top_lev, ixcldliq, ixcldice, ixnumliq, ixnumice, nwset",
         "iatype_liq, iatype_ice, do_detrain, cu_det_st, do_wtrc_detrain",
         "gravit, latice, cpair, t, pdel, dlf, dlf2, wtdlf, ptend_s, ptend_q",
         "det_s, det_ice, dlf_T, dlf_qv, dlf_ql, dlf_qi, dlf_nl, dlf_ni",
         "dpdlfliq, dpdlfice, shdlfliq, shdlfice, dpdlft, shdlft"],
        ["   integer,  intent(in)    :: ncol, top_lev, nwset",
         "   integer,  intent(in)    :: ixcldliq, ixcldice, ixnumliq, ixnumice",
         "   integer,  intent(in)    :: iatype_liq(nwset), iatype_ice(nwset)",
         "   logical,  intent(in)    :: do_detrain, cu_det_st, do_wtrc_detrain",
         "   real(r8), intent(in)    :: gravit, latice, cpair",
         "   real(r8), intent(in)    :: t(pcols,pver), pdel(pcols,pver)",
         "   real(r8), intent(in)    :: dlf(pcols,pver), dlf2(pcols,pver)",
         "   real(r8), intent(in)    :: wtdlf(pcols,pver,nwset)",
         "   real(r8), intent(inout) :: ptend_s(pcols,pver)",
         "   real(r8), intent(inout) :: ptend_q(pcols,pver,pcnst)",
         "   real(r8), intent(out)   :: det_s(pcols), det_ice(pcols)",
         "   ! Zeroed by the caller and written only under cu_det_st, so these",
         "   ! carry their incoming values out when that branch is not taken.",
         "   real(r8), intent(inout) :: dlf_T(pcols,pver), dlf_qv(pcols,pver)",
         "   real(r8), intent(inout) :: dlf_ql(pcols,pver), dlf_qi(pcols,pver)",
         "   real(r8), intent(inout) :: dlf_nl(pcols,pver), dlf_ni(pcols,pver)",
         "   real(r8), intent(out)   :: dpdlfliq(pcols,pver), dpdlfice(pcols,pver)",
         "   real(r8), intent(out)   :: shdlfliq(pcols,pver), shdlfice(pcols,pver)",
         "   real(r8), intent(out)   :: dpdlft(pcols,pver), shdlft(pcols,pver)"],
        ["   integer  :: i, k, m", "   real(r8) :: dum1"],
    ),
    Block(
        "macrop_clear_fraction", 895, 902,
        ["ncol, top_lev, concld, alst, ast, clrw_old, clri_old"],
        ["   integer,  intent(in)  :: ncol, top_lev",
         "   real(r8), intent(in)  :: concld(pcols,pver), alst(pcols,pver), ast(pcols,pver)",
         "   real(r8), intent(out) :: clrw_old(pcols,pver), clri_old(pcols,pver)"],
        ["   integer :: i, k"],
    ),
    Block(
        "macrop_advective_forcing", 978, 1019,
        ["ncol, top_lev, nstep, rdtime, t, q, qc, qi, nc, ni",
         "CC_T, CC_qv, CC_ql, CC_qi, CC_nl, CC_ni, CC_qlst",
         "tcwat, qcwat, lcwat, iccwat, nlwat, niwat",
         "ttend, qtend, ltend, itend, nltend, nitend, lmitend",
         "t_inout, qv_inout, ql_inout, qi_inout, nl_inout, ni_inout"],
        ["   integer,  intent(in)    :: ncol, top_lev, nstep",
         "   real(r8), intent(in)    :: rdtime",
         "   real(r8), intent(in)    :: t(pcols,pver), q(pcols,pver,pcnst)",
         "   real(r8), intent(in)    :: qc(pcols,pver), qi(pcols,pver)",
         "   real(r8), intent(in)    :: nc(pcols,pver), ni(pcols,pver)",
         "   ! At the first step these are overwritten from the state, so they",
         "   ! are in/out even though every later step only reads them.",
         "   real(r8), intent(inout) :: tcwat(pcols,pver), qcwat(pcols,pver)",
         "   real(r8), intent(inout) :: lcwat(pcols,pver), iccwat(pcols,pver)",
         "   real(r8), intent(inout) :: nlwat(pcols,pver), niwat(pcols,pver)",
         "   real(r8), intent(inout) :: CC_T(pcols,pver), CC_qv(pcols,pver)",
         "   real(r8), intent(inout) :: CC_ql(pcols,pver), CC_qi(pcols,pver)",
         "   real(r8), intent(inout) :: CC_nl(pcols,pver), CC_ni(pcols,pver)",
         "   real(r8), intent(inout) :: CC_qlst(pcols,pver)",
         "   real(r8), intent(out)   :: ttend(pcols,pver), qtend(pcols,pver)",
         "   real(r8), intent(out)   :: ltend(pcols,pver), itend(pcols,pver)",
         "   real(r8), intent(out)   :: nltend(pcols,pver), nitend(pcols,pver)",
         "   real(r8), intent(out)   :: lmitend(pcols,pver)",
         "   real(r8), intent(out)   :: t_inout(pcols,pver), qv_inout(pcols,pver)",
         "   real(r8), intent(out)   :: ql_inout(pcols,pver), qi_inout(pcols,pver)",
         "   real(r8), intent(out)   :: nl_inout(pcols,pver), ni_inout(pcols,pver)"],
        [],
    ),
    Block(
        "macrop_kernel_to_ptend", 1051, 1080,
        ["ncol, top_lev, ixcldliq, ixcldice, ixnumliq, ixnumice",
         "do_cldice, do_cldliq, tlat, qvlat, qcten, qiten, ncten, niten",
         "ptend_s, ptend_q, status"],
        ["   integer,  intent(in)    :: ncol, top_lev",
         "   integer,  intent(in)    :: ixcldliq, ixcldice, ixnumliq, ixnumice",
         "   logical,  intent(in)    :: do_cldice, do_cldliq",
         "   real(r8), intent(in)    :: tlat(pcols,pver), qvlat(pcols,pver)",
         "   real(r8), intent(in)    :: qcten(pcols,pver), qiten(pcols,pver)",
         "   real(r8), intent(in)    :: ncten(pcols,pver), niten(pcols,pver)",
         "   real(r8), intent(inout) :: ptend_s(pcols,pver)",
         "   real(r8), intent(inout) :: ptend_q(pcols,pver,pcnst)",
         "   ! The original calls endrun on each of these four; a routine Python",
         "   ! drives reports instead, and the caller decides to stop.",
         "   integer,  intent(out)   :: status"],
        ["   integer :: i, k"],
        renames={
            'call endrun("macrop_driver:ERROR - "// &': "status = 1",
            'call endrun("macrop_driver:ERROR -"// &': "status = 2",
            '"Cldwat is configured not to prognose cloud ice, but mmacro_pcond has ice mass tendencies.")': "",
            '" Cldwat is configured not to prognose cloud ice, but mmacro_pcond has ice number tendencies.")': "",
            '"Cldwat is configured not to prognose cloud liquid, but mmacro_pcond has liquid mass tendencies.")': "",
            '"Cldwat is configured not to prognose cloud liquid, but mmacro_pcond has liquid number tendencies.")': "",
        },
    ),
    Block(
        "macrop_tracer_rate_split", 1107, 1126,
        ["ncol, top_lev, qcten, qiten, pqctn, nqctn, pqitn, nqitn"],
        ["   integer,  intent(in)  :: ncol, top_lev",
         "   real(r8), intent(in)  :: qcten(pcols,pver), qiten(pcols,pver)",
         "   real(r8), intent(out) :: pqctn(pcols,pver), nqctn(pcols,pver)",
         "   real(r8), intent(out) :: pqitn(pcols,pver), nqitn(pcols,pver)"],
        ["   integer :: i, k"],
    ),
    Block(
        "macrop_cloud_mixing_ratio", 1182, 1197,
        ["ncol, top_lev, ixcldliq, ixcldice, q, cld",
         "mr_ccliq, mr_ccice, mr_lsliq, mr_lsice"],
        ["   integer,  intent(in)  :: ncol, top_lev, ixcldliq, ixcldice",
         "   real(r8), intent(in)  :: q(pcols,pver,pcnst), cld(pcols,pver)",
         "   real(r8), intent(out) :: mr_ccliq(pcols,pver), mr_ccice(pcols,pver)",
         "   real(r8), intent(out) :: mr_lsliq(pcols,pver), mr_lsice(pcols,pver)"],
        ["   integer :: i, k"],
    ),
    Block(
        "macrop_save_equilibrium", 1208, 1218,
        ["ncol, top_lev, ixcldliq, ixcldice, ixnumliq, ixnumice, tmelt, t, q",
         "tcwat, qcwat, lcwat, iccwat, nlwat, niwat, cldsice"],
        ["   integer,  intent(in)  :: ncol, top_lev",
         "   integer,  intent(in)  :: ixcldliq, ixcldice, ixnumliq, ixnumice",
         "   real(r8), intent(in)  :: tmelt",
         "   real(r8), intent(in)  :: t(pcols,pver), q(pcols,pver,pcnst)",
         "   real(r8), intent(out) :: tcwat(pcols,pver), qcwat(pcols,pver)",
         "   real(r8), intent(out) :: lcwat(pcols,pver), iccwat(pcols,pver)",
         "   real(r8), intent(out) :: nlwat(pcols,pver), niwat(pcols,pver)",
         "   real(r8), intent(out) :: cldsice(pcols,pver)"],
        ["   integer :: k"],
    ),
)


def _wrap(items, indent, width=76):
    lines, current = [], indent
    for index, name in enumerate(items):
        piece = name + ("," if index < len(items) - 1 else "")
        if len(current) + len(piece) + 2 > width:
            lines.append(current + " &")
            current = indent
        current += ("" if current == indent else " ") + piece
    return lines + [current + ")"]


def render_module(source: Path | None = None) -> str:
    lines = (source or PINNED).read_text().splitlines()
    nl = "\n"
    routines = []
    for block in BLOCKS:
        header = [f"  subroutine {block.name}( &"] + [
            "  " + line for line in _wrap(block.arguments, "       ")
        ]
        pieces = header + [""] + block.declarations
        if block.locals:
            pieces += [""] + block.locals
        if any("status" in d for d in block.declarations):
            pieces += ["", "   status = 0"]
        pieces += [""] + block.body(lines) + ["", f"  end subroutine {block.name}"]
        routines.append(nl.join(pieces))

    return f'''! macrop_driver_tend's inline arithmetic, as routines the image can be asked
! to run.
!
! GENERATED by tools/generate_pi_cam_macro_kernels.py from the pinned source
! macrop_driver.F90.  Do not edit by hand; edit the generator.
!
! Each body below is the original text with names substituted, never rewritten:
! `state_loc%t` becomes `t`, `ptend_loc%q` becomes `ptend_q`, a module flag
! becomes an argument.  Every expression, every loop nest and every bound is
! character for character what the pinned source computes, which is what lets
! the bit-for-bit gate mean something.
!
! This module is an addition to the source tree.  macrop_driver_tend is not
! changed to call it, so the oracle's own macrop_driver.o -- the machine code
! the gate validated -- stays untouched; only a Python-driven timestep runs
! these.  The Python-driven run is gated bit-for-bit against the oracle, and
! that gate is what proves the two copies of the arithmetic agree.
!
! Nothing here touches a derived type, the physics buffer, the clock or the
! history file.  That is deliberate: these are exactly the pieces of the
! driver layer that had no reason to stay welded to it.

module pycam_macro_kernels

  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid,       only: pcols, pver
  use constituents, only: pcnst

  implicit none
  private

{nl.join("  public :: " + block.name for block in BLOCKS)}

contains

{(nl + nl).join(routines)}

end module pycam_macro_kernels
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stale = []
    for path, rendered in ((MODULE, render_module()),):
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
