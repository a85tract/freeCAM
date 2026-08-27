#!/usr/bin/env python3
"""Lift micro_mg_cam_tend's inline arithmetic into routines the image can run.

The MG microphysics driver (``micro_mg_cam.F90:997-3184``) is 1490
statements, of which sixty-three on the live path compute a floating-point
number outside the routines it calls.  Python may compute none of them, so
each contiguous run becomes a Fortran routine here, its body the pinned
text with names substituted and never rewritten.  ``micro_mg_cam.F90`` is
**not** patched: this module is an addition, and only a Python-driven
timestep runs these copies.

Three things about this lift that the macrophysics and radiation lifts did
not have:

* Several blocks call elemental routines of ``micro_mg_utils`` inline --
  ``size_dist_param_liq``, ``size_dist_param_basic``, ``avg_diameter`` --
  with the module's own ``mg_liq_props`` / ``mg_ice_props``.  Those stay
  inline: the lifted routine ``use``s the same module, so the text is the
  driver's and the numbers are the utility's.
* One block encloses a branch that is dead under this configuration
  (``micro_mg_version > 1``).  The dead ``if ... else`` and its ``end if``
  are dropped by line number, and a test checks the drop is exactly those.
* The history block writes six fields through one ``ftem_grid`` buffer,
  each computed then handed to ``outfld``.  ``outfld`` only reads, so the
  six computations become six output arrays of one routine and Python
  writes the six fields after it; each renamed line is listed here so a
  test can check nothing but the name changed.

The unpacking of MG's packed tendencies (``packer%unpack`` with an
expression in its second argument, lines 2214-2222) is arithmetic that
needs the packer, which lives in the handles module; it is lifted there,
not here.

    tools/generate_pi_cam_micro_kernels.py            # write the module and descriptors
    tools/generate_pi_cam_micro_kernels.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/micro_mg_cam.F90"
MODULE = REPO / "native/pi_cam/support/pycam_micro_kernels.F90"
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_microphysics.yaml"

#: Renames applied to every carved body: a derived-type component the routine
#: now receives by name.  ``state_loc`` is the driver's local copy of the
#: state after the tendencies were applied; ``state`` the one it was given.
COMMON = {
    "state_loc%q": "q_loc",
    "state_loc%pmid": "pmid_loc",
    "state_loc%t": "t_loc",
    "state_loc%pdel": "pdel_loc",
    "state%q": "q",
    "state%pdel": "pdel",
    "state%pmid": "pmid",
    "state%t": "t",
}

P2, P1, P3, PP = "(pcols,pver)", "(pcols)", "(pcols,pver,pcnst)", "(pcols,pverp)"


class Block:
    """One carved routine: where its body is, and what it receives."""

    def __init__(self, name, first, last, arguments, *, skip=(), line_renames=None,
                 replace=None, locals_=(), note=""):
        self.name = name
        self.first = first
        self.last = last
        #: (dummy, fortran kind, dims, intent) in argument order
        self.arguments = arguments
        self.skip = frozenset(skip)
        self.line_renames = dict(line_renames or {})
        self.replace = dict(replace or {})
        self.locals = tuple(locals_)
        self.note = note

    @property
    def dummies(self) -> tuple[str, ...]:
        return tuple(a[0] for a in self.arguments)

    def declarations(self) -> list[str]:
        out = []
        for name, kind, dims, intent in self.arguments:
            out.append(f"   {kind:9s} intent({intent}) :: {name}{dims}")
        return out

    def body(self, lines: list[str]) -> list[str]:
        out = []
        for number in range(self.first, self.last + 1):
            if number in self.skip:
                continue
            if number in self.replace:
                out.extend("   " + l for l in self.replace[number])
                continue
            line = lines[number - 1]
            for old, new in COMMON.items():
                line = line.replace(old, new)
            for old, new in self.line_renames.get(number, {}).items():
                line = line.replace(old, new)
            out.append("   " + line if line.strip() else line)
        return out


def _real(*names, dims=P2, intent="inout"):
    return [(n, "real(r8),", dims, intent) for n in names]


def _int(*names, intent="in"):
    return [(n, "integer, ", "", intent) for n in names]


def _scalar(*names):
    return [(n, "real(r8),", "", "in") for n in names]


BLOCKS = (
    Block(
        "micro_initial_water_paths", 1724, 1736,
        _int("ncol", "top_lev", "ixcldliq", "ixcldice") + _scalar("mincld", "gravit")
        + _real("q", dims=P3, intent="in") + _real("ast", "pdel", intent="in")
        + _real("iclwpi", "iciwpi", dims=P1),
        locals_=("   integer :: i, k",),
    ),
    Block(
        "micro_precip_diagnostics", 2286, 2299,
        _int("ncol", "top_lev")
        + _real("naai", "naai_hom", "mnuccdo", "qrout", "qsout", intent="in")
        + _real("rflx", "sflx", dims=PP, intent="in")
        + _real("mnuccdohet", "mgmrprc", "mgmrsnw") + _real("mgflxprc", "mgflxsnw", dims=PP),
        locals_=("   integer :: i, k",),
    ),
    Block(
        "micro_macro_feedback", 2312, 2337,
        _int("ncol", "top_lev") + _scalar("cpair")
        + _real("vtrmc", "tlat", "qvlat", "qcten", "qiten", "ncten", "niten", "alst_mic",
                "cmeliq", "cmeiout", "ast", intent="in")
        + _real("prect", "preci", dims=P1, intent="in")
        + _real("wsedl", "cc_t", "cc_qv", "cc_ql", "cc_qi", "cc_nl", "cc_ni", "cc_qlst", "qme",
                "icecldf", "liqcldf")
        + _real("prec_pcw", "snow_pcw", "prec_sed", "snow_sed", "prec_str", "snow_str", dims=P1),
    ),
    Block(
        "micro_in_cloud_quantities", 2352, 2384,
        _int("ncol", "top_lev", "ixcldliq", "ixcldice", "ixnumliq", "ixnumice")
        + _scalar("mincld", "gravit")
        + _real("q_loc", dims=P3, intent="in")
        + _real("pmid_loc", "t_loc", "pdel_loc", "icecldf", "liqcldf", "ast", "cld", "concld",
                "qsout", intent="in")
        + _real("icimrst", "icwmrst", "icinc", "icwnc", "iciwpst", "iclwpst", "cldfsnow", "icswp"),
        locals_=("   integer :: i, k",),
    ),
    Block(
        "micro_split_signs", 2607, 2626,
        _int("ncol", "top_lev")
        + _real("cmeiout_grid", "meltso", "meltso_grid", intent="in")
        + _real("pcmei_grid", "ncmei_grid", "pmelts_grid", "nmelts_grid"),
        locals_=("   integer :: i, k",),
    ),
    Block(
        "micro_air_density", 2712, 2713,
        _int("ncol", "top_lev") + _scalar("rair")
        + _real("pmid", "t", intent="in") + _real("rho"),
    ),
    Block(
        "micro_liquid_size", 2721, 2759,
        _int("ngrdcol", "top_lev") + _scalar("mincld", "qsmall")
        + _real("icwmrst_grid", "rho_grid", "nc_grid", "liqcldf_grid", intent="in")
        + _real("mu_grid", "lambdac_grid", "rel_fn_grid", "ncic_grid", "rel_grid"),
        note="size_dist_param_liq with mg_liq_props stays inline: the same module procedure",
    ),
    Block(
        "micro_ice_and_precip_size", 2762, 2854,
        _int("ngrdcol", "top_lev") + _scalar("mincld", "qsmall")
        + _real("qrout_grid", "nrout_grid", "qsout_grid", "nsout_grid", "rho_grid", "ni_grid",
                "icecldf_grid", "icimrst_grid", "ast_grid", intent="in")
        + _real("drout2_grid", "reff_rain_grid", "des_grid", "dsout2_grid", "reff_snow_grid",
                "rei_grid", "niic_grid", "dei_grid", "mu_grid", "lambdac_grid",
                "mgreffrain_grid", "mgreffsnow_grid"),
        # 2768-2794 is `if (micro_mg_version > 1) then ... else`, dead under
        # version 1.0; 2820 is its `end if`.  The else-branch text stays.
        skip=tuple(range(2768, 2795)) + (2820,),
        locals_=("   integer :: i, k",),
        note="avg_diameter and size_dist_param_basic with mg_ice_props stay inline",
    ),
    Block(
        "micro_precip_efficiency", 2864, 2927,
        _int("ngrdcol", "top_lev") + _scalar("gravit", "rhoh2o")
        + _real("iclwpst_grid", "cld_grid", "cmeliq_grid", "pdel_grid", intent="in")
        + _real("prec_str_grid", dims=P1, intent="in")
        + _real("acgcme_grid", "acprecl_grid", dims=P1)
        + [("acnum_grid", "integer, ", P1, "inout")]
        + _real("tgliqwp_grid", "tgcmeliq_grid", "pe_grid", "tpr_grid", "pefrac_grid", dims=P1),
        locals_=("   integer  :: i, k", "   real(r8) :: minlwp"),
    ),
    Block(
        "micro_autoconversion_ratio", 2933, 2964,
        _int("ngrdcol", "top_lev") + _scalar("gravit")
        + _real("prao_grid", "prco_grid", "nc_grid", "pdel_grid", intent="in")
        + _real("vprao_grid", "vprco_grid", "racau_grid", "cdnumc_grid", dims=P1),
        locals_=("   integer :: k", "   integer :: cnt_grid(pcols)"),
    ),
    Block(
        "micro_effective_outputs", 2967, 3024,
        _int("ngrdcol", "top_lev")
        + _real("liqcldf_grid", "icecldf_grid", "icwmrst_grid", "icimrst_grid", "rel_grid",
                "rei_grid", "icwnc_grid", "icinc_grid", "nevapr_grid", intent="in")
        + _real("evpsnow_st_grid")
        + _real("efcout_grid", "efiout_grid", "ncout_grid", "niout_grid", "freql_grid",
                "freqi_grid", "icwmrst_grid_out", "icimrst_grid_out", "evprain_st_grid")
        + _real("fcti_grid", "fctl_grid", "ctrel_grid", "ctrei_grid", "ctnl_grid", "ctni_grid",
                dims=P1),
        locals_=("   integer :: i, k",),
    ),
    Block(
        "micro_history_tendencies", 3037, 3060,
        _int("ngrdcol", "top_lev")
        + _real("qcreso_grid", "melto_grid", "mnuccco_grid", "mnuccto_grid", "bergo_grid",
                "homoo_grid", "msacwio_grid", "prao_grid", "prco_grid", "psacwso_grid",
                "bergso_grid", "cmeiout_grid", "qireso_grid", "prcio_grid", "praio_grid",
                intent="in")
        + _real("mpdw2v", "mpdw2i", "mpdw2p", "mpdi2v", "mpdi2w", "mpdi2p"),
        # the six outfld calls: Python makes them, after this routine
        skip=(3040, 3045, 3049, 3052, 3057, 3060),
        # one ftem_grid buffer becomes six named outputs; outfld only reads
        replace={3037: [f"{n} = 0._r8" for n in ("mpdw2v", "mpdw2i", "mpdw2p", "mpdi2v", "mpdi2w", "mpdi2p")]},
        line_renames={3039: {"ftem_grid": "mpdw2v"}, 3042: {"ftem_grid": "mpdw2i"},
                      3047: {"ftem_grid": "mpdw2p"}, 3051: {"ftem_grid": "mpdi2v"},
                      3054: {"ftem_grid": "mpdi2w"}, 3059: {"ftem_grid": "mpdi2p"}},
    ),
)

#: The driver's own named constants, re-declared here from micro_mg_cam.F90
#: (a test pins them to the pinned lines).
PARAMETERS = (
    "  real(r8), parameter :: dcon   = 25.e-6_r8",
    "  real(r8), parameter :: mucon  = 5.3_r8",
    "  real(r8), parameter :: deicon = 50._r8",
)


def _wrap(items, indent, width=76):
    out, current = [], indent
    for index, name in enumerate(items):
        piece = name + ("," if index < len(items) - 1 else "")
        if len(current) + len(piece) + 2 > width:
            out.append(current + " &")
            current = indent
        current += ("" if current == indent else " ") + piece
    return out + [current + ")"]


def render_module(source: Path | None = None) -> str:
    lines = (source or PINNED).read_text().splitlines()
    nl = "\n"
    routines = []
    for block in BLOCKS:
        header = [f"  subroutine {block.name}( &"] + [
            "  " + line for line in _wrap(block.dummies, "       ")
        ]
        pieces = header + [""] + block.declarations()
        if block.locals:
            pieces += [""] + list(block.locals)
        pieces += [""] + block.body(lines) + ["", f"  end subroutine {block.name}"]
        routines.append(nl.join(pieces))

    return f'''! micro_mg_cam_tend's inline arithmetic, as routines the image can be asked
! to run.
!
! GENERATED by tools/generate_pi_cam_micro_kernels.py from the pinned source
! micro_mg_cam.F90.  Do not edit by hand; edit the generator.
!
! Each body below is the original text with names substituted, never
! rewritten: `state_loc%q` becomes `q_loc`, `state%pdel` becomes `pdel`.
! Every expression, every loop nest and every bound is character for
! character what the pinned source computes, which is what lets the
! bit-for-bit gate mean something.  The elemental size-distribution and
! diameter routines the driver calls inline are called inline here too, from
! the same module, with the same property objects.
!
! This module is an addition to the source tree.  micro_mg_cam_tend is not
! changed to call it, so the oracle's own micro_mg_cam.o -- the machine code
! the gate validated -- stays untouched; only a Python-driven timestep runs
! these.

module pycam_micro_kernels

  use shr_kind_mod,   only: r8 => shr_kind_r8
  use ppgrid,         only: pcols, pver, pverp
  use constituents,   only: pcnst
  use micro_mg_utils, only: size_dist_param_basic, size_dist_param_liq, &
       mg_liq_props, mg_ice_props, avg_diameter, rhoi, rhosn, rhow, rhows

  implicit none
  private

{nl.join(PARAMETERS)}

{nl.join("  public :: " + block.name for block in BLOCKS)}

contains

{(nl + nl).join(routines)}

end module pycam_micro_kernels
'''


# -- direct-kernel descriptors ------------------------------------------------

_DTYPE = {"real(r8),": "float64", "integer, ": "int32"}
_DIMS = {P2: ["pcols", "pver"], P1: ["pcols"], P3: ["pcols", "pver", "pcnst"],
         PP: ["pcols", "pverp"], "": []}


def render_descriptors() -> str:
    import yaml

    kernels = []
    for block in BLOCKS:
        arguments = []
        for name, kind, dims, intent in block.arguments:
            extents = [*_DIMS[dims], "chunks"]
            arguments.append({
                "field": f"micro.{name}", "dtype": _DTYPE[kind], "rank": len(extents),
                "intent": intent, "chunk_axis": len(extents), "extents": extents,
            })
        kernels.append({
            "name": block.name, "routine": block.name,
            "symbol": f"pycam_pi_cam_{block.name}_v1", "action_id": 0,
            "modules": {"pycam_micro_kernels": [block.name],
                        "ppgrid": ["pcols", "pver", "pverp"], "constituents": ["pcnst"]},
            "arguments": arguments,
        })
    header = (
        "# GENERATED by tools/generate_pi_cam_micro_kernels.py -- do not edit.\n"
        "#\n"
        "# The routines a Python-driven microphysics timestep calls in the Fortran\n"
        "# driver's place, as direct kernels: twelve lifted from micro_mg_cam_tend's\n"
        "# own arithmetic.  Every field is a `micro.<dummy>` StatePool array Python\n"
        "# owns, one chunk per slice.  Merged into direct_kernels_promoted.yaml by\n"
        "# tools/build_pi_cam_promoted_kernels.py.\n"
    )
    return header + yaml.safe_dump({"schema_version": 1, "kernels": kernels}, sort_keys=False, width=100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stale = []
    for path, rendered in ((MODULE, render_module()), (DESCRIPTORS, render_descriptors())):
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
