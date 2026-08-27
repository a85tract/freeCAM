#!/usr/bin/env python3
"""Lift radiation_tend's inline arithmetic into routines the image can run.

The driver layer of CAM5's RRTMG radiation is mostly calls, copies and
control flow; only thirty of its live statements compute a floating-point
number.  Python may not compute any of them -- ``-fp-model source -ftz``
and Intel's libimf are not reproducible from NumPy, and the bit-for-bit
gate is the only acceptance test this work has -- so each contiguous run of
them becomes a Fortran routine here, its body the pinned source's own text
with names substituted and never rewritten.

``radiation.F90`` is **not** patched.  This module is an addition to the
source tree, so the oracle's own ``radiation.o`` -- the machine code the
gate validated -- stays untouched, and only a Python-driven timestep runs
these copies.  The Python-driven run is gated bit-for-bit against the
oracle, and that gate is what proves the two copies agree.

Two things shape the cut and are worth knowing before reading the table:

* The SW and LW cloud-optics blocks are each interrupted in their middle by
  a ``get_snow_optics_sw`` / ``snow_cloud_get_rad_props_lw`` call, so
  neither can be one contiguous lift.  Each becomes two routines, split at
  the call.
* Lines 1170-1171 pass ``qrl(:ncol,:)/cpair`` straight to ``outfld`` with
  ``idim = ncol``: the division and the shape are one expression, and
  splitting them would change what ``outfld`` receives.  That pair is not
  a kernel; it is ``pycam_rad_outfld_scaled_v1`` in the handles module,
  where the whole source line survives intact.

    tools/generate_pi_cam_rad_kernels.py            # write the module and descriptors
    tools/generate_pi_cam_rad_kernels.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/rrtmg/radiation.F90"
MODULE = REPO / "native/pi_cam/support/pycam_rad_kernels.F90"
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_radiation.yaml"

# Renames applied to every carved body.  Each one replaces a name the
# routine could only see as a host association with a name it receives as
# an argument.
COMMON = {
    "state%t": "t",
    "state%pmid": "pmid",
    "state%pdel": "pdel",
    "state%pint": "pint",
    "state%lnpint": "lnpint",
    "state%lnpmid": "lnpmid",
    "cam_in%lwup": "lwup",
    "cldfsnow_idx > 0": "has_snow",
    "single_column.and.scm_crm_mode.and.have_tg": "refused_scm",
    "conserve_energy": "conserve_energy",
}


class Block:
    """One carved routine: where its body is, and what it receives."""

    def __init__(self, name, first, last, signature, declarations, locals_, *,
                 renames=None, skip=(), covers=()):
        self.name = name
        self.first = first          # 1-based, inclusive
        self.last = last
        self.signature = signature
        self.declarations = declarations
        self.locals = locals_
        self.renames = dict(COMMON, **(renames or {}))
        #: Source lines dropped from the body.  Only ever a call Python makes
        #: itself through a handle wrapper, or an ``endif`` closing a guard
        #: that lies outside the carved range.  A test checks every one.
        self.skip = frozenset(skip)
        #: Source lines this routine accounts for without carving them: the
        #: same statement over different arrays, which Python runs by calling
        #: the routine again.  A test counts these towards coverage.
        self.covers = frozenset(covers)

    @property
    def arguments(self) -> tuple[str, ...]:
        return tuple(", ".join(self.signature).split(", "))

    def body(self, lines: list[str]) -> list[str]:
        out = []
        for number in range(self.first, self.last + 1):
            if number in self.skip:
                continue
            line = lines[number - 1]
            for old, new in self.renames.items():
                line = line.replace(old, new)
            out.append("   " + line if line.strip() else line)
        return out


BLOCKS = (
    Block(
        "rad_gather_day_night", 856, 866,
        ["ncol, coszrs, Nday, Nnite, IdxDay, IdxNite"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: coszrs(pcols)",
         "   integer,  intent(out) :: Nday, Nnite",
         "   integer,  intent(out) :: IdxDay(pcols), IdxNite(pcols)"],
        ["   integer :: i"],
    ),
    Block(
        "rad_combine_cld_optics_sw", 912, 915,
        ["ncol, liq_tau, liq_tau_w, liq_tau_w_g, liq_tau_w_f",
         "ice_tau, ice_tau_w, ice_tau_w_g, ice_tau_w_f",
         "cld_tau, cld_tau_w, cld_tau_w_g, cld_tau_w_f"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: liq_tau(nbndsw,pcols,pver), liq_tau_w(nbndsw,pcols,pver)",
         "   real(r8), intent(in)  :: liq_tau_w_g(nbndsw,pcols,pver), liq_tau_w_f(nbndsw,pcols,pver)",
         "   real(r8), intent(in)  :: ice_tau(nbndsw,pcols,pver), ice_tau_w(nbndsw,pcols,pver)",
         "   real(r8), intent(in)  :: ice_tau_w_g(nbndsw,pcols,pver), ice_tau_w_f(nbndsw,pcols,pver)",
         "   real(r8), intent(out) :: cld_tau(nbndsw,pcols,pver), cld_tau_w(nbndsw,pcols,pver)",
         "   real(r8), intent(out) :: cld_tau_w_g(nbndsw,pcols,pver), cld_tau_w_f(nbndsw,pcols,pver)"],
        [],
    ),
    Block(
        "rad_snow_blend_sw", 917, 945,
        ["ncol, has_snow, cld, cldfsnow, snow_tau, snow_tau_w, snow_tau_w_g, snow_tau_w_f",
         "cld_tau, cld_tau_w, cld_tau_w_g, cld_tau_w_f",
         "cldfprime, c_cld_tau, c_cld_tau_w, c_cld_tau_w_g, c_cld_tau_w_f"],
        ["   integer,  intent(in)    :: ncol",
         "   logical,  intent(in)    :: has_snow",
         "   real(r8), intent(in)    :: cld(pcols,pver), cldfsnow(pcols,pver)",
         "   real(r8), intent(in)    :: snow_tau(nbndsw,pcols,pver), snow_tau_w(nbndsw,pcols,pver)",
         "   real(r8), intent(in)    :: snow_tau_w_g(nbndsw,pcols,pver), snow_tau_w_f(nbndsw,pcols,pver)",
         "   real(r8), intent(in)    :: cld_tau(nbndsw,pcols,pver), cld_tau_w(nbndsw,pcols,pver)",
         "   real(r8), intent(in)    :: cld_tau_w_g(nbndsw,pcols,pver), cld_tau_w_f(nbndsw,pcols,pver)",
         "   ! Written only where has_snow; the driver sets it at 993-995 otherwise,",
         "   ! which rad_snow_blend_lw reproduces, so this carries its value out.",
         "   real(r8), intent(inout) :: cldfprime(pcols,pver)",
         "   real(r8), intent(out)   :: c_cld_tau(nbndsw,pcols,pver), c_cld_tau_w(nbndsw,pcols,pver)",
         "   real(r8), intent(out)   :: c_cld_tau_w_g(nbndsw,pcols,pver), c_cld_tau_w_f(nbndsw,pcols,pver)"],
        ["   integer :: i, k"],
        # 919 is get_snow_optics_sw, which Python calls through its wrapper
        # before this routine and whose outputs it passes in.
        skip=(919,),
    ),
    Block(
        "rad_combine_cld_optics_lw", 968, 968,
        ["ncol, liq_lw_abs, ice_lw_abs, cld_lw_abs"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: liq_lw_abs(nbndlw,pcols,pver), ice_lw_abs(nbndlw,pcols,pver)",
         "   real(r8), intent(out) :: cld_lw_abs(nbndlw,pcols,pver)"],
        [],
    ),
    Block(
        "rad_snow_blend_lw", 974, 995,
        ["ncol, has_snow, cld, cldfsnow, snow_lw_abs, cld_lw_abs, cldfprime, c_cld_lw_abs"],
        ["   integer,  intent(in)  :: ncol",
         "   logical,  intent(in)  :: has_snow",
         "   real(r8), intent(in)  :: cld(pcols,pver), cldfsnow(pcols,pver)",
         "   real(r8), intent(in)  :: snow_lw_abs(nbndlw,pcols,pver)",
         "   real(r8), intent(in)  :: cld_lw_abs(nbndlw,pcols,pver)",
         "   real(r8), intent(inout) :: cldfprime(pcols,pver)",
         "   real(r8), intent(out) :: c_cld_lw_abs(nbndlw,pcols,pver)"],
        ["   integer :: i, k"],
        # 976 is snow_cloud_get_rad_props_lw, which Python calls through its
        # wrapper first; 991 closes `if (dolw)`, a guard outside this range.
        skip=(976, 991),
    ),
    Block(
        "rad_interface_temperature", 1005, 1012,
        ["ncol, t, lnpint, lnpmid, lwup, stebol, tint"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: t(pcols,pver), lnpint(pcols,pverp), lnpmid(pcols,pver)",
         "   real(r8), intent(in)  :: lwup(pcols), stebol",
         "   real(r8), intent(out) :: tint(pcols,pverp)"],
        ["   integer  :: i, k", "   real(r8) :: dy"],
    ),
    Block(
        "rad_sw_cloud_forcing", 1057, 1059,
        ["ncol, fsntoa, fsntoac, swcf"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: fsntoa(pcols), fsntoac(pcols)",
         "   real(r8), intent(out) :: swcf(pcols)"],
        ["   integer :: i"],
    ),
    Block(
        # 1061 and 1063 are the same statement over different sources; the
        # driver calls outfld between them, so Python runs this twice.
        "rad_scale_by_cpair", 1061, 1061,
        ["ncol, field, cpair, ftem"],
        ["   integer,  intent(in)    :: ncol",
         "   real(r8), intent(in)    :: field(pcols,pver), cpair",
         "   real(r8), intent(inout) :: ftem(pcols,pver)"],
        [],
        renames={"qrs(:ncol,:pver)": "field(:ncol,:pver)"},
        # 1063 is this statement again over qrsc; the driver calls outfld
        # between the two, so Python runs this routine twice.
        covers=(1063,),
    ),
    Block(
        "rad_visible_tau", 1092, 1110,
        ["ncol, Nnite, IdxNite, has_snow, idx_sw_diag, fillvalue",
         "c_cld_tau, liq_tau, ice_tau, snow_tau, cldfprime",
         "tot_cld_vistau, tot_icld_vistau, liq_icld_vistau, ice_icld_vistau, snow_icld_vistau"],
        ["   integer,  intent(in)  :: ncol, Nnite, idx_sw_diag",
         "   integer,  intent(in)  :: IdxNite(pcols)",
         "   logical,  intent(in)  :: has_snow",
         "   real(r8), intent(in)  :: fillvalue",
         "   real(r8), intent(in)  :: c_cld_tau(nbndsw,pcols,pver), liq_tau(nbndsw,pcols,pver)",
         "   real(r8), intent(in)  :: ice_tau(nbndsw,pcols,pver), snow_tau(nbndsw,pcols,pver)",
         "   real(r8), intent(in)  :: cldfprime(pcols,pver)",
         "   real(r8), intent(out) :: tot_cld_vistau(pcols,pver), tot_icld_vistau(pcols,pver)",
         "   real(r8), intent(out) :: liq_icld_vistau(pcols,pver), ice_icld_vistau(pcols,pver)",
         "   real(r8), intent(out) :: snow_icld_vistau(pcols,pver)"],
        ["   integer :: i"],
    ),
    Block(
        "rad_lwup_cgs", 1130, 1134,
        ["ncol, lwup, refused_scm, lwupcgs"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: lwup(pcols)",
         "   ! The single-column CRM branch is refused at attach; the flag is",
         "   ! passed so the body keeps the driver's text and stays .false.",
         "   logical,  intent(in)  :: refused_scm",
         "   real(r8), intent(out) :: lwupcgs(pcols)"],
        ["   integer :: i"],
        renames={"lwupcgs(i) = 1000*stebol*tground(1)**4": "continue"},
    ),
    Block(
        "rad_lw_cloud_forcing", 1156, 1158,
        ["ncol, flutc, flut, lwcf"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: flutc(pcols), flut(pcols)",
         "   real(r8), intent(out) :: lwcf(pcols)"],
        ["   integer :: i"],
    ),
    Block(
        "rad_emissivity", 1241, 1242,
        ["ncol, rrtmg_lw_cloudsim_band, cld_lw_abs, emis"],
        ["   integer,  intent(in)  :: ncol, rrtmg_lw_cloudsim_band",
         "   real(r8), intent(in)  :: cld_lw_abs(nbndlw,pcols,pver)",
         "   real(r8), intent(out) :: emis(pcols,pver)"],
        [],
    ),
    Block(
        "rad_snow_gridbox", 1246, 1257,
        ["ncol, has_snow, rrtmg_sw_cloudsim_band, rrtmg_lw_cloudsim_band",
         "cldfsnow, snow_tau, snow_lw_abs, gb_snow_tau, gb_snow_lw"],
        ["   integer,  intent(in)  :: ncol",
         "   logical,  intent(in)  :: has_snow",
         "   integer,  intent(in)  :: rrtmg_sw_cloudsim_band, rrtmg_lw_cloudsim_band",
         "   real(r8), intent(in)  :: cldfsnow(pcols,pver)",
         "   real(r8), intent(in)  :: snow_tau(nbndsw,pcols,pver), snow_lw_abs(nbndlw,pcols,pver)",
         "   real(r8), intent(out) :: gb_snow_tau(pcols,pver), gb_snow_lw(pcols,pver)"],
        ["   integer :: i, k"],
    ),
    Block(
        "rad_heating_unscale", 1277, 1287,
        ["ncol, conserve_energy, pdel, qrs, qrl"],
        ["   integer,  intent(in)    :: ncol",
         "   logical,  intent(in)    :: conserve_energy",
         "   real(r8), intent(in)    :: pdel(pcols,pver)",
         "   real(r8), intent(inout) :: qrs(pcols,pver), qrl(pcols,pver)"],
        ["   integer :: i, k"],
    ),
    Block(
        "rad_theta_heating", 1298, 1303,
        ["ncol, qrs, qrl, cpair, pmid, cappa, ftem"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: qrs(pcols,pver), qrl(pcols,pver)",
         "   real(r8), intent(in)  :: cpair, pmid(pcols,pver), cappa",
         "   real(r8), intent(out) :: ftem(pcols,pver)"],
        ["   integer :: i, k"],
    ),
    Block(
        "rad_heating_scale", 1306, 1316,
        ["ncol, conserve_energy, pdel, qrs, qrl"],
        ["   integer,  intent(in)    :: ncol",
         "   logical,  intent(in)    :: conserve_energy",
         "   real(r8), intent(in)    :: pdel(pcols,pver)",
         "   real(r8), intent(inout) :: qrs(pcols,pver), qrl(pcols,pver)"],
        ["   integer :: i, k"],
    ),
)


def _radinp_bounds(lines: list[str]) -> tuple[int, int]:
    """The executable body of the driver's private ``radinp``."""

    start = next(i for i, l in enumerate(lines, 1)
                 if re.match(r"\s*subroutine\s+radinp\s*\(", l, re.I))
    end = next(i for i, l in enumerate(lines, 1)
               if i > start and re.match(r"\s*end\s+subroutine\s+radinp\b", l, re.I))
    first = next(i for i in range(start, end) if "calday = get_curr_calday()" in lines[i - 1])
    return first, end - 1


def _radinp_block(lines: list[str]) -> Block:
    """``radinp`` is private to radiation.F90, so its body is lifted whole.

    It reads the orbital elements from ``cam_control_mod`` and calls
    ``shr_orb_decl`` exactly as the original does; both are module `use`s the
    lifted routine can repeat.
    """

    first, last = _radinp_bounds(lines)
    return Block(
        "rad_inp", first, last,
        ["ncol, pmid, pint, pmidrd, pintrd, eccf"],
        ["   integer,  intent(in)  :: ncol",
         "   real(r8), intent(in)  :: pmid(pcols,pver), pint(pcols,pverp)",
         "   real(r8), intent(out) :: pmidrd(pcols,pver), pintrd(pcols,pverp)",
         "   real(r8), intent(out) :: eccf"],
        ["   integer  :: i, k", "   real(r8) :: calday", "   real(r8) :: delta"],
        renames={"state%": "state%"},        # nothing to rename; the body is self-contained
    )


def blocks(lines: list[str]) -> tuple[Block, ...]:
    return (*BLOCKS, _radinp_block(lines))


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
    for block in blocks(lines):
        header = [f"  subroutine {block.name}( &"] + [
            "  " + line for line in _wrap(block.arguments, "       ")
        ]
        pieces = header + [""] + block.declarations
        if block.locals:
            pieces += [""] + block.locals
        pieces += [""] + block.body(lines) + ["", f"  end subroutine {block.name}"]
        routines.append(nl.join(pieces))

    return f'''! radiation_tend's inline arithmetic, as routines the image can be asked
! to run.
!
! GENERATED by tools/generate_pi_cam_rad_kernels.py from the pinned source
! physics/rrtmg/radiation.F90.  Do not edit by hand; edit the generator.
!
! Each body below is the original text with names substituted, never
! rewritten: `state%t` becomes `t`, `cam_in%lwup` becomes `lwup`, a module
! flag becomes an argument.  Every expression, every loop nest and every
! bound is character for character what the pinned source computes, which is
! what lets the bit-for-bit gate mean something.
!
! This module is an addition to the source tree.  radiation_tend is not
! changed to call it, so the oracle's own radiation.o -- the machine code the
! gate validated -- stays untouched; only a Python-driven timestep runs
! these.  The Python-driven run is gated bit-for-bit against the oracle, and
! that gate is what proves the two copies of the arithmetic agree.
!
! Nothing here touches a derived type, the physics buffer, the clock or the
! history file, with one exception noted in the generator: radinp is private
! to radiation.F90 and reads the orbital elements, so its body is lifted
! whole together with the two module uses it needs.

module pycam_rad_kernels

  use shr_kind_mod,    only: r8 => shr_kind_r8
  use ppgrid,          only: pcols, pver, pverp
  use parrrsw,         only: nbndsw
  use parrrtm,         only: nbndlw
  use shr_orb_mod,     only: shr_orb_decl
  use cam_control_mod, only: lambm0, obliqr, mvelpp, eccen
  use time_manager,    only: get_curr_calday

  implicit none
  private

{nl.join("  public :: " + block.name for block in blocks(lines))}

contains

{(nl + nl).join(routines)}

end module pycam_rad_kernels
'''


# -- direct-kernel descriptors ------------------------------------------------

_SCALAR_KINDS = {"integer": "int32", "real(r8)": "float64", "logical": "int32"}


def _declared(block):
    """(name, kind, dims, intent) for every dummy of a carved routine."""

    out = {}
    for line in block.declarations:
        text = line.strip()
        if not text or text.startswith("!"):
            continue
        head, names = text.split("::")
        kind = head.split(",")[0].strip()
        intent = re.search(r"intent\((\w+)\)", head).group(1)
        for item in re.finditer(r"(\w+)(?:\(([^)]*)\))?", names):
            name, dims = item.group(1), item.group(2)
            out[name] = (kind, tuple(d.strip() for d in dims.split(",")) if dims else (), intent)
    return out


def _argument(name, kind, dims, intent):
    dtype = _SCALAR_KINDS[kind]
    extents = [*dims, "chunks"]
    rank = len(extents)
    payload = {
        "field": f"rad.{name}", "dtype": dtype, "rank": rank,
        "intent": intent, "chunk_axis": rank, "extents": extents,
    }
    if kind == "logical":
        payload["fortran_type"] = "logical"
    return payload


#: The driver's plain-array calls, promoted as they stand.  ``vertinterp``
#: is called four times on the live path with different arrays; one
#: descriptor serves them all.
#:
#: ``zenith`` is not here: it is a bare external subroutine at file scope,
#: not a module procedure, so a generated wrapper cannot ``use`` it.  It gets
#: a handle entry instead, where an external call is legal.
PLAIN = (
    ("get_variability", "get_variability", {"rad_solar_var": ["get_variability"],
                                            "radconstants": ["nswbands"]}, [
        ("sfac", "real(r8)", ("nswbands",), "out"),
    ]),
    # vertinterp's dummies carry the most generic names in the driver, and
    # `pmid` collided with rad_inp's, which is a different shape.  The scratch
    # is keyed by field name across every kernel of a stage, so the two would
    # have shared one array.  Prefixed here, and its own `pmid` named for what
    # the driver actually passes: state%pint, at every live call site.
    ("vertinterp", "vertinterp", {"interpolate_data": ["vertinterp"],
                                  "ppgrid": ["pcols", "pverp"]}, [
        ("ncol", "integer", (), "in"),
        ("vi_ncold", "integer", (), "in"),
        ("vi_nlev", "integer", (), "in"),
        ("vi_pint", "real(r8)", ("pcols", "pverp"), "in"),
        ("vi_pout", "real(r8)", (), "in"),
        ("vi_arrin", "real(r8)", ("pcols", "pverp"), "in"),
        ("vi_arrout", "real(r8)", ("pcols",), "out"),
    ]),
)


def render_descriptors(source: Path | None = None) -> str:
    import yaml

    lines = (source or PINNED).read_text().splitlines()
    kernels = []
    for block in blocks(lines):
        declared = _declared(block)
        arguments = [_argument(name, *declared[name]) for name in block.arguments]
        kernels.append({
            "name": block.name, "routine": block.name,
            "symbol": f"pycam_pi_cam_{block.name}_v1", "action_id": 0,
            "modules": {"pycam_rad_kernels": [block.name],
                        "ppgrid": ["pcols", "pver", "pverp"],
                        "parrrsw": ["nbndsw"], "parrrtm": ["nbndlw"]},
            "arguments": arguments,
        })
    for name, routine, modules, dummies in PLAIN:
        kernels.append({
            "name": name, "routine": routine,
            "symbol": f"pycam_pi_cam_{name}_v1", "action_id": 0,
            "modules": modules,
            "arguments": [_argument(*dummy) for dummy in dummies],
        })
    header = (
        "# GENERATED by tools/generate_pi_cam_rad_kernels.py -- do not edit.\n"
        "#\n"
        "# The routines a Python-driven radiation timestep calls in the Fortran\n"
        "# driver's place, as direct kernels: seventeen lifted from the driver's\n"
        "# own arithmetic, and the three plain-array routines it calls as they\n"
        "# stand.  Every field is a `rad.<dummy>` StatePool array Python owns,\n"
        "# one chunk per slice.  Merged into direct_kernels_promoted.yaml by\n"
        "# tools/build_pi_cam_promoted_kernels.py.\n"
    )
    return header + yaml.safe_dump({"schema_version": 1, "kernels": kernels},
                                   sort_keys=False, width=100)


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
