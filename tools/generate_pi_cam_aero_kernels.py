#!/usr/bin/env python3
"""Lift microp_aero_run's inline arithmetic into routines the image can run.

The aerosol activation driver (``microp_aero.F90:345-713``) is 209
statements, of which twenty on the live path compute a floating-point
number outside the routines it calls: the activation outputs it zeroes,
the air density, the sub-grid vertical velocity from the PBL's turbulent
kinetic energy, the liquid cloud fractions, and the contact-freezing dust
bins.  Python may compute none of them, so each contiguous run becomes a
Fortran routine here, its body the pinned text with names substituted and
never rewritten.

Two branches this configuration never takes are dropped by line number and
a test checks the drop is exactly those: the bulk-aerosol arm of the
contact-freezing block (``.not. clim_modal_aero``) and the CLUBB arm of
the sub-grid velocity block.  The flags this configuration *does* read at
runtime -- ``separate_dust`` and ``use_preexisting_ice`` -- stay in the
text as arguments, so the routine still branches the way the source does.

``rn_dst1..4`` are private parameters of ``microp_aero``; their
declarations are copied here so the lifted text reads the same literals.

    tools/generate_pi_cam_aero_kernels.py            # write the module and descriptors
    tools/generate_pi_cam_aero_kernels.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/microp_aero.F90"
MODULE = REPO / "native/pi_cam/support/pycam_aero_kernels.F90"
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_aero.yaml"

#: Renames applied to every carved body: the associate block's names for the
#: state's components, which the routine now receives directly.
COMMON: dict[str, str] = {}

P2, P1, PP = "(pcols,pver)", "(pcols)", "(pcols,pverp)"
P2_4 = "(pcols,pver,4)"


#: microp_aero.F90:62-65, the fixed dust bin radii.
PARAMETERS_AT = (62, 63, 64, 65)


class Block:
    """One carved routine: where its body is, and what it receives."""

    def __init__(self, name, first, last, arguments, *, skip=(), locals_=(),
                 line_renames=None, note=""):
        self.name = name
        self.first = first
        self.last = last
        self.arguments = arguments
        self.skip = frozenset(skip)
        self.locals = tuple(locals_)
        self.line_renames = dict(line_renames or {})
        self.note = note

    @property
    def dummies(self) -> tuple[str, ...]:
        return tuple(a[0] for a in self.arguments)

    def declarations(self) -> list[str]:
        return [f"   {kind:9s} intent({intent}) :: {name}{dims}"
                for name, kind, dims, intent in self.arguments]

    def body(self, lines: list[str]) -> list[str]:
        out = []
        for number in range(self.first, self.last + 1):
            if number in self.skip:
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


def _logical(*names):
    return [(n, "logical,  ", "", "in") for n in names]


def _scalar(*names):
    return [(n, "real(r8),", "", "in") for n in names]


BLOCKS = (
    Block(
        # 449-457: the activation outputs start at zero and the dust bins at
        # their fixed radii
        "aero_initial_bins", 449, 457,
        _int("ncol") + _real("npccn") + _real("nacon", "rndst", dims=P2_4),
    ),
    Block(
        # 465-469: air density, from the state the routine was given
        "aero_air_density", 465, 469,
        _int("ncol", "top_lev") + _scalar("rair")
        + _real("pmid", "t", intent="in") + _real("rho"),
        locals_=("   integer :: i, k",),
    ),
    Block(
        # 515-549: the sub-grid vertical velocity.  The CLUBB arm of both
        # branches is dropped; `use_preexisting_ice` stays a runtime flag.
        "aero_subgrid_velocity", 515, 549,
        _int("ncol", "top_lev") + _logical("use_preexisting_ice")
        + _real("tke", dims=PP, intent="in") + _real("wsub", "wsubi"),
        # the select case keeps its 'diag_TKE' arm (523-524) and drops the
        # eddy-diffusivity arm; the wsubi test keeps its else arm
        skip=(521, 522, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534,
              536, 537, 538, 539, 544),
        locals_=("   integer :: i, k",),
        note="the select case on eddy_scheme and the CLUBB arm of the wsubi test",
    ),
    Block(
        # 564-568: the liquid cloud fraction the activation sees
        "aero_liquid_fraction", 564, 568,
        _int("ncol", "top_lev") + _scalar("mincld")
        + _real("ast", intent="in") + _real("lcldm"),
        locals_=("   integer :: i, k",),
    ),
    Block(
        # 578-588: the cloud fraction split into its liquid part, new and old
        "aero_cloud_fraction_split", 578, 588,
        _int("ncol", "top_lev") + _scalar("qsmall")
        + _real("qc", "qi", "cldn", "cldo", intent="in") + _real("lcldn", "lcldo"),
        locals_=("   integer :: i, k", "   real(r8) :: qcld"),
    ),
    Block(
        # 596: the activation tendency ndrop returned
        "aero_npccn_from_mixnuc", 596, 596,
        _int("ncol") + _real("nctend_mixnuc", intent="in") + _real("npccn"),
    ),
    Block(
        # 632-684: the contact-freezing dust bins.  The bulk-aerosol arm
        # (667-680) is dropped with its `else`.
        "aero_contact_freezing", 632, 684,
        _int("ncol", "top_lev") + _logical("separate_dust")
        + _real("t", "coarse_dust", "coarse_nacl", "num_coarse", "rho",
                "dgnumwet_coarse", intent="in")
        + _real("nacon", "rndst", dims=P2_4),
        # `if (clim_modal_aero) then` and the bulk arm it guards, with its end if
        skip=(637,) + tuple(range(667, 681)),
        # the source indexes one plane of the mode-resolved diameters; the
        # walk passes that plane, so the reference loses its third index and
        # nothing else changes
        line_renames={662: {"dgnumwet(i,k,mode_coarse_dst_idx)": "dgnumwet_coarse(i,k)"}},
        locals_=("   integer :: i, k", "   real(r8) :: dmc, ssmc, wght"),
        note="the bulk-aerosol arm and its `if (clim_modal_aero) then`",
    ),
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
    return f'''! microp_aero_run's inline arithmetic, as routines the image can be asked
! to run.
!
! GENERATED by tools/generate_pi_cam_aero_kernels.py from the pinned source
! microp_aero.F90.  Do not edit by hand; edit the generator.
!
! Each body below is the original text, never rewritten.  Two arms this
! configuration never takes are dropped by line number -- the bulk-aerosol
! contact freezing and the CLUBB sub-grid velocity -- and the flags the
! configuration does read stay as arguments, so what remains branches the
! way the source branches.
!
! This module is an addition to the source tree.  microp_aero_run is not
! changed to call it, so the oracle's own microp_aero.o stays untouched;
! only a Python-driven timestep runs these.

module pycam_aero_kernels

  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid,       only: pcols, pver, pverp

  implicit none
  private

{nl.join("  " + lines[n - 1].strip() for n in PARAMETERS_AT)}

{nl.join("  public :: " + block.name for block in BLOCKS)}

contains

{(nl + nl).join(routines)}

end module pycam_aero_kernels
'''


# -- direct-kernel descriptors ------------------------------------------------

_DTYPE = {"real(r8),": "float64", "integer, ": "int32", "logical,  ": "int32"}
#: a Fortran logical travels as one int32 per chunk and the wrapper declares
#: it logical again, which the descriptor has to say
_CARRIER = {"logical,  ": "logical"}
_DIMS = {P2: ["pcols", "pver"], P1: ["pcols"], PP: ["pcols", "pverp"],
         P2_4: ["pcols", "pver", "4"], "": []}


def render_descriptors() -> str:
    import yaml

    kernels = []
    for block in BLOCKS:
        arguments = []
        for name, kind, dims, intent in block.arguments:
            extents = [*_DIMS[dims], "chunks"]
            entry = {
                "field": f"aero.{name}", "dtype": _DTYPE[kind], "rank": len(extents),
                "intent": intent, "chunk_axis": len(extents), "extents": extents,
            }
            if kind in _CARRIER:
                entry["fortran_type"] = _CARRIER[kind]
            arguments.append(entry)
        kernels.append({
            "name": block.name, "routine": block.name,
            "symbol": f"pycam_pi_cam_{block.name}_v1", "action_id": 0,
            "modules": {"pycam_aero_kernels": [block.name],
                        "ppgrid": ["pcols", "pver", "pverp"]},
            "arguments": arguments,
        })
    header = (
        "# GENERATED by tools/generate_pi_cam_aero_kernels.py -- do not edit.\n"
        "#\n"
        "# The routines a Python-driven aerosol activation calls in the Fortran\n"
        "# driver's place, as direct kernels: seven lifted from microp_aero_run's\n"
        "# own arithmetic.  Every field is an `aero.<dummy>` StatePool array Python\n"
        "# owns, one chunk per slice.  Merged into direct_kernels_promoted.yaml by\n"
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
