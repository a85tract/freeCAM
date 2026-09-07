#!/usr/bin/env python3
"""Make the module state the pausable runners' hoisted drivers read visible to them.

A driver hoisted out of its module (native/pi_cam/pausable/*.yaml) still reads
that module's private variables: the per-chunk convective state zm_conv_tend
writes and zm_conv_tend_2 reads, vertical diffusion's namelist options and
field selectors, gravity wave drag's bands and source descriptions.  The
hoisted copy must read the very same storage, never a copy, so each of these
patches adds one ``public`` statement naming what the driver reads.  No
declaration, executable statement or numerical object changes; the device
build compiles the patched module for its .mod file alone and links the
oracle's object (``INTERFACE_MODULES`` in tools/build_pi_cam_devices.py).

    tools/generate_pi_cam_module_state_boundary.py            # write the patches
    tools/generate_pi_cam_module_state_boundary.py --check    # fail if any is stale
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import difflib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam"
PATCHES_DIR = REPO / "native/pi_cam/control_patches"


@dataclass(frozen=True)
class StatePatch:
    patch: str                 # file name under control_patches
    relative: str              # the module's path under components/cam
    anchor: str                # the declaration the statement follows (stripped text, unique)
    names: tuple[str, ...]     # what becomes public
    why: tuple[str, ...]       # the comment lines written above the statement

    @property
    def path(self) -> Path:
        return PATCHES_DIR / self.patch

    def lines(self) -> list[str]:
        return ["", *(f"   ! {line}" for line in self.why), f"   public :: {', '.join(self.names)}"]


STATE_PATCHES: tuple[StatePatch, ...] = (
    StatePatch(
        "0044-zm-conv-state-boundary.patch", "src/physics/cam/zm_conv_intr.F90",
        "integer  ::    nevapr_dpcu_idx  = 0",
        ("mu", "eu", "du", "md", "ed", "dp", "dsubcld", "jt", "maxg", "ideep", "lengath", "zmconv_org", "ixorg", "limcnv"),
        ("pyCAM (control patch 0044): the per-chunk convective state zm_conv_tend writes",
         "and zm_conv_tend_2 reads, and the options both test, readable by the pausable",
         "runners' hoisted copies of the two routines.  No executable statement changes."),
    ),
    StatePatch(
        "0045-vertical-diffusion-state-boundary.patch", "src/physics/cam/vertical_diffusion.F90",
        "integer, allocatable :: pmam_cnst_idx(:)             ! constituent indices of prognostic modal aerosols",
        ("eddy_scheme", "do_pseudocon_diff", "shallow_scheme", "fieldlist_wet", "fieldlist_dry", "fieldlist_molec",
         "ntop", "nbot", "tke_idx", "kvh_idx", "kvm_idx", "kvt_idx", "turbtype_idx", "smaw_idx", "tauresx_idx",
         "tauresy_idx", "vdiffnam", "ixcldice", "ixcldliq", "ixnumice", "ixnumliq", "qrl_idx", "wsedl_idx",
         "pblh_idx", "tpert_idx", "qpert_idx", "bprod_idx", "ipbl_idx", "kpblh_idx", "wstarPBL_idx", "tkes_idx",
         "went_idx", "qtl_flx_idx", "qti_flx_idx", "kv_top_pressure", "kv_top_scale", "kv_freetrop_scale",
         "diff_cnsrv_mass_check", "do_tms", "prog_modal_aero", "pmam_ncnst", "pmam_cnst_idx"),
        ("pyCAM (control patch 0045): the options, field selectors and buffer indices",
         "vertical_diffusion_tend reads, readable by the pausable runner's hoisted copy",
         "of the routine.  No executable statement changes."),
    ),
    StatePatch(
        "0046-gw-drag-state-boundary.patch", "src/physics/cam/gw_drag.F90",
        "logical          :: history_amwg                   ! output the variables used by the AMWG diag package",
        ("band_oro", "band_mid", "band_long", "effgw_oro", "effgw_cm", "effgw_cm_igw", "effgw_beres_dp",
         "effgw_beres_sh", "gw_polar_taper", "beres_dp_desc", "beres_sh_desc", "cm_desc", "cm_igw_desc",
         "kvt_idx", "ttend_dp_idx", "ttend_sh_idx", "frontgf_idx", "frontga_idx", "gw_spec_outflds"),
        ("pyCAM (control patch 0046): the wave bands, efficiencies, source descriptions,",
         "buffer indices and history writer gw_tend reads and calls, readable by the",
         "pausable runner's hoisted copy of the routine.  No executable statement changes."),
    ),
    StatePatch(
        "0047-chemistry-state-boundary.patch", "src/chemistry/mozart/chemistry.F90",
        "logical :: chem_rad_passive = .false.",
        ("chem_name", "chem_rad_passive", "ghg_chem", "ixcldice", "ixcldliq", "ixndrop", "ndx_cld", "ndx_cldtop",
         "ndx_cmfdqr", "ndx_nevapr", "ndx_pblh", "ndx_prain", "srcnam", "xactive_prates", "chem_step", "chem_freq"),
        ("pyCAM (control patch 0047): the options, indices and names chem_timestep_tend",
         "reads, readable by the pausable runner's hoisted copy of the routine.  No",
         "executable statement changes."),
    ),
    StatePatch(
        "0048-aero-model-state-boundary.patch", "src/chemistry/modal_aero/aero_model.F90",
        "logical :: modal_accum_coarse_exch = .false.",
        ("nmodes", "dgnumwet_idx", "qaerwat_idx", "fracis_idx", "wetdens_ap_idx", "nwetdep", "drydep_lq", "wetdep_lq",
         "sol_facti_cloud_borne", "sol_factb_interstitial", "sol_factic_interstitial", "modal_aero_bcscavcoef_get",
         "modal_aero_depvel_part"),
        ("pyCAM (control patch 0048): the indices, selectors and solubility factors the",
         "deposition drivers read, and the module's own procedures they call, readable by",
         "the pausable runners' hoisted copies.  No executable statement changes."),
    ),
)
BOUNDARIES = {entry.patch: entry.path for entry in STATE_PATCHES}


def edit(entry: StatePatch, lines: list[str]) -> list[str]:
    out = list(lines)
    anchors = [i for i, line in enumerate(out) if line.strip() == entry.anchor]
    if len(anchors) != 1:
        raise SystemExit(f"{entry.relative}: expected one anchor line {entry.anchor!r}, found {len(anchors)}")
    out[anchors[0] + 1:anchors[0] + 1] = entry.lines()
    return out


def _diff(entry: StatePatch, before: Path, after: Path) -> str:
    diff = subprocess.run(
        ["git", "diff", "--no-index", "--unified=0", "--no-prefix", str(before), str(after)],
        capture_output=True, text=True,
    ).stdout.splitlines()
    body = [line for line in diff if not line.startswith(("diff --git", "index ", "--- ", "+++ "))]
    return "\n".join([f"--- a/{entry.relative}", f"+++ b/{entry.relative}"] + body) + "\n"


def render() -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    with tempfile.TemporaryDirectory(prefix="pycam-module-state-") as temporary:
        root = Path(temporary)
        for entry in STATE_PATCHES:
            before = root / f"{Path(entry.relative).stem}-before.F90"
            after = root / f"{Path(entry.relative).stem}-after.F90"
            shutil.copy2(PINNED / entry.relative, before)
            after.write_text("\n".join(edit(entry, before.read_text().splitlines())) + "\n")
            rendered[entry.path] = _diff(entry, before, after)
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
                    fromfile=f"{path.name} (committed)", tofile=f"{path.name} (generated)")))
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(REPO)}")
    if stale:
        sys.stderr.write("\nstale: " + ", ".join(str(p.relative_to(REPO)) for p in stale) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
