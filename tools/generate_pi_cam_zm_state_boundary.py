#!/usr/bin/env python3
"""Make zm_conv_intr's per-chunk convective state readable by the pausable runners.

The Zhang-McFarlane driver keeps the mass fluxes, detrainment, cloud-top and
gathering indices of every chunk in private module arrays that zm_conv_tend
writes and zm_conv_tend_2 reads a few actions later.  The hoisted copies of
those two routines (native/pi_cam/pausable/deep_convection.yaml and
convective_tracer_transport.yaml) must read and write the very same storage,
never a copy, so ``0044-zm-conv-state-boundary.patch`` adds one ``public``
statement naming them, and the options the drivers test.  No declaration,
executable statement or numerical object changes.

    tools/generate_pi_cam_zm_state_boundary.py            # write the patch
    tools/generate_pi_cam_zm_state_boundary.py --check    # fail if stale
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
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam"
BOUNDARY = REPO / "native/pi_cam/control_patches/0044-zm-conv-state-boundary.patch"
RELATIVE = "src/physics/cam/zm_conv_intr.F90"

#: The declaration the statement follows: the last of the module's private indices.
ANCHOR = "integer  ::    nevapr_dpcu_idx  = 0"
PUBLIC = [
    "",
    "   ! pyCAM (control patch 0044): the per-chunk convective state zm_conv_tend writes",
    "   ! and zm_conv_tend_2 reads, and the options both test, readable by the pausable",
    "   ! runners' hoisted copies of the two routines.  No executable statement changes.",
    "   public :: mu, eu, du, md, ed, dp, dsubcld, jt, maxg, ideep, lengath, zmconv_org, ixorg, limcnv",
]


def edit(lines: list[str]) -> list[str]:
    out = list(lines)
    anchors = [i for i, line in enumerate(out) if line.strip() == ANCHOR]
    if len(anchors) != 1:
        raise SystemExit(f"{RELATIVE}: expected one anchor line {ANCHOR!r}, found {len(anchors)}")
    out[anchors[0] + 1:anchors[0] + 1] = PUBLIC
    return out


def _diff(before: Path, after: Path) -> str:
    diff = subprocess.run(
        ["git", "diff", "--no-index", "--unified=0", "--no-prefix", str(before), str(after)],
        capture_output=True, text=True,
    ).stdout.splitlines()
    body = [line for line in diff if not line.startswith(("diff --git", "index ", "--- ", "+++ "))]
    return "\n".join([f"--- a/{RELATIVE}", f"+++ b/{RELATIVE}"] + body) + "\n"


def render() -> dict[Path, str]:
    with tempfile.TemporaryDirectory(prefix="pycam-zm-state-") as temporary:
        root = Path(temporary)
        before = root / "before.F90"
        after = root / "after.F90"
        shutil.copy2(PINNED / RELATIVE, before)
        after.write_text("\n".join(edit(before.read_text().splitlines())) + "\n")
        return {BOUNDARY: _diff(before, after)}


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
