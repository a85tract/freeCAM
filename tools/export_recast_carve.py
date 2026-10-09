"""Write the carve manifest RecastEngine's refactor recipe reads (recast/carve.json).

freeCAM's carve-out of the pinned iCESM is the ordered control patches and the
support sources tools/apply_pi_cam_source_patches.py applies, under
components/cam of the external at the revision tools/prepare_pi_cam_source.py
pins.  The manifest says exactly that, as data (schema ``recast.carve.v1``),
plus where the prepared tree records what went into it (its
.pycam-source.json) and the arithmetic statements the support sources carry
that are neither copies of CAM's own lines nor plain kind conversions, each
with the reason it was accepted (recast/numerics_declared.json).  It is
generated from those lists so it cannot drift from them; run after changing
either:

    uv run python tools/export_recast_carve.py
    uv run python tools/export_recast_carve.py --check    # is the committed file current?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "recast" / "carve.json"
DECLARED = REPO / "recast" / "numerics_declared.json"
UPSTREAM = "external/iCESM1.3.1_fzhu"
BASE = "components/cam"


def build(repo: Path = REPO) -> dict:
    sys.path.insert(0, str(repo / "tools"))
    import apply_pi_cam_source_patches as patches
    import prepare_pi_cam_source as prepare

    declared = json.loads((repo / "recast" / "numerics_declared.json").read_text())
    return {
        "schema": "recast.carve.v1",
        "component": "pi-cam",
        "upstream": {"path": UPSTREAM, "revision": prepare.PINNED_REVISIONS["."], "base": BASE},
        "patches": list(patches.PATCHES),
        "files": [{"source": source, "target": target} for source, target in patches.SUPPORT_SOURCES],
        "carved": {
            "path": str(prepare.DEFAULT_OUTPUT.relative_to(prepare.REPO)),
            "provenance": ".pycam-source.json",
            "patches_key": "applied_patches",
            "files_key": "installed_support_sources",
            "revision_key": "revisions",
        },
        "numerics": {"declared": declared["declared"]},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the file's content differs from what the checkout produces")
    arguments = parser.parse_args(argv)
    text = json.dumps(build(), indent=2) + "\n"
    if arguments.check:
        if not arguments.output.is_file() or arguments.output.read_text() != text:
            print(f"{arguments.output} is not current; run tools/export_recast_carve.py",
                  file=sys.stderr)
            return 1
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(text)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
