"""Write the Workflow Builder's catalog snapshot for the published page.

The page cannot import freeCAM, so it reads this file: the default workflow,
the process library with its reasons, the kernel capabilities, the audited
parameters and the control rules, stamped with the commit and a content
hash.  It carries no path, account, weight, log or state array -- a test
checks that.  Run after anything that changes the step plan, the physics
catalog, the process-support record or the parameter table:

    uv run python tools/export_workflow_catalog.py
    uv run python tools/export_workflow_catalog.py --check    # is the committed file current?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "web" / "public" / "catalog.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the file's content differs from what the checkout produces")
    arguments = parser.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    from freecam.pi_cam.workflow_builder import build_snapshot

    snapshot = build_snapshot(root=REPO)
    if arguments.check:
        if not arguments.output.is_file():
            print(f"{arguments.output} does not exist", file=sys.stderr)
            return 1
        current = json.loads(arguments.output.read_text())
        if current.get("catalog_hash") != snapshot["catalog_hash"]:
            print(f"{arguments.output} is stale: {current.get('catalog_hash')} != {snapshot['catalog_hash']}",
                  file=sys.stderr)
            return 1
        print(f"{arguments.output} is current ({snapshot['catalog_hash'][:12]})")
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    print(f"wrote {arguments.output} ({arguments.output.stat().st_size // 1024} KB, "
          f"catalog {snapshot['catalog_hash'][:12]}, commit {(snapshot.get('commit') or '?')[:7]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
