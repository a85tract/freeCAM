"""Write the progress dashboard's snapshot for the published page.

The page at /freeCAM/progress/ cannot import freeCAM, so it reads this file:
the processes, the candidate kernels with their call relationships, the
capability and validation statuses with the records behind them, stamped with
the commit and a deterministic content hash.  It carries no personal path,
account, raw array or job log -- the builder refuses to write one.  Run after
anything that changes the ledger, the closure inventory, the relocation audit,
the runner manifest, or the Workflow Builder catalog:

    uv run python tools/export_progress_snapshot.py
    uv run python tools/export_progress_snapshot.py --check    # is the committed file current?
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "web" / "progress" / "public" / "progress.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the file's content differs from what the checkout produces")
    arguments = parser.parse_args(argv)
    sys.path.insert(0, str(REPO / "src"))
    from freecam.pi_cam.progress import build_progress_snapshot

    snapshot = build_progress_snapshot(REPO)
    snapshot["volatile"]["generated_at"] = (
        datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat())
    if arguments.check:
        if not arguments.output.is_file():
            print(f"{arguments.output} does not exist", file=sys.stderr)
            return 1
        current = json.loads(arguments.output.read_text())
        if current.get("content_hash") != snapshot["content_hash"]:
            print(f"{arguments.output} is stale: {current.get('content_hash')} != {snapshot['content_hash']}",
                  file=sys.stderr)
            return 1
        print(f"{arguments.output} is current ({snapshot['content_hash'][:12]})")
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n")
    print(f"wrote {arguments.output} ({arguments.output.stat().st_size // 1024} KB, "
          f"content {snapshot['content_hash'][:12]}, commit {(snapshot['volatile'].get('commit') or '?')[:7]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
