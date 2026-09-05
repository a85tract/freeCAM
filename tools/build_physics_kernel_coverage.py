#!/usr/bin/env python3
"""Write the physics kernel decoupling inventory, or check the committed one is current.

    uv run python tools/build_physics_kernel_coverage.py            # write validation/physics_kernel_decoupling.json
    uv run python tools/build_physics_kernel_coverage.py --check    # fail if the committed record is stale
    uv run python tools/build_physics_kernel_coverage.py --summary  # print the summary and the gaps

The record is built from the step plan, the physics catalog, the stage
classes, the segment-runner manifest, the function contracts and the
validation records; see freecam.pi_cam.kernel_coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUTPUT = REPO / "validation/physics_kernel_decoupling.json"


def main(argv: list[str] | None = None) -> int:
    from freecam.pi_cam.kernel_coverage import build_coverage, check_closure, write_coverage

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="compare with the committed record instead of writing")
    parser.add_argument("--summary", action="store_true", help="print the summary and the open items")
    arguments = parser.parse_args(argv)

    record = build_coverage()
    failures = check_closure(record)
    if arguments.summary or arguments.check:
        print(json.dumps(record["summary"], indent=1))
        for action in record["actions"]:
            if action["classification"] == "numeric_scheme" and action["coverage"] in ("gap", "partial"):
                print(f'{action["coverage"]:8s} {action["id"]:45s} kernels={action["kernels"]} '
                      f'candidates={len(action["kernel_candidates"])}')
        for kernel in record["kernels"]:
            print(f'{kernel["status"]:8s} {kernel["kernel"]:16s} missing={kernel["missing"]}')
        if failures:
            print("closure failures:", failures)
    if arguments.check:
        if not arguments.output.is_file():
            print(f"{arguments.output} does not exist; run without --check to write it", file=sys.stderr)
            return 1
        committed = json.loads(arguments.output.read_text())
        if committed.get("coverage_hash") != record["coverage_hash"]:
            print(f"{arguments.output} is stale: committed {committed.get('coverage_hash', '')[:12]}, "
                  f"current {record['coverage_hash'][:12]}", file=sys.stderr)
            return 1
        print(f"{arguments.output.relative_to(REPO)} is current ({record['coverage_hash'][:12]})")
        return 0 if not failures else 1
    write_coverage(arguments.output, record)
    print(f"wrote {arguments.output.relative_to(REPO)} ({record['coverage_hash'][:12]})")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
