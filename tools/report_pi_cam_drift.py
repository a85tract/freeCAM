#!/usr/bin/env python3
"""Measure how far a run's state drifted from a reference run's.

    tools/report_pi_cam_drift.py --reference <dir> --candidate <dir> [--file r|h0]
        [--variables T Q:1e3:g/kg ...] [--output drift.json]

The bit-for-bit verifier (tools/verify_pi_cam.py) answers "the same?"; this
answers "how different?", per field, in scaled units, on one CAM file of each
run (the restart file by default).  --output writes the report as JSON that
names the files' logical names only, so it may be committed as a record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.pi_cam.drift import DEFAULT_VARIABLES, DriftRow, format_table, parse_variables, report  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, required=True, help="the reference run's output directory")
    parser.add_argument("--candidate", type=Path, required=True, help="the run to measure")
    parser.add_argument("--file", default="r", help="which CAM file to compare: r (restart, default), h0, ...")
    parser.add_argument("--variables", nargs="*", metavar="NAME[:SCALE[:UNIT]]",
                        help="fields to measure (default: " + ", ".join(v[0] for v in DEFAULT_VARIABLES) + ")")
    parser.add_argument("--output", type=Path, help="write the report as JSON here")
    args = parser.parse_args()
    variables = parse_variables(args.variables) if args.variables else DEFAULT_VARIABLES
    payload = report(args.reference, args.candidate, args.file, variables)
    rows = [DriftRow(f["name"], f["unit"], f["scale"], f["rms"], f["max"], f["mean"], f["reference_rms"],
                     f["count"], f["note"]) for f in payload["fields"]]
    print(f"{payload['reference_file']}  vs  {payload['candidate_file']}")
    print(format_table(rows))
    if args.output:
        args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
