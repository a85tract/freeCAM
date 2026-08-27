#!/usr/bin/env python3
"""How far a run drifted from the oracle, field by field.

A surrogate in a kernel's place cannot be bit-for-bit: it is a different
computation, and the bit-for-bit tool answers only yes or no.  What is
worth knowing instead is how far each history field moved, in units of
the field's own variability, and whether anything went non-finite.

    tools/report_pi_cam_surrogate_drift.py --reference <oracle run> \\
        --candidate <surrogate run> --output <report>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def history_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.cam.h*.nc"))


def compare(reference: Path, candidate: Path) -> dict[str, object]:
    import netCDF4

    fields: list[dict[str, object]] = []
    with netCDF4.Dataset(reference) as a, netCDF4.Dataset(candidate) as b:
        for name, variable in a.variables.items():
            if name not in b.variables or not np.issubdtype(variable.dtype, np.floating):
                continue
            x = np.asarray(variable[:], dtype=np.float64)
            y = np.asarray(b.variables[name][:], dtype=np.float64)
            if x.shape != y.shape or x.size == 0:
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                continue
            difference = y[finite] - x[finite]
            spread = np.std(x[finite])
            scale = spread if spread > 0 else np.max(np.abs(x[finite]))
            fields.append({
                "name": name,
                "rms_difference": float(np.sqrt(np.mean(difference ** 2))),
                "relative_rms": float(np.sqrt(np.mean(difference ** 2)) / scale) if scale else 0.0,
                "max_difference": float(np.max(np.abs(difference))),
                "reference_spread": float(spread),
                "identical": bool(np.array_equal(x, y)),
                "candidate_non_finite": int(np.sum(~np.isfinite(y))),
            })
    return {"file": reference.name, "fields": fields}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=15)
    arguments = parser.parse_args()

    reports = []
    for path in history_files(arguments.reference):
        other = arguments.candidate / path.name
        if other.is_file():
            reports.append(compare(path, other))
    every = [field for report in reports for field in report["fields"]]
    if not every:
        raise SystemExit("no comparable history fields; is the candidate run's output there?")

    identical = sum(1 for field in every if field["identical"])
    non_finite = [field["name"] for field in every if field["candidate_non_finite"]]
    ranked = sorted(every, key=lambda field: -field["relative_rms"])
    summary = {
        "schema_version": 1,
        "files": [report["file"] for report in reports],
        "fields_compared": len(every),
        "fields_identical": identical,
        "fields_with_non_finite_values": non_finite,
        "relative_rms": {
            "median": float(np.median([f["relative_rms"] for f in every])),
            "p90": float(np.percentile([f["relative_rms"] for f in every], 90)),
            "max": float(max(f["relative_rms"] for f in every)),
        },
        "worst": ranked[:arguments.top],
        "unchanged": [f["name"] for f in every if f["identical"]][:arguments.top],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{len(every)} fields compared, {identical} bit-for-bit identical")
    print(f"relative RMS difference: median {summary['relative_rms']['median']:.3e}  "
          f"p90 {summary['relative_rms']['p90']:.3e}  max {summary['relative_rms']['max']:.3e}")
    if non_finite:
        print(f"NON-FINITE values in {len(non_finite)} fields: {non_finite[:6]}")
    print("worst:")
    for field in ranked[:arguments.top]:
        print(f"  {field['name']:14s} relative {field['relative_rms']:9.3e}  "
              f"rms {field['rms_difference']:.3e}  (spread {field['reference_spread']:.3e})")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
