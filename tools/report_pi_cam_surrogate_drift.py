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


def history_files(directory: Path, pattern: str = "*.cam.h*.nc") -> list[Path]:
    return sorted(p for p in directory.glob(pattern))


def is_restart(path: Path) -> bool:
    """A CAM restart holds the prognostic state at the end of the run.

    It has no time axis to slice by, and every field in it post-dates the
    start by construction -- so it is the file to read when a run wrote no
    history record after its initial one, as a 50-step run with monthly
    history does.
    """

    return ".cam.r." in path.name


def elapsed_records(dataset) -> np.ndarray:
    """Which time records post-date the start of the run.

    CAM writes a record for the initial state, before a single physics step
    has run.  Both the oracle and a surrogate hold the same initial condition,
    so that record is identical by construction and says nothing at all about
    the surrogate -- which is exactly how it reads as a passing comparison if
    it is not excluded.
    """

    for name in ("nsteph", "time"):
        if name in dataset.variables:
            values = np.asarray(dataset.variables[name][:]).reshape(-1)
            return np.flatnonzero(values > 0)
    return np.arange(len(dataset.dimensions.get("time", [])) or 1)


def compare(reference: Path, candidate: Path, *,
            whole_file_after_start: bool = False) -> dict[str, object]:
    import netCDF4

    fields: list[dict[str, object]] = []
    with netCDF4.Dataset(reference) as a, netCDF4.Dataset(candidate) as b:
        records = np.arange(0) if whole_file_after_start else elapsed_records(a)
        for name, variable in a.variables.items():
            if name not in b.variables or not np.issubdtype(variable.dtype, np.floating):
                continue
            x = np.asarray(variable[:], dtype=np.float64)
            y = np.asarray(b.variables[name][:], dtype=np.float64)
            if x.shape != y.shape or x.size == 0:
                continue
            if "time" in variable.dimensions and not whole_file_after_start:
                if records.size == 0:
                    continue
                x, y = x[records], y[records]
            finite = np.isfinite(x) & np.isfinite(y)
            if not finite.any():
                continue
            difference = y[finite] - x[finite]
            spread = np.std(x[finite])
            scale = spread if spread > 0 else np.max(np.abs(x[finite]))
            # A field with no time axis, or one the reference holds constant,
            # is a coordinate or a grid descriptor.  Comparing those and
            # reporting them identical says nothing about the run: a surrogate
            # that aborted at step three still writes the right latitudes.
            fields.append({
                "name": name,
                "after_start": whole_file_after_start
                or ("time" in variable.dimensions and records.size > 0),
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
    parser.add_argument("--pattern", default="*.cam.h*.nc",
                        help="which CAM files to compare; '*.cam.r.*.nc' reads the "
                             "restart, i.e. the state at the end of the run")
    arguments = parser.parse_args()

    reports = []
    for path in history_files(arguments.reference, arguments.pattern):
        other = arguments.candidate / path.name
        if other.is_file():
            reports.append(compare(path, other, whole_file_after_start=is_restart(path)))
    every = [field for report in reports for field in report["fields"]]
    if not every:
        raise SystemExit("no comparable history fields; is the candidate run's output there?")

    identical = sum(1 for field in every if field["identical"])
    non_finite = [field["name"] for field in every if field["candidate_non_finite"]]
    informative = [field for field in every
                   if field["after_start"] and field["reference_spread"] > 0]
    ranked = sorted(informative or every, key=lambda field: -field["relative_rms"])
    summary = {
        "schema_version": 2,
        "files": [report["file"] for report in reports],
        "fields_compared": len(every),
        "fields_informative": len(informative),
        "informative": bool(informative),
        "fields_identical": identical,
        "fields_with_non_finite_values": non_finite,
        "relative_rms": {
            "median": float(np.median([f["relative_rms"] for f in (informative or every)])),
            "p90": float(np.percentile([f["relative_rms"] for f in (informative or every)], 90)),
            "max": float(max(f["relative_rms"] for f in (informative or every))),
        },
        "worst": ranked[:arguments.top],
        "unchanged": [f["name"] for f in every if f["identical"]][:arguments.top],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{len(every)} fields compared, {len(informative)} of them written after the run "
          f"started, {identical} bit-for-bit identical")
    if not informative:
        # The loud case.  Every history record is the initial state, which both
        # runs hold identically before a step is taken; reporting that as
        # "identical" would read as a surrogate that reproduced the model.
        print("NOT A DRIFT MEASUREMENT: every history record is the run's initial state")
        print("(nsteph = 0), which both runs share by construction.  The candidate wrote no")
        print("record after a physics step; read the run log's first export difference and")
        print("the stage trace instead.")
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
