"""Which variables differ, file by file, between two PI-CAM run directories.

    compare_pi_cam_variables.py --reference DIR --candidate DIR [--output record.json]

The bit-for-bit gate (tools/verify_pi_cam.py) stops at the first difference.  When a
replacement leaves the model state exact but not every history accumulator -- a process
slot answering radiation_tend's computing branch produces none of the branch's own
diagnostics -- the question is which files and which variables differ.  Timestamps
(date_written, time_written) are reported apart, never counted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np

TIMESTAMPS = {"date_written", "time_written"}


def differing_variables(candidate: Path, reference: Path) -> tuple[list[str], list[str], list[str]]:
    import netCDF4 as nc

    with nc.Dataset(candidate) as a, nc.Dataset(reference) as b:
        differing, timestamps, missing = [], [], []
        for name in a.variables:
            if name not in b.variables:
                missing.append(name)
                continue
            x, y = np.asarray(a.variables[name][...]), np.asarray(b.variables[name][...])
            same = x.shape == y.shape and (np.array_equal(x, y) or (x.dtype.kind == "f" and np.array_equal(x, y, equal_nan=True)))
            if not same:
                (timestamps if name in TIMESTAMPS else differing).append(name)
        missing.extend(name for name in b.variables if name not in a.variables)
    return differing, timestamps, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    files: dict[str, dict] = {}
    for path in sorted(glob(str(args.candidate / "*.nc"))):
        name = os.path.basename(path)
        reference = args.reference / name
        if not reference.exists():
            files[name] = {"reference": "absent"}
            continue
        differing, timestamps, missing = differing_variables(Path(path), reference)
        files[name] = {"identical": not differing and not missing, "differing_variables": differing,
                       "timestamp_variables": timestamps, "missing_variables": missing}
        verdict = "identical" if files[name]["identical"] else f"{len(differing)} differing"
        print(f"{name}: {verdict}" + (f" ({', '.join(differing[:12])}{', ...' if len(differing) > 12 else ''})" if differing else ""))
    record = {"schema_version": 1, "reference": str(args.reference), "candidate": str(args.candidate),
              "state_files_identical": all(v.get("identical") for n, v in files.items() if ".cam.r." in n or ".cam.rs." in n),
              "files": files}
    if args.output:
        args.output.write_text(json.dumps(record, indent=1) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
