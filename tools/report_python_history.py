#!/usr/bin/env python3
"""Record the Python-owned CAM-format history files a run produced."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from netCDF4 import Dataset


def describe(path: Path) -> dict[str, object]:
    with Dataset(path) as dataset:
        column_variables = sorted(
            name
            for name, variable in dataset.variables.items()
            if "ncol" in variable.dimensions
        )
        return {
            "file": path.name,
            "bytes": path.stat().st_size,
            "ncol": len(dataset.dimensions["ncol"]),
            "samples": len(dataset.dimensions["time"]),
            "time": [float(item) for item in dataset.variables["time"][:]],
            "time_bnds": [
                [float(low), float(high)]
                for low, high in dataset.variables["time_bnds"][:]
            ],
            "date": [int(item) for item in dataset.variables["date"][:]],
            "datesec": [int(item) for item in dataset.variables["datesec"][:]],
            "column_variables": column_variables,
            "attributes": {
                name: str(dataset.getncattr(name))
                for name in ("Conventions", "source", "case", "ne", "np")
                if name in dataset.ncattrs()
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stream", default="h9")
    args = parser.parse_args()
    files = sorted(args.run_dir.glob(f"*.cam.{args.stream}.*.nc"))
    payload = {
        "schema_version": 1,
        "run_dir": str(args.run_dir),
        "stream": args.stream,
        "file_count": len(files),
        "files": [describe(path) for path in files],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(files)} Python-owned history files -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
