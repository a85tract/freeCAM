#!/usr/bin/env python3
"""Compare CAM output against a reference case that uses a different name.

``verify_pi_cam.py`` matches files by name, which only works when both runs
share a case name.  This comparator matches by the CAM output suffix instead
(``.cam.h0.0001-01.nc``), so a run can be checked against an independent
reference case such as a long production integration.  Character variables
that record when a file was written are metadata about the job, not results,
and are reported separately rather than compared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from netCDF4 import Dataset

SUFFIX = re.compile(r"\.cam\.(?P<stream>[a-z0-9]+)\.(?P<stamp>[0-9-]+)\.nc$")
WRITE_STAMPS = ("date_written", "time_written")


def index(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(root.glob("*.cam.*.nc")):
        match = SUFFIX.search(path.name)
        if match is None:
            continue
        found[f"{match.group('stream')}.{match.group('stamp')}"] = path
    return found


def compare(candidate: Path, reference: Path) -> dict[str, object]:
    identical: list[str] = []
    differing: list[dict[str, object]] = []
    skipped: list[str] = []
    with Dataset(candidate) as a, Dataset(reference) as b:
        dimensions_match = {k: len(v) for k, v in a.dimensions.items()} == {
            k: len(v) for k, v in b.dimensions.items()
        }
        only_candidate = sorted(set(a.variables) - set(b.variables))
        only_reference = sorted(set(b.variables) - set(a.variables))
        for name in sorted(set(a.variables) & set(b.variables)):
            first, second = a.variables[name], b.variables[name]
            if name in WRITE_STAMPS or first.dtype.kind not in "fiu":
                skipped.append(name)
                continue
            if first.shape != second.shape:
                differing.append({"variable": name, "reason": "shape"})
                continue
            left = np.asarray(first[:])
            right = np.asarray(second[:])
            if np.array_equal(left, right):
                identical.append(name)
            else:
                delta = np.abs(left.astype("f8") - right.astype("f8"))
                differing.append(
                    {
                        "variable": name,
                        "maximum_absolute_difference": float(delta.max()),
                        "differing_values": int(np.count_nonzero(delta)),
                    }
                )
    return {
        "candidate": candidate.name,
        "reference": reference.name,
        "dimensions_match": dimensions_match,
        "identical": len(identical),
        "differing": differing,
        "skipped": skipped,
        "only_in_candidate": only_candidate,
        "only_in_reference": only_reference,
        "bfb": (
            dimensions_match
            and not differing
            and not only_candidate
            and not only_reference
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stream",
        default="h0",
        help="compare only this CAM stream; 'all' compares every stream",
    )
    args = parser.parse_args()

    candidates = index(args.candidate)
    references = index(args.reference)
    if args.stream != "all":
        prefix = f"{args.stream}."
        candidates = {k: v for k, v in candidates.items() if k.startswith(prefix)}
        references = {k: v for k, v in references.items() if k.startswith(prefix)}
    shared = sorted(set(candidates) & set(references))
    files = [compare(candidates[key], references[key]) for key in shared]
    payload = {
        "schema_version": 1,
        "candidate_dir": str(args.candidate),
        "reference_dir": str(args.reference),
        "stream": args.stream,
        "candidate_files": len(candidates),
        "reference_files": len(references),
        "compared_files": len(files),
        "unmatched_candidates": sorted(set(candidates) - set(references)),
        "bfb": bool(files) and all(item["bfb"] for item in files),
        "files": files,
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(
            f"{payload['compared_files']} files compared, "
            f"bfb={payload['bfb']} -> {args.output}"
        )
    return 0 if payload["bfb"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
