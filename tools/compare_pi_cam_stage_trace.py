#!/usr/bin/env python3
"""Compare a Python-driven stage's kernel trace against the oracle's capture.

A ``NativeStage`` writes, when its ``TRACE_ENV`` names a directory, one JSON
line per swappable-kernel call per chunk per step: the live-lane hash of
every argument as it entered the kernel and of every returned value as it
left, tagged with the kernel's name.  The capture bundle
(``tools/convert_pi_cam_function_capture.py``) holds the same arguments the
oracle passed, for every chunk of every rank over the captured steps.  This
joins the two on ``(mpi_rank, lchnk, nstep)`` and reports, for each record,
the first argument whose bytes differ -- before the call (the
transliteration fed the kernel something else) or after it (the kernel
answered differently, or the answer was copied out wrongly).

``--kernel`` selects one kernel out of a trace that holds several, which is
what a stage with more than one swappable kernel writes; without it every
record is compared and a bundle that only covers one kernel reports the
others as unmatched.

    tools/compare_pi_cam_stage_trace.py \\
        --bundle /.../bundles/mmacro_pcond_capture.npz \\
        --trace  <run_dir>/macro_trace --prefix macro \\
        --output validation/pi_cam_macro_tend_trace_vs_capture.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics.capture import lane_sha256  # noqa: E402


def load_trace(directory: Path, prefix: str = "macro",
               kernel: str | None = None) -> dict[tuple[int, int, int], dict]:
    records: dict[tuple[int, int, int], dict] = {}
    for path in sorted(directory.glob(f"{prefix}_trace.rank-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if kernel is not None and record.get("kernel", kernel) != kernel:
                continue
            records[(int(record["mpi_rank"]), int(record["lchnk"]), int(record["nstep"]))] = record
    return records


def compare(bundle_path: Path, trace_dir: Path, prefix: str = "macro",
            kernel: str | None = None) -> dict:
    bundle = np.load(bundle_path)
    keys = {(int(r), int(c), int(s)): i for i, (r, c, s) in enumerate(
        zip(bundle["mpi_rank"], bundle["lchnk"], bundle["nstep"]))}
    ncols = bundle["ncol"]
    before_names = sorted(n.removeprefix("before__") for n in bundle.files if n.startswith("before__"))
    after_names = sorted(n.removeprefix("after__") for n in bundle.files if n.startswith("after__"))
    trace = load_trace(trace_dir, prefix, kernel)

    matched = 0
    unmatched: list[list[int]] = []
    per_record: list[dict] = []
    first_before: Counter = Counter()
    first_after: Counter = Counter()
    any_before: Counter = Counter()
    any_after: Counter = Counter()

    for key, record in sorted(trace.items()):
        index = keys.get(key)
        if index is None:
            unmatched.append(list(key))
            continue
        matched += 1
        ncol = int(ncols[index])
        differing_before = [
            name for name in before_names
            if name in record["before"]
            and lane_sha256(bundle[f"before__{name}"][..., index], ncol) != record["before"][name]
        ]
        differing_after = [
            name for name in after_names
            if name in record["after"]
            and lane_sha256(bundle[f"after__{name}"][..., index], ncol) != record["after"][name]
        ]
        any_before.update(differing_before)
        any_after.update(differing_after)
        if differing_before:
            first_before[differing_before[0]] += 1
        if differing_after:
            first_after[differing_after[0]] += 1
        if differing_before or differing_after:
            per_record.append({
                "mpi_rank": key[0], "lchnk": key[1], "nstep": key[2], "ncol": ncol,
                "before": differing_before, "after": differing_after,
            })

    return {
        "schema_version": 1,
        "bundle": str(bundle_path),
        "trace": str(trace_dir),
        "prefix": prefix,
        "kernel": kernel,
        "trace_records": len(trace),
        "matched_records": matched,
        "unmatched_records": unmatched[:16],
        "records_with_differences": len(per_record),
        "identical": matched > 0 and not per_record,
        "arguments_differing_before_call": dict(any_before.most_common()),
        "arguments_differing_after_call": dict(any_after.most_common()),
        "first_differing_before_call": dict(first_before.most_common()),
        "first_differing_after_call": dict(first_after.most_common()),
        "differing_records": per_record[:64],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefix", default="macro",
                        help="the stage's Fortran prefix, which names its trace files")
    parser.add_argument("--kernel", default=None,
                        help="compare only this kernel's records (default: all of them)")
    arguments = parser.parse_args()
    report = compare(arguments.bundle, arguments.trace, arguments.prefix, arguments.kernel)
    text = json.dumps(report, indent=2)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text + "\n")
    print(f"trace records {report['trace_records']}, matched {report['matched_records']}, "
          f"with differences {report['records_with_differences']}")
    if report["arguments_differing_before_call"]:
        print("first differing BEFORE the call:", report["first_differing_before_call"])
    if report["arguments_differing_after_call"]:
        print("first differing AFTER the call:", report["first_differing_after_call"])
    if report["identical"]:
        print("every matched record is identical to the capture, before and after")
    return 0 if report["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
