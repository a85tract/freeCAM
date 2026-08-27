#!/usr/bin/env python3
"""Sum a Python-driven stage's per-rank timing profiles into one table.

A ``NativeStage`` with ``PROFILE_ENV`` set writes, per rank, the seconds and
call count of every handle entry, kernel run, kernel copy, trace hash and
``tend`` under its own key.  This reads every rank's file and prints, per
key, the mean seconds per rank, the slowest rank, and the share of the
stage's ``tend`` total -- so a question like "where do 650 ms a step go" is
answered by name rather than by guess.

    tools/summarize_stage_profile.py <run_dir>/rad_profile --prefix rad [--top 30]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(directory: Path, prefix: str) -> dict:
    seconds: dict[str, list[float]] = defaultdict(list)
    calls: dict[str, list[int]] = defaultdict(list)
    files = sorted(directory.glob(f"{prefix}_profile.rank-*.json"))
    for path in files:
        report = json.loads(path.read_text())
        for key, value in report["seconds"].items():
            seconds[key].append(float(value))
            calls[key].append(int(report["calls"].get(key, 0)))
    ranks = len(files)
    rows = []
    for key, values in seconds.items():
        rows.append({
            "key": key,
            "mean_seconds": sum(values) / len(values),
            "max_seconds": max(values),
            "mean_calls": sum(calls[key]) / len(calls[key]),
            "ranks_reporting": len(values),
        })
    rows.sort(key=lambda r: -r["mean_seconds"])
    tend = next((r["mean_seconds"] for r in rows if r["key"] == "tend"), None)
    for row in rows:
        row["share_of_tend"] = (row["mean_seconds"] / tend) if tend else None
    # group totals, which is usually the answer
    groups: dict[str, float] = defaultdict(float)
    for row in rows:
        family = row["key"].split(":")[0]
        if family != "tend":
            groups[family] += row["mean_seconds"]
    return {"ranks": ranks, "tend_mean_seconds": tend, "rows": rows,
            "by_family": dict(sorted(groups.items(), key=lambda kv: -kv[1]))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", default="rad")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = summarize(arguments.directory, arguments.prefix)
    if arguments.output:
        arguments.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"{report['ranks']} ranks; tend mean {report['tend_mean_seconds']:.3f} s per rank")
    print("\nby family (mean seconds per rank):")
    for family, total in report["by_family"].items():
        share = total / report["tend_mean_seconds"] if report["tend_mean_seconds"] else 0
        print(f"  {family:18s} {total:9.3f} s  {100 * share:5.1f}%")
    print(f"\ntop {arguments.top} keys:")
    print(f"  {'key':40s} {'mean s':>9s} {'max s':>9s} {'calls':>8s} {'share':>6s}")
    for row in report["rows"][:arguments.top]:
        share = row["share_of_tend"]
        print(f"  {row['key']:40s} {row['mean_seconds']:9.3f} {row['max_seconds']:9.3f} "
              f"{row['mean_calls']:8.0f} {100 * share if share else 0:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
