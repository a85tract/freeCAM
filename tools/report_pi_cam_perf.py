"""Summarise a perf-instrumented online 50-step run.

Reads the ``stat.<rank>.csv`` counters every rank wrote and the
``record.<rank>.data`` samples of the recorded ranks (see
``tools/perf_rank_wrapper.sh``), and writes one JSON record: page faults
and task clock across ranks, and for each recorded rank the share of user
time by library and the heaviest symbols.  The shares are what the Fortran
memory work is budgeted against -- how much of a step is allocation, copying
and Python control rather than the numerical kernels.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path


_PERCENT_LINE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)%\s+(.+?)\s*$")


def parse_stat_csv(text: str) -> dict[str, float]:
    """Return ``perf stat -x,`` counters by event name (the ``:u`` suffix dropped)."""

    counters: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) < 3 or fields[0] in ("<not counted>", "<not supported>"):
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        name = fields[2].split(":")[0]
        counters[name] = value
    return counters


def parse_percent_lines(text: str) -> list[tuple[float, str]]:
    """Return ``(percent, label)`` pairs from a ``perf report --stdio`` listing."""

    rows: list[tuple[float, str]] = []
    for line in text.splitlines():
        match = _PERCENT_LINE.match(line)
        if match:
            rows.append((float(match.group(1)), match.group(2)))
    return rows


def classify_dso(name: str) -> str:
    """Bucket a shared object into what it stands for in the step."""

    lowered = name.lower()
    if "freecam_pi_cam" in lowered or "cesm.exe" in lowered:
        return "cam"
    if "pycesm" in lowered:
        return "coupler+components"
    if "numpy" in lowered or "_multiarray" in lowered or "umath" in lowered:
        return "numpy"
    if "libpython" in lowered or "python" in lowered:
        return "python"
    if "libmpi" in lowered or "libfabric" in lowered or "libpmi" in lowered or "libcxi" in lowered:
        return "mpi"
    if "libc.so" in lowered or "libc-" in lowered or "ld-linux" in lowered:
        return "libc"
    if "libimf" in lowered or "libsvml" in lowered or "libintlc" in lowered or "libirc" in lowered:
        return "intel-math"
    if "netcdf" in lowered or "hdf5" in lowered or "pnetcdf" in lowered or "pio" in lowered:
        return "io-libraries"
    if "[kernel" in lowered or "[unknown]" in lowered:
        return "kernel-or-unknown"
    return "other"


def perf_report(data: Path, sort: str, limit: int) -> list[tuple[float, str]]:
    command = ["perf", "report", "-i", str(data), "--stdio", "--sort", sort,
               "--percent-limit", "0.2", "--max-stack", "0"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    rows = parse_percent_lines(result.stdout)
    return rows[:limit]


def rank_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {"min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.fmean(values)}


def build_record(perf_dir: Path, summary: dict, bfb: dict, job: str | None,
                 git_commit: str | None, *, symbols: int = 40) -> dict:
    stats: dict[int, dict[str, float]] = {}
    for path in sorted(perf_dir.glob("stat.*.csv")):
        rank = int(path.stem.split(".")[1])
        stats[rank] = parse_stat_csv(path.read_text())
    steps = int(summary.get("steps", 0)) or None
    faults = [c.get("page-faults", 0.0) for c in stats.values()]
    clocks = [c.get("task-clock", 0.0) / 1000.0 for c in stats.values()]  # msec -> s
    recorded = {}
    for path in sorted(perf_dir.glob("record.*.data")):
        rank = int(path.stem.split(".")[1])
        by_dso = perf_report(path, "dso", 30)
        buckets: dict[str, float] = {}
        for percent, dso in by_dso:
            bucket = classify_dso(dso)
            buckets[bucket] = round(buckets.get(bucket, 0.0) + percent, 2)
        recorded[str(rank)] = {
            "by_dso": [{"percent": p, "dso": d} for p, d in by_dso],
            "by_bucket": dict(sorted(buckets.items(), key=lambda item: -item[1])),
            "top_symbols": [{"percent": p, "symbol": s}
                            for p, s in perf_report(path, "dso,symbol", symbols)],
        }
    timing = summary.get("timing", {})
    return {
        "schema_version": 1,
        "what": ("perf counters on every rank and user-space samples on a few, over the exact "
                 "online 50-step run; a diagnostic of where the step's time goes, not a gate"),
        "pbs_job_id": job,
        "git_commit": git_commit,
        "steps": steps,
        "bfb": bfb.get("bfb"),
        "ranks_counted": len(stats),
        "page_faults_per_rank": rank_stats(faults),
        "page_faults_per_rank_per_step": (None if not steps else rank_stats([f / steps for f in faults])),
        "task_clock_seconds_per_rank": rank_stats(clocks),
        "advance_seconds": timing.get("advance_seconds"),
        "initialize_seconds": timing.get("initialize_seconds"),
        "recorded_ranks": recorded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perf-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bfb", type=Path, required=True)
    parser.add_argument("--pbs-job-id")
    parser.add_argument("--git-commit")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summary = json.loads(arguments.summary.read_text())
    bfb = json.loads(arguments.bfb.read_text()) if arguments.bfb.exists() else {"bfb": None}
    record = build_record(arguments.perf_dir, summary, bfb, arguments.pbs_job_id, arguments.git_commit)
    arguments.output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"ranks counted {record['ranks_counted']}; page faults per rank per step "
          f"{record['page_faults_per_rank_per_step']}; bfb {record['bfb']}")
    for rank, detail in record["recorded_ranks"].items():
        print(f"rank {rank}: {detail['by_bucket']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
