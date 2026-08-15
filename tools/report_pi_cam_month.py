#!/usr/bin/env python3
"""Combine FreeCAM timing, CESM GPTL timing, and CAM BFB evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


_GPTL_LINE = re.compile(
    r'^\s*"(?P<name>[^"]+)"\s+-\s+(?P<calls>\d+)\s+-\s+'
    r'(?P<wall>[0-9.eE+-]+)'
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_gptl(path: Path) -> dict[str, dict[str, float | int]]:
    """Return call count and rank-zero wall time for every GPTL timer."""

    timers: dict[str, dict[str, float | int]] = {}
    for line in path.read_text().splitlines():
        match = _GPTL_LINE.match(line)
        if match is None:
            continue
        timers[match.group("name")] = {
            "calls": int(match.group("calls")),
            "seconds": float(match.group("wall")),
        }
    return timers


def build_report(
    *,
    freecam_summary: dict[str, Any],
    bfb: dict[str, Any],
    gptl: dict[str, dict[str, float | int]],
    config: Path,
    fortran_job_id: str | None = None,
) -> dict[str, Any]:
    timing = freecam_summary["timing"]
    original_init = float(gptl["CPL:INIT"]["seconds"])
    original_loop = float(gptl["CPL:RUN_LOOP"]["seconds"])
    original_atm = float(gptl["CPL:ATM_RUN"]["seconds"])
    original_final = float(gptl["CPL:FINAL"]["seconds"])
    original_total = original_init + original_loop + original_final
    freecam_advance = float(timing["advance_seconds"])
    freecam_total = float(timing["total_seconds"])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "schema_version": 1,
        "result": "passed" if bfb.get("bfb") else "failed",
        "git_commit": commit,
        "config": str(config.resolve()),
        "config_sha256": _sha256(config),
        "steps": freecam_summary["steps"],
        "simulated_days": timing["simulated_days"],
        "mpi_ranks": freecam_summary["mpi_ranks"],
        "pbs_job_id": freecam_summary.get("pbs_job_id"),
        "bfb": bfb,
        "freecam_seconds": {
            "initialize": timing["initialize_seconds"],
            "advance": freecam_advance,
            "finalize": timing["finalize_seconds"],
            "total": freecam_total,
            "advance_sypd": timing["advance_sypd"],
        },
        "original_fortran_seconds": {
            "pbs_job_id": fortran_job_id,
            "initialize_all_components": original_init,
            "coupled_run_loop": original_loop,
            "atmosphere_inside_coupled_loop": original_atm,
            "finalize_all_components": original_final,
            "lifecycle_sum": original_total,
            "timer_source": "CESM GPTL rank-zero wallclock",
        },
        "performance_ratios": {
            "freecam_advance_over_original_atmosphere": (
                freecam_advance / original_atm
            ),
            "freecam_total_over_original_cesm_lifecycle": (
                freecam_total / original_total
            ),
        },
        "comparison_scope": {
            "bfb": (
                "All numeric variables in every CAM history and restart NetCDF "
                "file are compared byte for byte."
            ),
            "performance": (
                "FreeCAM uses the recorded x2a/a2x boundary bundle for the "
                "simulated duration; "
                "the original atmosphere timer excludes other component compute "
                "but includes its live coupler exchange. The ratio is therefore "
                "an end-to-end CAM-path comparison, not a kernel-only benchmark."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freecam-summary", type=Path, required=True)
    parser.add_argument("--bfb", type=Path, required=True)
    parser.add_argument("--fortran-timing", type=Path, required=True)
    parser.add_argument("--fortran-job-id")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        freecam_summary=json.loads(args.freecam_summary.read_text()),
        bfb=json.loads(args.bfb.read_text()),
        gptl=parse_gptl(args.fortran_timing),
        config=args.config,
        fortran_job_id=args.fortran_job_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
