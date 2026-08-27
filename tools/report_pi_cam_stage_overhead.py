#!/usr/bin/env python3
"""What the Python cloud macro/microphysics stage costs over a model month.

Three runs of the same PI-atm month answer it, and the answer is only
worth reading because all three integrate the same steps to the same
numbers:

* the original Fortran model, timed by its own GPTL counters;
* freeCAM with tphysbc stage 7 still in Fortran -- Python owns the
  workflow, the clock and the coupling, but not the stage;
* freeCAM with :class:`CloudMacroMicrophysics` in the action's place --
  Python owns the stage's control flow too, statement for statement, and
  every floating-point number is still the oracle's.

Time is compared on the integration loop, which is the like-for-like
quantity: freeCAM's ``timing.advance_seconds`` against the Fortran
model's atmosphere time inside the coupled loop.  Memory is compared two
ways: the job-aggregate high-water mark PBS accounts for (``qhist``), and
the cross-rank total the driver samples for itself as it runs, which is
the only instrument the two freeCAM runs share.

    tools/report_pi_cam_stage_overhead.py \
        --fortran-report validation/pi_cam_1month_stage_fortran_performance.json \
        --python-report validation/pi_cam_1month_stage_python_performance.json \
        --fortran-summary validation/pi_cam_1month_stage_fortran.json \
        --python-summary validation/pi_cam_1month_stage_python.json \
        --qhist-job 7256750=fortran_model --qhist-job 7256751=freecam_stage_fortran \
        --qhist-job 7256752=freecam_stage_python \
        --output validation/pi_cam_stage_python_1month_overhead.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

GIGABYTE = 1024.0 ** 3


def qhist_memory(job: str) -> dict[str, Any]:
    """The high-water memory and elapsed time PBS accounted for one job."""

    try:
        output = subprocess.run(["qhist", "-j", job.split(".")[0]],
                                capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return {"job": job, "available": False, "reason": "qhist is not on this host"}
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0].startswith(job.split(".")[0]):
            try:
                return {"job": job, "available": True, "used_memory_gb": float(fields[-3]),
                        "cpu_hours": float(fields[-2]), "elapsed_hours": float(fields[-1])}
            except (IndexError, ValueError):
                break
    return {"job": job, "available": False, "reason": "no accounting row yet"}


def sampled_memory(summary: dict[str, Any]) -> dict[str, Any]:
    """What the driver saw of its own ranks: the cross-rank totals it sampled."""

    memory = summary.get("memory") or {}
    samples = memory.get("samples") or []
    if not samples:
        return {"available": False}
    by_label = {str(item.get("label") or f"step_{item.get('step')}"): item for item in samples}
    peak = max(samples, key=lambda item: item.get("total_pss_bytes") or 0)
    return {
        "available": True,
        "rank_count": memory.get("rank_count"),
        "sample_every_steps": memory.get("sample_every_steps"),
        "samples": len(samples),
        "initialized_total_pss_gb": (by_label.get("initialized", {}).get("total_pss_bytes", 0)
                                     / GIGABYTE) or None,
        "peak_total_pss_gb": (peak.get("total_pss_bytes") or 0) / GIGABYTE,
        "peak_total_rss_gb": (peak.get("total_rss_bytes") or 0) / GIGABYTE,
        "peak_label": str(peak.get("label") or f"step_{peak.get('step')}"),
        "peak_rank_rss_mb": (peak.get("maximum_rank_rss_bytes") or 0) / (1024.0 ** 2),
    }


def _ratio(candidate: float, reference: float) -> dict[str, float]:
    return {"seconds": candidate, "reference_seconds": reference,
            "ratio": candidate / reference,
            "overhead_percent": 100.0 * (candidate / reference - 1.0),
            "absolute_seconds": candidate - reference}


def gptl_atmosphere(path: Path) -> float:
    """``CPL:ATM_RUN`` from a GPTL timing file: the atmosphere's own time
    inside the coupled loop, which is what freeCAM's advance loop replaces."""

    for line in path.read_text().splitlines():
        if '"CPL:ATM_RUN"' in line:
            return float(line.split()[4])
    raise SystemExit(f"no CPL:ATM_RUN in {path}")


def build(fortran_report: dict[str, Any], python_report: dict[str, Any],
          fortran_summary: dict[str, Any], python_summary: dict[str, Any],
          accounting: dict[str, dict[str, Any]],
          same_day_fortran: float | None = None) -> dict[str, Any]:
    oracle = fortran_report["original_fortran_seconds"]["atmosphere_inside_coupled_loop"]
    # the same-day Fortran sample is the fair reference: the oracle month was
    # integrated on another day, on other nodes
    original = same_day_fortran if same_day_fortran is not None else oracle
    baseline = fortran_report["freecam_seconds"]["advance"]
    stage = python_report["freecam_seconds"]["advance"]
    steps = int(python_report["steps"])

    bfb = {name: report["bfb"]["bfb"] for name, report in
           (("freecam_stage_fortran", fortran_report), ("freecam_stage_python", python_report))}
    memory = {name: sampled_memory(summary) for name, summary in
              (("freecam_stage_fortran", fortran_summary),
               ("freecam_stage_python", python_summary))}

    report: dict[str, Any] = {
        "schema_version": 1,
        "what": "the cost of running tphysbc stage 7 as a Python class over a model month",
        "steps": steps,
        "simulated_days": python_report.get("simulated_days"),
        "mpi_ranks": python_report.get("mpi_ranks"),
        "bfb": bfb,
        "time": {
            "original_fortran_atmosphere_seconds": original,
            "original_fortran_reference": ("a Fortran month run the same day"
                                           if same_day_fortran is not None
                                           else "the oracle month's own timing"),
            "oracle_month_atmosphere_seconds": oracle,
            "freecam_stage_fortran_vs_original": _ratio(baseline, original),
            "freecam_stage_python_vs_original": _ratio(stage, original),
            "freecam_stage_python_vs_freecam": _ratio(stage, baseline),
            "stage_cost_ms_per_step": 1000.0 * (stage - baseline) / steps,
        },
        "memory_sampled_by_the_driver": memory,
        "memory_accounted_by_pbs": accounting,
    }
    if all(item.get("available") for item in memory.values()):
        base = memory["freecam_stage_fortran"]["peak_total_pss_gb"]
        with_stage = memory["freecam_stage_python"]["peak_total_pss_gb"]
        report["memory_sampled_by_the_driver"]["stage_over_freecam"] = {
            "peak_total_pss_gb": with_stage, "reference_gb": base,
            "overhead_percent": 100.0 * (with_stage / base - 1.0),
            "absolute_gb": with_stage - base,
        }
    rows = [item for item in accounting.values() if item.get("available")]
    if len(rows) == 3:
        by_name = {name: item for name, item in accounting.items()}
        original_gb = by_name["fortran_model"]["used_memory_gb"]
        report["memory_accounted_by_pbs"]["overhead_percent"] = {
            "freecam_stage_fortran": 100.0 * (
                by_name["freecam_stage_fortran"]["used_memory_gb"] / original_gb - 1.0),
            "freecam_stage_python": 100.0 * (
                by_name["freecam_stage_python"]["used_memory_gb"] / original_gb - 1.0),
        }
    report["result"] = "passed" if all(bfb.values()) else "failed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortran-report", type=Path, required=True)
    parser.add_argument("--python-report", type=Path, required=True)
    parser.add_argument("--fortran-summary", type=Path, required=True)
    parser.add_argument("--python-summary", type=Path, required=True)
    parser.add_argument("--qhist-job", action="append", default=[],
                        metavar="JOBID=NAME", help="a PBS job to read accounting for")
    parser.add_argument("--fortran-timing", type=Path, default=None,
                        help="GPTL file of a Fortran month run the same day")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    accounting: dict[str, dict[str, Any]] = {}
    for item in arguments.qhist_job:
        job, _, name = item.partition("=")
        accounting[name or job] = qhist_memory(job)

    report = build(
        json.loads(arguments.fortran_report.read_text()),
        json.loads(arguments.python_report.read_text()),
        json.loads(arguments.fortran_summary.read_text()),
        json.loads(arguments.python_summary.read_text()),
        accounting,
        gptl_atmosphere(arguments.fortran_timing) if arguments.fortran_timing else None,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    time = report["time"]
    print(f"steps {report['steps']}  bfb {report['bfb']}")
    print(f"  original Fortran atmosphere : {time['original_fortran_atmosphere_seconds']:9.2f} s")
    for key, label in (("freecam_stage_fortran_vs_original", "freeCAM, stage in Fortran"),
                       ("freecam_stage_python_vs_original", "freeCAM, stage in Python")):
        row = time[key]
        print(f"  {label:26s}: {row['seconds']:9.2f} s  {row['overhead_percent']:+6.2f}% "
              f"({row['absolute_seconds']:+.1f} s)")
    row = time["freecam_stage_python_vs_freecam"]
    print(f"  the stage alone            : {row['overhead_percent']:+6.2f}% over freeCAM, "
          f"{time['stage_cost_ms_per_step']:.1f} ms/step")
    print(f"wrote {arguments.output}")
    return 0 if report["result"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
