#!/usr/bin/env python3
"""Append one paired A/C measurement to the faster-than-Fortran record.

A is the original Fortran model, timed by its GPTL counters (``CPL:RUN_LOOP``
for the coupling loop, ``CPL:INIT`` and ``CPL:FINAL`` around it); C is
freeCAM in the same online configuration, timed by its own summary
(``advance_seconds`` over the same loop, ``initialize``/``finalize`` around
it).  Both ran in one allocation, in the order given.  The record keeps every
pair -- never the fastest -- and recomputes the paired ratios' median and a
bootstrap 95% interval each time one is added.

    tools/report_pi_cam_pair.py --a-timing A/run/timing/cesm_timing.000 \\
        --a-executable cesm.exe --c-summary C/summary.json --c-bfb C/bfb.json \\
        --order AC --stage-execution native-whole --pbs-job-id 1234 \\
        --record validation/pi_cam_faster_than_fortran.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_pi_cam_month import parse_gptl  # noqa: E402

TARGET_RATIO = 0.95


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gptl_seconds(gptl: dict, key: str) -> float | None:
    entry = gptl.get(key)
    return None if entry is None else float(entry["seconds"])


def pair_record(a_timing: Path, a_executable: Path | None, c_summary: Path, c_bfb: Path,
                order: str, policy: str, job: str | None, git_commit: str | None = None) -> dict:
    gptl = parse_gptl(a_timing)
    a_loop = gptl_seconds(gptl, "CPL:RUN_LOOP")
    if a_loop is None:
        raise SystemExit(f"{a_timing} has no CPL:RUN_LOOP")
    a_init, a_final = gptl_seconds(gptl, "CPL:INIT"), gptl_seconds(gptl, "CPL:FINAL")
    summary = json.loads(c_summary.read_text())
    timing = summary["timing"]
    bfb = json.loads(c_bfb.read_text()) if c_bfb.exists() else {"bfb": None}
    memory = summary.get("memory", {}).get("samples") or []
    c_memory = memory[-1] if memory else {}
    return {
        "pbs_job_id": job,
        "order": order,
        "stage_execution": policy,
        "steps": int(summary.get("steps", 0)),
        "a": {
            "coupling_loop_seconds": a_loop,
            "init_seconds": a_init,
            "final_seconds": a_final,
            "lifecycle_seconds": (None if a_init is None or a_final is None else a_init + a_loop + a_final),
            "executable_sha256": None if a_executable is None else sha256(a_executable),
            "timing_source": "GPTL CPL:RUN_LOOP in timing/cesm_timing.000",
        },
        "c": {
            "coupling_loop_seconds": float(timing["advance_seconds"]),
            "init_seconds": float(timing.get("initialize_seconds", 0.0)),
            "final_seconds": float(timing.get("finalize_seconds", 0.0)),
            "lifecycle_seconds": float(timing.get("total_seconds", 0.0)),
            "sypd": timing.get("advance_sypd"),
            "native_library_sha256": summary.get("native_library_sha256"),
            "git_commit": git_commit or summary.get("git_commit"),
            "collective_reductions": (summary.get("boundary_rank_zero") or {}).get(
                "collective_reductions"),
            "bfb": bfb.get("bfb"),
            "bfb_files": bfb.get("compared_files"),
            "peak_rank_rss_bytes": c_memory.get("maximum_rank_rss_bytes"),
            "total_rss_bytes": c_memory.get("total_rss_bytes"),
            "stage_execution_record": summary.get("stage_execution"),
            # which stage classes the run installed (every one of them for the P6 pairs)
            "python_stages": summary.get("python_stages"),
            "radiation_python": summary.get("radiation_python"),
            "cloud_macro_micro_python": summary.get("cloud_macro_micro_python"),
        },
        "ratio_c_over_a": float(timing["advance_seconds"]) / a_loop,
    }


def summarise(pairs: list[dict]) -> dict:
    ratios = [p["ratio_c_over_a"] for p in pairs]
    if not ratios:
        return {"pairs": 0}
    median = statistics.median(ratios)
    rng = random.Random(20260904)
    if len(ratios) >= 3:
        medians = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
        low, high = medians[int(0.025 * len(medians))], medians[int(0.975 * len(medians)) - 1]
    else:
        low = high = None
    return {
        "pairs": len(ratios),
        "ratios": ratios,
        "median_ratio": median,
        "bootstrap_95_interval": None if low is None else [low, high],
        "target_median_ratio": TARGET_RATIO,
        "target_met": len(ratios) >= 5 and median <= TARGET_RATIO and high is not None and high < 1.0,
        "all_bfb": all(p["c"]["bfb"] is True for p in pairs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-timing", type=Path, required=True)
    parser.add_argument("--a-executable", type=Path)
    parser.add_argument("--c-summary", type=Path, required=True)
    parser.add_argument("--c-bfb", type=Path, required=True)
    parser.add_argument("--order", choices=("AC", "CA"), required=True)
    parser.add_argument("--stage-execution", default="native-whole")
    parser.add_argument("--pbs-job-id")
    parser.add_argument("--git-commit", help="the freeCAM commit the C run executed")
    parser.add_argument("--duration", choices=("1month", "1year"), default="1month")
    parser.add_argument("--record", type=Path, required=True)
    arguments = parser.parse_args()
    record = (json.loads(arguments.record.read_text()) if arguments.record.exists()
              else {"schema_version": 1,
                    "what": "Paired online runs of the PI-atm case: A the original Fortran model, C freeCAM with "
                            "the Python stage class installed and nothing replaced, in one allocation each. "
                            "The goal is a median C/A coupling-loop ratio of 0.95 or better over the month and "
                            "the year, bit-for-bit, on 512 ranks.",
                    "1month": {"pairs": []}, "1year": {"pairs": []}})
    pair = pair_record(arguments.a_timing, arguments.a_executable, arguments.c_summary, arguments.c_bfb,
                       arguments.order, arguments.stage_execution, arguments.pbs_job_id,
                       arguments.git_commit)
    block = record.setdefault(arguments.duration, {"pairs": []})
    block["pairs"].append(pair)
    block["summary"] = summarise(block["pairs"])
    arguments.record.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{arguments.duration} pair {arguments.order} job {arguments.pbs_job_id}: "
          f"A {pair['a']['coupling_loop_seconds']:.2f} s | C {pair['c']['coupling_loop_seconds']:.2f} s | "
          f"C/A {pair['ratio_c_over_a']:.4f} | bfb {pair['c']['bfb']}")
    print(f"{arguments.duration} so far: {json.dumps(block['summary'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
