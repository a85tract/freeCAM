#!/usr/bin/env python
"""Merge one observation run into the versioned runtime-coverage record.

Execution evidence and instrumentation coverage are two independent axes, and
the record keeps them apart for every candidate:

- ``observed``                 counted calls > 0 on an instrumented path; where
                               coverage is partial the total is a lower bound.
- ``not-observed-in-this-run`` fully covered, the run completed on every rank,
                               and the count is zero.  Fifty steps of silence
                               say nothing about a month, and a month nothing
                               about reachability in general.
- ``unknown``                  no counting entry, excluded from this image's
                               scope, or partially covered with zero counts --
                               absence of a count is never evidence of absence.

A kernel observed in one process is never marked observed in another: the
per-context counts carry the attribution.  A candidate whose owning actions
are all inert in this configuration is annotated as a stub or reduced
implementation when it is nevertheless called.

Inputs: the raw observation the CLI wrote (kernel_observation.json), the
observability inventory, the run's summary and bit-for-bit records, and the
decoupling ledger (for action activity).  Output:
``validation/pi_cam_kernel_runtime_coverage_<label>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
VALIDATION = REPO / "validation"
OBSERVABILITY = VALIDATION / "pi_cam_kernel_observability.json"
LEDGER = VALIDATION / "physics_kernel_decoupling.json"
SCHEMA_VERSION = 1


def _read(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def kernel_rows(observation: Mapping[str, Any], observability: Mapping[str, Any],
                instrumented_indexes: set[int], inert_actions: set[str],
                run_complete: bool) -> tuple[list[dict], dict[str, dict[str, dict]]]:
    """Every candidate's status row, plus the per-process call map."""

    observed_by_index = {int(row["index"]): row for row in observation["kernels"]}
    rows: list[dict[str, Any]] = []
    processes: dict[str, dict[str, dict]] = {}
    for kernel in observability["kernels"]:
        index = int(kernel["index"])
        counted = observed_by_index.get(index)
        instrumented = index in instrumented_indexes
        row: dict[str, Any] = {
            "qualified": kernel["qualified"],
            "index": index,
            "entry": kernel["entry"],
            "coverage": kernel["coverage"] if instrumented else
            ("not-instrumented-in-this-image" if kernel["entry"] == "trampoline" else kernel["coverage"]),
            "blind_spots": kernel["blind_spots"],
            "processes": kernel["processes"],
        }
        if counted is not None and instrumented:
            calls = int(counted["calls_total"])
            row.update({
                "calls_total": calls,
                "calls_by_context": counted["calls_by_context"],
                "first_step": counted["first_step"],
                "last_step": counted["last_step"],
                "ranks_with_calls": counted["ranks_with_calls"],
                "rank_calls_min": counted["rank_calls_min"],
                "rank_calls_max": counted["rank_calls_max"],
            })
            if calls > 0:
                row["status"] = "observed"
                if kernel["coverage"] == "partial":
                    row["count_meaning"] = ("lower bound: this kernel has uncounted call paths "
                                            "(see blind_spots)")
                if kernel["processes"] and inert_actions.issuperset(kernel["processes"]):
                    row["note"] = "stub-or-reduced-implementation: every owning action is inert here"
                for context, count in counted["calls_by_context"].items():
                    if context.startswith(("initialization", "finalize", "run-unattributed", "slot-")):
                        continue
                    processes.setdefault(context, {})[kernel["qualified"]] = {
                        "calls": int(count),
                        "first_step": counted["first_step"],
                        "last_step": counted["last_step"],
                    }
            elif kernel["coverage"] == "full" and run_complete:
                row["status"] = "not-observed-in-this-run"
            else:
                row["status"] = "unknown"
                row["status_reason"] = ("zero counted calls with uncounted paths remaining"
                                        if kernel["coverage"] == "partial"
                                        else "the run did not complete cleanly; zero is not evidence")
        else:
            row["status"] = "unknown"
            row["status_reason"] = ("no counting entry exists for this kernel"
                                    if kernel["entry"] != "trampoline"
                                    else "not instrumented in this image's scope")
        rows.append(row)
    return rows, processes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--observation", type=Path, required=True,
                        help="the raw kernel_observation.json the run wrote")
    parser.add_argument("--summary", required=True,
                        help="the run's summary record name under validation/")
    parser.add_argument("--bfb", required=True,
                        help="the run's bit-for-bit comparison record name under validation/")
    parser.add_argument("--label", required=True, help="e.g. 50step or 1month")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    observation = _read(args.observation)
    observability = _read(OBSERVABILITY)
    ledger = _read(LEDGER)
    summary = _read(VALIDATION / args.summary)
    bfb = _read(VALIDATION / args.bfb)
    if observability.get("schema_version") != 1:
        raise SystemExit("unsupported observability schema")
    if observation.get("native_manifest_kernel_counts", {}).get("observability_sha256") \
            != observability["content_hash"]:
        raise SystemExit("the observation was made against a different observability inventory; "
                         "refusing to merge mismatched provenance")

    instrumented_indexes = {int(r["index"]) for r in observation["kernels"]}
    inert_actions = {a["id"] for a in ledger["actions"] if a.get("activity") == "inert"}
    run_complete = summary.get("run_status") == "passed" and bool(bfb.get("bfb"))
    rows, processes = kernel_rows(observation, observability, instrumented_indexes,
                                  inert_actions, run_complete)

    statuses = {status: sum(1 for r in rows if r["status"] == status)
                for status in ("observed", "not-observed-in-this-run", "unknown")}
    record = {
        "schema_version": SCHEMA_VERSION,
        "generator": "tools/record_kernel_runtime_coverage.py",
        "what": ("Which candidate kernels actually executed in this run, counted in place by the "
                 "counting image's trampolines and attributed to workflow processes, steps and "
                 "lifecycle phases; instrumentation gaps stay explicit and zero counts on "
                 "partially covered or uninstrumented paths are unknown, never absence."),
        "run": {
            "label": args.label,
            "summary_record": args.summary,
            "bfb_record": args.bfb,
            "bfb": bool(bfb.get("bfb")),
            "run_status": summary.get("run_status"),
            "pbs_job": (summary.get("pbs_job_id") or "").split(".", 1)[0] or None,
            "steps": observation.get("steps"),
            "final_model_step": observation.get("final_model_step"),
            "final_date": summary.get("final_date"),
            "mpi_ranks": observation.get("mpi_ranks"),
            "threads_per_rank": observation.get("threads_per_rank"),
            "advance_seconds": (summary.get("timing") or {}).get("advance_seconds"),
            "maximum_rank_hwm_bytes": max((s.get("maximum_rank_hwm_bytes", 0)
                                           for s in (summary.get("memory") or {}).get("samples", [])),
                                          default=None),
        },
        "image": {
            "native_library_sha256": observation.get("native_library_sha256"),
            "kernel_counts": observation.get("native_manifest_kernel_counts"),
            "observability_content_hash": observability["content_hash"],
        },
        "slots": observation.get("slot_names"),
        "kernels": rows,
        "process_calls": {pid: dict(sorted(kernels.items()))
                          for pid, kernels in sorted(processes.items())},
        "summary": {
            "candidates": len(rows),
            "instrumented": len(instrumented_indexes),
            **statuses,
            "all_ranks_reported": True,   # the reduction is collective; a missing rank aborts the run
        },
    }
    output = args.output or (VALIDATION / f"pi_cam_kernel_runtime_coverage_{args.label}.json")
    output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {output}: {json.dumps(record['summary'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
