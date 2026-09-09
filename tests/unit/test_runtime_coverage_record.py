"""The runtime-coverage merge: status semantics, per-process attribution, provenance refusal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "record_runtime_coverage", REPO / "tools/record_kernel_runtime_coverage.py")
merge = importlib.util.module_from_spec(spec)
sys.modules["record_runtime_coverage"] = merge
spec.loader.exec_module(merge)


def _observability_kernels():
    return [
        {"index": 0, "qualified": "m::hot", "entry": "trampoline", "coverage": "full",
         "blind_spots": [], "processes": ["cam_run1.a", "cam_run1.b"]},
        {"index": 1, "qualified": "m::cold", "entry": "trampoline", "coverage": "full",
         "blind_spots": [], "processes": ["cam_run1.a"]},
        {"index": 2, "qualified": "m::half", "entry": "trampoline", "coverage": "partial",
         "blind_spots": [{"object": "m.o", "reason": "inlined"}], "processes": ["cam_run1.a"]},
        {"index": 3, "qualified": "m::blind", "entry": "none", "coverage": "none",
         "blind_spots": [{"reason": "internal procedure"}], "processes": ["cam_run1.a"]},
        {"index": 4, "qualified": "m::outside", "entry": "trampoline", "coverage": "full",
         "blind_spots": [], "processes": ["cam_run1.b"]},
        {"index": 5, "qualified": "carma::stub", "entry": "trampoline", "coverage": "full",
         "blind_spots": [], "processes": ["cam_run1.inert"]},
    ]


def _observation(complete=True):
    def row(index, qualified, total, contexts, first=1, last=50):
        return {"index": index, "qualified": qualified, "calls_total": total,
                "calls_by_context": contexts, "first_step": first if total else -1,
                "last_step": last if total else -1, "ranks_with_calls": 4 if total else 0,
                "rank_calls_min": 0, "rank_calls_max": total,
                "steps_by_context": {c: [first + i, last - i] for i, c in enumerate(contexts)}}
    return {
        "kernels": [
            row(0, "m::hot", 1000, {"cam_run1.a": 900, "cam_run1.b": 90, "initialization": 10}),
            row(1, "m::cold", 0, {}),
            row(2, "m::half", 0, {}),
            row(5, "carma::stub", 8, {"cam_run1.inert": 8}),
            # index 4 is instrumented in the image but outside this fixture's scope list
        ],
        "steps": 50, "final_model_step": 50, "mpi_ranks": 4, "threads_per_rank": 1,
        "slot_names": {"0": "initialization"}, "native_library_sha256": "cafe",
        "native_manifest_kernel_counts": {"scope": "all", "observability_sha256": "obs-hash",
                                          "trampolines_sha256": "tramp"},
    }


def _rows(complete=True):
    observation = _observation()
    observability = {"kernels": _observability_kernels(), "content_hash": "obs-hash"}
    instrumented = {int(r["index"]) for r in observation["kernels"]}
    return merge.kernel_rows(observation, observability, instrumented,
                             inert_actions={"cam_run1.inert"}, run_complete=complete)


def test_observed_not_observed_and_unknown_are_kept_apart() -> None:
    rows, processes = _rows()
    by_name = {r["qualified"]: r for r in rows}
    assert by_name["m::hot"]["status"] == "observed"
    assert by_name["m::cold"]["status"] == "not-observed-in-this-run"     # full coverage, clean run, zero
    assert by_name["m::half"]["status"] == "unknown"                      # zero with uncounted paths
    assert "uncounted paths" in by_name["m::half"]["status_reason"]
    assert by_name["m::blind"]["status"] == "unknown"
    assert "no counting entry" in by_name["m::blind"]["status_reason"]
    assert by_name["m::outside"]["status"] == "unknown"                   # in the image, not in this scope
    assert "not instrumented" in by_name["m::outside"]["status_reason"]
    # a stub that is genuinely called is observed, and labeled
    assert by_name["carma::stub"]["status"] == "observed"
    assert "stub-or-reduced" in by_name["carma::stub"]["note"]
    # per-process attribution: a kernel observed in A and B counts in each, initialization stays out
    assert processes["cam_run1.a"]["m::hot"]["calls"] == 900
    assert processes["cam_run1.b"]["m::hot"]["calls"] == 90
    # each process carries its own step span, never the kernel's global one
    assert processes["cam_run1.a"]["m::hot"]["first_step"] == 1
    assert processes["cam_run1.b"]["m::hot"]["first_step"] == 2
    assert "m::hot" not in processes.get("initialization", {})


def test_an_incomplete_run_never_produces_not_observed() -> None:
    rows, _ = _rows(complete=False)
    by_name = {r["qualified"]: r for r in rows}
    assert by_name["m::cold"]["status"] == "unknown"
    assert "did not complete" in by_name["m::cold"]["status_reason"]
    assert by_name["m::hot"]["status"] == "observed"                      # positive counts still stand


def test_partial_coverage_marks_observed_totals_as_lower_bounds() -> None:
    observation = _observation()
    observation["kernels"][2]["calls_total"] = 5
    observation["kernels"][2]["calls_by_context"] = {"cam_run1.a": 5}
    observability = {"kernels": _observability_kernels(), "content_hash": "obs-hash"}
    instrumented = {int(r["index"]) for r in observation["kernels"]}
    rows, _ = merge.kernel_rows(observation, observability, instrumented, set(), True)
    half = next(r for r in rows if r["qualified"] == "m::half")
    assert half["status"] == "observed" and "lower bound" in half["count_meaning"]
