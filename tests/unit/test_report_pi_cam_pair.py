"""The paired A/C record: one pair appended, ratios summarised, nothing dropped."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

GPTL = """  "CPL:INIT"                                -       1    -      10.000000    10.000000    10.000000         0.000000
    "CPL:RUN_LOOP"                          -    1488    -     400.000000     5.829652     0.241992         0.000136
      "CPL:ATM_RUN"                         -    1488    -     300.000000     3.273668     0.211146         0.000136
    "CPL:FINAL"                             -       1    -       1.000000     0.002280     0.002280         0.000000
"""


def _pair(tmp_path, seconds, order):
    import report_pi_cam_pair as rp

    timing = tmp_path / f"cesm_timing_{order}.000"
    timing.write_text(GPTL)
    summary = tmp_path / f"summary_{order}.json"
    summary.write_text(json.dumps({"steps": 1488, "timing": {"advance_seconds": seconds, "initialize_seconds": 12.0,
                                                             "finalize_seconds": 0.5, "total_seconds": seconds + 12.5,
                                                             "advance_sypd": 15.0},
                                   "native_library_sha256": "abc", "git_commit": "def",
                                   "memory": {"samples": [{"maximum_rank_rss_bytes": 1, "total_rss_bytes": 2}]},
                                   "stage_execution": {"x": {"execution_mode": "native-whole"}}}))
    bfb = tmp_path / f"bfb_{order}.json"
    bfb.write_text(json.dumps({"bfb": True, "compared_files": 18}))
    record = tmp_path / "record.json"
    return rp.pair_record(timing, None, summary, bfb, order, "native-whole", "1"), record


def test_a_pair_is_recorded_with_both_loops_and_their_ratio(tmp_path) -> None:
    import report_pi_cam_pair as rp

    pair, _ = _pair(tmp_path, 380.0, "AC")
    assert pair["a"]["coupling_loop_seconds"] == 400.0 and pair["a"]["lifecycle_seconds"] == 411.0
    assert pair["c"]["coupling_loop_seconds"] == 380.0 and pair["c"]["bfb"] is True
    assert pair["ratio_c_over_a"] == 0.95
    assert pair["c"]["stage_execution_record"]["x"]["execution_mode"] == "native-whole"


def test_the_summary_is_the_median_of_every_pair_and_the_target_needs_five(tmp_path) -> None:
    import report_pi_cam_pair as rp

    pairs = [_pair(tmp_path, s, "AC")[0] for s in (376.0, 380.0, 372.0)]
    summary = rp.summarise(pairs)
    assert summary["pairs"] == 3 and summary["median_ratio"] == 0.94
    assert summary["target_met"] is False                       # fewer than five pairs
    more = pairs + [_pair(tmp_path, s, "CA")[0] for s in (378.0, 374.0)]
    summary = rp.summarise(more)
    assert summary["pairs"] == 5 and summary["median_ratio"] == 0.94
    assert summary["bootstrap_95_interval"][1] < 1.0 and summary["target_met"] is True
    slow = more + [_pair(tmp_path, 440.0, "AC")[0]]
    assert rp.summarise(slow)["ratios"][-1] == 1.1              # nothing is dropped
