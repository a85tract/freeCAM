"""The compile tool's measures and the GPU benchmark's batches and summary, without a GPU."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

torch = pytest.importorskip("torch")


def test_the_gap_is_measured_over_each_output_columns_own_range() -> None:
    from compile_torch_model import compare

    reference = torch.tensor([[1.0, 100.0, 0.0], [-2.0, 50.0, 0.0]], dtype=torch.float64)
    same = compare(reference, reference.clone())
    assert same["bit_for_bit"] and same["scaled_gap"] == 0.0 and same["values_differing"] == 0
    # 1e-3 off in a column ranging to 2, 1 off in a column ranging to 100: the larger share is 1e-2
    candidate = reference + torch.tensor([[0.0, 1.0, 0.0], [0.002, 0.0, 0.0]], dtype=torch.float64)
    gap = compare(reference, candidate)
    assert not gap["bit_for_bit"] and gap["scaled_gap"] == pytest.approx(1e-2) and gap["max_abs_diff"] == 1.0
    # a column the reference leaves at zero counts its gap as it is
    assert compare(reference, reference + torch.tensor([[0.0, 0.0, 3e-9], [0.0, 0.0, 0.0]],
                                                       dtype=torch.float64))["scaled_gap"] == pytest.approx(3e-9)
    nan = reference.clone()
    nan[0, 0] = float("nan")
    assert compare(reference, nan)["non_finite_mismatch"] == 1
    assert compare(nan, nan.clone())["bit_for_bit"]


def test_the_benchmark_takes_fortran_ordered_batches_and_summarises_its_processes(tmp_path: Path) -> None:
    import bench_torch_graph as bench

    names = bench.model_inputs("compute_uwshcu_inv")
    assert names[0] == "dt" and len(names) == 20
    anchors = tmp_path / "anchors.npz"
    shapes = {"dt": (60,), "tr0_inv": (60, 30, 57), "pblh": (60,), "cush": (60,)}
    np.savez(anchors, **{name: np.random.default_rng(0).random(shapes.get(name, (60, 30))) for name in names})
    batches = bench.batches(anchors, names, 27)
    assert len(batches) == 2 and batches[0][0].shape == (1,)
    assert batches[0][13].shape == (27, 30, 57) and batches[0][13].flags.f_contiguous
    out = tmp_path / "graph-32.jsonl"
    out.write_text("".join(json.dumps(row) + "\n" for row in (
        {"mode": "graph", "ms_per_forward": 20.0, "capture_seconds": 1.5},
        {"mode": "graph", "ms_per_forward": 30.0, "capture_seconds": 2.5, "compiled_gap": {"median": 1e-5, "max": 4e-5}},
        {"mode": "graph", "status": 4, "message": "capturing the forward failed"})))
    bench.main(["summarize", str(out), "--mode", "graph", "--procs", "3"])


def test_the_summary_names_the_slowest_process_and_the_first_refusal(tmp_path: Path, capsys) -> None:
    import bench_torch_graph as bench

    out = tmp_path / "compiled-2.jsonl"
    out.write_text(json.dumps({"ms_per_forward": 10.0, "compiled_gap": {"median": 1e-5, "max": 4e-5}, "capture_seconds": 3.0}) + "\n"
                   + json.dumps({"status": 8, "message": "the compiled forward is far"}) + "\n")
    bench.main(["summarize", str(out), "--mode", "compiled", "--procs", "2"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["finished"] == 1 and summary["ms_per_forward_slowest"] == 10.0
    assert summary["compiled_gap"] == {"median_largest": 1e-5, "max": 4e-5} and summary["refused"][0]["status"] == 8
