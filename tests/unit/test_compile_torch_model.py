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
    # 5e-4 off in a column ranging to 2, 1 off in a column ranging to 100: the larger share is 1e-2
    candidate = reference + torch.tensor([[0.0, 1.0, 0.0], [0.001, 0.0, 0.0]], dtype=torch.float64)
    gap = compare(reference, candidate)
    assert not gap["bit_for_bit"] and gap["scaled_gap"] == pytest.approx(1e-2) and gap["max_abs_diff"] == 1.0
    assert gap["columns"] == 3 and gap["columns_over_1e-3"] == 1
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


def test_a_tf32_phase_names_its_rows_and_its_largest_gap_from_float32(tmp_path: Path, capsys) -> None:
    import bench_torch_graph as bench

    out = tmp_path / "graph-1-864-tf32.jsonl"
    gaps = [{"scaled_gap_median": 1e-6, "scaled_gap": g, "columns_over_1e-3": n, "columns": 1190} for g, n in ((2e-3, 3), (5e-2, 9))]
    out.write_text("".join(json.dumps({"ms_per_forward": 4.0, "rows": 864, "tf32": True, "tf32_gap": g}) + "\n" for g in gaps))
    bench.main(["summarize", str(out), "--mode", "graph", "--procs", "2"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["rows"] == 864 and summary["tf32"] is True and summary["tf32_gap"]["scaled_gap"] == 5e-2


def test_a_trace_is_read_call_by_call_into_kernel_time_idle_time_and_launch_time(tmp_path: Path, capsys) -> None:
    import sqlite3

    import bench_torch_graph as bench

    path = tmp_path / "trace.sqlite"
    db = sqlite3.connect(str(path))
    db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "call"), (2, "cudaLaunchKernel_v7000"), (3, "cudaStreamSynchronize_v3020")])
    db.execute("CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, textId INTEGER)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER, end INTEGER)")
    for c in range(5):                                  # five calls of 10 ms, each three 1 us kernels 2 us apart
        t0 = c * 10_000_000
        db.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, NULL, 1)", (t0, t0 + 10_000_000))
        for k in range(3):
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?)", (t0 + 1000 + k * 3000, t0 + 2000 + k * 3000))
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 2)", (t0 + 100 + k * 500, t0 + 400 + k * 500))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 3)", (t0 + 2000, t0 + 9_000_000))
    db.commit()
    rows = bench._trace_calls(path)
    assert len(rows) == 5 and rows[0]["kernels"] == 3
    assert rows[0]["kernel_run_us_each"] == pytest.approx(1.0) and rows[0]["idle_us_between_each"] == pytest.approx(2.0)
    assert rows[0]["cpu_in_kernel_launches_ms"] == pytest.approx(0.0009) and rows[0]["cpu_waiting_ms"] == pytest.approx(8.998)
    bench.main(["profile-summary", str(path), "--mode", "ordinary", "--procs", "32"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["profiled"][0]["calls"] == 2 and summary["profiled"][0]["kernels_a_call"] == 3   # the first three left out
