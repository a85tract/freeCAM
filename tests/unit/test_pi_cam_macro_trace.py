"""The trace/capture comparison: same bytes, same hash; one bit off, named."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

import compare_pi_cam_macro_trace as tool  # noqa: E402
from freecam.physics.capture import lane_sha256  # noqa: E402


def _bundle_and_trace(tmp_path: Path, *, perturb: str | None = None):
    rng = np.random.default_rng(1)
    records = [(0, 1540, 1, 14), (0, 1541, 1, 13), (0, 1540, 2, 14)]
    n = len(records)
    arrays = {
        "mpi_rank": np.array([r[0] for r in records], dtype=np.int32),
        "lchnk": np.array([r[1] for r in records], dtype=np.int32),
        "nstep": np.array([r[2] for r in records], dtype=np.int32),
        "ncol": np.array([r[3] for r in records], dtype=np.int32),
        "before__t0": np.asfortranarray(rng.random((16, 30, n))),
        "before__landfrac": np.asfortranarray(rng.random((16, n))),
        "before__dt": np.full(n, 1800.0),
        "after__cld": np.asfortranarray(rng.random((16, 30, n))),
    }
    np.savez(tmp_path / "bundle.npz", **arrays)
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    lines = []
    for i, (rank, lchnk, nstep, ncol) in enumerate(records):
        before = {name.removeprefix("before__"): lane_sha256(arrays[name][..., i], ncol)
                  for name in arrays if name.startswith("before__")}
        after = {"cld": lane_sha256(arrays["after__cld"][..., i], ncol)}
        if perturb == "before" and i == 1:
            wrong = arrays["before__t0"][..., i].copy(); wrong[0, 0] = np.nextafter(wrong[0, 0], 1.0)
            before["t0"] = lane_sha256(wrong, ncol)
        if perturb == "after" and i == 2:
            wrong = arrays["after__cld"][..., i].copy(); wrong[5, 7] = np.nextafter(wrong[5, 7], 1.0)
            after["cld"] = lane_sha256(wrong, ncol)
        if perturb == "padding" and i == 0:
            wrong = arrays["before__t0"][..., i].copy(); wrong[15, :] = 12345.0   # a padding lane
            before["t0"] = lane_sha256(wrong, ncol)
        lines.append(json.dumps({"mpi_rank": rank, "lchnk": lchnk, "nstep": nstep, "ncol": ncol,
                                 "dt": 1800.0, "before": before, "after": after}))
    (trace_dir / "macro_trace.rank-1.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path / "bundle.npz", trace_dir


def test_identical_bytes_compare_identical(tmp_path: Path) -> None:
    report = tool.compare(*_bundle_and_trace(tmp_path))
    assert report["matched_records"] == 3 and report["identical"]
    assert report["records_with_differences"] == 0


def test_one_ulp_before_the_call_is_named_on_its_record(tmp_path: Path) -> None:
    report = tool.compare(*_bundle_and_trace(tmp_path, perturb="before"))
    assert not report["identical"]
    assert report["first_differing_before_call"] == {"t0": 1}
    assert report["differing_records"] == [
        {"mpi_rank": 0, "lchnk": 1541, "nstep": 1, "ncol": 13, "before": ["t0"], "after": []}]


def test_one_ulp_after_the_call_is_named_too(tmp_path: Path) -> None:
    report = tool.compare(*_bundle_and_trace(tmp_path, perturb="after"))
    assert report["first_differing_after_call"] == {"cld": 1}
    assert report["arguments_differing_before_call"] == {}


def test_a_padding_lane_never_counts(tmp_path: Path) -> None:
    """Lanes ncol..pcols-1 are not the routine's; garbage there is not a difference."""

    report = tool.compare(*_bundle_and_trace(tmp_path, perturb="padding"))
    assert report["identical"]
