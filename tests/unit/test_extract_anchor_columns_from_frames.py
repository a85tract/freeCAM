"""Anchors drawn from frames captured at a kernel's hook: whole columns, every user argument, provenance."""
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _tool():
    spec = importlib.util.spec_from_file_location("extract_anchors", REPO / "tools/extract_pi_cam_anchor_columns.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_capture_anchors_are_whole_columns_with_their_provenance(tmp_path: Path) -> None:
    from freecam.physics.spec import load_function_spec

    tool = _tool()
    spec = load_function_spec("uwshcu")
    by_name = {a.name: a for a in spec.arguments}
    dims = dict(spec.dimensions)
    capture = tmp_path / "frame-capture"; capture.mkdir()
    rng = np.random.default_rng(1)
    for rank in range(3):
        arrays, meta = {}, []
        for call in range(2):
            ncol = 4 if call == 0 else 2
            meta.append({"step": 1 + 12 * call, "ncol": ncol, "token": 1 + 50 * call, "kernel": "compute_uwshcu_inv"})
            for item in spec.user_arguments:
                shape = [ncol] + [int(dims[str(a)]) for a in item.native_shape[1:]]
                value = np.asarray(1800.0) if item.rank == 0 else rng.uniform(0, 1, shape)
                if item.name == "t0_inv":
                    value = np.tile((100.0 * rank + 10.0 * call + np.arange(ncol))[:, None], (1, 30))   # says which column it is
                arrays[f"in/{call}/{item.name}"] = value
            arrays[f"in/{call}/lchnk"] = np.asarray(700 + rank)
        arrays["meta"] = np.asarray(json.dumps(meta))
        np.savez(capture / f"compute_uwshcu_inv.rank-{rank:04d}.npz", **arrays)
    out = tmp_path / "anchors.npz"
    summary = tool.extract_frames("uwshcu", capture, "compute_uwshcu_inv", 7, out, seed=0)
    assert summary == {"live": 18, "taken": 7, "arguments": 20}
    anchors = np.load(out, allow_pickle=True)
    names = [a.name for a in spec.user_arguments]
    assert all(anchors[n].shape[0] == 7 for n in names) and anchors["tr0_inv"].shape == (7, 30, 57) and anchors["dt"].shape == (7,)
    # every anchor is one column: its temperature says rank, call and lane, and the metadata agree
    ranks = np.asarray(anchors["meta_mpi_rank"]); steps = np.asarray(anchors["meta_nstep"]); lchnk = np.asarray(anchors["meta_lchnk"])
    tag = anchors["t0_inv"][:, 0]
    assert np.array_equal(tag // 100, ranks) and np.array_equal((tag % 100) // 10, (steps - 1) // 12) and np.array_equal(lchnk, 700 + ranks)
    assert np.all(anchors["meta_dt"] == 1800.0) and np.all(anchors["dt"] == 1800.0)
    provenance = json.loads(str(anchors["provenance"]))
    assert provenance["live_columns"] == 18 and provenance["columns"] == 7 and provenance["kernel"] == "compute_uwshcu_inv"
    # all of them, when asked for more than there are
    tool.extract_frames("uwshcu", capture, "compute_uwshcu_inv", 100, tmp_path / "all.npz", seed=0)
    assert np.load(tmp_path / "all.npz", allow_pickle=True)["t0_inv"].shape[0] == 18
