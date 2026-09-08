"""The frame-replay tool reads a rank's capture in one pass and merges sharded records."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("replay_capture", REPO / "tools/replay_pi_cam_frame_capture.py")
replay = importlib.util.module_from_spec(spec)
sys.modules["replay_capture"] = replay
spec.loader.exec_module(replay)


def test_a_capture_file_is_read_call_by_call(tmp_path: Path) -> None:
    arrays = {"meta": np.array(json.dumps([{"step": 1, "ncol": 0, "token": 3, "kernel": "k"}, {"step": 2, "ncol": 0, "token": 5, "kernel": "k"}]))}
    for call in range(2):
        arrays[f"in/{call}/ps0"] = np.arange(31.0) + call
        arrays[f"in/{call}/cbmf"] = np.array(0.5 * call)
        arrays[f"out/{call}/xflx"] = np.full(31, float(call))
    path = tmp_path / "k.rank-0000.npz"
    np.savez(path, **arrays)
    calls = replay._load_capture(path)
    assert [c["meta"]["token"] for c in calls] == [3, 5]
    assert set(calls[1]["inputs"]) == {"ps0", "cbmf"} and calls[1]["inputs"]["ps0"][0] == 1.0
    assert calls[0]["outputs"]["xflx"].shape == (31,)


def test_shard_records_merge_into_one() -> None:
    def shard(i, calls, bfb, mismatches):
        return {"schema_version": 1, "kernel": "k", "rank_files": 2, "calls": calls, "samples": calls, "compared_values": 31 * calls,
                "statuses": {"ok": calls - len(mismatches), "invalid": len(mismatches)}, "steps_covered": [1, 2 + i],
                "mismatches": mismatches, "mismatches_truncated": False, "bfb": bfb, "seconds": 10.0 + i, "shard": f"{i}/2",
                "capture": {"pbs_job_id": "1.x"}, "image_sha256": "abc"}
    merged = replay.merge_shards([shard(0, 100, True, []), shard(1, 50, True, [])], max_mismatches=8)
    assert merged["calls"] == 150 and merged["rank_files"] == 4 and merged["compared_values"] == 31 * 150
    assert merged["steps_covered"] == [1, 2, 3] and merged["statuses"] == {"ok": 150, "invalid": 0}
    assert merged["bfb"] and merged["shards"] == 2 and merged["seconds"] == 21.0 and "shard" not in merged
    assert merged["capture"] == {"pbs_job_id": "1.x"} and merged["image_sha256"] == "abc"
    worse = replay.merge_shards([shard(0, 100, True, []), shard(1, 50, False, [{"call": 7}])], max_mismatches=8)
    assert not worse["bfb"] and worse["mismatches"] == [{"call": 7}]
