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


from freecam.physics.spec import parse_function_spec  # noqa: E402


def _toy(**overrides):
    """A column routine: a structural ncol, a per-constituent logical flag, a profile in, a count
    in/out, a profile out and a logical out."""

    document = {
        "schema_version": 1, "function": "toy", "qualified_name": "toymod::toy", "routine": "toy",
        "source": "toy.F90", "module": "toymod", "layout": "column",
        "dimensions": {"pcols": 4, "pver": 3, "ncnst": 2},
        "arguments": [
            {"name": "ncol", "role": "structural", "fortran_type": "integer", "dtype": "int32", "rank": 0,
             "intent": "in", "native_shape": [], "value": 1},
            {"name": "flag", "role": "input", "fortran_type": "logical", "dtype": "int32", "rank": 1, "intent": "in",
             "native_shape": ["ncnst"], "public_shape": ["ncnst"], "carrier": "logical"},
            {"name": "t", "role": "input", "fortran_type": "real", "dtype": "float64", "rank": 2, "intent": "in",
             "native_shape": ["pcols", "pver"], "public_shape": ["pver"]},
            {"name": "n", "role": "inout", "fortran_type": "integer", "dtype": "int32", "rank": 0, "intent": "inout",
             "native_shape": [], "public_shape": []},
            {"name": "g", "role": "output", "fortran_type": "real", "dtype": "float64", "rank": 2, "intent": "out",
             "native_shape": ["pcols", "pver"], "public_shape": ["pver"]},
            {"name": "done", "role": "output", "fortran_type": "logical", "dtype": "int32", "rank": 1, "intent": "out",
             "native_shape": ["pcols"], "public_shape": [], "carrier": "logical"},
        ],
        "image": {"archive_members": ["toy.o"], "stubs": {}, "base_address": 0x50000000},
    }
    document.update(overrides)
    return parse_function_spec(document)


def _call(ncol=3):
    return {"meta": {"step": 1, "ncol": ncol},
            "inputs": {"ncol": np.array(ncol, dtype=np.int32), "flag": np.array([-1, 0], dtype=np.int32),
                       "t": np.arange(ncol * 3.0).reshape(ncol, 3), "n": np.array(0, dtype=np.int32)},
            "outputs": {"n": np.array(2, dtype=np.int32), "g": np.ones((ncol, 3)),
                        "done": np.array([-1, 0, -1][:ncol], dtype=np.int32)}}


def test_lane_samples_split_only_arrays_with_a_column_axis() -> None:
    samples = list(replay._samples(_toy(), _call()))
    assert len(samples) == 3
    sample, expected = samples[1]
    assert sample["flag"].tolist() == [-1, 0]                    # per constituent: the call's, every lane
    assert sample["t"].tolist() == [3.0, 4.0, 5.0] and expected["g"].shape == (3,)


def test_a_chunk_pool_holds_every_live_lane_the_call_ncol_and_markers() -> None:
    spec = _toy()
    pool = replay.chunk_pool(spec, _call())
    assert pool["toy.ncol"][0] == 3                              # the call's, not the declared 1
    assert pool["toy.t"][:3, :, 0].tolist() == _call()["inputs"]["t"].tolist() and not pool["toy.t"][3].any()
    assert pool["toy.flag"][:, 0].tolist() == [-1, 0]
    assert np.all(pool["toy.g"].view(np.uint64) == replay.MARKERS["float64"].view(np.uint64))
    assert np.all(pool["toy.done"] == replay.MARKERS["int32"])


def test_chunk_comparison_skips_unwritten_positions_and_reads_a_logical_as_its_truth() -> None:
    spec, call = _toy(), _call()
    pool = replay.chunk_pool(spec, call)
    pool["toy.n"][0] = 2
    pool["toy.g"][:2, :, 0] = 1.0                                # lane 2 left unwritten
    pool["toy.done"][:3, 0] = [1, 0, 1]                          # the wrapper's spelling of .true.
    compared, differences, unwritten = replay.compare_chunk(spec, call, pool)
    assert differences == [] and unwritten == {"g": 3} and compared == 1 + 6 + 3
    pool["toy.g"][1, 2, 0] = 1.0 + 2.0 ** -52
    _, differences, _ = replay.compare_chunk(spec, call, pool)
    assert [(d["argument"], d["index"], d["count"]) for d in differences] == [("g", [1, 2], 1)]


def test_a_gathered_array_is_compared_up_to_the_gathered_count() -> None:
    spec = _toy(columns="gathered", gathered={"count": "n", "arguments": ["g"]})
    call = _call()
    pool = replay.chunk_pool(spec, call)
    pool["toy.n"][0] = 2
    pool["toy.g"][:, :, 0] = 1.0
    pool["toy.g"][2, :, 0] = 7.0                                 # past the count: storage, not an answer
    pool["toy.done"][:3, 0] = [1, 0, 1]
    compared, differences, unwritten = replay.compare_chunk(spec, call, pool)
    assert differences == [] and unwritten == {"g": 3}


def test_lane_mode_refuses_a_gathered_routine(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["replay", "--function", "zm_convr", "--capture-dir", str(tmp_path)])
    assert replay.main() == 2


def test_shards_merge_their_unwritten_counts() -> None:
    def shard(unwritten):
        return {"rank_files": 1, "calls": 1, "samples": 1, "compared_values": 3, "statuses": {"ok": 1},
                "steps_covered": [1], "mismatches": [], "mismatches_truncated": False, "bfb": True, "seconds": 1.0,
                "unwritten_positions": unwritten}
    merged = replay.merge_shards([shard({"g": 3}), shard({"g": 1, "done": 2})], max_mismatches=8)
    assert merged["unwritten_positions"] == {"g": 4, "done": 2}
