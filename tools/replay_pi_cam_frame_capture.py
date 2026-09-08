#!/usr/bin/env python3
"""Replay frames captured at a runner's pause through the routine's standalone function.

A capture run (validation/jobs/pi_cam_pausable_50step.pbs with
PYCAM_CAPTURE_KERNELS) answers a kernel with the original at its pause and
records every call's frame: the inputs the kernel was given and the outputs
it returned, per rank, as <dir>/<kernel>.rank-NNNN.npz.  This tool hands each
call to the same routine loaded standalone through freecam.physics.load_function
-- element by element for an elemental function, lane by lane for a chunk
routine, as declared for a profile routine -- and requires every output to
come back bit for bit.  The record it writes is the replay evidence of the
kernel-API closure; a mismatch is reported, never averaged away.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics import load_function  # noqa: E402
from freecam.physics.spec import load_function_spec  # noqa: E402


def _bits(value: float) -> str:
    return hex(int(np.asarray(value, dtype=np.float64).view(np.uint64)))


def _load_capture(path: Path) -> list[dict]:
    """Every call in one rank's capture file: its meta, inputs and outputs."""

    with np.load(path) as bundle:
        meta = json.loads(str(bundle["meta"]))
        keys: dict[int, dict[str, dict[str, str]]] = {}
        for key in bundle.files:                      # indexed once: a rank holds up to 700000 arrays
            parts = key.split("/", 2)
            if len(parts) == 3 and parts[0] in ("in", "out"):
                keys.setdefault(int(parts[1]), {"in": {}, "out": {}})[parts[0]][parts[2]] = key
        calls = []
        for index, item in enumerate(meta):
            names = keys.get(index, {"in": {}, "out": {}})
            inputs = {name: bundle[key] for name, key in names["in"].items()}
            outputs = {name: bundle[key] for name, key in names["out"].items()}
            calls.append({"meta": item, "inputs": inputs, "outputs": outputs})
    return calls


def merge_shards(records: list[dict], max_mismatches: int) -> dict:
    """One record from the shards of a replay split over rank files."""

    first = records[0]
    merged = dict(first)
    for key in ("rank_files", "calls", "samples", "compared_values"):
        merged[key] = sum(int(r[key]) for r in records)
    statuses: dict[str, int] = {}
    for r in records:
        for status, count in r["statuses"].items():
            statuses[status] = statuses.get(status, 0) + int(count)
    merged["statuses"] = statuses
    merged["steps_covered"] = sorted({int(s) for r in records for s in r["steps_covered"]})
    mismatches = [m for r in records for m in r["mismatches"]]
    merged["mismatches"] = mismatches[:max_mismatches]
    merged["mismatches_truncated"] = len(mismatches) > max_mismatches or any(r["mismatches_truncated"] for r in records)
    merged["bfb"] = all(r["bfb"] for r in records) and merged["samples"] > 0
    merged["seconds"] = round(sum(float(r["seconds"]) for r in records), 1)
    merged["shards"] = len(records)
    merged.pop("shard", None)
    return merged


def _samples(spec, call: dict):
    """Yield (inputs, expected outputs) samples for one captured call, by the spec's layout."""

    inputs, outputs, ncol = call["inputs"], call["outputs"], int(call["meta"]["ncol"])
    result_name = spec.result.name if spec.result is not None else None
    names = [item.name for item in spec.arguments if item.user_visible]
    if spec.layout == "direct" and all(inputs[name].ndim == 1 for name in names if name in inputs) \
            and all(spec.argument(name).rank == 0 for name in names):
        # an elemental routine applied to sections: one scalar call per element
        for element in range(ncol):
            sample = {name: inputs[name][element] for name in names if name in inputs}
            expected = {("result" if name == result_name else name): np.asarray(value[element])
                        for name, value in outputs.items()}
            yield sample, expected
        return
    if spec.layout == "column":
        # a chunk routine: the frame holds (pcols, ...) arrays with ncol live lanes
        for lane in range(ncol):
            sample = {name: inputs[name][lane] for name in names if name in inputs}
            expected = {name: np.asarray(value[lane]) for name, value in outputs.items()}
            yield sample, expected
        return
    sample = {name: inputs[name] for name in names if name in inputs}
    expected = {("result" if name == result_name else name): np.asarray(value) for name, value in outputs.items()}
    yield sample, expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--function", required=True, help="the function spec / standalone image name")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--kernel", help="the kernel name in the capture files (default: the function name)")
    parser.add_argument("--output", type=Path, help="the replay record (default validation/pi_cam_<function>_frame_replay.json)")
    parser.add_argument("--max-ranks", type=int, default=0, help="replay this many rank files only (0 = all)")
    parser.add_argument("--max-mismatches", type=int, default=8)
    parser.add_argument("--shard", help="I/N: replay every N-th rank file from the I-th (0-based); the record is "
                                        "written as <output>.shard-I-of-N.json for --merge-shards")
    parser.add_argument("--merge-shards", type=int, default=0,
                        help="merge <output>.shard-*-of-N.json into the record instead of replaying")
    arguments = parser.parse_args()

    spec = load_function_spec(arguments.function)
    kernel = arguments.kernel or arguments.function
    output = arguments.output or (REPO / "validation" / f"pi_cam_{arguments.function}_frame_replay.json")
    if arguments.merge_shards:
        parts = [output.with_name(f"{output.name}.shard-{i}-of-{arguments.merge_shards}.json") for i in range(arguments.merge_shards)]
        absent = [str(p) for p in parts if not p.is_file()]
        if absent:
            print(f"shard records are absent: {absent}", file=sys.stderr)
            return 1
        record = merge_shards([json.loads(p.read_text()) for p in parts], arguments.max_mismatches)
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{kernel}: {record['calls']} calls, {record['samples']} samples, {record['compared_values']} values compared, "
              f"bfb={record['bfb']}, {len(record['mismatches'])} mismatches recorded, {record['shards']} shards -> {output}")
        return 0 if record["bfb"] else 1
    files = sorted(arguments.capture_dir.glob(f"{kernel}.rank-*.npz"))
    if not files:
        print(f"no capture files for {kernel} under {arguments.capture_dir}", file=sys.stderr)
        return 1
    if arguments.max_ranks:
        files = files[: arguments.max_ranks]
    shard = None
    if arguments.shard:
        index, count = (int(x) for x in arguments.shard.split("/"))
        files = files[index::count]
        shard = f"{index}/{count}"
        output = output.with_name(f"{output.name}.shard-{index}-of-{count}.json")
    provenance_path = arguments.capture_dir / "capture.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}

    function = load_function(arguments.function)
    started = time.time()
    calls = samples = compared_values = 0
    mismatches: list[dict] = []
    statuses: dict[str, int] = {}
    steps: set[int] = set()
    try:
        for path in files:
            for call in _load_capture(path):
                calls += 1
                if call["meta"].get("step") is not None:
                    steps.add(int(call["meta"]["step"]))
                for sample, expected in _samples(spec, call):
                    samples += 1
                    result = function.try_run(sample)
                    statuses[result.status] = statuses.get(result.status, 0) + 1
                    if not result.ok:
                        if len(mismatches) < arguments.max_mismatches:
                            mismatches.append({"file": path.name, "call": calls, "status": result.status, "message": result.message})
                        continue
                    for name, want in expected.items():
                        got = np.asarray(result[name], dtype=want.dtype)
                        compared_values += int(want.size)
                        if got.shape != want.shape or not np.array_equal(got.view(np.uint64) if want.dtype == np.float64 else got,
                                                                          want.view(np.uint64) if want.dtype == np.float64 else want):
                            if len(mismatches) < arguments.max_mismatches:
                                flat_want, flat_got = np.ravel(want), np.ravel(got)
                                where = next((i for i in range(min(flat_want.size, flat_got.size)) if flat_want[i] != flat_got[i]), 0)
                                mismatches.append({
                                    "file": path.name, "call": calls, "argument": name, "index": int(where),
                                    "model": float(flat_want[where]), "model_bits": _bits(flat_want[where]),
                                    "standalone": float(flat_got[where]), "standalone_bits": _bits(flat_got[where]),
                                })
                            else:
                                mismatches.append({"truncated": True})
                                break
    finally:
        function.close()
    bfb = not mismatches and statuses.get("ok", 0) == samples and samples > 0
    record = {
        "schema_version": 1,
        "what": "every frame captured at the kernel's pause replayed through the standalone function; bit-for-bit or not",
        "function": spec.qualified_name,
        "kernel": kernel,
        "layout": spec.layout,
        "binding": spec.binding,
        "capture": {k: provenance.get(k) for k in ("pbs_job_id", "run_tag", "native_library_sha256", "kernels", "bfb_record")},
        "rank_files": len(files),
        "calls": calls,
        "samples": samples,
        "compared_values": compared_values,
        "steps_covered": sorted(steps),
        "statuses": statuses,
        "mismatches": [m for m in mismatches if not m.get("truncated")],
        "mismatches_truncated": any(m.get("truncated") for m in mismatches),
        "bfb": bfb,
        "image_sha256": function.metadata.get("image_sha256"),
        "module_state_digest": function.metadata.get("module_state_digest"),
        "seconds": round(time.time() - started, 1),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
    }
    if shard is not None:
        record["shard"] = shard
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{kernel}: {calls} calls, {samples} samples, {compared_values} values compared, bfb={bfb}, "
          f"{len(record['mismatches'])} mismatches recorded -> {output}")
    return 0 if bfb else 1


if __name__ == "__main__":
    raise SystemExit(main())
