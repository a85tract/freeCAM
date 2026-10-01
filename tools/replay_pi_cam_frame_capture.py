#!/usr/bin/env python3
"""Replay frames captured at a runner's pause through the routine's standalone function.

A capture run (validation/jobs/pi_cam_pausable_50step.pbs with
PYCAM_CAPTURE_KERNELS) answers a kernel with the original at its pause and
records every call's frame: the inputs the kernel was given and the outputs
it returned, per rank, as <dir>/<kernel>.rank-NNNN.npz.  This tool hands each
call to the same routine loaded standalone through freecam.physics.load_function
and requires every output to come back bit for bit.  The record it writes is
the replay evidence of the kernel-API closure; a mismatch is reported, never
averaged away.

Two modes.  ``lane`` (the default) goes through the public function --
element by element for an elemental function, lane by lane for a chunk
routine, as declared for a profile routine -- so a column is proven to stand
alone.  ``chunk`` hands the routine each captured call whole, every live lane
at the model's ncol, which is what a routine whose columns are gathered
(``columns: gathered``) admits, and what a routine whose compiled column loop
rounds a lane by its position needs.  In chunk mode every intent(out) array
is filled with a marker before the call, and only the positions the routine
wrote are compared: the rest held, in the model, whatever the storage had,
and are counted in the record as unwritten, not hidden.
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
from freecam.physics.column import empty_pool, presence_field  # noqa: E402
from freecam.physics.spec import ArgumentSpec, FunctionSpec, load_function_spec  # noqa: E402

#: what an intent(out) array holds before a chunk call: a quiet NaN with a payload no
#: computation produces, and integers no index or count takes
MARKERS = {
    "float64": np.array([0x7FF8DEADBEEF0001], dtype=np.uint64).view(np.float64)[0],
    "int32": np.int32(-2147483647),
    "int64": np.int64(-9223372036854775807),
}


def _bits(value) -> str:
    array = np.asarray(value)
    if array.dtype == np.float64:
        return hex(int(array.view(np.uint64)))
    return hex(int(array))


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
    if any("unwritten_positions" in r for r in records):
        unwritten: dict[str, int] = {}
        for r in records:
            for name, count in (r.get("unwritten_positions") or {}).items():
                unwritten[name] = unwritten.get(name, 0) + int(count)
        merged["unwritten_positions"] = unwritten
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
        # a chunk routine: the frame holds (pcols, ...) arrays with ncol live lanes; an argument
        # without a column axis (dt, a per-constituent flag) is the call's, the same for every lane
        def lane_of(name: str, value: np.ndarray, lane: int) -> np.ndarray:
            return value[lane] if _by_lane(spec, name) else value

        for lane in range(ncol):
            sample = {name: lane_of(name, inputs[name], lane) for name in names if name in inputs}
            expected = {name: np.asarray(lane_of(name, value, lane)) for name, value in outputs.items()}
            yield sample, expected
        return
    sample = {name: inputs[name] for name in names if name in inputs}
    expected = {("result" if name == result_name else name): np.asarray(value) for name, value in outputs.items()}
    yield sample, expected


def _by_lane(spec: FunctionSpec, name: str) -> bool:
    """Whether an argument's first axis is the chunk's columns."""

    try:
        item = spec.argument(name)
    except KeyError:
        return False
    return bool(item.native_shape) and item.native_shape[0] == "pcols"


def _differs(item: ArgumentSpec | None, want: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Where two values differ: bitwise, except that a logical is its truth.

    ifort stores .true. as -1 and tests the low bit; the standalone wrapper hands a
    logical back as 1 -- the same truth in another spelling.
    """

    if item is not None and item.carrier == "logical":
        return (np.asarray(want).astype(np.int64) & 1) != (np.asarray(got).astype(np.int64) & 1)
    if want.dtype == np.float64:
        return np.ascontiguousarray(want).view(np.uint64) != np.ascontiguousarray(got, dtype=np.float64).view(np.uint64)
    return np.asarray(want) != np.asarray(got, dtype=want.dtype)


def _argument_or_none(spec: FunctionSpec, name: str) -> ArgumentSpec | None:
    try:
        return spec.argument(name)
    except KeyError:
        return None


def chunk_pool(spec: FunctionSpec, call: dict) -> dict[str, np.ndarray]:
    """One captured call as the routine's own pool: every live lane, the model's ncol.

    Inputs and in/out values are the captured ones; an intent(out) array holds the
    marker, so what the routine leaves unwritten can be told from what it wrote.  A
    structural argument takes the captured value (the call's ncol, top_lev, chunk)
    unless it sizes the arrays: the image's wrapper is built on the declared extents,
    and a call packed narrower (MG's mgncol as pcols) is laid into their first lanes.
    """

    ncol = int(call["meta"]["ncol"])
    pool = empty_pool(spec, 1)
    for item in spec.arguments:
        target = pool[f"{spec.function}.{item.name}"]
        if item.role == "structural":
            captured = call["inputs"].get(item.name)
            if captured is None or item.name in spec.dimensions:
                if captured is not None and int(captured) > int(item.value):
                    raise ValueError(f"{item.name} was {int(captured)} in the call, beyond the declared {item.value}")
                target[...] = item.value
            else:
                target[...] = np.asarray(captured).reshape(-1)[0]
            continue
        if item.role in ("output", "result"):
            target[...] = MARKERS[item.dtype]
            continue
        value = call["inputs"].get(item.name)
        if item.optional:
            pool[presence_field(spec, item)][0] = 0 if value is None else 1
        if value is None or (item.rank and value.size == 0):
            continue                        # workspace the call never touches, an absent optional
        if item.rank == 0:
            target[0] = np.asarray(value).reshape(-1)[0]
        elif item.native_shape[0] == "pcols":
            target[:ncol, ..., 0] = value
        else:
            target[..., 0] = value
    return pool


def compare_chunk(spec: FunctionSpec, call: dict, pool: dict[str, np.ndarray]) -> tuple[int, list[dict], dict[str, int]]:
    """Values compared, differences, and positions not compared, for one chunk call.

    A position is not compared when the routine left an intent(out) array unwritten
    there, or when it lies past the gathered count of an array indexed by gathered
    position: the routine's answer ends at the count, and what follows is storage.
    """

    ncol = int(call["meta"]["ncol"])
    count = None
    if spec.gather_count is not None:
        value = call["outputs"].get(spec.gather_count, call["inputs"].get(spec.gather_count))
        count = int(np.asarray(value).reshape(-1)[0])
    compared, differences, unwritten = 0, [], {}
    for name, want in call["outputs"].items():
        item = _argument_or_none(spec, name)
        if item is None or not item.returned:
            continue
        got = pool[f"{spec.function}.{name}"][..., 0]
        if item.rank == 0:
            got = got.reshape(-1)[0]
        elif item.native_shape[0] == "pcols":
            got = got[:ncol]
        got = np.asarray(got).reshape(want.shape)
        written = np.ones(want.shape, dtype=bool)
        if item.role in ("output", "result"):
            marker = np.asarray(MARKERS[item.dtype])
            written = (np.asarray(got).view(np.uint64) != marker.view(np.uint64)) if item.dtype == "float64" \
                else np.asarray(got) != marker
        if name in spec.gathered:
            written[count:] = False
        if not written.all():
            unwritten[name] = int((~written).sum())
        compared += int(written.sum())
        differ = _differs(item, want, got) & written
        if differ.any():
            where = tuple(int(i) for i in np.argwhere(differ)[0]) if differ.ndim else ()
            differences.append({"argument": name, "index": list(where), "count": int(differ.sum()),
                                "model": float(want[where]), "model_bits": _bits(want[where]),
                                "standalone": float(got[where]), "standalone_bits": _bits(got[where])})
    return compared, differences, unwritten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--function", required=True, help="the function spec / standalone image name")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--kernel", help="the kernel name in the capture files (default: the function name)")
    parser.add_argument("--mode", choices=("lane", "chunk"), default="lane",
                        help="lane: every column alone through the public function; chunk: every call whole")
    parser.add_argument("--output", type=Path, help="the replay record (default validation/pi_cam_<function>_frame_replay.json, "
                                                    "..._frame_replay_chunk.json in chunk mode)")
    parser.add_argument("--max-ranks", type=int, default=0, help="replay this many rank files only (0 = all)")
    parser.add_argument("--max-mismatches", type=int, default=8)
    parser.add_argument("--shard", help="I/N: replay every N-th rank file from the I-th (0-based); the record is "
                                        "written as <output>.shard-I-of-N.json for --merge-shards")
    parser.add_argument("--merge-shards", type=int, default=0,
                        help="merge <output>.shard-*-of-N.json into the record instead of replaying")
    arguments = parser.parse_args()

    spec = load_function_spec(arguments.function)
    kernel = arguments.kernel or arguments.function
    suffix = "_chunk" if arguments.mode == "chunk" else ""
    output = arguments.output or (REPO / "validation" / f"pi_cam_{arguments.function}_frame_replay{suffix}.json")
    if arguments.mode == "lane" and spec.columns == "gathered" and not arguments.merge_shards:
        print(f"{kernel}: its columns are gathered (spec columns: gathered); one column cannot be "
              "replayed alone -- use --mode chunk", file=sys.stderr)
        return 2
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
    unwritten: dict[str, int] = {}
    returned = tuple(f"{spec.function}.{item.name}" for item in spec.arguments if item.returned)
    try:
        for path in files:
            for call in _load_capture(path):
                calls += 1
                if call["meta"].get("step") is not None:
                    steps.add(int(call["meta"]["step"]))
                if arguments.mode == "chunk":
                    samples += 1
                    outcome = function.host.call(chunk_pool(spec, call), returned)
                    statuses[outcome.status] = statuses.get(outcome.status, 0) + 1
                    if outcome.status != "ok":
                        if len(mismatches) < arguments.max_mismatches:
                            mismatches.append({"file": path.name, "call": calls, "status": outcome.status, "message": outcome.message})
                        continue
                    compared, differences, left = compare_chunk(spec, call, outcome.pool)
                    compared_values += compared
                    for name, count in left.items():
                        unwritten[name] = unwritten.get(name, 0) + count
                    for difference in differences:
                        if len(mismatches) < arguments.max_mismatches:
                            mismatches.append({"file": path.name, "call": calls, **difference})
                        else:
                            mismatches.append({"truncated": True})
                            break
                    continue
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
                        item = _argument_or_none(spec, name if name != "result" else (spec.result.name if spec.result else name))
                        if got.shape != want.shape or np.any(_differs(item, want, got)):
                            if len(mismatches) < arguments.max_mismatches:
                                flat_want, flat_got = np.ravel(want), np.ravel(got)
                                where = next((i for i in range(min(flat_want.size, flat_got.size))
                                              if _differs(item, flat_want[i:i + 1], flat_got[i:i + 1])[0]), 0)
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
        "what": ("every call captured at the kernel's pause replayed whole through the standalone routine, every live "
                 "lane at the model's ncol; positions an intent(out) array was left unwritten, and positions past the "
                 "gathered count of a gathered array, are counted, not compared"
                 if arguments.mode == "chunk" else
                 "every frame captured at the kernel's pause replayed through the standalone function; bit-for-bit or not"),
        "mode": arguments.mode,
        "function": spec.qualified_name,
        "kernel": kernel,
        "layout": spec.layout,
        "columns": spec.columns,
        "structural_from_capture": ([item.name for item in spec.structural if item.name not in spec.dimensions]
                                    if arguments.mode == "chunk" else []),
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
    if arguments.mode == "chunk":
        record["unwritten_positions"] = unwritten
    if shard is not None:
        record["shard"] = shard
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{kernel}: {calls} calls, {samples} samples, {compared_values} values compared, bfb={bfb}, "
          f"{len(record['mismatches'])} mismatches recorded -> {output}")
    return 0 if bfb else 1


if __name__ == "__main__":
    raise SystemExit(main())
