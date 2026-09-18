#!/usr/bin/env python3
"""Draw real columns out of a capture bundle to anchor a sampling space.

A capture bundle is the model's own state: every argument of every call, on
the manifold the model actually visits.  A sampling space built by perturbing
one hand-picked column cannot reach that manifold -- the vertical structure of
cloud water in one column is rank one, and no amount of retuning the noise
makes it rank twenty-four.  What does reach it is anchoring on the captured
columns themselves.

This writes a compact anchor file: ``N`` live columns, every user-visible
argument, the same column index used for all of them so a sample stays one
coherent atmospheric state rather than a mix of six unrelated ones.

    tools/extract_pi_cam_anchor_columns.py \\
        --function mmacro_pcond --bundle <capture>.npz \\
        --columns 200000 --output anchors.npz

    tools/extract_pi_cam_anchor_columns.py \\
        --function uwshcu --kernel compute_uwshcu_inv --frame-capture <run>/frame-capture \\
        --columns 200000 --output anchors.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics.spec import load_function_spec  # noqa: E402


def column_index(ncol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(record, lane) for every live column, padding lanes dropped."""

    records = np.repeat(np.arange(ncol.size, dtype=np.int64), ncol)
    lanes = np.concatenate([np.arange(int(n), dtype=np.int64) for n in ncol])
    return records, lanes


def extract(function: str, bundle_path: Path, columns: int, output: Path,
            seed: int) -> dict:
    spec = load_function_spec(function)
    bundle = np.load(bundle_path, allow_pickle=True)
    ncol = np.asarray(bundle["ncol"]).reshape(-1).astype(np.int64)
    records, lanes = column_index(ncol)
    live = records.size

    take = min(columns, live)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(live, take, replace=False)) if take < live \
        else np.arange(live)
    rec, lane = records[chosen], lanes[chosen]

    out: dict[str, np.ndarray] = {}
    for item in spec.user_arguments:
        array = np.asarray(bundle[f"before__{item.name}"])
        if item.rank == 2:                       # (pcols, pver, records)
            out[item.name] = array[lane, :, rec].astype(np.float64)
        elif item.rank == 1:                     # (pcols, records)
            out[item.name] = array[lane, rec].astype(np.float64)
        else:                                    # one value per call
            flat = array.reshape(-1) if array.ndim == 1 else array[0]
            out[item.name] = flat[rec].astype(np.float64)
        del array
        print(f"  {item.name:12s} {out[item.name].shape}", flush=True)

    for key in ("nstep", "lchnk", "mpi_rank", "dt"):
        out[f"meta_{key}"] = np.asarray(bundle[key]).reshape(-1)[rec]

    out["provenance"] = np.array(json.dumps({
        "function": function, "bundle": str(bundle_path),
        "live_columns": int(live), "columns": int(take), "seed": seed,
    }))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    return {"live": int(live), "taken": int(take),
            "arguments": len(spec.user_arguments)}


def extract_frames(function: str, capture: Path, kernel: str, columns: int, output: Path,
                   seed: int) -> dict:
    """The same anchors from frames captured at a kernel's hook: ``<capture>/<kernel>.rank-NNNN.npz``
    holding ``in/<call>/<argument>`` with the live columns as the leading axis and a ``meta`` list
    per call (step, ncol, token).  Every rank file is read once; the anchors are a seeded random
    subset of all live columns, sorted by rank, call and lane.
    """

    import re

    spec = load_function_spec(function)
    files = sorted(capture.glob(f"{kernel}.rank-*.npz"))
    if not files:
        raise SystemExit(f"{capture}: no {kernel}.rank-NNNN.npz files")
    counts = []                                  # (rank, call, ncol)
    for path in files:
        rank = int(re.search(r"rank-(\d+)\.npz$", path.name).group(1))
        with np.load(path, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"])) if "meta" in z.files else []
            calls = sorted({int(k.split("/")[1]) for k in z.files if k.startswith("in/")})
            for call in calls:
                ncol = int(meta[call]["ncol"]) if call < len(meta) else int(np.asarray(z[f"in/{call}/{spec.user_arguments[1].name}"]).shape[0])
                counts.append((rank, call, ncol))
    live = sum(n for _, _, n in counts)
    take = min(columns, live)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(live, take, replace=False)) if take < live else np.arange(live)
    # which (rank, call, lane) each chosen global column index is
    starts = np.cumsum([0] + [n for _, _, n in counts])
    owner = np.searchsorted(starts, chosen, side="right") - 1
    lane = chosen - starts[owner]
    out: dict[str, list] = {item.name: [] for item in spec.user_arguments}
    meta_out: dict[str, list] = {"nstep": [], "lchnk": [], "mpi_rank": [], "dt": []}
    by_entry: dict[int, list[int]] = {}
    for position, entry in enumerate(owner):
        by_entry.setdefault(int(entry), []).append(position)
    by_rank: dict[int, list[int]] = {}                       # one open per rank file, its calls in order
    for entry in sorted(by_entry):
        by_rank.setdefault(counts[entry][0], []).append(entry)
    for rank, entries in sorted(by_rank.items()):
        path = next(p for p in files if p.name.endswith(f"rank-{rank:04d}.npz"))
        with np.load(path, allow_pickle=True) as z:
            names = set(z.files)
            meta = json.loads(str(z["meta"])) if "meta" in names else []
            for entry in entries:
                _, call, _ = counts[entry]
                record = meta[call] if call < len(meta) else {}
                lanes = np.asarray([lane[p] for p in by_entry[entry]])
                for item in spec.user_arguments:
                    key = f"in/{call}/{item.name}"
                    if key not in names:
                        raise SystemExit(f"{path.name}: call {call} has no input {item.name!r}")
                    array = np.asarray(z[key], dtype=np.float64)
                    out[item.name].append(array[lanes] if item.rank >= 1 else np.repeat(array.reshape(-1)[:1], lanes.size))
                meta_out["nstep"].append(np.full(lanes.size, int(record.get("step", -1))))
                meta_out["lchnk"].append(np.full(lanes.size, int(np.asarray(z[f"in/{call}/lchnk"]).reshape(-1)[0]) if f"in/{call}/lchnk" in names else -1))
                meta_out["mpi_rank"].append(np.full(lanes.size, rank))
                meta_out["dt"].append(np.full(lanes.size, float(np.asarray(z[f"in/{call}/dt"]).reshape(-1)[0]) if f"in/{call}/dt" in names else np.nan))
    result: dict[str, np.ndarray] = {name: np.concatenate(parts, axis=0) for name, parts in out.items()}
    for item in spec.user_arguments:
        print(f"  {item.name:14s} {result[item.name].shape}", flush=True)
    for key, parts in meta_out.items():
        result[f"meta_{key}"] = np.concatenate(parts)
    result["provenance"] = np.array(json.dumps({
        "function": function, "kernel": kernel, "frame_capture": str(capture),
        "live_columns": int(live), "columns": int(take), "seed": seed,
    }))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **result)
    return {"live": int(live), "taken": int(take), "arguments": len(spec.user_arguments)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", default="mmacro_pcond")
    parser.add_argument("--bundle", type=Path, default=None, help="a function-capture bundle (<fn>_capture.npz)")
    parser.add_argument("--frame-capture", type=Path, default=None,
                        help="a run's frame-capture directory instead of a bundle: frames recorded at a kernel's hook")
    parser.add_argument("--kernel", default=None, help="the kernel whose frames to read (default: the function's name)")
    parser.add_argument("--columns", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if (arguments.bundle is None) == (arguments.frame_capture is None):
        raise SystemExit("give --bundle or --frame-capture, not both")
    if arguments.frame_capture is not None:
        summary = extract_frames(arguments.function, arguments.frame_capture, arguments.kernel or arguments.function,
                                 arguments.columns, arguments.output, arguments.seed)
    else:
        summary = extract(arguments.function, arguments.bundle,
                          arguments.columns, arguments.output, arguments.seed)
    print(f"  {summary['taken']} of {summary['live']} live columns, "
          f"{summary['arguments']} arguments -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
