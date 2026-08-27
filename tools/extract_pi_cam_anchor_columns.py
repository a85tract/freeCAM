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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", default="mmacro_pcond")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    summary = extract(arguments.function, arguments.bundle,
                      arguments.columns, arguments.output, arguments.seed)
    print(f"  {summary['taken']} of {summary['live']} live columns, "
          f"{summary['arguments']} arguments -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
