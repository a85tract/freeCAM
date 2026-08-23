#!/usr/bin/env python3
"""Replay captured model calls through a standalone image and compare bit for bit.

Two gates.  Gate A hands the image every captured chunk exactly as the model
passed it -- all lanes, the model's ncol, the model's module state from the
snapshot -- and requires every output and in/out argument to come back bit
for bit.  Gate B is the product claim: one captured column packed into lane
0 with ncol = 1 must equal that column's result in the model.  Both run in a
plain Python process: no Driver, no MPI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from freecam.physics.image import StandaloneImage, _first_difference, bitwise_equal, reexec_with_math_library
from freecam.physics.spec import FunctionSpec

REPO = Path(__file__).resolve().parents[1]


def _pool_from_record(image: StandaloneImage, bundle: dict[str, np.ndarray], index: int) -> dict[str, np.ndarray]:
    """The native pool for one captured chunk, every lane as the model had it."""

    spec = image.spec
    pool = image.empty_pool(1)
    for item in spec.arguments:
        key = f"{spec.function}.{item.name}"
        tag = item.name
        if f"before__{tag}__associated" in bundle and not bool(bundle[f"before__{tag}__associated"][index]):
            continue  # unassociated pointer: zeros, never dereferenced
        values = bundle[f"before__{tag}"][..., index]
        target = pool[key]
        if item.rank == 0:
            target[0] = np.asarray(values).reshape(-1)[0]
        else:
            target[..., 0] = values
    return pool


def _pack_column(image: StandaloneImage, full: dict[str, np.ndarray], lane: int) -> dict[str, np.ndarray]:
    """Lane ``lane`` of a full-chunk pool moved into lane 0 with ncol = 1."""

    spec = image.spec
    pool = image.empty_pool(1)
    for item in spec.arguments:
        key = f"{spec.function}.{item.name}"
        if item.rank == 0:
            pool[key][0] = 1 if item.name == "ncol" else full[key][0]
        else:
            pool[key][0, ..., 0] = full[key][lane, ..., 0]
    return pool


def _compare(spec: FunctionSpec, pool: dict[str, np.ndarray], bundle, index: int, *, lanes: slice | int) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in spec.arguments:
        if not item.returned:
            continue
        reference = bundle[f"after__{item.name}"][..., index]
        candidate = pool[f"{spec.function}.{item.name}"][..., 0]
        if isinstance(lanes, int):
            reference = reference[lanes]
            candidate = candidate[0]
        else:
            reference = reference[lanes]
            candidate = candidate[lanes]
        difference = _first_difference(reference, candidate)
        result[item.name] = {"equal": difference is None, **({"first_difference": difference} if difference else {})}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--single-column-output", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--single-column-records", type=int, default=8)
    args = parser.parse_args()

    manifest = args.manifest or REPO / "build/pi_cam_standalone" / args.function / "manifest.json"
    snapshot_path = args.snapshot or REPO / "validation" / f"pi_cam_{args.function}_module_state.json"
    reexec_with_math_library(manifest)
    image = StandaloneImage(manifest)
    spec = image.spec
    snapshot = json.loads(snapshot_path.read_text())
    verification = image.initialize(snapshot)
    # An .npz re-reads an array on every key access; load each once.
    with np.load(args.bundle) as archive:
        bundle = {key: archive[key] for key in archive.files}
    count = int(bundle["ncol"].shape[0])
    if args.max_records is not None:
        count = min(count, args.max_records)

    # Gate A: full chunk, model ncol, all lanes.
    records = []
    all_equal = True
    for index in range(count):
        pool = _pool_from_record(image, bundle, index)
        image.call(pool)
        ncol = int(bundle["ncol"][index])
        active = _compare(spec, pool, bundle, index, lanes=slice(0, ncol))
        padding = _compare(spec, pool, bundle, index, lanes=slice(ncol, None))
        equal = all(item["equal"] for item in active.values()) and all(item["equal"] for item in padding.values())
        all_equal = all_equal and equal
        records.append({
            "index": index, "nstep": int(bundle["nstep"][index]), "lchnk": int(bundle["lchnk"][index]),
            "ncol": ncol, "mpi_rank": int(bundle["mpi_rank"][index]), "equal": equal,
            "active_lanes": active, "padding_lanes_unchanged": all(item["equal"] for item in padding.values()),
        })
    full_chunk = {
        "schema_version": 1,
        "gate": f"pi_cam_{args.function}_full_chunk_vs_capture",
        "function": spec.function,
        "qualified_name": spec.qualified_name,
        "passed": all_equal,
        "driver_free": True,
        "mpi_initialized": "mpi4py" in sys.modules,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "image": str(image.library_path),
        "image_sha256": image.manifest["library_sha256"],
        "original_call_proof": image.manifest["original_call_proof"],
        "intel_math_library": str(image.math_library),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "snapshot": str(snapshot_path),
        "snapshot_digest": snapshot.get("digest"),
        "module_state_verification": verification,
        "bundle": str(args.bundle),
        "records_compared": count,
        "records_equal": sum(1 for item in records if item["equal"]),
        "outputs_compared": [item.name for item in spec.arguments if item.returned],
        "records": records,
    }
    output = args.output or REPO / "validation" / f"pi_cam_{args.function}_full_chunk_vs_capture.json"
    output.write_text(json.dumps(full_chunk, indent=2, sort_keys=True) + "\n")
    print(f"gate A full chunk: {'passed' if all_equal else 'FAILED'} ({full_chunk['records_equal']}/{count} records)")

    # Gate B: one column at a time, lane 0, ncol = 1.
    columns = []
    single_equal = True
    for index in range(min(count, args.single_column_records)):
        full = _pool_from_record(image, bundle, index)
        ncol = int(bundle["ncol"][index])
        for lane in range(ncol):
            pool = _pack_column(image, full, lane)
            image.call(pool)
            comparison = _compare(spec, pool, bundle, index, lanes=lane)
            equal = all(item["equal"] for item in comparison.values())
            single_equal = single_equal and equal
            columns.append({"index": index, "lane": lane, "equal": equal,
                            **({"outputs": comparison} if not equal else {})})
    single = {
        "schema_version": 1,
        "gate": f"pi_cam_{args.function}_single_column_vs_capture",
        "function": spec.function,
        "passed": single_equal,
        "driver_free": True,
        "mpi_initialized": "mpi4py" in sys.modules,
        "image_sha256": image.manifest["library_sha256"],
        "intel_math_library": str(image.math_library),
        "ld_preload": os.environ.get("LD_PRELOAD"),
        "snapshot_digest": snapshot.get("digest"),
        "columns_compared": len(columns),
        "columns_equal": sum(1 for item in columns if item["equal"]),
        "columns": columns,
    }
    output = args.single_column_output or REPO / "validation" / f"pi_cam_{args.function}_single_column_vs_capture.json"
    output.write_text(json.dumps(single, indent=2, sort_keys=True) + "\n")
    print(f"gate B single column: {'passed' if single_equal else 'FAILED'} ({single['columns_equal']}/{len(columns)} columns)")
    return 0 if all_equal and single_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
