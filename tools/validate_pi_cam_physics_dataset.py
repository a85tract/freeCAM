#!/usr/bin/env python3
"""Generate a small training dataset Driver-free and prove its properties.

Extracts one real column from the capture bundle as an anchor (saved under
validation/ with its provenance and the hybrid coordinate, so notebooks can
reproduce it), then generates a dataset twice with one seed -- same inputs
both times -- checks that every stored sample re-executes to its stored
output, that failed samples carry a status rather than fabricated data, and
that the NetCDF round-trips.  Runs in a plain Python process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from freecam.physics import Anchored, Dataset, HybridPressure, Normal, Uniform, load_function
from freecam.physics.spec import load_function_spec

REPO = Path(__file__).resolve().parents[1]


def anchor_from_bundle(spec, bundle_path: Path, history: Path, *, index: int, lane: int) -> dict:
    from netCDF4 import Dataset as NetCDF

    with np.load(bundle_path) as archive:
        inputs = {}
        for item in spec.user_arguments:
            values = archive[f"before__{item.name}"][..., index]
            inputs[item.name] = (np.asarray(values).reshape(-1)[0] if item.rank == 0 else values[lane]).tolist()
        meta = {key: int(archive[key][index]) for key in ("nstep", "lchnk", "ncol", "mpi_rank")}
        meta["dt"] = float(archive["dt"][index])
    with NetCDF(str(history)) as handle:
        hybrid = {
            "hyai": np.asarray(handle.variables["hyai"][...], dtype=np.float64).tolist(),
            "hybi": np.asarray(handle.variables["hybi"][...], dtype=np.float64).tolist(),
            "p0": float(np.asarray(handle.variables["P0"][...])),
        }
    return {
        "schema_version": 1,
        "function": spec.function,
        "source": {"bundle": str(bundle_path), "record": index, "lane": lane, **meta},
        "hybrid_coordinate": {**hybrid, "source": str(history)},
        "inputs": inputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True, help="CAM file carrying hyai/hybi/P0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    spec = load_function_spec(args.function)
    anchor = anchor_from_bundle(spec, args.bundle, args.history, index=args.record, lane=args.lane)
    anchor_path = REPO / "validation" / f"pi_cam_{args.function}_anchor_column.json"
    anchor_path.write_text(json.dumps(anchor, indent=2, sort_keys=True) + "\n")
    column = {name: np.asarray(value) for name, value in anchor["inputs"].items()}
    hybrid = anchor["hybrid_coordinate"]

    # Anchored draws for the thermodynamic column, a structural pressure
    # profile, and one parameter as an extra dimension.
    pressure_names = {
        "dadadj": ("pmid", "pdel", "pint"),
        "mmacro_pcond": ("p", "dp", None),
    }[args.function]
    distributions: dict = {}
    midpoint, thickness, interface = pressure_names
    produces = tuple(name for name in (midpoint, thickness, interface or "__pint") if name)
    surface = float(column[interface][-1]) if interface else float(column[midpoint][-1] + 0.5 * column[thickness][-1])
    distributions[midpoint] = HybridPressure(
        np.asarray(hybrid["hyai"]), np.asarray(hybrid["hybi"]), hybrid["p0"],
        Uniform(surface * 0.95, surface * 1.05), produces=(midpoint, thickness, interface or "__pint"),
    )
    if args.function == "dadadj":
        distributions["t"] = Anchored(column["t"], 2.0)
        distributions["q"] = Anchored(column["q"], 0.05, relative=True, clip=(0.0, None))
        distributions["nlvdry"] = Uniform(2, 6)
    else:
        distributions["t0"] = Anchored(column["t0"], 1.0)
        distributions["qv0"] = Anchored(column["qv0"], 0.05, relative=True, clip=(0.0, None))
        distributions["ql0"] = Anchored(column["ql0"], 0.05, relative=True, clip=(0.0, None))
        distributions["qi0"] = Anchored(column["qi0"], 0.05, relative=True, clip=(0.0, None))
        distributions["cldfrc_rhminl"] = Uniform(0.80, 0.95)
    fixed = {name: value for name, value in column.items() if name not in distributions and name not in produces}

    function = load_function(args.function)
    try:
        first = function.generate_dataset(args.samples, distributions, seed=args.seed, inputs=fixed)
        second = function.generate_dataset(args.samples, distributions, seed=args.seed, inputs=fixed)
        same_inputs = all(np.array_equal(first.inputs[k], second.inputs[k]) for k in first.inputs) and all(
            np.array_equal(first.parameters[k], second.parameters[k]) for k in first.parameters
        )
        same_outputs = all(np.array_equal(first.outputs[k], second.outputs[k], equal_nan=True) for k in first.outputs) and all(
            np.array_equal(first.updated[k], second.updated[k], equal_nan=True) for k in first.updated
        )
        valid = np.nonzero(first.valid)[0]
        replays = 0
        replay_equal = True
        for index in valid[:: max(1, len(valid) // 10)][:10]:
            sample = first.sample(int(index))
            again = function.run(sample["inputs"], sample["parameters"])
            replays += 1
            for name, value in {**first.outputs, **first.updated}.items():
                if not np.array_equal(again[name], value[index]):
                    replay_equal = False
        dataset_path = args.dataset or Path(os.environ.get("SCRATCH", "/glade/derecho/scratch/" + os.environ.get("USER", ""))) / "freecam-physics" / f"{args.function}_training.nc"
        first.save(dataset_path)
        loaded = Dataset.load(dataset_path)
        round_trip = all(np.array_equal(loaded.inputs[k], first.inputs[k]) for k in first.inputs) and np.array_equal(loaded.status, first.status)
    finally:
        function.close()

    failed = [str(item) for item in first.message if item]
    record = {
        "schema_version": 1,
        "gate": f"pi_cam_{args.function}_dataset_validation",
        "function": spec.qualified_name,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "driver_free": True,
        "mpi_initialized": "mpi4py" in sys.modules,
        "image_sha256": function.metadata["image_sha256"],
        "module_state_digest": function.metadata["module_state_digest"],
        "anchor_column": str(anchor_path),
        "samples": int(args.samples),
        "seed": int(args.seed),
        "sampling_space": first.attributes["sampling_space"],
        "status_counts": first.status_counts,
        "failure_messages": sorted(set(failed))[:10],
        "same_seed_same_inputs": bool(same_inputs),
        "same_seed_same_outputs": bool(same_outputs),
        "stored_samples_replayed": replays,
        "stored_samples_reexecute_identically": bool(replay_equal),
        "failed_samples_have_nan_outputs": bool(all(np.isnan(first.outputs[k][~first.valid]).all() for k in first.outputs) if (~first.valid).any() else True),
        "netcdf": str(dataset_path),
        "netcdf_round_trip": bool(round_trip),
        "worker_restarts": function.host.restarts,
        "passed": bool(same_inputs and same_outputs and replay_equal and round_trip),
    }
    output = args.output or REPO / "validation" / f"pi_cam_{args.function}_dataset_validation.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{args.function}: {'passed' if record['passed'] else 'FAILED'} {record['status_counts']} -> {dataset_path}")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
