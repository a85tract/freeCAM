#!/usr/bin/env python3
"""Exercise one promoted original CAM routine, then run the 50-step BFB gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from mpi4py import MPI
import numpy as np

from freecam.pi_cam import PICAMCase, ReplayBoundaryProvider
from freecam.pi_cam.native import NativeCAMDevice


def _hash(array: np.ndarray) -> str:
    return sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    backend = NativeCAMDevice(args.native_manifest)
    with case.runtime(
        boundary=ReplayBoundaryProvider(args.boundary),
        backend=backend,
        communicator=world,
        run_dir=args.run_dir,
    ) as cam:
        cam.initialize()
        temperature = cam.pool["phys_state.t"]
        temperature_hash = _hash(temperature)
        result = cam.physics.cloud_fraction_fice()
        process = cam.process_contexts.record("cloud_fraction_fice")
        binding = {item.argument: item.field for item in process.bindings}
        output_names = (binding["fice"], binding["fsnow"])
        output_addresses = {
            name: int(cam.pool[name].ctypes.data) for name in output_names
        }
        trace = result.trace
        local = {
            "rank": world.Get_rank(),
            "native_available": process.native_available,
            "temperature_unchanged": _hash(temperature) == temperature_hash,
            "output_addresses_stable": all(
                int(cam.pool[name].ctypes.data) == output_addresses[name]
                for name in output_names
            ),
            "outputs_finite": all(
                bool(np.isfinite(cam.pool[name]).all()) for name in output_names
            ),
            "fice_min": float(cam.pool[binding["fice"]].min()),
            "fice_max": float(cam.pool[binding["fice"]].max()),
            "fsnow_min": float(cam.pool[binding["fsnow"]].min()),
            "fsnow_max": float(cam.pool[binding["fsnow"]].max()),
            "trace_phase": trace.phase,
            "trace_operation": trace.operation,
            "created_fields": process.created_fields,
        }
        result.remove()
        local["outputs_removed"] = all(name not in cam.pool for name in output_names)
        cam.advance(args.steps)
        local["final_step"] = cam.clock.nstep

    records = world.gather(local, root=0)
    if world.Get_rank() == 0:
        manifest = json.loads(args.native_manifest.read_text())
        native = next(
            record
            for record in manifest["direct_kernels"]["kernels"]
            if record["name"] == "cloud_fraction_fice"
        )
        promoted_device = manifest["promoted_kernel_device"]
        summary = {
            "schema_version": 1,
            "process": "cloud_fraction_fice",
            "original_routine": "cloud_fraction::cldfrc_fice",
            "mpi_ranks": world.Get_size(),
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "steps_after_probe": args.steps,
            "generated_symbol": native["symbol"],
            "original_call_proof": native["original_call_proof"],
            "library_sha256": manifest["library_sha256"],
            "promoted_library_sha256": promoted_device["library_sha256"],
            "promoted_library_load_policy": promoted_device["load_policy"],
            "all_ranks_native_available": all(
                record["native_available"] for record in records
            ),
            "all_temperature_inputs_unchanged": all(
                record["temperature_unchanged"] for record in records
            ),
            "all_output_addresses_stable": all(
                record["output_addresses_stable"] for record in records
            ),
            "all_outputs_finite": all(
                record["outputs_finite"] for record in records
            ),
            "all_outputs_removed": all(
                record["outputs_removed"] for record in records
            ),
            "all_final_steps": sorted({record["final_step"] for record in records}),
            "rank0_output_range": {
                key: records[0][key]
                for key in ("fice_min", "fice_max", "fsnow_min", "fsnow_max")
            },
            "rank0_created_fields": list(records[0]["created_fields"]),
        }
        required = (
            "all_ranks_native_available",
            "all_temperature_inputs_unchanged",
            "all_output_addresses_stable",
            "all_outputs_finite",
            "all_outputs_removed",
        )
        failures = tuple(name for name in required if not summary[name])
        if failures or summary["all_final_steps"] != [args.steps]:
            raise RuntimeError(f"promoted StatePool validation failed: {failures}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
