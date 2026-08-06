#!/usr/bin/env python3
"""Run PI-CAM, then exercise a raw-array kernel without mutating the model."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from mpi4py import MPI
import numpy as np

from pycam_sima.pi_cam import PICAMCase, ReplayBoundaryProvider
from pycam_sima.pi_cam.native import NativeCAMDevice


def _hash(array: np.ndarray) -> str:
    return sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--kernel", default="dadadj")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path)
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    boundary = ReplayBoundaryProvider(args.boundary)
    backend = (
        None
        if args.native_manifest is None
        else NativeCAMDevice(args.native_manifest)
    )
    with case.runtime(
        boundary=boundary,
        backend=backend,
        communicator=world,
        run_dir=args.run_dir,
    ) as cam:
        cam.initialize()
        cam.advance(args.steps)
        if not isinstance(cam.backend, NativeCAMDevice):
            raise RuntimeError("direct-kernel validation requires NativeCAMDevice")
        names = cam.backend.kernel_fields(args.kernel)
        arrays = {
            name: cam.pool[name].copy(order="F")
            for name in names
        }
        if args.kernel == "dadadj":
            # Construct one mild, analytically unstable layer per local chunk
            # in the disposable probe arrays.  This proves the inout path even
            # when the evolved 50-step atmosphere is already dry stable.
            cappa = 0.28571658640413355
            ncols = arrays["grid.chunk_ncols"]
            pmid = arrays["phys_state.pmid"]
            pint = arrays["phys_state.pint"]
            temperature = arrays["phys_state.t"]
            probe_unstable_columns = 0
            for chunk, ncol in enumerate(ncols):
                if int(ncol) < 1:
                    continue
                for column in range(int(ncol)):
                    alpha = (
                        0.5
                        * cappa
                        * (pmid[column, 1, chunk] - pmid[column, 0, chunk])
                        / pint[column, 1, chunk]
                    )
                    temperature[column, 1, chunk] = (
                        temperature[column, 0, chunk]
                        * (1.0 + alpha)
                        / (1.0 - alpha)
                        + 1.0
                    )
                    dtdp = (
                        temperature[column, 1, chunk]
                        - temperature[column, 0, chunk]
                    ) / (pmid[column, 1, chunk] - pmid[column, 0, chunk])
                    gamma = (
                        cappa
                        * 0.5
                        * (
                            temperature[column, 1, chunk]
                            + temperature[column, 0, chunk]
                        )
                        / pint[column, 1, chunk]
                    )
                    probe_unstable_columns += int(dtdp + 2.0e-5 > gamma)
        else:
            probe_unstable_columns = 0
        addresses = {name: int(array.ctypes.data) for name, array in arrays.items()}
        before = {name: _hash(array) for name, array in arrays.items()}
        cam.backend.execute_kernel(args.kernel, arrays, fcomm=cam.fcomm)
        after = {name: _hash(array) for name, array in arrays.items()}
        address_stable = all(
            int(array.ctypes.data) == addresses[name]
            for name, array in arrays.items()
        )
        unchanged_inputs = all(
            before[name] == after[name]
            for name in names[:-2]
        )
        local = {
            "rank": world.Get_rank(),
            "address_stable": address_stable,
            "unchanged_inputs": unchanged_inputs,
            "temperature_changed": before["phys_state.t"] != after["phys_state.t"],
            "water_vapor_changed": before["phys_state.q"] != after["phys_state.q"],
            "probe_unstable_columns": probe_unstable_columns,
            "field_shapes": {name: list(array.shape) for name, array in arrays.items()},
        }

    records = world.gather(local, root=0)
    if world.Get_rank() == 0:
        manifest_path = args.native_manifest or case.config.native_manifest
        if manifest_path is None:
            raise RuntimeError("native manifest is unavailable")
        manifest = json.loads(Path(manifest_path).read_text())
        kernel_record = next(
            record
            for record in manifest["direct_kernels"]["kernels"]
            if record["name"] == args.kernel
        )
        summary = {
            "schema_version": 1,
            "kernel": args.kernel,
            "steps_before_probe": args.steps,
            "probe": "one mild synthetic dry instability per local chunk",
            "mpi_ranks": world.Get_size(),
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "library_sha256": manifest["library_sha256"],
            "generated_symbol": kernel_record["symbol"],
            "original_call_proof": kernel_record["original_call_proof"],
            "all_addresses_stable": all(record["address_stable"] for record in records),
            "all_readonly_inputs_unchanged": all(
                record["unchanged_inputs"] for record in records
            ),
            "ranks_with_temperature_change": sum(
                record["temperature_changed"] for record in records
            ),
            "ranks_with_water_vapor_change": sum(
                record["water_vapor_changed"] for record in records
            ),
            "probe_unstable_columns": sum(
                record["probe_unstable_columns"] for record in records
            ),
            "rank0_shapes": records[0]["field_shapes"],
        }
        if not summary["all_addresses_stable"] or not summary["all_readonly_inputs_unchanged"]:
            raise RuntimeError("direct kernel violated its pointer-table contract")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if summary["probe_unstable_columns"] < 1:
            raise RuntimeError("direct dadadj probe did not construct an instability")
        if summary["ranks_with_temperature_change"] < 1:
            raise RuntimeError("direct dadadj write path was not exercised")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
