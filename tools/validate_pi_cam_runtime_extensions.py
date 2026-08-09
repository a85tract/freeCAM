#!/usr/bin/env python3
"""Validate dynamic PI-CAM fields plus Python and Fortran runtime processes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from mpi4py import MPI
import numpy as np

from freecam.pi_cam import (
    NativeCAMDevice,
    PICAMCase,
    PICAMVariableSpec,
    ReplayBoundaryProvider,
)


def _array_hash(values: np.ndarray) -> str:
    return sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def tracer_source(fields, context, *, rate: float) -> None:
    """Trusted Notebook-style callback executed on each rank-local array."""

    fields["tracer"][...] += rate * context.timestep_seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # CAM changes into run_dir while its native runtime is live.  Resolve the
    # validation destination first so the evidence always lands in the repo.
    output_path = args.output.expanduser().resolve()

    comm = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    cam = case.runtime(
        boundary=ReplayBoundaryProvider(args.boundary),
        backend=NativeCAMDevice(args.native_manifest),
        communicator=comm,
        run_dir=args.run_dir,
    )
    cam.initialize()
    try:
        tracer = cam.define_variable(
            PICAMVariableSpec(
                "experiment_tracer",
                ("pcols", "pver", "chunks"),
                units="kg kg-1",
                initial=0.0,
                aliases=("tracer",),
                standard_name="experiment_tracer",
            )
        )
        temperature = cam.define_variable(
            PICAMVariableSpec(
                "runtime_temperature",
                ("nphys_local", "pver"),
                units="K",
                initial=240.0,
                standard_name="runtime_plugin_temperature",
            )
        )
        cam.define_variable(
            PICAMVariableSpec(
                "runtime_temperature_increment",
                (),
                units="K",
                initial=1.5,
                writable=False,
                standard_name="runtime_plugin_temperature_increment",
            )
        )
        addresses = {
            name: int(cam.pool[name].ctypes.data)
            for name in (
                "experiment_tracer",
                "runtime_temperature",
                "runtime_temperature_increment",
            )
        }

        python_process = cam.physics.install_python(
            tracer_source,
            name="runtime_tracer_source",
            phase="cam_run1",
            after="dadadj",
            writes=("tracer",),
            parameters={"rate": 1.0e-6},
        )
        fortran_process = cam.physics.install_fortran(
            args.device,
            project_root=args.project_root,
            process="runtime_temperature_offset",
            phase="cam_run1",
            after="dadadj",
            unsafe=True,
        )
        python_process.move(after="runtime_temperature_offset")
        python_process.disable()
        python_process.enable()
        fortran_process.disable()
        fortran_process.enable()

        cam.advance(args.steps)

        expected_tracer = 0.0
        for _ in range(args.steps):
            expected_tracer += 1.0e-6 * cam.clock.dt_seconds
        expected_temperature = 240.0
        for _ in range(args.steps):
            expected_temperature += 1.5
        local_ok = bool(
            np.array_equal(tracer, np.full_like(tracer, expected_tracer))
            and np.array_equal(
                temperature, np.full_like(temperature, expected_temperature)
            )
            and all(
                int(cam.pool[name].ctypes.data) == address
                for name, address in addresses.items()
            )
        )
        all_ok = bool(comm.allreduce(local_ok, op=MPI.LAND))
        if not all_ok:
            raise RuntimeError("runtime extension arrays or pointers differ")
        rank_record = {
            "rank": comm.rank,
            "tracer_shape": tuple(int(value) for value in tracer.shape),
            "temperature_shape": tuple(int(value) for value in temperature.shape),
            "tracer_hash": _array_hash(tracer),
            "temperature_hash": _array_hash(temperature),
            "tracer_min": float(tracer.min()),
            "tracer_max": float(tracer.max()),
            "temperature_min": float(temperature.min()),
            "temperature_max": float(temperature.max()),
            "addresses_unchanged": local_ok,
        }
        rank_records = comm.gather(rank_record, root=0)
        python_payload_hash = cam.python_processes.installed[
            "runtime_tracer_source"
        ].spec.payload_hash

        python_process.remove()
        fortran_process.remove()
        for name in (
            "experiment_tracer",
            "runtime_temperature",
            "runtime_temperature_increment",
        ):
            cam.delete_variable(name)
        if comm.rank == 0:
            payload = {
                "schema_version": 1,
                "pbs_job_id": os.environ.get("PBS_JOBID"),
                "mpi_ranks": comm.size,
                "steps": args.steps,
                "fine_grained_actions_per_step": len(tuple(cam.step_plan)),
                "python_process": "runtime_tracer_source",
                "python_payload_hash": python_payload_hash,
                "fortran_process": "runtime_temperature_offset",
                "runtime_extensions_passed": all_ok,
                "rank_count": len(rank_records),
                "tracer_shapes": sorted(
                    {tuple(record["tracer_shape"]) for record in rank_records}
                ),
                "temperature_shapes": sorted(
                    {tuple(record["temperature_shape"]) for record in rank_records}
                ),
                "tracer_hashes": sorted(
                    {str(record["tracer_hash"]) for record in rank_records}
                ),
                "temperature_hashes": sorted(
                    {str(record["temperature_hash"]) for record in rank_records}
                ),
                "tracer_min": min(float(record["tracer_min"]) for record in rank_records),
                "tracer_max": max(float(record["tracer_max"]) for record in rank_records),
                "temperature_min": min(
                    float(record["temperature_min"]) for record in rank_records
                ),
                "temperature_max": max(
                    float(record["temperature_max"]) for record in rank_records
                ),
                "all_addresses_unchanged": all(
                    bool(record["addresses_unchanged"]) for record in rank_records
                ),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2) + "\n")
    finally:
        cam.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
