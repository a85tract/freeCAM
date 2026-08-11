#!/usr/bin/env python3
"""Prove that admitted CAM actions no longer depend on a hidden call cursor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mpi4py import MPI

from freecam.pi_cam import PICAMCase, ReplayBoundaryProvider
from freecam.pi_cam.native import NativeCAMDevice


REORDERED_ACTIONS = (
    ("cam_run1", "deep_convection"),
    ("cam_run1", "dry_adjustment"),
    ("cam_run2", "gravity_wave_drag"),
    ("cam_run2", "vertical_diffusion"),
)

REORDERED_LEAVES = (
    ("cam_run1", "cloud_diagnostics_leaf"),
    ("cam_run1", "state_and_convection_diagnostics_leaf"),
    ("cam_run2", "carma_statistics_leaf"),
    ("cam_run2", "aerosol_dry_deposition_leaf"),
    ("cam_run4", "flush_leaf"),
    ("cam_run4", "step_cost_leaf"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
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
        traces = []
        for phase, name in REORDERED_ACTIONS:
            traces.append(
                cam.run_action(name, phase=phase, experimental=True)
            )

        cam.expand_cam_run1_leaves(experimental=True)
        cam.expand_cam_run2_run4_leaves(experimental=True)
        for phase, name in REORDERED_LEAVES:
            traces.append(
                cam.run_action(name, phase=phase, experimental=True)
            )

        local = {
            "rank": world.rank,
            "operations": [trace.operation for trace in traces],
            "native_ids": [trace.native_id for trace in traces],
            "final_step": cam.clock.nstep,
            "leaf_device_loaded": bool(backend.leaf_loaded),
        }

    records = world.gather(local, root=0)
    if world.rank == 0:
        operations = records[0]["operations"]
        if any(record["operations"] != operations for record in records):
            raise RuntimeError("MPI ranks executed different reordered actions")
        if any(record["final_step"] != 0 for record in records):
            raise RuntimeError("isolated process calls unexpectedly advanced time")
        if not all(record["leaf_device_loaded"] for record in records):
            raise RuntimeError("not every MPI rank loaded the leaf device")
        payload = {
            "schema_version": 1,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "mpi_ranks": world.size,
            "model_steps_advanced": 0,
            "reordered_actions": [
                {"phase": phase, "name": name}
                for phase, name in (*REORDERED_ACTIONS, *REORDERED_LEAVES)
            ],
            "native_operations": operations,
            "all_ranks_completed": len(records) == world.size,
            "all_ranks_loaded_leaf_device": all(
                record["leaf_device_loaded"] for record in records
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
