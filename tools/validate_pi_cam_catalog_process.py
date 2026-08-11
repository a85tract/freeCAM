#!/usr/bin/env python3
"""Execute a newly generated catalog process, then preserve 50-step BFB."""

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


def _hash(values: np.ndarray) -> str:
    return sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(arguments.config)
    backend = NativeCAMDevice(arguments.native_manifest)
    with case.runtime(
        boundary=ReplayBoundaryProvider(arguments.boundary),
        backend=backend,
        communicator=world,
        run_dir=arguments.run_dir,
    ) as cam:
        cam.initialize()
        state_hash = _hash(cam.pool["phys_state.t"])
        result = cam.physics.calc_hltalt(t=250.0)
        local = {
            "rank": world.rank,
            "process": result.process.name,
            "qualified_name": result.process.qualified_name,
            "native_available": result.process.runnable,
            "hltalt_finite": bool(np.isfinite(result.hltalt).all()),
            "tterm_finite": bool(np.isfinite(result.tterm).all()),
            "state_unchanged": _hash(cam.pool["phys_state.t"]) == state_hash,
            "chunks": int(cam.pool.dimensions["chunks"]),
        }
        result.remove()
        cam.advance(arguments.steps)
        local["final_step"] = cam.clock.nstep

    records = world.gather(local, root=0)
    if world.rank == 0:
        report = {
            "schema_version": 1,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "mpi_ranks": world.size,
            "process": "calc_hltalt",
            "qualified_name": "wv_saturation::calc_hltalt",
            "steps_after_probe": arguments.steps,
            "all_ranks_native_available": all(
                item["native_available"] for item in records
            ),
            "all_outputs_finite": all(
                item["hltalt_finite"] and item["tterm_finite"]
                for item in records
            ),
            "all_state_unchanged": all(item["state_unchanged"] for item in records),
            "all_final_steps": sorted({item["final_step"] for item in records}),
            "rank0_chunks": records[0]["chunks"],
        }
        report["passed"] = bool(
            report["all_ranks_native_available"]
            and report["all_outputs_finite"]
            and report["all_state_unchanged"]
            and report["all_final_steps"] == [arguments.steps]
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        if not report["passed"]:
            raise RuntimeError(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
