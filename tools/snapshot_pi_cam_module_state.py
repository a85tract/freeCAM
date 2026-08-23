#!/usr/bin/env python3
"""Record the module state a physics routine reads, from an initialized model.

A standalone image cannot run cam_init, so every module variable the routine
depends on is poked or initialized and then verified against this snapshot,
taken on rank 0 of a fully initialized 512-rank model.  The values are read
again after one timestep to confirm they are init-time constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mpi4py import MPI

from freecam.physics.image import read_module_state, state_digest
from freecam.physics.spec import load_function_spec
from freecam.pi_cam.boundary import ReplayBoundaryProvider
from freecam.pi_cam.case import PICAMCase
from freecam.pi_cam.native import NativeCAMDevice


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--function", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("validation"))
    args = parser.parse_args()

    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    backend = NativeCAMDevice(args.native_manifest)
    specs = {name: load_function_spec(name) for name in args.function}
    with case.runtime(
        boundary=ReplayBoundaryProvider(args.boundary),
        backend=backend,
        communicator=world,
        run_dir=args.run_dir,
    ) as cam:
        cam.initialize()
        library = backend._library
        initial = {name: read_module_state(library, spec.module_state) for name, spec in specs.items()}
        cam.advance(1)
        later = {name: read_module_state(library, spec.module_state) for name, spec in specs.items()}

    if world.Get_rank() != 0:
        return 0
    manifest = json.loads(args.native_manifest.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in specs.items():
        drifted = sorted(
            symbol for symbol in initial[name]
            if initial[name][symbol]["sha256"] != later[name][symbol]["sha256"]
        )
        record = {
            "schema_version": 1,
            "function": name,
            "qualified_name": spec.qualified_name,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "mpi_ranks": world.Get_size(),
            "config": str(args.config),
            "atm_in_sha256": _sha256(args.run_dir / "atm_in"),
            "native_manifest": str(args.native_manifest),
            "library_sha256": manifest.get("library_sha256"),
            "spec_sha256": _sha256(spec.path) if spec.path else None,
            "entries": initial[name],
            "digest": state_digest(initial[name]),
            "stable_after_one_step": not drifted,
            "drifted_after_one_step": drifted,
        }
        path = args.output_dir / f"pi_cam_{name}_module_state.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"{name}: {len(initial[name])} entries, stable={not drifted} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
