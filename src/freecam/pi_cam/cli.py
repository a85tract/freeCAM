"""MPI command line for one PI-CAM-only model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from dataclasses import replace

from mpi4py import MPI

from .boundary import ReplayBoundaryProvider
from .case import PICAMCase
from .native import NativeCAMDevice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m freecam.pi_cam.cli")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--native-manifest",
        type=Path,
        help="override native_manifest from the case YAML",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--execution-mode",
        choices=("fine_grained", "source_compat"),
        help="override the YAML control-path mode for validation",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    if args.execution_mode is not None:
        case = PICAMCase(replace(case.config, execution_mode=args.execution_mode))
    boundary = ReplayBoundaryProvider(args.boundary)
    backend = (
        None
        if args.native_manifest is None
        else NativeCAMDevice(args.native_manifest)
    )
    cam = case.runtime(
        boundary=boundary,
        backend=backend,
        communicator=world,
        run_dir=args.run_dir,
    )
    created_addresses = {
        name: int(values.ctypes.data) for name, values in cam.pool.items()
    }
    with cam:
        cam.initialize()
        python_initialized_addresses = cam.python_initialized_addresses
        initialized_addresses = {
            name: int(values.ctypes.data) for name, values in cam.pool.items()
        }
        cam.advance(args.steps)
        final_addresses = {
            name: int(values.ctypes.data) for name, values in cam.pool.items()
        }
        initial_names = set(python_initialized_addresses)
        stable_names = initial_names & set(initialized_addresses) & set(final_addresses)
        changed_addresses = tuple(
            sorted(
                name
                for name in stable_names
                if not (
                    python_initialized_addresses[name]
                    == initialized_addresses[name]
                    == final_addresses[name]
                )
            )
        )
        if changed_addresses:
            raise RuntimeError(
                "Python-owned PI-CAM arrays changed address: "
                + ", ".join(changed_addresses[:8])
            )
        local = {
            "rank": world.Get_rank(),
            "step": cam.clock.nstep,
            "date": cam.clock.yyyymmdd,
            "seconds": cam.clock.seconds,
            "fields": len(cam.pool),
            "state_bytes": cam.pool.nbytes,
            "actions": len(cam.trace),
            "created_fields": len(created_addresses),
            "python_initialized_fields": len(python_initialized_addresses),
            "stable_preinitialized_addresses": len(stable_names),
            "addresses_unchanged": not changed_addresses,
        }
    finalized_addresses = {
        name: int(values.ctypes.data) for name, values in cam.pool.items()
    }
    if any(
        finalized_addresses.get(name) != final_addresses[name]
        for name in final_addresses
    ):
        raise RuntimeError("native finalization changed Python-owned array addresses")
    records = world.gather(local, root=0)
    if world.Get_rank() == 0:
        manifest_path = (
            args.native_manifest
            if args.native_manifest is not None
            else case.config.native_manifest
        )
        native_evidence: dict[str, object] = {}
        if manifest_path is not None:
            manifest_path = Path(manifest_path).resolve()
            manifest_payload = json.loads(manifest_path.read_text())
            state_bridge = manifest_payload.get("state_bridge", {})
            native_evidence = {
                "native_manifest": str(manifest_path),
                "native_library_sha256": manifest_payload.get("library_sha256"),
                "native_state_ownership": (
                    state_bridge.get("ownership")
                    if isinstance(state_bridge, dict)
                    else None
                ),
            }
        summary = {
            "schema_version": 1,
            "case": case.config.case_name,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "mpi_ranks": world.Get_size(),
            "steps": args.steps if args.steps is not None else case.config.stop_n,
            "rank_state_bytes": [record["state_bytes"] for record in records],
            "rank_fields": [record["fields"] for record in records],
            "rank_created_fields": [record["created_fields"] for record in records],
            "rank_preinitialized_fields": [
                record["python_initialized_fields"] for record in records
            ],
            "rank_stable_preinitialized_addresses": [
                record["stable_preinitialized_addresses"] for record in records
            ],
            "addresses_unchanged": all(
                record["addresses_unchanged"] for record in records
            ),
            "final_date": records[0]["date"],
            "final_seconds": records[0]["seconds"],
            **native_evidence,
        }
        text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.summary is None:
            print(text, end="")
        else:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # A rank-local BFB or native-state failure must not leave the other
        # ranks blocked in their next CAM collective until PBS walltime.  The
        # failing rank prints the useful diagnostic first, then terminates the
        # complete validation communicator.
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
        raise
