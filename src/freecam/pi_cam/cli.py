"""MPI command line for one PI-CAM-only model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback
from dataclasses import replace

import cloudpickle

from .boundary import (
    CAMBoundaryProvider,
    CESMOnlineBoundaryProvider,
    OnlineBoundaryProvider,
    ReplayBoundaryProvider,
)
from .case import PICAMCase
from .native import NativeCAMDevice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freecam")
    parser.add_argument("--config", type=Path, required=True)
    boundary_group = parser.add_mutually_exclusive_group(required=True)
    boundary_group.add_argument("--boundary", type=Path)
    boundary_group.add_argument("--boundary-provider", type=Path)
    boundary_group.add_argument("--online-bootstrap", type=Path)
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
    parser.add_argument(
        "--expand-cam-run1-leaves",
        action="store_true",
        help=(
            "replace three cam_run1 composites with ordered native leaf actions"
        ),
    )
    parser.add_argument(
        "--expand-cam-run2-run4-leaves",
        action="store_true",
        help=(
            "replace admitted cam_run2 and cam_run4 composites with ordered "
            "native leaf actions"
        ),
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    if args.execution_mode is not None:
        case = PICAMCase(replace(case.config, execution_mode=args.execution_mode))
    if args.boundary is not None:
        boundary: CAMBoundaryProvider = ReplayBoundaryProvider(args.boundary)
    elif args.online_bootstrap is not None:
        boundary = OnlineBoundaryProvider.held(args.online_bootstrap)
    else:
        boundary = cloudpickle.loads(args.boundary_provider.read_bytes())
        if not isinstance(boundary, CAMBoundaryProvider):
            raise TypeError(
                "boundary provider payload is not a CAMBoundaryProvider"
            )
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
    if args.expand_cam_run1_leaves:
        cam.step_plan.expand_cam_run1_leaves(experimental=True)
    if args.expand_cam_run2_run4_leaves:
        cam.step_plan.expand_cam_run2_run4_leaves(experimental=True)
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
        operation_counts = Counter(trace.operation for trace in cam.trace)
        leaf_operations = tuple(
            action.operation
            for action in cam.step_plan.actions
            if action.operation.startswith("leaf_")
        )
        local = {
            "rank": world.Get_rank(),
            "step": cam.clock.nstep,
            "date": cam.clock.yyyymmdd,
            "seconds": cam.clock.seconds,
            "fields": len(cam.pool),
            "state_bytes": cam.pool.nbytes,
            "actions": len(cam.trace),
            "step_plan_actions": len(tuple(cam.step_plan)),
            "action_catalog_size": len(cam.step_plan.actions),
            "cam_run1_actions": len(cam.step_plan.in_phase("cam_run1")),
            "cam_run1_catalog_size": sum(
                action.phase == "cam_run1" for action in cam.step_plan.actions
            ),
            "cam_run2_actions": len(cam.step_plan.in_phase("cam_run2")),
            "cam_run2_catalog_size": sum(
                action.phase == "cam_run2" for action in cam.step_plan.actions
            ),
            "cam_run3_actions": len(cam.step_plan.in_phase("cam_run3")),
            "cam_run3_catalog_size": sum(
                action.phase == "cam_run3" for action in cam.step_plan.actions
            ),
            "cam_run4_actions": len(cam.step_plan.in_phase("cam_run4")),
            "cam_run4_catalog_size": sum(
                action.phase == "cam_run4" for action in cam.step_plan.actions
            ),
            "expanded_cam_run1_leaves": args.expand_cam_run1_leaves,
            "expanded_cam_run2_run4_leaves": (
                args.expand_cam_run2_run4_leaves
            ),
            "leaf_operation_counts": {
                name: operation_counts[name] for name in leaf_operations
            },
            "native_leaf_loaded": bool(
                getattr(cam.backend, "leaf_loaded", False)
            ),
            "created_fields": len(created_addresses),
            "python_initialized_fields": len(python_initialized_addresses),
            "stable_preinitialized_addresses": len(stable_names),
            "addresses_unchanged": not changed_addresses,
            "boundary": dict(getattr(boundary, "diagnostics", {})),
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
            leaf_device = manifest_payload.get("leaf_device", {})
            native_evidence = {
                "native_manifest": str(manifest_path),
                "native_library_sha256": manifest_payload.get("library_sha256"),
                "native_state_ownership": (
                    state_bridge.get("ownership")
                    if isinstance(state_bridge, dict)
                    else None
                ),
                "native_leaf_library_sha256": (
                    leaf_device.get("library_sha256")
                    if isinstance(leaf_device, dict)
                    else None
                ),
            }
        boundary_diagnostics = records[0]["boundary"]
        exact_oracle_bfb = (
            isinstance(boundary, CESMOnlineBoundaryProvider)
            and boundary.oracle is not None
            and all(
                record["boundary"].get("imports")
                == record["boundary"].get("oracle_x2a_bfb_steps")
                and record["boundary"].get("exports")
                == record["boundary"].get("oracle_a2x_bfb_steps")
                for record in records
            )
        )
        summary = {
            "schema_version": 1,
            "run_status": "passed",
            "case": case.config.case_name,
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "mpi_ranks": world.Get_size(),
            "steps": args.steps if args.steps is not None else case.config.stop_n,
            "execution_mode": case.config.execution_mode,
            "boundary_mode": case.config.boundary_mode,
            "boundary_provider": type(boundary).__name__,
            "validation_scope": (
                "online original-CESM components/coupler with per-boundary BFB oracle"
                if isinstance(boundary, CESMOnlineBoundaryProvider)
                else "online boundary control path with held-surface technical model"
                if isinstance(boundary, OnlineBoundaryProvider)
                and type(boundary.update).__name__ == "HeldSurfaceModel"
                else "configured CAM runtime"
            ),
            "scientific_bfb": True if exact_oracle_bfb else None,
            "boundary_rank_zero": boundary_diagnostics,
            "all_rank_generated_imports": [
                record["boundary"].get("generated_imports") for record in records
            ],
            "step_plan_actions": records[0]["step_plan_actions"],
            "action_catalog_size": records[0]["action_catalog_size"],
            "cam_run1_actions": records[0]["cam_run1_actions"],
            "cam_run1_catalog_size": records[0]["cam_run1_catalog_size"],
            "cam_run2_actions": records[0]["cam_run2_actions"],
            "cam_run2_catalog_size": records[0]["cam_run2_catalog_size"],
            "cam_run3_actions": records[0]["cam_run3_actions"],
            "cam_run3_catalog_size": records[0]["cam_run3_catalog_size"],
            "cam_run4_actions": records[0]["cam_run4_actions"],
            "cam_run4_catalog_size": records[0]["cam_run4_catalog_size"],
            "expanded_cam_run1_leaves": records[0][
                "expanded_cam_run1_leaves"
            ],
            "expanded_cam_run2_run4_leaves": records[0][
                "expanded_cam_run2_run4_leaves"
            ],
            "leaf_operation_counts": records[0]["leaf_operation_counts"],
            "all_ranks_loaded_leaf_device": all(
                record["native_leaf_loaded"] for record in records
            ),
            "ranks_loaded_leaf_device": sum(
                bool(record["native_leaf_loaded"]) for record in records
            ),
            "rank_trace_actions": [record["actions"] for record in records],
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
        try:
            from mpi4py import MPI
        except ImportError:
            raise
        MPI.COMM_WORLD.Abort(1)
        raise
