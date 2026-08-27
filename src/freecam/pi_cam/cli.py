"""MPI command line for one PI-CAM-only model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from dataclasses import replace
import resource

import cloudpickle

from .boundary import (
    CAMBoundaryProvider,
    CESMOnlineBoundaryProvider,
    OnlineBoundaryProvider,
    ReplayBoundaryProvider,
)
from .case import PICAMCase
from .errors import PICAMConfigurationError
from .namelist import (
    NamelistEntry,
    apply_overrides,
    coerce_text,
    load_catalog,
)
from .native import NativeCAMDevice


def _namelist_requests(items: list[str]) -> dict[str, str]:
    """Split repeatable ``NAME=VALUE`` flags, rejecting malformed ones."""

    requested: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name.strip():
            raise PICAMConfigurationError(
                f"--namelist expects NAME=VALUE, got {item!r}"
            )
        requested[name.strip().lower()] = value
    return requested


def _catalog_entry(
    catalog: dict[str, NamelistEntry], name: str
) -> NamelistEntry:
    entry = catalog.get(name)
    if entry is None:
        raise PICAMConfigurationError(
            f"unknown CAM namelist variable {name!r}"
        )
    return entry


def _process_memory(label: str, step: int) -> dict[str, int | str]:
    """Return one rank-local Linux memory sample in bytes."""

    sample: dict[str, int | str] = {"label": label, "step": int(step)}
    for path, names in (
        (
            Path("/proc/self/status"),
            {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"},
        ),
        (
            Path("/proc/self/smaps_rollup"),
            {"Pss", "Private_Clean", "Private_Dirty", "Shared_Clean", "Shared_Dirty"},
        ),
    ):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            name, separator, raw = line.partition(":")
            if not separator or name not in names:
                continue
            fields = raw.split()
            if fields:
                sample[f"{name}_bytes"] = int(fields[0]) * 1024
    # Linux reports ru_maxrss in KiB. Keep this independent source beside
    # VmHWM so long runs can diagnose parser or procfs availability issues.
    sample["ru_maxrss_bytes"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ) * 1024
    return sample


def _history_stream_request(text: str) -> dict[str, object]:
    """Read one history-stream request from JSON text or a JSON file."""

    # Inline JSON is longer than a file name may be, so probing the
    # filesystem first would raise instead of parsing.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        candidate = Path(text)
        if not candidate.is_file():
            raise ValueError(
                "a history stream request must be JSON text or a JSON file"
            ) from None
        payload = json.loads(candidate.read_text())
    if not isinstance(payload, dict):
        raise ValueError("a history stream request must be a JSON object")
    if "name" not in payload:
        raise ValueError("history stream request needs 'name'")
    return payload


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
    parser.add_argument(
        "--history-stream",
        action="append",
        default=[],
        dest="history_streams",
        metavar="JSON",
        help=(
            "add Python-owned fields to a CAM history stream from a JSON "
            "object or a path to one; Python-owned fields already join the "
            "model's own h0 output by default, so this only overrides that"
        ),
    )
    parser.add_argument(
        "--namelist",
        action="append",
        default=[],
        dest="namelist_overrides",
        metavar="NAME=VALUE",
        help=(
            "override one CAM namelist variable in the run directory's "
            "atm_in before initialization, validated against the pinned "
            "source's namelist definition; repeatable"
        ),
    )
    parser.add_argument(
        "--split-macrophysics",
        action="store_true",
        help=(
            "run the macrophysics stage as its two halves, stopped before "
            "macrop_driver_tend and resumed after it; with no Python process "
            "installed the resume calls the original driver, which gates the "
            "boundary alone"
        ),
    )
    parser.add_argument(
        "--split-radiation",
        action="store_true",
        help=(
            "run the radiation stage as its two halves, stopped before "
            "radiation_tend and resumed after it; with no Python process "
            "installed the resume calls the original driver, which gates the "
            "boundary alone"
        ),
    )
    parser.add_argument(
        "--macrophysics-python",
        action="store_true",
        help=(
            "install Macrophysics.tend on every rank between the two halves of "
            "the split macrophysics stage, so the driver layer runs as Python "
            "with every FLOP still in Fortran; requires --split-macrophysics"
        ),
    )
    parser.add_argument(
        "--radiation-python",
        action="store_true",
        help=(
            "install Radiation.tend on every rank between the two halves of "
            "the split radiation stage, so the driver layer runs as Python "
            "with every FLOP still in Fortran; requires --split-radiation"
        ),
    )
    parser.add_argument(
        "--cloud-macro-micro-python",
        action="store_true",
        help=(
            "disable the cloud_macro_microphysics action (tphysbc stage 7) and "
            "install CloudMacroMicrophysics.tend in its place on every rank, so "
            "the stage's driver layer runs as Python with every FLOP still in "
            "Fortran; the whole action is replaced, nothing is split"
        ),
    )
    parser.add_argument(
        "--cloud-macro-micro-whole-drivers",
        action="store_true",
        help=(
            "with --cloud-macro-micro-python, call macrop_driver_tend whole "
            "instead of running the Macrophysics sub-walk in its place (Gate M-1's form)"
        ),
    )
    parser.add_argument(
        "--cloud-macro-micro-whole-micro",
        action="store_true",
        help=(
            "with --cloud-macro-micro-python, call microp_driver_tend whole "
            "instead of running the Microphysics sub-walk in its place (Gate M-2's form)"
        ),
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--memory-sample-every",
        type=int,
        default=0,
        metavar="STEPS",
        help=(
            "record per-rank procfs memory at initialization, finalization, "
            "step 1, and every STEPS complete model steps"
        ),
    )
    args = parser.parse_args(argv)
    from mpi4py import MPI

    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    # Every rank validates the overrides (a rank-0-only failure between
    # barriers would hang the rest); only rank 0 rewrites the file.
    applied_namelist: dict[str, tuple[str | None, str]] = {}
    if args.namelist_overrides:
        requested = _namelist_requests(args.namelist_overrides)
        catalog = load_catalog(case.config.source_root)
        overrides = {
            name: coerce_text(text, _catalog_entry(catalog, name))
            for name, text in requested.items()
        }
        if world.rank == 0:
            applied_namelist = apply_overrides(
                Path(args.run_dir) / "atm_in", overrides, catalog
            )
        world.Barrier()
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
    history_streams = tuple(
        _history_stream_request(item) for item in args.history_streams
    )
    if args.expand_cam_run1_leaves:
        cam.step_plan.expand_cam_run1_leaves(experimental=True)
    if args.expand_cam_run2_run4_leaves:
        cam.step_plan.expand_cam_run2_run4_leaves(experimental=True)
    if args.split_macrophysics:
        cam.step_plan.split_macrophysics(experimental=True)
    if args.macrophysics_python and not args.split_macrophysics:
        raise SystemExit("--macrophysics-python requires --split-macrophysics")
    if args.split_radiation:
        cam.step_plan.split_radiation(experimental=True)
    if args.radiation_python and not args.split_radiation:
        raise SystemExit("--radiation-python requires --split-radiation")
    created_addresses = {
        name: int(values.ctypes.data) for name, values in cam.pool.items()
    }
    # Measure collective lifecycle boundaries, not rank 0's local arrival
    # time. These barriers do not alter any CAM numerical operation.
    world.Barrier()
    total_started = MPI.Wtime()
    memory_samples = [_process_memory("python_ready", cam.clock.nstep)]
    with cam:
        world.Barrier()
        initialize_started = MPI.Wtime()
        cam.initialize()
        if args.macrophysics_python:
            # The Python driver layer, installed collectively on every rank
            # once the model exists: a native, non-transactional process (it
            # owns no StatePool field; every array it touches is CAM's own
            # storage reached by handle) sitting between the two halves.
            from freecam.model.python_processes import PythonProcessSpec
            from freecam.physics.macrophysics import Macrophysics

            scheme = Macrophysics()
            cam.python_processes.install(
                PythonProcessSpec.from_callable(
                    scheme.tend,
                    name="macro_tend",
                    group="cam_run1",
                    after="macro_tend_pre_leaf",
                    native=True,
                    transactional=False,
                ),
                unsafe=True,
            )
        if args.radiation_python:
            # The same shape for radiation: Radiation.tend between the two
            # halves of the split stage, native and non-transactional for the
            # same reason -- it owns no StatePool field.
            from freecam.model.python_processes import PythonProcessSpec
            from freecam.physics.radiation import Radiation

            scheme = Radiation()
            cam.python_processes.install(
                PythonProcessSpec.from_callable(
                    scheme.tend,
                    name="rad_tend",
                    group="cam_run1",
                    after="rad_tend_pre_leaf",
                    native=True,
                    transactional=False,
                ),
                unsafe=True,
            )
        if args.cloud_macro_micro_python:
            # A whole workflow action rather than a call inside one: the
            # Fortran stage is disabled and the Python stage sits where it
            # was.  Native and non-transactional for the same reason as the
            # other two -- it owns no StatePool field.
            from freecam.model.python_processes import PythonProcessSpec
            from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics

            scheme = CloudMacroMicrophysics(
                whole_drivers=bool(args.cloud_macro_micro_whole_drivers),
                whole_micro=bool(args.cloud_macro_micro_whole_micro))
            cam.step_plan.set_enabled("cloud_macro_microphysics", False, phase="cam_run1", experimental=True)
            cam.python_processes.install(
                PythonProcessSpec.from_callable(
                    scheme.tend,
                    name="cloud_macro_microphysics_python",
                    group="cam_run1",
                    after="cloud_macro_microphysics",
                    native=True,
                    transactional=False,
                ),
                unsafe=True,
            )
        for request in history_streams:
            cam.install_history_stream(
                str(request["name"]),
                fields=(
                    None if request.get("fields") is None
                    else tuple(request["fields"])
                ),
                stream=str(request.get("stream", "h0")),
                nhtfrq=int(request.get("nhtfrq", 0)),
                before=request.get("before"),
                after=request.get("after", "wshist"),
                time_period=str(request.get("time_period", "mean")),
                precision=str(request.get("precision", "float32")),
            )
        memory_samples.append(_process_memory("initialized", cam.clock.nstep))
        world.Barrier()
        initialize_seconds = MPI.Wtime() - initialize_started
        python_initialized_addresses = cam.python_initialized_addresses
        initialized_addresses = {
            name: int(values.ctypes.data) for name, values in cam.pool.items()
        }
        world.Barrier()
        advance_started = MPI.Wtime()
        steps = args.steps if args.steps is not None else case.config.stop_n
        if args.memory_sample_every > 0:
            for completed in range(1, steps + 1):
                cam.step()
                if (
                    completed == 1
                    or completed == steps
                    or completed % args.memory_sample_every == 0
                ):
                    memory_samples.append(
                        _process_memory(f"step_{completed}", cam.clock.nstep)
                    )
        else:
            cam.advance(steps)
        world.Barrier()
        advance_seconds = MPI.Wtime() - advance_started
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
        operation_counts = cam.operation_counts
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
            "macrophysics_split": bool(args.split_macrophysics),
            "radiation_split": bool(args.split_radiation),
            "state_bytes": cam.pool.nbytes,
            "actions": cam.trace_count,
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
            "initialize_seconds": initialize_seconds,
            "advance_seconds": advance_seconds,
            "memory_samples": memory_samples,
        }
        memory_samples.append(_process_memory("pre_finalize", cam.clock.nstep))
        world.Barrier()
        finalize_started = MPI.Wtime()
    world.Barrier()
    local["finalize_seconds"] = MPI.Wtime() - finalize_started
    local["total_seconds"] = MPI.Wtime() - total_started
    memory_samples.append(_process_memory("finalized", cam.clock.nstep))
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
        steps = args.steps if args.steps is not None else case.config.stop_n
        memory_labels = tuple(
            sample["label"] for sample in records[0]["memory_samples"]
        )
        memory = {
            "sample_every_steps": args.memory_sample_every,
            "rank_count": world.Get_size(),
            "samples": [
                {
                    "label": label,
                    "step": records[0]["memory_samples"][index]["step"],
                    "total_rss_bytes": sum(
                        int(record["memory_samples"][index].get("VmRSS_bytes", 0))
                        for record in records
                    ),
                    "maximum_rank_rss_bytes": max(
                        int(record["memory_samples"][index].get("VmRSS_bytes", 0))
                        for record in records
                    ),
                    "total_pss_bytes": sum(
                        int(record["memory_samples"][index].get("Pss_bytes", 0))
                        for record in records
                    ),
                    "total_hwm_bytes": sum(
                        int(record["memory_samples"][index].get("VmHWM_bytes", 0))
                        for record in records
                    ),
                    "maximum_rank_hwm_bytes": max(
                        int(record["memory_samples"][index].get("VmHWM_bytes", 0))
                        for record in records
                    ),
                    "maximum_rank_ru_maxrss_bytes": max(
                        int(
                            record["memory_samples"][index].get(
                                "ru_maxrss_bytes", 0
                            )
                        )
                        for record in records
                    ),
                }
                for index, label in enumerate(memory_labels)
            ],
        }
        timing = {
            "clock": "MPI.Wtime with MPI barriers",
            "aggregation": "maximum across all MPI ranks",
            "initialize_seconds": max(
                record["initialize_seconds"] for record in records
            ),
            "advance_seconds": max(record["advance_seconds"] for record in records),
            "finalize_seconds": max(
                record["finalize_seconds"] for record in records
            ),
            "total_seconds": max(record["total_seconds"] for record in records),
        }
        simulated_days = steps * case.config.timestep_seconds / 86400.0
        timing["simulated_days"] = simulated_days
        timing["advance_sypd"] = (
            simulated_days / 365.0 * 86400.0 / timing["advance_seconds"]
        )
        summary = {
            "schema_version": 1,
            "run_status": "passed",
            "case": case.config.case_name,
            "applied_namelist_overrides": {
                name: {"previous": old, "value": new}
                for name, (old, new) in applied_namelist.items()
            },
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "mpi_ranks": world.Get_size(),
            "steps": steps,
            "execution_mode": case.config.execution_mode,
            "macrophysics_split": bool(args.split_macrophysics),
            "macrophysics_python": bool(args.macrophysics_python),
            "radiation_split": bool(args.split_radiation),
            "radiation_python": bool(args.radiation_python),
            "cloud_macro_micro_python": bool(args.cloud_macro_micro_python),
            "cloud_macro_micro_whole_drivers": bool(args.cloud_macro_micro_whole_drivers),
            "cloud_macro_micro_whole_micro": bool(args.cloud_macro_micro_whole_micro),
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
            "timing": timing,
            "memory": memory,
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
        # complete validation communicator.  It also leaves the diagnostic in
        # the run directory: an abort can kill a rank before its stderr is
        # flushed, and the message that names the first differing field is
        # the one worth keeping.
        traceback.print_exc()
        sys.stderr.flush()
        try:
            run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
            rank = os.environ.get("PMI_RANK") or os.environ.get("PMIX_RANK") or "unknown"
            (run_dir / f"error.rank-{rank}.json").write_text(json.dumps(
                {"rank": rank, "traceback": traceback.format_exc()}, indent=2))
        except Exception:
            pass
        try:
            from mpi4py import MPI
        except ImportError:
            raise
        MPI.COMM_WORLD.Abort(1)
        raise
