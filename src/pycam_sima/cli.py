"""Command-line entry points for the Python-owned CAM model."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from .core.mpi import world_comm
from .core.runtime_env import ensure_mpi_loader_environment
from .model import (
    BranchSpec,
    CAMDriver,
    ModelConfig,
    SegmentPlan,
    execute_segment_plan,
    read_checkpoint,
    restore_driver,
)
from .model.checkpoint import CHECKPOINT_SCHEMA_VERSION
from .model.device_codegen import build_device
from .model.validation import compare_history_directories


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_run(args: argparse.Namespace) -> int:
    """Run the complete fixed-case CAM model."""

    config = ModelConfig.from_yaml(args.config)
    ensure_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_size:
        raise SystemExit(
            f"configuration requires {config.mpi_size} MPI ranks, got {comm.size}"
        )
    history_dir = Path(args.history_dir).resolve()
    exists = history_dir.exists() if comm.rank == 0 else None
    if comm.bcast(exists, root=0):
        raise SystemExit(
            f"refusing to replace existing history directory: {history_dir}"
        )

    driver = CAMDriver(
        config,
        run_dir=args.run_dir,
        comm=comm,
        kernel_library=args.library,
        history_dir=history_dir,
    ).start()
    initialized_native_calls = driver.backend.call_count
    initialized_abi_checked = driver.backend._abi_checked
    for _ in range(args.steps):
        driver.step()
    result = driver.stats()
    result.update(
        initialized_native_calls=initialized_native_calls,
        initialized_abi_checked=initialized_abi_checked,
    )
    driver.finalize()
    if comm.rank == 0:
        print(json.dumps(result, sort_keys=True))
    return 0


def command_run_segment(args: argparse.Namespace) -> int:
    """Run one restartable MPI segment without a persistent socket worker."""

    config = ModelConfig.from_yaml(args.config)
    ensure_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_size:
        raise SystemExit(
            f"configuration requires {config.mpi_size} MPI ranks, got {comm.size}"
        )

    if args.branch_spec and args.segment_plan:
        raise SystemExit("use --branch-spec or --segment-plan, not both")
    if args.segment_plan:
        payload = _read_json_collective(Path(args.segment_plan), comm)
        plan = SegmentPlan.from_mapping(payload)
        if args.steps is not None:
            raise SystemExit("use actions in --segment-plan or --steps, not both")
    elif args.branch_spec:
        payload = _read_json_collective(Path(args.branch_spec), comm)
        plan = BranchSpec.from_mapping(payload).to_segment_plan()
        if args.steps is not None:
            raise SystemExit("use steps in --branch-spec or --steps, not both")
    else:
        plan = BranchSpec(
            name=args.name,
            steps=1 if args.steps is None else args.steps,
        ).to_segment_plan()

    history_dir = Path(args.history_dir).resolve()
    if args.input_checkpoint:
        snapshot = read_checkpoint(args.input_checkpoint, comm)
        driver = restore_driver(
            snapshot,
            run_dir=args.run_dir,
            comm=comm,
            kernel_library=args.library,
            history_dir=history_dir,
            expected_config=config,
        )
    else:
        exists = history_dir.exists() if comm.rank == 0 else None
        if comm.bcast(exists, root=0):
            raise SystemExit(
                f"refusing to replace existing history directory: {history_dir}"
            )
        driver = CAMDriver(
            config,
            run_dir=args.run_dir,
            comm=comm,
            kernel_library=args.library,
            history_dir=history_dir,
        ).start()

    action_trace = execute_segment_plan(driver, plan)
    checkpoint = driver.write_checkpoint(args.output_checkpoint)
    result = driver.stats()
    result.update(
        branch=plan.name,
        segment_steps=plan.step_count,
        action_count=len(plan.actions),
        action_trace=action_trace,
        segment_plan=plan.as_dict(),
        checkpoint=str(checkpoint),
        checkpoint_schema=CHECKPOINT_SCHEMA_VERSION,
        pbs_job_id=os.environ.get("PBS_JOBID"),
    )
    driver.finalize()
    if comm.rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        if args.result_json:
            result_path = Path(args.result_json).resolve()
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(encoded)
        print(encoded)
    return 0


def _read_json_collective(path: Path, comm) -> dict:
    """Read one controller payload once and broadcast it to every MPI rank."""

    payload = None
    error = None
    if comm.rank == 0:
        try:
            payload = json.loads(path.read_text())
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error:
        raise SystemExit(f"cannot read segment payload {path}: {error}")
    payload = comm.bcast(payload, root=0)
    if not isinstance(payload, dict):
        raise SystemExit(f"segment payload must be a JSON object: {path}")
    return payload


def command_build_kernels(_args: argparse.Namespace) -> int:
    subprocess.run(
        ["make", "-C", str(_repo_root() / "native" / "kernels"), "clean", "all"],
        check=True,
    )
    return 0


def command_build_device(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root or _repo_root() / "build/devices")
    for descriptor in args.descriptors:
        manifest = build_device(
            descriptor,
            project_root=_repo_root(),
            output_root=output_root,
            compiler=args.compiler,
            fflags=shlex.split(args.fflags),
            ldflags=shlex.split(args.ldflags),
        )
        print(manifest)
    return 0


def command_compare_history(args: argparse.Namespace) -> int:
    compare_history_directories(
        args.reference_dir,
        args.candidate_dir,
        expected_files=args.files,
        expected_numeric_variables=args.numeric_variables,
    )
    print("BFB")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="pycam-sima")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the complete Python-owned CAM model")
    run.add_argument("config")
    run.add_argument("--run-dir", required=True)
    run.add_argument("--history-dir", required=True)
    run.add_argument("--library")
    run.add_argument("--steps", type=int, default=50)
    run.set_defaults(func=command_run)

    segment = sub.add_parser(
        "run-segment",
        help="run one checkpointed model segment for Dask fan-out",
    )
    segment.add_argument("config")
    segment.add_argument("--run-dir", required=True)
    segment.add_argument("--history-dir", required=True)
    segment.add_argument("--library")
    segment.add_argument("--input-checkpoint")
    segment.add_argument("--output-checkpoint", required=True)
    segment.add_argument("--branch-spec")
    segment.add_argument("--segment-plan")
    segment.add_argument("--name", default="segment")
    segment.add_argument("--steps", type=int)
    segment.add_argument("--result-json")
    segment.set_defaults(func=command_run_segment)

    build = sub.add_parser(
        "build-kernels",
        help="build main model kernels and generated Fortran devices",
    )
    build.set_defaults(func=command_build_kernels)

    device = sub.add_parser(
        "build-device",
        help="generate and build source-preserving CCPP Fortran devices",
    )
    device.add_argument("descriptors", nargs="+")
    device.add_argument("--output-root")
    device.add_argument(
        "--compiler", default="/opt/cray/pe/gcc/12.2.0/bin/gfortran"
    )
    device.add_argument(
        "--fflags",
        default=(
            "-O2 -march=znver3 -fPIC -ffp-contract=off -fno-fast-math "
            "-ffree-line-length-none -cpp"
        ),
    )
    device.add_argument(
        "--ldflags", default="-Wl,--as-needed -Wl,--no-undefined"
    )
    device.set_defaults(func=command_build_device)

    compare = sub.add_parser("compare-history", help="run the fixed BFB gate")
    compare.add_argument("reference_dir")
    compare.add_argument("candidate_dir")
    compare.add_argument("--files", type=int, default=51)
    compare.add_argument("--numeric-variables", type=int, default=26)
    compare.set_defaults(func=command_compare_history)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
