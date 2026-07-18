from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .config import CaseConfig
from .driver import FKesslerDriver
from .full_driver import FullCAMDriver
from .history_compare import compare_history, history_manifest
from .mpi_runtime import world_comm
from .native import NativeKesslerBackend, RecordingBackend
from .runtime_env import ensure_mpi_loader_environment
from .snapshot import NpzSnapshotWriter


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_run(args: argparse.Namespace) -> int:
    config = CaseConfig.from_yaml(args.config)
    ensure_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_ranks and not args.allow_rank_mismatch:
        raise SystemExit(
            f"configuration requires {config.mpi_ranks} MPI ranks, got {comm.size}; "
            "use mpiexec -n 24 or --allow-rank-mismatch for a local kernel smoke"
        )
    backend = (
        RecordingBackend()
        if args.backend == "recording"
        else NativeKesslerBackend(config.native.kessler_library)
    )
    driver = FKesslerDriver(config, comm, backend=backend)
    for field in args.watch:
        def show(context, field=field):
            array = context.state.require(field)
            if context.rank == 0:
                if args.watch_mode == "values":
                    print(f"event={args.watch_event} step={context.step} field={field} value={array!r}")
                else:
                    print(
                        f"event={args.watch_event} step={context.step} field={field} "
                        f"shape={array.shape} min={array.min():.17g} max={array.max():.17g}"
                    )
        driver.observe(args.watch_event, show, access="readonly")
    if args.snapshot_dir:
        driver.observe(
            args.snapshot_event,
            NpzSnapshotWriter(args.snapshot_dir, tuple(args.snapshot_field)),
            access="readonly",
        )
    driver.initialize()
    driver.run(args.steps)
    driver.finalize()
    if driver.comm.rank == 0:
        print(f"completed steps={driver.clock.step} backend={args.backend} dynamics=identity")
    return 0


def command_run_full(args: argparse.Namespace) -> int:
    config = CaseConfig.from_yaml(args.config)
    ensure_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_ranks and not args.allow_rank_mismatch:
        raise SystemExit(
            f"configuration requires {config.mpi_ranks} MPI ranks, got {comm.size}; "
            "run the full backend with mpiexec -n 24"
        )
    library = args.library or config.native.se_library
    driver = FullCAMDriver(
        config,
        comm,
        library=library,
        run_dir=args.run_dir,
    )
    for field in args.watch:
        def show(context, field=field):
            array = context.state.require(field)
            if context.rank == 0:
                if args.watch_mode == "values":
                    print(
                        f"event={args.watch_event} step={context.step} "
                        f"field={field} value={array!r}"
                    )
                else:
                    print(
                        f"event={args.watch_event} step={context.step} field={field} "
                        f"shape={array.shape} min={array.min():.17g} max={array.max():.17g}"
                    )
        driver.observe(args.watch_event, show, access="readonly")
    if args.snapshot_dir:
        driver.observe(
            args.snapshot_event,
            NpzSnapshotWriter(args.snapshot_dir, tuple(args.snapshot_field)),
            access="readonly",
        )
    driver.initialize()
    driver.run(args.steps)
    driver.finalize()
    if comm.rank == 0:
        print(f"completed steps={driver.clock.step} backend=full-native dynamics=se")
    return 0


def command_build_native(args: argparse.Namespace) -> int:
    root = _repo_root()
    build = root / "build" / "native"
    compiler = shutil.which("gfortran")
    if compiler is None:
        raise SystemExit("gfortran is required to build the CAM-SIMA kernels")
    subprocess.run(
        ["cmake", "--fresh", "-S", str(root / "native"), "-B", str(build),
         "-G", "Ninja", f"-DCMAKE_Fortran_COMPILER={compiler}"],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build), "-j"], check=True)
    return 0


def command_inspect_contract(args: argparse.Namespace) -> int:
    contract = _repo_root() / "native" / "generated" / "contract.json"
    if not contract.is_file():
        raise SystemExit("contract is missing; run tools/generate_kessler_contract.py")
    print(json.dumps(json.loads(contract.read_text()), indent=2))
    return 0


def command_compare_history(args: argparse.Namespace) -> int:
    fields = args.field or ("T", "Q", "U", "V", "PS")
    result = compare_history(args.reference_dir, args.candidate_dir, fields=fields)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.bfb else 1


def command_history_manifest(args: argparse.Namespace) -> int:
    fields = args.field or ("T", "Q", "U", "V", "PS")
    manifest = history_manifest(args.history_dir, fields=fields)
    payload = json.dumps(manifest, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload)
    else:
        print(payload, end="")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="pycam-sima")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("--steps", type=int, default=None)
    run.add_argument("--backend", choices=("recording", "native"), default="native")
    run.add_argument("--dynamics", choices=("identity",), default="identity")
    run.add_argument("--allow-rank-mismatch", action="store_true")
    run.add_argument("--watch", action="append", default=[], metavar="FIELD")
    run.add_argument("--watch-event", default="step_end")
    run.add_argument("--watch-mode", choices=("summary", "values"), default="summary")
    run.add_argument("--snapshot-dir")
    run.add_argument("--snapshot-field", action="append", default=[])
    run.add_argument("--snapshot-event", default="step_end")
    run.set_defaults(func=command_run)

    full = sub.add_parser("run-full")
    full.add_argument("config")
    full.add_argument("--run-dir", required=True)
    full.add_argument("--library")
    full.add_argument("--steps", type=int, default=None)
    full.add_argument("--allow-rank-mismatch", action="store_true")
    full.add_argument("--watch", action="append", default=[], metavar="FIELD")
    full.add_argument("--watch-event", default="step_end")
    full.add_argument("--watch-mode", choices=("summary", "values"), default="summary")
    full.add_argument("--snapshot-dir")
    full.add_argument("--snapshot-field", action="append", default=[])
    full.add_argument("--snapshot-event", default="step_end")
    full.set_defaults(func=command_run_full)

    build = sub.add_parser("build-native")
    build.set_defaults(func=command_build_native)

    inspect = sub.add_parser("inspect-contract")
    inspect.set_defaults(func=command_inspect_contract)

    compare = sub.add_parser("compare-history")
    compare.add_argument("reference_dir")
    compare.add_argument("candidate_dir")
    compare.add_argument("--field", action="append")
    compare.set_defaults(func=command_compare_history)

    manifest = sub.add_parser("history-manifest")
    manifest.add_argument("history_dir")
    manifest.add_argument("--field", action="append")
    manifest.add_argument("--output")
    manifest.set_defaults(func=command_history_manifest)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
