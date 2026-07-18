from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from .config import CaseConfig
from .driver import FKesslerDriver
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

    build = sub.add_parser("build-native")
    build.set_defaults(func=command_build_native)

    inspect = sub.add_parser("inspect-contract")
    inspect.set_defaults(func=command_inspect_contract)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
