"""Command-line entry points for the Python-owned CAM model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .core.mpi import world_comm
from .core.runtime_env import ensure_mpi_loader_environment
from .model import CAMDriver, ModelConfig
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


def command_build_kernels(_args: argparse.Namespace) -> int:
    subprocess.run(
        ["make", "-C", str(_repo_root() / "native" / "kernels"), "clean", "all"],
        check=True,
    )
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

    build = sub.add_parser("build-kernels", help="build stateless model kernels")
    build.set_defaults(func=command_build_kernels)

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
