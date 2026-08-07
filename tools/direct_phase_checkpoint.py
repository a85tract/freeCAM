"""Run one CAM phase directly, without the SegmentPlan executor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from freecam.core.runtime_env import mpi_loader_environment
from freecam.model import ModelConfig, read_checkpoint, restore_driver
from freecam.model.comm import world_comm


def _ensure_direct_mpi_loader_environment() -> None:
    """Re-exec this script, rather than the package CLI, when MPI needs it."""

    try:
        from mpi4py import MPI  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        if (
            "libmpi.so" not in str(exc)
            or os.environ.get("FREECAM_MPI_ENV_READY")
        ):
            raise RuntimeError(f"mpi4py cannot load the MPI runtime: {exc}") from exc
    else:
        return

    environment = mpi_loader_environment()
    environment["FREECAM_MPI_ENV_READY"] = "1"
    os.execve(
        sys.executable,
        [
            sys.executable,
            str(Path(__file__).resolve()),
            *sys.argv[1:],
        ],
        environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    config = ModelConfig.from_yaml(args.config)
    _ensure_direct_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_size:
        raise SystemExit(
            f"configuration requires {config.mpi_size} MPI ranks, got {comm.size}"
        )

    run_dir = Path(args.run_dir).resolve()
    if comm.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(
            Path(args.initial_run_dir).resolve() / "atm_in",
            run_dir / "atm_in",
        )
    comm.barrier()

    snapshot = read_checkpoint(args.input_checkpoint, comm)
    driver = restore_driver(
        snapshot,
        run_dir=run_dir,
        comm=comm,
        kernel_library=args.library,
        history_dir=args.history_dir,
        expected_config=config,
    )
    calls_before = driver.backend.call_count
    driver.run_phase(args.phase)
    comm.barrier()
    checkpoint = driver.write_checkpoint(args.output_checkpoint)
    result = driver.stats()
    result.update(
        phase=args.phase,
        native_calls_delta=driver.backend.call_count - calls_before,
        checkpoint=str(checkpoint),
        pbs_job_id=os.environ.get("PBS_JOBID"),
    )
    driver.finalize()

    if comm.rank == 0:
        encoded = json.dumps(result, indent=2, sort_keys=True)
        result_path = Path(args.result_json).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(encoded)
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
