"""Validate that Dask reuses one live MPI model across many Actor calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import dask
from distributed import Client
import distributed

import pycam_sima
from pycam_sima import (
    DaskExperimentClient,
    ObserveFields,
    RunSteps,
    SegmentPlan,
)


def _compact_status(status: dict) -> dict:
    names = (
        "name",
        "running",
        "worker_host",
        "worker_pid",
        "ranks",
        "step",
        "native_calls",
        "mpi_launch_count",
        "launch_mode",
        "pbs_job_id",
        "outer_pbs_job_id",
        "run_dir",
        "history_dir",
        "log_path",
        "field_count",
    )
    return {name: status[name] for name in names if name in status}


def _compact_plan_result(result: dict) -> dict:
    trace = []
    for record in result["action_trace"]:
        compact = {
            name: record[name]
            for name in (
                "index",
                "type",
                "step_before",
                "step_after",
                "native_calls_delta",
                "last_phase",
                "last_scheme",
                "last_scheme_group",
            )
        }
        if "observations" in record:
            compact["observations"] = [
                {
                    "field": observation["field"],
                    "global": observation["global"],
                }
                for observation in record["observations"]
            ]
        trace.append(compact)
    return {
        "name": result["name"],
        "step": result["step"],
        "native_calls": result["native_calls"],
        "mpi_launch_count": result["mpi_launch_count"],
        "action_trace": trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    outer_job_id = os.environ.get("PBS_JOBID")
    if not outer_job_id:
        raise SystemExit("dask_persistent_smoke.py must run inside a PBS allocation")

    project = Path(__file__).resolve().parents[1]
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            config=project / "configs/fkessler_model.yaml",
            initial_run_dir=Path(args.initial_run_dir).resolve(),
            run_root=Path(args.run_root).resolve(),
            library=project / "build/libpycam_sima_kernels.so",
            environment_script=(
                project
                / "reference/cases/FKESSLER_ne3pg3_gnu_24x50"
                / ".env_mach_specific.sh"
            ),
            python_executable=project / ".venv/bin/python",
            execution_mode="allocation",
        )
        model = experiments.start_persistent("live")
        try:
            started = model.describe().result()
            first = model.step().result()
            observed = model.run_plan(
                SegmentPlan(
                    "second-command",
                    (
                        ObserveFields(("air_temperature",)),
                        RunSteps(1),
                    ),
                )
            ).result()
            field_stats = model.field_stats("air_temperature", rank=0).result()
            checkpoint = model.checkpoint().result()
            final_status = model.describe().result()
        finally:
            closed = model.close().result()

    launch_counts = {
        started["mpi_launch_count"],
        first["mpi_launch_count"],
        observed["mpi_launch_count"],
        checkpoint["mpi_launch_count"],
        final_status["mpi_launch_count"],
        closed["mpi_launch_count"],
    }
    if launch_counts != {1}:
        raise RuntimeError(f"persistent Actor relaunched MPI: {launch_counts}")
    if (
        started["step"],
        first["step"],
        observed["step"],
        final_status["step"],
    ) != (0, 1, 2, 2):
        raise RuntimeError("persistent Actor did not retain model clock state")
    if final_status["outer_pbs_job_id"] != outer_job_id:
        raise RuntimeError("persistent Actor escaped the outer PBS allocation")
    if final_status["pbs_job_id"] is not None:
        raise RuntimeError("persistent allocation mode submitted a nested PBS job")
    if not Path(checkpoint["checkpoint_dir"]).is_dir():
        raise RuntimeError("persistent checkpoint was not written")

    payload = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(project),
            "parent_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"),
                cwd=project,
                text=True,
            ).strip(),
            "working_tree_dirty": (
                subprocess.run(
                    ("git", "diff", "--quiet"),
                    cwd=project,
                    check=False,
                ).returncode
                != 0
            ),
        },
        "software": {
            "pycam_sima": pycam_sima.__version__,
            "dask": dask.__version__,
            "distributed": distributed.__version__,
        },
        "outer_pbs_job_id": outer_job_id,
        "dask_workers": 1,
        "mpi_ranks": 24,
        "mpi_launches": 1,
        "actor_method_calls": 7,
        "actor_worker": model.worker,
        "started": _compact_status(started),
        "first_step": _compact_status(first),
        "second_command": _compact_plan_result(observed),
        "field_stats_rank_0": field_stats,
        "checkpoint": checkpoint,
        "final_status": _compact_status(final_status),
        "closed": closed,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "PYCAM_SIMA_DASK_PERSISTENT_OK "
        f"job={outer_job_id} mpi_launches=1 actor_calls=7 step=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
