"""Validate the single-launch persistent MPI pool inside one PBS allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from distributed import Client
import numpy as np

import freecam
from freecam import DaskExperimentClient
from freecam.model.validation import compare_history_directories


def _snapshots_equal(left: Any, right: Any) -> tuple[bool, int]:
    left_payloads = left.rank_payloads()
    right_payloads = right.rank_payloads()
    if len(left_payloads) != len(right_payloads):
        return False, 0
    compared = 0
    for left_payload, right_payload in zip(left_payloads, right_payloads):
        with (
            np.load(BytesIO(left_payload[1]), allow_pickle=False) as lhs,
            np.load(BytesIO(right_payload[1]), allow_pickle=False) as rhs,
        ):
            if set(lhs.files) != set(rhs.files):
                return False, compared
            for name in lhs.files:
                if not np.array_equal(lhs[name], rhs[name]):
                    return False, compared
                compared += 1
    return True, compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--oracle-history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("persistent pool validation requires a PBS allocation")

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    output = Path(args.output).resolve()

    with Client(
        processes=False,
        n_workers=3,
        threads_per_worker=1,
        dashboard_address=None,
        local_directory=str(run_root / "dask-worker-space"),
    ) as client:
        experiments = DaskExperimentClient(
            client,
            config=project / "configs/fkessler_model.yaml",
            initial_run_dir=Path(args.initial_run_dir).resolve(),
            run_root=run_root / "models",
            library=project / "build/libpycam_sima_kernels.so",
            environment_script=(
                project
                / "reference/cases/FKESSLER_ne3pg3_gnu_24x50"
                / ".env_mach_specific.sh"
            ),
            python_executable=project / ".venv/bin/python",
            execution_mode="allocation",
        )
        resource_plan = experiments.plan_pool(
            max_concurrent_models=2,
            ranks_per_model=None,
            memory_per_model="auto",
            available_nodes=2,
            cpus_per_node=24,
            memory_per_node="80GB",
        )
        with experiments.pool(
            "validation-pool",
            resource_plan=resource_plan,
        ) as pool:
            with pool.model("base") as base:
                base.advance(args.steps)
                base_status = base.status
                base_snapshot = base.snapshot()
                base_temperature = base.fields.air_temperature.get(rank=0)

                with base.fork("control", require_concurrent=True) as children:
                    control = children.control
                    control_status = control.status
                    control_snapshot = control.snapshot()
                    exact_fork, arrays_compared = _snapshots_equal(
                        base_snapshot,
                        control_snapshot,
                    )
                    control.fields.air_temperature += 1.0
                    changed_temperature = (
                        control.fields.air_temperature.get(rank=0)
                    )
                    parent_temperature_after_edit = (
                        base.fields.air_temperature.get(rank=0)
                    )
                    exact_edit = np.array_equal(
                        changed_temperature,
                        np.add(base_temperature, 1.0),
                    )
                    parent_unchanged = np.array_equal(
                        parent_temperature_after_edit,
                        base_temperature,
                    )
                    children.advance(steps=1)
                    child_final_status = control.status

                slots_after_child_close = pool.slots
                pool_status = pool.status

            if pool_status["mpi_launch_count"] != 1:
                raise RuntimeError("pool launched MPI more than once")
            if resource_plan.ranks_per_model != 24:
                raise RuntimeError("validation did not use 24 ranks per model")
            if resource_plan.model_slots != 2 or resource_plan.world_size != 48:
                raise RuntimeError("validation did not create the requested 2x24 pool")
            if not exact_fork or not exact_edit or not parent_unchanged:
                raise RuntimeError("pool fork/edit isolation validation failed")
            if control_status.step != base_status.step:
                raise RuntimeError("child did not inherit the parent clock")
            if child_final_status.step != base_status.step + 1:
                raise RuntimeError("child did not advance independently")
            if not any(row["state"] == "idle" for row in slots_after_child_close):
                raise RuntimeError("closing the child did not return its slot")

            base_history = (
                run_root / "models" / "validation-pool" / "base" / "history"
            )
            compare_history_directories(
                Path(args.oracle_history).resolve(),
                base_history,
                expected_files=args.steps + 1,
                expected_numeric_variables=26,
            )

    payload = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=project,
            text=True,
        ).strip(),
        "pbs_job_id": os.environ["PBS_JOBID"],
        "freecam": freecam.__version__,
        "resource_plan": resource_plan.describe(),
        "mpi_launch_count": pool_status["mpi_launch_count"],
        "base_steps": base_status.step,
        "child_final_step": child_final_status.step,
        "fork_transport": control_status.snapshot_transport,
        "fork_all_fields_bitwise_equal": exact_fork,
        "rank_local_arrays_compared": arrays_compared,
        "child_exact_add_1K": exact_edit,
        "parent_unchanged_after_child_edit": parent_unchanged,
        "slot_released_after_child_close": True,
        "history_bfb": True,
        "history_files": args.steps + 1,
        "nested_qsub": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "FREECAM_PERSISTENT_POOL_BFB "
        f"job={payload['pbs_job_id']} world=2x24 "
        f"arrays={arrays_compared} history={args.steps + 1}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
