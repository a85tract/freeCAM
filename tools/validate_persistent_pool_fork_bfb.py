"""Validate 25-step parent + MPI-memory fork + 25-step child BFB."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from distributed import Client
import numpy as np

import pycam_sima
from pycam_sima import DaskExperimentClient
from pycam_sima.model.validation import compare_history_directories


def _snapshots_equal(
    left: Any,
    right: Any,
) -> tuple[bool, int, Mapping[str, Any] | None]:
    left_payloads = left.rank_payloads()
    right_payloads = right.rank_payloads()
    if len(left_payloads) != len(right_payloads):
        return False, 0, {
            "reason": "rank_count",
            "left": len(left_payloads),
            "right": len(right_payloads),
        }
    compared = 0
    for rank, (left_payload, right_payload) in enumerate(
        zip(left_payloads, right_payloads)
    ):
        with (
            np.load(BytesIO(left_payload[1]), allow_pickle=False) as lhs,
            np.load(BytesIO(right_payload[1]), allow_pickle=False) as rhs,
        ):
            if set(lhs.files) != set(rhs.files):
                return False, compared, {
                    "reason": "field_inventory",
                    "rank": rank,
                    "missing": sorted(set(lhs.files) - set(rhs.files)),
                    "extra": sorted(set(rhs.files) - set(lhs.files)),
                }
            for name in sorted(lhs.files):
                if not np.array_equal(lhs[name], rhs[name]):
                    difference = np.abs(lhs[name] - rhs[name])
                    return False, compared, {
                        "reason": "array_values",
                        "rank": rank,
                        "field": name,
                        "maximum_absolute_difference": float(
                            np.nanmax(difference)
                        ),
                    }
                compared += 1
    return True, compared, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--oracle-history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fork-step", type=int, default=25)
    parser.add_argument("--final-step", type=int, default=50)
    args = parser.parse_args()

    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("persistent fork BFB validation requires PBS")
    if args.fork_step <= 0 or args.final_step <= args.fork_step:
        raise ValueError("require 0 < fork-step < final-step")

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    continuation_steps = args.final_step - args.fork_step

    with Client(
        processes=False,
        n_workers=1,
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
            ranks_per_model=24,
            memory_per_model="auto",
            available_nodes=2,
            cpus_per_node=24,
            memory_per_node="80GB",
        )
        with experiments.pool(
            "fork-25x25-bfb",
            resource_plan=resource_plan,
        ) as pool:
            with pool.model("base") as base:
                base.advance(steps=args.fork_step)
                base_at_fork = base.status

                with base.fork(
                    "control",
                    require_concurrent=True,
                ) as children:
                    control = children.control
                    child_at_fork = control.status
                    if child_at_fork.step != args.fork_step:
                        raise RuntimeError(
                            "child did not inherit parent fork step"
                        )

                    control.advance(steps=continuation_steps)
                    base.advance(steps=continuation_steps)

                    child_final = control.status
                    base_final = base.status
                    child_snapshot = control.snapshot()
                    base_snapshot = base.snapshot()
                    exact, arrays_compared, first_difference = (
                        _snapshots_equal(base_snapshot, child_snapshot)
                    )

                pool_status = pool.status

            base_history = (
                run_root
                / "models"
                / "fork-25x25-bfb"
                / "base"
                / "history"
            )
            compare_history_directories(
                Path(args.oracle_history).resolve(),
                base_history,
                expected_files=args.final_step + 1,
                expected_numeric_variables=26,
            )

    if pool_status["mpi_launch_count"] != 1:
        raise RuntimeError("pool launched MPI more than once")
    if not exact:
        raise RuntimeError(
            f"fork continuation is not BFB: {first_difference!r}"
        )
    if base_final.step != args.final_step:
        raise RuntimeError(f"base ended at step {base_final.step}")
    if child_final.step != args.final_step:
        raise RuntimeError(f"child ended at step {child_final.step}")
    if child_at_fork.snapshot_transport != "mpi":
        raise RuntimeError(
            f"unexpected fork transport {child_at_fork.snapshot_transport!r}"
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
        "pycam_sima": pycam_sima.__version__,
        "allocation": {
            "world_size": resource_plan.world_size,
            "model_slots": resource_plan.model_slots,
            "ranks_per_model": resource_plan.ranks_per_model,
            "mpi_launch_count": pool_status["mpi_launch_count"],
            "nested_qsub": 0,
        },
        "path": {
            "base_start_step": 0,
            "fork_step": base_at_fork.step,
            "child_inherited_step": child_at_fork.step,
            "continuation_steps": continuation_steps,
            "base_final_step": base_final.step,
            "child_final_step": child_final.step,
            "fork_transport": child_at_fork.snapshot_transport,
        },
        "validation": {
            "all_statepool_arrays_bitwise_identical": exact,
            "rank_local_arrays_compared": arrays_compared,
            "first_difference": first_difference,
            "continuous_base_history_bfb": True,
            "history_files": args.final_step + 1,
            "numeric_variables": 26,
        },
        "completion_marker": (
            "PYCAM_SIMA_POOL_FORK_25X25_BFB "
            f"job={os.environ['PBS_JOBID']} arrays={arrays_compared}"
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(payload["completion_marker"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
