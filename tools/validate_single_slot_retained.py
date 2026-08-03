"""Validate single-model rank-local retained-state continuation."""

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
from pycam_sima.model.validation import (
    HISTORY_FIELD_NAMES,
    compare_history_directories,
)


# The pinned legacy FKESSLER oracle predates the three tendency-only history
# diagnostics.  Keep the comparison contract explicit: all 26 variables that
# exist in both the oracle and the current history output must be bitwise
# identical, while TTEND/UTEND/VTEND remain current-run diagnostics.
_LEGACY_ORACLE_FIELDS = tuple(
    name
    for name in HISTORY_FIELD_NAMES
    if name not in {"TTEND", "UTEND", "VTEND"}
)
if len(_LEGACY_ORACLE_FIELDS) != 26:
    raise RuntimeError(
        "legacy FKESSLER history contract changed: expected 26 fields, "
        f"found {len(_LEGACY_ORACLE_FIELDS)}"
    )


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
                    return False, compared, {
                        "reason": "array_values",
                        "rank": rank,
                        "field": name,
                    }
                compared += 1
    return True, compared, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--oracle-history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-step", type=int, default=25)
    parser.add_argument("--final-step", type=int, default=50)
    args = parser.parse_args()

    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("single-slot validation requires a PBS allocation")
    if args.split_step <= 0 or args.final_step <= args.split_step:
        raise ValueError("require 0 < split-step < final-step")

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    continuation = args.final_step - args.split_step

    with Client(
        processes=False,
        n_workers=2,
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
            ranks_per_model=24,
            retained_snapshots=1,
            available_nodes=1,
            cpus_per_node=25,
            memory_per_node="80GB",
        )
        if resource_plan.model_slots != 1 or resource_plan.world_size != 24:
            raise RuntimeError(
                f"unexpected single-slot plan: {resource_plan.describe()}"
            )

        with experiments.pool(
            "single-slot-retained",
            resource_plan=resource_plan,
        ) as pool:
            print(
                "single-slot pool ready: "
                f"world={resource_plan.world_size}, launches=1",
                flush=True,
            )
            with pool.model("continuous") as continuous:
                print(
                    f"continuous path: advancing {args.final_step} steps",
                    flush=True,
                )
                continuous.advance(steps=args.final_step)
                continuous_status = continuous.status
                continuous_snapshot = continuous.snapshot()

            retained_base = pool.model("retained-base")
            print(
                "retained path: "
                f"{args.split_step} + {continuation} steps",
                flush=True,
            )
            retained_base.advance(steps=args.split_step)
            retained_state = retained_base.retain("after-split")
            retained_descriptor = {
                "snapshot_id": retained_state.snapshot_id,
                "source_slot": retained_state.source_slot,
                "step": retained_state.step,
                "rank_count": retained_state.rank_count,
                "nbytes": retained_state.nbytes,
                "config_hash": retained_state.config_hash,
            }
            retained_base.close()
            idle_with_snapshot = dict(pool.slots[0])

            with pool.restore_retained(
                "retained-control", retained_state
            ) as retained_control:
                restore_status = retained_control.status
                retained_control.advance(steps=continuation)
                retained_snapshot = retained_control.snapshot()
                retained_final = retained_control.status

            with pool.restore_retained(
                "retained-warm", retained_state
            ) as retained_warm:
                before = retained_warm.fields.air_temperature.get(rank=0)
                retained_warm.fields.air_temperature += 1.0
                after = retained_warm.fields.air_temperature.get(rank=0)
                warm_exact_add = np.array_equal(after, np.add(before, 1.0))

            retained_state.close()
            pool_status = pool.status
            retained_after_drop = pool.retained_states

    print("comparing complete rank-local snapshots", flush=True)
    retained_equal, retained_arrays, retained_difference = _snapshots_equal(
        continuous_snapshot, retained_snapshot
    )
    if not retained_equal:
        raise RuntimeError(
            f"retained continuation differs: {retained_difference}"
        )
    if not warm_exact_add:
        raise RuntimeError("retained warm branch did not apply exact NumPy +1 K")
    if retained_after_drop:
        raise RuntimeError("retained state was not deleted")
    if pool_status["mpi_launch_count"] != 1:
        raise RuntimeError("pool launched MPI more than once")

    oracle = Path(args.oracle_history).resolve()
    history_results = {}
    for name in ("continuous", "retained-control"):
        history = (
            run_root
            / "models"
            / "single-slot-retained"
            / name
            / "history"
        )
        expected = (
            args.final_step + 1
            if name == "continuous"
            else continuation + 1
        )
        if name == "continuous":
            compare_history_directories(
                oracle,
                history,
                expected_files=expected,
                expected_numeric_variables=26,
                fields=_LEGACY_ORACLE_FIELDS,
            )
            history_results[name] = "BFB"

    payload = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=project, text=True
        ).strip(),
        "pbs_job_id": os.environ["PBS_JOBID"],
        "pycam_sima": pycam_sima.__version__,
        "allocation": {
            "model_slots": resource_plan.model_slots,
            "ranks_per_model": resource_plan.ranks_per_model,
            "world_size": resource_plan.world_size,
            "mpi_launch_count": pool_status["mpi_launch_count"],
            "nested_qsub": 0,
            "pool_mpi_launch_id": pool_status["pool_mpi_launch_id"],
        },
        "retained": {
            **retained_descriptor,
            "idle_slot_state_bytes": idle_with_snapshot["state_bytes"],
            "idle_slot_retained_bytes": idle_with_snapshot[
                "retained_snapshot_bytes"
            ],
            "restore_transport": restore_status.snapshot_transport,
            "final_step": retained_final.step,
            "bitwise_identical": retained_equal,
            "arrays_compared": retained_arrays,
            "first_difference": retained_difference,
            "warm_exact_add_1K": warm_exact_add,
            "deleted": not retained_after_drop,
        },
        "continuous": {
            "final_step": continuous_status.step,
            "history": history_results,
        },
        "completion_marker": (
            "PYCAM_SIMA_SINGLE_SLOT_RETAINED_BFB "
            f"job={os.environ['PBS_JOBID']} world=1x24 "
            f"retained_arrays={retained_arrays}"
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["completion_marker"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
