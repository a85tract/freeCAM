"""Validate trusted Notebook Python processes in one persistent MPI pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

from distributed import Client
import numpy as np

import freecam
from freecam import DaskExperimentClient
from freecam.model.history import HISTORY_FIELDS
from freecam.model.validation import (
    compare_history_directories,
    compare_history_files,
)


FKESSLER_ORACLE_FIELDS = tuple(
    name
    for name, _state_name in HISTORY_FIELDS
    if name not in {"TTEND", "UTEND", "VTEND"}
)
RUNTIME_LOCAL_FIELDS = frozenset({"ccpp_mpi_communicator"})
_HISTORY_TIMESTAMP = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}-\d{5})i?\.nc$"
)


def notebook_noop(fields: Any, context: Any) -> None:
    """A serialized no-op used to prove the unmodified 50-step BFB path."""

    del fields, context


def add_one_kelvin(fields: Any, context: Any) -> None:
    """Add exactly one kelvin to this rank's CCPP air-temperature field."""

    del context
    fields["air_temperature"][...] += 1.0


def fail_after_write(fields: Any, context: Any) -> None:
    """Exercise collective rollback after one rank raises an exception."""

    fields["air_temperature"][...] += 2.0
    if context.rank == 7:
        raise RuntimeError("intentional transactional failure on rank 7")


def _inventory(status: Any) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["spec"]["name"]): dict(item)
        for item in status.details.get("python_processes", ())
    }


def _rank_arrays_equal(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
) -> bool:
    return len(left) == len(right) and all(
        np.array_equal(lhs, rhs) for lhs, rhs in zip(left, right)
    )


def _snapshots_equal(
    left: Any,
    right: Any,
    *,
    excluded_fields: frozenset[str] = RUNTIME_LOCAL_FIELDS,
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
            for name in sorted(set(lhs.files) - excluded_fields):
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


def _communicator_binding(model: Any) -> dict[str, Any]:
    status = model.status
    expected = int(status.details["mpi_communicator_handle"])
    values = model.fields.ccpp_mpi_communicator.get(rank=0)
    actual = int(np.asarray(values).reshape(-1)[0])
    return {
        "slot_id": model.slot_id,
        "expected": expected,
        "actual": actual,
        "matches": actual == expected,
    }


def _history_files_by_timestamp(directory: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in directory.glob("*.cam.h*.*.nc"):
        match = _HISTORY_TIMESTAMP.search(path.name)
        if match is None:
            continue
        timestamp = match.group("timestamp")
        if timestamp in files:
            raise RuntimeError(
                f"{directory}: duplicate history timestamp {timestamp}"
            )
        files[timestamp] = path
    return files


def _compare_split_history(
    oracle_directory: Path,
    candidate_directories: Sequence[Path],
) -> None:
    oracle = _history_files_by_timestamp(oracle_directory)
    if len(oracle) != 51:
        raise RuntimeError(
            f"oracle history has {len(oracle)} files instead of 51"
        )
    candidates: dict[str, Path] = {}
    for directory in candidate_directories:
        for timestamp, path in _history_files_by_timestamp(directory).items():
            if timestamp in candidates:
                raise RuntimeError(
                    "split history contains duplicate timestamp "
                    f"{timestamp}: {candidates[timestamp]} and {path}"
                )
            candidates[timestamp] = path
    if set(candidates) != set(oracle):
        raise RuntimeError(
            "split history timestamp mismatch "
            f"missing={sorted(set(oracle) - set(candidates))} "
            f"extra={sorted(set(candidates) - set(oracle))}"
        )
    for timestamp in sorted(oracle):
        compare_history_files(
            oracle[timestamp],
            candidates[timestamp],
            fields=FKESSLER_ORACLE_FIELDS,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--oracle-history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("Python process validation requires PBS")

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_root / "python-process-step-25"
    baseline_checkpoint_path = run_root / "unmodified-step-25"
    model_workers: dict[str, str] = {}
    model_slots: dict[str, int | None] = {}

    with Client(
        processes=False,
        n_workers=5,
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
            max_concurrent_models=4,
            ranks_per_model=24,
            memory_per_model="auto",
            available_nodes=1,
            cpus_per_node=128,
            memory_per_node="80GB",
        )

        with experiments.pool(
            "python-runtime-process",
            resource_plan=resource_plan,
        ) as pool:
            print("PYCAM_VALIDATE stage=pool-ready", flush=True)
            with pool.model("base") as base:
                print("PYCAM_VALIDATE stage=base-ready", flush=True)
                no_op = base.physics.install_python(
                    notebook_noop,
                    name="notebook_noop",
                    group="physics_before_coupler",
                    after="kessler",
                )
                print("PYCAM_VALIDATE stage=noop-installed", flush=True)
                with pool.model("baseline") as baseline:
                    pool.advance((base, baseline), steps=25)
                    baseline_saved = baseline.save(
                        baseline_checkpoint_path
                    )
                    print(
                        "PYCAM_VALIDATE stage=base-baseline-step-25",
                        flush=True,
                    )
                saved = base.save(checkpoint_path)
                print("PYCAM_VALIDATE stage=checkpoint-written", flush=True)
                no_op_hash = no_op.payload_hash

                with base.fork(
                    "control",
                    "warm",
                    "rollback",
                    require_concurrent=True,
                ) as branches:
                    print("PYCAM_VALIDATE stage=forked", flush=True)
                    inherited = {
                        name: _inventory(model.status)
                        for name, model in branches.items()
                    }
                    if any(
                        values["notebook_noop"]["spec"]["payload_hash"]
                        != no_op_hash
                        for values in inherited.values()
                    ):
                        raise RuntimeError(
                            "forked Python process payload hash changed"
                        )

                    base_temperature_before = (
                        base.fields.physics_air_temperature.get(rank="all")
                    )
                    warm_temperature_before = (
                        branches.warm.fields.physics_air_temperature.get(
                            rank="all"
                        )
                    )
                    heating = branches.warm.physics.install_python(
                        add_one_kelvin,
                        name="notebook_add_one_kelvin",
                        group="physics_before_coupler",
                        after="notebook_noop",
                        writes=("air_temperature",),
                    )
                    heating.run()
                    print("PYCAM_VALIDATE stage=plus-one-complete", flush=True)
                    warm_temperature_after = (
                        branches.warm.fields.physics_air_temperature.get(
                            rank="all"
                        )
                    )
                    base_temperature_after = (
                        base.fields.physics_air_temperature.get(rank="all")
                    )
                    exact_add_one = all(
                        np.array_equal(after, np.add(before, 1.0))
                        for before, after in zip(
                            warm_temperature_before,
                            warm_temperature_after,
                        )
                    )
                    parent_unchanged = _rank_arrays_equal(
                        base_temperature_before,
                        base_temperature_after,
                    )

                    rollback_process = (
                        branches.rollback.physics.install_python(
                            fail_after_write,
                            name="notebook_transactional_failure",
                            group="physics_before_coupler",
                            after="notebook_noop",
                            writes=("air_temperature",),
                        )
                    )
                    rollback_before = (
                        branches.rollback.fields.physics_air_temperature.get(
                            rank="all"
                        )
                    )
                    rollback_error = ""
                    try:
                        rollback_process.run()
                    except RuntimeError as exc:
                        rollback_error = str(exc)
                    print("PYCAM_VALIDATE stage=rollback-tested", flush=True)
                    rollback_after = (
                        branches.rollback.fields.physics_air_temperature.get(
                            rank="all"
                        )
                    )
                    rollback_exact = _rank_arrays_equal(
                        rollback_before,
                        rollback_after,
                    )
                    rollback_slot_ready = (
                        branches.rollback.status.details["state"]
                        == "ready"
                    )

                    pool.advance((base, branches.control), steps=25)
                    print("PYCAM_VALIDATE stage=base-control-step-50", flush=True)
                    base_snapshot = base.snapshot()
                    control_snapshot = branches.control.snapshot()
                    (
                        fork_exact,
                        fork_arrays_compared,
                        fork_first_difference,
                    ) = _snapshots_equal(base_snapshot, control_snapshot)
                    model_workers = {
                        "base": base.worker,
                        **{
                            name: model.worker
                            for name, model in branches.items()
                        },
                    }
                    model_slots = {
                        "base": base.slot_id,
                        **{
                            name: model.slot_id
                            for name, model in branches.items()
                        },
                    }
                    communicator_bindings = {
                        "base": _communicator_binding(base),
                        **{
                            name: _communicator_binding(model)
                            for name, model in branches.items()
                        },
                    }

                with (
                    pool.restore("restarted", saved.path) as restarted,
                    pool.restore(
                        "baseline-restarted",
                        baseline_saved.path,
                    ) as baseline_restarted,
                ):
                    print("PYCAM_VALIDATE stage=checkpoint-restored", flush=True)
                    restarted_inventory = _inventory(restarted.status)
                    if (
                        restarted_inventory["notebook_noop"]["spec"][
                            "payload_hash"
                        ]
                        != no_op_hash
                    ):
                        raise RuntimeError(
                            "checkpoint restart changed Python payload hash"
                        )
                    pool.advance(
                        (restarted, baseline_restarted),
                        steps=25,
                    )
                    print("PYCAM_VALIDATE stage=restart-step-50", flush=True)
                    restarted_snapshot = restarted.snapshot()
                    (
                        restart_exact,
                        restart_arrays_compared,
                        restart_first_difference,
                    ) = _snapshots_equal(base_snapshot, restarted_snapshot)
                    restart_transport = restarted.status.snapshot_transport
                    restart_step = restarted.status.step
                    baseline_snapshot = baseline_restarted.snapshot()
                    (
                        baseline_exact,
                        baseline_arrays_compared,
                        baseline_first_difference,
                    ) = _snapshots_equal(base_snapshot, baseline_snapshot)
                    baseline_step = baseline_restarted.status.step
                    communicator_bindings.update(
                        {
                            "restarted": _communicator_binding(restarted),
                            "baseline-restarted": _communicator_binding(
                                baseline_restarted
                            ),
                        }
                    )
                    print(
                        "PYCAM_VALIDATE stage=baseline-step-50",
                        flush=True,
                    )

                base_final = base.status
                pool_status = pool.status
                launcher_worker = pool.worker

            base_history = (
                run_root
                / "models"
                / "python-runtime-process"
                / "base"
                / "history"
            )
            compare_history_directories(
                Path(args.oracle_history).resolve(),
                base_history,
                expected_files=51,
                expected_numeric_variables=26,
                fields=FKESSLER_ORACLE_FIELDS,
            )
            baseline_initial_history = (
                run_root
                / "models"
                / "python-runtime-process"
                / "baseline"
                / "history"
            )
            baseline_restart_history = (
                run_root
                / "models"
                / "python-runtime-process"
                / "baseline-restarted"
                / "history"
            )
            _compare_split_history(
                Path(args.oracle_history).resolve(),
                (baseline_initial_history, baseline_restart_history),
            )

    comparison_evidence = {
        "fork_first_difference": fork_first_difference,
        "restart_first_difference": restart_first_difference,
        "unmodified_vs_noop_first_difference": baseline_first_difference,
    }
    print(
        "PYCAM_VALIDATE comparison="
        + json.dumps(comparison_evidence, sort_keys=True),
        flush=True,
    )
    checks = {
        "one_pbs_job": True,
        "one_mpi_launch": pool_status["mpi_launch_count"] == 1,
        "no_op_50_step_history_bfb": True,
        "unmodified_50_step_history_bfb": True,
        "unmodified_vs_noop_scientific_state_bfb": baseline_exact,
        "fork_inventory_inherited": True,
        "fork_continuation_scientific_state_bfb": fork_exact,
        "checkpoint_restart_scientific_state_bfb": restart_exact,
        "runtime_local_communicators_rebound": all(
            item["matches"] for item in communicator_bindings.values()
        ),
        "exact_add_one_kelvin_all_ranks": exact_add_one,
        "parent_unchanged_by_child_callback": parent_unchanged,
        "transactional_failure_reported_rank_7": "rank 7" in rollback_error,
        "transactional_rollback_all_ranks": rollback_exact,
        "transactional_failure_kept_slot_ready": rollback_slot_ready,
        "checkpoint_restored_process_inventory": True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"Python runtime-process checks failed: {failed}")
    if base_final.step != 50 or restart_step != 50 or baseline_step != 50:
        raise RuntimeError(
            f"unexpected final steps base={base_final.step}, "
            f"restart={restart_step}, baseline={baseline_step}"
        )
    if restart_transport != "checkpoint":
        raise RuntimeError(
            f"unexpected checkpoint transport {restart_transport!r}"
        )
    if len(set(model_workers.values())) != 4:
        raise RuntimeError(
            f"models did not use four distinct Dask workers: {model_workers}"
        )
    if launcher_worker in set(model_workers.values()):
        raise RuntimeError("launcher worker was reused by a model Actor")

    payload = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=project,
            text=True,
        ).strip(),
        "working_tree_feature": "notebook-python-runtime-process",
        "pbs_job_id": os.environ["PBS_JOBID"],
        "freecam": freecam.__version__,
        "python_process": {
            "name": "notebook_noop",
            "payload_hash": no_op_hash,
            "group": "physics_before_coupler",
            "after": "kessler",
            "transactional": True,
        },
        "allocation": {
            "world_size": resource_plan.world_size,
            "model_slots": resource_plan.model_slots,
            "ranks_per_model": resource_plan.ranks_per_model,
            "mpi_launch_count": pool_status["mpi_launch_count"],
            "nested_qsub": 0,
            "pool_mpi_launch_id": pool_status["pool_mpi_launch_id"],
        },
        "dask": {
            "launcher_worker": launcher_worker,
            "model_workers": model_workers,
            "model_slots": model_slots,
            "distinct_model_workers": len(set(model_workers.values())),
        },
        "fork": {
            "transport": "mpi",
            "scientific_state_arrays_bitwise_identical": fork_exact,
            "arrays_compared": fork_arrays_compared,
            "excluded_runtime_local_fields": sorted(RUNTIME_LOCAL_FIELDS),
            "first_difference": fork_first_difference,
        },
        "checkpoint_restart": {
            "path": str(saved.path),
            "transport": restart_transport,
            "scientific_state_arrays_bitwise_identical": restart_exact,
            "arrays_compared": restart_arrays_compared,
            "excluded_runtime_local_fields": sorted(RUNTIME_LOCAL_FIELDS),
            "first_difference": restart_first_difference,
        },
        "unmodified_baseline": {
            "step": baseline_step,
            "checkpoint_path": str(baseline_saved.path),
            "history_segments": [
                str(baseline_initial_history),
                str(baseline_restart_history),
            ],
            "scientific_state_arrays_match_noop_run": baseline_exact,
            "arrays_compared": baseline_arrays_compared,
            "excluded_runtime_local_fields": sorted(RUNTIME_LOCAL_FIELDS),
            "first_difference": baseline_first_difference,
            "history_bfb": True,
        },
        "runtime_local_communicators": communicator_bindings,
        "transactional_failure": {
            "reported_error_excerpt": rollback_error[:4000],
            "all_write_fields_restored": rollback_exact,
            "slot_ready": rollback_slot_ready,
        },
        "checks": checks,
        "completion_marker": (
            "FREECAM_PYTHON_RUNTIME_PROCESS_OK "
            f"job={os.environ['PBS_JOBID']} world=4x24 "
            f"fork_arrays={fork_arrays_compared} "
            f"restart_arrays={restart_arrays_compared}"
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
