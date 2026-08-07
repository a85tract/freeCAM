"""Validate diskless Dask fan-out into independent persistent MPI models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import subprocess
from typing import Any

import dask
from distributed import Client
import distributed
import numpy as np

import freecam
from freecam import (
    BranchSpec,
    CheckpointBundle,
    DaskExperimentClient,
    DaskPBSOptions,
    FieldEdit,
)


def _status_subset(status: dict[str, Any]) -> dict[str, Any]:
    names = (
        "name",
        "parent_name",
        "running",
        "worker_host",
        "worker_pid",
        "ranks",
        "step",
        "native_calls",
        "mpi_launch_count",
        "snapshot_transport",
        "source_snapshot_nbytes",
        "launch_mode",
        "pbs_job_id",
        "run_dir",
        "history_dir",
        "log_path",
    )
    return {name: status[name] for name in names if name in status}


def _all_equal(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
    return all(
        np.array_equal(left_rank, right_rank)
        for left_rank, right_rank in zip(left, right)
    )


def _compare_snapshot_arrays(
    source: CheckpointBundle,
    candidate: CheckpointBundle,
    *,
    add_temperature: float | None = None,
) -> tuple[bool, int]:
    source_payloads = source.rank_payloads()
    candidate_payloads = candidate.rank_payloads()
    if len(source_payloads) != len(candidate_payloads):
        return False, 0
    compared = 0
    for source_payload, candidate_payload in zip(
        source_payloads,
        candidate_payloads,
    ):
        with np.load(
            BytesIO(source_payload[1]), allow_pickle=False
        ) as source_arrays:
            with np.load(
                BytesIO(candidate_payload[1]), allow_pickle=False
            ) as candidate_arrays:
                if set(source_arrays.files) != set(candidate_arrays.files):
                    return False, compared
                for name in source_arrays.files:
                    expected = source_arrays[name]
                    if (
                        add_temperature is not None
                        and name == "air_temperature"
                    ):
                        expected = np.add(expected, add_temperature)
                    if not np.array_equal(candidate_arrays[name], expected):
                        return False, compared
                    compared += 1
    return True, compared


def _step_subset(status: dict[str, Any]) -> dict[str, Any]:
    kessler = next(
        row
        for row in status["scheme_status"]["plan"]["schemes"]
        if row["name"] == "kessler"
        and row["source_group"] == "physics_before_coupler"
    )
    return {
        "step": status["step"],
        "native_calls": status["native_calls"],
        "mpi_launch_count": status["mpi_launch_count"],
        "last_phase": status["phase_status"]["last_phase"],
        "last_scheme": status["scheme_status"]["last_scheme"],
        "sequence_safe": status["scheme_status"]["sequence_safe"],
        "kessler_enabled": kessler["enabled"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-steps", type=int, default=10)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    base: Any = None
    base_closed = False
    children: dict[str, Any] = {}
    with Client(
        processes=True,
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
            execution_mode="pbs",
            pbs=DaskPBSOptions(walltime="00:20:00"),
        )
        try:
            base = experiments.start_persistent("base")
            base.step(args.base_steps).result()
            base_status = base.describe().result()
            base_temperature = base.field(
                "air_temperature", rank="all"
            ).result()
            base_snapshot = base.memory_checkpoint().result()

            children = experiments.fork_persistent(
                base,
                (
                    BranchSpec("control", steps=0),
                    BranchSpec(
                        "no-kessler",
                        steps=0,
                        disable_schemes=("kessler",),
                    ),
                    BranchSpec(
                        "warm",
                        steps=0,
                        field_edits=(
                            FieldEdit("air_temperature", "add", 1.0),
                        ),
                    ),
                ),
                close_parent=True,
            )
            base_closed = True
            child_status = {
                name: child.describe().result()
                for name, child in children.items()
            }
            initial_temperature = {
                name: child.field("air_temperature", rank="all").result()
                for name, child in children.items()
            }
            control_snapshot = children["control"].memory_checkpoint().result()
            no_kessler_snapshot = children[
                "no-kessler"
            ].memory_checkpoint().result()
            warm_snapshot = children["warm"].memory_checkpoint().result()
            control_all_fields, arrays_compared = _compare_snapshot_arrays(
                base_snapshot,
                control_snapshot,
            )
            no_kessler_all_fields, no_kessler_arrays_compared = (
                _compare_snapshot_arrays(
                    base_snapshot,
                    no_kessler_snapshot,
                )
            )
            warm_only_temperature, warm_arrays_compared = (
                _compare_snapshot_arrays(
                    base_snapshot,
                    warm_snapshot,
                    add_temperature=1.0,
                )
            )
            control_equal = _all_equal(
                base_temperature,
                initial_temperature["control"],
            )
            no_kessler_equal = _all_equal(
                base_temperature,
                initial_temperature["no-kessler"],
            )
            warm_exact = all(
                np.array_equal(warm, np.add(source, 1.0))
                for source, warm in zip(
                    base_temperature,
                    initial_temperature["warm"],
                )
            )
            step_futures = {
                name: child.step(1) for name, child in children.items()
            }
            stepped = {
                name: future.result()
                for name, future in step_futures.items()
            }
            final_stats = {
                name: child.field_stats(
                    "air_temperature", rank=0
                ).result()
                for name, child in children.items()
            }
        finally:
            for child in children.values():
                try:
                    child.close().result()
                except BaseException:
                    pass
            if base is not None and not base_closed:
                try:
                    base.close().result()
                except BaseException:
                    pass

    expected_step = args.base_steps + 1
    if (
        not control_equal
        or not no_kessler_equal
        or not warm_exact
        or not control_all_fields
        or not no_kessler_all_fields
        or not warm_only_temperature
    ):
        raise RuntimeError("persistent fork did not restore independent exact state")
    if len(
        {
            arrays_compared,
            no_kessler_arrays_compared,
            warm_arrays_compared,
        }
    ) != 1:
        raise RuntimeError("persistent branches compared different field inventories")
    if any(status["step"] != args.base_steps for status in child_status.values()):
        raise RuntimeError("persistent child did not inherit the base model clock")
    if any(status["step"] != expected_step for status in stepped.values()):
        raise RuntimeError("persistent child did not advance independently")
    if any(
        status["snapshot_transport"] != "memory"
        for status in child_status.values()
    ):
        raise RuntimeError("persistent child used a non-memory snapshot transport")
    if any(status["parent_name"] != "base" for status in child_status.values()):
        raise RuntimeError("persistent child lost its parent identity")
    checkpoint_artifacts = sorted(
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if path.name == "manifest.json"
        or path.suffix == ".npz"
        or path.name == "checkpoints"
    )
    if checkpoint_artifacts:
        raise RuntimeError(
            "diskless fork wrote checkpoint artifacts: "
            f"{checkpoint_artifacts}"
        )

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
            "freecam": freecam.__version__,
            "dask": dask.__version__,
            "distributed": distributed.__version__,
        },
        "dask_workers": 3,
        "mpi_ranks_per_model": 24,
        "base_steps": args.base_steps,
        "branch_steps": 1,
        "snapshot_transport": "memory",
        "checkpoint_artifacts": checkpoint_artifacts,
        "base": _status_subset(base_status),
        "children_after_restore": {
            name: _status_subset(status)
            for name, status in child_status.items()
        },
        "inheritance": {
            "control_bitwise_equal": control_equal,
            "no_kessler_bitwise_equal": no_kessler_equal,
            "warm_exact_numpy_add_1K": warm_exact,
            "control_all_fields_bitwise_equal": control_all_fields,
            "no_kessler_all_fields_bitwise_equal": no_kessler_all_fields,
            "warm_only_air_temperature_changed": warm_only_temperature,
            "rank_local_arrays_compared_per_branch": arrays_compared,
        },
        "children_after_step": {
            name: _step_subset(status) for name, status in stepped.items()
        },
        "field_stats_rank_0": final_stats,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "FREECAM_DASK_PERSISTENT_FORK_OK "
        f"base_steps={args.base_steps} children=3 "
        "transport=memory checkpoint_artifacts=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
