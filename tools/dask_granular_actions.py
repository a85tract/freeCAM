"""Validate granular Dask actions in one PBS allocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from distributed import Client
import numpy as np

import freecam
from freecam import (
    BranchSpec,
    DaskExperimentClient,
    ObserveFields,
    RunPhase,
    RunScheme,
    SegmentPlan,
)
from freecam.notebook.dask import _allocation_launcher


def _action_name(index: int, action: object) -> str:
    return f"chain-{index:02d}-{type(action).__name__.lower()}"


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()


def _compare_checkpoint_bits(left: Path, right: Path) -> dict[str, object]:
    left_files = sorted(left.glob("rank-*.npz"))
    right_files = sorted(right.glob("rank-*.npz"))
    if [path.name for path in left_files] != [path.name for path in right_files]:
        raise RuntimeError("batch and chained checkpoint rank files differ")
    arrays_compared = 0
    bytes_compared = 0
    for left_file, right_file in zip(left_files, right_files):
        with (
            np.load(left_file, allow_pickle=False) as left_arrays,
            np.load(right_file, allow_pickle=False) as right_arrays,
        ):
            if left_arrays.files != right_arrays.files:
                raise RuntimeError(
                    f"array inventory differs for {left_file.name}"
                )
            for name in left_arrays.files:
                left_value = left_arrays[name]
                right_value = right_arrays[name]
                if (
                    left_value.shape != right_value.shape
                    or left_value.dtype != right_value.dtype
                    or left_value.tobytes(order="A")
                    != right_value.tobytes(order="A")
                ):
                    raise RuntimeError(
                        f"batch/chained difference at {left_file.name}:{name}"
                    )
                arrays_compared += 1
                bytes_compared += int(left_value.nbytes)
    return {
        "bitwise_identical": True,
        "ranks": len(left_files),
        "arrays_compared": arrays_compared,
        "bytes_compared": bytes_compared,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--control-steps", type=int, default=50)
    args = parser.parse_args()

    outer_job_id = os.environ.get("PBS_JOBID")
    if not outer_job_id:
        raise SystemExit(
            "dask_granular_actions.py must run inside a PBS allocation"
        )

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    actions = (
        RunPhase("dynamics_to_physics"),
        RunPhase("physics_timestep_initial"),
        RunScheme("calc_exner", group="physics_before_coupler"),
        RunScheme(
            "temp_to_potential_temp",
            group="physics_before_coupler",
        ),
        RunScheme(
            "calc_dry_air_ideal_gas_density",
            group="physics_before_coupler",
        ),
        RunScheme(
            "wet_to_dry_water_vapor",
            group="physics_before_coupler",
        ),
        RunScheme(
            "wet_to_dry_cloud_liquid_water",
            group="physics_before_coupler",
        ),
        RunScheme("wet_to_dry_rain", group="physics_before_coupler"),
        RunScheme("kessler", group="physics_before_coupler"),
        ObserveFields(
            (
                "potential_temperature",
                "large_scale_precipitation_rate",
            )
        ),
    )
    batch_plan = SegmentPlan(
        "granular-batch",
        actions=actions,
        unsafe=True,
    )

    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            config=project / "configs/fkessler_model.yaml",
            initial_run_dir=args.initial_run_dir,
            run_root=run_root,
            library=project / "build/libpycam_sima_kernels.so",
            environment_script=(
                project
                / "reference/cases/FKESSLER_ne3pg3_gnu_24x50/.env_mach_specific.sh"
            ),
            python_executable=project / ".venv/bin/python",
            execution_mode="allocation",
        )
        root = experiments.submit_base(BranchSpec("root", steps=0))
        control = experiments.submit_branch(
            root,
            BranchSpec("control-50step", steps=args.control_steps),
        )
        batch = experiments.submit_plan(root, batch_plan)
        chain_parent = root
        chain_futures = {}
        for index, action in enumerate(actions):
            name = _action_name(index, action)
            chain_parent = experiments.submit_action(
                chain_parent,
                name=name,
                action=action,
            )
            chain_futures[name] = chain_parent
        extracted = experiments.field(
            batch, "potential_temperature", rank=0
        )
        summaries = {
            "root": experiments.summary(root).result(),
            "control": experiments.summary(control).result(),
            "batch": experiments.summary(batch).result(),
            **experiments.summaries(chain_futures),
        }
        extracted_value = extracted.result()

    job_ids = {summary["pbs_job_id"] for summary in summaries.values()}
    modes = {summary["execution_mode"] for summary in summaries.values()}
    if job_ids != {outer_job_id}:
        raise RuntimeError(
            f"segments escaped outer PBS job {outer_job_id}: {job_ids!r}"
        )
    if modes != {"allocation"}:
        raise RuntimeError(f"unexpected execution modes: {sorted(modes)}")
    nested_scripts = tuple(run_root.glob("*/job.pbs"))
    if nested_scripts:
        raise RuntimeError(f"allocation mode wrote nested PBS jobs: {nested_scripts}")

    chain_final_name = _action_name(len(actions) - 1, actions[-1])
    batch_checkpoint = Path(summaries["batch"]["checkpoint_dir"])
    chain_checkpoint = Path(summaries[chain_final_name]["checkpoint_dir"])
    comparison = _compare_checkpoint_bits(batch_checkpoint, chain_checkpoint)

    direct_root = run_root / "direct-dynamics-to-physics"
    direct_log = project / "logs" / (
        f"pycam_direct_phase_{outer_job_id.split('.', 1)[0]}.log"
    )
    direct_command = [
        *_allocation_launcher(os.environ, 24),
        str(project / ".venv/bin/python"),
        str(project / "tools/direct_phase_checkpoint.py"),
        "--config",
        str(project / "configs/fkessler_model.yaml"),
        "--initial-run-dir",
        str(Path(args.initial_run_dir).resolve()),
        "--run-dir",
        str(direct_root / "run"),
        "--history-dir",
        str(direct_root / "history"),
        "--input-checkpoint",
        summaries["root"]["checkpoint_dir"],
        "--output-checkpoint",
        str(direct_root / "checkpoint"),
        "--library",
        str(project / "build/libpycam_sima_kernels.so"),
        "--phase",
        "dynamics_to_physics",
        "--result-json",
        str(direct_root / "result.json"),
    ]
    with direct_log.open("w") as log:
        subprocess.run(
            direct_command,
            check=True,
            cwd=project,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    direct_phase_result = json.loads(
        (direct_root / "result.json").read_text()
    )
    if direct_phase_result.get("pbs_job_id") != outer_job_id:
        raise RuntimeError("direct phase comparison escaped the PBS allocation")
    direct_phase_comparison = _compare_checkpoint_bits(
        direct_root / "checkpoint",
        Path(summaries[_action_name(0, actions[0])]["checkpoint_dir"]),
    )

    kessler_records = [
        record
        for record in summaries["batch"]["action_trace"]
        if record["type"] == "run_scheme"
        and record["action"]["name"] == "kessler"
    ]
    if len(kessler_records) != 1:
        raise RuntimeError(f"expected one Kessler action, got {kessler_records}")
    if kessler_records[0]["native_calls_delta"] != 1:
        raise RuntimeError("Kessler action did not call exactly one native kernel")

    chain_kessler_name = _action_name(8, actions[8])
    chain_kessler_trace = summaries[chain_kessler_name]["action_trace"]
    if (
        len(chain_kessler_trace) != 1
        or chain_kessler_trace[0]["native_calls_delta"] != 1
    ):
        raise RuntimeError("single-action Kessler segment did not call one kernel")

    repository_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    summary = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "root": str(project),
            "parent_commit": repository_commit,
            "package_version": freecam.__version__,
        },
        "pbs_allocation": {
            "job_id": outer_job_id,
            "execution_mode": "allocation",
            "dask_workers": 1,
            "nested_qsub": 0,
        },
        "plan": batch_plan.as_dict(),
        "segments": summaries,
        "batch_vs_chained": comparison,
        "run_phase_vs_direct_driver": {
            **direct_phase_comparison,
            "phase": "dynamics_to_physics",
            "direct_result": direct_phase_result,
            "log_path": str(direct_log),
        },
        "field_extraction": {
            "field": "potential_temperature",
            "rank": 0,
            "shape": list(extracted_value.shape),
            "dtype": extracted_value.dtype.str,
            "min": float(extracted_value.min()),
            "max": float(extracted_value.max()),
            "mean": float(extracted_value.mean()),
        },
        "checkpoint_manifests": {
            "batch": _manifest_sha256(batch_checkpoint),
            "chained": _manifest_sha256(chain_checkpoint),
        },
    }
    summary_path = run_root / "dask-granular-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        "FREECAM_DASK_GRANULAR_OK "
        f"job={outer_job_id} actions={len(actions)} "
        f"ranks={comparison['ranks']} nested_qsub=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
