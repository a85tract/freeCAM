"""Validate a granular SegmentPlan through Dask's nested-PBS mode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from distributed import Client

import freecam
from freecam import (
    BranchSpec,
    DaskExperimentClient,
    DaskPBSOptions,
    ObserveFields,
    RunPhase,
    SegmentPlan,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    plan = SegmentPlan(
        "pbs-granular-phase",
        actions=(
            RunPhase("dynamics_to_physics"),
            ObserveFields(("air_temperature",)),
        ),
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
            pbs=DaskPBSOptions(walltime="00:10:00"),
            execution_mode="pbs",
        )
        root = experiments.submit_base(BranchSpec("pbs-root", steps=0))
        granular = experiments.submit_plan(root, plan)
        summaries = {
            "root": experiments.summary(root).result(),
            "granular": experiments.summary(granular).result(),
        }
        field = experiments.field(
            granular, "air_temperature", rank=0
        ).result()

    job_ids = {summary["pbs_job_id"] for summary in summaries.values()}
    if None in job_ids or len(job_ids) != 2:
        raise RuntimeError(f"expected two independent PBS jobs, got {job_ids}")
    if {
        summary["execution_mode"] for summary in summaries.values()
    } != {"pbs"}:
        raise RuntimeError("a granular segment did not use PBS mode")
    for name in ("pbs-root", "pbs-granular-phase"):
        if not (run_root / name / "job.pbs").is_file():
            raise RuntimeError(f"PBS segment {name!r} did not write job.pbs")

    result = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "package_version": freecam.__version__,
        "plan": plan.as_dict(),
        "segments": summaries,
        "job_ids": sorted(job_ids),
        "field_extraction": {
            "field": "air_temperature",
            "rank": 0,
            "shape": list(field.shape),
            "dtype": field.dtype.str,
            "min": float(field.min()),
            "max": float(field.max()),
            "mean": float(field.mean()),
        },
    }
    output = run_root / "dask-granular-pbs-summary.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        "FREECAM_DASK_GRANULAR_PBS_OK "
        f"jobs={','.join(sorted(job_ids))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
