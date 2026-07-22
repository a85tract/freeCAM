"""Run checkpoint fan-out as direct MPI tasks inside one PBS allocation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from distributed import Client

from pycam_sima import BranchSpec, DaskExperimentClient, FieldEdit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--base-steps", type=int, default=25)
    parser.add_argument("--continuation-steps", type=int, default=25)
    args = parser.parse_args()

    outer_job_id = os.environ.get("PBS_JOBID")
    if not outer_job_id:
        raise SystemExit(
            "dask_allocation_fanout.py must run inside a PBS allocation"
        )

    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
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
        base = experiments.submit_base(
            BranchSpec("base", steps=args.base_steps)
        )
        branches = experiments.fork(
            base,
            (
                BranchSpec("control", steps=args.continuation_steps),
                BranchSpec(
                    "warm",
                    steps=0,
                    field_edits=(
                        FieldEdit("air_temperature", "add", 1.0),
                    ),
                ),
            ),
        )
        base_summary = experiments.summary(base).result()
        summaries = experiments.summaries(branches)

    all_summaries = {"base": base_summary, **summaries}
    job_ids = {summary["pbs_job_id"] for summary in all_summaries.values()}
    modes = {summary["execution_mode"] for summary in all_summaries.values()}
    if job_ids != {outer_job_id}:
        raise RuntimeError(
            f"segments escaped outer PBS job {outer_job_id}: {job_ids!r}"
        )
    if modes != {"allocation"}:
        raise RuntimeError(f"unexpected execution modes: {sorted(modes)}")
    nested_scripts = tuple(run_root.glob("*/job.pbs"))
    if nested_scripts:
        raise RuntimeError(f"allocation mode wrote nested PBS jobs: {nested_scripts}")

    summary_path = run_root / "dask-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "outer_pbs_job_id": outer_job_id,
                "dask_workers": 1,
                "segments": all_summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(all_summaries, indent=2, sort_keys=True))
    print(
        "PYCAM_SIMA_DASK_ALLOCATION_OK "
        f"job={outer_job_id} segments={len(all_summaries)} nested_qsub=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
