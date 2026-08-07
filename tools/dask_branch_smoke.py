"""Submit a zero-step base and two checkpoint-restored Dask branches."""

from __future__ import annotations

import argparse
from pathlib import Path

from distributed import Client

from freecam import BranchSpec, DaskExperimentClient, FieldEdit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    with Client(
        processes=False,
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            config=project / "configs/fkessler_model.yaml",
            initial_run_dir=args.initial_run_dir,
            run_root=args.run_root,
            library=project / "build/libpycam_sima_kernels.so",
            environment_script=(
                project
                / "reference/cases/FKESSLER_ne3pg3_gnu_24x50/.env_mach_specific.sh"
            ),
            python_executable=project / ".venv/bin/python",
        )
        base = experiments.submit_base(BranchSpec("base", steps=0))
        branches = experiments.fork(
            base,
            (
                BranchSpec("control", steps=0),
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

    print("base", base_summary)
    for name, summary in summaries.items():
        print(name, summary)
    print("FREECAM_DASK_BRANCH_OK branches=2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
