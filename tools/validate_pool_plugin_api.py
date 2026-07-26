"""Real-MPI smoke test for the pooled dynamic-physics public API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from distributed import Client
import numpy as np

from pycam_sima import DaskExperimentClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--initial-run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.environ.get("PBS_JOBID"):
        raise RuntimeError("pool plugin validation requires a PBS allocation")
    project = Path(__file__).resolve().parents[1]
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=False)

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
        plan = experiments.plan_pool(
            max_concurrent_models=1,
            ranks_per_model=24,
            available_nodes=1,
            cpus_per_node=24,
            memory_per_node="80GB",
        )
        with experiments.pool("plugin-api", resource_plan=plan) as pool:
            with pool.model("base") as base:
                installed = base.physics.install(
                    source=(
                        project
                        / "examples/plugins/runtime_temperature_offset/device.yaml"
                    ),
                    project_root=project,
                    after="kessler",
                    inputs={
                        "runtime_plugin_temperature": 240.0,
                        "runtime_plugin_temperature_increment": 1.5,
                    },
                )
                if installed["name"] != "runtime_temperature_offset":
                    raise RuntimeError(
                        f"unexpected pooled install result: {installed!r}"
                    )
                field = base.fields.ccpp_runtime_plugin_temperature
                before = field.stats(rank=0)
                base.physics.scheme(
                    "runtime_temperature_offset", group="before"
                ).run()
                after = field.stats(rank=0)
                delta = after["mean"] - before["mean"]
                if not np.isclose(delta, 1.5):
                    raise RuntimeError(f"plugin field delta is {delta}, expected 1.5")
                launch_count = pool.status["mpi_launch_count"]

    payload = {
        "pbs_job_id": os.environ["PBS_JOBID"],
        "mpi_ranks": 24,
        "mpi_launch_count": launch_count,
        "installed_result": installed,
        "plugin_delta": delta,
        "passed": True,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(
        "PYCAM_SIMA_POOL_PLUGIN_API_OK "
        f"job={payload['pbs_job_id']} name={installed['name']} delta={delta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
