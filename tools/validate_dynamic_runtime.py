"""Validate runtime fields and source/prebuilt physics on every MPI rank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from freecam.core.runtime_env import mpi_loader_environment
from freecam.model import (
    CAMDriver,
    ModelConfig,
    read_checkpoint,
    restore_driver,
)
from freecam.model.comm import world_comm


def _ensure_mpi_loader_environment() -> None:
    """Re-exec this validation entrypoint when mpi4py cannot find libmpi."""

    try:
        from mpi4py import MPI  # noqa: F401
    except (ImportError, RuntimeError) as exc:
        if (
            "libmpi.so" not in str(exc)
            or os.environ.get("FREECAM_MPI_ENV_READY")
        ):
            raise RuntimeError(f"mpi4py cannot load the MPI runtime: {exc}") from exc
    else:
        return
    environment = mpi_loader_environment()
    environment["FREECAM_MPI_ENV_READY"] = "1"
    os.execve(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def _array_hash(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="F")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config = ModelConfig.from_yaml(args.config)
    _ensure_mpi_loader_environment()
    comm = world_comm()
    if comm.size != config.mpi_size:
        raise RuntimeError(
            f"configuration requires {config.mpi_size} ranks, got {comm.size}"
        )

    driver = CAMDriver(
        config,
        run_dir=args.run_dir,
        comm=comm,
        kernel_library=args.library,
        history_dir=args.history_dir,
    ).start()
    pointers_before = driver.pool.pointer_records()

    control_initial = np.float64(comm.rank) + np.float64(0.25)
    control_values = driver.fields.create(
        "runtime_control",
        standard_name="runtime_control",
        dtype="float64",
        dims=("column",),
        units="1",
        initial=control_initial,
    )
    if not np.all(control_values == control_initial):
        raise AssertionError("rank-local runtime_control initialization failed")

    descriptor = (
        project_root
        / "examples/plugins/runtime_temperature_offset/device.yaml"
    )
    plugin = driver.physics.install(
        descriptor,
        project_root=project_root,
        after="kessler",
        inputs={
            "runtime_plugin_temperature": np.float64(240.0),
            "runtime_plugin_temperature_increment": np.float64(1.5),
        },
    )
    driver.pool.assert_pointer_stability(pointers_before)

    native_calls_before = driver.backend.call_count
    driver.physics.scheme(
        "runtime_temperature_offset", group="before"
    ).run()
    native_calls_delta = driver.backend.call_count - native_calls_before
    temperature_name = driver.pool.ccpp_field_name(
        "runtime_plugin_temperature"
    )
    temperature = driver.pool.get(temperature_name)
    if native_calls_delta != 1:
        raise AssertionError(
            f"expected one native plugin call, got {native_calls_delta}"
        )
    if not np.array_equal(
        temperature, np.full(temperature.shape, 241.5, dtype=np.float64)
    ):
        raise AssertionError("source plugin did not add exactly 1.5 K")
    if not driver.pool.is_initialized(temperature_name):
        raise AssertionError("native output was not marked initialized")

    checkpoint = driver.write_checkpoint(args.checkpoint_dir)
    plugin_checkpoint_record = plugin.as_dict()
    source_temperature_hash = _array_hash(temperature)
    source_control_hash = _array_hash(control_values)
    snapshot = read_checkpoint(checkpoint, comm)
    driver.finalize()

    restored = restore_driver(
        snapshot,
        run_dir=args.run_dir,
        comm=comm,
        kernel_library=args.library,
        history_dir=Path(args.history_dir).with_name("history-restored"),
        expected_config=config,
    )
    restored_control = restored.pool.get("runtime_control")
    restored_temperature = restored.pool.get_ccpp(
        "runtime_plugin_temperature"
    )
    if _array_hash(restored_control) != source_control_hash:
        raise AssertionError("dynamic field changed across checkpoint restore")
    if _array_hash(restored_temperature) != source_temperature_hash:
        raise AssertionError("plugin field changed across checkpoint restore")
    if restored.plugins.inventory() != (plugin_checkpoint_record,):
        raise AssertionError("plugin inventory changed across restore")

    deleted = restored.fields.delete("runtime_control")
    if deleted["standard_name"] != "runtime_control":
        raise AssertionError("dynamic deletion returned the wrong contract")
    if "runtime_control" in restored.pool.contracts:
        raise AssertionError("deleted dynamic field remains in the StatePool")

    restored.physics.scheme(
        "runtime_temperature_offset", group="before"
    ).run()
    final_temperature = restored.pool.get_ccpp(
        "runtime_plugin_temperature"
    )
    if not np.array_equal(
        final_temperature,
        np.full(final_temperature.shape, 243.0, dtype=np.float64),
    ):
        raise AssertionError("restored prebuilt plugin did not add 1.5 K")

    local = {
        "rank": int(comm.rank),
        "runtime_control_hash": source_control_hash,
        "source_temperature_hash": source_temperature_hash,
        "final_temperature_hash": _array_hash(final_temperature),
        "shape": list(final_temperature.shape),
        "native_calls_delta": native_calls_delta,
        "dynamic_fields": sorted(restored.pool.dynamic_fields),
        "manifest_path": plugin.manifest_path,
        "manifest_hash": plugin.manifest_hash,
        "library_hash": plugin.library_hash,
    }
    records = comm.gather(local, root=0)
    restored.finalize()

    if comm.rank == 0:
        assert records is not None
        result = {
            "schema_version": 1,
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                text=True,
            ).strip(),
            "pbs_job_id": os.environ.get("PBS_JOBID"),
            "mpi_ranks": int(comm.size),
            "source_plugin_built": True,
            "prebuilt_plugin_restored": True,
            "dynamic_variable_deleted_collectively": True,
            "old_pointer_stability": True,
            "checkpoint_schema": 2,
            "checkpoint": str(checkpoint),
            "plugin": plugin_checkpoint_record,
            "ranks": records,
            "result": "PASS",
        }
        output = Path(args.result_json).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
