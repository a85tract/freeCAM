"""Validate one-allocation Dask execution and its exact branch edit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _manifest(path: Path) -> dict:
    payload = json.loads((path / "manifest.json").read_text())
    if payload.get("schema_version") != 1 or payload.get("mpi_size") != 24:
        raise RuntimeError(f"unexpected checkpoint manifest: {path}")
    if len(payload.get("ranks", ())) != 24:
        raise RuntimeError(f"checkpoint does not contain 24 rank records: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--field", default="air_temperature")
    parser.add_argument("--delta", type=float, default=1.0)
    args = parser.parse_args()

    root = Path(args.run_root).resolve()
    summary = json.loads((root / "dask-summary.json").read_text())
    outer_job_id = summary["outer_pbs_job_id"]
    segments = summary["segments"]
    if set(segments) != {"base", "control", "warm"}:
        raise RuntimeError(f"unexpected segment inventory: {sorted(segments)}")
    expected_progress = {
        "base": (25, 26),
        "control": (50, 51),
        "warm": (25, 26),
    }
    for name, values in segments.items():
        if values["execution_mode"] != "allocation":
            raise RuntimeError(f"{name} did not use allocation execution")
        if values["pbs_job_id"] != outer_job_id:
            raise RuntimeError(f"{name} did not remain in PBS job {outer_job_id}")
        progress = (values["step"], values["history_samples"])
        if progress != expected_progress[name]:
            raise RuntimeError(
                f"{name} progress is {progress}, expected {expected_progress[name]}"
            )
    nested_scripts = tuple(root.glob("*/job.pbs"))
    if nested_scripts:
        raise RuntimeError(f"found nested PBS scripts: {nested_scripts}")

    base = root / "base/checkpoint"
    changed = root / "warm/checkpoint"
    if _manifest(base) != _manifest(changed):
        raise RuntimeError("warm checkpoint metadata differs from the base")

    changed_fields: set[str] = set()
    delta_exact = True
    for rank in range(24):
        filename = f"rank-{rank:03d}.npz"
        with (
            np.load(base / filename, allow_pickle=False) as base_rank,
            np.load(changed / filename, allow_pickle=False) as changed_rank,
        ):
            if set(base_rank.files) != set(changed_rank.files):
                raise RuntimeError(f"rank {rank} field inventory differs")
            for name in base_rank.files:
                if not np.array_equal(base_rank[name], changed_rank[name]):
                    changed_fields.add(name)
                    if name == args.field:
                        delta_exact &= np.array_equal(
                            changed_rank[name],
                            np.add(base_rank[name], np.float64(args.delta)),
                        )

    if changed_fields != {args.field}:
        raise RuntimeError(
            f"modified branch changed {sorted(changed_fields)}, expected {[args.field]}"
        )
    if not delta_exact:
        raise RuntimeError(f"{args.field} does not have the exact requested delta")
    print(
        "PYCAM_SIMA_DASK_ALLOCATION_VALID "
        f"job={outer_job_id} ranks=24 nested_qsub=0 "
        f"changed_field={args.field} delta={args.delta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
