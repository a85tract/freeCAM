"""Validate exact state isolation from the Dask branch smoke."""

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
    base = root / "base/checkpoint"
    control = root / "control/checkpoint"
    changed = root / "warm/checkpoint"
    manifests = tuple(_manifest(path) for path in (base, control, changed))
    if manifests[0] != manifests[1] or manifests[0] != manifests[2]:
        raise RuntimeError("branch checkpoint metadata differs from the base")

    control_changes: set[str] = set()
    changed_fields: set[str] = set()
    delta_exact = True
    for rank in range(24):
        filename = f"rank-{rank:03d}.npz"
        with (
            np.load(base / filename, allow_pickle=False) as base_rank,
            np.load(control / filename, allow_pickle=False) as control_rank,
            np.load(changed / filename, allow_pickle=False) as changed_rank,
        ):
            if set(base_rank.files) != set(control_rank.files) or set(
                base_rank.files
            ) != set(changed_rank.files):
                raise RuntimeError(f"rank {rank} field inventory differs")
            for name in base_rank.files:
                if not np.array_equal(base_rank[name], control_rank[name]):
                    control_changes.add(name)
                if not np.array_equal(base_rank[name], changed_rank[name]):
                    changed_fields.add(name)
                    if name == args.field:
                        delta_exact &= np.array_equal(
                            changed_rank[name],
                            np.add(base_rank[name], np.float64(args.delta)),
                        )

    if control_changes:
        raise RuntimeError(f"control branch changed fields: {sorted(control_changes)}")
    if changed_fields != {args.field}:
        raise RuntimeError(
            f"modified branch changed {sorted(changed_fields)}, expected {[args.field]}"
        )
    if not delta_exact:
        raise RuntimeError(f"{args.field} does not have the exact requested delta")
    print(
        "PYCAM_SIMA_DASK_BRANCH_BFB "
        f"ranks=24 control_changed=0 changed_field={args.field} "
        f"delta={args.delta}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
