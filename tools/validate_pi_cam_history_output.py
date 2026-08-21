#!/usr/bin/env python3
"""Prove Python-owned fields reach every CAM history sample of a run.

The stream that adds those fields closes a sample's averaging window one
step after CAM writes the sample itself, so the run's last window is still
open at finalization.  This gate drives an hourly tape at the half-hour
PI-atm timestep, which makes both the ordinary samples and that final one
observable in a twelve-step run, and checks three things a notebook user
depends on:

* every sample CAM wrote after the field existed carries it, including the
  last one, which only finalization can complete;
* a field created with ``output=False`` reaches no history file at all;
* asking a live model to describe its history streams returns instead of
  deadlocking, which it did while only rank 0 entered the collective column
  map that resolving a stream's fields builds.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import netCDF4
import numpy as np

import freecam as fc


def _tapes(run_dir: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(glob.glob(f"{run_dir}/*.cam.h0.*.nc")):
        with netCDF4.Dataset(path) as dataset:
            rows.append(
                {
                    "file": Path(path).name,
                    "date": int(dataset.variables["date"][0]),
                    "datesec": int(dataset.variables["datesec"][0]),
                    "dt_macro": "dt_macro" in dataset.variables,
                    "scratch_probe": "scratch_probe" in dataset.variables,
                }
            )
    return rows


def _attribute_kind(value: object) -> str:
    """Text or the numeric type a reader sees, not the value itself."""

    if isinstance(value, (str, bytes)):
        return "text"
    return str(np.asarray(value).dtype)


def _added_variable_matches_cam(path: str | Path) -> dict[str, object]:
    """Compare the added field's declaration with a CAM field in the same file."""

    with netCDF4.Dataset(path) as dataset:
        native = dataset.variables["T"]
        added = dataset.variables["dt_macro"]
        return {
            "file_format": dataset.data_model,
            "dimensions_match": added.dimensions == native.dimensions,
            "dtype_matches": str(added.dtype) == str(native.dtype),
            "attributes_match": sorted(added.ncattrs()) == sorted(native.ncattrs()),
            "attribute_types_match": {
                name: _attribute_kind(added.getncattr(name)) for name in added.ncattrs()
            }
            == {
                name: _attribute_kind(native.getncattr(name))
                for name in native.ncattrs()
            },
            "dimensions": list(added.dimensions),
            "dtype": str(added.dtype),
            "attribute_types": {
                name: _attribute_kind(added.getncattr(name)) for name in added.ncattrs()
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # nhtfrq=-1 writes one h0 sample per model hour, i.e. every two steps.
    driver = fc.Driver(case="PI-atm", nsteps=args.steps, namelist={"nhtfrq": -1})
    driver.initialize()
    state = driver.cam.state
    state.create("dt_macro", like="T", units="K")
    state.dt_macro[:] = 1.0
    state.create("scratch_probe", like="T", units="K", output=False)
    state.scratch_probe[:] = 2.0

    streams = [dict(row) for row in driver.cam.history_streams]
    driver.run(steps=args.steps)
    before_close = _tapes(driver.run_dir)
    run_dir = str(driver.run_dir)
    driver.close()
    after_close = _tapes(run_dir)

    samples = [row for row in after_close if row["dt_macro"]]
    fidelity = _added_variable_matches_cam(f"{run_dir}/{after_close[-1]['file']}")
    resolved = [
        list(row.get("resolved_fields", ())) for row in streams
    ]
    passed = (
        resolved == [["dt_macro"]]
        and len(samples) == args.steps // 2
        and bool(after_close[-1]["dt_macro"])
        and not bool(before_close[-1]["dt_macro"])
        and not any(row["scratch_probe"] for row in after_close)
        and fidelity["dimensions_match"]
        and fidelity["dtype_matches"]
        and fidelity["attributes_match"]
        and fidelity["attribute_types_match"]
    )
    record = {
        "schema_version": 1,
        "gate": "pi_cam_python_history_output_12step",
        "expectation": (
            "every CAM history sample written after a Python-owned field "
            "exists carries it, including the run's last sample, which only "
            "finalization completes; a field created with output=False "
            "reaches no history file; the added variable is declared exactly "
            "as CAM declares its own; describing history streams on a live "
            "model returns on every rank"
        ),
        "run_status": "passed" if passed else "failed",
        "steps": args.steps,
        "nhtfrq": -1,
        "history_streams": streams,
        "tapes_before_close": before_close,
        "tapes_after_close": after_close,
        "samples_with_python_field": len(samples),
        "final_sample_completed_at_finalization": (
            bool(after_close[-1]["dt_macro"])
            and not bool(before_close[-1]["dt_macro"])
        ),
        "output_false_field_written_anywhere": any(
            row["scratch_probe"] for row in after_close
        ),
        "added_variable_matches_cam": fidelity,
        "run_dir": run_dir,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"history output gate: {record['run_status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
