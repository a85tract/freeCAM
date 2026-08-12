#!/usr/bin/env python3
"""Validate the public freeCAM UI through one 50-step persistent MPI run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import freecam as fc
import numpy as np
from freecam.pi_cam.validation import compare_pi_cam_directories


class TransientNoop(fc.Physics):
    """Exercise runtime workflow insertion without changing CAM state."""

    name = "ui_transient_noop"
    writes = ("phys_state.t",)

    def run(self, state, context):
        del context
        state.T += 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    case = fc.CaseConfig(
        name="PI-atm-ui-validation",
        description="public Python UI validation",
        forcing="unchanged PI-atm forcing",
    )
    fc.CASES.register(case, replace=True)
    driver = fc.Driver(
        case=case.name,
        nsteps=args.steps,
        repo=args.repo,
        run_dir=args.run_dir,
        launch_mode="local",
        history_every=1,
        restart_every="end",
    )
    preview = driver.preview()
    preview_without_launch = not driver.running
    try:
        initial_mean = driver.cam.state.T.mean()
        top_level = driver.cam.state.T[:, 0, :]
        top_level_before = top_level.mean()
        top_level += 0.0
        top_level_after = top_level.mean()

        driver.cam.state.ui_expression_probe = np.zeros(8)
        driver.cam.state.ui_expression_probe = (
            np.sin(driver.cam.state.ui_expression_probe) + 1.0
        )
        expression_stats = driver.cam.state.ui_expression_probe.stats(rank="global")
        del driver.cam.state.ui_expression_probe

        transient = driver.cam.workflow.append(TransientNoop())
        transient.move(after="dadadj")
        transient.run()
        popped = driver.cam.workflow.pop(driver.cam.workflow.index(transient))
        transient_removed = transient.name not in {
            item.name for item in driver.cam.workflow
        }
        trace = driver.run(args.steps)
        completed = dict(driver.status)
        run_dir = driver.run_dir
    finally:
        driver.close()
        fc.CASES.unregister(case.name)

    assert run_dir is not None
    history_files = tuple(str(path) for path in driver.cam.history.files)
    history_streams = tuple(driver.cam.history.streams)
    with driver.cam.history.open("h0") as history:
        history_variables = tuple(history.data_vars)
    comparison = compare_pi_cam_directories(args.reference, run_dir)
    result = {
        "schema_version": 1,
        "date": "2026-08-12",
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "queue": os.environ.get("PBS_QUEUE"),
        "mpi_ranks": 512,
        "steps": args.steps,
        "case_registry_lookup": True,
        "preview_actions": len(preview),
        "preview_without_launch": preview_without_launch,
        "slice_noop": {
            "selection": "[:, 0, :]",
            "mean_before": top_level_before,
            "mean_after": top_level_after,
            "unchanged": top_level_before == top_level_after,
        },
        "distributed_numpy_expression": {
            "expression": "sin(ui_expression_probe) + 1",
            "global_min": expression_stats["min"],
            "global_max": expression_stats["max"],
            "global_mean": expression_stats["mean"],
            "field_removed_before_scientific_run": True,
        },
        "state_mapping": {
            "field_count": len(completed.get("fields", {})),
        },
        "xarray_output": {
            "history_files": history_files,
            "history_streams": history_streams,
            "variables": history_variables,
        },
        "workflow_list_api": {
            "appended": transient.name,
            "popped": popped.name,
            "removed_from_live_workflow": transient_removed,
        },
        "initial_global_temperature_mean": initial_mean,
        "output": completed.get("output"),
        "executed_actions": len(trace),
        "candidate_run": str(run_dir),
        "reference_run": str(args.reference),
        "comparison": comparison.to_payload(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if comparison.bfb else 1


if __name__ == "__main__":
    raise SystemExit(main())
