#!/usr/bin/env python3
"""Run the persistent Notebook controller for the full PI-CAM BFB gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pycam_sima.pi_cam import PICAMNotebookSession
from pycam_sima.pi_cam.validation import compare_pi_cam_directories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--env-script", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    with PICAMNotebookSession(
        args.config,
        boundary=args.boundary,
        run_dir=args.run_dir,
        env_script=args.env_script,
        launch_mode="local",
        log_path=args.run_dir / "pi_cam_session_worker.log",
    ) as cam:
        started = dict(cam.status)
        native_state_fields = tuple(
            name
            for name in started.get("fields", {})
            if name.startswith(("cam_in.", "cam_out.", "phys_state.", "phys_tend."))
            and name not in {"cam_in.x2a_rattr", "cam_out.a2x_rattr"}
        )
        if "phys_state.t" not in native_state_fields:
            raise RuntimeError("Python StatePool does not own phys_state.t")
        def active_rank_temperature() -> dict[str, object]:
            temperature = cam.field("phys_state.t", rank=0)
            ncol = cam.field("phys_state.ncol", rank=0)
            active = np.concatenate(
                [
                    temperature[: int(ncol[chunk]), :, chunk]
                    for chunk in range(temperature.shape[-1])
                ],
                axis=0,
            )
            return {
                "rank": 0,
                "count": int(active.size),
                "min": float(active.min()),
                "max": float(active.max()),
                "mean": float(active.mean()),
            }

        initial_temperature = active_rank_temperature()
        completed = dict(cam.step(args.steps))
        export_stats = dict(cam.stats("cam_out.a2x_rattr", rank="global"))
        final_temperature = active_rank_temperature()

    bfb = compare_pi_cam_directories(args.reference, args.run_dir)
    result = {
        "schema_version": 1,
        "execution": "persistent_notebook_session",
        "mpi_launch_count": 1,
        "steps_requested": args.steps,
        "python_owned_native_state_fields": len(native_state_fields),
        "started": started,
        "completed": completed,
        "initial_rank0_active_physics_temperature_statistics": initial_temperature,
        "final_rank0_active_physics_temperature_statistics": final_temperature,
        "rank_global_export_statistics": export_stats,
        "bfb": bfb.to_payload(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bfb.bfb else 1


if __name__ == "__main__":
    raise SystemExit(main())
