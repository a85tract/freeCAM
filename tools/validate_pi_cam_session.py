#!/usr/bin/env python3
"""Run the persistent Notebook controller for the full PI-CAM BFB gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
        completed = dict(cam.step(args.steps))
        export_stats = dict(cam.stats("cam_out.a2x_rattr", rank="global"))

    bfb = compare_pi_cam_directories(args.reference, args.run_dir)
    result = {
        "schema_version": 1,
        "execution": "persistent_notebook_session",
        "mpi_launch_count": 1,
        "steps_requested": args.steps,
        "started": started,
        "completed": completed,
        "rank_global_export_statistics": export_stats,
        "bfb": bfb.to_payload(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bfb.bfb else 1


if __name__ == "__main__":
    raise SystemExit(main())
