"""MPI command line for one PI-CAM-only model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpi4py import MPI

from .boundary import ReplayBoundaryProvider
from .case import PICAMCase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pycam_sima.pi_cam.cli")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    world = MPI.COMM_WORLD
    case = PICAMCase.from_yaml(args.config)
    boundary = ReplayBoundaryProvider(args.boundary)
    with case.runtime(
        boundary=boundary,
        communicator=world,
        run_dir=args.run_dir,
    ) as cam:
        cam.initialize()
        cam.advance(args.steps)
        local = {
            "rank": world.Get_rank(),
            "step": cam.clock.nstep,
            "date": cam.clock.yyyymmdd,
            "seconds": cam.clock.seconds,
            "fields": len(cam.pool),
            "state_bytes": cam.pool.nbytes,
            "actions": len(cam.trace),
        }
    records = world.gather(local, root=0)
    if world.Get_rank() == 0:
        summary = {
            "schema_version": 1,
            "case": case.config.case_name,
            "mpi_ranks": world.Get_size(),
            "steps": args.steps if args.steps is not None else case.config.stop_n,
            "rank_state_bytes": [record["state_bytes"] for record in records],
            "rank_fields": [record["fields"] for record in records],
            "final_date": records[0]["date"],
            "final_seconds": records[0]["seconds"],
        }
        text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.summary is None:
            print(text, end="")
        else:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
