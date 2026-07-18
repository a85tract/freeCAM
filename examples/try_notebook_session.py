#!/usr/bin/env python3
"""Small live-field demo for PyCAM-SIMA's Jupyter MPI session.

In a Notebook running on a compute node, execute for example:

    %run /glade/work/ruitong/pycam-sima/examples/try_notebook_session.py --steps 2

After ``%run`` completes, ``last_field``, ``last_stats``, and ``run_dir_used``
remain available in the Notebook namespace.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pycam_sima import NotebookSession


REPO = Path("/glade/work/ruitong/pycam-sima")
REFERENCE_RUN = Path(
    "/glade/derecho/scratch/ruitong/pycam-sima/"
    "FKESSLER_ne3pg3_gnu_24x50/FKESSLER_ne3pg3_gnu_24x50/run"
)
CASE = REPO / "reference/cases/FKESSLER_ne3pg3_gnu_24x50"

# These names intentionally remain global so IPython's %run copies the final
# values into the interactive Notebook namespace.
last_field: np.ndarray | None = None
last_stats: dict[str, Any] | None = None
run_dir_used: Path | None = None


def default_run_dir() -> Path:
    user = os.environ.get("USER", "ruitong")
    scratch = Path(os.environ.get("SCRATCH", f"/glade/derecho/scratch/{user}"))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return scratch / "pycam-sima/notebook_trials" / stamp / "run"


def prepare_run_dir(run_dir: Path) -> None:
    existing_history = tuple(run_dir.glob("*.cam.h*.nc")) if run_dir.exists() else ()
    if existing_history:
        raise RuntimeError(
            f"refusing to overwrite {len(existing_history)} history files in {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "atm_in"
    if not target.exists():
        shutil.copy2(REFERENCE_RUN / "atm_in", target)


def describe(step: int, field: str, rank: int, array: np.ndarray, stats: dict[str, Any]) -> None:
    first = array.ravel(order="F")[0]
    print(
        f"step={step} rank={rank} field={field} shape={array.shape} "
        f"first={first:.17g} min={stats['min']:.17g} "
        f"max={stats['max']:.17g} mean={stats['mean']:.17g}"
    )


def main() -> int:
    global last_field, last_stats, run_dir_used

    parser = argparse.ArgumentParser(
        description="Run PyCAM-SIMA interactively and print one live field after every step."
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--field", default="air_temperature")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--print-array", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")

    run_dir_used = (args.run_dir or default_run_dir()).resolve()
    prepare_run_dir(run_dir_used)
    worker_log = run_dir_used / "mpi-worker.log"

    print(f"run directory: {run_dir_used}")
    print(f"worker log:   {worker_log}")
    print("starting one controller plus 24 MPI CAM-SIMA workers ...")

    with NotebookSession(
        REPO / "configs/fkessler_ne3pg3.yaml",
        run_dir=run_dir_used,
        env_script=CASE / ".env_mach_specific.sh",
        log_path=worker_log,
    ) as model:
        print(f"session ready: {len(model.field_names)} fields")
        if args.field not in model.field_names:
            available = "\n  ".join(model.field_names)
            raise KeyError(f"unknown field {args.field!r}; available fields:\n  {available}")

        last_field = model.get_field(args.field, rank=args.rank)
        assert isinstance(last_field, np.ndarray)
        last_stats = model.get_field_stats(args.field, rank=args.rank)
        describe(model.current_step, args.field, args.rank, last_field, last_stats)

        for _ in range(args.steps):
            model.step()
            last_field = model.get_field(args.field, rank=args.rank)
            assert isinstance(last_field, np.ndarray)
            last_stats = model.get_field_stats(args.field, rank=args.rank)
            describe(model.current_step, args.field, args.rank, last_field, last_stats)
            if args.print_array:
                print(last_field)

    print("session closed cleanly")
    print("Jupyter variables available: last_field, last_stats, run_dir_used")
    return 0


if __name__ == "__main__":
    main()
