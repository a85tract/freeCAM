"""Two-rank integration smoke for FreeCAM timing report aggregation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from mpi4py import MPI

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    RecordingCAMBackend,
)


def main() -> int:
    comm = MPI.COMM_WORLD
    run_dir = Path(os.environ["FREECAM_TIMING_SMOKE_RUN"])
    boundary = InMemoryBoundaryProvider(
        {
            (step, comm.rank): {"sst": np.full((2,), 280.0 + step)}
            for step in range(3)
        }
    )
    driver = PICAMDriver(
        PICAMConfig(
            case_name="freecam-timing-mpi-smoke",
            source_root=Path("/tmp/source"),
            mpi_size=comm.size,
            stop_n=1,
        ),
        boundary,
        RecordingCAMBackend(),
        rank=comm.rank,
        size=comm.size,
        fcomm=comm.py2f(),
        communicator=comm,
        run_dir=run_dir,
    )
    driver.initialize()
    driver.step()
    driver.finalize()

    if comm.rank == 0:
        detail = run_dir / "timing" / "freecam_timing.0000"
        stats = run_dir / "timing" / "freecam_timing_stats"
        assert detail.is_file()
        assert stats.is_file()
        text = stats.read_text()
        assert "MPI tasks: 2" in text
        assert "FREECAM:TOTAL/FREECAM:STEP/CAM:dadadj" in text
        (run_dir / "result.json").write_text(
            json.dumps(
                {
                    "mpi_ranks": comm.size,
                    "detail": str(detail),
                    "global_stats": str(stats),
                    "passed": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
