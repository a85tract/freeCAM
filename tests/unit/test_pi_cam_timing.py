from pathlib import Path

import numpy as np

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    RecordingCAMBackend,
)
from freecam.pi_cam.timing import FreeCAMProfiler, _format_global_report


class _Clock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return float(next(self._values))


class _Comm:
    rank = 0
    size = 1

    @staticmethod
    def allgather(value):
        return [value]

    @staticmethod
    def gather(value, root=0):
        del root
        return [value]


def test_profiler_records_nested_regions_without_runtime_collectives() -> None:
    profiler = FreeCAMProfiler(
        rank=0,
        size=1,
        clock=_Clock((0.0, 1.0, 3.0, 5.0)),
    )

    profiler.start_total()
    with profiler.region("FREECAM:STEP"):
        pass
    profiler.stop_total()

    records = profiler.records
    assert records[("FREECAM:TOTAL",)]["calls"] == 1
    assert records[("FREECAM:TOTAL",)]["walltotal"] == 5.0
    assert records[("FREECAM:TOTAL", "FREECAM:STEP")]["walltotal"] == 2.0


def test_global_report_aggregates_rank_local_timer_totals() -> None:
    snapshots = (
        {
            "rank": 0,
            "size": 2,
            "timers": {
                "FREECAM:TOTAL/FREECAM:STEP": {
                    "calls": 2,
                    "walltotal": 4.0,
                    "wallmax": 2.5,
                    "wallmin": 1.5,
                }
            },
        },
        {
            "rank": 1,
            "size": 2,
            "timers": {
                "FREECAM:TOTAL/FREECAM:STEP": {
                    "calls": 2,
                    "walltotal": 6.0,
                    "wallmax": 3.5,
                    "wallmin": 2.5,
                }
            },
        },
    )

    report = _format_global_report(snapshots)

    assert "FreeCAM global timing statistics" in report
    assert "FREECAM:TOTAL/FREECAM:STEP" in report
    assert "10.000000" in report
    assert "6.000000" in report


def test_driver_finalize_writes_freecam_timing_products(tmp_path: Path) -> None:
    (tmp_path / "atm_in").write_text("&dummy /\n", encoding="utf-8")
    config = PICAMConfig(
        case_name="timed-unit-pi-cam",
        source_root=Path("/tmp/source"),
        mpi_size=1,
        stop_n=1,
    )
    boundary = InMemoryBoundaryProvider(
        {
            (0, 0): {"sst": np.full((2,), 280.0)},
            (1, 0): {"sst": np.full((2,), 281.0)},
            (2, 0): {"sst": np.full((2,), 282.0)},
        }
    )
    driver = PICAMDriver(
        config,
        boundary,
        RecordingCAMBackend(),
        rank=0,
        size=1,
        communicator=_Comm(),
        run_dir=tmp_path,
    )

    driver.initialize()
    driver.step()
    driver.finalize()

    detail = tmp_path / "timing" / "freecam_timing.0000"
    global_stats = tmp_path / "timing" / "freecam_timing_stats"
    assert detail.is_file()
    assert global_stats.is_file()
    detail_text = detail.read_text()
    assert "FREECAM:INITIALIZE" in detail_text
    assert "FREECAM:STEP" in detail_text
    assert "CAM:dadadj" in detail_text
    timer_lines = detail_text.splitlines()[6:]
    assert timer_lines[0].lstrip().startswith("FREECAM:TOTAL")
    assert "FREECAM:TOTAL/FREECAM:STEP/CAM:dadadj" in global_stats.read_text()
