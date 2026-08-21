from pathlib import Path

import numpy as np

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    RecordingCAMBackend,
)
from freecam.pi_cam.timing import (
    CESMTimingContext,
    FreeCAMProfiler,
    _format_global_report,
    _PhaseTotals,
    format_cesm_timing_profile,
)


def _profile_context(**overrides) -> CESMTimingContext:
    defaults = dict(
        case_name="PI-atm",
        lid="7.desched1.260821-100000",
        machine="derecho",
        caseroot="/run/PI-atm",
        user="ruitong",
        curr_date="Fri Aug 21 10:00:00 2026",
        driver="freeCAM",
        grid="a%ne16np4 cam5-se",
        compset="1850_CAM%CAM5_CLM_CICE_DOCN_RTM_SGLC_SWAV_SESP",
        run_type="startup, continue_run = FALSE (inittype = TRUE)",
        timestep_seconds=1800,
        mpi_ranks=512,
        tasks_per_node=128,
        components=(
            ("cpl", "cpl"),
            ("atm", "cam"),
            ("lnd", "clm"),
            ("ice", "cice"),
            ("ocn", "docn"),
            ("rof", "rtm"),
            ("glc", "sglc"),
            ("wav", "swav"),
            ("esp", "sesp"),
        ),
    )
    defaults.update(overrides)
    return CESMTimingContext(**defaults)


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


def test_cesm_timing_profile_cost_and_throughput_match_the_formula() -> None:
    context = _profile_context()
    totals = _PhaseTotals(
        steps=50, init_seconds=42.0, run_seconds=90.0, final_seconds=1.5
    )

    report = format_cesm_timing_profile(context, totals)

    # 50 steps x 1800 s = 1.041666... simulated days on 512 whole-node PEs.
    assert "---------------- TIMING PROFILE" in report
    assert "  Case        : PI-atm" in report
    assert "  stop option : nsteps, stop_n = 50" in report
    assert "  pe count for cost estimate : 512 " in report
    assert "Model Cost:           4485.12   pe-hrs/simulated_year " in report
    assert "Model Throughput:        2.74   simulated_years/day " in report
    assert "Init Time   :      42.000 seconds " in report
    assert "Run Time    :      90.000 seconds       86.400 seconds/day " in report
    assert "Final Time  :       1.500 seconds " in report
    # freeCAM is atmosphere-only: TOT == ATM, every other component reads zero.
    assert "TOT Run Time:      90.000 seconds" in report
    assert "ATM Run Time:      90.000 seconds" in report
    assert "CPL Run Time:       0.000 seconds" in report
    assert "esp = sesp          1         0         1" in report


def test_cesm_timing_profile_is_safe_when_no_steps_ran() -> None:
    report = format_cesm_timing_profile(
        _profile_context(),
        _PhaseTotals(steps=0, init_seconds=3.0, run_seconds=0.0, final_seconds=0.1),
    )

    assert "Model Cost:              0.00   pe-hrs/simulated_year " in report
    assert "Model Throughput:        0.00   simulated_years/day " in report
    assert "  stop option : nsteps, stop_n = 0" in report


def test_driver_finalize_writes_a_cesm_timing_profile(tmp_path: Path) -> None:
    (tmp_path / "atm_in").write_text("&dummy /\n", encoding="utf-8")
    config = PICAMConfig(
        case_name="profiled-unit-pi-cam",
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

    profiles = list((tmp_path / "timing").glob("cesm_timing.profiled-unit-pi-cam.*"))
    assert len(profiles) == 1
    text = profiles[0].read_text()
    assert "---------------- TIMING PROFILE" in text
    assert "  Case        : profiled-unit-pi-cam" in text
    assert "  atm = cam" in text
    assert "  stop option : nsteps, stop_n = 1" in text
    assert "ATM Run Time:" in text
