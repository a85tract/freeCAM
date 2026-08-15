import importlib.util
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "report_pi_cam_month",
    PROJECT / "tools/report_pi_cam_month.py",
)
assert SPEC is not None and SPEC.loader is not None
REPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORTER)
build_report = REPORTER.build_report
parse_gptl = REPORTER.parse_gptl


def test_parse_gptl_and_build_month_report(tmp_path: Path) -> None:
    timing_file = tmp_path / "cesm_timing.000"
    timing_file.write_text(
        '    "CPL:INIT" - 1 - 22.0 22.0 22.0 0.0\n'
        '    "CPL:RUN_LOOP" - 1488 - 495.0 1.0 0.1 0.0\n'
        '    "CPL:ATM_RUN" - 1488 - 404.0 1.0 0.1 0.0\n'
        '    "CPL:FINAL" - 1 - 1.0 1.0 1.0 0.0\n'
    )
    config = tmp_path / "case.yaml"
    config.write_text("stop_n: 1488\n")
    report = build_report(
        freecam_summary={
            "steps": 1488,
            "mpi_ranks": 512,
            "pbs_job_id": "1.server",
            "timing": {
                "initialize_seconds": 20.0,
                "advance_seconds": 505.0,
                "finalize_seconds": 2.0,
                "total_seconds": 527.0,
                "simulated_days": 31.0,
                "advance_sypd": 14.5,
            },
        },
        bfb={"bfb": True, "compared_files": 18},
        gptl=parse_gptl(timing_file),
        config=config,
    )

    assert report["result"] == "passed"
    assert report["original_fortran_seconds"]["lifecycle_sum"] == 518.0
    assert report["performance_ratios"][
        "freecam_advance_over_original_atmosphere"
    ] == 505.0 / 404.0
