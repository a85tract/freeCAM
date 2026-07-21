#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def append_once(path: Path, line: str) -> None:
    text = path.read_text()
    if line not in text.splitlines():
        path.write_text(text.rstrip() + "\n" + line + "\n")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--case-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    cam = repo / "external/CAM-SIMA"
    case_name = f"FKESSLER_ne3pg3_gnu_24x{args.steps}"
    case = (args.case_root or repo / "reference/cases" / case_name).resolve()
    output_root = (
        args.output_root
        or Path("/glade/derecho/scratch/ruitong/pycam-sima") / case_name
    ).resolve()
    if case.exists():
        raise SystemExit(f"case already exists: {case}")
    usermods = cam / "cime_config/testdefs/testmods_dirs/cam/outfrq_se_cslam"
    run(
        [
            "./cime/scripts/create_newcase",
            "--case",
            str(case),
            "--compset",
            "FKESSLER",
            "--res",
            "ne3pg3_ne3pg3_mg37",
            "--machine",
            "derecho",
            "--compiler",
            "gnu",
            "--project",
            "UCUB0188",
            "--pecount",
            "24",
            "--user-mods-dirs",
            str(usermods),
            "--output-root",
            str(output_root),
            "--walltime",
            "00:30:00",
            "--run-unsupported",
        ],
        cwd=cam,
    )
    run(
        [
            "./xmlchange",
            f"STOP_OPTION=nsteps,STOP_N={args.steps},REST_OPTION=nsteps,"
            f"REST_N={args.steps},DOUT_S=FALSE",
        ],
        cwd=case,
    )
    run(["./case.setup"], cwd=case)
    append_once(case / "user_nl_cam", "hist_precision;h1: REAL64")
    append_once(case / "user_nl_cam", "hist_max_frames;h1: 1")
    run(["./preview_namelists"], cwd=case)
    if args.build or args.submit:
        run(["./case.build"], cwd=case)
        run(["./check_input_data", "--download"], cwd=case)
    if args.submit:
        run(["./case.submit"], cwd=case)
    print(case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
