#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MARKER = "# pycam-sima full shared-library flags"
PIC_LINES = """\
# pycam-sima full shared-library flags
string(APPEND CFLAGS " -fPIC")
string(APPEND CXXFLAGS " -fPIC")
string(APPEND FFLAGS " -fPIC")
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_root", type=Path)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    case = args.case_root.resolve()
    macro = case / "cmake_macros/gnu_derecho.cmake"
    text = macro.read_text()
    if MARKER not in text:
        macro.write_text(PIC_LINES + text)
    if args.rebuild:
        subprocess.run(["./case.build", "--clean-all"], cwd=case, check=True)
        subprocess.run(["./case.build"], cwd=case, check=True)
    else:
        print("PIC flags added; run ./case.build --clean-all && ./case.build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
