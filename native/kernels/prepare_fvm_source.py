#!/usr/bin/env python3
"""Prepare selected CAM FVM source for the stateless model library."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.read_text()
    if "write_pycam_fvm_stage" in source:
        raise RuntimeError("refusing to compile an oracle-instrumented FVM source")
    marker = "  public :: run_consistent_se_cslam\n"
    if source.count(marker) != 1:
        raise RuntimeError("unexpected FVM source export declaration")
    generated = source.replace(
        marker,
        marker + "  public :: compute_displacements_for_swept_areas\n",
        1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)


if __name__ == "__main__":
    main()
