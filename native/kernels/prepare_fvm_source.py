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
        (
            marker
            + "  public :: compute_displacements_for_swept_areas\n"
            + "  public :: large_courant_number_increment\n"
            + "  integer, public :: pycam_transport_stage = 0\n"
        ),
        1,
    )
    large_courant_marker = """    !
    !***************************************
    !
    ! Large Courant number increment
"""
    if generated.count(large_courant_marker) != 1:
        raise RuntimeError("unexpected FVM large-Courant section")
    generated = generated.replace(
        large_courant_marker,
        "    if (pycam_transport_stage /= 1) then\n\n" + large_courant_marker,
        1,
    )
    parallel_end_marker = "    !$OMP END PARALLEL\n"
    if generated.count(parallel_end_marker) != 1:
        raise RuntimeError("unexpected FVM OpenMP section")
    generated = generated.replace(
        parallel_end_marker,
        "    endif\n" + parallel_end_marker,
        1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)


if __name__ == "__main__":
    main()
