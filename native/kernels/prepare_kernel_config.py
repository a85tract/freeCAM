#!/usr/bin/env python3
"""Generate the BFB kernel specialization from the Python model config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    values = yaml.safe_load(args.config.read_text())
    nc = int(values["fv_nphys"])
    nlev = int(values["pver"])
    ntrac = int(
        values.get(
            "advected_constituent_count",
            values["constituent_count"],
        )
    )
    np_ = int(values["np"])
    generated = f"""! Generated from {args.config}; do not edit.
module pycam_sima_build_config
  implicit none
  integer, parameter :: build_nc={nc}, build_nlev={nlev}
  integer, parameter :: build_ntrac={ntrac}, build_np={np_}
  integer, parameter :: build_ngpc={nc}, build_irecons=6
  integer, parameter :: build_nhe=1, build_nhr=2, build_nht=3
  integer, parameter :: build_ns=build_nc, build_nhc=3
  integer, parameter :: build_kmin_jet=1, build_kmax_jet={nlev}
  integer, parameter :: build_ngp=build_np*build_np
  logical, parameter :: build_large_courant=.true.
end module pycam_sima_build_config
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)


if __name__ == "__main__":
    main()
