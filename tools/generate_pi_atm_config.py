#!/usr/bin/env python3
"""Generate the fail-closed PI-atm YAML from a configured CIME case."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pycam_sima.cesm import CESMCaseConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--library-root", type=Path)
    args = parser.parse_args()

    config = CESMCaseConfig.from_cime_case(
        args.case,
        library_root=args.library_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config.to_mapping(), sort_keys=False, width=100)
    )
    print(args.output)


if __name__ == "__main__":
    main()
