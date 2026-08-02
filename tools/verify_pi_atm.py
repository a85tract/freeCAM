#!/usr/bin/env python3
"""Fail-closed bitwise comparison for PI-atm history and restart files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycam_sima.cesm.validation import compare_cesm_directories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_cesm_directories(args.reference, args.candidate)
    payload = result.to_mapping()
    text = json.dumps(payload, indent=2, default=str) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    return 0 if result.bfb else 1


if __name__ == "__main__":
    raise SystemExit(main())
