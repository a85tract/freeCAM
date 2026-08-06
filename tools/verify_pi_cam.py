#!/usr/bin/env python3
"""Compare every stored CAM variable without numerical tolerance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycam_sima.pi_cam.validation import compare_pi_cam_directories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_pi_cam_directories(args.reference, args.candidate)
    text = json.dumps(result.to_payload(), indent=2, default=str) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    return 0 if result.bfb else 1


if __name__ == "__main__":
    raise SystemExit(main())
