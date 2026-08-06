#!/usr/bin/env python3
"""Generate PI-CAM StatePool bridge source and its machine-readable schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pycam_sima.pi_cam.state_codegen import (  # noqa: E402
    generate_fortran_include,
    load_state_bridge,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--description", type=Path, default=REPO / "native/pi_cam/state_bridge.yaml"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fortran", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    bridge = load_state_bridge(args.description, args.source_root)
    args.fortran.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.fortran.write_text(generate_fortran_include(bridge))
    args.manifest.write_text(json.dumps(bridge.manifest(), indent=2) + "\n")
    print(f"generated {len(bridge.fields)} Python-owned state fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
