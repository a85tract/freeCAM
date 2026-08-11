#!/usr/bin/env python3
"""Write the complete 262-interface PI-CAM process support audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freecam.pi_cam.physics_catalog import PICAMPhysicsCatalog
from freecam.pi_cam.plan import PICAMStepPlan
from freecam.pi_cam.process_support import build_process_support_report


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--compilation", type=Path, required=True)
    parser.add_argument("--loading", type=Path, required=True)
    parser.add_argument("--runtime-validation", type=Path)
    parser.add_argument("--bfb-validation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = build_process_support_report(
        catalog=PICAMPhysicsCatalog.load_default(),
        runtime_records=PICAMStepPlan.default().describe(),
        generation=_read(arguments.generation),
        compilation=_read(arguments.compilation),
        loading=_read(arguments.loading),
        runtime_validation=(
            None
            if arguments.runtime_validation is None
            else _read(arguments.runtime_validation)
        ),
        bfb_validation=(
            None if arguments.bfb_validation is None else _read(arguments.bfb_validation)
        ),
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"supported {report['adapters_compiled']}/"
        f"{report['formerly_catalog_only_interfaces']} former catalog-only "
        f"interfaces; {report['current_case_loadable']} load in PI-CAM"
    )
    return 0 if report["all_catalog_only_interfaces_supported"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
