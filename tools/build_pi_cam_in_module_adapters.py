#!/usr/bin/env python3
"""Generate module-internal adapters for all reachable PI-CAM processes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freecam.pi_cam.in_module_adapter import (
    adapter_strategy_counts,
    generate_in_module_source_tree,
)
from freecam.pi_cam.physics_catalog import PICAMPhysicsRules, build_physics_catalog
from freecam.pi_cam.source_catalog import PICAMSourceCatalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--inventory-report",
        type=Path,
        default=Path("validation/pi_cam_kernel_inventory.json"),
    )
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()

    project = arguments.project_root.resolve()
    source = (
        arguments.source_root.resolve()
        if arguments.source_root is not None
        else (project / "external/iCESM1.3.1_fzhu").resolve()
    )
    source_catalog = PICAMSourceCatalog.discover(
        project,
        source_root=source,
        workers=arguments.workers,
    )
    source_record = source_catalog.machine_record()
    arguments.inventory_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.inventory_report.write_text(
        json.dumps(source_record, indent=2, sort_keys=True) + "\n"
    )
    physics = build_physics_catalog(
        source_record,
        rules=PICAMPhysicsRules.load(
            project / "native/pi_cam/physics_process_rules.yaml"
        ),
    )
    by_key = {
        (procedure.qualified_name, procedure.source): procedure
        for procedure in source_catalog.procedures
    }
    selected = tuple(
        by_key[(process.qualified_name, process.source)]
        for process in physics.physics_processes
    )
    generated = generate_in_module_source_tree(
        selected,
        source_root=source,
        output_root=arguments.output_root,
    )
    generated_names = {adapter.procedure.qualified_name for adapter in generated}
    report = {
        "schema_version": 1,
        "generator": "tools/build_pi_cam_in_module_adapters.py",
        "source_revision": source_catalog.cam_source_revision,
        "physical_processes": len(selected),
        "generated_in_module_adapters": len(generated),
        "strategy_counts": adapter_strategy_counts(selected),
        "generated": [adapter.manifest() for adapter in generated],
        "not_generated": [
            process.as_dict()
            for process in physics.physics_processes
            if process.qualified_name not in generated_names
        ],
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"generated {len(generated)}/{len(selected)} in-module adapters below "
        f"{arguments.output_root.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
