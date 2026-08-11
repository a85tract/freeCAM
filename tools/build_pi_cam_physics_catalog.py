#!/usr/bin/env python3
"""Generate the flat PI-atm physics catalog from committed validation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freecam.pi_cam.physics_catalog import PICAMPhysicsRules, build_physics_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", default="validation/pi_cam_kernel_inventory.json"
    )
    parser.add_argument(
        "--adapter-validation",
        default="validation/pi_cam_in_module_adapter_generation.json",
    )
    parser.add_argument(
        "--output",
        default="src/freecam/pi_cam/data/pi_cam_physics_catalog.json",
    )
    parser.add_argument(
        "--rules", default="native/pi_cam/physics_process_rules.yaml"
    )
    arguments = parser.parse_args()
    inventory = json.loads(Path(arguments.inventory).read_text())
    adapters = json.loads(Path(arguments.adapter_validation).read_text())
    generated = {
        (str(item.get("name")), str(item.get("original_source"))): item
        for item in adapters.get("generated", ())
    }
    # The adapter generator uses the freshest AST record.  Overlay its exact
    # signature (including function results and kind scope) onto older
    # committed inventories before producing the user-facing catalog.
    for procedure in inventory.get("procedures", ()):
        record = generated.get(
            (str(procedure.get("qualified_name")), str(procedure.get("source")))
        )
        if record is None:
            continue
        procedure["arguments"] = record.get("arguments", procedure.get("arguments", ()))
        procedure["result"] = record.get("result")
    rules = PICAMPhysicsRules.load(arguments.rules)
    catalog = build_physics_catalog(
        inventory,
        generated_adapters=adapters,
        rules=rules,
    )
    output = catalog.write(arguments.output)
    summary = catalog.machine_record()
    print(
        f"wrote {summary['physics_interfaces']} flat physics interfaces "
        f"from {summary['reachable_procedures']} reachable procedures to {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
