#!/usr/bin/env python3
"""Exercise a non-zero Notebook Python process inside complete PI-CAM steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import freecam as fc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--increment", type=float, default=1.0e-4)
    parser.add_argument("--queue", default="develop")
    parser.add_argument("--verify-exports", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    class TemperatureIncrement(fc.Physics):
        name = "temperature_increment"
        writes = ("phys_state.t",)

        def run(self, state, context):
            del context
            state.T += args.increment

    def workflow(default):
        configured = default.copy()
        configured.insert_after("dadadj", TemperatureIncrement())
        configured.insert_before("radiation_tend", TemperatureIncrement())
        return configured

    case = fc.CaseConfig(
        name="PI-atm-python-temperature-validation",
        description="Two non-zero Python temperature processes",
        forcing="1850 prescribed SST and sea ice",
        make_atm=lambda: fc.FreeCAM(workflow=workflow),
    )
    driver = fc.Driver(
        case=case,
        nsteps=args.steps,
        queue=args.queue,
        verify_boundary_exports=args.verify_exports,
    )
    try:
        result = driver.run()
        status = driver.status
        payload = {
            "schema_version": 1,
            "steps": args.steps,
            "increment": args.increment,
            "actions": result.actions,
            "first_action": result.first_process,
            "last_action": result.last_process,
            "model_step": int(status["step"]),
            "native_step": int(status["native_step"]),
            "boundary_export_verification": bool(
                status["boundary_export_verification"]
            ),
            "run_dir": str(driver.run_dir),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        print(json.dumps(payload, indent=2, default=str))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
