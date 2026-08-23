#!/usr/bin/env python3
"""Check reviewed function specs against the kernel inventory and the source.

The YAML under native/pi_cam/functions is the runtime's only authority for a
routine's boundary.  This keeps it honest: argument order, dtype, rank, intent
and extents must match the inventory record, and units must match the
bracketed units in the routine's declaration comments.  Exit status is
non-zero on any disagreement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from freecam.physics.spec import default_functions_dir, load_function_spec
from freecam.physics.verify import (
    VerificationReport,
    verify_against_inventory,
    verify_against_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, action="append", default=[])
    parser.add_argument(
        "--inventory", type=Path, default=Path("validation/pi_cam_kernel_inventory.json")
    )
    parser.add_argument("--source-root", type=Path, default=Path("external/iCESM1.3.1_fzhu"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    specs = args.spec or sorted(default_functions_dir().glob("*.yaml"))
    inventory = json.loads(args.inventory.read_text())
    records = {
        str(item.get("qualified_name", "")).lower(): item
        for item in inventory.get("procedures", ())
    }
    reports: list[VerificationReport] = []
    for path in specs:
        spec = load_function_spec(path)
        report = VerificationReport(spec.function)
        record = records.get(spec.qualified_name.lower())
        if record is None:
            report.fail(f"{spec.qualified_name} is not in {args.inventory}")
        else:
            verify_against_inventory(spec, record, report)
            source = args.source_root / record["source"]
            if str(record["source"]) != spec.source:
                report.fail(f"inventory source {record['source']!r} differs from spec {spec.source!r}")
            elif not source.is_file():
                report.fail(f"source file not found: {source}")
            else:
                verify_against_source(
                    spec,
                    source.read_text(encoding="utf-8", errors="replace").splitlines(),
                    line_start=int(record["line_start"]),
                    line_end=int(record["line_end"]),
                    report=report,
                )
        reports.append(report)
        status = "ok" if report.passed else "FAILED"
        print(f"{spec.function}: {status}")
        for text in report.checks:
            print(f"  {text}")
        for text in report.failures:
            print(f"  ! {text}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"schema_version": 1, "reports": [item.as_dict() for item in reports]}, indent=2)
            + "\n"
        )
    return 0 if all(item.passed for item in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
