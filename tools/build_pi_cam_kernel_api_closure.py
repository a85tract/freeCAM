#!/usr/bin/env python3
"""Generate or check the PI-atm kernel-API closure record.

Scans the configured pinned CAM sources with the build's macros and search
path, resolves every call site, and writes
validation/pi_cam_kernel_api_closure.json.  ``--check`` verifies that the
committed record still matches the pinned source, the rules, the generator,
and the reference case's configuration without rescanning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam import site  # noqa: E402
from freecam.pi_cam.call_tree import CallTree  # noqa: E402
from freecam.pi_cam.kernel_api_closure import (  # noqa: E402
    DEFAULT_RECORD,
    DEFAULT_RULES,
    PATCHED_PHYSPKG,
    ClosureInputs,
    build_closure,
    check_closure,
    load_closure_rules,
    write_record,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=REPO / DEFAULT_RULES)
    parser.add_argument("--output", type=Path, default=REPO / DEFAULT_RECORD)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check", action="store_true", help="verify the committed record against its inputs")
    parser.add_argument("--save-scans", type=Path, help="pickle the file scans for later reuse (development)")
    parser.add_argument("--load-scans", type=Path, help="reuse pickled file scans instead of rescanning (development)")
    parser.add_argument("--no-site", action="store_true", help="do not consult the site's reference case; use the previous record's configuration")
    arguments = parser.parse_args()

    rules = load_closure_rules(arguments.rules)
    previous = json.loads(arguments.output.read_text()) if arguments.output.is_file() else None
    where = {} if arguments.no_site else site.resolved(repo=REPO)
    inputs = ClosureInputs(
        project_root=REPO,
        rules_path=arguments.rules.relative_to(REPO) if arguments.rules.is_absolute() and arguments.rules.is_relative_to(REPO) else arguments.rules,
        rules=rules,
        reference_run=where.get("reference run"),
        reference_case=where.get("reference case"),
        native_manifest=where.get("native manifest"),
        previous_record=previous,
        patched_physpkg=REPO / PATCHED_PHYSPKG,
    )
    if arguments.check:
        if previous is None:
            print(f"missing record {arguments.output}", file=sys.stderr)
            return 1
        problems = check_closure(previous, inputs)
        if problems:
            for problem in problems:
                print(f"stale: {problem}", file=sys.stderr)
            return 1
        summary = previous.get("summary", {})
        print(
            f"current: {summary.get('procedures_configured')} configured procedures, "
            f"{summary.get('unresolved_references')} unresolved references"
        )
        return 0

    tree = None
    if arguments.load_scans is not None:
        with arguments.load_scans.open("rb") as handle:
            tree = CallTree(pickle.load(handle))
        tree.resolve()
    record = build_closure(inputs, workers=arguments.workers, tree=tree)
    if arguments.save_scans is not None and tree is None:
        # scans are rebuilt inside build_closure; rescan once more to save them
        from freecam.pi_cam.kernel_api_closure import scan_tree

        scanned, _ = scan_tree(inputs, workers=arguments.workers)
        with arguments.save_scans.open("wb") as handle:
            pickle.dump(scanned.scans, handle)
    path = write_record(record, arguments.output)
    summary = record["summary"]
    print(f"wrote {path}")
    for key in (
        "procedures_static",
        "procedures_configured",
        "procedures_config_disabled",
        "by_category",
        "sites_by_kind",
        "sites_disabled_by_guards",
        "function_sites_in_expressions",
        "unresolved_references",
        "undecided_guard_conditions",
        "initialization_only_procedures",
        "parse_failures",
        "duplicate_modules",
        "actions_with_procedures",
        "stage_attribution",
    ):
        print(f"  {key}: {summary.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
