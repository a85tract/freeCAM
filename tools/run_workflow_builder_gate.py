"""Run what the Workflow Builder generates, through the Driver the page runs it with.

The page applies a document to a live ``freecam.Driver``; the code it
generates does the same through ``configure(driver)``.  This harness takes a
document -- the validated default, or the default with a Python process that
does nothing -- generates its setup code with the same bundle the page uses,
executes that code, and drives the exact-online 50-step case through a
Driver inside the allocation: initialize, configure, run, close.  The output
is then compared with the oracle byte for byte, the run's timing read from
its own report, and whether the Python process was called counted from the
action trace.  The record says all of that, and nothing it did not measure.

Run inside a PBS allocation (the Driver launches its ranks locally there):

    python tools/run_workflow_builder_gate.py --variant default --root <dir> --record <json> \\
        --config configs/pi_cam_icesm131_exact_online_50step.yaml --reference <oracle CAM run> \\
        --seed-run <original CESM 50-step run> --boundary-oracle <x2a/a2x capture>

``--dry-run`` generates and shows the configuration code without a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

NOOP_SOURCE = '''class NoopProcess(fc.Physics):
    """Does nothing; proves the step reaches a Python process in this slot."""

    name = "noop_process"
    after = "dry_adjustment"
    reads = ()
    writes = ()

    def run(self, state, context):
        pass
'''


def build_document(variant: str, steps: int):
    from freecam.pi_cam.workflow_builder import (
        WorkflowEditSession, build_snapshot, catalog_entries, python_process_template,
    )
    from freecam.pi_cam.workflow_builder.document import WorkflowDocument

    snapshot = build_snapshot(root=REPO)
    default = WorkflowDocument.from_payload(snapshot["default_document"])
    entries = catalog_entries(default.nodes, root=REPO)
    session = WorkflowEditSession(default, entries, python_template=python_process_template)
    session.apply({"operation": "set_nsteps", "nsteps": int(steps)})
    if variant == "noop-python":
        session.apply({"operation": "add_python", "name": "noop_process", "after": "cam_run1.dry_adjustment",
                       "source": NOOP_SOURCE})
    elif variant != "default":
        raise SystemExit(f"unknown variant {variant!r}")
    return snapshot, session.default_document, session.document


def generate_setup(document, snapshot) -> tuple[str, list[str]]:
    """The setup code and the generator's own list of changes against the default."""

    from freecam.pi_cam.workflow_builder import codegen

    if not codegen.available():
        raise SystemExit("the generator bundle is not built or node is missing; run `npm run build` under web/")
    artifacts = codegen.generate(document, snapshot)
    return artifacts.setup, list(artifacts.changes)


def load_configure(setup: str):
    import freecam as fc

    namespace: dict[str, Any] = {"__name__": "workflow_builder_gate", "fc": fc}
    exec(compile(setup, "<generated setup>", "exec"), namespace)   # noqa: S102 -- the generated code, on purpose
    return namespace["configure"]


def step_seconds(run_dir: Path) -> dict[str, float | None]:
    stats = run_dir / "timing" / "freecam_timing_stats"
    values: dict[str, float | None] = {"step_total_wallavg": None, "step_total_wallmax": None}
    if not stats.is_file():
        return values
    for line in stats.read_text().splitlines():
        parts = line.split()
        if len(parts) > 8 and parts[0] == "FREECAM:TOTAL/FREECAM:STEP":
            values["step_total_wallavg"] = float(parts[-1])
            values["step_total_wallmax"] = float(parts[4])
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=("default", "noop-python"), default="default")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--root", type=Path, help="where the run directories go")
    parser.add_argument("--config", type=Path, default=REPO / "configs" / "pi_cam_icesm131_exact_online_50step.yaml")
    parser.add_argument("--reference", type=Path, help="the oracle CAM run the output is compared with")
    parser.add_argument("--seed-run", type=Path, help="the original CESM run the online provider seeds from")
    parser.add_argument("--boundary-oracle", type=Path, help="captured x2a/a2x to check every exchange against")
    parser.add_argument("--record", type=Path, help="JSON record to append this variant to")
    parser.add_argument("--pbs-job-id")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)

    snapshot, _default, document = build_document(arguments.variant, arguments.steps)
    setup, changes = generate_setup(document, snapshot)
    configure = load_configure(setup)
    print(f"variant {arguments.variant}: workflow {document.workflow_hash[:12]}, "
          f"{len(setup.splitlines())} lines of setup code", file=sys.stderr)
    if arguments.dry_run:
        print(setup)
        return 0
    for name in ("root", "reference", "seed_run"):
        if getattr(arguments, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required without --dry-run")

    import freecam as fc
    from freecam.pi_cam.validation import compare_pi_cam_directories

    root = arguments.root / arguments.variant
    run_dir = root / "run"
    driver = fc.Driver(
        case="PI-atm", nsteps=arguments.steps, config=arguments.config, run_dir=run_dir,
        online_seed_run=arguments.seed_run, online_oracle=arguments.boundary_oracle,
        trace_limit=None,
    )
    started = time.time()
    driver.initialize()
    initialized = time.time()
    configure(driver)
    configured = time.time()
    order_after = [row["name"] for row in driver.cam.workflow.describe()]
    result = driver.run()
    finished = time.time()
    status = dict(driver.status)
    trace = list(driver.trace)
    python_processes = list(status.get("python_processes", ()))
    calls = sum(1 for record in trace if record.get("name") == "noop_process")
    driver.close()
    closed = time.time()

    bfb = compare_pi_cam_directories(arguments.reference, run_dir).to_payload()
    record = {
        "variant": arguments.variant,
        "pbs_job_id": arguments.pbs_job_id,
        "steps": arguments.steps,
        "workflow_hash": document.workflow_hash,
        "catalog_hash": snapshot["catalog_hash"],
        "setup_sha256": hashlib.sha256(setup.encode()).hexdigest(),
        "changes_against_the_default": changes,
        "configure_changed_the_model": bool(changes),
        "python_processes_installed": python_processes,
        "noop_process_calls_in_trace": calls if arguments.variant == "noop-python" else None,
        "trace_actions": len(trace),
        "workflow_after_configure": order_after,
        "run_result": str(result)[:500],
        "seconds": {
            "initialize": round(initialized - started, 3),
            "configure": round(configured - initialized, 3),
            "run_wall": round(finished - configured, 3),
            "close": round(closed - finished, 3),
            **step_seconds(run_dir),
        },
        "run_dir": str(run_dir),
        "bfb": bfb,
    }
    print(json.dumps({k: v for k, v in record.items() if k not in ("bfb", "workflow_after_configure")}, indent=1))
    print("bfb:", bfb.get("bfb"), "| compared", bfb.get("compared_files"))
    if arguments.record:
        existing = json.loads(arguments.record.read_text()) if arguments.record.is_file() else {
            "schema_version": 1,
            "what": ("The Workflow Builder's generated configuration run through freecam.Driver -- the path the "
                     "page runs -- on the exact-online 50-step case, compared byte for byte with the oracle."),
            "variants": {},
        }
        existing["variants"][arguments.variant] = record
        arguments.record.parent.mkdir(parents=True, exist_ok=True)
        arguments.record.write_text(json.dumps(existing, indent=2, default=str) + "\n")
    return 0 if bfb.get("bfb") else 1


if __name__ == "__main__":
    raise SystemExit(main())
