#!/usr/bin/env python
"""The kernel work-item ledger: who is implementing what, right now.

``native/pi_cam/kernel_work_items.yaml`` is the single source of truth for
work in flight -- development coordination only, never scientific evidence.
An item's ``state`` says a kernel is being worked on (``in_progress``),
stuck (``blocked``, with the blocker named), or done working (``closed``);
whether the kernel is actually *finished* is always computed from the
validation ledgers, and the dashboard shows the two axes separately.

Rules this tool enforces:

- one active claim (in_progress or blocked) per kernel across the repository;
- an in_progress item carries owner, branch, stage, dates and the next gate;
- ``close`` is allowed only when the scientific record supports it -- the
  decoupling ledger for replacement and gate stages, existing replay/build
  evidence for the earlier stages, or an explicit validation record --
  otherwise the item must be ``block``ed instead;
- kernels are canonical ``module::routine`` ids from the observability
  inventory, and target processes are plan action ids from the ledger.

Usage:
    uv run python tools/kernel_work_items.py claim --kernel cloud_fraction::cldfrc_fice \\
        --process cam_run1.cloud_macro_microphysics --owner-class freecam.physics.macrophysics.Macrophysics \\
        --owner claude --branch physics-kernel-api-closure --stage in_model_replacement \\
        --next-gate "512-rank 50-step BFB" --note "..."
    uv run python tools/kernel_work_items.py update --kernel ... --stage bfb_gate --next-gate ...
    uv run python tools/kernel_work_items.py block --kernel ... --blocker "..."
    uv run python tools/kernel_work_items.py close --kernel ... [--evidence <validation record>]
    uv run python tools/kernel_work_items.py check
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPO / "native/pi_cam/kernel_work_items.yaml"
OBSERVABILITY = REPO / "validation/pi_cam_kernel_observability.json"
DECOUPLING = REPO / "validation/physics_kernel_decoupling.json"
VALIDATION = REPO / "validation"
SCHEMA_VERSION = 1

STATES = ("in_progress", "blocked", "closed")
STAGES = ("classification", "contract", "adapter", "capture_replay",
          "in_model_replacement", "bfb_gate", "performance_gate")
#: stages whose closure the decoupling ledger itself must witness
LEDGER_GATED_STAGES = ("capture_replay", "in_model_replacement", "bfb_gate")


class WorkItemError(RuntimeError):
    pass


def _today() -> str:
    return datetime.date.today().isoformat()


def load_ledger(path: Path | None = None) -> dict[str, Any]:
    path = path or LEDGER_PATH
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "items": []}
    payload = yaml.safe_load(path.read_text()) or {}
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise WorkItemError(f"{path}: unsupported schema_version {payload.get('schema_version')!r}")
    payload.setdefault("items", [])
    return payload


def save_ledger(payload: Mapping[str, Any], path: Path | None = None) -> None:
    (path or LEDGER_PATH).write_text(yaml.safe_dump(dict(payload), sort_keys=False, width=100))


def _known_kernels() -> set[str]:
    record = json.loads(OBSERVABILITY.read_text())
    return {k["qualified"] for k in record["kernels"]}


def _known_processes() -> set[str]:
    record = json.loads(DECOUPLING.read_text())
    return {a["id"] for a in record["actions"]}


def active_item(ledger: Mapping[str, Any], kernel: str) -> dict[str, Any] | None:
    for item in ledger["items"]:
        if item["kernel"] == kernel and item["state"] in ("in_progress", "blocked"):
            return item
    return None


def check_ledger(ledger: Mapping[str, Any], *, known_kernels: set[str] | None = None,
                 known_processes: set[str] | None = None) -> list[str]:
    """Every rule violation in the ledger; an empty list is a clean ledger."""

    problems: list[str] = []
    known_kernels = known_kernels if known_kernels is not None else _known_kernels()
    known_processes = known_processes if known_processes is not None else _known_processes()
    active: dict[str, int] = {}
    for index, item in enumerate(ledger["items"]):
        where = f"items[{index}] ({item.get('kernel', '?')})"
        kernel = item.get("kernel")
        if not kernel or kernel not in known_kernels:
            problems.append(f"{where}: kernel is not a canonical candidate id")
        if item.get("state") not in STATES:
            problems.append(f"{where}: state must be one of {STATES}")
        if item.get("stage") not in STAGES:
            problems.append(f"{where}: stage must be one of {STAGES}")
        for pid in item.get("target_processes") or []:
            if pid not in known_processes:
                problems.append(f"{where}: unknown target process {pid!r}")
        if item.get("state") in ("in_progress", "blocked"):
            active[kernel] = active.get(kernel, 0) + 1
            for field in ("owner", "branch", "started_at", "updated_at", "next_gate",
                          "target_processes", "owner_class"):
                if not item.get(field):
                    problems.append(f"{where}: an active item needs {field}")
        if item.get("state") == "blocked" and not item.get("blocker"):
            problems.append(f"{where}: a blocked item must name its blocker")
        if item.get("state") == "closed" and not item.get("closed_at"):
            problems.append(f"{where}: a closed item needs closed_at")
        for field in ("started_at", "updated_at", "closed_at"):
            value = item.get(field)
            if value is not None:
                try:
                    datetime.date.fromisoformat(str(value))
                except ValueError:
                    problems.append(f"{where}: {field} is not an ISO date")
    for kernel, count in active.items():
        if count > 1:
            problems.append(f"{kernel}: {count} active claims; one kernel takes at most one")
    return problems


def _tracked(kernel: str) -> Mapping[str, Any] | None:
    record = json.loads(DECOUPLING.read_text())
    routine = kernel.split("::")[-1]
    return next((k for k in record["kernels"] if (k.get("routine") or k["kernel"]) == routine), None)


def closure_obstacles(item: Mapping[str, Any], evidence: str | None) -> list[str]:
    """Why this item may not close yet: the scientific record must witness the stage."""

    stage = item["stage"]
    kernel = item["kernel"]
    tracked = _tracked(kernel)
    problems: list[str] = []
    if stage in LEDGER_GATED_STAGES:
        if tracked is None:
            problems.append(f"{kernel} is not tracked in the decoupling ledger; nothing witnesses {stage}")
            return problems
        if stage == "capture_replay":
            replays = [n for step in ("replay_full_chunk", "replay_single_column", "replay_public_api")
                       for n in tracked["evidence"].get(step, [])]
            if not replays:
                problems.append("no replay evidence in the decoupling ledger")
        else:
            gates = [g for g in tracked.get("in_model_gates") or [] if g.get("present") and g.get("bfb")]
            if not (tracked.get("validated_through_runner") and gates):
                problems.append("no present, bit-for-bit in-model gate in the decoupling ledger")
    else:
        if not evidence:
            problems.append(f"closing a {stage} item needs --evidence naming a validation record")
        elif not (VALIDATION / evidence).is_file():
            problems.append(f"validation/{evidence} does not exist")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_kernel(p):
        p.add_argument("--kernel", required=True, help="canonical module::routine id")

    claim = sub.add_parser("claim")
    add_kernel(claim)
    claim.add_argument("--process", action="append", required=True, dest="processes")
    claim.add_argument("--owner-class", required=True)
    claim.add_argument("--owner", required=True)
    claim.add_argument("--branch", required=True)
    claim.add_argument("--stage", required=True, choices=STAGES)
    claim.add_argument("--next-gate", required=True)
    claim.add_argument("--note", default=None)

    update = sub.add_parser("update")
    add_kernel(update)
    update.add_argument("--stage", choices=STAGES)
    update.add_argument("--next-gate")
    update.add_argument("--note")

    block = sub.add_parser("block")
    add_kernel(block)
    block.add_argument("--blocker", required=True)

    close = sub.add_parser("close")
    add_kernel(close)
    close.add_argument("--evidence", help="a validation record witnessing the stage, for pre-gate stages")
    close.add_argument("--note")

    sub.add_parser("check")

    args = parser.parse_args(argv)
    ledger = load_ledger()

    if args.command == "check":
        problems = check_ledger(ledger)
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{LEDGER_PATH.name}: {len(ledger['items'])} items, "
              f"{sum(1 for i in ledger['items'] if i['state'] == 'in_progress')} in progress, "
              f"{sum(1 for i in ledger['items'] if i['state'] == 'blocked')} blocked"
              + ("" if not problems else f", {len(problems)} problems"))
        return 1 if problems else 0

    if args.command == "claim":
        if args.kernel not in _known_kernels():
            raise SystemExit(f"{args.kernel} is not a canonical candidate id (see the observability inventory)")
        unknown = [p for p in args.processes if p not in _known_processes()]
        if unknown:
            raise SystemExit(f"unknown target processes: {unknown}")
        existing = active_item(ledger, args.kernel)
        if existing is not None:
            raise SystemExit(f"{args.kernel} already has an active claim "
                             f"(owner {existing['owner']}, branch {existing['branch']}, "
                             f"state {existing['state']}); one kernel takes at most one")
        item = {
            "kernel": args.kernel,
            "target_processes": list(args.processes),
            "owner_class": args.owner_class,
            "state": "in_progress",
            "stage": args.stage,
            "owner": args.owner,
            "branch": args.branch,
            "started_at": _today(),
            "updated_at": _today(),
            "next_gate": args.next_gate,
        }
        if args.note:
            item["note"] = args.note
        ledger["items"].append(item)
    else:
        item = active_item(ledger, args.kernel)
        if item is None:
            raise SystemExit(f"{args.kernel} has no active claim")
        if args.command == "update":
            if args.stage:
                item["stage"] = args.stage
            if args.next_gate:
                item["next_gate"] = args.next_gate
            if args.note:
                item["note"] = args.note
            item["state"] = "in_progress"
            item.pop("blocker", None)
        elif args.command == "block":
            item["state"] = "blocked"
            item["blocker"] = args.blocker
        elif args.command == "close":
            obstacles = closure_obstacles(item, args.evidence)
            if obstacles:
                for obstacle in obstacles:
                    print(obstacle, file=sys.stderr)
                raise SystemExit(f"{args.kernel} may not close; block it instead")
            item["state"] = "closed"
            item["closed_at"] = _today()
            if args.evidence:
                item["evidence"] = args.evidence
            if args.note:
                item["note"] = args.note
            item.pop("blocker", None)
        item["updated_at"] = _today()

    problems = check_ledger(ledger)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        raise SystemExit("the change would leave the ledger invalid; nothing was written")
    save_ledger(ledger)
    print(f"{args.command}: {args.kernel if hasattr(args, 'kernel') else ''} -> {LEDGER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
