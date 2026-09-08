#!/usr/bin/env python
"""Which candidate kernels a count-only entry can observe, and where the blind spots are.

Counting a kernel's executions without touching its machine code uses the same
link-time redirection the replacement hooks use: the defining object's symbol
is weakened and given an alias, and a generated tail-jump trampoline takes the
original name -- it bumps one rank-local counter for the current process
context and jumps to the alias with every register and stack byte untouched,
so no calling convention is guessed and no floating-point operation changes.

That entry counts exactly the calls that go through the symbol.  This tool
classifies every candidate of the call-tree inventory against the relocation
audit and the source call sites:

- ``entry: trampoline`` -- one symbol, one defining object; a counting
  trampoline can take the symbol.
- ``entry: none`` -- an internal procedure with no symbol of its own, or a
  symbol no text section references (every call site inlined or eliminated).

Coverage compares, per referencing object, the compiled call relocations with
the source call sites the inventory attributes to that object:

- ``full``      -- every object with source call sites shows at least as many
                   call relocations (the compiler kept every site out of line;
                   the build uses no cross-object inlining).
- ``partial``   -- some object has fewer relocations than source sites: those
                   sites were inlined or eliminated and are counting blind
                   spots.  Observed counts are then lower bounds.
- ``none``      -- no countable entry at all.

The record is deterministic and checked in CI:

    uv run python tools/build_pi_cam_kernel_observability.py
    uv run python tools/build_pi_cam_kernel_observability.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO = Path(__file__).resolve().parents[1]
CLOSURE = REPO / "validation/pi_cam_kernel_api_closure.json"
AUDIT = REPO / "validation/pi_cam_kernel_api_redirectable_calls.json"
HOOKS = REPO / "native/pi_cam/hooks.yaml"
RUNNERS = REPO / "native/pi_cam/segment_runners.yaml"
DEFAULT_OUTPUT = REPO / "validation/pi_cam_kernel_observability.json"
SCHEMA_VERSION = 1

#: Context slots of the counting table: slot 0 is initialization/unattributed,
#: slot 1 finalize/unattributed, slot 2 run/unattributed, slots 3.. are
#: assigned to workflow processes by the Python driver at run time.
RESERVED_SLOTS = ("initialization", "finalize", "run-unattributed")
TABLE_SLOTS = 64


def _object_of(source: str | None) -> str | None:
    """The archive member a source compiles into (one object per source file)."""

    if not source:
        return None
    stem = source.rsplit("/", 1)[-1]
    for suffix in (".F90", ".f90", ".F", ".f"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + ".o"
    return None


def _call_sites_by_object(candidate: Mapping[str, Any], procedures: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """Source call sites targeting the candidate, grouped by the caller's object."""

    sites: dict[str, int] = {}
    qualified = candidate["qualified"]
    for caller_name in candidate.get("callers") or []:
        caller = procedures.get(caller_name)
        if caller is None:
            continue
        obj = _object_of(caller.get("source"))
        if obj is None:
            continue
        count = sum(1 for site in caller.get("sites") or []
                    if site.get("target") == qualified or qualified in (site.get("candidates") or []))
        if count:
            sites[obj] = sites.get(obj, 0) + count
    return sites


def classify(candidate: Mapping[str, Any], audit_row: Mapping[str, Any] | None,
             procedures: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One candidate's counting entry, coverage, and blind spots."""

    qualified = candidate["qualified"]
    record: dict[str, Any] = {
        "qualified": qualified,
        "routine": candidate["name"],
        "module": candidate.get("module"),
        "kind": candidate.get("kind"),
        "host": candidate.get("host"),
        "source": candidate.get("source"),
        "processes": sorted(candidate.get("parent_actions") or []),
    }
    classification = (audit_row or {}).get("classification")
    record["relocation_classification"] = classification
    symbols = (audit_row or {}).get("symbols") or []
    defined = [(s["symbol"], obj) for s in symbols for obj in (s.get("defined_in") or {})]

    if classification in (None, "not-in-archive") or not defined:
        record.update(entry="none", coverage="none", blind_spots=[
            {"reason": "internal procedure absorbed into its host; it has no symbol a trampoline could take"}])
        return record
    if len(defined) != 1:
        record.update(entry="none", coverage="none", blind_spots=[
            {"reason": f"ambiguous symbols for one procedure: {sorted(s for s, _ in defined)}"}])
        return record

    symbol, defining_object = defined[0]
    record["symbol"] = symbol
    record["defining_object"] = defining_object
    site_counts = _call_sites_by_object(candidate, procedures)
    relocations = {ref["object"]: int(ref["relocations"])
                   for s in symbols for ref in (s.get("references") or [])}
    record["source_call_sites_by_object"] = dict(sorted(site_counts.items()))
    record["call_relocations_by_object"] = dict(sorted(relocations.items()))

    if classification == "no-call-relocation":
        record.update(entry="none", coverage="none", blind_spots=[
            {"reason": "no text section references the symbol: every call site was inlined or eliminated; "
                       "a trampoline would count nothing",
             "source_call_sites": sum(site_counts.values())}])
        return record

    blind: list[dict[str, Any]] = []
    for obj, sites in sorted(site_counts.items()):
        kept = relocations.get(obj, 0)
        if kept < sites:
            blind.append({"object": obj, "source_call_sites": sites, "call_relocations": kept,
                          "reason": "call sites without a matching relocation were inlined or eliminated; "
                                    "calls from them bypass the symbol and are not counted"})
    record.update(
        entry="trampoline",
        redirection="weaken-definition",
        coverage="partial" if blind else "full",
        blind_spots=blind,
    )
    return record


def build_observability(root: Path | str = REPO) -> dict[str, Any]:
    root = Path(root)
    closure = json.loads((root / CLOSURE.relative_to(REPO)).read_text())
    audit = json.loads((root / AUDIT.relative_to(REPO)).read_text())
    hooks = yaml.safe_load((root / HOOKS.relative_to(REPO)).read_text())
    runners = yaml.safe_load((root / RUNNERS.relative_to(REPO)).read_text())
    if closure.get("schema_version") != 1 or audit.get("schema_version") != 1:
        raise SystemExit("unsupported input schema")

    procedures = {p["qualified"]: p for p in closure["procedures"]}
    audit_by_q = {p["qualified"]: p for p in audit["procedures"]}
    candidates = sorted(
        (p for p in closure["procedures"]
         if p["category"] == "numeric_kernel" and p["in_configuration"] and not p["inert_in_configuration"]),
        key=lambda p: p["qualified"])

    hook_kernels = sorted(h["kernel"] for h in hooks.get("hooks") or [])
    replaceable = sorted({k["name"] for r in runners["runners"] for k in r.get("kernels") or []})

    entries = []
    for index, candidate in enumerate(candidates):
        record = classify(candidate, audit_by_q.get(candidate["qualified"]), procedures)
        record["index"] = index
        record["replacement_hook"] = record["routine"] in hook_kernels
        record["replaceable_in_runner"] = record["routine"] in replaceable
        entries.append(record)

    summary = {
        "candidates": len(entries),
        "by_entry": {kind: sum(1 for e in entries if e["entry"] == kind) for kind in ("trampoline", "none")},
        "by_coverage": {kind: sum(1 for e in entries if e["coverage"] == kind)
                        for kind in ("full", "partial", "none")},
        "blind_spot_objects": sum(len(e["blind_spots"]) for e in entries if e["entry"] == "trampoline"),
    }
    content = {
        "schema_version": SCHEMA_VERSION,
        "generator": "tools/build_pi_cam_kernel_observability.py",
        "what": ("For every candidate kernel of the configured call tree: whether a count-only tail-jump "
                 "trampoline can observe its executions, which call paths it covers, and which remain "
                 "blind spots.  A trampoline forwards every argument untouched (a tail jump preserves all "
                 "registers and the stack), so no calling convention is guessed; inlined call sites and "
                 "internal procedures without symbols stay uncovered by design -- this stage does not "
                 "disable inlining or recompile numerical code."),
        "reading": ("static classification against the linked oracle objects; observed counts from a "
                    "counting image are lower bounds wherever coverage is partial"),
        "table": {"slots": TABLE_SLOTS, "reserved_slots": list(RESERVED_SLOTS),
                  "layout": "int64 counts[candidate_index][slot]; slot meanings are assigned by the "
                            "driver at run time and written into the runtime coverage record"},
        "inputs": {
            "closure_inputs_hash": closure.get("inputs_hash"),
            "audit_archive_sha256": (audit.get("archive") or {}).get("sha256"),
        },
        "kernels": entries,
        "summary": summary,
    }
    content["content_hash"] = hashlib.sha256(
        json.dumps(content, sort_keys=True).encode()).hexdigest()
    return content


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    record = build_observability()
    if args.check:
        if not args.output.is_file():
            print(f"{args.output} does not exist", file=sys.stderr)
            return 1
        current = json.loads(args.output.read_text())
        if current.get("content_hash") != record["content_hash"]:
            print(f"{args.output} is stale", file=sys.stderr)
            return 1
        print(f"{args.output} is current ({record['content_hash'][:12]})")
        return 0
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
