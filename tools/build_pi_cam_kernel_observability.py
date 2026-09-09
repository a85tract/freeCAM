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

Coverage is an auditable per-call-site map, not a count comparison: every
source call site the inventory attributes to an object must match a call
relocation whose debug-line entry falls on that site's line (continuation
lines allowed), and the record keeps the map -- site line, relocation offset,
relocation line -- per object:

- ``full``      -- every source call site has its own mapped relocation.
- ``partial``   -- some site has no mapped relocation (inlined or eliminated;
                   its calls bypass the counter, observed totals are lower
                   bounds), or an object carries no debug line table and the
                   map cannot be audited.
- ``none``      -- no countable entry at all.

The record is deterministic and checked in CI:

    uv run python tools/build_pi_cam_kernel_observability.py
    uv run python tools/build_pi_cam_kernel_observability.py --check
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO = Path(__file__).resolve().parents[1]
CLOSURE = REPO / "validation/pi_cam_kernel_api_closure.json"
AUDIT = REPO / "validation/pi_cam_kernel_api_redirectable_calls.json"
HOOKS = REPO / "native/pi_cam/hooks.yaml"
RUNNERS = REPO / "native/pi_cam/segment_runners.yaml"
AUDIT_RECORD = REPO / "validation/pi_cam_kernel_api_redirectable_calls.json"
DEFAULT_OUTPUT = REPO / "validation/pi_cam_kernel_observability.json"
DEFAULT_WORK = REPO / "build/pi_cam_observability_work"
#: a Fortran call statement can span continuation lines; the call instruction's
#: debug line must fall inside this window below the statement's first line
CONTINUATION_WINDOW = 40
SCHEMA_VERSION = 1

#: Context slots of the counting table: slot 0 is initialization/unattributed,
#: slot 1 finalize/unattributed, slot 2 run/unattributed, slots 3.. are
#: assigned to workflow processes by the Python driver at run time.
RESERVED_SLOTS = ("initialization", "finalize", "run-unattributed")
TABLE_SLOTS = 64


class ObjectReader:
    """Cached debug-line tables and call relocations of the oracle archive's objects.

    The line table (compiled with -debug minimal) maps every text offset to its
    source line, so each call relocation is attributed to the exact source line
    of its call instruction -- the auditable half of the site-to-relocation map.
    """

    def __init__(self, archive: Path, work: Path) -> None:
        self.archive = archive
        self.work = work
        self.work.mkdir(parents=True, exist_ok=True)
        self._lines: dict[str, list[tuple[int, int]]] = {}
        self._relocations: dict[str, dict[str, list[int]]] = {}

    def _extract(self, object_name: str) -> Path | None:
        target = self.work / object_name
        if not target.is_file():
            result = subprocess.run(["ar", "x", str(self.archive), object_name],
                                    cwd=self.work, capture_output=True, text=True)
            if result.returncode != 0 or not target.is_file():
                return None
        return target

    def decoded_lines(self, object_name: str) -> list[tuple[int, int]]:
        if object_name not in self._lines:
            rows: list[tuple[int, int]] = []
            target = self._extract(object_name)
            if target is not None:
                out = subprocess.run(["readelf", "--debug-dump=decodedline", str(target)],
                                     capture_output=True, text=True).stdout
                for line in out.splitlines():
                    match = re.match(r"\S+\s+(\d+)\s+(0x[0-9a-f]+|0)\b", line.strip())
                    if match:
                        address = int(match.group(2), 16) if match.group(2) != "0" else 0
                        rows.append((address, int(match.group(1))))
                rows.sort()
            self._lines[object_name] = rows
        return self._lines[object_name]

    def call_relocations(self, object_name: str) -> dict[str, list[int]]:
        if object_name not in self._relocations:
            table: dict[str, list[int]] = {}
            target = self._extract(object_name)
            if target is not None:
                out = subprocess.run(["readelf", "-rW", str(target)],
                                     capture_output=True, text=True).stdout
                section = ""
                for line in out.splitlines():
                    if line.startswith("Relocation section"):
                        section = line.split("'")[1]
                        continue
                    parts = line.split()
                    if section.startswith(".rela.text") and len(parts) >= 5 \
                            and parts[0].strip("0123456789abcdef") == "" and parts[2] != "Type":
                        table.setdefault(parts[4], []).append(int(parts[0], 16))
            self._relocations[object_name] = table
        return self._relocations[object_name]

    def line_of(self, object_name: str, offset: int) -> int | None:
        rows = self.decoded_lines(object_name)
        addresses = [address for address, _ in rows]
        index = bisect.bisect_right(addresses, offset) - 1
        return rows[index][1] if index >= 0 else None


def map_call_sites(site_lines: list[int], relocation_lines: list[tuple[int, int | None]]) -> tuple[list[dict], list[int], list[dict]]:
    """Greedy site-to-relocation assignment by source line.

    A site matches the relocation whose debug line equals its own line, or falls
    within the continuation window below it.  Returns (matched map, unmatched
    site lines, extra relocations) -- an unmatched site is a blind spot, an
    extra relocation (a compiler-duplicated call) is recorded, never counted as
    coverage of anything.
    """

    available = sorted(relocation_lines, key=lambda item: (item[1] is None, item[1] or 0, item[0]))
    used = [False] * len(available)
    matched: list[dict] = []
    unmatched: list[int] = []
    for site in sorted(site_lines):
        best = None
        for index, (offset, line) in enumerate(available):
            if used[index] or line is None:
                continue
            if site <= line <= site + CONTINUATION_WINDOW:
                if best is None or line < available[best][1]:
                    best = index
        if best is None:
            unmatched.append(site)
        else:
            used[best] = True
            offset, line = available[best]
            matched.append({"site_line": site, "relocation_offset": f"0x{offset:x}", "relocation_line": line})
    extras = [{"relocation_offset": f"0x{offset:x}", "relocation_line": line}
              for index, (offset, line) in enumerate(available) if not used[index]]
    return matched, unmatched, extras


def _object_of(source: str | None) -> str | None:
    """The archive member a source compiles into (one object per source file)."""

    if not source:
        return None
    stem = source.rsplit("/", 1)[-1]
    for suffix in (".F90", ".f90", ".F", ".f"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] + ".o"
    return None


def _call_site_lines_by_object(candidate: Mapping[str, Any],
                               procedures: Mapping[str, Mapping[str, Any]]) -> dict[str, list[int]]:
    """Source call-site lines targeting the candidate, grouped by the caller's object."""

    sites: dict[str, list[int]] = {}
    qualified = candidate["qualified"]
    for caller_name in candidate.get("callers") or []:
        caller = procedures.get(caller_name)
        if caller is None:
            continue
        obj = _object_of(caller.get("source"))
        if obj is None:
            continue
        for site in caller.get("sites") or []:
            if site.get("target") == qualified or qualified in (site.get("candidates") or []):
                sites.setdefault(obj, []).append(int(site["line"]))
    return sites


def classify(candidate: Mapping[str, Any], audit_row: Mapping[str, Any] | None,
             procedures: Mapping[str, Mapping[str, Any]], reader: "ObjectReader | None") -> dict[str, Any]:
    """One candidate's counting entry, coverage, blind spots, and site map."""

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
    site_lines = _call_site_lines_by_object(candidate, procedures)
    relocations = {ref["object"]: int(ref["relocations"])
                   for s in symbols for ref in (s.get("references") or [])}
    record["source_call_sites_by_object"] = {obj: len(lines) for obj, lines in sorted(site_lines.items())}
    record["call_relocations_by_object"] = dict(sorted(relocations.items()))

    if classification == "no-call-relocation":
        record.update(entry="none", coverage="none", blind_spots=[
            {"reason": "no text section references the symbol: every call site was inlined or eliminated; "
                       "a trampoline would count nothing",
             "source_call_sites": sum(len(v) for v in site_lines.values())}])
        return record

    blind: list[dict[str, Any]] = []
    site_map: dict[str, dict[str, Any]] = {}
    for obj, lines in sorted(site_lines.items()):
        offsets = (reader.call_relocations(obj) if reader else {}).get(symbol, [])
        if reader is None or (offsets and not reader.decoded_lines(obj)):
            blind.append({"object": obj, "source_call_sites": len(lines),
                          "reason": "no debug line table: the site-to-relocation map cannot be audited"})
            continue
        relocation_lines = [(offset, reader.line_of(obj, offset)) for offset in offsets]
        matched, unmatched, extras = map_call_sites(lines, relocation_lines)
        site_map[obj] = {"matched": matched}
        if extras:
            site_map[obj]["extra_relocations"] = extras
        for site in unmatched:
            blind.append({"object": obj, "site_line": site,
                          "reason": "no call relocation maps to this site's line: the call was inlined "
                                    "or eliminated and bypasses the counter"})
    record["call_site_map"] = site_map
    record.update(
        entry="trampoline",
        redirection="weaken-definition",
        coverage="partial" if blind else "full",
        blind_spots=blind,
    )
    return record


def build_observability(root: Path | str = REPO, *, archive: Path | None = None,
                        work: Path | None = None) -> dict[str, Any]:
    root = Path(root)
    closure = json.loads((root / CLOSURE.relative_to(REPO)).read_text())
    audit = json.loads((root / AUDIT.relative_to(REPO)).read_text())
    archive = archive or Path((audit.get("archive") or {}).get("path", ""))
    reader = ObjectReader(archive, work or DEFAULT_WORK) if archive and archive.is_file() else None
    if reader is None:
        raise SystemExit(f"the oracle archive is unavailable ({archive}); the site-to-relocation "
                         f"map cannot be audited without it")
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
        record = classify(candidate, audit_by_q.get(candidate["qualified"]), procedures, reader)
        record["index"] = index
        record["replacement_hook"] = record["routine"] in hook_kernels
        record["replaceable_in_runner"] = record["routine"] in replaceable
        entries.append(record)

    summary = {
        "candidates": len(entries),
        "by_entry": {kind: sum(1 for e in entries if e["entry"] == kind) for kind in ("trampoline", "none")},
        "by_coverage": {kind: sum(1 for e in entries if e["coverage"] == kind)
                        for kind in ("full", "partial", "none")},
        "blind_spots": sum(len(e["blind_spots"]) for e in entries if e["entry"] == "trampoline"),
        "mapped_call_sites": sum(len(obj_map["matched"]) for e in entries
                                 for obj_map in (e.get("call_site_map") or {}).values()),
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
