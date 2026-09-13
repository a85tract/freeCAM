#!/usr/bin/env python
"""Which numeric kernels of the PI-atm call tree a hook can reach.

A hook redirects one call by renaming the callee's symbol in a caller object's
relocation table (``tools/build_pi_cam_devices.py``, ``native/pi_cam/hooks.yaml``).
That needs a relocation: a call the compiler left as a call.  A routine the
compiler inlined at every site has no relocation and no hook can reach it; a
routine reached only through a pausable runner's pause (``virtem``) needs none.

This tool reads the oracle archive the image links -- the very objects whose
machine code the hooks leave untouched -- and records, for every numeric kernel
of the call-tree inventory, the objects defining its symbol and the objects
whose text sections reference it.  It is a static reading of the linked objects:
a relocation says a call site exists, not that this configuration executes it;
the call-tree inventory says the latter.

Usage:
    uv run python tools/audit_pi_cam_call_relocations.py            # write the record
    uv run python tools/audit_pi_cam_call_relocations.py --check    # the record matches the archive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CLOSURE = REPO / "validation/pi_cam_kernel_api_closure.json"
DEFAULT_MANIFEST = REPO / "build/pi_cam_promoted/native_cam_manifest.json"
DEFAULT_OUTPUT = REPO / "validation/pi_cam_kernel_api_redirectable_calls.json"
DEFAULT_WORK = REPO / "build/pi_cam_relocation_audit"
SCHEMA_VERSION = 1

#: Sections whose relocations are calls or references from machine code.  Debug,
#: trace and unwind sections also name the symbol and are not calls (the
#: ``compute_alpha`` finding: inlined, referenced only from ``.rela.trace``).
TEXT_SECTIONS = re.compile(r"^\.rela\.text")

CLASSIFICATIONS = {
    "rename-references": "every reference comes from another object: rename the callee's symbol in those objects' copies",
    "weaken-definition": "a reference comes from the defining object itself: weaken the definition, give it a second name, and let the hook take the original name",
    "no-call-relocation": "the symbol is defined but no text section references it: inlined at every call site, or unreferenced by the linked objects",
    "not-in-archive": "no object defines a symbol for the procedure: an internal procedure the compiler absorbed into its host, or a name the archive does not carry",
}


def _run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_archive(archive: Path, work: Path) -> list[str]:
    """Extract every object of ``archive`` into ``work``; return the member names."""

    work.mkdir(parents=True, exist_ok=True)
    members = [line.strip() for line in _run(["ar", "t", str(archive)]).splitlines() if line.strip()]
    _run(["ar", "x", str(archive)], cwd=work)
    missing = [member for member in members if not (work / member).is_file()]
    if missing:
        raise RuntimeError(f"archive members were not extracted: {missing[:5]}")
    return members


def defined_functions(path: Path) -> dict[str, dict[str, object]]:
    """Function symbols an object defines: name -> binding and size."""

    functions: dict[str, dict[str, object]] = {}
    for line in _run(["readelf", "-sW", str(path)]).splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[3] != "FUNC" or parts[6] == "UND":
            continue
        functions[parts[7]] = {"binding": parts[4], "size": int(parts[2], 0)}
    return functions


def text_references(path: Path) -> dict[str, Counter]:
    """Symbols an object's text sections reference: name -> relocation type counts."""

    references: dict[str, Counter] = defaultdict(Counter)
    section = ""
    for line in _run(["readelf", "-rW", str(path)]).splitlines():
        if line.startswith("Relocation section"):
            section = line.split("'")[1]
            continue
        if not TEXT_SECTIONS.match(section):
            continue
        parts = line.split()
        if len(parts) < 5 or not parts[0].strip("0123456789abcdef") == "" or parts[2] == "Type":
            continue
        references[parts[4]][parts[2]] += 1
    return references


def scan_archive(archive: Path, work: Path) -> tuple[list[str], dict[str, dict[str, dict]], dict[str, dict[str, Counter]]]:
    """Every object's defined functions and text references."""

    members = extract_archive(archive, work)
    definitions: dict[str, dict[str, dict]] = defaultdict(dict)     # symbol -> object -> {binding, size}
    references: dict[str, dict[str, Counter]] = defaultdict(dict)   # symbol -> object -> types
    for member in members:
        path = work / member
        for name, record in defined_functions(path).items():
            definitions[name][member] = record
        for name, counts in text_references(path).items():
            references[name][member] = counts
    return members, definitions, references


def procedure_symbols(procedure: dict, definitions: dict[str, dict[str, dict]]) -> list[str]:
    """The symbols an ifort object would carry for a call-tree procedure."""

    name = procedure["name"].lower()
    module = (procedure.get("module") or "").lower()
    host = procedure.get("host")
    if host:
        # an internal procedure: ifort names it <module>_mp_<host>_ip_<name>_ when it keeps one
        host_name = host.split("::")[-1].lower()
        pattern = re.compile(rf"^({re.escape(module)}_mp_)?{re.escape(host_name)}_ip_{re.escape(name)}_?$")
        return sorted(symbol for symbol in definitions if pattern.match(symbol))
    if module:
        return [f"{module}_mp_{name}_"]
    return [f"{name}_"]


def classify(symbol_records: list[dict]) -> str:
    if not symbol_records or all(not record["defined_in"] for record in symbol_records):
        return "not-in-archive"
    references = [ref for record in symbol_records for ref in record["references"]]
    if not references:
        return "no-call-relocation"
    definers = {obj for record in symbol_records for obj in record["defined_in"]}
    if any(ref["object"] in definers for ref in references):
        return "weaken-definition"
    return "rename-references"


def audit(closure: dict, archive: Path, work: Path) -> dict:
    members, definitions, references = scan_archive(archive, work)
    # the candidates: numeric kernels this configuration reaches (the inventory's 601)
    procedures = [p for p in closure["procedures"]
                  if p["category"] == "numeric_kernel" and p["in_configuration"] and not p["inert_in_configuration"]]
    sources = {p["qualified"]: p["source"] for p in closure["procedures"]}
    entries = []
    for procedure in sorted(procedures, key=lambda p: p["qualified"]):
        callers = procedure.get("callers") or []
        symbol_records = []
        for symbol in procedure_symbols(procedure, definitions):
            defined_in = {obj: definitions[symbol][obj] for obj in sorted(definitions.get(symbol, {}))}
            refs = [{"object": obj, "relocations": sum(counts.values()), "types": sorted(counts)}
                    for obj, counts in sorted(references.get(symbol, {}).items())]
            symbol_records.append({"symbol": symbol, "defined_in": defined_in, "references": refs})
        classification = classify(symbol_records)
        entries.append({
            "qualified": procedure["qualified"],
            "kind": procedure["kind"],
            "host": procedure.get("host"),
            "public": procedure["public"],
            "source": procedure["source"],
            "source_callers": len(callers),
            # callers compiled in the same object are where ifort inlines at -O2
            "callers_in_same_source": sum(1 for caller in callers if sources.get(caller) == procedure["source"]),
            "symbols": symbol_records,
            "classification": classification,
            "redirectable": classification in ("rename-references", "weaken-definition"),
        })
    by_class = Counter(entry["classification"] for entry in entries)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "tools/audit_pi_cam_call_relocations.py",
        "what": ("For every numeric kernel of the call-tree inventory that this configuration reaches, the "
                 "objects of the oracle archive defining its symbol and the objects whose text sections "
                 "reference it.  A hook can redirect a call only where such a reference exists."),
        "reading": "static: a relocation is a compiled call site, not proof this configuration executes it",
        "archive": {"path": str(archive), "sha256": _sha256(archive), "objects": len(members)},
        "closure_inputs_hash": closure.get("inputs_hash"),
        "classifications": CLASSIFICATIONS,
        "summary": {
            "procedures": len(entries),
            "redirectable": sum(1 for entry in entries if entry["redirectable"]),
            "by_classification": dict(sorted(by_class.items())),
            "no_call_relocation_with_every_caller_in_the_same_source": sum(
                1 for entry in entries
                if entry["classification"] == "no-call-relocation"
                and entry["source_callers"] and entry["callers_in_same_source"] == entry["source_callers"]),
            "internal_procedures": sum(1 for entry in entries if entry["host"]),
            "private_procedures": sum(1 for entry in entries if not entry["public"]),
        },
        "procedures": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--closure", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="image manifest naming the numerical archive (numerical_archive.path)")
    parser.add_argument("--archive", type=Path, help="the oracle archive itself; overrides --manifest")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK, help="where the objects are extracted")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail when the record differs from a fresh audit")
    args = parser.parse_args(argv)

    closure = json.loads(args.closure.read_text())
    archive = args.archive
    if archive is None:
        archive = Path(json.loads(args.manifest.read_text())["numerical_archive"]["path"])
    record = audit(closure, archive, args.work)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not args.output.is_file():
            print(f"{args.output} is absent", file=sys.stderr)
            return 1
        current = json.loads(args.output.read_text())
        if current != record:
            print(f"{args.output} differs from a fresh audit of {archive}", file=sys.stderr)
            return 1
        print(f"{args.output} is current: {record['summary']}")
        return 0
    args.output.write_text(text)
    print(json.dumps(record["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
