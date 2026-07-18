#!/usr/bin/env python3
"""Generate the Python/native contract from the pinned CCPP suite metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ERROR_NAMES = {"ccpp_error_message", "ccpp_error_code"}


def parse_tables(path: Path) -> dict[str, list[dict[str, str]]]:
    tables: dict[str, list[dict[str, str]]] = {}
    current_table: str | None = None
    current_variable: dict[str, str] | None = None
    expecting_table_name = False

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "[ccpp-arg-table]":
            current_table = None
            current_variable = None
            expecting_table_name = True
            continue
        if line == "[ccpp-table-properties]":
            current_table = None
            current_variable = None
            expecting_table_name = False
            continue
        section = re.fullmatch(r"\[\s*([^]]+?)\s*\]", line)
        if section:
            if current_table is not None:
                current_variable = {"local_name": section.group(1)}
                tables[current_table].append(current_variable)
            continue
        if "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if expecting_table_name and key == "name":
            current_table = value
            tables.setdefault(current_table, [])
            current_variable = None
            expecting_table_name = False
        elif current_variable is not None:
            current_variable[key] = value
    return tables


def discover_metadata(schemes_root: Path) -> dict[str, tuple[Path, list[dict[str, str]]]]:
    discovered: dict[str, tuple[Path, list[dict[str, str]]]] = {}
    for path in sorted(schemes_root.rglob("*.meta")):
        for table, variables in parse_tables(path).items():
            discovered[table] = (path, variables)
    return discovered


def normalized_argument(argument: dict[str, str]) -> dict[str, object]:
    dimensions = argument.get("dimensions", "()").strip()
    dim_values = [
        item.strip()
        for item in dimensions.strip("()").split(",")
        if item.strip()
    ]
    raw_type = argument.get("type", "unknown")
    base_type = raw_type.split("|", 1)[0].strip()
    return {
        "local_name": argument["local_name"],
        "standard_name": argument.get("standard_name", argument["local_name"]),
        "type": base_type,
        "kind": argument.get("kind", raw_type.split("kind =", 1)[-1].strip() if "kind =" in raw_type else ""),
        "dimensions": dim_values,
        "intent": argument.get("intent", "in"),
        "persistence": argument.get("persistence", ""),
    }


def generate(cam_sima: Path, output: Path) -> dict[str, object]:
    suite = cam_sima / "src/physics/ncar_ccpp/suites/suite_kessler.xml"
    schemes_root = cam_sima / "src/physics/ncar_ccpp/schemes"
    root = ET.parse(suite).getroot()
    metadata = discover_metadata(schemes_root)
    groups: dict[str, list[dict[str, object]]] = {}

    for group in root.findall("group"):
        entries: list[dict[str, object]] = []
        for scheme_node in group.findall("scheme"):
            scheme = (scheme_node.text or "").strip()
            table = f"{scheme}_run"
            if table not in metadata:
                raise RuntimeError(f"CCPP metadata table {table!r} was not found")
            meta_path, arguments = metadata[table]
            entries.append(
                {
                    "name": scheme,
                    "subroutine": table,
                    "metadata": str(meta_path.relative_to(cam_sima)),
                    "arguments": [
                        normalized_argument(arg)
                        for arg in arguments
                        if arg.get("standard_name") not in ERROR_NAMES
                    ],
                }
            )
        groups[group.attrib["name"]] = entries

    contract: dict[str, object] = {
        "abi_version": 1,
        "cam_sima_commit": "f8daa568eae2696b7c4ebff7768f02f5d097d9df",
        "suite": "kessler",
        "suite_xml": str(suite.relative_to(cam_sima)),
        "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
        "groups": groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-sima", type=Path, default=Path("external/CAM-SIMA"))
    parser.add_argument("--output", type=Path, default=Path("native/generated/contract.json"))
    args = parser.parse_args()
    contract = generate(args.cam_sima.resolve(), args.output.resolve())
    groups = contract["groups"]
    assert isinstance(groups, dict)
    print("generated", args.output)
    print("before", len(groups["physics_before_coupler"]))
    print("after", len(groups["physics_after_coupler"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
