#!/usr/bin/env python3
"""Emit the physics-buffer field table a Python-driven routine reads.

A transliterated driver reaches CAM's physics buffer by index: the module
integer the Fortran routine itself uses.  Macrophysics' thirty fields were
typed by hand; microphysics reads sixty-five, so the table is read off the
pinned source instead.  For every ``pbuf_get_field(pbuf, <idx>, <ptr>, ...)``
in the routine this records the field's registered name (from the module's
``pbuf_add_field('NAME', ..., <idx>)`` or ``<idx> = pbuf_get_index('NAME')``),
the module symbol holding the index, whether the routine reads the older
time sample (``start=`` with ``itim_old``), and the rank and kind it declares
for the field -- which is what the rank-aware accessor needs.

    tools/generate_pi_cam_pbuf_table.py --routine micro_mg_cam_tend \\
        --source physics/cam/micro_mg_cam.F90 --module micro_mg_cam \\
        --output native/pi_cam/pbuf_fields_micro.yaml
    tools/generate_pi_cam_pbuf_table.py ... --check      # fail if stale

An index variable the module neither registers nor looks up is refused
rather than guessed; ``--index-module NAME=module`` names where such an
index lives (``physpkg`` owns ``prec_str_idx``, say).
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAM_SRC = REPO / "external/iCESM1.3.1_fzhu/components/cam/src"

KIND = {"real(r8)": "float64", "integer": "int32", "logical": "int32"}


def statements(lines: list[str], first: int, last: int) -> list[tuple[int, str]]:
    """(first_line, text) with continuations joined and comments stripped."""

    out, buffer, start = [], "", None
    for number in range(first, last + 1):
        text = lines[number - 1].split("!")[0].rstrip()
        if not text.strip():
            continue
        if start is None:
            start = number
        text = (buffer + " " + text.strip()) if buffer else text
        if text.rstrip().endswith("&"):
            buffer = text.rstrip()[:-1]
            continue
        out.append((start, text.strip()))
        buffer, start = "", None
    return out


def routine_span(lines: list[str], name: str) -> tuple[int, int]:
    first = next(i for i, l in enumerate(lines, 1)
                 if re.match(rf"\s*subroutine\s+{name}\s*\(", l, re.I))
    last = next(i for i, l in enumerate(lines, 1)
                if i > first and re.match(rf"\s*end\s+subroutine\s+{name}\b", l, re.I))
    return first, last


def registered_names(source: Path) -> dict[str, str]:
    """index variable -> registered pbuf name, over one module's text."""

    lines = source.read_text(errors="ignore").splitlines()
    names: dict[str, str] = {}
    for _, text in statements(lines, 1, len(lines)):
        # CAM pads some registered names ('CLDFSNOW '); the name is the trimmed one
        for match in re.finditer(r"pbuf_add_field\s*\(\s*['\"]\s*(\w+)\s*['\"]\s*,.*?,\s*(\w+_idx)\s*[,)]", text):
            names[match.group(2)] = match.group(1)
        for match in re.finditer(r"(\w+_idx)\s*=\s*pbuf_get_index\s*\(\s*['\"]\s*(\w+)\s*['\"]", text):
            names[match.group(1)] = match.group(2)
    return names


def pointer_shapes(body: list[tuple[int, str]]) -> dict[str, tuple[int, str]]:
    """pointer name -> (rank, dtype) from the routine's declarations."""

    shapes: dict[str, tuple[int, str]] = {}
    for _, text in body:
        match = re.match(r"(real\(r8\)|integer|logical)\s*,\s*pointer\b(.*?)::\s*(.*)$", text, re.I)
        if not match:
            continue
        kind = KIND[match.group(1).lower()]
        attributes, names = match.group(2), match.group(3)
        dimension = re.search(r"dimension\s*\(([^)]*)\)", attributes, re.I)
        default_rank = dimension.group(1).count(",") + 1 if dimension else None
        for item in re.finditer(r"(\w+)\s*(?:\(([^)]*)\))?", names):
            name, dims = item.group(1), item.group(2)
            if name.lower() in ("null", "dimension"):
                continue
            rank = dims.count(",") + 1 if dims else default_rank
            if rank is not None:
                shapes[name.lower()] = (rank, kind)
    return shapes


def build_table(routine: str, source: Path, module: str,
                index_modules: dict[str, str], only: set[str] | None) -> dict:
    lines = source.read_text(errors="ignore").splitlines()
    first, last = routine_span(lines, routine)
    body = statements(lines, first, last)
    names = registered_names(source)
    for symbol_module in set(index_modules.values()):
        extra = next(CAM_SRC.rglob(f"{symbol_module}.F90"), None)
        if extra is not None:
            names.update(registered_names(extra))
    shapes = pointer_shapes(body)
    rows: dict[str, dict] = {}
    for number, text in body:
        # a read guarded on one line -- `if (qrain_idx > 0) call pbuf_get_field(...)`
        # -- is a read of a field the configuration may not register
        match = re.match(r"(?:if\s*\([^)]*\)\s*)?call\s+pbuf_get_field\s*\(\s*pbuf\s*,\s*(\w+)\s*,"
                         r"\s*(\w+)(.*)\)\s*$", text, re.I)
        if not match:
            continue
        idx, pointer, rest = match.group(1), match.group(2).lower(), match.group(3)
        if only is not None and idx not in only:
            continue
        if idx in rows:
            continue
        if idx not in names:
            raise SystemExit(
                f"{routine}:{number}: {idx} is neither registered nor looked up in "
                f"{source.name}; name its module with --index-module {idx}=<module>")
        if pointer not in shapes:
            raise SystemExit(f"{routine}:{number}: no pointer declaration found for {pointer}")
        rank, dtype = shapes[pointer]
        owner = index_modules.get(idx, module)
        rows[idx] = {
            "name": names[idx],
            "symbol": f"{owner}_mp_{idx}_",
            # a start= that indexes the buffer's older time sample, not one
            # that merely slices a third dimension (DGNUMWET's modes, say)
            "time_sliced": "itim_old" in rest.replace(" ", ""),
            "rank": rank,
            "dtype": dtype,
            "line": number,
        }
    return {
        "schema_version": 1,
        "routine": routine,
        "source": str(source.relative_to(CAM_SRC)),
        "fields": [rows[k] for k in sorted(rows, key=lambda k: rows[k]["line"])],
    }


def render(table: dict) -> str:
    import yaml

    header = (
        "# GENERATED by tools/generate_pi_cam_pbuf_table.py -- do not edit.\n"
        f"#\n# The physics-buffer fields {table['routine']} reads, from the pinned\n"
        f"# {table['source']}: each with the module integer holding its index,\n"
        "# whether the routine takes the older time sample, and the rank and kind\n"
        "# of the pointer it declares.  Loaded by freecam.pi_cam.pbuf.load_pbuf_table.\n"
    )
    return header + yaml.safe_dump(table, sort_keys=False, width=100)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routine", required=True)
    parser.add_argument("--source", required=True, help="path under components/cam/src")
    parser.add_argument("--module", required=True, help="the module that owns the index variables")
    parser.add_argument("--index-module", action="append", default=[],
                        metavar="IDX=MODULE", help="an index variable owned elsewhere")
    parser.add_argument("--only", default=None,
                        help="comma-separated index variables to include (default: all)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    index_modules = dict(item.split("=", 1) for item in arguments.index_module)
    only = set(arguments.only.split(",")) if arguments.only else None
    table = build_table(arguments.routine, CAM_SRC / arguments.source, arguments.module,
                        index_modules, only)
    rendered = render(table)
    if arguments.check:
        current = arguments.output.read_text() if arguments.output.is_file() else ""
        if current != rendered:
            sys.stderr.write("".join(difflib.unified_diff(
                current.splitlines(keepends=True), rendered.splitlines(keepends=True),
                fromfile="committed", tofile="generated"))[:3000])
            sys.stderr.write(f"\nstale: {arguments.output}\n")
            return 1
        return 0
    arguments.output.write_text(rendered)
    print(f"wrote {arguments.output} ({len(table['fields'])} fields)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
