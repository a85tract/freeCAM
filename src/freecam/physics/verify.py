"""Build-time checks of a function spec against the inventory and the source.

The reviewed YAML is the runtime's authority, but it must not drift from the
routine it describes.  These checks compare it with the kernel inventory
(argument order, dtype, rank, intent, declared extents) and with the trailing
declaration comments in the pinned Fortran source (units in brackets).  They
run from ``tools/verify_pi_cam_function_spec.py``; the runtime never calls
them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .spec import FunctionSpec

_UNIT_BRACKET = re.compile(r"\[([^\]]+)\]")
_DECLARATION = re.compile(
    r"^\s*(?:real\s*\(\s*r8\s*\)|real|integer|logical)\s*(?:,[^:]*)?::\s*(?P<names>[^!]*?)\s*(?:!(?P<comment>.*))?$",
    re.IGNORECASE,
)
_CONTINUED_COMMENT = re.compile(r"^\s*!(?P<comment>.*)$")


@dataclass
class VerificationReport:
    """What was compared and every disagreement found."""

    function: str
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def ok(self, text: str) -> None:
        self.checks.append(text)

    def fail(self, text: str) -> None:
        self.failures.append(text)

    def as_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "passed": self.passed,
            "checks": list(self.checks),
            "failures": list(self.failures),
        }


def _inventory_dtype(argument: Mapping[str, Any]) -> tuple[str, str | None]:
    dtype = str(argument.get("dtype") or "")
    if dtype == "logical":
        return "int32", "logical"
    return dtype, None


def verify_against_inventory(
    spec: FunctionSpec, record: Mapping[str, Any], report: VerificationReport | None = None
) -> VerificationReport:
    """Argument order, dtype, rank, intent and extents must match the inventory."""

    report = report or VerificationReport(spec.function)
    if str(record.get("qualified_name", "")).lower() != spec.qualified_name.lower():
        report.fail(
            f"inventory record is {record.get('qualified_name')!r}, spec says {spec.qualified_name!r}"
        )
        return report
    arguments = [item for item in record.get("arguments", ()) if not item.get("procedure")]
    if len(arguments) != len(spec.arguments):
        report.fail(f"inventory declares {len(arguments)} arguments, spec {len(spec.arguments)}")
        return report
    for index, (declared, reviewed) in enumerate(zip(arguments, spec.arguments), start=1):
        name = str(declared["name"])
        where = f"argument {index} ({reviewed.name})"
        if name.lower() != reviewed.name.lower():
            report.fail(f"{where}: inventory has {name!r} at this position")
            continue
        dtype, carrier = _inventory_dtype(declared)
        if dtype != reviewed.dtype or carrier != reviewed.carrier:
            report.fail(
                f"{where}: inventory dtype {declared.get('dtype')!r}, spec dtype "
                f"{reviewed.dtype!r} carrier {reviewed.carrier!r}"
            )
        if int(declared.get("rank", -1)) != reviewed.rank:
            report.fail(f"{where}: inventory rank {declared.get('rank')}, spec rank {reviewed.rank}")
        pointer = bool(declared.get("pointer"))
        if pointer != reviewed.pointer:
            report.fail(f"{where}: inventory pointer={pointer}, spec pointer={reviewed.pointer}")
        intent = declared.get("intent")
        if intent is None:
            if not reviewed.pointer:
                report.fail(f"{where}: inventory has no intent and the dummy is not a pointer")
        elif str(intent).lower() != reviewed.intent:
            report.fail(f"{where}: inventory intent {intent!r}, spec intent {reviewed.intent!r}")
        dimensions = tuple(str(item) for item in declared.get("dimensions") or ())
        # A routine names its own extents (uwshcu writes mix/mkx, not
        # pcols/pver).  The spec declares that correspondence; nothing is
        # inferred here, so an undeclared name still fails.
        resolved = tuple(
            spec.dimension_aliases.get(" ".join(item.split()), item) for item in dimensions
        )
        if not pointer and resolved != reviewed.native_shape:
            report.fail(
                f"{where}: inventory extents {list(dimensions)}"
                + (f" (aliased to {list(resolved)})" if resolved != dimensions else "")
                + f", spec native_shape {list(reviewed.native_shape)}"
            )
        if bool(declared.get("optional")):
            report.fail(f"{where}: optional dummies are not supported")
    report.ok(f"inventory: {len(arguments)} arguments agree in order, dtype, rank, intent and extents")
    return report


def declaration_units(
    lines: Sequence[str], names: Sequence[str]
) -> dict[str, str | None]:
    """Units in brackets from each dummy's declaration comment, if any.

    A declaration's comment may continue on the next line as a bare comment
    (``C_qlst`` does this); the two are joined before the bracket is read.
    """

    wanted = {name.lower() for name in names}
    found: dict[str, str | None] = {}
    for index, line in enumerate(lines):
        match = _DECLARATION.match(line)
        if match is None:
            continue
        declared = [item.split("(")[0].strip().lower() for item in match.group("names").split(",")]
        comment = match.group("comment") or ""
        if index + 1 < len(lines):
            continued = _CONTINUED_COMMENT.match(lines[index + 1])
            if continued is not None and not _DECLARATION.match(lines[index + 1]):
                comment = comment + " " + continued.group("comment")
        unit = _UNIT_BRACKET.search(comment)
        for name in declared:
            if name in wanted and name not in found:
                found[name] = unit.group(1).strip() if unit else None
    return found


def verify_against_source(
    spec: FunctionSpec,
    source_lines: Sequence[str],
    *,
    line_start: int,
    line_end: int,
    report: VerificationReport | None = None,
) -> VerificationReport:
    """Units the source declares in brackets must match the reviewed units."""

    report = report or VerificationReport(spec.function)
    block = source_lines[max(0, line_start - 1) : line_end]
    found = declaration_units(block, [item.name for item in spec.arguments])
    missing = [item.name for item in spec.arguments if item.name.lower() not in found]
    if missing:
        report.fail("no declaration found in the routine for: " + ", ".join(missing))
    compared = 0
    for item in spec.arguments:
        source_unit = found.get(item.name.lower())
        if source_unit is None:
            continue
        compared += 1
        if item.units is None:
            report.fail(f"argument {item.name}: source declares [{source_unit}] but spec has no units")
        elif _normalize_unit(item.units) != _normalize_unit(source_unit):
            report.fail(
                f"argument {item.name}: source declares [{source_unit}], spec says {item.units!r}"
            )
    report.ok(f"source: {compared} bracketed units agree with the spec")
    return report


def _normalize_unit(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


__all__ = [
    "VerificationReport",
    "declaration_units",
    "verify_against_inventory",
    "verify_against_source",
]
