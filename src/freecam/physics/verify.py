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
    r"^\s*(?:real\s*\(\s*r8\s*\)|real|integer|logical|character\s*\(\s*[^)]*\s*\))"
    # attributes: intent(in), optional, dimension(0:mkx) -- parentheses may hold colons
    r"\s*(?:,\s*(?:[^:(!]|\([^)]*\))*)?::\s*(?P<names>[^!]*?)\s*(?:!(?P<comment>.*))?$",
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
    """What a reviewed spec must say for the type the inventory records.

    Two Fortran types have no NumPy equivalent and travel as one int32 with a
    carrier naming what they really are: a logical, and the character an
    error channel returns.
    """

    dtype = str(argument.get("dtype") or "")
    if dtype == "logical":
        return "int32", "logical"
    if dtype == "character":
        return "int32", "character"
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
    dummies = [item for item in spec.arguments if item.role != "result"]
    if len(arguments) != len(dummies):
        report.fail(f"inventory declares {len(arguments)} arguments, spec {len(dummies)}")
        return report
    result = spec.result
    declared_result = record.get("result")
    is_function = str(record.get("procedure_kind", "")).lower() == "function" or declared_result is not None
    if (result is None) != (not is_function):
        report.fail("inventory and spec disagree on whether the routine is a function")
    elif result is not None and declared_result is None:
        # the inventory saw a function whose result type sits in the function statement
        # (an elemental prefix) and recorded none; the reviewed spec supplies it
        report.ok(f"inventory: a function whose result the spec types as {result.dtype}")
    elif result is not None:
        dtype, _ = _inventory_dtype(declared_result)
        if dtype != result.dtype or int(declared_result.get("rank", -1)) != result.rank:
            report.fail(
                f"result {result.name}: inventory dtype {declared_result.get('dtype')!r} rank "
                f"{declared_result.get('rank')}, spec dtype {result.dtype!r} rank {result.rank}"
            )
        elif str(declared_result.get("name", "")).lower() != result.name.lower():
            report.fail(f"result: inventory names it {declared_result.get('name')!r}, spec {result.name!r}")
    for index, (declared, reviewed) in enumerate(zip(arguments, dummies), start=1):
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
    # a function's result may be typed in the function statement itself, with no declaration line
    missing = [item.name for item in spec.arguments if item.name.lower() not in found and item.role != "result"]
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
    verify_overwritten_on_entry(spec, source_lines, line_start=line_start, line_end=line_end, report=report)
    return report


def _code(line: str) -> str:
    """A source line without its trailing comment (a ``!`` outside quotes)."""

    quote = None
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == "!":
            return line[:index]
    return line


def _top_level_split(text: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return [part.strip().lower() for part in parts]


def verify_overwritten_on_entry(
    spec: FunctionSpec,
    source_lines: Sequence[str],
    *,
    line_start: int,
    line_end: int,
    report: VerificationReport | None = None,
) -> VerificationReport:
    """A dummy said to be overwritten on entry is assigned whole at that line, and appears nowhere before it.

    Before the line, the name may stand only in the routine's argument list and in
    declarations; after a ``contains`` it may not stand at all, since an internal
    procedure called earlier could read it.  Anything else fails: the claim is that
    the value handed in cannot matter.
    """

    report = report or VerificationReport(spec.function)
    for item in spec.arguments:
        line = item.overwritten_on_entry
        if line is None:
            continue
        where = f"argument {item.name}: overwritten_on_entry {line}"
        if not line_start < line <= line_end:
            report.fail(f"{where} is outside the routine ({line_start}-{line_end})")
            continue
        statement = _code(source_lines[line - 1])
        match = re.match(rf"\s*{re.escape(item.name)}\s*(?:\((?P<sections>.*)\))?\s*=(?!=)", statement, re.IGNORECASE)
        if match is None:
            report.fail(f"{where} is not an assignment to it: {statement.strip()!r}")
            continue
        if match.group("sections") is not None:
            sections = _top_level_split(match.group("sections"))
            aliases = {native.lower(): canonical for native, canonical in spec.dimension_aliases.items()}
            whole = len(sections) == item.rank
            for axis, lower, section in zip(item.native_shape, item.lower_bounds, sections):
                bounds = {axis.lower(), *(native for native, canonical in aliases.items() if canonical == axis)}
                if axis == "pcols":
                    bounds.add("ncol")                    # the live lanes are the whole of what the call sees
                if section != ":" and not (section.startswith(f"{lower}:") and section[len(f"{lower}:"):] in bounds):
                    whole = False
            if not whole:
                report.fail(f"{where} assigns a part of it, not its whole live section: {statement.strip()!r}")
                continue
        header_open = True
        inside = False
        name = re.compile(rf"(?<![\w%]){re.escape(item.name)}(?!\w)", re.IGNORECASE)
        for number in range(line_start, line_end + 1):
            text = _code(source_lines[number - 1])
            if number == line_start or header_open:
                header_open = text.rstrip().endswith("&")
                continue
            if re.match(r"\s*contains\s*$", text, re.IGNORECASE):
                inside = True
            if number == line or text.lstrip().startswith("#") or not name.search(text):
                continue
            if inside:
                report.fail(f"{where}: an internal procedure names it at line {number}")
                break
            if number < line and "::" not in text:
                report.fail(f"{where}: line {number} uses it before the assignment: {text.strip()!r}")
                break
        else:
            report.ok(f"source: {item.name} is assigned whole at line {line} before any use")
    return report


def _normalize_unit(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


__all__ = [
    "VerificationReport",
    "declaration_units",
    "verify_against_inventory",
    "verify_against_source",
    "verify_overwritten_on_entry",
]
