"""Validated editing of CAM's ``atm_in`` Fortran namelist.

CAM reads ``atm_in`` once, at initialization, and its failure modes are
asymmetric: a bad variable name inside a group it reads aborts the whole job
with a message that does not name the variable, while a misspelled group --
or a group the active configuration never reads -- is ignored without a
word, leaving the user convinced they tuned something.  Every override is
therefore validated here, in Python, against CAM's own catalogue
(``namelist_definition.xml`` in the pinned iCESM source) before a byte of
the file changes, and an override whose group is absent from the file is
refused rather than appended: CAM resolves duplicate groups as "first
occurrence wins", so appending is exactly the silent no-op this module
exists to prevent.

Only the overridden assignments are rewritten.  Every other line of the
file is preserved byte for byte, and applying an empty override set does
not open the file for writing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
import difflib
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from .errors import PICAMConfigurationError

_DEFINITION_RELATIVE_PATH = Path(
    "components/cam/bld/namelist_files/namelist_definition.xml"
)

# The vendor XML is not well formed (raw ampersands and one unclosed entry),
# so entries are extracted leniently instead of with an XML parser.
_XML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_XML_ENTRY = re.compile(r"<entry\b(.*?)>", re.DOTALL)
_XML_ATTRIBUTE = re.compile(r"([\w.:-]+)\s*=\s*\"([^\"]*)\"")

_TYPE_PATTERN = re.compile(
    r"^(?P<base>char|real|integer|logical)"
    r"(?:\*(?P<length>\d+))?"
    r"(?:\((?P<dims>[^)]+)\))?$"
)

# One assignment per logical line; values may continue onto following lines
# (and continuation lines may contain '=' inside quoted strings), so only a
# line matching this pattern starts a new assignment.
_ASSIGNMENT_HEAD = re.compile(r"^(\s*)([A-Za-z]\w*)(\s*=)")
_GROUP_HEADER = re.compile(r"^\s*&(\w+)")
_GROUP_TERMINATOR = re.compile(r"^\s*/\s*$")


@dataclass(frozen=True, slots=True)
class NamelistEntry:
    """One variable from CAM's namelist definition catalogue."""

    id: str
    base_type: str
    char_length: int | None
    array_dims: int | str | None
    group: str
    valid_values: tuple[str, ...]

    @property
    def is_array(self) -> bool:
        return self.array_dims is not None

    @property
    def type_label(self) -> str:
        label = self.base_type
        if self.char_length is not None:
            label = f"char*{self.char_length}"
        if self.array_dims is not None:
            label = f"{label}({self.array_dims})"
        return label


@dataclass(frozen=True, slots=True)
class _Assignment:
    """One ``name = value`` occurrence, possibly spanning several lines."""

    name: str
    group: str
    start: int
    end: int
    raw_value: str


def load_catalog(source_root: str | Path) -> dict[str, NamelistEntry]:
    """Parse CAM's namelist definition into a validation catalogue."""

    path = Path(source_root) / _DEFINITION_RELATIVE_PATH
    if not path.is_file():
        raise PICAMConfigurationError(
            f"CAM namelist definition not found: {path}"
        )
    text = _XML_COMMENT.sub("", path.read_text())
    catalog: dict[str, NamelistEntry] = {}
    for match in _XML_ENTRY.finditer(text):
        attributes = dict(_XML_ATTRIBUTE.findall(match.group(1)))
        identifier = attributes.get("id", "").strip().lower()
        type_text = attributes.get("type", "").strip()
        group = attributes.get("group", "").strip().lower()
        if not identifier or not group:
            continue
        parsed = _TYPE_PATTERN.match(type_text)
        if parsed is None:
            continue
        dims_text = parsed.group("dims")
        dims: int | str | None
        if dims_text is None:
            dims = None
        elif dims_text.strip().isdigit():
            dims = int(dims_text.strip())
        else:
            dims = dims_text.strip()
        length_text = parsed.group("length")
        valid_text = attributes.get("valid_values", "").strip()
        entry = NamelistEntry(
            id=identifier,
            base_type=parsed.group("base"),
            char_length=int(length_text) if length_text else None,
            array_dims=dims,
            group=group,
            valid_values=(
                tuple(item.strip() for item in valid_text.split(","))
                if valid_text
                else ()
            ),
        )
        # First occurrence wins, matching CAM's own build tooling.
        catalog.setdefault(identifier, entry)
    if not catalog:
        raise PICAMConfigurationError(
            f"no namelist entries could be read from {path}"
        )
    return catalog


def parse_namelist(text: str) -> tuple[_Assignment, ...]:
    """Locate every assignment in a Fortran namelist file, group-aware."""

    lines = text.split("\n")
    assignments: list[_Assignment] = []
    group: str | None = None
    head: tuple[str, int] | None = None
    value_parts: list[str] = []

    def flush(end: int) -> None:
        nonlocal head
        if head is not None:
            assignments.append(
                _Assignment(
                    name=head[0],
                    group=group or "",
                    start=head[1],
                    end=end,
                    raw_value="\n".join(value_parts).strip(),
                )
            )
        head = None
        value_parts.clear()

    for index, line in enumerate(lines):
        header = _GROUP_HEADER.match(line)
        if header is not None:
            flush(index - 1)
            group = header.group(1).lower()
            continue
        if _GROUP_TERMINATOR.match(line):
            flush(index - 1)
            group = None
            continue
        if group is None:
            continue
        assignment = _ASSIGNMENT_HEAD.match(line)
        if assignment is not None:
            flush(index - 1)
            head = (assignment.group(2).lower(), index)
            value_parts.append(line[assignment.end(3):])
            continue
        if head is not None:
            value_parts.append(line)
            continue
        if line.strip():
            raise PICAMConfigurationError(
                f"unrecognised namelist line {index + 1}: {line.strip()!r}"
            )
    flush(len(lines) - 1)
    return tuple(assignments)


def coerce_value(value: Any, entry: NamelistEntry) -> Any:
    """Validate one Python value against a catalogue entry, fail closed."""

    if entry.is_array and isinstance(value, (list, tuple)):
        if isinstance(entry.array_dims, int) and len(value) > entry.array_dims:
            raise PICAMConfigurationError(
                f"{entry.id} accepts at most {entry.array_dims} elements, "
                f"got {len(value)}"
            )
        if not value:
            raise PICAMConfigurationError(f"{entry.id} cannot be empty")
        return tuple(_coerce_scalar(item, entry) for item in value)
    if isinstance(value, (list, tuple)):
        raise PICAMConfigurationError(
            f"{entry.id} is a scalar {entry.type_label}, got a sequence"
        )
    return _coerce_scalar(value, entry)


def _coerce_scalar(value: Any, entry: NamelistEntry) -> Any:
    base = entry.base_type
    # bool is an int subclass, so it must be recognised before integers.
    if base == "logical":
        if not isinstance(value, bool):
            raise PICAMConfigurationError(
                f"{entry.id} is a Fortran logical; pass True or False, "
                f"not {value!r}"
            )
        return value
    if isinstance(value, bool):
        raise PICAMConfigurationError(
            f"{entry.id} is a Fortran {entry.type_label}, got a boolean"
        )
    if base == "integer":
        if not isinstance(value, int):
            raise PICAMConfigurationError(
                f"{entry.id} is a Fortran integer, got {value!r}"
            )
        result: Any = value
    elif base == "real":
        if not isinstance(value, (int, float)):
            raise PICAMConfigurationError(
                f"{entry.id} is a Fortran real, got {value!r}"
            )
        result = float(value)
    elif base == "char":
        if not isinstance(value, str):
            raise PICAMConfigurationError(
                f"{entry.id} is a Fortran {entry.type_label}, got {value!r}"
            )
        if entry.char_length is not None and len(value) > entry.char_length:
            raise PICAMConfigurationError(
                f"{entry.id} is limited to {entry.char_length} characters, "
                f"got {len(value)}"
            )
        result = value
    else:  # pragma: no cover - the type pattern admits nothing else
        raise PICAMConfigurationError(
            f"{entry.id} has unsupported type {entry.type_label!r}"
        )
    if entry.valid_values:
        literal = result if isinstance(result, str) else _plain_text(result)
        if literal not in entry.valid_values:
            raise PICAMConfigurationError(
                f"{entry.id} must be one of "
                f"{', '.join(entry.valid_values)}; got {literal!r}"
            )
    return result


def coerce_text(text: str, entry: NamelistEntry) -> Any:
    """Interpret one command-line string as a value for ``entry``."""

    stripped = text.strip()
    if entry.is_array and "," in stripped:
        return coerce_value(
            [_scalar_from_text(item, entry) for item in stripped.split(",")],
            entry,
        )
    return coerce_value(_scalar_from_text(stripped, entry), entry)


def _scalar_from_text(text: str, entry: NamelistEntry) -> Any:
    stripped = text.strip()
    base = entry.base_type
    if base == "logical":
        lowered = stripped.lower().strip(".")
        if lowered in {"true", "t"}:
            return True
        if lowered in {"false", "f"}:
            return False
        raise PICAMConfigurationError(
            f"{entry.id} is a Fortran logical; use true or false, "
            f"not {stripped!r}"
        )
    if base == "integer":
        try:
            return int(stripped)
        except ValueError as error:
            raise PICAMConfigurationError(
                f"{entry.id} is a Fortran integer, got {stripped!r}"
            ) from error
    if base == "real":
        try:
            return float(stripped.replace("D", "e").replace("d", "e"))
        except ValueError as error:
            raise PICAMConfigurationError(
                f"{entry.id} is a Fortran real, got {stripped!r}"
            ) from error
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == "'":
        return stripped[1:-1].replace("''", "'")
    return stripped


def format_fortran_value(value: Any, entry: NamelistEntry) -> str:
    """Render a validated value in the notation CAM's files use."""

    if isinstance(value, tuple):
        return ", ".join(_format_scalar(item, entry) for item in value)
    return _format_scalar(value, entry)


def _format_scalar(value: Any, entry: NamelistEntry) -> str:
    if entry.base_type == "logical":
        return ".true." if value else ".false."
    if entry.base_type == "char":
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
    if entry.base_type == "real":
        text = repr(float(value))
        if "e" in text or "E" in text:
            return text.replace("e", "D").replace("E", "D")
        return f"{text}D0"
    return str(value)


def _plain_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def validate_overrides(
    overrides: Mapping[str, Any], catalog: Mapping[str, NamelistEntry]
) -> dict[str, Any]:
    """Coerce every override against the catalogue, fail closed on any."""

    validated: dict[str, Any] = {}
    for name, value in overrides.items():
        key = str(name).strip().lower()
        entry = catalog.get(key)
        if entry is None:
            raise PICAMConfigurationError(
                f"unknown CAM namelist variable {key!r}"
                + _suggestions(key, catalog)
            )
        validated[key] = coerce_value(value, entry)
    return validated


def _suggestions(name: str, catalog: Mapping[str, NamelistEntry]) -> str:
    close = difflib.get_close_matches(name, catalog, n=3, cutoff=0.6)
    if not close:
        return ""
    return f"; did you mean {', '.join(close)}?"


def apply_overrides(
    path: str | Path,
    overrides: Mapping[str, Any],
    catalog: Mapping[str, NamelistEntry],
) -> dict[str, tuple[str | None, str]]:
    """Rewrite the overridden assignments of one namelist file in place.

    Returns ``{name: (previous_raw_value_or_None, new_raw_value)}``.  With no
    overrides the file is not opened for writing at all.
    """

    validated = validate_overrides(overrides, catalog)
    if not validated:
        return {}
    path = Path(path)
    text = path.read_text()
    assignments = parse_namelist(text)
    by_name: dict[str, list[_Assignment]] = {}
    groups: set[str] = set()
    for assignment in assignments:
        by_name.setdefault(assignment.name, []).append(assignment)
        groups.add(assignment.group)
    group_headers = {
        match.group(1).lower()
        for match in map(_GROUP_HEADER.match, text.split("\n"))
        if match is not None
    }
    groups |= group_headers

    lines = text.split("\n")
    report: dict[str, tuple[str | None, str]] = {}
    # Collect edits first so a failure changes nothing.
    replacements: list[tuple[int, int, list[str]]] = []
    for name, value in validated.items():
        entry = catalog[name]
        found = by_name.get(name, [])
        rendered = format_fortran_value(value, entry)
        if len(found) > 1:
            raise PICAMConfigurationError(
                f"{name} appears {len(found)} times in {path}; refusing an "
                "ambiguous edit"
            )
        if found:
            assignment = found[0]
            head = _ASSIGNMENT_HEAD.match(lines[assignment.start])
            prefix = lines[assignment.start][: head.end(3)]
            replacements.append(
                (assignment.start, assignment.end, [f"{prefix} {rendered}"])
            )
            report[name] = (assignment.raw_value, rendered)
        else:
            if entry.group not in groups:
                raise PICAMConfigurationError(
                    f"{name} belongs to namelist group &{entry.group}, which "
                    f"{path.name} does not contain. CAM ignores unknown "
                    "groups silently, so writing it would have no effect; "
                    "this configuration does not read that group."
                )
            terminator = _group_terminator_line(lines, entry.group)
            replacements.append(
                (terminator, terminator - 1, [f" {name}\t\t= {rendered}"])
            )
            report[name] = (None, rendered)

    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start : end + 1] = replacement

    output = "\n".join(lines)
    directory = path.parent
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=directory
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(output)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return report


def _group_terminator_line(lines: Sequence[str], group: str) -> int:
    inside = False
    for index, line in enumerate(lines):
        header = _GROUP_HEADER.match(line)
        if header is not None:
            inside = header.group(1).lower() == group
            continue
        if inside and _GROUP_TERMINATOR.match(line):
            return index
    raise PICAMConfigurationError(
        f"namelist group &{group} has no terminating '/'"
    )


def read_values(path: str | Path) -> dict[str, str]:
    """Read every assignment's raw value from one namelist file."""

    path = Path(path)
    return {
        assignment.name: assignment.raw_value
        for assignment in parse_namelist(path.read_text())
    }
