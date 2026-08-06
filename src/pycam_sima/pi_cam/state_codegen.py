"""Generate a legacy-derived-type bridge to the Python PI-CAM StatePool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping

import yaml

from .errors import PICAMConfigurationError


_TYPE_START = re.compile(r"^\s*type\s*(?:::)?\s*([a-zA-Z]\w*)\s*$", re.I)
_TYPE_END = re.compile(r"^\s*end\s*type(?:\s+[a-zA-Z]\w*)?\s*$", re.I)
_VARIABLE = re.compile(r"^([a-zA-Z]\w*)(?:\((.*)\))?(?:\s*=.*)?$")


@dataclass(frozen=True, slots=True)
class LegacyStateOwner:
    name: str
    type_name: str
    source: Path


@dataclass(frozen=True, slots=True)
class LegacyStateField:
    field_id: int
    owner: LegacyStateOwner
    member: str
    dtype: str
    source_dimensions: tuple[str, ...]
    allocatable: bool
    pointer: bool

    @property
    def name(self) -> str:
        return f"{self.owner.name}.{self.member}"

    @property
    def component_rank(self) -> int:
        return len(self.source_dimensions)

    @property
    def python_rank(self) -> int:
        return self.component_rank + 1

    @property
    def expression(self) -> str:
        return f"{self.owner.name}(c)%{self.member}"

    @property
    def reference(self) -> str:
        owner = self.owner.name
        return f"{owner}(lbound({owner},1))%{self.member}"

    def to_payload(self) -> dict[str, object]:
        dtype = "float64" if self.dtype == "real" else "int32"
        return {
            "field_id": self.field_id,
            "name": self.name,
            "dtype": dtype,
            "component_rank": self.component_rank,
            "rank": self.python_rank,
            "source_dimensions": list(self.source_dimensions),
            "storage": (
                "pointer" if self.pointer else "allocatable" if self.allocatable else "inline"
            ),
        }


@dataclass(frozen=True, slots=True)
class LegacyStateBridge:
    owners: tuple[LegacyStateOwner, ...]
    fields: tuple[LegacyStateField, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "ownership": "python-authoritative-with-legacy-fortran-mirror",
            "fields": [field.to_payload() for field in self.fields],
            "symbols": {
                "count": "pycam_pi_cam_state_count_v1",
                "metadata": "pycam_pi_cam_state_metadata_v1",
                "transfer": "pycam_pi_cam_state_transfer_v1",
            },
            "directions": {"native_to_python": 1, "python_to_native": 2},
        }


def _without_comment(line: str) -> str:
    # The admitted type declarations do not contain quoted exclamation marks.
    return line.split("!", 1)[0]


def _logical_statements(text: str) -> Iterable[str]:
    parts: list[str] = []
    for raw in text.splitlines():
        line = _without_comment(raw).strip()
        if not line:
            continue
        if line.startswith("&"):
            line = line[1:].lstrip()
        continued = line.endswith("&")
        if continued:
            line = line[:-1].rstrip()
        parts.append(line)
        if not continued:
            yield " ".join(parts)
            parts.clear()
    if parts:
        yield " ".join(parts)


def _split_top_level(text: str) -> tuple[str, ...]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    result.append(text[start:].strip())
    return tuple(item for item in result if item)


def _type_body(source: Path, type_name: str) -> tuple[str, ...]:
    active = False
    body: list[str] = []
    for statement in _logical_statements(source.read_text(errors="replace")):
        if not active:
            match = _TYPE_START.match(statement)
            active = bool(match and match.group(1).lower() == type_name.lower())
            continue
        if _TYPE_END.match(statement):
            return tuple(body)
        body.append(statement)
    raise PICAMConfigurationError(
        f"cannot find derived type {type_name!r} in {source}"
    )


def _declaration_fields(
    owner: LegacyStateOwner,
    statement: str,
) -> tuple[tuple[str, str, tuple[str, ...], bool, bool], ...]:
    if "::" not in statement:
        return ()
    declaration, variables = (part.strip() for part in statement.split("::", 1))
    lower = declaration.lower()
    if lower.startswith("real"):
        dtype = "real"
    elif lower.startswith("integer"):
        dtype = "integer"
    else:
        return ()
    allocatable = "allocatable" in lower
    pointer = "pointer" in lower
    dimension_match = re.search(r"\bdimension\s*\(([^)]*)\)", declaration, re.I)
    inherited_dimensions = (
        _split_top_level(dimension_match.group(1)) if dimension_match else ()
    )
    result: list[tuple[str, str, tuple[str, ...], bool, bool]] = []
    for variable in _split_top_level(variables):
        match = _VARIABLE.match(variable.strip())
        if match is None:
            raise PICAMConfigurationError(
                f"cannot parse {owner.type_name} declaration item {variable!r}"
            )
        dimensions = (
            _split_top_level(match.group(2))
            if match.group(2) is not None
            else inherited_dimensions
        )
        result.append((match.group(1), dtype, dimensions, allocatable, pointer))
    return tuple(result)


def load_state_bridge(description: str | Path, source_root: str | Path) -> LegacyStateBridge:
    descriptor = Path(description).resolve()
    payload = yaml.safe_load(descriptor.read_text())
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", 0)) != 1:
        raise PICAMConfigurationError("legacy state bridge requires schema_version: 1")
    root = Path(source_root).resolve()
    raw_owners = payload.get("owners")
    if not isinstance(raw_owners, list) or not raw_owners:
        raise PICAMConfigurationError("legacy state bridge must declare owners")
    owners: list[LegacyStateOwner] = []
    fields: list[LegacyStateField] = []
    excluded = {str(name) for name in payload.get("exclude", ())}
    for raw_owner in raw_owners:
        if not isinstance(raw_owner, Mapping):
            raise PICAMConfigurationError("legacy state owner must be a mapping")
        owner = LegacyStateOwner(
            name=str(raw_owner["name"]),
            type_name=str(raw_owner["type"]),
            source=root / str(raw_owner["source"]),
        )
        if not owner.source.is_file():
            raise PICAMConfigurationError(f"state type source is absent: {owner.source}")
        owners.append(owner)
        for statement in _type_body(owner.source, owner.type_name):
            for member, dtype, dimensions, allocatable, pointer in _declaration_fields(
                owner, statement
            ):
                qualified = f"{owner.name}.{member}"
                if qualified in excluded:
                    continue
                fields.append(
                    LegacyStateField(
                        field_id=len(fields) + 1,
                        owner=owner,
                        member=member,
                        dtype=dtype,
                        source_dimensions=dimensions,
                        allocatable=allocatable,
                        pointer=pointer,
                    )
                )
    return LegacyStateBridge(tuple(owners), tuple(fields))


def _metadata_case(field: LegacyStateField) -> list[str]:
    owner = field.owner.name
    lines = [
        f"  case ({field.field_id})",
        f"    name = '{field.name}'",
        f"    dtype_code = {1 if field.dtype == 'real' else 2}",
        f"    field_rank = {field.python_rank}",
        "    if (max_rank < field_rank) then",
        "      status = 3",
        "      return",
        "    endif",
    ]
    if field.allocatable or field.pointer:
        inquiry = "associated" if field.pointer else "allocated"
        lines.extend(
            [
                f"    do c = lbound({owner},1), ubound({owner},1)",
                f"      if (.not. {inquiry}({field.expression})) active = .false.",
                "    enddo",
                "    if (.not. active) return",
            ]
        )
    for axis in range(field.component_rank):
        lines.append(f"    extents({axis + 1}) = size({field.reference},{axis + 1})")
    lines.append(f"    extents({field.python_rank}) = size({owner})")
    return lines


def _transfer_case(field: LegacyStateField) -> list[str]:
    owner = field.owner.name
    values = "real_values" if field.dtype == "real" else "integer_values"
    lines = [f"  case ({field.field_id})", "    offset = 1"]
    lines.append(f"    do c = lbound({owner},1), ubound({owner},1)")
    if field.component_rank == 0:
        lines.extend(
            [
                "      if (direction == 1) then",
                f"        {values}(offset) = {field.expression}",
                "      else",
                f"        {field.expression} = {values}(offset)",
                "      endif",
                "      offset = offset + 1",
            ]
        )
    else:
        lines.extend(
            [
                f"      component_values = size({field.expression})",
                "      if (direction == 1) then",
                f"        {values}(offset:offset+component_values-1) = &",
                f"             reshape({field.expression}, (/ component_values /))",
                "      else",
                f"        {field.expression} = reshape(&",
                f"             {values}(offset:offset+component_values-1), &",
                f"             shape({field.expression}))",
                "      endif",
                "      offset = offset + component_values",
            ]
        )
    lines.append("    enddo")
    return lines


def generate_fortran_include(bridge: LegacyStateBridge) -> str:
    """Return procedures included inside ``module cam_comp``."""

    metadata_cases: list[str] = []
    transfer_cases: list[str] = []
    for field in bridge.fields:
        metadata_cases.extend(_metadata_case(field))
        transfer_cases.extend(_transfer_case(field))
    return "\n".join(
        [
            "! Generated by pycam_sima.pi_cam.state_codegen; do not edit.",
            "integer function cam_python_state_count() result(count)",
            f"  count = {len(bridge.fields)}",
            "end function cam_python_state_count",
            "",
            "subroutine cam_python_state_metadata(field_id, cam_in, cam_out, name, &",
            "     dtype_code, field_rank, extents, max_rank, active, status)",
            "  integer, intent(in) :: field_id, max_rank",
            "  type(cam_in_t), pointer :: cam_in(:)",
            "  type(cam_out_t), pointer :: cam_out(:)",
            "  character(len=*), intent(out) :: name",
            "  integer, intent(out) :: dtype_code, field_rank, extents(max_rank), status",
            "  logical, intent(out) :: active",
            "  integer :: c",
            "  name = ''",
            "  dtype_code = 0",
            "  field_rank = 0",
            "  extents = 0",
            "  active = .true.",
            "  status = 0",
            "  select case (field_id)",
            *metadata_cases,
            "  case default",
            "    status = 2",
            "  end select",
            "end subroutine cam_python_state_metadata",
            "",
            "subroutine cam_python_state_transfer(field_id, direction, data, nvalues, &",
            "     cam_in, cam_out, status)",
            "  use, intrinsic :: iso_c_binding, only: c_f_pointer, c_int32_t, c_ptr",
            "  integer, intent(in) :: field_id, direction, nvalues",
            "  type(c_ptr), value :: data",
            "  type(cam_in_t), pointer :: cam_in(:)",
            "  type(cam_out_t), pointer :: cam_out(:)",
            "  integer, intent(out) :: status",
            "  real(r8), pointer :: real_values(:)",
            "  integer(c_int32_t), pointer :: integer_values(:)",
            "  integer :: dtype_code, field_rank, extents(8), expected",
            "  integer :: c, offset, component_values",
            "  logical :: active",
            "  character(len=128) :: name",
            "  call cam_python_state_metadata(field_id, cam_in, cam_out, name, &",
            "       dtype_code, field_rank, extents, size(extents), active, status)",
            "  if (status /= 0) return",
            "  if (.not. active) then",
            "    status = 4",
            "    return",
            "  endif",
            "  if (direction /= 1 .and. direction /= 2) then",
            "    status = 5",
            "    return",
            "  endif",
            "  expected = product(extents(1:field_rank))",
            "  if (nvalues /= expected) then",
            "    status = 6",
            "    return",
            "  endif",
            "  if (dtype_code == 1) then",
            "    call c_f_pointer(data, real_values, (/ nvalues /))",
            "  else if (dtype_code == 2) then",
            "    call c_f_pointer(data, integer_values, (/ nvalues /))",
            "  else",
            "    status = 7",
            "    return",
            "  endif",
            "  select case (field_id)",
            *transfer_cases,
            "  case default",
            "    status = 2",
            "  end select",
            "end subroutine cam_python_state_transfer",
            "",
        ]
    )


def instrument_cam_comp(source: str, include_name: str) -> str:
    """Expose generated state routines without editing the source checkout."""

    public_anchor = "   public cam_final     ! CAM Finalization"
    if public_anchor not in source:
        raise PICAMConfigurationError("cam_comp public anchor changed")
    source = source.replace(
        public_anchor,
        public_anchor
        + "\n   public cam_python_state_count, cam_python_state_metadata, &\n"
        + "        cam_python_state_transfer",
        1,
    )
    end_anchor = "end module cam_comp"
    if source.count(end_anchor) != 1:
        raise PICAMConfigurationError("cam_comp end-module anchor changed")
    return source.replace(end_anchor, f"include '{include_name}'\n\n{end_anchor}", 1)
