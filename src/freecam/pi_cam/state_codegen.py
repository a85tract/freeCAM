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
    python_dimensions: tuple[str, ...]
    active_by_default: bool = True

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
            "dimensions": [*self.python_dimensions, "chunks"],
            "active_by_default": self.active_by_default,
            "storage": (
                "pointer" if self.pointer else "allocatable" if self.allocatable else "inline"
            ),
        }


@dataclass(frozen=True, slots=True)
class LegacyStateBridge:
    owners: tuple[LegacyStateOwner, ...]
    fields: tuple[LegacyStateField, ...]
    dimension_defaults: tuple[tuple[str, int], ...] = ()
    initializers: tuple[tuple[str, str | int | float], ...] = ()

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "ownership": "python-authoritative-with-legacy-fortran-mirror",
            "fields": [field.to_payload() for field in self.fields],
            "owners": [
                {
                    "owner_id": owner_id,
                    "name": owner.name,
                    "type": owner.type_name,
                    "raw_field": f"__native_owner.{owner.name}",
                    "layout_symbol": f"pycam_pi_cam_layout_{owner.name}_v1",
                    "inline_field_ids": [
                        field.field_id
                        for field in self.fields
                        if field.owner == owner
                        and field.active_by_default
                        and not field.allocatable
                        and not field.pointer
                    ],
                }
                for owner_id, owner in enumerate(self.owners, start=1)
            ],
            "dimension_defaults": dict(self.dimension_defaults),
            "initializers": dict(self.initializers),
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
    raw_bindings = payload.get("dimension_bindings", {})
    if not isinstance(raw_bindings, Mapping):
        raise PICAMConfigurationError("dimension_bindings must be a mapping")
    dimension_bindings: dict[str, tuple[str, ...]] = {}
    for name, dimensions in raw_bindings.items():
        if not isinstance(dimensions, list) or any(
            not isinstance(dimension, str) or not re.fullmatch(r"[A-Za-z_]\w*", dimension)
            for dimension in dimensions
        ):
            raise PICAMConfigurationError(
                f"state dimension binding for {name!r} must be a list of names"
            )
        dimension_bindings[str(name)] = tuple(dimensions)
    inactive_fields = {str(name) for name in payload.get("inactive_fields", ())}
    raw_defaults = payload.get("dimension_defaults", {})
    if not isinstance(raw_defaults, Mapping):
        raise PICAMConfigurationError("dimension_defaults must be a mapping")
    dimension_defaults: list[tuple[str, int]] = []
    for name, value in raw_defaults.items():
        if not re.fullmatch(r"[A-Za-z_]\w*", str(name)) or int(value) < 0:
            raise PICAMConfigurationError(
                f"invalid state dimension default {name!r}={value!r}"
            )
        dimension_defaults.append((str(name), int(value)))
    raw_initializers = payload.get("initializers", {})
    if not isinstance(raw_initializers, Mapping):
        raise PICAMConfigurationError("initializers must be a mapping")
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
                if any(dimension == ":" for dimension in dimensions):
                    try:
                        python_dimensions = dimension_bindings[qualified]
                    except KeyError as exc:
                        raise PICAMConfigurationError(
                            f"deferred-shape state field {qualified!r} requires "
                            "dimension_bindings"
                        ) from exc
                else:
                    python_dimensions = tuple(dimensions)
                if len(python_dimensions) != len(dimensions):
                    raise PICAMConfigurationError(
                        f"state field {qualified!r} dimension binding has the wrong rank"
                    )
                fields.append(
                    LegacyStateField(
                        field_id=len(fields) + 1,
                        owner=owner,
                        member=member,
                        dtype=dtype,
                        source_dimensions=dimensions,
                        allocatable=allocatable,
                        pointer=pointer,
                        python_dimensions=python_dimensions,
                        active_by_default=qualified not in inactive_fields,
                    )
                )
    discovered = {field.name for field in fields}
    unknown_bindings = sorted(set(dimension_bindings) - discovered)
    unknown_inactive = sorted(inactive_fields - discovered)
    unknown_initializers = sorted(set(map(str, raw_initializers)) - discovered)
    if unknown_bindings or unknown_inactive or unknown_initializers:
        raise PICAMConfigurationError(
            "state descriptor references unknown fields: "
            + ", ".join(
                (*unknown_bindings, *unknown_inactive, *unknown_initializers)
            )
        )
    admitted_symbols = {"chunk_id", "chunk_ncols", "pcols", "inf", "posinf"}
    initializers: list[tuple[str, str | int | float]] = []
    for raw_name, raw_value in raw_initializers.items():
        name = str(raw_name)
        if isinstance(raw_value, str):
            value = raw_value.strip().lower()
            if value not in admitted_symbols:
                raise PICAMConfigurationError(
                    f"state initializer for {name!r} has unknown symbol {raw_value!r}"
                )
        elif isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            value = raw_value
        else:
            raise PICAMConfigurationError(
                f"state initializer for {name!r} must be numeric or symbolic"
            )
        initializers.append((name, value))
    return LegacyStateBridge(
        tuple(owners),
        tuple(fields),
        tuple(dimension_defaults),
        tuple(initializers),
    )


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
            "! Generated by freecam.pi_cam.state_codegen; do not edit.",
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
            *_pbuf_accessor(),
            *_macro_host_binding(),
            *_rad_host_binding(),
            *_micro_host_binding(),
            *_mm_host_binding(),
            *_aero_host_binding(),
        ]
    )


def _macro_host_binding() -> tuple[str, ...]:
    """Hand the macrophysics handles the state array and the physics buffer.

    Both live in cam_comp, which the handles module cannot use (cam_comp uses
    it), so the binding is done from here, once, when Python asks for it
    after initialization.
    """

    return (
        "integer(c_int) function pycam_macro_bind_hosts_v1() &",
        "     bind(C, name='pycam_macro_bind_hosts_v1') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_int",
        "  use pycam_macro_handles, only: pycam_macro_bind_hosts",
        "  status = 1_c_int",
        "  if (.not. associated(phys_state) .or. .not. associated(pbuf2d)) return",
        "  call pycam_macro_bind_hosts(phys_state, pbuf2d)",
        "  status = 0_c_int",
        "end function pycam_macro_bind_hosts_v1",
        "",
    )


def _rad_host_binding() -> tuple[str, ...]:
    """The same for the radiation handles, for the same reason."""

    return (
        "integer(c_int) function pycam_rad_bind_hosts_v1() &",
        "     bind(C, name='pycam_rad_bind_hosts_v1') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_int",
        "  use pycam_rad_handles, only: pycam_rad_bind_hosts",
        "  status = 1_c_int",
        "  if (.not. associated(phys_state) .or. .not. associated(pbuf2d)) return",
        "  call pycam_rad_bind_hosts(phys_state, pbuf2d)",
        "  status = 0_c_int",
        "end function pycam_rad_bind_hosts_v1",
        "",
    )


def _micro_host_binding() -> tuple[str, ...]:
    """The same for the microphysics handles."""

    return (
        "integer(c_int) function pycam_micro_bind_hosts_v1() &",
        "     bind(C, name='pycam_micro_bind_hosts_v1') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_int",
        "  use pycam_micro_handles, only: pycam_micro_bind_hosts",
        "  status = 1_c_int",
        "  if (.not. associated(phys_state) .or. .not. associated(pbuf2d)) return",
        "  call pycam_micro_bind_hosts(phys_state, pbuf2d)",
        "  status = 0_c_int",
        "end function pycam_micro_bind_hosts_v1",
        "",
    )


def _mm_host_binding() -> tuple[str, ...]:
    """The cloud macro/microphysics stage also updates ``phys_tend``.

    ``physics_update(state, ptend, ztodt, tend)`` in tphysbc accumulates the
    stage's tendencies into cam_comp's ``phys_tend(lchnk)``, so that array
    is bound alongside the state and the buffer.
    """

    return (
        "integer(c_int) function pycam_mm_bind_hosts_v1() &",
        "     bind(C, name='pycam_mm_bind_hosts_v1') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_int",
        "  use pycam_mm_handles, only: pycam_mm_bind_hosts",
        "  status = 1_c_int",
        "  if (.not. associated(phys_state) .or. .not. associated(phys_tend) &",
        "       .or. .not. associated(pbuf2d)) return",
        "  call pycam_mm_bind_hosts(phys_state, phys_tend, pbuf2d)",
        "  status = 0_c_int",
        "end function pycam_mm_bind_hosts_v1",
        "",
    )


def _aero_host_binding() -> tuple[str, ...]:
    """The same for the aerosol activation handles."""

    return (
        "integer(c_int) function pycam_aero_bind_hosts_v1() &",
        "     bind(C, name='pycam_aero_bind_hosts_v1') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_int",
        "  use pycam_aero_handles, only: pycam_aero_bind_hosts",
        "  status = 1_c_int",
        "  if (.not. associated(phys_state) .or. .not. associated(pbuf2d)) return",
        "  call pycam_aero_bind_hosts(phys_state, pbuf2d)",
        "  status = 0_c_int",
        "end function pycam_aero_bind_hosts_v1",
        "",
    )


def _pbuf_accessor() -> tuple[str, ...]:
    """A zero-copy handle on one physics-buffer field of one chunk.

    The physics buffer is the last host service Python could not see.  It is
    not a state owner -- CAM allocates it, indexes it by a name registered at
    initialization, and rotates two time samples -- so it gets a handle rather
    than a StatePool field: the caller asks for a field by the index CAM gave
    it and receives the address CAM already holds.  Nothing is copied and
    nothing is allocated, so a Python-driven process reads and writes the same
    storage the Fortran process would have.

    Only the two shapes CAM's physics actually asks for are served: a plain
    ``(pcols, n)`` field, and the ``(pcols, pver)`` plane of a time-rotated
    field at the older sample.  Anything else is refused rather than guessed.
    """

    return (
        "integer(c_int) function pycam_pbuf_field_v1(lchnk, field_index, &",
        "     time_sliced, ptr, extents) &",
        "     bind(C, name='pycam_pbuf_field_v1') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_int, &",
        "       c_int64_t, c_null_ptr",
        "  use physics_buffer, only: physics_buffer_desc, pbuf_get_chunk, &",
        "       pbuf_get_field, pbuf_old_tim_idx",
        "  use ppgrid, only: pcols, pver",
        "  integer(c_int), value, intent(in)  :: lchnk, field_index, time_sliced",
        "  type(c_ptr),           intent(out) :: ptr",
        "  integer(c_int64_t),    intent(out) :: extents(2)",
        "  type(physics_buffer_desc), pointer :: pbuf(:)",
        "  real(r8), pointer :: plane(:,:)",
        "  integer :: itim",
        "  status = 0_c_int",
        "  ptr = c_null_ptr",
        "  extents = 0_c_int64_t",
        "  nullify(plane)",
        "  ! An index of -1 is how CAM says a field was never registered; the",
        "  ! six UNICON fields are like that in this configuration.",
        "  if (field_index < 1) then",
        "    status = 1_c_int",
        "    return",
        "  endif",
        "  if (lchnk < begchunk .or. lchnk > endchunk) then",
        "    status = 2_c_int",
        "    return",
        "  endif",
        "  if (.not. associated(pbuf2d)) then",
        "    status = 3_c_int",
        "    return",
        "  endif",
        "  pbuf => pbuf_get_chunk(pbuf2d, lchnk)",
        "  if (time_sliced /= 0_c_int) then",
        "    itim = pbuf_old_tim_idx()",
        "    call pbuf_get_field(pbuf, field_index, plane, &",
        "         start=(/1,1,itim/), kount=(/pcols,pver,1/))",
        "  else",
        "    call pbuf_get_field(pbuf, field_index, plane)",
        "  endif",
        "  if (.not. associated(plane)) then",
        "    status = 4_c_int",
        "    return",
        "  endif",
        "  ptr = c_loc(plane(1,1))",
        "  extents(1) = int(size(plane,1), c_int64_t)",
        "  extents(2) = int(size(plane,2), c_int64_t)",
        "end function pycam_pbuf_field_v1",
        "",
        *_pbuf_accessor_v2(),
    )


def _pbuf_accessor_v2() -> tuple[str, ...]:
    """The same handle for every shape the physics actually asks for.

    v1 serves ``(pcols, n)`` doubles.  micro_mg_cam_tend also reads rank-1
    fields (``PREC_STR`` and friends), rank-3 ones (``NACON``, ``RNDST``),
    and one integer field (``ACNUM``), so the caller now says the rank and
    the kind and receives ``ndims`` extents back.  A time-rotated field is
    always the ``(pcols, pver)`` plane at the older sample, as in v1.  A
    request the buffer cannot honour -- wrong rank for the field, wrong
    kind -- comes back unassociated and is refused with status 4, never
    guessed.
    """

    return (
        "integer(c_int) function pycam_pbuf_field_v2(lchnk, field_index, &",
        "     time_sliced, rank, is_integer, ptr, ndims, extents) &",
        "     bind(C, name='pycam_pbuf_field_v2') result(status)",
        "  use, intrinsic :: iso_c_binding, only: c_ptr, c_loc, c_int, &",
        "       c_int64_t, c_null_ptr",
        "  use physics_buffer, only: physics_buffer_desc, pbuf_get_chunk, &",
        "       pbuf_get_field, pbuf_old_tim_idx",
        "  use ppgrid, only: pcols, pver",
        "  integer(c_int), value, intent(in)  :: lchnk, field_index, time_sliced",
        "  integer(c_int), value, intent(in)  :: rank, is_integer",
        "  type(c_ptr),           intent(out) :: ptr",
        "  integer(c_int),        intent(out) :: ndims",
        "  integer(c_int64_t),    intent(out) :: extents(3)",
        "  type(physics_buffer_desc), pointer :: pbuf(:)",
        "  real(r8), pointer :: r1(:), r2(:,:), r3(:,:,:)",
        "  integer,  pointer :: i1(:), i2(:,:)",
        "  integer :: itim",
        "  status = 0_c_int",
        "  ptr = c_null_ptr",
        "  ndims = 0_c_int",
        "  extents = 0_c_int64_t",
        "  nullify(r1, r2, r3, i1, i2)",
        "  if (field_index < 1) then",
        "    status = 1_c_int",
        "    return",
        "  endif",
        "  if (lchnk < begchunk .or. lchnk > endchunk) then",
        "    status = 2_c_int",
        "    return",
        "  endif",
        "  if (.not. associated(pbuf2d)) then",
        "    status = 3_c_int",
        "    return",
        "  endif",
        "  pbuf => pbuf_get_chunk(pbuf2d, lchnk)",
        "  status = 4_c_int",
        "  if (is_integer /= 0_c_int) then",
        "    select case (rank)",
        "    case (1)",
        "      call pbuf_get_field(pbuf, field_index, i1)",
        "      if (.not. associated(i1)) return",
        "      ptr = c_loc(i1(1)); ndims = 1_c_int",
        "      extents(1) = int(size(i1,1), c_int64_t)",
        "    case (2)",
        "      call pbuf_get_field(pbuf, field_index, i2)",
        "      if (.not. associated(i2)) return",
        "      ptr = c_loc(i2(1,1)); ndims = 2_c_int",
        "      extents(1) = int(size(i2,1), c_int64_t)",
        "      extents(2) = int(size(i2,2), c_int64_t)",
        "    case default",
        "      return",
        "    end select",
        "  else",
        "    select case (rank)",
        "    case (1)",
        "      call pbuf_get_field(pbuf, field_index, r1)",
        "      if (.not. associated(r1)) return",
        "      ptr = c_loc(r1(1)); ndims = 1_c_int",
        "      extents(1) = int(size(r1,1), c_int64_t)",
        "    case (2)",
        "      if (time_sliced /= 0_c_int) then",
        "        itim = pbuf_old_tim_idx()",
        "        call pbuf_get_field(pbuf, field_index, r2, &",
        "             start=(/1,1,itim/), kount=(/pcols,pver,1/))",
        "      else",
        "        call pbuf_get_field(pbuf, field_index, r2)",
        "      endif",
        "      if (.not. associated(r2)) return",
        "      ptr = c_loc(r2(1,1)); ndims = 2_c_int",
        "      extents(1) = int(size(r2,1), c_int64_t)",
        "      extents(2) = int(size(r2,2), c_int64_t)",
        "    case (3)",
        "      call pbuf_get_field(pbuf, field_index, r3)",
        "      if (.not. associated(r3)) return",
        "      ptr = c_loc(r3(1,1,1)); ndims = 3_c_int",
        "      extents(1) = int(size(r3,1), c_int64_t)",
        "      extents(2) = int(size(r3,2), c_int64_t)",
        "      extents(3) = int(size(r3,3), c_int64_t)",
        "    case default",
        "      return",
        "    end select",
        "  endif",
        "  status = 0_c_int",
        "end function pycam_pbuf_field_v2",
        "",
    )


def _rename_cam_init_header(header: str, name: str) -> str:
    result, count = re.subn(
        r"(?im)^(\s*subroutine\s+)cam_init\b",
        rf"\1{name}",
        header,
        count=1,
    )
    if count != 1:
        raise PICAMConfigurationError("cam_init declaration changed")
    return result


def split_cam_init_for_python_state(source: str) -> str:
    """Split CAM initialization around the first state-allocation boundary.

    The admitted initial run first creates the native grid and chunk map.  It
    must then return to Python so rank-local arrays can be allocated and
    initialized with exact native chunk metadata.  ``cam_init_finish`` resumes
    at the original allocation point.  The original ``cam_init`` remains as a
    compatibility wrapper for non-Python callers.
    """

    match = re.search(
        r"(?ims)^\s*subroutine\s+cam_init\b.*?"
        r"^\s*end\s+subroutine\s+cam_init\s*$",
        source,
    )
    if match is None:
        raise PICAMConfigurationError("cannot find cam_init for initialization split")
    block = match.group(0)
    executable = re.search(r"(?im)^\s*etamid\s*=\s*nan\s*$", block)
    initial = re.search(
        r"(?im)^\s*call\s+cam_initial\(dyn_in,\s*dyn_out,\s*NLFileName=filein\)\s*$",
        block,
    )
    allocations = re.search(
        r"(?ims)^\s*! Allocate and setup surface exchange data\s*$.*?"
        r"^\s*call\s+hub2atm_alloc\(cam_in\)\s*$",
        block,
    )
    tail = re.search(
        r"(?ims)^\s*call\s+phys_init\(\s*phys_state,\s*phys_tend,\s*pbuf2d,\s*cam_out\s*\)\s*$.*?"
        r"^\s*end\s+subroutine\s+cam_init\s*$",
        block,
    )
    if any(value is None for value in (executable, initial, allocations, tail)):
        raise PICAMConfigurationError("cam_init allocation boundary changed")
    assert executable is not None
    assert initial is not None
    assert allocations is not None
    assert tail is not None
    if not (executable.start() < initial.end() <= allocations.start() < tail.start()):
        raise PICAMConfigurationError("cam_init initialization order changed")

    header = block[: executable.start()]
    prepare_header = _rename_cam_init_header(header, "cam_init_prepare")
    finish_header = _rename_cam_init_header(header, "cam_init_finish")
    prepare_body = block[executable.start() : initial.end()]
    finish_tail = block[tail.start() :]
    finish_tail = re.sub(
        r"(?im)^\s*end\s+subroutine\s+cam_init\s*$",
        "end subroutine cam_init_finish",
        finish_tail,
        count=1,
    )
    arguments = (
        "cam_out, cam_in, mpicom_atm, start_ymd, start_tod, ref_ymd, "
        "ref_tod, stop_ymd, stop_tod, perpetual_run, perpetual_ymd, calendar"
    )
    prepare = (
        prepare_header
        + prepare_body
        + "\n   else\n"
        + "      call endrun('cam_init_prepare: restart mode is not admitted')\n"
        + "   endif\n\nend subroutine cam_init_prepare"
    )
    finish = (
        finish_header
        + "   if (nsrest /= 0) then\n"
        + "      call endrun('cam_init_finish: restart mode is not admitted')\n"
        + "   endif\n\n"
        + block[allocations.start() : allocations.end()]
        + "\n\n"
        + finish_tail
    )
    wrapper = (
        header
        + f"   call cam_init_prepare({arguments})\n"
        + f"   call cam_init_finish({arguments})\n\n"
        + "end subroutine cam_init"
    )
    replacement = wrapper + "\n\n" + prepare + "\n\n" + finish
    return source[: match.start()] + replacement + source[match.end() :]


def instrument_cam_comp(
    source: str,
    include_name: str,
    *,
    split_initialization: bool = False,
) -> str:
    """Expose generated state routines without editing the source checkout."""

    if split_initialization:
        source = split_cam_init_for_python_state(source)

    public_anchor = "   public cam_final     ! CAM Finalization"
    if public_anchor not in source:
        raise PICAMConfigurationError("cam_comp public anchor changed")
    source = source.replace(
        public_anchor,
        public_anchor
        + ("\n   public cam_init_prepare, cam_init_finish" if split_initialization else "")
        + "\n   public cam_python_state_count, cam_python_state_metadata, &\n"
        + "        cam_python_state_transfer",
        1,
    )
    end_anchor = "end module cam_comp"
    if source.count(end_anchor) != 1:
        raise PICAMConfigurationError("cam_comp end-module anchor changed")
    return source.replace(end_anchor, f"include '{include_name}'\n\n{end_anchor}", 1)


def active_state_fields(bridge: LegacyStateBridge) -> tuple[LegacyStateField, ...]:
    """Return fields whose numerical storage exists in the admitted case."""

    return tuple(field for field in bridge.fields if field.active_by_default)


def generate_pointer_registry(bridge: LegacyStateBridge) -> str:
    """Generate the C ABI table retaining Python StatePool addresses.

    Fixed-size members live inside Python-owned raw derived-type records, so
    the pointer table only needs one raw buffer per owner plus the separately
    allocated/pointer members.  This preserves the production record layout
    while avoiding a second copy of every inline CAM field.
    """

    active = active_state_fields(bridge)
    external = tuple(field for field in active if field.allocatable or field.pointer)
    maximum_rank = max((field.python_rank for field in external), default=1)
    argument_count = len(bridge.owners) + len(external)
    lines = [
        "! Generated by freecam.pi_cam.state_codegen; do not edit.",
        "module pycam_python_state_registry",
        "  use, intrinsic :: iso_c_binding, only: c_char, c_int, c_int64_t, &",
        "       c_null_char, c_ptr",
        "  implicit none",
        "  private",
        f"  integer, parameter :: pycam_state_field_count = {len(bridge.fields)}",
        f"  integer, parameter :: pycam_state_owner_count = {len(bridge.owners)}",
        f"  integer, parameter :: pycam_state_max_rank = {maximum_rank}",
        "  type(c_ptr), save :: pycam_pointers(pycam_state_field_count)",
        "  type(c_ptr), save :: pycam_owner_pointers(pycam_state_owner_count)",
        "  integer(c_int64_t), save :: pycam_owner_bytes(pycam_state_owner_count) = 0_c_int64_t",
        "  integer(c_int64_t), save :: pycam_shapes(pycam_state_field_count, &",
        "       pycam_state_max_rank) = 0_c_int64_t",
        "  logical, save :: pycam_ready = .false.",
        "  public :: pycam_pi_cam_bind_state_v1, pycam_state_pointer, &",
        "       pycam_state_extent, pycam_state_is_ready, &",
        "       pycam_owner_pointer, pycam_owner_nbytes",
        "contains",
        "",
        "integer(c_int) function pycam_pi_cam_bind_state_v1(action_id, nfields, &",
        "     pointers, ndims, shapes, table_rank, fortran_comm, errmsg, &",
        "     errmsg_len) bind(C, name='pycam_pi_cam_bind_state_v1') result(status)",
        "  integer(c_int), value, intent(in) :: action_id, nfields, table_rank",
        "  integer(c_int), value, intent(in) :: fortran_comm, errmsg_len",
        "  type(c_ptr), intent(in) :: pointers(*)",
        "  integer(c_int), intent(in) :: ndims(*)",
        "  integer(c_int64_t), intent(in) :: shapes(*)",
        "  character(kind=c_char), intent(out) :: errmsg(*)",
        "  integer :: axis, index, argument_index",
        "  do index = 1, max(1, int(errmsg_len))",
        "    errmsg(index) = c_null_char",
        "  enddo",
        "  status = 0_c_int",
        "  if (action_id /= 0 .or. &",
        f"       nfields /= {argument_count} .or. table_rank < pycam_state_max_rank) then",
        "    status = 1_c_int",
        "    return",
        "  endif",
        "  pycam_shapes = 0_c_int64_t",
        "  argument_index = 0",
    ]
    for owner_id, _owner in enumerate(bridge.owners, start=1):
        lines.extend(
            (
                "  argument_index = argument_index + 1",
                "  if (ndims(argument_index) /= 1) then",
                f"    status = {1000 + owner_id}_c_int",
                "    return",
                "  endif",
                f"  pycam_owner_pointers({owner_id}) = pointers(argument_index)",
                f"  pycam_owner_bytes({owner_id}) = &",
                "       shapes((argument_index-1)*table_rank+1)",
            )
        )
    for field in external:
        lines.extend(
            (
                "  argument_index = argument_index + 1",
                f"  if (ndims(argument_index) /= {field.python_rank}) then",
                f"    status = {10 + field.field_id}_c_int",
                "    return",
                "  endif",
                f"  pycam_pointers({field.field_id}) = pointers(argument_index)",
                f"  do axis = 1, {field.python_rank}",
                f"    pycam_shapes({field.field_id},axis) = &",
                "         shapes((argument_index-1)*table_rank+axis)",
                "  enddo",
            )
        )
    lines.extend(
        (
            "  pycam_ready = .true.",
            "end function pycam_pi_cam_bind_state_v1",
            "",
            "logical function pycam_state_is_ready() result(ready)",
            "  ready = pycam_ready",
            "end function pycam_state_is_ready",
            "",
            "function pycam_state_pointer(field_id) result(value)",
            "  integer, intent(in) :: field_id",
            "  type(c_ptr) :: value",
            "  value = pycam_pointers(field_id)",
            "end function pycam_state_pointer",
            "",
            "integer function pycam_state_extent(field_id, axis) result(value)",
            "  integer, intent(in) :: field_id, axis",
            "  value = int(pycam_shapes(field_id,axis))",
            "end function pycam_state_extent",
            "",
            "function pycam_owner_pointer(owner_id) result(value)",
            "  integer, intent(in) :: owner_id",
            "  type(c_ptr) :: value",
            "  value = pycam_owner_pointers(owner_id)",
            "end function pycam_owner_pointer",
            "",
            "integer(c_int64_t) function pycam_owner_nbytes(owner_id) result(value)",
            "  integer, intent(in) :: owner_id",
            "  value = pycam_owner_bytes(owner_id)",
            "end function pycam_owner_nbytes",
            "",
            "end module pycam_python_state_registry",
            "",
        )
    )
    return "\n".join(lines)


def _shell_declaration(field: LegacyStateField) -> str:
    """Preserve inline layout and pointerize only separately stored arrays.

    Intel uses layout-compatible descriptors for allocatable and pointer array
    components.  Keeping inline scalars/fixed-size arrays inline lets the
    unchanged production numerical object operate on the generated shell,
    while deferred storage can be associated with Python arrays.
    """

    declaration = "real(r8)" if field.dtype == "real" else "integer"
    if not field.allocatable and not field.pointer:
        dimensions = "" if field.component_rank == 0 else "(" + ",".join(
            field.source_dimensions
        ) + ")"
        return f"     {declaration} :: {field.member}{dimensions}"
    dimensions = "" if field.component_rank == 0 else "(" + ",".join(
        ":" for _ in range(field.component_rank)
    ) + ")"
    # An allocatable component is always contiguous.  StatePool arrays have
    # the same guarantee, so retain it explicitly after replacing the
    # component by a pointer.  Without CONTIGUOUS Intel can legally select a
    # different scalar/vector path in every consumer of the generated .mod,
    # which changes the final bits even though the source expression is the
    # same.
    attributes = "pointer" if field.component_rank == 0 else "pointer, contiguous"
    return (
        f"     {declaration}, {attributes} :: "
        f"{field.member}{dimensions} => null()"
    )


def generate_owner_binder(
    owner: LegacyStateOwner,
    fields: tuple[LegacyStateField, ...],
    owner_id: int,
) -> str:
    """Generate one Python raw-record binder and layout query."""

    active = tuple(field for field in fields if field.active_by_default)
    external = tuple(field for field in active if field.allocatable or field.pointer)
    inline = tuple(field for field in active if field not in external)
    lines = [
        f"subroutine pycam_bind_{owner.name}({owner.name}, begchunk_in, endchunk_in)",
        "  use, intrinsic :: iso_c_binding, only: c_f_pointer",
        "  use pycam_python_state_registry, only: pycam_state_pointer, &",
        "       pycam_state_extent, pycam_state_is_ready, pycam_owner_pointer",
        f"  type({owner.type_name}), pointer :: {owner.name}(:)",
        f"  type({owner.type_name}), pointer :: pycam_owner_records(:)",
        "  integer, intent(in) :: begchunk_in, endchunk_in",
        "  integer :: c, local_chunk, nchunks",
    ]
    for field in external:
        kind = "real(r8)" if field.dtype == "real" else "integer"
        dimensions = "(" + ",".join(":" for _ in range(field.python_rank)) + ")"
        lines.append(
            f"  {kind}, pointer, contiguous :: "
            f"pycam_field_{field.field_id}{dimensions}"
        )
    lines.extend(
        (
            "  if (.not. pycam_state_is_ready()) then",
            f"    call endrun('pycam_bind_{owner.name}: Python state table is absent')",
            "  endif",
            "  nchunks = endchunk_in - begchunk_in + 1",
            "  if (nchunks < 1) then",
            f"    call endrun('pycam_bind_{owner.name}: invalid chunk range')",
            "  endif",
            f"  call c_f_pointer(pycam_owner_pointer({owner_id}), &",
            "       pycam_owner_records, (/ nchunks /))",
            f"  {owner.name}(begchunk_in:endchunk_in) => pycam_owner_records",
            f"  if (size({owner.name}) /= nchunks) then",
            f"    call endrun('pycam_bind_{owner.name}: chunk count differs')",
            "  endif",
        )
    )
    for field in external:
        shape = ", ".join(
            f"pycam_state_extent({field.field_id},{axis})"
            for axis in range(1, field.python_rank + 1)
        )
        lines.extend(
            (
                f"  call c_f_pointer(pycam_state_pointer({field.field_id}), &",
                f"       pycam_field_{field.field_id}, (/ {shape} /))",
            )
        )
    lines.extend(
        (
            f"  do c = lbound({owner.name},1), ubound({owner.name},1)",
            f"    local_chunk = c - lbound({owner.name},1) + 1",
        )
    )
    for field in external:
        if field.component_rank == 0:
            section = "local_chunk"
        else:
            section = ",".join(
                (*((":" for _ in range(field.component_rank))), "local_chunk")
            )
        lines.append(
            f"    {owner.name}(c)%{field.member} => "
            f"pycam_field_{field.field_id}({section})"
        )
    lines.extend(("  enddo", f"end subroutine pycam_bind_{owner.name}", ""))
    layout_symbol = f"pycam_pi_cam_layout_{owner.name}_v1"
    lines.extend(
        (
            f"integer function {layout_symbol}(record_bytes, nfields, &",
            "     field_ids, offsets) bind(C, &",
            f"     name='{layout_symbol}') result(status)",
            "  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_intptr_t",
            "  integer(c_int64_t), intent(out) :: record_bytes",
            "  integer(c_int), value, intent(in) :: nfields",
            "  integer(c_int), intent(out) :: field_ids(*)",
            "  integer(c_int64_t), intent(out) :: offsets(*)",
            f"  type({owner.type_name}), target :: probe(2)",
            "  integer(c_intptr_t) :: base_address",
            "  status = 0_c_int",
            f"  if (nfields < {len(inline)}) then",
            "    status = 1_c_int",
            "    return",
            "  endif",
            "  base_address = loc(probe(1))",
            "  record_bytes = int(loc(probe(2)) - base_address, c_int64_t)",
        )
    )
    for index, field in enumerate(inline, start=1):
        lines.extend(
            (
                f"  field_ids({index}) = {field.field_id}_c_int",
                f"  offsets({index}) = int(loc(probe(1)%{field.member}) - &",
                "       base_address, c_int64_t)",
            )
        )
    lines.extend((f"end function {layout_symbol}", ""))
    return "\n".join(lines)


def pointerize_owner_type(
    source: str,
    owner: LegacyStateOwner,
    fields: tuple[LegacyStateField, ...],
) -> str:
    """Replace one numeric state type by a pointer-only shell."""

    lines = source.splitlines()
    start = end = None
    for index, line in enumerate(lines):
        match = _TYPE_START.match(_without_comment(line).strip())
        if start is None and match and match.group(1).lower() == owner.type_name.lower():
            start = index
            continue
        if start is not None and _TYPE_END.match(_without_comment(line).strip()):
            end = index
            break
    if start is None or end is None:
        raise PICAMConfigurationError(
            f"cannot pointerize type {owner.type_name!r} in {owner.source}"
        )
    replacement = [f"  type {owner.type_name}"]
    replacement.extend(_shell_declaration(field) for field in fields)
    replacement.append(f"  end type {owner.type_name}")
    return "\n".join((*lines[:start], *replacement, *lines[end + 1 :])) + "\n"


def insert_owner_binder(source: str, owner: LegacyStateOwner, binder: str) -> str:
    """Insert a generated binder as the first module procedure."""

    match = re.search(r"(?im)^\s*contains\s*$", source)
    if match is None:
        raise PICAMConfigurationError(f"{owner.source} has no module CONTAINS")
    return source[: match.end()] + "\n\n" + binder + source[match.end() :]
