#!/usr/bin/env python3
"""Generate CAM SourceMods whose numerical state storage is Python-owned."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.pi_cam.errors import PICAMConfigurationError  # noqa: E402
from freecam.pi_cam.state_codegen import (  # noqa: E402
    LegacyStateField,
    LegacyStateOwner,
    generate_owner_binder,
    generate_pointer_registry,
    insert_owner_binder,
    load_state_bridge,
    pointerize_owner_type,
)


def _routine(source: str, name: str) -> tuple[slice, str]:
    match = re.search(
        rf"(?ims)^\s*subroutine\s+{re.escape(name)}\b.*?"
        rf"^\s*end\s+subroutine(?:\s+{re.escape(name)})?\s*$",
        source,
    )
    if match is None:
        raise PICAMConfigurationError(f"cannot find subroutine {name!r}")
    return slice(match.start(), match.end()), match.group(0)


def _replace_routine(source: str, name: str, transform) -> str:
    location, block = _routine(source, name)
    return source[: location.start] + transform(block) + source[location.stop :]


def _component_statement(line: str, fields: tuple[LegacyStateField, ...]) -> str | None:
    lower = line.lower()
    for field in fields:
        token = rf"%{re.escape(field.member.lower())}\b"
        if re.search(token, lower):
            return field.member
    return None


def _remove_component_allocations(
    block: str,
    fields: tuple[LegacyStateField, ...],
) -> str:
    result: list[str] = []
    removed_allocation = False
    for line in block.splitlines():
        member = _component_statement(line, fields)
        lower = line.lower()
        if member is not None and re.search(r"\ballocate\s*\(", lower):
            result.append(f"  ! Python owns %{member}; allocation removed by generator.")
            removed_allocation = True
        elif removed_allocation and re.search(
            r"\bif\s*\([^)]*(?:ierror|ierr)\s*/=\s*0[^)]*\)",
            lower,
        ):
            result.append("  ! Allocation status check removed with Python-owned storage.")
            removed_allocation = False
        else:
            result.append(line)
            if line.strip() and not line.lstrip().startswith("!"):
                removed_allocation = False
    return "\n".join(result)


def _remove_outer_allocation(block: str, argument: str) -> str:
    """Remove allocation/error handling for a Python-owned record array."""

    pattern = re.compile(
        rf"(?ims)^\s*allocate\s*\(\s*{re.escape(argument)}\s*"
        r"\(begchunk\s*:\s*endchunk\)\s*,\s*stat\s*=\s*"
        r"(?:ierror|ierr)\s*\)\s*$.*?^\s*end\s*if\s*$"
    )
    result, count = pattern.subn(
        f"    ! Python owns the {argument} record array; allocation removed.",
        block,
        count=1,
    )
    if count != 1:
        raise PICAMConfigurationError(
            f"cannot find outer allocation for {argument!r}"
        )
    return result


def _make_allocator_python_aware(
    block: str,
    *,
    argument: str,
    fields: tuple[LegacyStateField, ...],
) -> str:
    """Keep the legacy allocator for temporaries, but skip it for Python state."""

    remainder, replacements = re.subn(
        r"(?im)^(\s*subroutine\s+[A-Za-z_]\w*\s*\([^\n)]*)(\))",
        r"\1,python_storage\2",
        block,
        count=1,
    )
    if replacements != 1:
        raise PICAMConfigurationError("allocator declaration changed")

    declaration = re.search(
        rf"(?im)^\s*type\([^\n]+\),\s*intent\(inout\)\s*::\s*{re.escape(argument)}\s*$",
        remainder,
    )
    if declaration is None:
        raise PICAMConfigurationError(
            f"cannot find {argument} declaration in generated allocator"
        )
    insertion = (
        "\n  logical, optional, intent(in) :: python_storage"
        "\n  logical :: use_python_storage"
    )
    remainder = (
        remainder[: declaration.end()]
        + insertion
        + remainder[declaration.end() :]
    )

    local_declarations = re.search(r"(?im)^\s*integer\s*::\s*ierr[^\n]*$", remainder)
    if local_declarations is None:
        raise PICAMConfigurationError("allocator ierr declaration changed")
    initialization = (
        "\n\n  use_python_storage = .false."
        "\n  if (present(python_storage)) use_python_storage = python_storage"
    )
    remainder = (
        remainder[: local_declarations.end()]
        + initialization
        + remainder[local_declarations.end() :]
    )

    scalar_fields = tuple(
        field
        for field in fields
        if field.component_rank == 0 and (field.allocatable or field.pointer)
    )
    first_assignment = min(
        (
            match.start()
            for field in scalar_fields
            if (
                match := re.search(
                    rf"(?im)^\s*{re.escape(argument)}%{re.escape(field.member)}\s*=",
                    remainder,
                )
            )
        ),
        default=None,
    )
    if scalar_fields and first_assignment is None:
        raise PICAMConfigurationError(
            f"cannot find scalar initialization for {argument}"
        )
    if first_assignment is not None:
        allocations = ["  if (.not. use_python_storage) then"]
        for field in scalar_fields:
            allocations.extend(
                (
                    f"    allocate({argument}%{field.member}, stat=ierr)",
                    f"    if (ierr /= 0) call endrun('{argument}%{field.member} scalar allocation failed')",
                )
            )
        allocations.append("  endif")
        remainder = (
            remainder[:first_assignment]
            + "\n".join(allocations)
            + "\n\n"
            + remainder[first_assignment:]
        )

    lines: list[str] = []
    for line in remainder.splitlines():
        member = _component_statement(line, fields)
        if (
            member is not None
            and re.search(r"\ballocate\s*\(", line, re.I)
            and "use_python_storage" not in line
        ):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(f"{indent}if (.not. use_python_storage) {line.lstrip()}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _add_scalar_deallocations(
    block: str,
    *,
    argument: str,
    fields: tuple[LegacyStateField, ...],
) -> str:
    scalar_fields = tuple(
        field
        for field in fields
        if field.component_rank == 0 and (field.allocatable or field.pointer)
    )
    if not scalar_fields:
        return block
    end = re.search(r"(?im)^\s*end\s+subroutine", block)
    if end is None:
        raise PICAMConfigurationError("deallocator end changed")
    lines = [""]
    for field in scalar_fields:
        lines.extend(
            (
                f"  deallocate({argument}%{field.member}, stat=ierr)",
                f"  if (ierr /= 0) call endrun('{argument}%{field.member} scalar deallocation failed')",
            )
        )
    lines.append("")
    return block[: end.start()] + "\n".join(lines) + block[end.start() :]


def _replace_component_deallocations(
    block: str,
    fields: tuple[LegacyStateField, ...],
) -> str:
    result: list[str] = []
    expression = re.compile(
        r"deallocate\s*\(\s*([^,%]+)%([A-Za-z_]\w*)\s*\)", re.I
    )
    for line in block.splitlines():
        match = expression.search(line)
        if match and any(match.group(2).lower() == field.member.lower() for field in fields):
            indent = line[: len(line) - len(line.lstrip())]
            result.append(
                f"{indent}nullify({match.group(1)}%{match.group(2)})"
            )
        else:
            result.append(line)
    return "\n".join(result)


def _transform_camsrfexch(
    source: str,
    owners: dict[str, tuple[LegacyStateOwner, tuple[LegacyStateField, ...]]],
    owner_ids: dict[str, int],
) -> str:
    for owner, fields in owners.values():
        source = pointerize_owner_type(source, owner, fields)
        source = insert_owner_binder(
            source,
            owner,
            generate_owner_binder(owner, fields, owner_ids[owner.name]),
        )

    cam_in_fields = owners["cam_in"][1]
    cam_out_fields = owners["cam_out"][1]

    def hub_alloc(block: str) -> str:
        block = _remove_outer_allocation(block, "cam_in")
        block = _remove_component_allocations(block, cam_in_fields)
        block = re.sub(
            r"(?im)^\s*nullify\(cam_in\(c\)%[A-Za-z_]\w*\)\s*$",
            "  ! Pointer association is supplied by Python.",
            block,
        )
        anchor = re.search(r"(?im)^\s*do\s+c\s*=\s*begchunk\s*,\s*endchunk\s*$", block)
        if anchor is None:
            raise PICAMConfigurationError("hub2atm_alloc loop anchor changed")
        return (
            block[: anchor.start()]
            + "    call pycam_bind_cam_in(cam_in, begchunk, endchunk)\n"
            + "    ! Python already initialized every admitted component.\n"
            + "    return\n"
            + block[anchor.start() :]
        )

    def out_alloc(block: str) -> str:
        block = _remove_outer_allocation(block, "cam_out")
        anchor = re.search(r"(?im)^\s*do\s+c\s*=\s*begchunk\s*,\s*endchunk\s*$", block)
        if anchor is None:
            raise PICAMConfigurationError("atm2hub_alloc loop anchor changed")
        return (
            block[: anchor.start()]
            + "    call pycam_bind_cam_out(cam_out, begchunk, endchunk)\n"
            + "    ! Python already initialized every admitted component.\n"
            + "    return\n"
            + block[anchor.start() :]
        )

    source = _replace_routine(source, "hub2atm_alloc", hub_alloc)
    source = _replace_routine(source, "atm2hub_alloc", out_alloc)
    source = _replace_routine(
        source,
        "atm2hub_deallocate",
        lambda block: re.sub(
            r"(?im)^\s*deallocate\(cam_out\)\s*$",
            "       nullify(cam_out)",
            block,
        ),
    )
    source = _replace_routine(
        source,
        "hub2atm_deallocate",
        lambda block: re.sub(
            r"(?im)^\s*deallocate\(cam_in\)\s*$",
            "       nullify(cam_in)",
            _replace_component_deallocations(block, cam_in_fields),
        ),
    )
    return source


def _transform_physics_types(
    source: str,
    owners: dict[str, tuple[LegacyStateOwner, tuple[LegacyStateField, ...]]],
    owner_ids: dict[str, int],
) -> str:
    for owner, fields in owners.values():
        source = pointerize_owner_type(source, owner, fields)
        source = insert_owner_binder(
            source,
            owner,
            generate_owner_binder(owner, fields, owner_ids[owner.name]),
        )

    state_fields = owners["phys_state"][1]
    tend_fields = owners["phys_tend"][1]

    def type_alloc(block: str) -> str:
        block = _remove_outer_allocation(block, "phys_state")
        block = _remove_outer_allocation(block, "phys_tend")
        state_anchor = "    do lchnk=begchunk,endchunk\n       call physics_state_alloc"
        tend_anchor = "    do lchnk=begchunk,endchunk\n       call physics_tend_alloc"
        if state_anchor not in block or tend_anchor not in block:
            raise PICAMConfigurationError("physics_type_alloc loop anchor changed")
        block = block.replace(
            "    do lchnk=begchunk,endchunk\n"
            "       call physics_state_alloc(phys_state(lchnk),lchnk,pcols)\n"
            "    end do",
            "    call pycam_bind_phys_state(phys_state, begchunk, endchunk)",
            1,
        )
        block = block.replace(
            "    do lchnk=begchunk,endchunk\n"
            "       call physics_tend_alloc(phys_tend(lchnk),phys_state(lchnk)%psetcols)\n"
            "    end do",
            "    call pycam_bind_phys_tend(phys_tend, begchunk, endchunk)\n\n"
            "    ! Python already initialized every admitted component.\n"
            "    return",
            1,
        )
        return block

    source = _replace_routine(source, "physics_type_alloc", type_alloc)
    source = _replace_routine(
        source,
        "physics_state_alloc",
        lambda block: _make_allocator_python_aware(
            block,
            argument="state",
            fields=state_fields,
        ),
    )
    source = _replace_routine(
        source,
        "physics_tend_alloc",
        lambda block: _make_allocator_python_aware(
            block,
            argument="tend",
            fields=tend_fields,
        ),
    )
    source = _replace_routine(
        source,
        "physics_state_dealloc",
        lambda block: _add_scalar_deallocations(
            block,
            argument="state",
            fields=state_fields,
        ),
    )
    source = _replace_routine(
        source,
        "physics_tend_dealloc",
        lambda block: _add_scalar_deallocations(
            block,
            argument="tend",
            fields=tend_fields,
        ),
    )
    source = re.sub(
        r"\ballocated\(\s*tend%dtdt\s*\)",
        "associated(tend%dtdt)",
        source,
        flags=re.I,
    )
    source = source.replace("state%q(1,1,m)", "state%q(:,:,m)")
    return source


def _transform_pointer_compatibility(source: str) -> str:
    source = re.sub(
        r"\ballocated\(\s*((?:state|tend)(?:_sc)?%[A-Za-z_]\w*)\s*\)",
        r"associated(\1)",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"(?im)^(\s*\{VTYPE\},\s*)allocatable(\s*::\s*field_sc\{DIMSTR\}.*)$",
        r"\1pointer    \2",
        source,
    )
    source = re.sub(
        r"\ballocated\(\s*field_sc\s*\)",
        "associated(field_sc)",
        source,
        flags=re.I,
    )
    return source


def _transform_dp_coupling(source: str) -> str:
    old = "phys_state(lchnk)%q(1,1,m), pcols,lchnk"
    new = "phys_state(lchnk)%q(:,:,m), pcols,lchnk"
    if source.count(old) != 1:
        raise PICAMConfigurationError("dp_coupling initial-history call changed")
    return source.replace(old, new)


def _transform_cam_diagnostics(source: str) -> str:
    """Replace legacy sequence association with explicit pointer sections."""

    source = re.sub(
        r"state%q\(1\s*,\s*1\s*,\s*([^\n,)]+(?:\([^\n)]*\))?)\)",
        r"state%q(:,:,\1)",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"state%q\(1\s*,\s*pver\s*,\s*([^\n,)]+(?:\([^\n)]*\))?)\)",
        r"state%q(:,pver,\1)",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"state%(u|v|zm)\(1\s*,\s*pver\s*\)",
        r"state%\1(:,pver)",
        source,
        flags=re.I,
    )
    source = re.sub(
        r"cam_in%cflx\(1\s*,\s*([^\n,)]+(?:\([^\n)]*\))?)\)",
        r"cam_in%cflx(:,\1)",
        source,
        flags=re.I,
    )
    return source


def _transform_stratiform(source: str) -> str:
    old = "state1%q(1,1,1)"
    if source.count(old) != 1:
        raise PICAMConfigurationError("stratiform pcond call changed")
    return source.replace(old, "state1%q(:,:,1)")


def _transform_physpkg(source: str) -> str:
    replacements = {
        "state%q(1,pver,1)": "state%q(:,pver,1)",
        "state%rpdel(1,pver)": "state%rpdel(:,pver)",
        "state%q(1,1,n)": "state%q(:,:,n)",
    }
    for old, new in replacements.items():
        if old not in source:
            raise PICAMConfigurationError(f"physpkg call changed: {old}")
        source = source.replace(old, new)
    source = re.sub(
        r"(?im)^\s*deallocate\(phys_(state|tend)\)\s*$",
        r"    nullify(phys_\1)",
        source,
    )
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO / "build/iCESM1.3.1_PI_cam_only",
    )
    parser.add_argument(
        "--description",
        type=Path,
        default=REPO / "native/pi_cam/state_bridge.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    bridge = load_state_bridge(args.description.resolve(), source_root)
    grouped: dict[Path, dict[str, tuple[LegacyStateOwner, tuple[LegacyStateField, ...]]]] = defaultdict(dict)
    for owner in bridge.owners:
        grouped[owner.source][owner.name] = (
            owner,
            tuple(field for field in bridge.fields if field.owner == owner),
        )

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    owner_ids = {
        owner.name: owner_id
        for owner_id, owner in enumerate(bridge.owners, start=1)
    }
    for path, owners in grouped.items():
        source = path.read_text()
        names = set(owners)
        if names == {"cam_in", "cam_out"}:
            transformed = _transform_camsrfexch(source, owners, owner_ids)
        elif names == {"phys_state", "phys_tend"}:
            transformed = _transform_physics_types(source, owners, owner_ids)
        else:
            raise PICAMConfigurationError(
                f"unsupported Python-state owner group {sorted(names)}"
            )
        (output / path.name).write_text(transformed)

    registry = output / "pycam_python_state_registry.F90"
    registry.write_text(generate_pointer_registry(bridge))
    compatibility_sources = (
        (
            source_root / "components/cam/src/physics/cam/subcol_utils.F90.in",
            _transform_pointer_compatibility,
        ),
        (
            source_root / "components/cam/src/physics/cam/subcol.F90",
            _transform_pointer_compatibility,
        ),
        (
            source_root / "components/cam/src/dynamics/se/dp_coupling.F90",
            _transform_dp_coupling,
        ),
        (
            source_root / "components/cam/src/physics/cam/cam_diagnostics.F90",
            _transform_cam_diagnostics,
        ),
        (
            source_root / "components/cam/src/physics/cam/stratiform.F90",
            _transform_stratiform,
        ),
        (
            source_root / "components/cam/src/physics/cam/physpkg.F90",
            _transform_physpkg,
        ),
    )
    for path, transform in compatibility_sources:
        (output / path.name).write_text(transform(path.read_text()))
    print(
        f"generated {registry}, {len(grouped)} pointer-shell modules, and "
        f"{len(compatibility_sources)} pointer-compatible consumers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
