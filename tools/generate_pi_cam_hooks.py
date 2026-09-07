#!/usr/bin/env python3
"""Emit pycam_hooks.F90: one hook procedure per kernel in native/pi_cam/hooks.yaml.

    tools/generate_pi_cam_hooks.py            # write native/pi_cam/support/pycam_hooks.F90
    tools/generate_pi_cam_hooks.py --check    # fail if the committed module is stale

A hook has the callee's own argument list (from its reviewed function
contract) and the symbol its redirected callers call.  It counts every call.
Unarmed, it calls the original and returns.  Armed -- the owning stage has a
replacement installed and runs on the fiber -- it records the frame (each
argument's address, rank, extents, dtype and intent, from the contract),
yields the fiber to Python with NEEDS_PYTHON_KERNEL, and when resumed returns
to its caller without calling the original: Python has written the outputs.
The original can be run on the paused frame from the main context
(pycam_hooks_original_v1), which is what the validation gates do.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from freecam.physics.spec import FunctionSpec, load_function_spec  # noqa: E402
from freecam.pi_cam.hooks import HOOK_MODULE, Hook, HookTable, load_hooks  # noqa: E402

FRAME_MAX_RANK = 5
C_TYPES = {"float64": "real(c_double)", "int32": "integer(c_int)", "int64": "integer(c_int64_t)"}
DTYPE_CODE = {"float64": 1, "int32": 2, "int64": 2}
INTENT_CODE = {"in": 0, "out": 1, "inout": 2}
ROLE_INTENT = {"structural": "in", "input": "in", "inout": "inout", "output": "out", "workspace": "inout", "result": "out"}


def _contract(hook: Hook) -> FunctionSpec:
    return load_function_spec(str(REPO / hook.contract))


def _dummies(spec: FunctionSpec):
    return [item for item in spec.arguments if item.role != "result"]


def _hook_procedure(hook: Hook, spec: FunctionSpec, index: int) -> str:
    dummies = _dummies(spec)
    if spec.result is not None:
        raise SystemExit(f"hook {hook.kernel}: function kernels are not hooked yet (the result needs a value slot)")
    if any(item.optional or item.carrier or item.pointer for item in dummies):
        raise SystemExit(f"hook {hook.kernel}: optional, logical, character or pointer dummies are not hooked yet")
    names = ", ".join(item.name for item in dummies)
    lines = [f"  subroutine hook_{hook.kernel}({names}) bind(C, name='{hook.symbol}')",
             f"    ! {spec.qualified_name}, as its redirected callers call it; {hook.redirect}"]
    for item in dummies:
        shape = "(*)" if item.rank else ""
        lines.append(f"    {C_TYPES[item.dtype]}, intent({ROLE_INTENT[item.role]}), target :: {item.name}{shape}")
    lines.append("    integer :: slot")
    lines.append(f"    calls({index}) = calls({index}) + 1_c_int64_t")
    lines.append(f"    if (.not. armed({index})) then")
    lines.append(f"      call original_{hook.kernel}({names})")
    lines.append("      return")
    lines.append("    end if")
    lines.append("    if (pycam_fiber_running_v1() == 0_c_int) then")
    lines.append("      ! armed, yet not on the owning runner's fiber: the original answers, and the miss is counted")
    lines.append(f"      missed({index}) = missed({index}) + 1_c_int64_t")
    lines.append(f"      call original_{hook.kernel}({names})")
    lines.append("      return")
    lines.append("    end if")
    lines.append(f"    paused({index}) = paused({index}) + 1_c_int64_t")
    lines.append("    slot = 0")
    lines.append("    frame_shapes = 0_c_int64_t")
    for item in dummies:
        first = f"{item.name}(1)" if item.rank else item.name
        extents = ", ".join(f"int({spec.dimensions[axis]}, c_int64_t)" for axis in item.native_shape)
        lines.append("    slot = slot + 1")
        lines.append(f"    frame_ptrs(slot) = c_loc({first})")
        lines.append(f"    frame_ndims(slot) = {item.rank}_c_int")
        if item.rank:
            lines.append(f"    frame_shapes(1:{item.rank}, slot) = (/ {extents} /)")
        lines.append(f"    frame_dtypes(slot) = {DTYPE_CODE[item.dtype]}_c_int")
        lines.append(f"    frame_intents(slot) = {INTENT_CODE[ROLE_INTENT[item.role]]}_c_int")
    lines.append(f"    frame_nslots = {len(dummies)}_c_int")
    ncol = next((item.name for item in dummies if item.name.lower() == "ncol"), None)
    lines.append(f"    frame_ncol = int({ncol}, c_int)" if ncol else "    frame_ncol = 0_c_int")
    lines.append(f"    paused_hook = {index}")
    lines.append("    call pycam_fiber_yield_v1(ev_needs_kernel)")
    lines.append("    ! resumed: Python wrote the outputs (or ran the original on this frame); the call is done")
    lines.append("    paused_hook = 0")
    lines.append(f"  end subroutine hook_{hook.kernel}")
    return "\n".join(lines)


def _original_procedure(hook: Hook, spec: FunctionSpec) -> str:
    """The original callee, reached through its module or its own symbol."""

    dummies = _dummies(spec)
    names = ", ".join(item.name for item in dummies)
    lines = [f"  subroutine original_{hook.kernel}({names})"]
    if hook.original_module:
        lines.append(f"    use {hook.original_module}, only: {hook.original_routine}")
    for item in dummies:
        shape = "(*)" if item.rank else ""
        lines.append(f"    {C_TYPES[item.dtype]}, intent({ROLE_INTENT[item.role]}) :: {item.name}{shape}")
    if hook.original_module:
        lines.append(f"    call {hook.original_routine}({names})")
    else:
        lines.append(f"    call {hook.kernel}_by_symbol({names})")
    lines.append(f"  end subroutine original_{hook.kernel}")
    return "\n".join(lines)


def _symbol_interface(hook: Hook, spec: FunctionSpec) -> list[str]:
    """An explicit interface to a callee reached by symbol (a private or renamed procedure)."""

    if hook.original_module:
        return []
    dummies = _dummies(spec)
    names = ", ".join(item.name for item in dummies)
    lines = [f"    subroutine {hook.kernel}_by_symbol({names}) bind(C, name='{hook.original_symbol}')",
             "      import :: c_double, c_int, c_int64_t"]
    for item in dummies:
        shape = "(*)" if item.rank else ""
        lines.append(f"      {C_TYPES[item.dtype]}, intent({ROLE_INTENT[item.role]}) :: {item.name}{shape}")
    lines.append(f"    end subroutine {hook.kernel}_by_symbol")
    return lines


def _frame_original(hook: Hook, spec: FunctionSpec, index: int) -> str:
    """Run the original on the paused frame from the main context: pointers back to arrays."""

    dummies = _dummies(spec)
    lines = [f"    case ({index})"]
    for slot, item in enumerate(dummies, start=1):
        if item.rank:
            extents = ", ".join(str(spec.dimensions[axis]) for axis in item.native_shape)
            lines.append(f"      call c_f_pointer(frame_ptrs({slot}), p{slot}_{item.dtype[0]}{item.rank}, (/ {extents} /))")
        else:
            lines.append(f"      call c_f_pointer(frame_ptrs({slot}), s{slot}_{item.dtype[0]})")
    actuals = ", ".join(f"p{slot}_{item.dtype[0]}{item.rank}" if item.rank else f"s{slot}_{item.dtype[0]}"
                        for slot, item in enumerate(dummies, start=1))
    lines.append(f"      call original_{hook.kernel}({actuals})")
    return "\n".join(lines)


def _pointer_declarations(table: HookTable) -> list[str]:
    seen: set[str] = set()
    lines = []
    for hook in table.hooks:
        spec = _contract(hook)
        for slot, item in enumerate(_dummies(spec), start=1):
            if item.rank:
                name = f"p{slot}_{item.dtype[0]}{item.rank}"
                declaration = f"    {C_TYPES[item.dtype]}, pointer :: {name}({','.join(':' for _ in range(item.rank))})"
            else:
                name = f"s{slot}_{item.dtype[0]}"
                declaration = f"    {C_TYPES[item.dtype]}, pointer :: {name}"
            if name not in seen:
                seen.add(name)
                lines.append(declaration)
    return lines


def render(table: HookTable) -> str:
    specs = [_contract(hook) for hook in table.hooks]
    max_slots = max(len(_dummies(spec)) for spec in specs) if specs else 1
    interfaces = []
    for hook, spec in zip(table.hooks, specs):
        interfaces.extend(_symbol_interface(hook, spec))
    hook_names = ", ".join(hook.kernel for hook in table.hooks)
    procedures = []
    for index, (hook, spec) in enumerate(zip(table.hooks, specs), start=1):
        procedures.append(_hook_procedure(hook, spec, index))
        procedures.append("")
        procedures.append(_original_procedure(hook, spec))
        procedures.append("")
    original_cases = "\n".join(_frame_original(hook, spec, index) for index, (hook, spec) in enumerate(zip(table.hooks, specs), start=1))
    kernel_names = "\n".join(f"  character(len=*), parameter :: name_{i} = '{hook.kernel}'" for i, hook in enumerate(table.hooks, start=1))
    return f'''! Hooks: kernels reached inside compiled routines, their callers' references
! redirected at link time to these procedures.  Each counts its calls, calls the
! original when unarmed, and when armed hands Python the frame by yielding the
! stage's fiber.  Fortran never calls Python.
!
! GENERATED by tools/generate_pi_cam_hooks.py from native/pi_cam/hooks.yaml and
! the kernels' function contracts.  Do not edit by hand.
module pycam_hooks
  use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t, c_double, c_ptr, c_loc, c_f_pointer, c_null_ptr, c_char, c_null_char
  implicit none
  private
  public :: pycam_hooks_arm_v1, pycam_hooks_counts_v1, pycam_hooks_paused_v1, pycam_hooks_frame_v1, &
            pycam_hooks_original_v1, pycam_hooks_reset_v1, pycam_hooks_count_v1, pycam_hooks_name_v1

  integer, parameter :: nhooks = {len(table.hooks)}
  integer(c_int), parameter :: ev_needs_kernel = 1_c_int
  integer, parameter :: max_slots = {max_slots}
  integer, parameter :: max_rank = {FRAME_MAX_RANK}
{kernel_names}

  logical, save :: armed(nhooks) = .false.
  integer(c_int64_t), save :: calls(nhooks) = 0_c_int64_t
  integer(c_int64_t), save :: paused(nhooks) = 0_c_int64_t
  integer(c_int64_t), save :: missed(nhooks) = 0_c_int64_t
  integer, save :: paused_hook = 0
  type(c_ptr), save :: frame_ptrs(max_slots)
  integer(c_int), save :: frame_ndims(max_slots) = 0_c_int, frame_dtypes(max_slots) = 0_c_int, frame_intents(max_slots) = 0_c_int
  integer(c_int64_t), save :: frame_shapes(max_rank, max_slots) = 0_c_int64_t
  integer(c_int), save :: frame_nslots = 0_c_int, frame_ncol = 0_c_int

  interface
    integer(c_int) function pycam_fiber_running_v1() bind(C, name='pycam_fiber_running_v1')
      import :: c_int
    end function pycam_fiber_running_v1
    subroutine pycam_fiber_yield_v1(event) bind(C, name='pycam_fiber_yield_v1')
      import :: c_int
      integer(c_int), value :: event
    end subroutine pycam_fiber_yield_v1
{chr(10).join(interfaces)}
  end interface

contains

{chr(10).join(procedures)}
  ! ------------------------------------------------------------------ !
  ! The ABI the runners and Python drive
  ! ------------------------------------------------------------------ !

  integer(c_int) function pycam_hooks_count_v1() bind(C, name='pycam_hooks_count_v1') result(count)
    count = int(nhooks, c_int)
  end function pycam_hooks_count_v1

  integer(c_int) function pycam_hooks_name_v1(hook, buffer, length) bind(C, name='pycam_hooks_name_v1') result(status)
    integer(c_int), value, intent(in) :: hook, length
    character(kind=c_char), intent(out) :: buffer(*)
    character(len=64) :: name
    integer :: i
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    select case (hook)
{chr(10).join(f"    case ({i})" + chr(10) + f"      name = name_{i}" for i in range(1, len(table.hooks) + 1))}
    end select
    do i = 1, min(int(length) - 1, len_trim(name))
      buffer(i) = name(i:i)
    end do
    buffer(min(int(length), len_trim(name) + 1)) = c_null_char
    status = 0_c_int
  end function pycam_hooks_name_v1

  integer(c_int) function pycam_hooks_arm_v1(hook, flag) bind(C, name='pycam_hooks_arm_v1') result(status)
    ! arm (flag /= 0) or disarm one hook: the owning stage does this around its run
    integer(c_int), value, intent(in) :: hook, flag
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    armed(hook) = flag /= 0_c_int
    status = 0_c_int
  end function pycam_hooks_arm_v1

  integer(c_int) function pycam_hooks_counts_v1(hook, calls_out, paused_out) bind(C, name='pycam_hooks_counts_v1') result(status)
    integer(c_int), value, intent(in) :: hook
    integer(c_int64_t), intent(out) :: calls_out, paused_out
    calls_out = 0_c_int64_t; paused_out = 0_c_int64_t
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    calls_out = calls(hook); paused_out = paused(hook)
    status = 0_c_int
  end function pycam_hooks_counts_v1

  integer(c_int) function pycam_hooks_missed_v1(hook, missed_out) bind(C, name='pycam_hooks_missed_v1') result(status)
    ! calls that found the hook armed while no fiber was running: the original answered them
    integer(c_int), value, intent(in) :: hook
    integer(c_int64_t), intent(out) :: missed_out
    missed_out = 0_c_int64_t
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    missed_out = missed(hook)
    status = 0_c_int
  end function pycam_hooks_missed_v1

  integer(c_int) function pycam_hooks_paused_v1() bind(C, name='pycam_hooks_paused_v1') result(hook)
    ! the hook suspended on the fiber now, or 0
    hook = int(paused_hook, c_int)
  end function pycam_hooks_paused_v1

  integer(c_int) function pycam_hooks_frame_v1(count, ptrs, ndims, shapes, dtypes, intents, ncol_out) &
       bind(C, name='pycam_hooks_frame_v1') result(status)
    ! the paused call's arguments in the callee's order, where they live
    integer(c_int), value, intent(in) :: count
    type(c_ptr), intent(out) :: ptrs(count)
    integer(c_int), intent(out) :: ndims(count), dtypes(count), intents(count), ncol_out
    integer(c_int64_t), intent(out) :: shapes(max_rank, count)
    integer :: slot
    status = 1_c_int
    ncol_out = 0_c_int
    if (paused_hook == 0) return
    if (count < frame_nslots) then
      status = 3_c_int; return
    end if
    do slot = 1, count
      ptrs(slot) = c_null_ptr; ndims(slot) = 0_c_int; dtypes(slot) = 0_c_int; intents(slot) = 0_c_int
      shapes(:, slot) = 0_c_int64_t
    end do
    do slot = 1, frame_nslots
      ptrs(slot) = frame_ptrs(slot); ndims(slot) = frame_ndims(slot)
      dtypes(slot) = frame_dtypes(slot); intents(slot) = frame_intents(slot)
      shapes(:, slot) = frame_shapes(:, slot)
    end do
    ncol_out = frame_ncol
    status = 0_c_int
  end function pycam_hooks_frame_v1

  integer(c_int) function pycam_hooks_original_v1() bind(C, name='pycam_hooks_original_v1') result(status)
    ! run the original on the paused frame, from the main context
{chr(10).join(_pointer_declarations(table))}
    status = 1_c_int
    if (paused_hook == 0) return
    select case (paused_hook)
{original_cases}
    end select
    status = 0_c_int
  end function pycam_hooks_original_v1

  subroutine pycam_hooks_reset_v1() bind(C, name='pycam_hooks_reset_v1')
    armed = .false.
    paused_hook = 0
  end subroutine pycam_hooks_reset_v1

end module pycam_hooks
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    text = render(load_hooks())
    if arguments.check:
        current = HOOK_MODULE.read_text() if HOOK_MODULE.is_file() else ""
        if current != text:
            sys.stderr.write("".join(difflib.unified_diff(current.splitlines(keepends=True), text.splitlines(keepends=True),
                                                          fromfile="pycam_hooks.F90 (committed)", tofile="pycam_hooks.F90 (generated)"))[:4000])
            print(f"stale: {HOOK_MODULE.relative_to(REPO)}", file=sys.stderr)
            return 1
        print("current: pycam_hooks.F90")
        return 0
    HOOK_MODULE.write_text(text)
    print(f"wrote {HOOK_MODULE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
