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

A hook whose table entry has a ``model`` block can also be *bound* to a
TorchScript model (pycam_hooks_bind_model_v1): the hook then answers every
call by handing the named arguments to the model through FTorch, inside the
image, and writing the model's outputs back over the live columns.  No fiber,
no Python: a step with a bound model crosses the Python/Fortran boundary once.
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


def _axis(hook: Hook, spec: FunctionSpec, axis: str) -> str:
    """One extent as the hook must express it.

    A bind(C) hook serves module arrays of the contract's fixed extents.  A
    Fortran-bound hook may receive packed arrays whose leading extents are
    the callee's own integer dummies (micro_mg_tend is told ``pcols`` and
    ``pver`` for arrays of that very size), so an axis named like an integer
    dummy is that dummy, ``<axis>p`` is ``<axis>+1``, and only the rest are
    the contract's constants.
    """

    if hook.binding == "fortran":
        ints = {item.name for item in _dummies(spec) if item.rank == 0 and item.dtype == "int32" and not item.carrier}
        if axis in ints:
            return axis
        if axis.endswith("p") and axis[:-1] in ints:
            return f"{axis[:-1]}+1"
    return str(spec.dimensions[axis])


def _extents(spec: FunctionSpec, item, hook: Hook | None = None) -> str:
    if hook is None:
        return ", ".join(str(spec.dimensions[axis]) for axis in item.native_shape)
    return ", ".join(_axis(hook, spec, axis) for axis in item.native_shape)


def _declaration(hook: Hook, spec: FunctionSpec, item, *, target: bool) -> str:
    """One dummy as the hook (or the original's wrapper) declares it.

    A bind(C) hook declares interoperable scalars and assumed-size arrays; a
    Fortran-bound hook declares the callee's own kinds -- default logical,
    assumed-length character, pointer arrays -- and explicit-shape arrays with
    the contract's extents so the model branch can copy sections.
    """

    intent = ROLE_INTENT[item.role]
    if hook.binding == "c":
        shape = "(*)" if item.rank else ""
        return f"    {C_TYPES[item.dtype]}, intent({intent}){', target' if target else ''} :: {item.name}{shape}"
    if item.carrier == "logical":
        return f"    logical, intent({intent}) :: {item.name}"
    if item.carrier == "character":
        return f"    character(len=*), intent({intent}) :: {item.name}"
    if item.pointer:
        colons = ",".join(":" for _ in range(item.rank))
        return f"    {C_TYPES[item.dtype]}, pointer, intent({intent}) :: {item.name}({colons})"
    if item.rank:
        return f"    {C_TYPES[item.dtype]}, intent({intent}){', target' if target else ''} :: {item.name}({_extents(spec, item, hook)})"
    return f"    {C_TYPES[item.dtype]}, intent({intent}) :: {item.name}"


def _check_binding(hook: Hook, spec: FunctionSpec) -> None:
    dummies = _dummies(spec)
    if spec.result is not None:
        raise SystemExit(f"hook {hook.kernel}: function kernels are not hooked yet (the result needs a value slot)")
    if any(item.optional for item in dummies):
        raise SystemExit(f"hook {hook.kernel}: optional dummies are not hooked yet")
    awkward = [item.name for item in dummies if item.carrier or item.pointer]
    if hook.binding == "c" and awkward:
        raise SystemExit(f"hook {hook.kernel}: {awkward} are logical, character or pointer dummies; a bind(C) hook "
                         f"cannot carry them -- declare the hook `binding: fortran`")


def _hook_procedure(hook: Hook, spec: FunctionSpec, index: int) -> str:
    _check_binding(hook, spec)
    dummies = _dummies(spec)
    names = ", ".join(item.name for item in dummies)
    if hook.binding == "fortran":
        # a plain module procedure with the callee's own dummies: it answers with the
        # original or with a bound model, and has no frame to hand Python
        lines = [f"  subroutine hook_{hook.kernel}({names})",
                 f"    ! {spec.qualified_name}, as its redirected callers call it; {hook.redirect}, Fortran-bound"]
        lines.extend(_declaration(hook, spec, item, target=True) for item in dummies)
        lines.append(f"    calls({index}) = calls({index}) + 1_c_int64_t")
        if hook.takes_model:
            lines.append(f"    if (modeled({index})) then")
            lines.append(f"      answered({index}) = answered({index}) + 1_c_int64_t")
            lines.append(f"      call model_{hook.kernel}({names})")
            lines.append(f"      if (.not. shadow({index})) return")
            lines.append("      ! shadow: the model ran for its cost alone; the original answers")
            lines.append("    end if")
        lines.append(f"    call original_{hook.kernel}({names})")
        lines.append(f"  end subroutine hook_{hook.kernel}")
        return "\n".join(lines)
    lines = [f"  subroutine hook_{hook.kernel}({names}) bind(C, name='{hook.symbol}')",
             f"    ! {spec.qualified_name}, as its redirected callers call it; {hook.redirect}"]
    lines.extend(_declaration(hook, spec, item, target=True) for item in dummies)
    lines.append("    integer :: slot")
    lines.append(f"    calls({index}) = calls({index}) + 1_c_int64_t")
    if hook.takes_model:
        lines.append(f"    if (modeled({index})) then")
        lines.append("      ! a TorchScript model is bound here: the image answers the call itself")
        lines.append(f"      answered({index}) = answered({index}) + 1_c_int64_t")
        lines.append(f"      call model_{hook.kernel}({names})")
        lines.append(f"      if (.not. shadow({index})) return")
        lines.append("      ! shadow: the model ran for its cost alone; the original answers")
        lines.append("    end if")
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


def _model_arguments(hook: Hook, spec: FunctionSpec):
    """The model's input and output dummies, checked against the contract."""

    by_name = {item.name: item for item in _dummies(spec)}
    inputs, outputs = [], []
    for name in hook.model_inputs:
        item = by_name.get(name)
        if item is None:
            raise SystemExit(f"hook {hook.kernel}: model input {name!r} is not an argument of the contract")
        if item.carrier or item.rank > FRAME_MAX_RANK or (item.rank and item.dtype != "float64"):
            raise SystemExit(f"hook {hook.kernel}: model input {name!r} must be a numeric scalar or a real array")
        inputs.append(item)
    for name in hook.model_outputs:
        item = by_name.get(name)
        if item is None:
            raise SystemExit(f"hook {hook.kernel}: model output {name!r} is not an argument of the contract")
        if item.carrier or item.pointer or not item.rank or item.dtype != "float64" or ROLE_INTENT[item.role] == "in":
            raise SystemExit(f"hook {hook.kernel}: model output {name!r} must be a real output (or inout) array")
        outputs.append(item)
    both = {i.name for i in inputs} & {o.name for o in outputs}
    for name in sorted(both):
        if ROLE_INTENT[by_name[name].role] != "inout":
            raise SystemExit(f"hook {hook.kernel}: {name!r} is listed as model input and output but is not inout")
    return inputs, outputs


def _section(item, lead: str) -> str:
    """``name(1:n, :, :)``-style section over the live columns of a rank-N array."""

    return f"({lead}" + "".join(", :" for _ in range(item.rank - 1)) + ")"


def _model_procedure(hook: Hook, spec: FunctionSpec, index: int) -> str:
    """Answer the call with the bound TorchScript model: tensors over the arrays, forward, write back."""

    dummies = _dummies(spec)
    inputs, outputs = _model_arguments(hook, spec)
    names = ", ".join(item.name for item in dummies)
    ncol = next((item.name for item in dummies if item.name.lower() == "ncol"), None)
    plain = hook.binding == "fortran"
    first = (lambda item: item.name) if plain else (lambda item: f"{item.name}(1)")
    lines = [f"  subroutine model_{hook.kernel}({names})",
             f"    ! {spec.qualified_name} answered by the model bound at hook {index}, inside the image"]
    lines.extend(_declaration(hook, spec, item, target=True) for item in dummies)
    lines.append(f"    type(torch_tensor) :: in_t({len(inputs)}), out_t({len(outputs)})")
    for item in inputs:
        colons = ",".join(":" for _ in range(max(item.rank, 1)))
        if item.rank == 0:
            lines.append(f"    real(c_double), target :: s_{item.name}(1)")
            lines.append(f"    real(c_double), pointer, contiguous :: sp_{item.name}(:)")
        elif item.pointer:
            lines.append(f"    real(c_double), target :: l_{item.name}({_extents(spec, item, hook)})")
            lines.append(f"    real(c_double), pointer, contiguous :: v_{item.name}({colons})")
        else:
            lines.append(f"    real(c_double), pointer, contiguous :: v_{item.name}({colons})")
    for item in outputs:
        colons = ",".join(":" for _ in range(item.rank))
        lines.append(f"    real(c_double), target :: o_{item.name}({_extents(spec, item, hook)})")
        lines.append(f"    real(c_double), pointer, contiguous :: op_{item.name}({colons})")
        lines.append(f"    real(c_double), pointer, contiguous :: w_{item.name}({colons})")
    lines.append("    integer :: n")
    lines.append("    integer(c_int64_t) :: t0, t1, t2")
    lines.append("    call system_clock(t0)")
    for slot, item in enumerate(inputs, start=1):
        if item.rank == 0:
            value = item.name if item.dtype == "float64" else f"real({item.name}, c_double)"
            lines.append(f"    s_{item.name}(1) = {value}")
            lines.append(f"    sp_{item.name} => s_{item.name}")
            lines.append(f"    call torch_tensor_from_array(in_t({slot}), sp_{item.name}, torch_kCPU)")
        elif item.pointer:
            bounds = ", ".join(f"1:{_axis(hook, spec, axis)}" for axis in item.native_shape)
            lines.append(f"    if (associated({item.name})) then")
            lines.append(f"      l_{item.name} = {item.name}({bounds})")
            lines.append("    else")
            lines.append(f"      l_{item.name} = 0.0_c_double")
            lines.append("    end if")
            lines.append(f"    v_{item.name} => l_{item.name}")
            lines.append(f"    call torch_tensor_from_array(in_t({slot}), v_{item.name}, torch_kCPU)")
        else:
            lines.append(f"    call c_f_pointer(c_loc({first(item)}), v_{item.name}, (/ {_extents(spec, item, hook)} /))")
            lines.append(f"    call torch_tensor_from_array(in_t({slot}), v_{item.name}, torch_kCPU)")
    for slot, item in enumerate(outputs, start=1):
        lines.append(f"    op_{item.name} => o_{item.name}")
        lines.append(f"    call torch_tensor_from_array(out_t({slot}), op_{item.name}, torch_kCPU)")
    lines.append("    call system_clock(t1)")
    lines.append(f"    call torch_model_forward(models({index}), in_t, out_t)")
    lines.append("    call system_clock(t2)")
    lines.append(f"    forward_ticks({index}) = forward_ticks({index}) + (t2 - t1)")
    lines.append(f"    if (.not. shadow({index})) then")
    for item in outputs:
        lead = _axis(hook, spec, item.native_shape[0])
        lines.append(f"      n = min(int({ncol}), {lead})" if ncol else f"      n = {lead}")
        lines.append(f"      call c_f_pointer(c_loc({first(item)}), w_{item.name}, (/ {_extents(spec, item, hook)} /))")
        lines.append(f"      w_{item.name}{_section(item, '1:n')} = o_{item.name}{_section(item, '1:n')}")
    for item in dummies:
        if item.carrier == "character" and ROLE_INTENT[item.role] != "in":
            lines.append(f"      {item.name} = ' '")
    lines.append("    end if")
    lines.append("    call torch_delete(in_t)")
    lines.append("    call torch_delete(out_t)")
    lines.append("    call system_clock(t1)")
    lines.append(f"    model_ticks({index}) = model_ticks({index}) + (t1 - t0)")
    lines.append(f"  end subroutine model_{hook.kernel}")
    return "\n".join(lines)


def _original_procedure(hook: Hook, spec: FunctionSpec) -> str:
    """The original callee, reached through its module or its own symbol."""

    dummies = _dummies(spec)
    names = ", ".join(item.name for item in dummies)
    lines = [f"  subroutine original_{hook.kernel}({names})"]
    if hook.original_module:
        lines.append(f"    use {hook.original_module}, only: {hook.original_routine}")
    lines.extend(_declaration(hook, spec, item, target=False) for item in dummies)
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
        if not hook.pausable:
            continue
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
    framed = [len(_dummies(spec)) for hook, spec in zip(table.hooks, specs) if hook.pausable]
    max_slots = max(framed) if framed else 1
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
        if hook.takes_model:
            procedures.append(_model_procedure(hook, spec, index))
            procedures.append("")
    has_model = ", ".join(".true." if hook.takes_model else ".false." for hook in table.hooks)
    original_cases = "\n".join(_frame_original(hook, spec, index) for index, (hook, spec) in enumerate(zip(table.hooks, specs), start=1)
                               if hook.pausable)
    can_pause = ", ".join(".true." if hook.pausable else ".false." for hook in table.hooks)
    kernel_names = "\n".join(f"  character(len=*), parameter :: name_{i} = '{hook.kernel}'" for i, hook in enumerate(table.hooks, start=1))
    return f'''! Hooks: kernels reached inside compiled routines, their callers' references
! redirected at link time to these procedures.  Each counts its calls, calls the
! original when unarmed, and when armed hands Python the frame by yielding the
! stage's fiber.  Fortran never calls Python.  A hook with a model block in the
! table can be bound to a TorchScript model instead, which the image runs itself
! through FTorch: no fiber, no Python, one crossing a step.
!
! GENERATED by tools/generate_pi_cam_hooks.py from native/pi_cam/hooks.yaml and
! the kernels' function contracts.  Do not edit by hand.
module pycam_hooks
  use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t, c_double, c_ptr, c_loc, c_f_pointer, c_null_ptr, c_char, c_null_char
  use ftorch, only: torch_model, torch_tensor, torch_kCPU, torch_model_load, torch_model_forward, &
                    torch_tensor_from_array, torch_delete
  implicit none
  private
  public :: pycam_hooks_arm_v1, pycam_hooks_counts_v1, pycam_hooks_paused_v1, pycam_hooks_frame_v1, &
            pycam_hooks_original_v1, pycam_hooks_reset_v1, pycam_hooks_count_v1, pycam_hooks_name_v1, &
            pycam_hooks_bind_model_v1, pycam_hooks_unbind_model_v1, pycam_hooks_modeled_v1, &
            pycam_hooks_model_seconds_v1

  integer, parameter :: nhooks = {len(table.hooks)}
  integer(c_int), parameter :: ev_needs_kernel = 1_c_int
  integer, parameter :: max_slots = {max_slots}
  integer, parameter :: max_rank = {FRAME_MAX_RANK}
{kernel_names}

  logical, save :: armed(nhooks) = .false.
  integer(c_int64_t), save :: calls(nhooks) = 0_c_int64_t
  integer(c_int64_t), save :: paused(nhooks) = 0_c_int64_t
  integer(c_int64_t), save :: missed(nhooks) = 0_c_int64_t
  ! TorchScript models bound at hooks whose table entry has a model block
  logical, parameter :: has_model(nhooks) = (/ {has_model} /)
  ! bind(C) hooks carry a frame and can pause for Python; Fortran-bound hooks cannot
  logical, parameter :: can_pause(nhooks) = (/ {can_pause} /)
  logical, save :: modeled(nhooks) = .false.
  ! shadow: the bound model runs on every call and its answer is discarded; the original answers.
  ! The run stays bit-for-bit and the stage's extra time is the model path's cost alone.
  logical, save :: shadow(nhooks) = .false.
  integer(c_int64_t), save :: answered(nhooks) = 0_c_int64_t
  ! wall time inside the model branch (tensors, forward, write-back) and in the forward alone
  integer(c_int64_t), save :: model_ticks(nhooks) = 0_c_int64_t, forward_ticks(nhooks) = 0_c_int64_t
  type(torch_model), save :: models(nhooks)
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
    if (flag /= 0_c_int .and. modeled(hook)) then
      status = 3_c_int; return          ! a bound model and a Python replacement cannot share a hook
    end if
    if (flag /= 0_c_int .and. .not. can_pause(hook)) then
      status = 5_c_int; return          ! a Fortran-bound hook has no frame to hand Python
    end if
    armed(hook) = flag /= 0_c_int
    status = 0_c_int
  end function pycam_hooks_arm_v1

  integer(c_int) function pycam_hooks_bind_model_v1(hook, path, length, shadow_flag) &
       bind(C, name='pycam_hooks_bind_model_v1') result(status)
    ! load the TorchScript file at path (length bytes) and answer the hook's calls with it from
    ! now on; with shadow_flag /= 0 the model runs but the original keeps answering
    integer(c_int), value, intent(in) :: hook, length, shadow_flag
    character(kind=c_char), intent(in) :: path(*)
    character(len=4096) :: filename
    integer :: i
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    if (.not. has_model(hook)) then
      status = 2_c_int; return
    end if
    if (armed(hook)) then
      status = 3_c_int; return
    end if
    if (length < 1 .or. length > len(filename)) then
      status = 4_c_int; return
    end if
    filename = ' '
    do i = 1, length
      filename(i:i) = path(i)
    end do
    if (modeled(hook)) call torch_delete(models(hook))
    call torch_model_load(models(hook), filename(1:length), torch_kCPU)
    modeled(hook) = .true.
    shadow(hook) = shadow_flag /= 0_c_int
    status = 0_c_int
  end function pycam_hooks_bind_model_v1

  integer(c_int) function pycam_hooks_unbind_model_v1(hook) bind(C, name='pycam_hooks_unbind_model_v1') result(status)
    ! release the bound model: the hook answers with the original again
    integer(c_int), value, intent(in) :: hook
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    if (modeled(hook)) then
      call torch_delete(models(hook))
      modeled(hook) = .false.
      shadow(hook) = .false.
    end if
    status = 0_c_int
  end function pycam_hooks_unbind_model_v1

  integer(c_int) function pycam_hooks_model_seconds_v1(hook, model_seconds, forward_seconds) &
       bind(C, name='pycam_hooks_model_seconds_v1') result(status)
    ! wall seconds this rank spent in the hook's model branch, and in the model's forward alone
    integer(c_int), value, intent(in) :: hook
    real(c_double), intent(out) :: model_seconds, forward_seconds
    integer(c_int64_t) :: rate
    model_seconds = 0.0_c_double; forward_seconds = 0.0_c_double
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    call system_clock(count_rate=rate)
    model_seconds = real(model_ticks(hook), c_double) / real(rate, c_double)
    forward_seconds = real(forward_ticks(hook), c_double) / real(rate, c_double)
    status = 0_c_int
  end function pycam_hooks_model_seconds_v1

  integer(c_int) function pycam_hooks_modeled_v1(hook, answered_out) bind(C, name='pycam_hooks_modeled_v1') result(status)
    ! calls the bound model answered inside the image
    integer(c_int), value, intent(in) :: hook
    integer(c_int64_t), intent(out) :: answered_out
    answered_out = 0_c_int64_t
    status = 1_c_int
    if (hook < 1 .or. hook > nhooks) return
    answered_out = answered(hook)
    status = 0_c_int
  end function pycam_hooks_modeled_v1

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
    integer :: hook
    armed = .false.
    paused_hook = 0
    do hook = 1, nhooks
      if (modeled(hook)) then
        call torch_delete(models(hook))
        modeled(hook) = .false.
        shadow(hook) = .false.
      end if
    end do
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
