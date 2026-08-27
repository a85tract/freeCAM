#!/usr/bin/env python3
"""Emit the handles module a Python-driven microphysics timestep calls.

``pycam_micro_kernels`` holds the driver's arithmetic.  What is left of
``micro_mg_cam_tend`` divides in two.

Most of it Python walks statement for statement through small ``bind(C)``
entries here: the physics-state copy, the two tendency objects, the
water-tracer rates, the history writes, and views of every array the
lifted kernels read.

The packer section does not divide that way.  Lines 1764-2286 choose the
cloudy columns, build an ``MGPacker`` and an ``MGPostProc``, allocate 117
packed arrays, gather 27 inputs into them, register 89 outputs for the
scatter back, run the MG core over one or more substeps, unpack the
tendencies into ``ptend_loc``, apply them, accumulate, average, scatter,
and free everything.  It binds 241 locals and carries no meaning Python
could use: it is bookkeeping around one call.  So it is lifted **verbatim**
into this module as five procedures, with the 241 locals as chunk-local
module state, and the configuration flags its text tests passed in from
Python, which reads them off the image once.  The bodies are the pinned
text; a test holds them to it line by line.

The core call itself is one of the five.  When a model owns the core the
procedure is skipped, Python reads the packed inputs through views, and
writes the packed outputs the same way before the unpack procedure runs.

    tools/generate_pi_cam_micro_handles.py            # write the module and the view table
    tools/generate_pi_cam_micro_handles.py --check    # fail if either is stale
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/micro_mg_cam.F90"
MODULE = REPO / "native/pi_cam/support/pycam_micro_handles.F90"
VIEW_TABLE = REPO / "native/pi_cam/micro_views.yaml"

ROUTINE = (997, 3184)
DECLARATIONS = (997, 1553)

#: The verbatim procedures: (name, first, last).  Every line in a range is
#: kept; the driver's own `if`s on module flags are evaluated by the same
#: flags, passed in.
VERBATIM = (
    ("micro_pack_prelude", 1768, 2069),
    ("micro_substep_pack", 2074, 2086),
    ("micro_core", 2087, 2209),          # the core call and its error check
    ("micro_substep_unpack", 2210, 2247),
    ("micro_post_proc", 2252, 2286),
)

#: micro_mg_cam module variables the verbatim text reads.  Private to that
#: module, so Python reads each off the image (ifort emits the symbol) and
#: passes it to `pycam_micro_configure_v1` once; here they are module state.
CONFIGURATION = (
    ("micro_mg_version", "integer"), ("micro_mg_sub_version", "integer"),
    ("num_steps", "integer"), ("microp_uniform", "logical"),
    ("do_cldice", "logical"), ("do_cldliq", "logical"),
    ("ixcldliq", "integer"), ("ixcldice", "integer"),
    ("ixnumliq", "integer"), ("ixnumice", "integer"),
    ("ixrain", "integer"), ("ixsnow", "integer"),
    ("ixnumrain", "integer"), ("ixnumsnow", "integer"),
)


def _statements(lines, first, last):
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
        out.append((start, number, re.sub(r"\s+", " ", text.strip())))
        buffer, start = "", None
    return out


def declarations(lines) -> dict[str, tuple[str, str, str]]:
    """name -> (kind, attributes, dims) for every local the routine declares."""

    out = {}
    for _, _, text in _statements(lines, *DECLARATIONS):
        match = re.match(r"(real\(r8\)|integer|logical|character\(len=\d+\)|type\(\w+\))(.*?)::\s*(.*)$",
                         text, re.I)
        if not match:
            continue
        kind, attributes, names = match.groups()
        attributes = re.sub(r"intent\(\w+\)", "", attributes).strip(" ,")
        for item in re.finditer(r"(\w+)\s*(\(([^()]*(?:\([^()]*\)[^()]*)*)\))?(\s*=>?\s*[^,]+)?", names):
            name, dims, init = item.group(1), item.group(3) or "", item.group(4) or ""
            dims = dims.replace("state%psetcols", "pcols").replace("psetcols", "pcols")
            out[name.lower()] = (kind, attributes, dims, init.strip())
    return out


def used_names(lines) -> set[str]:
    names = set()
    for _, first, last in VERBATIM:
        for number in range(first, last + 1):
            for word in re.findall(r"\b([A-Za-z_]\w*)\b", lines[number - 1].split("!")[0]):
                names.add(word.lower())
    return names


#: Names the verbatim text uses that are not the routine's locals: dummies,
#: module flags passed in, or things this module imports.
NOT_LOCAL = {"state", "ptend", "pbuf", "dtime", "it", "pcols", "pver", "pverp", "top_lev",
             "use_hetfrz_classnuc", "trace_water", "wtrc_ncnst", "wtrc_indices", "size",
             "allocate", "deallocate", "nullify", "associated", "any", "call", "if", "then",
             "else", "end", "do", "select", "case", "endrun", "handle_errmsg", "subname",
             "true", "false", "r8", "p", "accum_null", "accum_mean", "fillvalue", "accum_method",
             "ls", "lq", "psetcols", "ncol", "nlev", "lchnk", "errstring", "mgpacker", "mgpostproc",
             "micro_mg_tend1_0", "micro_mg_tend1_5", "micro_mg_tend2_0",
             "micro_mg_get_cols1_0", "micro_mg_get_cols1_5", "micro_mg_get_cols2_0",
             "physics_ptend_init", "physics_ptend_sum", "physics_update", "physics_ptend_scale",
             "post_proc", "packer", "state_loc", "ptend_loc", "pckdptr", "qrain_idx", "qsnow_idx",
             "nrain_idx", "nsnow_idx", "e", "in", "out", "i", "k"}


def module_state(lines) -> list[tuple[str, str]]:
    """(declaration line, name) for every local the verbatim text needs."""

    declared = declarations(lines)
    configured = {n for n, _ in CONFIGURATION}
    rows = []
    wanted = used_names(lines) | {n for n, _ in INPUT_ALIASES}
    for name in sorted(wanted):
        if name in NOT_LOCAL or name in configured or name not in declared:
            continue
        kind, attributes, dims, init = declared[name]
        if "intent" in attributes:
            continue
        attrs = ", " + attributes if attributes else ""
        shape = f"({dims})" if dims else ""
        rows.append((f"  {kind}{attrs}, save :: {name}{shape}{(' ' + init) if init else ''}", name))
    # the four the driver holds as dummies or derived types, held here instead
    rows += [
        ("  type(physics_state), pointer, save :: state => null()", "state"),
        ("  type(physics_ptend), pointer, save :: ptend => null()", "ptend"),
        ("  type(physics_state), save :: state_loc", "state_loc"),
        ("  type(physics_ptend), save :: ptend_loc", "ptend_loc"),
        ("  type(physics_ptend), allocatable, target, save :: micro_ptend(:)", "micro_ptend"),
        ("  type(MGPacker), save :: packer", "packer"),
        ("  type(MGPostProc), save :: post_proc", "post_proc"),
        ("  real(r8), pointer, save :: pckdptr(:,:) => null()", "pckdptr"),
        ("  integer, save :: ncol = 0, nlev = 0, psetcols = 0, lchnk = 0, it = 0", "ncol"),
        ("  real(r8), save :: dtime = 0._r8", "dtime"),
        ("  logical, save :: lq(pcnst)", "lq"),
        ("  character(len=128), save :: errstring", "errstring"),
    ]
    return rows


#: The pbuf-backed pointer aliases the prelude packs (the driver's
#: `pbuf_get_field` targets), bound by Python before the prelude.  Codes are
#: their positions here; the view table repeats them for Python.
INPUT_ALIASES = (
    ("naai", 2), ("npccn", 2), ("rndst", 3), ("nacon", 3), ("relvar", 2),
    ("accre_enhan", 2), ("ast", 2), ("alst_mic", 2), ("aist_mic", 2),
    ("tnd_qsnow", 2), ("tnd_nsnow", 2), ("re_ice", 2),
    ("frzimm", 2), ("frzcnt", 2), ("frzdep", 2),
    ("rel", 2), ("rei", 2), ("dei", 2), ("des", 2), ("mu", 2), ("lambdac", 2),
    ("prain", 2), ("nevapr", 2), ("prer_evap", 2), ("rate1ord_cw2pr_st", 2),
)


#: What Python views by code: (name, rank).  The MG outputs the lifted
#: kernels read, the two local derived types, and the packed arrays a model
#: in the core's place would read and write.
def view_table(lines) -> list[tuple[str, str, int, str]]:
    declared = declarations(lines)
    rows = []
    code = 0
    # the host state, the copy, and the tendencies.  The routine reads both:
    # `state%` before the copy (1730-1734) and after the substep updated the
    # copy (2712-2713), `state_loc%` in between.  They are different arrays
    # from the substep on, so the walk must be able to name each.
    for name, expr, rank in (("state_t", "state%t", 2), ("state_q", "state%q", 3),
                             ("state_pmid", "state%pmid", 2), ("state_pdel", "state%pdel", 2),
                             ("state_loc_t", "state_loc%t", 2), ("state_loc_q", "state_loc%q", 3),
                             ("state_loc_pmid", "state_loc%pmid", 2), ("state_loc_pdel", "state_loc%pdel", 2),
                             ("ptend_loc_s", "ptend_loc%s", 2), ("ptend_loc_q", "ptend_loc%q", 3),
                             ("ptend_s", "micro_ptend(lchnk)%s", 2), ("ptend_q", "micro_ptend(lchnk)%q", 3)):
        code += 1
        rows.append((name, expr, rank, str(code)))
    # the MG outputs and other target locals the walk reads
    for name in sorted(used_names(lines)):
        if name in NOT_LOCAL or name not in declared:
            continue
        kind, attributes, dims, _ = declared[name]
        if kind.lower() != "real(r8)" or "pointer" in attributes or not dims:
            continue
        if name.startswith("packed_") or name.endswith("_dum"):
            continue
        code += 1
        rows.append((name, name, dims.count(",") + 1, str(code)))
    # the packed arrays, for a model in the core's place
    for name in sorted(used_names(lines)):
        if not name.startswith("packed_") or name not in declared:
            continue
        kind, attributes, dims, _ = declared[name]
        code += 1
        rank = dims.count(",") + 1 if dims else 1          # (:) / (:,:) / (:,:,:)
        rows.append((name, name, rank, str(code)))
    return rows


def _verbatim(lines, name, first, last) -> str:
    body = []
    for number in range(first, last + 1):
        line = lines[number - 1]
        body.append("    " + line if line.strip() else line)
    return "\n".join([f"  subroutine {name}()", "    ! micro_mg_cam.F90:{}-{}, verbatim".format(first, last),
                      "    integer :: i", ""] + body + ["", f"  end subroutine {name}"])


def render_module(source: Path | None = None) -> str:
    lines = (source or PINNED).read_text().splitlines()
    nl = "\n"
    rows = module_state(lines)
    state = nl.join(row for row, _ in rows)
    # The routine's allocatable locals were freed by its exit; as module
    # state they must be freed by hand before the prelude allocates them
    # again for the next chunk.  Every allocatable array of the state, in
    # declaration order; a deallocate of what exit would have deallocated.
    releases = nl.join(f"    if (allocated({name})) deallocate({name})" for row, name in rows
                       if ", allocatable" in row and not row.lstrip().startswith("type("))
    config = nl.join(f"  {kind}, save :: {name}" for name, kind in CONFIGURATION)
    views = view_table(lines)
    codes = nl.join(f"  integer(c_int), parameter, public :: view_{n} = {c}" for n, _, _, c in views)
    cases = []
    for name, expr, rank, code in views:
        cases.append(f"    case (view_{name})")
        if expr.startswith("packed_") or name.startswith("packed_"):
            kind, attributes, _, _ = declarations(lines)[name]
            guard = "associated" if "pointer" in attributes else "allocated"
            cases.append(f"      if (.not. {guard}({expr})) return")
        elif expr.startswith("micro_ptend") or expr.startswith("ptend_loc"):
            cases.append(f"      if (.not. allocated({expr})) return")
        elif expr.startswith("state_loc"):
            cases.append("      if (.not. state_live) return")
        elif expr.startswith("state%"):
            cases.append("      if (.not. associated(state)) return")
        cases.append(f"      call view{rank}({expr}, ptr, ndims, extents)")
    procedures = (nl + nl).join(_verbatim(lines, *v) for v in VERBATIM)
    # micro_mg_cam.F90:3186-3196: the pointer helpers add_field takes
    helpers = nl.join(("  " + lines[n - 1]) if lines[n - 1].strip() else "" for n in range(3186, 3197))
    procedures += nl + nl + helpers
    lq_build = nl.join(("    " + lines[n - 1]) if lines[n - 1].strip() else "" for n in range(1743, 1767))
    lq_build = lq_build
    binds = []
    for code, (name, rank) in enumerate(INPUT_ALIASES, start=1):
        shape = {2: "(/n1, n2/)", 3: "(/n1, n2, n3/)"}[rank]
        binds.append(f"    case ({code})")
        binds.append(f"      call c_f_pointer(ptr, {name}, {shape})")
    input_binds = nl.join(binds)
    configure_args = ", &\n       ".join(f"{n}_in" for n, _ in CONFIGURATION)
    configure_decl = nl.join(
        f"    integer(c_int), value, intent(in) :: {n}_in" for n, _ in CONFIGURATION)
    configure_body = nl.join(
        (f"    {n} = {n}_in /= 0_c_int" if kind == "logical" else f"    {n} = int({n}_in)")
        for n, kind in CONFIGURATION)
    return f'''! The calls a Python-driven microphysics timestep makes into CAM, and the
! packer section of micro_mg_cam_tend, verbatim.
!
! GENERATED by tools/generate_pi_cam_micro_handles.py.  Do not edit by hand;
! edit the generator.
!
! micro_mg_cam_tend's arithmetic is lifted into pycam_micro_kernels.  Of what
! is left, the packer section (lines 1764-2286 of the pinned source) is kept
! here as five procedures whose bodies are the pinned text, character for
! character, with the routine's 241 locals as chunk-local module state.  The
! rest -- the state copy, the tendencies, the tracer rates, history -- Python
! walks through the small bind(C) entries below.
!
! This module is an addition to the source tree.  It calls the oracle's own
! routines and replaces none of them.

module pycam_micro_handles

  use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int, c_int64_t, &
       c_loc, c_null_ptr, c_ptr
  use shr_kind_mod,     only: r8 => shr_kind_r8
  use ppgrid,           only: pcols, pver, pverp, begchunk, endchunk
  use constituents,     only: pcnst
  use physconst,        only: gravit, rair, tmelt, cpair, rh2o, rhoh2o, latvap, latice, mwh2o
  use physics_types,    only: physics_state, physics_ptend, physics_ptend_init, &
       physics_state_copy, physics_update, physics_state_dealloc, physics_ptend_sum, &
       physics_ptend_scale
  use physics_buffer,   only: physics_buffer_desc, pbuf_get_chunk
  use phys_control,     only: use_hetfrz_classnuc
  use cam_abortutils,   only: endrun
  use error_messages,   only: handle_errmsg
  use ref_pres,         only: top_lev => trop_cloud_top_lev
  use micro_mg_data,    only: MGPacker, MGPostProc, accum_null, accum_mean
  use micro_mg1_0,      only: micro_mg_tend1_0 => micro_mg_tend, micro_mg_get_cols1_0 => micro_mg_get_cols
  use micro_mg1_5,      only: micro_mg_tend1_5 => micro_mg_tend, micro_mg_get_cols1_5 => micro_mg_get_cols
  use micro_mg2_0,      only: micro_mg_tend2_0 => micro_mg_tend, micro_mg_get_cols2_0 => micro_mg_get_cols
  use water_tracer_vars, only: trace_water, wtrc_ncnst, wtrc_indices
  use water_tracers,    only: wtrc_apply_rates, wtrc_init_rates, wtrc_add_rates, wtrc_output_precip
  use water_types,      only: pwtype, iwtvap, iwtliq, iwtice, iwtstrain, iwtstsnow
  use time_manager,     only: get_nstep, get_step_size

  implicit none
  private

  public :: pycam_micro_bind_hosts, python_owns_micro, micro_ptend

  logical, save :: python_owns_micro = .false.
  logical, save :: python_owns_core = .false.
  logical, save :: state_live = .false.
  type(physics_state), pointer, save :: host_state(:) => null()
  type(physics_buffer_desc), pointer, save :: host_pbuf2d(:,:) => null()

  ! the driver's own flags, read off the image by Python and set once
{config}

  ! the packer section's locals, held per chunk call
{state}

  integer, save :: rate1_cw2pr_st_idx = 0, qrain_idx = 0, qsnow_idx = 0, nrain_idx = 0, nsnow_idx = 0

  interface p
     module procedure p1
     module procedure p2
  end interface p

{codes}

contains

  subroutine release_locals()
    ! What the routine's exit did for its allocatable locals: the packed
    ! arrays the prelude allocates are module state here and would be
    ! allocated twice without this.  Called before every chunk's state copy
    ! and after its state dealloc.
{releases}
  end subroutine release_locals

  logical function chunk_ok(lchnk_in)
    integer(c_int), intent(in) :: lchnk_in
    chunk_ok = allocated(micro_ptend) .and. lchnk_in >= begchunk .and. lchnk_in <= endchunk
  end function chunk_ok

  subroutine pycam_micro_bind_hosts(state, pbuf2d)
    type(physics_state), pointer, intent(in) :: state(:)
    type(physics_buffer_desc), pointer, intent(in) :: pbuf2d(:,:)
    host_state => state
    host_pbuf2d => pbuf2d
    if (.not. allocated(micro_ptend)) allocate(micro_ptend(begchunk:endchunk))
  end subroutine pycam_micro_bind_hosts

  subroutine view1(field, ptr, ndims, extents)
    real(r8), intent(in), target :: field(:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(field(1)); ndims = 1_c_int
    extents(1) = int(size(field, 1), c_int64_t)
  end subroutine view1

  subroutine view2(field, ptr, ndims, extents)
    real(r8), intent(in), target :: field(:,:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(field(1,1)); ndims = 2_c_int
    extents(1) = int(size(field, 1), c_int64_t)
    extents(2) = int(size(field, 2), c_int64_t)
  end subroutine view2

  subroutine view3(field, ptr, ndims, extents)
    real(r8), intent(in), target :: field(:,:,:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(field(1,1,1)); ndims = 3_c_int
    extents(1) = int(size(field, 1), c_int64_t)
    extents(2) = int(size(field, 2), c_int64_t)
    extents(3) = int(size(field, 3), c_int64_t)
  end subroutine view3

  ! ------------------------------------------------------------------ !
  ! The packer section, verbatim
  ! ------------------------------------------------------------------ !

{procedures}

  ! ------------------------------------------------------------------ !
  ! Entries
  ! ------------------------------------------------------------------ !

  integer(c_int) function pycam_micro_set_owner_v1(owns) &
       bind(C, name='pycam_micro_set_owner_v1') result(status)
    integer(c_int), value, intent(in) :: owns
    python_owns_micro = owns /= 0_c_int
    status = 0_c_int
  end function pycam_micro_set_owner_v1

  integer(c_int) function pycam_micro_set_core_owner_v1(owns_core) &
       bind(C, name='pycam_micro_set_core_owner_v1') result(status)
    ! A model in micro_mg_tend's place: the core procedure is then skipped
    ! and Python fills the packed outputs before the unpack.
    integer(c_int), value, intent(in) :: owns_core
    python_owns_core = owns_core /= 0_c_int
    status = 0_c_int
  end function pycam_micro_set_core_owner_v1

  integer(c_int) function pycam_micro_configure_v1({configure_args}) &
       bind(C, name='pycam_micro_configure_v1') result(status)
{configure_decl}
{configure_body}
    status = 0_c_int
  end function pycam_micro_configure_v1

  integer(c_int) function pycam_micro_nstep_v1() bind(C, name='pycam_micro_nstep_v1') result(status)
    status = int(get_nstep(), c_int)
  end function pycam_micro_nstep_v1

  integer(c_int) function pycam_micro_dt_v1() bind(C, name='pycam_micro_dt_v1') result(status)
    status = int(get_step_size(), c_int)
  end function pycam_micro_dt_v1

  integer(c_int) function pycam_micro_begin_v1(lchnk_in, ncol_in, dtime_in) &
       bind(C, name='pycam_micro_begin_v1') result(status)
    ! micro_mg_cam.F90:1556-1559 and 1741: the chunk's sizes and the state copy
    integer(c_int), value, intent(in) :: lchnk_in, ncol_in
    real(c_double), value, intent(in) :: dtime_in
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in)) return
    if (.not. associated(host_state)) return
    lchnk = lchnk_in
    ncol = ncol_in
    psetcols = host_state(lchnk)%psetcols
    dtime = dtime_in
    nlev = pver - top_lev + 1
    state => host_state(lchnk)
    ptend => micro_ptend(lchnk)
    call release_locals()
    if (state_live) call physics_state_dealloc(state_loc)
    call physics_state_copy(state, state_loc)
    state_live = .true.
    status = 0_c_int
  end function pycam_micro_begin_v1

  integer(c_int) function pycam_micro_ptend_init_v1(lchnk_in) &
       bind(C, name='pycam_micro_ptend_init_v1') result(status)
    ! micro_mg_cam.F90:1743-1766 verbatim: the flag build and the cldwat ptend
    integer(c_int), value, intent(in) :: lchnk_in
    integer :: i
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk) return
{lq_build}
    status = 0_c_int
  end function pycam_micro_ptend_init_v1

  integer(c_int) function pycam_micro_bind_input_v1(lchnk_in, code, ptr, n1, n2, n3) &
       bind(C, name='pycam_micro_bind_input_v1') result(status)
    ! The driver reads these from the physics buffer and packs them; Python
    ! reads the same buffer through PBuf and hands the address here, so the
    ! prelude's `packer%pack(naai)` packs the buffer's own storage.
    use, intrinsic :: iso_c_binding, only: c_f_pointer
    integer(c_int), value, intent(in) :: lchnk_in, code
    type(c_ptr), value, intent(in) :: ptr
    integer(c_int), value, intent(in) :: n1, n2, n3
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk) return
    status = 2_c_int
    select case (code)
{input_binds}
    case default
      status = 3_c_int
      return
    end select
    status = 0_c_int
  end function pycam_micro_bind_input_v1

  integer(c_int) function pycam_micro_pack_prelude_v1(lchnk_in) &
       bind(C, name='pycam_micro_pack_prelude_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk_in
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk .or. .not. state_live) return
    call micro_pack_prelude()
    status = 0_c_int
  end function pycam_micro_pack_prelude_v1

  integer(c_int) function pycam_micro_substep_pack_v1(lchnk_in, it_in) &
       bind(C, name='pycam_micro_substep_pack_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk_in, it_in
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk .or. .not. state_live) return
    it = it_in
    call micro_substep_pack()
    status = 0_c_int
  end function pycam_micro_substep_pack_v1

  integer(c_int) function pycam_micro_core_v1(lchnk_in) &
       bind(C, name='pycam_micro_core_v1') result(status)
    ! The original core.  A model in its place skips this and writes the
    ! packed outputs through the views instead.
    integer(c_int), value, intent(in) :: lchnk_in
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk .or. .not. state_live) return
    if (.not. python_owns_core) call micro_core()
    status = 0_c_int
  end function pycam_micro_core_v1

  integer(c_int) function pycam_micro_substep_unpack_v1(lchnk_in) &
       bind(C, name='pycam_micro_substep_unpack_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk_in
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk .or. .not. state_live) return
    call micro_substep_unpack()
    status = 0_c_int
  end function pycam_micro_substep_unpack_v1

  integer(c_int) function pycam_micro_post_proc_v1(lchnk_in) &
       bind(C, name='pycam_micro_post_proc_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk_in
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk .or. .not. state_live) return
    call micro_post_proc()
    status = 0_c_int
  end function pycam_micro_post_proc_v1

  integer(c_int) function pycam_micro_wtrc_apply_v1(lchnk_in, pre_rates, sed_rates, post_rates, &
       alst_mic_in, aist_mic_in) bind(C, name='pycam_micro_wtrc_apply_v1') result(status)
    ! micro_mg_cam.F90:2650-2653: the driver's exact keyword set
    integer(c_int), value, intent(in) :: lchnk_in
    real(c_double), intent(in) :: pre_rates(pcols,pver,pwtype,pwtype,pwtype)
    real(c_double), intent(in) :: sed_rates(pcols,pver,pwtype)
    real(c_double), intent(in) :: post_rates(pcols,pver,pwtype,pwtype,pwtype)
    real(c_double), intent(in) :: alst_mic_in(pcols,pver), aist_mic_in(pcols,pver)
    type(physics_buffer_desc), pointer :: pbuf(:)
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk) return
    if (.not. associated(host_pbuf2d)) return
    pbuf => pbuf_get_chunk(host_pbuf2d, lchnk)
    call wtrc_apply_rates(host_state(lchnk), micro_ptend(lchnk), pbuf, top_lev, dtime, .true., &
         pre_rates=pre_rates, sed_rates=sed_rates, &
         post_rates=post_rates, do_stprecip=.true., liqcldf=alst_mic_in, icecldf=aist_mic_in, &
         fc=wtfc, fi=wtfi, prelat=wtprelat, postlat=wtpostlat, &
         frzro=frzro, meltso=meltso)
    status = 0_c_int
  end function pycam_micro_wtrc_apply_v1

  integer(c_int) function pycam_micro_wtrc_add_sum_v1(rates, ncol_in, top_lev_in, isrc, idst, rtype, &
       a, b, c, terms) bind(C, name='pycam_micro_wtrc_add_sum_v1') result(status)
    ! The four wtrc_add_rates calls whose rate is a sum of two or three
    ! arrays (2631-2641): the sum is formed here, in Fortran, and the call is
    ! the driver's.
    real(c_double), intent(inout) :: rates(pcols,pver,pwtype,pwtype,pwtype)
    integer(c_int), value, intent(in) :: ncol_in, top_lev_in, isrc, idst, rtype, terms
    real(c_double), intent(in) :: a(pcols,pver), b(pcols,pver), c(pcols,pver)
    status = 1_c_int
    if (terms == 2_c_int) then
       call wtrc_add_rates(rates, ncol_in, top_lev_in, isrc, idst, rtype, a + b)
    else if (terms == 3_c_int) then
       call wtrc_add_rates(rates, ncol_in, top_lev_in, isrc, idst, rtype, a + b + c)
    else
       return
    end if
    status = 0_c_int
  end function pycam_micro_wtrc_add_sum_v1

  integer(c_int) function pycam_micro_output_precip_v1(lchnk_in) &
       bind(C, name='pycam_micro_output_precip_v1') result(status)
    ! micro_mg_cam.F90:3179
    integer(c_int), value, intent(in) :: lchnk_in
    type(physics_buffer_desc), pointer :: pbuf(:)
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk .or. .not. state_live) return
    if (.not. associated(host_pbuf2d)) return
    pbuf => pbuf_get_chunk(host_pbuf2d, lchnk)
    call wtrc_output_precip(state_loc, pbuf)
    status = 0_c_int
  end function pycam_micro_output_precip_v1

  integer(c_int) function pycam_micro_end_v1(lchnk_in) &
       bind(C, name='pycam_micro_end_v1') result(status)
    ! micro_mg_cam.F90:3182
    integer(c_int), value, intent(in) :: lchnk_in
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in) .or. lchnk_in /= lchnk .or. .not. state_live) return
    call physics_state_dealloc(state_loc)
    state_live = .false.
    call release_locals()
    status = 0_c_int
  end function pycam_micro_end_v1

  integer(c_int) function pycam_micro_view_v1(lchnk_in, code, ptr, ndims, extents) &
       bind(C, name='pycam_micro_view_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk_in, code
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_null_ptr; ndims = 0_c_int; extents = 0_c_int64_t
    status = 1_c_int
    if (.not. chunk_ok(lchnk_in)) return
    lchnk = lchnk_in
    status = 2_c_int
    select case (code)
{nl.join(cases)}
    case default
      status = 3_c_int
      return
    end select
    status = 0_c_int
  end function pycam_micro_view_v1

end module pycam_micro_handles
'''


def render_views(source: Path | None = None) -> str:
    import yaml

    lines = (source or PINNED).read_text().splitlines()
    rows = [{"name": n, "code": int(c), "rank": r, "expression": e} for n, e, r, c in view_table(lines)]
    inputs = [{"name": n, "code": i, "rank": r} for i, (n, r) in enumerate(INPUT_ALIASES, start=1)]
    return ("# GENERATED by tools/generate_pi_cam_micro_handles.py -- do not edit.\n"
            "# pycam_micro_view_v1 codes: the arrays a Python-driven microphysics step reads\n"
            "# or writes, by name; and pycam_micro_bind_input_v1 codes: the pbuf-backed\n"
            "# inputs Python binds before the packer prelude.\n"
            + yaml.safe_dump({"views": rows, "inputs": inputs}, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    stale = []
    for path, rendered in ((MODULE, render_module()), (VIEW_TABLE, render_views())):
        if arguments.check:
            current = path.read_text() if path.is_file() else ""
            if current != rendered:
                stale.append(path)
                sys.stderr.write("".join(difflib.unified_diff(
                    current.splitlines(keepends=True), rendered.splitlines(keepends=True),
                    fromfile=f"{path.name} (committed)", tofile=f"{path.name} (generated)"))[:3000])
        else:
            path.write_text(rendered)
            print(f"wrote {path.relative_to(REPO)}")
    if stale:
        sys.stderr.write("\nstale: " + ", ".join(str(p.relative_to(REPO)) for p in stale) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
