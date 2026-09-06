#!/usr/bin/env python3
"""Emit the segment runner for tphysbc stage 7: the original Fortran, pausable at mmacro_pcond.

The runner is the stage's own code -- physpkg.F90's stage-7 block and the
whole of macrop_driver_tend -- transcribed statement for statement into a
module that can stop at the ``call mmacro_pcond`` site and be resumed from
it by Python.  A Fortran subroutine loses its locals when it returns, so the
two routines' locals are module variables here, one chunk in flight at a
time; the derived types the glue owns (``ptend``, ``ptend_aero``) and its
hosts (the state, the tendencies, the physics buffer) are the ones the
cloud macro/microphysics handles already hold -- the same storage the
Python glue reaches by handle, reached here directly.  Nothing numerical is
new: every arithmetic statement below is the pinned source's, the order is
the source's, and the routines called are the originals.

Two boundaries pause: mmacro_pcond, inside the macrophysics driver, and
micro_mg_tend, inside the microphysics driver.  The microphysics driver is
not transcribed here a second time: pycam_micro_handles holds
micro_mg_cam_tend verbatim in pieces, with the routine's locals as module
state, and this runner calls those pieces in the source's order around the
core call.  microp_aero_run runs whole, as the source calls it.

The original ranges this transcribes are pinned by hash: if the pinned
source moves, --check fails and the module is not silently reused.

    tools/generate_pi_cam_stage7_runner.py            # write the module
    tools/generate_pi_cam_stage7_runner.py --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "native/pi_cam/support/pycam_stage7_runner.F90"
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_macrophysics.yaml"
SOURCE = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam"
# the pinned ranges transcribed below, and what they hash to
ANCHORS = {
    "physpkg.F90": (2188, 2393, "3cd2d2a3a185edb2"),
    "macrop_driver.F90": (374, 1224, "97ade69716d4d792"),
}


def range_digest(name: str, first: int, last: int) -> str:
    lines = (SOURCE / name).read_text().splitlines()[first - 1:last]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16]


def kernel_arguments() -> list[dict]:
    payload = yaml.safe_load(DESCRIPTORS.read_text())
    (kernel,) = [k for k in payload["kernels"] if k["name"] == "mmacro_pcond"]
    return kernel["arguments"]


def micro_frame_slots() -> int:
    """How many slots the paused micro_mg_tend call takes, from the micro generator's table."""

    sys.path.insert(0, str(REPO / "tools"))
    import generate_pi_cam_micro_handles as micro

    return len(micro.frame_table(micro.PINNED.read_text().splitlines()))


# What stands behind each of mmacro_pcond's arguments in the runner: the
# expression whose address the frame hands Python, in the call's order.
# (name without prefix) -> (fortran expression, rank of the frame array)
FRAME_SOURCES = {
    "lchnk": ("frame_lchnk", 0), "ncol": ("frame_ncol", 0), "dt": ("frame_dt", 0),
    "p": ("state_loc%pmid", 2), "dp": ("state_loc%pdel", 2),
    "t0": ("t_inout", 2), "qv0": ("qv_inout", 2), "ql0": ("ql_inout", 2), "qi0": ("qi_inout", 2),
    "nl0": ("nl_inout", 2), "ni0": ("ni_inout", 2),
    "a_t": ("ttend", 2), "a_qv": ("qtend", 2), "a_ql": ("lmitend", 2), "a_qi": ("itend", 2),
    "a_nl": ("nltend", 2), "a_ni": ("nitend", 2),
    "c_t": ("cc_t", 2), "c_qv": ("cc_qv", 2), "c_ql": ("cc_ql", 2), "c_qi": ("cc_qi", 2),
    "c_nl": ("cc_nl", 2), "c_ni": ("cc_ni", 2), "c_qlst": ("cc_qlst", 2),
    "d_t": ("dlf_t", 2), "d_qv": ("dlf_qv", 2), "d_ql": ("dlf_ql", 2), "d_qi": ("dlf_qi", 2),
    "d_nl": ("dlf_nl", 2), "d_ni": ("dlf_ni", 2),
    "a_cud": ("concld_old", 2), "a_cu0": ("concld", 2), "clrw_old": ("clrw_old", 2), "clri_old": ("clri_old", 2),
    "landfrac": ("cam_in(lchnk)%landfrac", 1), "snowh": ("cam_in(lchnk)%snowhland", 1),
    # the kernel's workspace pointers: the physics-buffer field where the
    # configuration registers one (the UW PBL's tke), as the original passes
    # it, and zeros where it is unassociated (UNICON's detrainment fields)
    "tke": ("tke|ws_tke", 2), "qtl_flx": ("qtl_flx|ws_qtl_flx", 2), "qti_flx": ("qti_flx|ws_qti_flx", 2),
    "cmfr_det": ("cmfr_det|ws_cmfr_det", 2), "qlr_det": ("qlr_det|ws_qlr_det", 2),
    "qir_det": ("qir_det|ws_qir_det", 2),
    "s_tendout": ("tlat", 2), "qv_tendout": ("qvlat", 2), "ql_tendout": ("qcten", 2),
    "qi_tendout": ("qiten", 2), "nl_tendout": ("ncten", 2), "ni_tendout": ("niten", 2),
    "qme": ("cmeliq", 2), "qvadj": ("qvadj", 2), "qladj": ("qladj", 2), "qiadj": ("qiadj", 2),
    "qllim": ("qllim", 2), "qilim": ("qilim", 2),
    "cld": ("cld", 2), "al_st_star": ("alst", 2), "ai_st_star": ("aist", 2),
    "ql_st_star": ("qlst", 2), "qi_st_star": ("qist", 2),
    "do_cldice": ("frame_do_cldice", 0),
}
INTENT_CODE = {"in": 0, "out": 1, "inout": 2}


def frame_cases(arguments: list[dict]) -> str:
    lines = []
    for index, argument in enumerate(arguments, start=1):
        name = argument["field"].split(".", 1)[1]
        expression, rank = FRAME_SOURCES[name]
        intent = INTENT_CODE[argument["intent"]]
        dtype = 1 if argument["dtype"] == "float64" else 2
        if "|" in expression:
            pointer, fallback = expression.split("|")
            lines.append(f"    call slot2_or({index}, {pointer}, {fallback}, {dtype}, {intent}, ptrs, ndims, shapes, dtypes, intents)")
        elif rank == 0:
            lines.append(f"    call scalar_slot({index}, c_loc({expression}), {dtype}, {intent}, ptrs, ndims, shapes, dtypes, intents)")
        elif rank == 1:
            lines.append(f"    call slot1({index}, {expression}, {dtype}, {intent}, ptrs, ndims, shapes, dtypes, intents)")
        else:
            lines.append(f"    call slot2({index}, {expression}, {dtype}, {intent}, ptrs, ndims, shapes, dtypes, intents)")
    return "\n".join(lines)


def render_module() -> str:
    arguments = kernel_arguments()
    nargs = len(arguments)
    slots = max(nargs, micro_frame_slots())
    physpkg = ANCHORS["physpkg.F90"]
    macro = ANCHORS["macrop_driver.F90"]
    return f'''! The segment runner for tphysbc stage 7: the original Fortran, pausable at mmacro_pcond.
!
! GENERATED by tools/generate_pi_cam_stage7_runner.py.  Do not edit by hand;
! edit the generator.
!
! What runs here is physpkg.F90:{physpkg[0]}-{physpkg[1]} (the MG branch of stage 7, for
! one chunk at a time) and macrop_driver.F90:{macro[0]}-{macro[1]} (macrop_driver_tend,
! whole), statement for statement, with these substitutions and no other:
!   state          -> host_state(lchnk)         (the chunk's physics_state, from cam_comp)
!   tend           -> host_tend(lchnk)          (cam_comp's phys_tend)
!   ptend          -> mm_ptend(lchnk)           (the stage's tendency object, held by the mm handles)
!   ptend_aero     -> mm_ptend_aero(lchnk)
!   cam_in%x       -> cam_in(lchnk)%x           (the adapter's surface input)
!   dtime          -> cld_macmic_ztodt          (the driver's timestep, as tphysbc passes it)
!   pbuf           -> pbuf                      (pbuf_get_chunk(host_pbuf2d, lchnk), fetched per segment)
!   the drivers' locals -> module variables of the same names
! The tphysbc locals the stage reads from earlier stages -- dlf, dlf2, wtdlf,
! cmfmc, cmfmc2, zdu, rliq -- are the pycesm_bc_* carries physpkg keeps
! (control patch 0039), reached through pycam_macro_forcing_v1 as the Python
! glue reaches them.  Every routine called is the original.
!
! The runner is a state machine: start() runs from the top of the stage and
! returns either DONE or, when mmacro_pcond is replaced, NEEDS_PYTHON_KERNEL
! with the program counter parked at the call site; frame() describes the
! call's arguments where they live; resume() continues past the call.  Python
! makes every call; Fortran never calls Python.
module pycam_stage7_runner
  use, intrinsic :: iso_c_binding, only: c_int, c_int32_t, c_int64_t, c_double, c_ptr, c_loc, &
                                         c_f_pointer, c_char, c_null_ptr, c_associated
  use shr_kind_mod,    only: r8 => shr_kind_r8
  use ppgrid,          only: pcols, pver, pverp, begchunk, endchunk
  use constituents,    only: pcnst, cnst_get_ind
  use physconst,       only: cpair, tmelt, gravit, latice, latvap
  use physics_types,   only: physics_state, physics_ptend, physics_tend, physics_ptend_init, &
                             physics_ptend_sum, physics_ptend_scale, physics_ptend_dealloc, &
                             physics_update, physics_state_copy, physics_state_dealloc
  use physics_buffer,  only: physics_buffer_desc, pbuf_get_chunk, pbuf_get_field, pbuf_get_index, &
                             pbuf_old_tim_idx
  use pycam_pi_cam_adapter, only: cam_in
  use check_energy,    only: check_energy_chng
  use cloud_fraction,  only: cldfrc, cldfrc_fice
  use cldwat2m_macro,  only: mmacro_pcond
  use macrop_driver,   only: do_cldice, do_cldliq, do_detrain
  use microp_aero,     only: microp_aero_run
  use water_tracers,   only: wtrc_mass_fixer, wtrc_init_rates, wtrc_add_rates, wtrc_apply_rates
  use water_tracer_vars, only: trace_water, wtrc_detrain_in_macrop, wtrc_nwset, wtrc_iatype, &
                               wtrc_indices, wtrc_ncnst
  use water_types,     only: pwtype, iwtvap, iwtliq, iwtice
  use ref_pres,        only: top_lev => trop_cloud_top_lev
  use cam_history,     only: outfld
  use time_manager,    only: get_nstep, get_step_size
  use perf_mod,        only: t_startf, t_stopf
  use cam_abortutils,  only: endrun
  use phys_control,    only: phys_getopts
  use convect_shallow, only: convect_shallow_use_shfrc
  use subcol_utils,    only: is_subcol_on
  use pycam_mm_handles, only: mm_ptend, mm_ptend_aero, host_state, host_tend, host_pbuf2d
  use pycam_micro_handles, only: micro_runner_bind, micro_run_head, micro_runner_end, &
                                 micro_runner_ready, micro_substep, micro_num_steps, &
                                 micro_pack_prelude, micro_substep_pack, micro_core, &
                                 micro_substep_unpack, micro_post_proc, micro_tail, micro_core_frame
  implicit none
  private

  ! events and program counters
  integer(c_int), parameter :: ev_done = 0_c_int, ev_needs_kernel = 1_c_int, ev_error = 2_c_int
  integer, parameter :: pc_idle = 0, pc_chunk_begin = 1, pc_substep = 2, pc_at_pcond = 3, &
                        pc_after_pcond = 4, pc_chunk_end = 5, pc_micro_substep = 6, &
                        pc_at_mg = 7, pc_after_mg = 8, pc_micro_post = 9
  integer(c_int), parameter :: kernel_mmacro_pcond = 1_c_int, kernel_micro_mg_tend = 2_c_int
  integer, parameter :: nkernels = 2
  integer, parameter :: frame_slots = {slots}
  integer, parameter :: context_id = 1

  ! the context: one per rank, one stage
  logical, save :: created = .false.
  integer, save :: pc = pc_idle
  integer, save :: lchnk = 0, macmic_it = 0, ncol = 0, nstep = 0, micro_it = 0
  integer(c_int), save :: token = 0_c_int, call_index = 0_c_int
  logical, save :: replace_pcond = .false., replace_mg = .false.
  character(len=256), save :: last_error = ' '

  ! the configuration this transcription is admitted for, read once
  character(len=16), save :: microp_scheme = ' ', macrop_scheme = ' '
  integer, save :: cld_macmic_num_steps = 0
  logical, save :: micro_do_icesupersat = .false., use_subcol_microp = .false., use_shfrc = .false.
  logical, parameter :: cu_det_st = .false.   ! macrop_driver.F90: a parameter of the module
  integer, save :: ixcldliq = 0, ixcldice = 0, ixnumliq = 0, ixnumice = 0
  integer, save :: prec_str_idx = 0, snow_str_idx = 0, prec_sed_idx = 0, snow_sed_idx = 0, &
                   prec_pcw_idx = 0, snow_pcw_idx = 0
  integer, save :: qcwat_idx = 0, tcwat_idx = 0, lcwat_idx = 0, iccwat_idx = 0, nlwat_idx = 0, &
                   niwat_idx = 0, cc_t_idx = 0, cc_qv_idx = 0, cc_ql_idx = 0, cc_qi_idx = 0, &
                   cc_nl_idx = 0, cc_ni_idx = 0, cc_qlst_idx = 0, cld_idx = 0, concld_idx = 0, &
                   ast_idx = 0, aist_idx = 0, alst_idx = 0, qist_idx = 0, qlst_idx = 0, &
                   cmeliq_idx = 0, fice_idx = 0, shfrc_idx = 0
  integer, save :: tke_idx = -1, qtl_flx_idx = -1, qti_flx_idx = -1, cmfr_det_idx = -1, &
                   qlr_det_idx = -1, qir_det_idx = -1

  ! tphysbc's locals for stage 7 (physpkg.F90), for the chunk in flight
  real(r8), save :: ztodt = 0._r8, cld_macmic_ztodt = 0._r8
  real(r8), save :: zero(pcols), flx_cnd(pcols), flx_heat(pcols)
  real(r8), save, target :: det_s(pcols), det_ice(pcols)
  real(r8), save :: prec_sed_macmic(pcols), snow_sed_macmic(pcols), prec_pcw_macmic(pcols), &
                    snow_pcw_macmic(pcols)
  real(r8), pointer, save :: prec_str(:) => null(), snow_str(:) => null(), prec_sed(:) => null(), &
                             snow_sed(:) => null(), prec_pcw(:) => null(), snow_pcw(:) => null()
  real(r8), pointer, save :: dlf(:,:) => null(), dlf2(:,:) => null(), cmfmc(:,:) => null(), &
                             cmfmc2(:,:) => null(), zdu(:,:) => null(), rliq(:) => null(), &
                             wtdlf(:,:,:) => null()
  type(physics_buffer_desc), pointer, save :: pbuf(:) => null()

  ! macrop_driver_tend's locals (macrop_driver.F90), for the chunk in flight
  type(physics_state), save, target :: state_loc
  type(physics_ptend), save :: ptend_loc
  integer, save :: i, k, m, itim_old
  real(r8), pointer, save :: qcwat(:,:) => null(), tcwat(:,:) => null(), lcwat(:,:) => null(), &
       iccwat(:,:) => null(), nlwat(:,:) => null(), niwat(:,:) => null(), cc_t(:,:) => null(), &
       cc_qv(:,:) => null(), cc_ql(:,:) => null(), cc_qi(:,:) => null(), cc_nl(:,:) => null(), &
       cc_ni(:,:) => null(), cc_qlst(:,:) => null(), cld(:,:) => null(), ast(:,:) => null(), &
       aist(:,:) => null(), alst(:,:) => null(), qist(:,:) => null(), qlst(:,:) => null(), &
       concld(:,:) => null(), shfrc(:,:) => null(), cmeliq(:,:) => null(), fice_ql(:,:) => null(), &
       tke(:,:) => null(), qtl_flx(:,:) => null(), qti_flx(:,:) => null(), cmfr_det(:,:) => null(), &
       qlr_det(:,:) => null(), qir_det(:,:) => null()
  real(r8), save, target :: shfrc_zero(pcols,pver)
  real(r8), save :: cldst(pcols,pver), rhcloud(pcols,pver), clc(pcols), rhu00(pcols,pver), &
                    icecldf(pcols,pver), liqcldf(pcols,pver), relhum(pcols,pver)
  real(r8), save :: rdtime, dum1
  real(r8), save, target :: qtend(pcols,pver), ttend(pcols,pver), ltend(pcols,pver)
  real(r8), save :: fice(pcols,pver), fsnow(pcols,pver)
  real(r8), save :: dpdlfliq(pcols,pver), dpdlfice(pcols,pver), shdlfliq(pcols,pver), &
                    shdlfice(pcols,pver), dpdlft(pcols,pver), shdlft(pcols,pver)
  real(r8), save :: qc(pcols,pver), qi(pcols,pver), nc(pcols,pver), ni(pcols,pver)
  logical, save :: lq(pcnst)
  real(r8), save, target :: tlat(pcols,pver), qvlat(pcols,pver), qcten(pcols,pver), &
                            qiten(pcols,pver), ncten(pcols,pver), niten(pcols,pver)
  real(r8), save, target :: qvadj(pcols,pver), qladj(pcols,pver), qiadj(pcols,pver), &
                            qllim(pcols,pver), qilim(pcols,pver)
  real(r8), save, target :: itend(pcols,pver), lmitend(pcols,pver), zeros(pcols,pver), &
                            t_inout(pcols,pver), qv_inout(pcols,pver), ql_inout(pcols,pver), &
                            qi_inout(pcols,pver), concld_old(pcols,pver), clrw_old(pcols,pver), &
                            clri_old(pcols,pver), nl_inout(pcols,pver), ni_inout(pcols,pver), &
                            nltend(pcols,pver), nitend(pcols,pver)
  real(r8), save, target :: dlf_t(pcols,pver), dlf_qv(pcols,pver), dlf_ql(pcols,pver), &
                            dlf_qi(pcols,pver), dlf_nl(pcols,pver), dlf_ni(pcols,pver)
  real(r8), save :: mr_lsliq(pcols,pver), mr_lsice(pcols,pver), mr_ccliq(pcols,pver), &
                    mr_ccice(pcols,pver), cldsice(pcols,pver)
  real(r8), save :: process_rates(pcols,pver,pwtype,pwtype,pwtype)
  real(r8), save :: pqctn(pcols,pver), nqctn(pcols,pver), pqitn(pcols,pver), nqitn(pcols,pver)
  ! the paused kernel's frame: the scalars by value, and zeros standing in
  ! for whichever workspace pointers are unassociated in this configuration
  integer(c_int32_t), save, target :: frame_lchnk = 0, frame_ncol = 0, frame_do_cldice = 0
  real(c_double), save, target :: frame_dt = 0._c_double
  real(r8), save, target :: ws_tke(pcols,pverp), ws_qtl_flx(pcols,pverp), ws_qti_flx(pcols,pverp), &
                            ws_cmfr_det(pcols,pver), ws_qlr_det(pcols,pver), ws_qir_det(pcols,pver)

  interface
    integer(c_int) function pycam_macro_forcing_v1(lchnk, code, ptr, ndims, extents) &
         bind(C, name='pycam_macro_forcing_v1')
      import :: c_int, c_int64_t, c_ptr
      integer(c_int), value, intent(in) :: lchnk, code
      type(c_ptr), intent(out) :: ptr
      integer(c_int), intent(out) :: ndims
      integer(c_int64_t), intent(out) :: extents(4)
    end function pycam_macro_forcing_v1
  end interface

contains

  ! ------------------------------------------------------------------ !
  ! The ABI Python drives
  ! ------------------------------------------------------------------ !

  integer(c_int) function pycam_stage7_create_v1(context) &
       bind(C, name='pycam_stage7_create_v1') result(status)
    ! Read the configuration once and refuse anything this transcription
    ! was not written for: it is physpkg's MG branch without CLUBB,
    ! sub-columns or the ice-supersaturation adjustment, with water tracers.
    integer(c_int), intent(out) :: context
    integer :: istat
    context = 0_c_int
    status = 1_c_int
    if (created) then
      last_error = 'stage 7 context already exists'
      return
    end if
    if (.not. associated(host_state) .or. .not. associated(host_tend) &
        .or. .not. associated(host_pbuf2d) .or. .not. associated(cam_in)) then
      last_error = 'stage 7 hosts are not bound (call pycam_mm_bind_hosts_v1 first)'
      return
    end if
    call phys_getopts(microp_scheme_out=microp_scheme, macrop_scheme_out=macrop_scheme, &
                      cld_macmic_num_steps_out=cld_macmic_num_steps, &
                      micro_do_icesupersat_out=micro_do_icesupersat, &
                      use_subcol_microp_out=use_subcol_microp)
    status = 2_c_int
    if (trim(microp_scheme) /= 'MG') then
      last_error = 'stage 7 runner is written for microp_scheme MG'; return
    end if
    if (trim(macrop_scheme) == 'CLUBB_SGS') then
      last_error = 'stage 7 runner is written for the Park macrophysics, not CLUBB'; return
    end if
    if (micro_do_icesupersat) then
      last_error = 'stage 7 runner is written without micro_do_icesupersat'; return
    end if
    if (use_subcol_microp .or. is_subcol_on()) then
      last_error = 'stage 7 runner is written without sub-columns'; return
    end if
    if (cld_macmic_num_steps < 1) then
      last_error = 'cld_macmic_num_steps must be positive'; return
    end if
    if (.not. micro_runner_ready()) then
      last_error = 'the microphysics pieces are not configured (pycam_micro_configure_v1 first)'; return
    end if
    ! macrop_driver_init's indices, by the same names
    call cnst_get_ind('CLDLIQ', ixcldliq)
    call cnst_get_ind('CLDICE', ixcldice)
    call cnst_get_ind('NUMLIQ', ixnumliq)
    call cnst_get_ind('NUMICE', ixnumice)
    prec_str_idx = pbuf_get_index('PREC_STR')
    snow_str_idx = pbuf_get_index('SNOW_STR')
    prec_sed_idx = pbuf_get_index('PREC_SED')
    snow_sed_idx = pbuf_get_index('SNOW_SED')
    prec_pcw_idx = pbuf_get_index('PREC_PCW')
    snow_pcw_idx = pbuf_get_index('SNOW_PCW')
    ast_idx    = pbuf_get_index('AST')
    aist_idx   = pbuf_get_index('AIST')
    alst_idx   = pbuf_get_index('ALST')
    qist_idx   = pbuf_get_index('QIST')
    qlst_idx   = pbuf_get_index('QLST')
    cld_idx    = pbuf_get_index('CLD')
    concld_idx = pbuf_get_index('CONCLD')
    qcwat_idx  = pbuf_get_index('QCWAT')
    lcwat_idx  = pbuf_get_index('LCWAT')
    iccwat_idx = pbuf_get_index('ICCWAT')
    nlwat_idx  = pbuf_get_index('NLWAT')
    niwat_idx  = pbuf_get_index('NIWAT')
    tcwat_idx  = pbuf_get_index('TCWAT')
    fice_idx   = pbuf_get_index('FICE')
    cmeliq_idx = pbuf_get_index('CMELIQ')
    cc_t_idx    = pbuf_get_index('CC_T')
    cc_qv_idx   = pbuf_get_index('CC_qv')
    cc_ql_idx   = pbuf_get_index('CC_ql')
    cc_qi_idx   = pbuf_get_index('CC_qi')
    cc_nl_idx   = pbuf_get_index('CC_nl')
    cc_ni_idx   = pbuf_get_index('CC_ni')
    cc_qlst_idx = pbuf_get_index('CC_qlst')
    use_shfrc = convect_shallow_use_shfrc()
    if (use_shfrc) shfrc_idx = pbuf_get_index('shfrc')
    ! the optional workspace fields: -1 when the configuration has none, as
    ! macrop_driver_init leaves them
    tke_idx      = pbuf_get_index('tke', istat)
    qtl_flx_idx  = pbuf_get_index('qtl_flx', istat)
    qti_flx_idx  = pbuf_get_index('qti_flx', istat)
    cmfr_det_idx = pbuf_get_index('cmfr_det', istat)
    qlr_det_idx  = pbuf_get_index('qlr_det', istat)
    qir_det_idx  = pbuf_get_index('qir_det', istat)
    shfrc_zero = 0._r8
    ws_tke = 0._r8; ws_qtl_flx = 0._r8; ws_qti_flx = 0._r8
    ws_cmfr_det = 0._r8; ws_qlr_det = 0._r8; ws_qir_det = 0._r8
    zero = 0._r8
    created = .true.
    pc = pc_idle
    context = int(context_id, c_int)
    status = 0_c_int
  end function pycam_stage7_create_v1

  integer(c_int) function pycam_stage7_start_v1(context, count, mask, event) &
       bind(C, name='pycam_stage7_start_v1') result(status)
    ! Run stage 7 from its top for every chunk of this rank; mask(1) says
    ! whether mmacro_pcond is Python's, mask(2) whether micro_mg_tend is.
    integer(c_int), value, intent(in) :: context, count
    integer(c_int), intent(in) :: mask(count)
    integer(c_int), intent(out) :: event
    event = ev_error
    status = 1_c_int
    if (.not. created .or. context /= context_id) then
      last_error = 'no stage 7 context'; return
    end if
    if (pc /= pc_idle) then
      last_error = 'stage 7 is not idle; resume or reset it first'; status = 2_c_int; return
    end if
    if (count < nkernels) then
      last_error = 'replacement mask is too short'; status = 3_c_int; return
    end if
    replace_pcond = mask(kernel_mmacro_pcond) /= 0_c_int
    replace_mg = mask(kernel_micro_mg_tend) /= 0_c_int
    ! tphysbc's per-call values
    ztodt = get_step_size()
    nstep = get_nstep()
    zero = 0._r8
    call_index = 0_c_int
    lchnk = begchunk
    pc = pc_chunk_begin
    call advance(event)
    status = 0_c_int
  end function pycam_stage7_start_v1

  integer(c_int) function pycam_stage7_resume_v1(context, kernel, token_in, event) &
       bind(C, name='pycam_stage7_resume_v1') result(status)
    integer(c_int), value, intent(in) :: context, kernel, token_in
    integer(c_int), intent(out) :: event
    event = ev_error
    status = 1_c_int
    if (.not. created .or. context /= context_id) then
      last_error = 'no stage 7 context'; return
    end if
    if (pc /= pc_at_pcond .and. pc /= pc_at_mg) then
      last_error = 'stage 7 is not paused'; status = 2_c_int; return
    end if
    if (pc == pc_at_pcond .and. kernel /= kernel_mmacro_pcond) then
      last_error = 'stage 7 is paused on mmacro_pcond, not on the kernel resumed'; status = 3_c_int; return
    end if
    if (pc == pc_at_mg .and. kernel /= kernel_micro_mg_tend) then
      last_error = 'stage 7 is paused on micro_mg_tend, not on the kernel resumed'; status = 3_c_int; return
    end if
    if (token_in /= token) then
      last_error = 'stale resume: the frame token does not match the pause'; status = 4_c_int; return
    end if
    token = token + 1_c_int
    if (pc == pc_at_pcond) then
      pc = pc_after_pcond
    else
      pc = pc_after_mg
    end if
    call advance(event)
    status = 0_c_int
  end function pycam_stage7_resume_v1

  integer(c_int) function pycam_stage7_frame_v1(context, kernel, index_out, lchnk_out, ncol_out, &
       substep_out, token_out, count, ptrs, ndims, shapes, dtypes, intents) &
       bind(C, name='pycam_stage7_frame_v1') result(status)
    ! The paused call's arguments, where they live: one slot per argument in
    ! the call's order, with the address, rank, extents, dtype code (1 double,
    ! 2 int32) and intent code (0 in, 1 out, 2 inout).
    integer(c_int), value, intent(in) :: context, count
    integer(c_int), intent(out) :: kernel, index_out, lchnk_out, ncol_out, substep_out, token_out
    type(c_ptr), intent(out) :: ptrs(count)
    integer(c_int), intent(out) :: ndims(count), dtypes(count), intents(count)
    integer(c_int64_t), intent(out) :: shapes(5, count)
    kernel = 0_c_int; index_out = 0_c_int; lchnk_out = 0_c_int; ncol_out = 0_c_int
    substep_out = 0_c_int; token_out = 0_c_int
    status = 1_c_int
    if (.not. created .or. context /= context_id) then
      last_error = 'no stage 7 context'; return
    end if
    if (pc /= pc_at_pcond .and. pc /= pc_at_mg) then
      last_error = 'stage 7 is not paused; there is no frame'; status = 2_c_int; return
    end if
    if (count < frame_slots) then
      last_error = 'frame table is too short'; status = 3_c_int; return
    end if
    if (pc == pc_at_mg) then
      ! the packed columns of this micro substep, where pycam_micro_handles holds them
      call micro_core_frame(ptrs, ndims, shapes, dtypes, intents, ncol_out)
      kernel = kernel_micro_mg_tend
      index_out = call_index
      lchnk_out = int(lchnk, c_int)
      substep_out = int((macmic_it - 1) * micro_num_steps() + micro_it, c_int)
      token_out = token
      status = 0_c_int
      return
    end if
    frame_lchnk = int(lchnk, c_int32_t)
    frame_ncol = int(ncol, c_int32_t)
    frame_dt = real(cld_macmic_ztodt, c_double)
    frame_do_cldice = merge(1_c_int32_t, 0_c_int32_t, do_cldice)
{frame_cases(arguments)}
    kernel = kernel_mmacro_pcond
    index_out = call_index
    lchnk_out = int(lchnk, c_int)
    ncol_out = int(ncol, c_int)
    substep_out = int(macmic_it, c_int)
    token_out = token
    status = 0_c_int
  end function pycam_stage7_frame_v1

  integer(c_int) function pycam_stage7_error_v1(context, buffer, length) &
       bind(C, name='pycam_stage7_error_v1') result(status)
    integer(c_int), value, intent(in) :: context, length
    character(kind=c_char), intent(out) :: buffer(length)
    integer :: n
    n = min(length - 1, len_trim(last_error))
    do i = 1, n
      buffer(i) = last_error(i:i)
    end do
    buffer(n + 1) = c_char_'\\0'
    status = 0_c_int
    if (context /= context_id) status = 1_c_int
  end function pycam_stage7_error_v1

  integer(c_int) function pycam_stage7_reset_v1(context) &
       bind(C, name='pycam_stage7_reset_v1') result(status)
    ! Back to idle after a failure mid-stage.  The Fortran already run stays
    ! run: the caller has been told the stage is tainted.
    integer(c_int), value, intent(in) :: context
    status = 1_c_int
    if (.not. created .or. context /= context_id) return
    pc = pc_idle
    status = 0_c_int
  end function pycam_stage7_reset_v1

  integer(c_int) function pycam_stage7_destroy_v1(context) &
       bind(C, name='pycam_stage7_destroy_v1') result(status)
    integer(c_int), value, intent(in) :: context
    status = 1_c_int
    if (.not. created .or. context /= context_id) return
    created = .false.
    pc = pc_idle
    status = 0_c_int
  end function pycam_stage7_destroy_v1

  ! ------------------------------------------------------------------ !
  ! The state machine
  ! ------------------------------------------------------------------ !

  subroutine advance(event)
    integer(c_int), intent(out) :: event
    do
      select case (pc)
      case (pc_chunk_begin)
        if (lchnk > endchunk) then
          pc = pc_idle
          event = ev_done
          return
        end if
        call chunk_begin()
        macmic_it = 1
        pc = pc_substep
      case (pc_substep)
        if (macmic_it > cld_macmic_num_steps) then
          pc = pc_chunk_end
          cycle
        end if
        call macro_before_pcond()
        if (replace_pcond) then
          token = token + 1_c_int
          pc = pc_at_pcond
          event = ev_needs_kernel
          return
        end if
        call original_pcond()
        pc = pc_after_pcond
      case (pc_at_pcond)
        last_error = 'stage 7 is paused; only resume continues it'
        event = ev_error
        return
      case (pc_after_pcond)
        call_index = call_index + 1_c_int
        call macro_after_pcond()
        call substep_tail_to_micro()
        micro_it = 1
        pc = pc_micro_substep
      case (pc_micro_substep)
        ! micro_mg_cam.F90:2071, the substep loop, one iteration a visit
        if (micro_it > micro_num_steps()) then
          pc = pc_micro_post
          cycle
        end if
        call micro_substep(micro_it)
        call micro_substep_pack()
        if (replace_mg) then
          token = token + 1_c_int
          pc = pc_at_mg
          event = ev_needs_kernel
          return
        end if
        call micro_core()
        pc = pc_after_mg
      case (pc_at_mg)
        last_error = 'stage 7 is paused; only resume continues it'
        event = ev_error
        return
      case (pc_after_mg)
        call_index = call_index + 1_c_int
        call micro_substep_unpack()
        micro_it = micro_it + 1
        pc = pc_micro_substep
      case (pc_micro_post)
        call micro_post_proc()
        call micro_tail()
        call micro_runner_end()
        call substep_tail_after_micro()
        macmic_it = macmic_it + 1
        pc = pc_substep
      case (pc_chunk_end)
        call chunk_end()
        lchnk = lchnk + 1
        pc = pc_chunk_begin
      case default
        last_error = 'stage 7 program counter is corrupt'
        event = ev_error
        return
      end select
    end do
  end subroutine advance

  subroutine fetch_forcing(code, ptr2)
    ! One of tphysbc's carries for this chunk, as pycam_macro_forcing_v1 hands it
    integer, intent(in) :: code
    real(r8), pointer, intent(out) :: ptr2(:,:)
    type(c_ptr) :: address
    integer(c_int) :: ndims, status
    integer(c_int64_t) :: extents(4)
    status = pycam_macro_forcing_v1(int(lchnk, c_int), int(code, c_int), address, ndims, extents)
    if (status /= 0_c_int .or. ndims /= 2_c_int) call endrun('pycam_stage7_runner: forcing view refused')
    call c_f_pointer(address, ptr2, (/ int(extents(1)), int(extents(2)) /))
  end subroutine fetch_forcing

  subroutine chunk_begin()
    ! physpkg.F90:{physpkg[0]}-2211, the top of the MG branch, for this chunk: the
    ! buffer and the carries the stage reads, then the substep setup.
    type(c_ptr) :: address
    integer(c_int) :: ndims, status
    integer(c_int64_t) :: extents(4)
    ncol = host_state(lchnk)%ncol
    pbuf => pbuf_get_chunk(host_pbuf2d, lchnk)
    call pbuf_get_field(pbuf, prec_str_idx, prec_str)
    call pbuf_get_field(pbuf, snow_str_idx, snow_str)
    call pbuf_get_field(pbuf, prec_sed_idx, prec_sed)
    call pbuf_get_field(pbuf, snow_sed_idx, snow_sed)
    call pbuf_get_field(pbuf, prec_pcw_idx, prec_pcw)
    call pbuf_get_field(pbuf, snow_pcw_idx, snow_pcw)
    call fetch_forcing(1, zdu)
    call fetch_forcing(2, cmfmc)
    call fetch_forcing(3, cmfmc2)
    call fetch_forcing(4, dlf)
    call fetch_forcing(5, dlf2)
    status = pycam_macro_forcing_v1(int(lchnk, c_int), 6_c_int, address, ndims, extents)
    if (status /= 0_c_int .or. ndims /= 1_c_int) call endrun('pycam_stage7_runner: rliq view refused')
    call c_f_pointer(address, rliq, (/ int(extents(1)) /))
    status = pycam_macro_forcing_v1(int(lchnk, c_int), 7_c_int, address, ndims, extents)
    if (status /= 0_c_int .or. ndims /= 3_c_int) call endrun('pycam_stage7_runner: wtdlf view refused')
    call c_f_pointer(address, wtdlf, (/ int(extents(1)), int(extents(2)), int(extents(3)) /))
    ! Start co-substepping of macrophysics and microphysics
    cld_macmic_ztodt = ztodt/cld_macmic_num_steps
    ! Clear precip fields that should accumulate.
    prec_sed_macmic = 0._r8
    snow_sed_macmic = 0._r8
    prec_pcw_macmic = 0._r8
    snow_pcw_macmic = 0._r8
  end subroutine chunk_begin

  subroutine macro_before_pcond()
    ! macrop_driver.F90:{macro[0]}-1050: macrop_driver_tend from its top to the
    ! mmacro_pcond call, with tphysbc's macrop_tend timer around it as the
    ! glue has it (physpkg.F90:2239).  micro_do_icesupersat is false here
    ! (refused at create), so its two blocks are omitted.
    call t_startf('macrop_tend')
    lchnk = host_state(lchnk)%lchnk
    ncol  = host_state(lchnk)%ncol
    call physics_state_copy(host_state(lchnk), state_loc)            ! Copy state to local state_loc.
    ! Associate pointers with physics buffer fields
    itim_old = pbuf_old_tim_idx()
    call pbuf_get_field(pbuf, qcwat_idx,   qcwat,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, tcwat_idx,   tcwat,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, lcwat_idx,   lcwat,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, iccwat_idx,  iccwat,  start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, nlwat_idx,   nlwat,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, niwat_idx,   niwat,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cc_t_idx,    cc_t,    start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cc_qv_idx,   cc_qv,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cc_ql_idx,   cc_ql,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cc_qi_idx,   cc_qi,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cc_nl_idx,   cc_nl,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cc_ni_idx,   cc_ni,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cc_qlst_idx, cc_qlst, start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cld_idx,     cld,    start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, concld_idx,  concld, start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, ast_idx,     ast,    start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, aist_idx,    aist,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, alst_idx,    alst,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, qist_idx,    qist,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, qlst_idx,    qlst,   start=(/1,1,itim_old/), kount=(/pcols,pver,1/) )
    call pbuf_get_field(pbuf, cmeliq_idx,  cmeliq)
    ! For purposes of convective ql.
    call pbuf_get_field(pbuf, fice_idx,     fice_ql )
    ! Initialize convective detrainment tendency
    dlf_t(:,:)  = 0._r8
    dlf_qv(:,:) = 0._r8
    dlf_ql(:,:) = 0._r8
    dlf_qi(:,:) = 0._r8
    dlf_nl(:,:) = 0._r8
    dlf_ni(:,:) = 0._r8
    ! ----------------------------------------------------------------------------- !
    ! Detrainment of convective condensate into the environment or stratiform cloud !
    ! ----------------------------------------------------------------------------- !
    lq(:)        = .FALSE.
    lq(ixcldliq) = .TRUE.
    lq(ixcldice) = .TRUE.
    lq(ixnumliq) = .TRUE.
    lq(ixnumice) = .TRUE.
    !allow water tracers to change:
    if ((trace_water) .and. (wtrc_detrain_in_macrop)) then
      do m=1,wtrc_nwset
        lq(wtrc_iatype(m,iwtliq)) = .TRUE.
        lq(wtrc_iatype(m,iwtice)) = .TRUE.
      end do
    end if
    call physics_ptend_init(ptend_loc, host_state(lchnk)%psetcols, 'pcwdetrain', ls=.true., lq=lq)
    det_s(:)   = 0._r8
    det_ice(:) = 0._r8
    dpdlfliq = 0._r8
    dpdlfice = 0._r8
    shdlfliq = 0._r8
    shdlfice = 0._r8
    dpdlft   = 0._r8
    shdlft   = 0._r8
    do k = top_lev, pver
    do i = 1, state_loc%ncol
       if( state_loc%t(i,k) > 268.15_r8 ) then
           dum1 = 0.0_r8
       elseif( state_loc%t(i,k) < 238.15_r8 ) then
           dum1 = 1.0_r8
       else
           dum1 = ( 268.15_r8 - state_loc%t(i,k) ) / 30._r8
       endif
      if (do_detrain) then
       ptend_loc%q(i,k,ixcldliq) = dlf(i,k) * ( 1._r8 - dum1 )
       ptend_loc%q(i,k,ixcldice) = dlf(i,k) * dum1
       ptend_loc%q(i,k,ixnumliq) = 3._r8 * ( max(0._r8, ( dlf(i,k) - dlf2(i,k) )) * ( 1._r8 - dum1 ) ) / &
            (4._r8*3.14_r8* 8.e-6_r8**3*997._r8) + & ! Deep    Convection
            3._r8 * (                         dlf2(i,k)    * ( 1._r8 - dum1 ) ) / &
            (4._r8*3.14_r8*10.e-6_r8**3*997._r8)     ! Shallow Convection
       ptend_loc%q(i,k,ixnumice) = 3._r8 * ( max(0._r8, ( dlf(i,k) - dlf2(i,k) )) *  dum1 ) / &
            (4._r8*3.14_r8*25.e-6_r8**3*500._r8) + & ! Deep    Convection
            3._r8 * (                         dlf2(i,k)    *  dum1 ) / &
            (4._r8*3.14_r8*50.e-6_r8**3*500._r8)     ! Shallow Convection
       ptend_loc%s(i,k)          = dlf(i,k) * dum1 * latice
      else
         ptend_loc%q(i,k,ixcldliq) = 0._r8
         ptend_loc%q(i,k,ixcldice) = 0._r8
         ptend_loc%q(i,k,ixnumliq) = 0._r8
         ptend_loc%q(i,k,ixnumice) = 0._r8
         ptend_loc%s(i,k)          = 0._r8
      end if
       det_s(i)                  = det_s(i) + ptend_loc%s(i,k)*state_loc%pdel(i,k)/gravit
       det_ice(i)                = det_ice(i) - ptend_loc%q(i,k,ixcldice)*state_loc%pdel(i,k)/gravit
       if ((trace_water) .and. (wtrc_detrain_in_macrop)) then
         do m=1,wtrc_nwset
           ptend_loc%q(i,k,wtrc_iatype(m,iwtliq)) = wtdlf(i,k,m) * (1._r8 - dum1)
           ptend_loc%q(i,k,wtrc_iatype(m,iwtice)) = wtdlf(i,k,m) * dum1
         end do
       end if
       if( cu_det_st ) then
           dlf_t(i,k)  = ptend_loc%s(i,k)/cpair
           dlf_qv(i,k) = 0._r8
           dlf_ql(i,k) = ptend_loc%q(i,k,ixcldliq)
           dlf_qi(i,k) = ptend_loc%q(i,k,ixcldice)
           dlf_nl(i,k) = ptend_loc%q(i,k,ixnumliq)
           dlf_ni(i,k) = ptend_loc%q(i,k,ixnumice)
           ptend_loc%q(i,k,ixcldliq) = 0._r8
           ptend_loc%q(i,k,ixcldice) = 0._r8
           ptend_loc%q(i,k,ixnumliq) = 0._r8
           ptend_loc%q(i,k,ixnumice) = 0._r8
           ptend_loc%s(i,k)          = 0._r8
           dpdlfliq(i,k)             = 0._r8
           dpdlfice(i,k)             = 0._r8
           shdlfliq(i,k)             = 0._r8
           shdlfice(i,k)             = 0._r8
           dpdlft  (i,k)             = 0._r8
           shdlft  (i,k)             = 0._r8
        else
           dpdlfliq(i,k) = ( dlf(i,k) - dlf2(i,k) ) * ( 1._r8 - dum1 )
           dpdlfice(i,k) = ( dlf(i,k) - dlf2(i,k) ) * ( dum1 )
           shdlfliq(i,k) = dlf2(i,k) * ( 1._r8 - dum1 )
           shdlfice(i,k) = dlf2(i,k) * ( dum1 )
           dpdlft  (i,k) = ( dlf(i,k) - dlf2(i,k) ) * dum1 * latice/cpair
           shdlft  (i,k) = dlf2(i,k) * dum1 * latice/cpair
       endif
    end do
    end do
    call outfld( 'DPDLFLIQ ', dpdlfliq, pcols, lchnk )
    call outfld( 'DPDLFICE ', dpdlfice, pcols, lchnk )
    call outfld( 'SHDLFLIQ ', shdlfliq, pcols, lchnk )
    call outfld( 'SHDLFICE ', shdlfice, pcols, lchnk )
    call outfld( 'DPDLFT   ', dpdlft  , pcols, lchnk )
    call outfld( 'SHDLFT   ', shdlft  , pcols, lchnk )
    call outfld( 'ZMDLF',     dlf     , pcols, state_loc%lchnk )
    det_ice(:ncol) = det_ice(:ncol)/1000._r8  ! divide by density of water
    ! Add the detrainment tendency to the output tendency
    call physics_ptend_init(mm_ptend(lchnk), host_state(lchnk)%psetcols, 'macrop')
    call physics_ptend_sum(ptend_loc, mm_ptend(lchnk), ncol)
    ! update local copy of state with the detrainment tendency
    ! ptend_loc is reset to zero by this call
    call physics_update(state_loc, ptend_loc, cld_macmic_ztodt)
    ! -------------------------------------- !
    ! Computation of Various Cloud Fractions !
    ! -------------------------------------- !
    concld_old(:ncol,top_lev:pver) = concld(:ncol,top_lev:pver)
    nullify(tke, qtl_flx, qti_flx, cmfr_det, qlr_det, qir_det)
    if (tke_idx      > 0) call pbuf_get_field(pbuf, tke_idx, tke)
    if (qtl_flx_idx  > 0) call pbuf_get_field(pbuf, qtl_flx_idx,  qtl_flx)
    if (qti_flx_idx  > 0) call pbuf_get_field(pbuf, qti_flx_idx,  qti_flx)
    if (cmfr_det_idx > 0) call pbuf_get_field(pbuf, cmfr_det_idx, cmfr_det)
    if (qlr_det_idx  > 0) call pbuf_get_field(pbuf, qlr_det_idx,  qlr_det)
    if (qir_det_idx  > 0) call pbuf_get_field(pbuf, qir_det_idx,  qir_det)
    clrw_old(:ncol,:top_lev-1) = 0._r8
    clri_old(:ncol,:top_lev-1) = 0._r8
    do k = top_lev, pver
       do i = 1, ncol
          clrw_old(i,k) = max( 0._r8, min( 1._r8, 1._r8 - concld(i,k) - alst(i,k) ) )
          clri_old(i,k) = max( 0._r8, min( 1._r8, 1._r8 - concld(i,k) -  ast(i,k) ) )
       end do
    end do
    if( use_shfrc ) then
        call pbuf_get_field(pbuf, shfrc_idx, shfrc )
    else
        ! the driver allocates a fresh zero array here and never frees it; a
        ! held zero array is the same values
        shfrc => shfrc_zero
        shfrc(:,:) = 0._r8
    endif
    call t_startf("cldfrc")
    call cldfrc( lchnk, ncol, pbuf,                                                 &
                 state_loc%pmid, state_loc%t, state_loc%q(:,:,1), state_loc%omega,  &
                 state_loc%phis, shfrc, use_shfrc,                                  &
                 cld, rhcloud, clc, state_loc%pdel,                                 &
                 cmfmc, cmfmc2, cam_in(lchnk)%landfrac, cam_in(lchnk)%snowhland, concld, cldst, &
                 cam_in(lchnk)%ts, cam_in(lchnk)%sst, state_loc%pint(:,pverp), zdu, cam_in(lchnk)%ocnfrac, rhu00, &
                 state_loc%q(:,:,ixcldice), icecldf, liqcldf,                       &
                 relhum, 0 )
    call t_stopf("cldfrc")
    ! ---------------------------------------------- !
    ! Stratiform Cloud Macrophysics and Microphysics !
    ! ---------------------------------------------- !
    lchnk  = state_loc%lchnk
    ncol   = state_loc%ncol
    rdtime = 1._r8/cld_macmic_ztodt
    call cldfrc_fice( ncol, state_loc%t, fice, fsnow )
    lq(:)        = .FALSE.
    lq(1)        = .true.
    lq(ixcldice) = .true.
    lq(ixcldliq) = .true.
    lq(ixnumliq) = .true.
    lq(ixnumice) = .true.
    !Water tracers:
    do m=1,wtrc_ncnst
      lq(wtrc_indices(m)) = .true.
    end do
    ! Initialize local physics_ptend object again
    call physics_ptend_init(ptend_loc, host_state(lchnk)%psetcols, 'macro_park', &
         ls=.true., lq=lq )
    ! --------------------------------- !
    ! Liquid Macrop_Driver Macrophysics !
    ! --------------------------------- !
    call t_startf('mmacro_pcond')
    zeros(:ncol,top_lev:pver)  = 0._r8
    qc(:ncol,top_lev:pver) = state_loc%q(:ncol,top_lev:pver,ixcldliq)
    qi(:ncol,top_lev:pver) = state_loc%q(:ncol,top_lev:pver,ixcldice)
    nc(:ncol,top_lev:pver) = state_loc%q(:ncol,top_lev:pver,ixnumliq)
    ni(:ncol,top_lev:pver) = state_loc%q(:ncol,top_lev:pver,ixnumice)
    if( get_nstep() .le. 1 ) then
        tcwat(:ncol,:)   = state_loc%t(:ncol,:)
        qcwat(:ncol,:)   = state_loc%q(:ncol,:,1)
        lcwat(:ncol,:)   = qc(:ncol,:) + qi(:ncol,:)
        iccwat(:ncol,:)  = qi(:ncol,:)
        nlwat(:ncol,:)   = nc(:ncol,:)
        niwat(:ncol,:)   = ni(:ncol,:)
        ttend(:ncol,:)   = 0._r8
        qtend(:ncol,:)   = 0._r8
        ltend(:ncol,:)   = 0._r8
        itend(:ncol,:)   = 0._r8
        nltend(:ncol,:)  = 0._r8
        nitend(:ncol,:)  = 0._r8
        cc_t(:ncol,:)    = 0._r8
        cc_qv(:ncol,:)   = 0._r8
        cc_ql(:ncol,:)   = 0._r8
        cc_qi(:ncol,:)   = 0._r8
        cc_nl(:ncol,:)   = 0._r8
        cc_ni(:ncol,:)   = 0._r8
        cc_qlst(:ncol,:) = 0._r8
    else
        ttend(:ncol,top_lev:pver)   = ( state_loc%t(:ncol,top_lev:pver)   -  tcwat(:ncol,top_lev:pver)) * rdtime &
             - cc_t(:ncol,top_lev:pver)
        qtend(:ncol,top_lev:pver)   = ( state_loc%q(:ncol,top_lev:pver,1) -  qcwat(:ncol,top_lev:pver)) * rdtime &
             - cc_qv(:ncol,top_lev:pver)
        ltend(:ncol,top_lev:pver)   = ( qc(:ncol,top_lev:pver) + qi(:ncol,top_lev:pver) - lcwat(:ncol,top_lev:pver) ) * rdtime &
             - (cc_ql(:ncol,top_lev:pver) + cc_qi(:ncol,top_lev:pver))
        itend(:ncol,top_lev:pver)   = ( qi(:ncol,top_lev:pver)         - iccwat(:ncol,top_lev:pver)) * rdtime &
             - cc_qi(:ncol,top_lev:pver)
        nltend(:ncol,top_lev:pver)  = ( nc(:ncol,top_lev:pver)         -  nlwat(:ncol,top_lev:pver)) * rdtime &
             - cc_nl(:ncol,top_lev:pver)
        nitend(:ncol,top_lev:pver)  = ( ni(:ncol,top_lev:pver)         -  niwat(:ncol,top_lev:pver)) * rdtime &
             - cc_ni(:ncol,top_lev:pver)
    endif
    lmitend(:ncol,top_lev:pver) = ltend(:ncol,top_lev:pver) - itend(:ncol,top_lev:pver)
    t_inout(:ncol,top_lev:pver)  =  tcwat(:ncol,top_lev:pver)
    qv_inout(:ncol,top_lev:pver) =  qcwat(:ncol,top_lev:pver)
    ql_inout(:ncol,top_lev:pver) =  lcwat(:ncol,top_lev:pver) - iccwat(:ncol,top_lev:pver)
    qi_inout(:ncol,top_lev:pver) = iccwat(:ncol,top_lev:pver)
    nl_inout(:ncol,top_lev:pver) =  nlwat(:ncol,top_lev:pver)
    ni_inout(:ncol,top_lev:pver) =  niwat(:ncol,top_lev:pver)
  end subroutine macro_before_pcond

  subroutine original_pcond()
    ! macrop_driver.F90:1028-1037: the call itself, the original routine on
    ! the same arguments, when mmacro_pcond is not replaced
    call mmacro_pcond( lchnk, ncol, cld_macmic_ztodt, state_loc%pmid, state_loc%pdel, &
                       t_inout, qv_inout, ql_inout, qi_inout, nl_inout, ni_inout,     &
                       ttend, qtend, lmitend, itend, nltend, nitend,                  &
                       cc_t, cc_qv, cc_ql, cc_qi, cc_nl, cc_ni, cc_qlst,              &
                       dlf_t, dlf_qv, dlf_ql, dlf_qi, dlf_nl, dlf_ni,                 &
                       concld_old, concld, clrw_old, clri_old, cam_in(lchnk)%landfrac, cam_in(lchnk)%snowhland, &
                       tke, qtl_flx, qti_flx, cmfr_det, qlr_det, qir_det,             &
                       tlat, qvlat, qcten, qiten, ncten, niten,                       &
                       cmeliq, qvadj, qladj, qiadj, qllim, qilim,                     &
                       cld, alst, aist, qlst, qist, do_cldice )
  end subroutine original_pcond

  subroutine macro_after_pcond()
    ! macrop_driver.F90:1038-{macro[1]}: from the call to the end of the driver
    fice_ql(:ncol,:top_lev-1)     = 0._r8
    fice_ql(:ncol,top_lev:pver)   = fice(:ncol,top_lev:pver)
    ast(:ncol,:top_lev-1) = 0._r8
    ast(:ncol,top_lev:pver) = max( alst(:ncol,top_lev:pver), aist(:ncol,top_lev:pver) )
    call t_stopf('mmacro_pcond')
    do k = top_lev, pver
       do i = 1, ncol
          ptend_loc%s(i,k)          =  tlat(i,k)
          ptend_loc%q(i,k,1)        = qvlat(i,k)
          ptend_loc%q(i,k,ixcldliq) = qcten(i,k)
          ptend_loc%q(i,k,ixcldice) = qiten(i,k)
          ptend_loc%q(i,k,ixnumliq) = ncten(i,k)
          ptend_loc%q(i,k,ixnumice) = niten(i,k)
          if ((.not. do_cldice) .and. (qiten(i,k) /= 0.0_r8)) then
             call endrun("macrop_driver:ERROR - "// &
                  "Cldwat is configured not to prognose cloud ice, but mmacro_pcond has ice mass tendencies.")
          end if
          if ((.not. do_cldice) .and. (niten(i,k) /= 0.0_r8)) then
             call endrun("macrop_driver:ERROR -"// &
                  " Cldwat is configured not to prognose cloud ice, but mmacro_pcond has ice number tendencies.")
          end if
          if ((.not. do_cldliq) .and. (qcten(i,k) /= 0.0_r8)) then
             call endrun("macrop_driver:ERROR - "// &
                  "Cldwat is configured not to prognose cloud liquid, but mmacro_pcond has liquid mass tendencies.")
          end if
          if ((.not. do_cldliq) .and. (ncten(i,k) /= 0.0_r8)) then
             call endrun("macrop_driver:ERROR - "// &
                  "Cldwat is configured not to prognose cloud liquid, but mmacro_pcond has liquid number tendencies.")
          end if
       end do
    end do
    if (trace_water) then
      call wtrc_init_rates(top_lev, process_rates)
      pqctn(:,top_lev:) = 0._r8
      nqctn(:,top_lev:) = 0._r8
      pqitn(:,top_lev:) = 0._r8
      nqitn(:,top_lev:) = 0._r8
      do i=1,ncol
        do k=top_lev,pver
          if(qcten(i,k) .lt. 0._r8) then
            nqctn(i,k) = qcten(i,k)
          else
            pqctn(i,k) = qcten(i,k)
          end if
          if(qiten(i,k) .lt. 0._r8) then
            nqitn(i,k) = qiten(i,k)
          else
            pqitn(i,k) = qiten(i,k)
          end if
        end do
      end do
      call wtrc_add_rates(process_rates, ncol, top_lev, iwtvap, iwtvap, iwtvap, qvlat + qcten + qiten)
      call wtrc_add_rates(process_rates, ncol, top_lev, iwtvap, iwtliq, iwtvap, pqctn)
      call wtrc_add_rates(process_rates, ncol, top_lev, iwtvap, iwtliq, iwtliq, nqctn)
      call wtrc_add_rates(process_rates, ncol, top_lev, iwtvap, iwtice, iwtvap, pqitn)
      call wtrc_add_rates(process_rates, ncol, top_lev, iwtvap, iwtice, iwtice, nqitn)
      call wtrc_apply_rates(state_loc, ptend_loc, pbuf, top_lev, cld_macmic_ztodt, .false., pre_rates=process_rates, &
                            prelat=tlat)
    end if !water tracers
    ! update the output tendencies with the mmacro_pcond tendencies
    call physics_ptend_sum(ptend_loc, mm_ptend(lchnk), ncol)
    ! state_loc is the equlibrium state after macrophysics
    call physics_update(state_loc, ptend_loc, cld_macmic_ztodt)
    call outfld('CLR_LIQ', clrw_old,  pcols, lchnk)
    call outfld('CLR_ICE', clri_old,  pcols, lchnk)
    call outfld( 'MACPDT   ', tlat ,  pcols, lchnk )
    call outfld( 'MACPDQ   ', qvlat,  pcols, lchnk )
    call outfld( 'MACPDLIQ ', qcten,  pcols, lchnk )
    call outfld( 'MACPDICE ', qiten,  pcols, lchnk )
    call outfld( 'CLDVAPADJ', qvadj,  pcols, lchnk )
    call outfld( 'CLDLIQADJ', qladj,  pcols, lchnk )
    call outfld( 'CLDICEADJ', qiadj,  pcols, lchnk )
    call outfld( 'CLDLIQDET', dlf_ql, pcols, lchnk )
    call outfld( 'CLDICEDET', dlf_qi, pcols, lchnk )
    call outfld( 'CLDLIQLIM', qllim,  pcols, lchnk )
    call outfld( 'CLDICELIM', qilim,  pcols, lchnk )
    call outfld( 'ICECLDF ', aist,   pcols, lchnk )
    call outfld( 'LIQCLDF ', alst,   pcols, lchnk )
    call outfld( 'AST',      ast,    pcols, lchnk )
    call outfld( 'CONCLD  ', concld, pcols, lchnk )
    call outfld( 'CLDST   ', cldst,  pcols, lchnk )
    call outfld( 'CMELIQ'  , cmeliq, pcols, lchnk )
    mr_ccliq = 0._r8   !! not seen by radiation, so setting to 0
    mr_ccice = 0._r8   !! not seen by radiation, so setting to 0
    mr_lsliq = 0._r8
    mr_lsice = 0._r8
    do k=top_lev,pver
       do i=1,ncol
          if (cld(i,k) .gt. 0._r8) then
             mr_lsliq(i,k) = state_loc%q(i,k,ixcldliq)
             mr_lsice(i,k) = state_loc%q(i,k,ixcldice)
          else
             mr_lsliq(i,k) = 0._r8
             mr_lsice(i,k) = 0._r8
          end if
       end do
    end do
    call outfld( 'CLDLIQSTR  ', mr_lsliq,    pcols, lchnk )
    call outfld( 'CLDICESTR  ', mr_lsice,    pcols, lchnk )
    call outfld( 'CLDLIQCON  ', mr_ccliq,    pcols, lchnk )
    call outfld( 'CLDICECON  ', mr_ccice,    pcols, lchnk )
    cldsice = 0._r8
    do k = top_lev, pver
       tcwat(:ncol,k)  = state_loc%t(:ncol,k)
       qcwat(:ncol,k)  = state_loc%q(:ncol,k,1)
       lcwat(:ncol,k)  = state_loc%q(:ncol,k,ixcldliq) + state_loc%q(:ncol,k,ixcldice)
       iccwat(:ncol,k) = state_loc%q(:ncol,k,ixcldice)
       nlwat(:ncol,k)  = state_loc%q(:ncol,k,ixnumliq)
       niwat(:ncol,k)  = state_loc%q(:ncol,k,ixnumice)
       cldsice(:ncol,k) = lcwat(:ncol,k) * min(1.0_r8, max(0.0_r8, (tmelt - tcwat(:ncol,k)) / 20._r8))
    end do
    call outfld( 'CLDSICE'    , cldsice,   pcols, lchnk )
    ! ptend_loc is deallocated in physics_update above
    call physics_state_dealloc(state_loc)
  end subroutine macro_after_pcond

  subroutine substep_tail_to_micro()
    ! physpkg.F90:2251-2351: after the macrophysics driver returns, to the
    ! microphysics driver -- energy check, aerosol activation -- and then
    ! microp_driver.F90:156-183 and the head of micro_mg_cam_tend, its
    ! pieces held by pycam_micro_handles.  ncol is tphysbc's.
    ncol = host_state(lchnk)%ncol
    !  Since we "added" the reserved liquid back in this routine, we need
    !    to account for it in the energy checker
    flx_cnd(:ncol) = -1._r8*rliq(:ncol)
    flx_heat(:ncol) = det_s(:ncol)
    call physics_ptend_scale(mm_ptend(lchnk), 1._r8/cld_macmic_num_steps, ncol)
    call physics_update(host_state(lchnk), mm_ptend(lchnk), ztodt, host_tend(lchnk))
    call check_energy_chng(host_state(lchnk), host_tend(lchnk), "macrop_tend", nstep, ztodt, &
         zero, flx_cnd/cld_macmic_num_steps, &
         det_ice/cld_macmic_num_steps, flx_heat/cld_macmic_num_steps)
    call t_stopf('macrop_tend')
    !===================================================
    ! Calculate cloud microphysics
    !===================================================
    call t_startf('microp_aero_run')
    call microp_aero_run(host_state(lchnk), mm_ptend_aero(lchnk), cld_macmic_ztodt, pbuf)
    call t_stopf('microp_aero_run')
    call t_startf('microp_tend')
    ! microp_driver_tend: its select on microp_scheme is 'MG' here (refused otherwise at create)
    call t_startf('microp_mg_tend')
    ! micro_mg_cam_tend(state, ptend, dtime, pbuf), in its pieces: the dummies
    call micro_runner_bind(lchnk, mm_ptend(lchnk), cld_macmic_ztodt)
    call micro_run_head()
    call micro_pack_prelude()
  end subroutine substep_tail_to_micro

  subroutine substep_tail_after_micro()
    ! micro_mg_cam_tend has returned; microp_driver.F90:184-192 and
    ! physpkg.F90:2352-2378 -- energy check, precipitation accumulation
    call t_stopf('microp_mg_tend')
    ! combine aero and micro tendencies for the grid
    call physics_ptend_sum(mm_ptend_aero(lchnk), mm_ptend(lchnk), ncol)
    call physics_ptend_dealloc(mm_ptend_aero(lchnk))
    ! Have to scale and apply for full timestep to get tend right
    ! (see above note for macrophysics).
    call physics_ptend_scale(mm_ptend(lchnk), 1._r8/cld_macmic_num_steps, ncol)
    call physics_update (host_state(lchnk), mm_ptend(lchnk), ztodt, host_tend(lchnk))
    call check_energy_chng(host_state(lchnk), host_tend(lchnk), "microp_tend", nstep, ztodt, &
         zero, prec_str/cld_macmic_num_steps, &
         snow_str/cld_macmic_num_steps, zero)
    call t_stopf('microp_tend')
    prec_sed_macmic(:ncol) = prec_sed_macmic(:ncol) + prec_sed(:ncol)
    snow_sed_macmic(:ncol) = snow_sed_macmic(:ncol) + snow_sed(:ncol)
    prec_pcw_macmic(:ncol) = prec_pcw_macmic(:ncol) + prec_pcw(:ncol)
    snow_pcw_macmic(:ncol) = snow_pcw_macmic(:ncol) + snow_pcw(:ncol)
  end subroutine substep_tail_after_micro

  subroutine chunk_end()
    ! physpkg.F90:2379-{physpkg[1]}: after the substeps -- the means, and the
    ! tracer mass fixer.  CARMA is off in this configuration.
    ncol = host_state(lchnk)%ncol
    prec_sed(:ncol) = prec_sed_macmic(:ncol)/cld_macmic_num_steps
    snow_sed(:ncol) = snow_sed_macmic(:ncol)/cld_macmic_num_steps
    prec_pcw(:ncol) = prec_pcw_macmic(:ncol)/cld_macmic_num_steps
    snow_pcw(:ncol) = snow_pcw_macmic(:ncol)/cld_macmic_num_steps
    prec_str(:ncol) = prec_pcw(:ncol) + prec_sed(:ncol)
    snow_str(:ncol) = snow_pcw(:ncol) + snow_sed(:ncol)
    if(trace_water) then
      call wtrc_mass_fixer(host_state(lchnk))
    end if
  end subroutine chunk_end

  ! ------------------------------------------------------------------ !
  ! Frame slots: where an argument lives, for Python
  ! ------------------------------------------------------------------ !

  subroutine scalar_slot(index, address, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    integer, intent(in) :: index, dtype, intent
    type(c_ptr), intent(in) :: address
    type(c_ptr), intent(inout) :: ptrs(:)
    integer(c_int), intent(inout) :: ndims(:), dtypes(:), intents(:)
    integer(c_int64_t), intent(inout) :: shapes(:,:)
    ptrs(index) = address
    ndims(index) = 0_c_int
    shapes(:, index) = 0_c_int64_t
    dtypes(index) = int(dtype, c_int)
    intents(index) = int(intent, c_int)
  end subroutine scalar_slot

  subroutine slot1(index, array, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    integer, intent(in) :: index, dtype, intent
    real(r8), target, intent(in) :: array(:)
    type(c_ptr), intent(inout) :: ptrs(:)
    integer(c_int), intent(inout) :: ndims(:), dtypes(:), intents(:)
    integer(c_int64_t), intent(inout) :: shapes(:,:)
    ptrs(index) = c_loc(array(1))
    ndims(index) = 1_c_int
    shapes(:, index) = 0_c_int64_t
    shapes(1, index) = int(size(array, 1), c_int64_t)
    dtypes(index) = int(dtype, c_int)
    intents(index) = int(intent, c_int)
  end subroutine slot1

  subroutine slot2_or(index, field, fallback, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    ! a physics-buffer pointer where the configuration has the field, the
    ! zero workspace where it is unassociated -- what the original passes
    integer, intent(in) :: index, dtype, intent
    real(r8), pointer, intent(in) :: field(:,:)
    real(r8), target, intent(in) :: fallback(:,:)
    type(c_ptr), intent(inout) :: ptrs(:)
    integer(c_int), intent(inout) :: ndims(:), dtypes(:), intents(:)
    integer(c_int64_t), intent(inout) :: shapes(:,:)
    if (associated(field)) then
      call slot2(index, field, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    else
      call slot2(index, fallback, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    end if
  end subroutine slot2_or

  subroutine slot2(index, array, dtype, intent, ptrs, ndims, shapes, dtypes, intents)
    integer, intent(in) :: index, dtype, intent
    real(r8), target, intent(in) :: array(:,:)
    type(c_ptr), intent(inout) :: ptrs(:)
    integer(c_int), intent(inout) :: ndims(:), dtypes(:), intents(:)
    integer(c_int64_t), intent(inout) :: shapes(:,:)
    ptrs(index) = c_loc(array(1,1))
    ndims(index) = 2_c_int
    shapes(:, index) = 0_c_int64_t
    shapes(1, index) = int(size(array, 1), c_int64_t)
    shapes(2, index) = int(size(array, 2), c_int64_t)
    dtypes(index) = int(dtype, c_int)
    intents(index) = int(intent, c_int)
  end subroutine slot2

end module pycam_stage7_runner
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-digests", action="store_true",
                        help="print the pinned ranges' digests and exit")
    arguments = parser.parse_args()
    if arguments.print_digests:
        for name, (first, last, _) in ANCHORS.items():
            print(name, first, last, range_digest(name, first, last))
        return 0
    for name, (first, last, digest) in ANCHORS.items():
        actual = range_digest(name, first, last)
        if actual != digest:
            sys.stderr.write(f"{name}:{first}-{last} hashes to {actual}, not {digest}: the pinned "
                             f"source moved under the runner; re-read the transcription\n")
            return 2
    rendered = render_module()
    if arguments.check:
        current = MODULE.read_text() if MODULE.exists() else ""
        if current != rendered:
            sys.stderr.write("".join(difflib.unified_diff(
                current.splitlines(True), rendered.splitlines(True),
                str(MODULE), "generated")))
            return 1
        return 0
    MODULE.write_text(rendered)
    print(f"wrote {MODULE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
