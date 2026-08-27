#!/usr/bin/env python3
"""Emit the handles module a Python-driven radiation timestep calls.

``pycam_rad_kernels`` holds the driver's arithmetic.  Everything else the
driver does is a call into CAM that takes a derived type -- ``physics_state``,
``physics_buffer_desc``, ``cam_in_t``, ``cam_out_t``, ``physics_ptend``,
``rrtmg_state_t`` -- and so cannot be promoted as a direct kernel.  This
module is where those calls live: one ``bind(C)`` wrapper each, holding the
derived types in Fortran and handing Python zero-copy views of the numeric
components it needs.

What the module owns, and why:

* ``rad_ptend(begchunk:endchunk)`` -- ``radheat_tend`` declares its ptend
  ``intent(out)`` and allocates inside the timestep, the same contract
  problem macrophysics had.  The resume half of ``tphysbc`` takes it back.
* ``rad_net_flx(pcols, begchunk:endchunk)`` -- a ``tphysbc`` local the
  resume needs, so it has to survive the return to Python and back.
* ``python_rstate`` -- ``rrtmg_state_create`` allocates fourteen arrays and
  ``rrtmg_state_destroy`` frees them, both inside one chunk's work.  One
  pointer is enough because the driver's own lifetime is chunk-local.
* ``host_cam_in`` / ``host_cam_out`` -- ``cam_in`` and ``cam_out`` are
  dummies of ``tphysbc``, not module state anywhere below it, and the
  StatePool's copy of them is a transfer rather than a live view (see
  ``cam_python_state_transfer``).  The boundary patch binds the chunk's two
  objects by address at the stop.

What the module does **not** need: the surface flux arrays.  ``fsns``,
``fsnt``, ``flns``, ``flnt`` and ``fsds`` are module-scope allocatables in
``comsrf``, so a ``use`` reaches the same storage ``tphysbc`` was passed.

    tools/generate_pi_cam_rad_handles.py            # write the module
    tools/generate_pi_cam_rad_handles.py --check    # fail if it is stale
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "native/pi_cam/support/pycam_rad_handles.F90"

#: ``pycam_rad_view_v1`` codes.  Python's ``VIEW`` table mirrors this and a
#: test keeps the two equal.
#: ``pycam_rad_view_v1`` codes: name -> (code, rank, expression, liveness
#: guard).  Python's ``VIEW`` table mirrors the codes and a test keeps the
#: two equal.  A view whose storage is not alive is refused, never handed
#: out as a dangling address.
VIEWS = {
    "state_t": (1, 2, "host_state(lchnk)%t", None),
    "state_pmid": (2, 2, "host_state(lchnk)%pmid", None),
    "state_pint": (3, 2, "host_state(lchnk)%pint", None),
    "state_pdel": (4, 2, "host_state(lchnk)%pdel", None),
    "state_lnpint": (5, 2, "host_state(lchnk)%lnpint", None),
    "state_lnpmid": (6, 2, "host_state(lchnk)%lnpmid", None),
    "ptend_s": (21, 2, "rad_ptend(lchnk)%s", "allocated(rad_ptend(lchnk)%s)"),
    "cam_in_lwup": (41, 1, "host_cam_in(lchnk)%p%lwup", "associated(host_cam_in(lchnk)%p)"),
    "cam_in_asdir": (42, 1, "host_cam_in(lchnk)%p%asdir", "associated(host_cam_in(lchnk)%p)"),
    "cam_in_asdif": (43, 1, "host_cam_in(lchnk)%p%asdif", "associated(host_cam_in(lchnk)%p)"),
    "cam_in_aldir": (44, 1, "host_cam_in(lchnk)%p%aldir", "associated(host_cam_in(lchnk)%p)"),
    "cam_in_aldif": (45, 1, "host_cam_in(lchnk)%p%aldif", "associated(host_cam_in(lchnk)%p)"),
    "cam_out_sols": (51, 1, "host_cam_out(lchnk)%p%sols", "associated(host_cam_out(lchnk)%p)"),
    "cam_out_soll": (52, 1, "host_cam_out(lchnk)%p%soll", "associated(host_cam_out(lchnk)%p)"),
    "cam_out_solsd": (53, 1, "host_cam_out(lchnk)%p%solsd", "associated(host_cam_out(lchnk)%p)"),
    "cam_out_solld": (54, 1, "host_cam_out(lchnk)%p%solld", "associated(host_cam_out(lchnk)%p)"),
    "cam_out_flwds": (55, 1, "host_cam_out(lchnk)%p%flwds", "associated(host_cam_out(lchnk)%p)"),
    "cam_out_netsw": (56, 1, "host_cam_out(lchnk)%p%netsw", "associated(host_cam_out(lchnk)%p)"),
    "fsns": (61, 1, "fsns(:,lchnk)", "allocated(fsns)"),
    "fsnt": (62, 1, "fsnt(:,lchnk)", "allocated(fsnt)"),
    "flns": (63, 1, "flns(:,lchnk)", "allocated(flns)"),
    "flnt": (64, 1, "flnt(:,lchnk)", "allocated(flnt)"),
    "fsds": (65, 1, "fsds(:,lchnk)", "allocated(fsds)"),
    "net_flx": (71, 1, "rad_net_flx(:,lchnk)", None),
    "rstate_h2ovmr": (81, 2, "python_rstate%h2ovmr", "rstate_live"),
    "rstate_o3vmr": (82, 2, "python_rstate%o3vmr", "rstate_live"),
    "rstate_co2vmr": (83, 2, "python_rstate%co2vmr", "rstate_live"),
    "rstate_ch4vmr": (84, 2, "python_rstate%ch4vmr", "rstate_live"),
    "rstate_o2vmr": (85, 2, "python_rstate%o2vmr", "rstate_live"),
    "rstate_n2ovmr": (86, 2, "python_rstate%n2ovmr", "rstate_live"),
    "rstate_cfc11vmr": (87, 2, "python_rstate%cfc11vmr", "rstate_live"),
    "rstate_cfc12vmr": (88, 2, "python_rstate%cfc12vmr", "rstate_live"),
    "rstate_cfc22vmr": (89, 2, "python_rstate%cfc22vmr", "rstate_live"),
    "rstate_ccl4vmr": (90, 2, "python_rstate%ccl4vmr", "rstate_live"),
    "rstate_pmidmb": (91, 2, "python_rstate%pmidmb", "rstate_live"),
    "rstate_pintmb": (92, 2, "python_rstate%pintmb", "rstate_live"),
    "rstate_tlay": (93, 2, "python_rstate%tlay", "rstate_live"),
    "rstate_tlev": (94, 2, "python_rstate%tlev", "rstate_live"),
}

#: Views that live on ``python_rstate``, which only exists between create and
#: destroy; asking for one outside that window is a refusal, not a crash.
RSTATE_VIEWS = tuple(name for name in VIEWS if name.startswith("rstate_"))

_SW = "nbndsw,pcols,pver"
_LW = "nbndlw,pcols,pver"
_AER_SW = "pcols,0:pver,nbndsw"


class Wrapper:
    """One bind(C) entry: its C dummies, and the Fortran call it makes."""

    def __init__(self, name, dummies, declarations, body, *, needs_pbuf=False,
                 needs_state=False, needs_rstate=False, needs_cam=False):
        self.name = name
        self.dummies = dummies
        self.declarations = declarations
        self.body = body
        self.needs_pbuf = needs_pbuf
        self.needs_state = needs_state
        self.needs_rstate = needs_rstate
        self.needs_cam = needs_cam


def _optics_sw(name, routine):
    return Wrapper(
        name, ["lchnk", "tau", "tau_w", "tau_w_g", "tau_w_f"],
        [f"    real(c_double), intent(inout) :: tau({_SW}), tau_w({_SW})",
         f"    real(c_double), intent(inout) :: tau_w_g({_SW}), tau_w_f({_SW})"],
        [f"    call {routine}(host_state(lchnk), pbuf, tau, tau_w, tau_w_g, tau_w_f)"],
        needs_pbuf=True, needs_state=True)


def _props_lw(name, routine):
    return Wrapper(
        name, ["lchnk", "abs_od"],
        [f"    real(c_double), intent(inout) :: abs_od({_LW})"],
        [f"    call {routine}(host_state(lchnk), pbuf, abs_od)"],
        needs_pbuf=True, needs_state=True)


WRAPPERS = (
    # -- cloud optics: the branches the admitted configuration takes ---------
    _optics_sw("ice_optics_sw", "get_ice_optics_sw"),
    _optics_sw("liquid_optics_sw", "get_liquid_optics_sw"),
    _optics_sw("snow_optics_sw", "get_snow_optics_sw"),
    _props_lw("ice_props_lw", "ice_cloud_get_rad_props_lw"),
    _props_lw("liquid_props_lw", "liquid_cloud_get_rad_props_lw"),
    _props_lw("snow_props_lw", "snow_cloud_get_rad_props_lw"),
    # -- aerosol optics ------------------------------------------------------
    Wrapper(
        "aer_props_sw", ["lchnk", "list_idx", "nnite", "idxnite",
                         "tau", "tau_w", "tau_w_g", "tau_w_f"],
        ["    integer(c_int), value, intent(in) :: list_idx, nnite",
         "    integer(c_int), intent(in) :: idxnite(pcols)",
         f"    real(c_double), intent(inout) :: tau({_AER_SW}), tau_w({_AER_SW})",
         f"    real(c_double), intent(inout) :: tau_w_g({_AER_SW}), tau_w_f({_AER_SW})"],
        ["    call aer_rad_props_sw(list_idx, host_state(lchnk), pbuf, nnite, idxnite, &",
         "         tau, tau_w, tau_w_g, tau_w_f)"],
        needs_pbuf=True, needs_state=True),
    Wrapper(
        "aer_props_lw", ["lchnk", "list_idx", "odap_aer"],
        ["    integer(c_int), value, intent(in) :: list_idx",
         "    real(c_double), intent(inout) :: odap_aer(pcols,pver,nbndlw)"],
        ["    call aer_rad_props_lw(list_idx, host_state(lchnk), pbuf, odap_aer)"],
        needs_pbuf=True, needs_state=True),
    # -- the RRTMG state's lifetime -----------------------------------------
    Wrapper(
        "rstate_create", ["lchnk"], [],
        ["    if (rstate_live) return",
         "    python_rstate => rrtmg_state_create(host_state(lchnk), host_cam_in(lchnk)%p)",
         "    rstate_live = .true."],
        needs_state=True, needs_cam=True),
    Wrapper(
        "rstate_update", ["lchnk", "icall"],
        ["    integer(c_int), value, intent(in) :: icall"],
        ["    if (.not. rstate_live) return",
         "    call rrtmg_state_update(host_state(lchnk), pbuf, icall, python_rstate)"],
        needs_pbuf=True, needs_state=True),
    Wrapper(
        "rstate_destroy", ["lchnk"], [],
        ["    if (.not. rstate_live) return",
         "    call rrtmg_state_destroy(python_rstate)",
         "    nullify(python_rstate)",
         "    rstate_live = .false."]),
    # -- the two numerical cores --------------------------------------------
    Wrapper(
        "rrtmg_sw",
        ["lchnk", "ncol", "rrtmg_levs", "nday", "nnite", "idxday", "idxnite",
         "e_pmid", "e_cld", "e_aer_tau", "e_aer_tau_w", "e_aer_tau_w_g", "e_aer_tau_w_f",
         "eccf", "e_coszrs", "e_asdir", "e_asdif", "e_aldir", "e_aldif", "sfac",
         "e_cld_tau", "e_cld_tau_w", "e_cld_tau_w_g", "e_cld_tau_w_f",
         "solin", "qrs", "qrsc", "fsnt_o", "fsntc_o", "fsntoa", "fsutoa", "fsntoac",
         "fsnirtoa", "fsnrtoac", "fsnrtoaq", "fsns_o", "fsnsc", "fsdsc", "fsds_o",
         "sols", "soll", "solsd", "solld", "fns", "fcns"],
        ["    integer(c_int), value, intent(in) :: ncol, rrtmg_levs, nday, nnite",
         "    integer(c_int), intent(in) :: idxday(pcols), idxnite(pcols)",
         "    real(c_double), intent(in) :: e_pmid(pcols,pver), e_cld(pcols,pver)",
         f"    real(c_double), intent(in) :: e_aer_tau({_AER_SW}), e_aer_tau_w({_AER_SW})",
         f"    real(c_double), intent(in) :: e_aer_tau_w_g({_AER_SW}), e_aer_tau_w_f({_AER_SW})",
         "    real(c_double), value, intent(in) :: eccf",
         "    real(c_double), intent(in) :: e_coszrs(pcols)",
         "    real(c_double), intent(in) :: e_asdir(pcols), e_asdif(pcols)",
         "    real(c_double), intent(in) :: e_aldir(pcols), e_aldif(pcols)",
         "    real(c_double), intent(in) :: sfac(nbndsw)",
         f"    real(c_double), intent(in) :: e_cld_tau({_SW}), e_cld_tau_w({_SW})",
         f"    real(c_double), intent(in) :: e_cld_tau_w_g({_SW}), e_cld_tau_w_f({_SW})",
         "    real(c_double), intent(inout) :: solin(pcols), qrs(pcols,pver), qrsc(pcols,pver)",
         "    real(c_double), intent(inout) :: fsnt_o(pcols), fsntc_o(pcols)",
         "    real(c_double), intent(inout) :: fsntoa(pcols), fsutoa(pcols), fsntoac(pcols)",
         "    real(c_double), intent(inout) :: fsnirtoa(pcols), fsnrtoac(pcols), fsnrtoaq(pcols)",
         "    real(c_double), intent(inout) :: fsns_o(pcols), fsnsc(pcols)",
         "    real(c_double), intent(inout) :: fsdsc(pcols), fsds_o(pcols)",
         "    real(c_double), intent(inout) :: sols(pcols), soll(pcols)",
         "    real(c_double), intent(inout) :: solsd(pcols), solld(pcols)",
         "    real(c_double), intent(inout) :: fns(pcols,pverp), fcns(pcols,pverp)"],
        ["    if (.not. rstate_live) return",
         "    ! The driver's call, argument for argument: spectralflux is refused",
         "    ! at attach so su and sd stay the null pointers the driver passes,",
         "    ! and old_convert is .false. there as here.",
         "    call rad_rrtmg_sw( &",
         "         lchnk,        ncol,         rrtmg_levs,   python_rstate,               &",
         "         e_pmid,       e_cld,                                                   &",
         "         e_aer_tau,    e_aer_tau_w,  e_aer_tau_w_g, e_aer_tau_w_f,              &",
         "         eccf,         e_coszrs,     solin,        sfac,                        &",
         "         e_asdir,      e_asdif,      e_aldir,      e_aldif,                     &",
         "         qrs,          qrsc,         fsnt_o,       fsntc_o,      fsntoa, fsutoa, &",
         "         fsntoac,      fsnirtoa,     fsnrtoac,     fsnrtoaq,     fsns_o,        &",
         "         fsnsc,        fsdsc,        fsds_o,       sols,         soll,          &",
         "         solsd,        solld,        fns,          fcns,                        &",
         "         nday,         nnite,        idxday,       idxnite,                     &",
         "         null_su,      null_sd,                                                 &",
         "         E_cld_tau=e_cld_tau, E_cld_tau_w=e_cld_tau_w, &",
         "         E_cld_tau_w_g=e_cld_tau_w_g, E_cld_tau_w_f=e_cld_tau_w_f, &",
         "         old_convert = .false.)"],
        needs_rstate=True),
    Wrapper(
        "rrtmg_lw",
        ["lchnk", "ncol", "rrtmg_levs", "pmid", "aer_lw_abs", "cld", "tauc_lw",
         "qrl", "qrlc", "flns_o", "flnt_o", "flnsc", "flntc", "flwds",
         "flut", "flutc", "fnl", "fcnl", "fldsc"],
        ["    integer(c_int), value, intent(in) :: ncol, rrtmg_levs",
         "    real(c_double), intent(in) :: pmid(pcols,pver), cld(pcols,pver)",
         "    real(c_double), intent(in) :: aer_lw_abs(pcols,pver,nbndlw)",
         f"    real(c_double), intent(in) :: tauc_lw({_LW})",
         "    real(c_double), intent(inout) :: qrl(pcols,pver), qrlc(pcols,pver)",
         "    real(c_double), intent(inout) :: flns_o(pcols), flnt_o(pcols)",
         "    real(c_double), intent(inout) :: flnsc(pcols), flntc(pcols), flwds(pcols)",
         "    real(c_double), intent(inout) :: flut(pcols), flutc(pcols), fldsc(pcols)",
         "    real(c_double), intent(inout) :: fnl(pcols,pverp), fcnl(pcols,pverp)"],
        ["    if (.not. rstate_live) return",
         "    call rad_rrtmg_lw( &",
         "         lchnk,        ncol,         rrtmg_levs,   python_rstate,               &",
         "         pmid,         aer_lw_abs,   cld,          tauc_lw,                     &",
         "         qrl,          qrlc,                                                    &",
         "         flns_o,       flnt_o,       flnsc,        flntc,        flwds,         &",
         "         flut,         flutc,        fnl,          fcnl,         fldsc,         &",
         "         null_lu,      null_ld)"],
        needs_rstate=True),
    # -- the rest of the driver's derived-type calls -------------------------
    Wrapper(
        "tropopause_find", ["lchnk", "troplev", "trop_p"],
        ["    integer(c_int), intent(inout) :: troplev(pcols)",
         "    real(c_double), intent(inout) :: trop_p(pcols)"],
        ["    call tropopause_find(host_state(lchnk), troplev, tropP=trop_p, &",
         "         primary=TROP_ALG_HYBSTOB, backup=TROP_ALG_CLIMATE)"],
        needs_state=True),
    Wrapper(
        "cnst_out", ["lchnk", "list_idx"],
        ["    integer(c_int), value, intent(in) :: list_idx"],
        ["    call rad_cnst_out(list_idx, host_state(lchnk), pbuf)"],
        needs_pbuf=True, needs_state=True),
    Wrapper(
        "data_write", ["lchnk", "coszrs"],
        ["    real(c_double), intent(in) :: coszrs(pcols)"],
        ["    call rad_data_write(pbuf, host_state(lchnk), host_cam_in(lchnk)%p, coszrs)"],
        needs_pbuf=True, needs_state=True, needs_cam=True),
    Wrapper(
        "radheat", ["lchnk", "qrl", "qrs"],
        ["    real(c_double), intent(in) :: qrl(pcols,pver), qrs(pcols,pver)"],
        ["    ! ptend is intent(out): radheat_tend initialises and fills it, and",
         "    ! the resume half of tphysbc takes rad_ptend(lchnk) back.",
         "    call radheat_tend(host_state(lchnk), pbuf, rad_ptend(lchnk), qrl, qrs, &",
         "         fsns(:,lchnk), fsnt(:,lchnk), flns(:,lchnk), flnt(:,lchnk), &",
         "         host_cam_in(lchnk)%p%asdir, rad_net_flx(:,lchnk))"],
        needs_pbuf=True, needs_state=True, needs_cam=True),
    # -- zenith: a bare external subroutine, so not a direct kernel ---------
    Wrapper(
        "zenith", ["lchnk", "ncol", "calday", "clat", "clon", "coszrs"],
        ["    integer(c_int), value, intent(in) :: ncol",
         "    real(c_double), value, intent(in) :: calday",
         "    real(c_double), intent(in) :: clat(pcols), clon(pcols)",
         "    real(c_double), intent(inout) :: coszrs(pcols)"],
        ["    ! zenith lives at file scope in zenith.F90, not in a module, so a",
         "    ! generated direct-kernel wrapper cannot use-associate it.  An",
         "    ! external call needs no interface and is legal here.",
         "    call zenith(calday, clat, clon, coszrs, ncol)"]),

    # -- the one history call that carries arithmetic ------------------------
    Wrapper(
        "outfld_scaled", ["lchnk", "ncol", "name", "name_len", "field", "cpair_in"],
        ["    integer(c_int), value, intent(in) :: ncol, name_len",
         "    character(kind=c_char), intent(in) :: name(name_len)",
         "    real(c_double), intent(in) :: field(pcols,pver)",
         "    real(c_double), value, intent(in) :: cpair_in",
         "    character(len=name_len) :: label",
         "    integer :: i"],
        ["    do i = 1, name_len",
         "      label(i:i) = name(i)",
         "    end do",
         "    ! radiation.F90:1170-1171 -- the division and the (:ncol,:) shape are",
         "    ! one expression, and outfld is given idim = ncol.  Splitting them",
         "    ! would change what outfld receives, so the line is kept whole.",
         "    call outfld(label, field(:ncol,:)/cpair_in, ncol, lchnk)"]),
)


def _entry(wrapper: Wrapper) -> str:
    name = f"pycam_rad_{wrapper.name}_v1"
    args = ", ".join(wrapper.dummies)
    lines = [f"  integer(c_int) function {name}( &",
             f"       {args}) &",
             f"       bind(C, name='{name}') result(status)",
             "    integer(c_int), value, intent(in) :: lchnk"]
    lines += wrapper.declarations
    if wrapper.needs_pbuf:
        lines.append("    type(physics_buffer_desc), pointer :: pbuf(:)")
    lines += ["", "    status = 1_c_int", "    if (.not. chunk_ok(lchnk)) return"]
    if wrapper.needs_state:
        lines.append("    if (.not. associated(host_state)) return")
    if wrapper.needs_cam:
        lines.append("    if (.not. associated(host_cam_in(lchnk)%p)) return")
    if wrapper.needs_pbuf:
        lines += ["    if (.not. associated(host_pbuf2d)) return",
                  "    pbuf => pbuf_get_chunk(host_pbuf2d, lchnk)"]
    if wrapper.needs_rstate:
        lines.append("    status = 2_c_int")
    lines += wrapper.body + ["    status = 0_c_int", f"  end function {name}"]
    return "\n".join(lines)


def _view_cases() -> list[str]:
    out = []
    for name, (code, rank, expression, guard) in sorted(VIEWS.items(), key=lambda kv: kv[1][0]):
        out.append(f"    case (view_{name})")
        if guard:
            out.append(f"      if (.not. {guard}) return")
        out.append(f"      call view{rank}({expression}, ptr, ndims, extents)")
    return out


def render_module() -> str:
    nl = "\n"
    codes = nl.join(
        f"  integer(c_int), parameter, public :: view_{name} = {row[0]}"
        for name, row in sorted(VIEWS.items(), key=lambda kv: kv[1][0]))
    entries = (nl + nl).join(_entry(w) for w in WRAPPERS)
    return f'''! The calls a Python-driven radiation timestep makes into CAM.
!
! GENERATED by tools/generate_pi_cam_rad_handles.py.  Do not edit by hand;
! edit the generator.
!
! radiation_tend's arithmetic is lifted into pycam_rad_kernels.  What is left
! is calls that take a derived type -- physics_state, the physics buffer,
! cam_in_t, cam_out_t, physics_ptend, rrtmg_state_t -- and so cannot be
! promoted as direct kernels.  Each gets one bind(C) wrapper here, making the
! driver's call argument for argument.  The derived types stay in Fortran;
! Python gets zero-copy views of the numeric components by code.
!
! This module is an addition to the source tree.  It calls the oracle's own
! routines and replaces none of them.

module pycam_rad_handles

  use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int, c_int64_t, &
       c_loc, c_null_ptr, c_ptr
  use shr_kind_mod,     only: r8 => shr_kind_r8
  use ppgrid,           only: pcols, pver, pverp, begchunk, endchunk
  use parrrsw,          only: nbndsw
  use parrrtm,          only: nbndlw
  use physics_types,    only: physics_state, physics_ptend
  use physics_buffer,   only: physics_buffer_desc, pbuf_get_chunk
  use camsrfexch,       only: cam_in_t, cam_out_t
  use comsrf,           only: fsns, fsnt, flns, flnt, fsds
  use cam_history,      only: outfld, hist_fld_active
  use time_manager,     only: get_nstep, get_step_size, get_curr_calday
  use phys_grid,        only: get_rlat_all_p, get_rlon_all_p
  use rrtmg_state,      only: rrtmg_state_t, rrtmg_state_create, &
       rrtmg_state_update, rrtmg_state_destroy, num_rrtmg_levs
  use radsw,            only: rad_rrtmg_sw
  use radlw,            only: rad_rrtmg_lw
  use cloud_rad_props,  only: get_ice_optics_sw, get_liquid_optics_sw, &
       get_snow_optics_sw, ice_cloud_get_rad_props_lw, &
       liquid_cloud_get_rad_props_lw, snow_cloud_get_rad_props_lw
  use aer_rad_props,    only: aer_rad_props_sw, aer_rad_props_lw
  use rad_constituents, only: rad_cnst_out, rad_cnst_get_call_list, N_DIAG, &
       oldcldoptics, liqcldoptics, icecldoptics
  use radiation_data,   only: rad_data_write
  use radheat,          only: radheat_tend
  use tropopause,       only: tropopause_find, TROP_ALG_HYBSTOB, TROP_ALG_CLIMATE
  use radiation,        only: radiation_do, radiation_nextsw_cday

  implicit none
  private

  public :: pycam_rad_bind_hosts, pycam_rad_bind_chunk, python_owns_rad
  public :: rad_ptend, rad_net_flx

  ! Whether Python, rather than the Fortran driver, produced this step's
  ! tendency.  The resume half of tphysbc reads it.
  logical, save :: python_owns_rad = .false.

  ! The tendency radheat_tend fills and the net flux tphysbc keeps as a local,
  ! both of which must survive the return to Python and back.
  type(physics_ptend), allocatable, target, save :: rad_ptend(:)
  real(r8), allocatable, target, save :: rad_net_flx(:,:)

  ! The RRTMG state, alive only between create and destroy inside one chunk.
  type(rrtmg_state_t), pointer, save :: python_rstate => null()
  logical, save :: rstate_live = .false.

  ! Null pointers for the spectral-flux arguments.  spectralflux is .false. in
  ! the admitted configuration and refused at attach, so the driver passes
  ! these unassociated and so do we.
  real(r8), pointer, dimension(:,:,:) :: null_su => null(), null_sd => null()
  real(r8), pointer, dimension(:,:,:) :: null_lu => null(), null_ld => null()

  ! The hosts, bound from above so this module never uses the control layer.
  type(physics_state), pointer, save :: host_state(:) => null()
  type(physics_buffer_desc), pointer, save :: host_pbuf2d(:,:) => null()

  ! cam_in and cam_out are dummies of tphysbc, so one chunk's objects are
  ! bound by address at the stop rather than reached through module state.
  type :: cam_in_ref
    type(cam_in_t), pointer :: p => null()
  end type cam_in_ref
  type :: cam_out_ref
    type(cam_out_t), pointer :: p => null()
  end type cam_out_ref
  type(cam_in_ref), allocatable, save :: host_cam_in(:)
  type(cam_out_ref), allocatable, save :: host_cam_out(:)

{codes}

contains

  logical function chunk_ok(lchnk)
    integer(c_int), intent(in) :: lchnk
    chunk_ok = allocated(rad_ptend) .and. lchnk >= begchunk .and. lchnk <= endchunk
  end function chunk_ok

  subroutine pycam_rad_bind_hosts(state, pbuf2d)
    type(physics_state), pointer, intent(in) :: state(:)
    type(physics_buffer_desc), pointer, intent(in) :: pbuf2d(:,:)
    host_state => state
    host_pbuf2d => pbuf2d
    if (.not. allocated(rad_ptend)) allocate(rad_ptend(begchunk:endchunk))
    if (.not. allocated(rad_net_flx)) then
      allocate(rad_net_flx(pcols,begchunk:endchunk))
      rad_net_flx = 0._r8
    end if
    call ensure_chunk_refs()
  end subroutine pycam_rad_bind_hosts

  subroutine pycam_rad_bind_chunk(lchnk, chunk_cam_in, chunk_cam_out)
    ! Called from tphysbc's stop, where cam_in and cam_out are the chunk's own
    ! objects.  Only the address leaves; the objects stay where they are.
    !
    ! The stop runs before Python's first tend, so this cannot wait for
    ! pycam_rad_bind_hosts to make the storage: it allocates its own.  Gate
    ! R-B2 failed once on exactly that ordering, silently, because the guard
    ! here used to return instead.
    integer, intent(in) :: lchnk
    type(cam_in_t), intent(in), target :: chunk_cam_in
    type(cam_out_t), intent(inout), target :: chunk_cam_out
    if (lchnk < begchunk .or. lchnk > endchunk) return
    call ensure_chunk_refs()
    host_cam_in(lchnk)%p => chunk_cam_in
    host_cam_out(lchnk)%p => chunk_cam_out
  end subroutine pycam_rad_bind_chunk

  subroutine ensure_chunk_refs()
    if (.not. allocated(host_cam_in)) allocate(host_cam_in(begchunk:endchunk))
    if (.not. allocated(host_cam_out)) allocate(host_cam_out(begchunk:endchunk))
  end subroutine ensure_chunk_refs

  subroutine view1(field, ptr, ndims, extents)
    ! A TARGET dummy so c_loc is legal whatever the actual argument's
    ! attributes; no copy is made for a contiguous actual argument.
    real(r8), intent(in), target :: field(:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(field(1))
    ndims = 1_c_int
    extents(1) = int(size(field, 1), c_int64_t)
  end subroutine view1

  subroutine view2(field, ptr, ndims, extents)
    real(r8), intent(in), target :: field(:,:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(field(1,1))
    ndims = 2_c_int
    extents(1) = int(size(field, 1), c_int64_t)
    extents(2) = int(size(field, 2), c_int64_t)
  end subroutine view2

  ! ------------------------------------------------------------------ !
  ! Ownership, the clock, and the driver's own predicates
  ! ------------------------------------------------------------------ !

  integer(c_int) function pycam_rad_set_owner_v1(owns) &
       bind(C, name='pycam_rad_set_owner_v1') result(status)
    integer(c_int), value, intent(in) :: owns
    python_owns_rad = owns /= 0_c_int
    status = 0_c_int
  end function pycam_rad_set_owner_v1

  integer(c_int) function pycam_rad_nstep_v1() &
       bind(C, name='pycam_rad_nstep_v1') result(status)
    status = int(get_nstep(), c_int)
  end function pycam_rad_nstep_v1

  integer(c_int) function pycam_rad_dt_v1() &
       bind(C, name='pycam_rad_dt_v1') result(status)
    status = int(get_step_size(), c_int)
  end function pycam_rad_dt_v1

  integer(c_int) function pycam_rad_calday_v1(calday) &
       bind(C, name='pycam_rad_calday_v1') result(status)
    real(c_double), intent(out) :: calday
    calday = get_curr_calday()
    status = 0_c_int
  end function pycam_rad_calday_v1

  integer(c_int) function pycam_rad_do_v1(op, clat, clon) &
       bind(C, name='pycam_rad_do_v1') result(status)
    ! radiation_do is the driver's own predicate; Python branches on it rather
    ! than deriving the cadence a second time.  op: 1 shortwave, 2 longwave.
    integer(c_int), value, intent(in) :: op
    real(c_double), intent(out) :: clat, clon
    clat = 0._r8
    clon = 0._r8
    select case (op)
    case (1)
      status = merge(1_c_int, 0_c_int, radiation_do('sw'))
    case (2)
      status = merge(1_c_int, 0_c_int, radiation_do('lw'))
    case default
      status = -1_c_int
    end select
  end function pycam_rad_do_v1

  integer(c_int) function pycam_rad_latlon_v1(lchnk, ncol, clat, clon) &
       bind(C, name='pycam_rad_latlon_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk, ncol
    real(c_double), intent(inout) :: clat(pcols), clon(pcols)
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    call get_rlat_all_p(lchnk, ncol, clat)
    call get_rlon_all_p(lchnk, ncol, clon)
    status = 0_c_int
  end function pycam_rad_latlon_v1

  integer(c_int) function pycam_rad_options_v1(codes) &
       bind(C, name='pycam_rad_options_v1') result(status)
    ! The character and derived module state module_view cannot read, as
    ! integer codes: 1 oldcldoptics, 2 icecldoptics is 'mitchell',
    ! 3 liqcldoptics is 'gammadist', 4 how many radiation calls are active.
    integer(c_int), intent(out) :: codes(4)
    logical :: active_calls(0:N_DIAG)
    integer :: i
    codes(1) = merge(1_c_int, 0_c_int, oldcldoptics)
    codes(2) = merge(1_c_int, 0_c_int, trim(icecldoptics) == 'mitchell')
    codes(3) = merge(1_c_int, 0_c_int, trim(liqcldoptics) == 'gammadist')
    call rad_cnst_get_call_list(active_calls)
    codes(4) = 0_c_int
    do i = 0, N_DIAG
      if (active_calls(i)) codes(4) = codes(4) + 1_c_int
    end do
    status = 0_c_int
  end function pycam_rad_options_v1

  integer(c_int) function pycam_rad_hist_active_v1(name, name_len) &
       bind(C, name='pycam_rad_hist_active_v1') result(status)
    ! FSNR and FLNR guard two vertinterp loops.  Neither is on a history tape
    ! in the admitted configuration; Python refuses if that ever changes.
    integer(c_int), value, intent(in) :: name_len
    character(kind=c_char), intent(in) :: name(name_len)
    character(len=name_len) :: label
    integer :: i
    do i = 1, name_len
      label(i:i) = name(i)
    end do
    status = merge(1_c_int, 0_c_int, hist_fld_active(label))
  end function pycam_rad_hist_active_v1

  integer(c_int) function pycam_rad_view_v1(lchnk, code, ptr, ndims, extents) &
       bind(C, name='pycam_rad_view_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk, code
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_null_ptr
    ndims = 0_c_int
    extents = 0_c_int64_t
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    if (.not. associated(host_state)) return
    status = 2_c_int
    select case (code)
{nl.join(_view_cases())}
    case default
      status = 3_c_int
      return
    end select
    status = 0_c_int
  end function pycam_rad_view_v1

  ! ------------------------------------------------------------------ !
  ! The driver's calls, one wrapper each
  ! ------------------------------------------------------------------ !

{entries}

end module pycam_rad_handles
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = render_module()
    if arguments.check:
        current = MODULE.read_text() if MODULE.is_file() else ""
        if current != rendered:
            sys.stderr.write("".join(difflib.unified_diff(
                current.splitlines(keepends=True), rendered.splitlines(keepends=True),
                fromfile=f"{MODULE.name} (committed)", tofile=f"{MODULE.name} (generated)",
            ))[:4000])
            sys.stderr.write(f"\nstale: {MODULE.relative_to(REPO)}\n")
            return 1
        return 0
    MODULE.write_text(rendered)
    print(f"wrote {MODULE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
