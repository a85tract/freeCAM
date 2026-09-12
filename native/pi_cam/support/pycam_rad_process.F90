! The radiation process slot inside the image: a compiled plugin answers the computing branch
! of radiation_tend in the driver's place, with no Python in the step.
!
! This module is an addition to the source tree.  Control patch 0044 asks it, at the top of the
! driver's radiative branch (radiation.F90:875), whether a plugin is bound; if one is, this module
! hands the plugin what the driver has in hand -- the state's fields and constituents, the buffer's
! cloud fraction and the optics' inputs, the surface albedos and upward longwave, the zenith angle
! and the RRTMG state's gas profiles -- as pointer and extent tables, in the order Python's
! radiation_process.TABLE_INPUTS lists (a test keeps the two lists equal), takes the two heating
! rates and the ten fluxes back, writes them where the driver writes them, and writes the history
! fields of what it produced as the driver does.  The branch's other diagnostics have no source
! and are not written.  In shadow the plugin runs for its cost and the original branch answers.
! The plugin is a C function of the same interface the kernel hooks use (pycam_hooks).
module pycam_rad_process

  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_double, c_ptr, c_loc, c_funptr, &
                                         c_null_funptr, c_f_procpointer
  use shr_kind_mod,    only: r8 => shr_kind_r8
  use ppgrid,          only: pcols, pver, pverp
  use constituents,    only: pcnst
  use physics_types,   only: physics_state
  use physics_buffer,  only: physics_buffer_desc, pbuf_get_field, pbuf_get_index, pbuf_old_tim_idx
  use camsrfexch,      only: cam_in_t, cam_out_t
  use rrtmg_state,     only: rrtmg_state_t, rrtmg_state_create, rrtmg_state_update, rrtmg_state_destroy
  use cam_history,     only: outfld
  use physconst,       only: cpair
  use time_manager,    only: get_nstep, get_curr_calday
  use phys_grid,       only: get_rlat_all_p, get_rlon_all_p

  implicit none
  private
  public :: pycam_rad_process_answer, pycam_rad_process_bind_v1, pycam_rad_process_unbind_v1, &
            pycam_rad_process_counts_v1

  integer, parameter :: n_in = 46, n_out = 12
  !> the table's order, for the test that pins it to Python's radiation_process.TABLE_INPUTS
  character(len=16), parameter :: table_inputs(n_in) = [character(len=16) :: &
       'nstep', 'lchnk', 'ncol', 'calday', 'dosw', 'dolw', &
       'coszrs', 'clat', 'clon', &
       'state_t', 'state_pmid', 'state_pint', 'state_pdel', 'state_lnpint', 'state_lnpmid', 'state_q', &
       'cld', 'cldfsnow', 'dei', 'mu', 'lambdac', 'iciwp', 'iclwp', 'des', 'icswp', 'dgnumwet', 'qaerwat', &
       'cam_in_lwup', 'cam_in_asdir', 'cam_in_asdif', 'cam_in_aldir', 'cam_in_aldif', &
       'rstate_h2ovmr', 'rstate_o3vmr', 'rstate_co2vmr', 'rstate_ch4vmr', 'rstate_o2vmr', 'rstate_n2ovmr', &
       'rstate_cfc11vmr', 'rstate_cfc12vmr', 'rstate_cfc22vmr', 'rstate_ccl4vmr', &
       'rstate_pmidmb', 'rstate_pintmb', 'rstate_tlay', 'rstate_tlev']
  character(len=16), parameter :: table_outputs(n_out) = [character(len=16) :: &
       'qrs', 'qrl', 'fsns', 'fsnt', 'flns', 'flnt', 'fsds', 'sols', 'soll', 'solsd', 'solld', 'flwds']

  logical, save :: bound = .false.
  logical, save :: shadow = .false.
  logical, save :: indices_ready = .false.
  type(c_funptr), save :: plugin = c_null_funptr
  integer(c_int64_t), save :: calls = 0_c_int64_t, ticks = 0_c_int64_t, first_ticks = 0_c_int64_t

  ! the optics' buffer fields; an unregistered one (index -1) is handed as zeros
  integer, save :: cld_idx = -1, cldfsnow_idx = -1, dei_idx = -1, mu_idx = -1, lambda_idx = -1, &
                   iciwp_idx = -1, iclwp_idx = -1, des_idx = -1, icswp_idx = -1, &
                   dgnumwet_idx = -1, qaerwat_idx = -1

  ! scalars travel as one-element doubles; missing fields as zeros; the outputs as temporaries
  real(c_double), target, save :: s_nstep(1), s_lchnk(1), s_ncol(1), s_calday(1), s_dosw(1), s_dolw(1)
  real(r8), target, save :: clat(pcols), clon(pcols)
  real(r8), target, save :: zero2(pcols, pver) = 0.0_r8
  real(r8), target, save :: zero3(pcols, pver, 3) = 0.0_r8
  real(r8), target, save :: o_qrs(pcols, pver), o_qrl(pcols, pver)
  real(r8), target, save :: o_fsns(pcols), o_fsnt(pcols), o_flns(pcols), o_flnt(pcols), o_fsds(pcols), &
                            o_sols(pcols), o_soll(pcols), o_solsd(pcols), o_solld(pcols), o_flwds(pcols)
  real(r8), save :: ftem(pcols, pver)

  abstract interface
    integer(c_int) function plugin_interface(n_in, in_ptrs, in_shapes, n_out, out_ptrs, out_shapes) bind(C)
      import :: c_int, c_ptr, c_int64_t
      integer(c_int), value, intent(in) :: n_in, n_out
      type(c_ptr), intent(in) :: in_ptrs(*), out_ptrs(*)
      integer(c_int64_t), intent(in) :: in_shapes(*), out_shapes(*)
    end function plugin_interface
  end interface

contains

  subroutine resolve_indices()
    integer :: err
    if (indices_ready) return
    cld_idx      = pbuf_get_index('CLD', errcode=err)
    cldfsnow_idx = pbuf_get_index('CLDFSNOW', errcode=err)
    dei_idx      = pbuf_get_index('DEI', errcode=err)
    mu_idx       = pbuf_get_index('MU', errcode=err)
    lambda_idx   = pbuf_get_index('LAMBDAC', errcode=err)
    iciwp_idx    = pbuf_get_index('ICIWP', errcode=err)
    iclwp_idx    = pbuf_get_index('ICLWP', errcode=err)
    des_idx      = pbuf_get_index('DES', errcode=err)
    icswp_idx    = pbuf_get_index('ICSWP', errcode=err)
    dgnumwet_idx = pbuf_get_index('DGNUMWET', errcode=err)
    qaerwat_idx  = pbuf_get_index('QAERWAT', errcode=err)
    indices_ready = .true.
  end subroutine resolve_indices

  subroutine plane(pbuf, index, sliced, itim, ptr)
    ! a (pcols, pver) buffer field, at the older time sample when it has one; zeros when absent
    type(physics_buffer_desc), pointer :: pbuf(:)
    integer, intent(in) :: index, itim
    logical, intent(in) :: sliced
    real(r8), pointer :: ptr(:,:)
    if (index <= 0) then
      ptr => zero2
    else if (sliced) then
      call pbuf_get_field(pbuf, index, ptr, start=(/1, 1, itim/), kount=(/pcols, pver, 1/))
    else
      call pbuf_get_field(pbuf, index, ptr)
    end if
  end subroutine plane

  subroutine set2(slot, in_p, in_s, array)
    integer, intent(in) :: slot
    type(c_ptr), intent(inout) :: in_p(:)
    integer(c_int64_t), intent(inout) :: in_s(:,:)
    real(r8), target, intent(in) :: array(:,:)
    in_p(slot) = c_loc(array)
    in_s(:, slot) = (/ int(size(array, 1), c_int64_t), int(size(array, 2), c_int64_t), 0_c_int64_t /)
  end subroutine set2

  subroutine set1(slot, in_p, in_s, array)
    integer, intent(in) :: slot
    type(c_ptr), intent(inout) :: in_p(:)
    integer(c_int64_t), intent(inout) :: in_s(:,:)
    real(r8), target, intent(in) :: array(:)
    in_p(slot) = c_loc(array)
    in_s(:, slot) = (/ int(size(array, 1), c_int64_t), 0_c_int64_t, 0_c_int64_t /)
  end subroutine set1

  logical function pycam_rad_process_answer(state, pbuf, cam_in, cam_out, coszrs, dosw, dolw, &
       qrs, qrl, fsns, fsnt, flns, flnt, fsds) result(handled)
    ! Called at the top of the radiative branch.  .false. leaves the branch to the driver (no plugin,
    ! or a shadow plugin, which ran for its cost); .true. means the plugin answered and the
    ! driver skips the branch.
    type(physics_state), target, intent(in) :: state
    type(physics_buffer_desc), pointer :: pbuf(:)
    type(cam_in_t), target, intent(in) :: cam_in
    type(cam_out_t), target, intent(inout) :: cam_out
    real(r8), target, intent(in) :: coszrs(pcols)
    logical, intent(in) :: dosw, dolw
    real(r8), intent(inout) :: qrs(:,:), qrl(:,:)
    real(r8), intent(inout) :: fsns(pcols), fsnt(pcols), flns(pcols), flnt(pcols), fsds(pcols)

    type(c_ptr) :: in_p(n_in), out_p(n_out)
    integer(c_int64_t) :: in_s(3, n_in), out_s(3, n_out)
    real(r8), pointer :: cld(:,:), cldfsnow(:,:), dei(:,:), mu(:,:), lambdac(:,:), iciwp(:,:), &
                         iclwp(:,:), des(:,:), icswp(:,:)
    real(r8), pointer :: dgnumwet(:,:,:), qaerwat(:,:,:)
    type(rrtmg_state_t), pointer :: r_state
    procedure(plugin_interface), pointer :: call_plugin => null()
    integer(c_int) :: status
    integer(c_int64_t) :: t0, t1
    integer :: lchnk, ncol, itim, k

    handled = .false.
    if (.not. bound) return
    call resolve_indices()
    lchnk = state%lchnk
    ncol = state%ncol
    itim = pbuf_old_tim_idx()

    ! the scalars, the geometry
    s_nstep(1) = real(get_nstep(), c_double); s_lchnk(1) = real(lchnk, c_double); s_ncol(1) = real(ncol, c_double)
    s_calday(1) = real(get_curr_calday(), c_double)
    s_dosw(1) = merge(1.0_c_double, 0.0_c_double, dosw); s_dolw(1) = merge(1.0_c_double, 0.0_c_double, dolw)
    call get_rlat_all_p(lchnk, ncol, clat)
    call get_rlon_all_p(lchnk, ncol, clon)
    ! the buffer's cloud fraction and the optics' inputs
    call plane(pbuf, cld_idx, .true., itim, cld)
    call plane(pbuf, cldfsnow_idx, .true., itim, cldfsnow)
    call plane(pbuf, dei_idx, .false., itim, dei)
    call plane(pbuf, mu_idx, .false., itim, mu)
    call plane(pbuf, lambda_idx, .false., itim, lambdac)
    call plane(pbuf, iciwp_idx, .false., itim, iciwp)
    call plane(pbuf, iclwp_idx, .false., itim, iclwp)
    call plane(pbuf, des_idx, .false., itim, des)
    call plane(pbuf, icswp_idx, .false., itim, icswp)
    if (dgnumwet_idx > 0) then
      call pbuf_get_field(pbuf, dgnumwet_idx, dgnumwet)
    else
      dgnumwet => zero3
    end if
    if (qaerwat_idx > 0) then
      call pbuf_get_field(pbuf, qaerwat_idx, qaerwat)
    else
      qaerwat => zero3
    end if
    ! the RRTMG state: the gas profiles, as the driver builds them (radiation.F90:878, 1026)
    r_state => rrtmg_state_create(state, cam_in)
    call rrtmg_state_update(state, pbuf, 0, r_state)

    k = 0
    k = k + 1; call set1(k, in_p, in_s, s_nstep)
    k = k + 1; call set1(k, in_p, in_s, s_lchnk)
    k = k + 1; call set1(k, in_p, in_s, s_ncol)
    k = k + 1; call set1(k, in_p, in_s, s_calday)
    k = k + 1; call set1(k, in_p, in_s, s_dosw)
    k = k + 1; call set1(k, in_p, in_s, s_dolw)
    k = k + 1; call set1(k, in_p, in_s, coszrs)
    k = k + 1; call set1(k, in_p, in_s, clat)
    k = k + 1; call set1(k, in_p, in_s, clon)
    k = k + 1; call set2(k, in_p, in_s, state%t)
    k = k + 1; call set2(k, in_p, in_s, state%pmid)
    k = k + 1; call set2(k, in_p, in_s, state%pint)
    k = k + 1; call set2(k, in_p, in_s, state%pdel)
    k = k + 1; call set2(k, in_p, in_s, state%lnpint)
    k = k + 1; call set2(k, in_p, in_s, state%lnpmid)
    k = k + 1
    in_p(k) = c_loc(state%q)
    in_s(:, k) = (/ int(size(state%q, 1), c_int64_t), int(size(state%q, 2), c_int64_t), int(size(state%q, 3), c_int64_t) /)
    k = k + 1; call set2(k, in_p, in_s, cld)
    k = k + 1; call set2(k, in_p, in_s, cldfsnow)
    k = k + 1; call set2(k, in_p, in_s, dei)
    k = k + 1; call set2(k, in_p, in_s, mu)
    k = k + 1; call set2(k, in_p, in_s, lambdac)
    k = k + 1; call set2(k, in_p, in_s, iciwp)
    k = k + 1; call set2(k, in_p, in_s, iclwp)
    k = k + 1; call set2(k, in_p, in_s, des)
    k = k + 1; call set2(k, in_p, in_s, icswp)
    k = k + 1
    in_p(k) = c_loc(dgnumwet)
    in_s(:, k) = (/ int(size(dgnumwet, 1), c_int64_t), int(size(dgnumwet, 2), c_int64_t), int(size(dgnumwet, 3), c_int64_t) /)
    k = k + 1
    in_p(k) = c_loc(qaerwat)
    in_s(:, k) = (/ int(size(qaerwat, 1), c_int64_t), int(size(qaerwat, 2), c_int64_t), int(size(qaerwat, 3), c_int64_t) /)
    k = k + 1; call set1(k, in_p, in_s, cam_in%lwup)
    k = k + 1; call set1(k, in_p, in_s, cam_in%asdir)
    k = k + 1; call set1(k, in_p, in_s, cam_in%asdif)
    k = k + 1; call set1(k, in_p, in_s, cam_in%aldir)
    k = k + 1; call set1(k, in_p, in_s, cam_in%aldif)
    k = k + 1; call set2(k, in_p, in_s, r_state%h2ovmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%o3vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%co2vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%ch4vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%o2vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%n2ovmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%cfc11vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%cfc12vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%cfc22vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%ccl4vmr)
    k = k + 1; call set2(k, in_p, in_s, r_state%pmidmb)
    k = k + 1; call set2(k, in_p, in_s, r_state%pintmb)
    k = k + 1; call set2(k, in_p, in_s, r_state%tlay)
    k = k + 1; call set2(k, in_p, in_s, r_state%tlev)

    o_qrs = 0.0_r8; o_qrl = 0.0_r8
    o_fsns = 0.0_r8; o_fsnt = 0.0_r8; o_flns = 0.0_r8; o_flnt = 0.0_r8; o_fsds = 0.0_r8
    o_sols = 0.0_r8; o_soll = 0.0_r8; o_solsd = 0.0_r8; o_solld = 0.0_r8; o_flwds = 0.0_r8
    k = 0
    k = k + 1; call set2(k, out_p, out_s, o_qrs)
    k = k + 1; call set2(k, out_p, out_s, o_qrl)
    k = k + 1; call set1(k, out_p, out_s, o_fsns)
    k = k + 1; call set1(k, out_p, out_s, o_fsnt)
    k = k + 1; call set1(k, out_p, out_s, o_flns)
    k = k + 1; call set1(k, out_p, out_s, o_flnt)
    k = k + 1; call set1(k, out_p, out_s, o_fsds)
    k = k + 1; call set1(k, out_p, out_s, o_sols)
    k = k + 1; call set1(k, out_p, out_s, o_soll)
    k = k + 1; call set1(k, out_p, out_s, o_solsd)
    k = k + 1; call set1(k, out_p, out_s, o_solld)
    k = k + 1; call set1(k, out_p, out_s, o_flwds)

    call system_clock(t0)
    call c_f_procpointer(plugin, call_plugin)
    status = call_plugin(int(n_in, c_int), in_p, in_s, int(n_out, c_int), out_p, out_s)
    call system_clock(t1)
    if (calls == 0_c_int64_t) first_ticks = t1 - t0
    ticks = ticks + (t1 - t0)
    calls = calls + 1_c_int64_t
    call rrtmg_state_destroy(r_state)
    if (status /= 0_c_int) error stop 'pycam_rad_process: the bound plugin returned a non-zero status'
    if (shadow) return

    ! the outputs, where the driver's branch leaves them (radiation.F90:1034-1051, 1148-1154)
    qrs(:ncol, :) = o_qrs(:ncol, :)
    qrl(:ncol, :) = o_qrl(:ncol, :)
    fsns(:ncol) = o_fsns(:ncol); fsnt(:ncol) = o_fsnt(:ncol); flns(:ncol) = o_flns(:ncol)
    flnt(:ncol) = o_flnt(:ncol); fsds(:ncol) = o_fsds(:ncol)
    cam_out%sols(:ncol) = o_sols(:ncol); cam_out%soll(:ncol) = o_soll(:ncol)
    cam_out%solsd(:ncol) = o_solsd(:ncol); cam_out%solld(:ncol) = o_solld(:ncol)
    cam_out%flwds(:ncol) = o_flwds(:ncol)
    ! the history of what the plugin produced, as the driver writes it (1061-1090, 1170-1187)
    if (dosw) then
      ftem(:ncol, :pver) = qrs(:ncol, :pver) / cpair
      call outfld('QRS', ftem, pcols, lchnk)
      call outfld('FSDS', fsds, pcols, lchnk)
      call outfld('FSNT', fsnt, pcols, lchnk)
      call outfld('FSNS', fsns, pcols, lchnk)
      call outfld('SOLS', cam_out%sols, pcols, lchnk)
      call outfld('SOLL', cam_out%soll, pcols, lchnk)
      call outfld('SOLSD', cam_out%solsd, pcols, lchnk)
      call outfld('SOLLD', cam_out%solld, pcols, lchnk)
    end if
    if (dolw) then
      call outfld('QRL', qrl(:ncol, :) / cpair, ncol, lchnk)
      call outfld('FLNT', flnt, pcols, lchnk)
      call outfld('FLNS', flns, pcols, lchnk)
      call outfld('FLDS', cam_out%flwds, pcols, lchnk)
    end if
    handled = .true.
  end function pycam_rad_process_answer

  integer(c_int) function pycam_rad_process_bind_v1(funptr, shadow_flag) &
       bind(C, name='pycam_rad_process_bind_v1') result(status)
    ! bind a compiled plugin (the address of a C function of plugin_interface) at the radiation
    ! process slot; shadow_flag /= 0 runs it for its cost while the driver's branch answers
    type(c_funptr), value, intent(in) :: funptr
    integer(c_int), value, intent(in) :: shadow_flag
    plugin = funptr
    shadow = shadow_flag /= 0_c_int
    bound = .true.
    status = 0_c_int
  end function pycam_rad_process_bind_v1

  integer(c_int) function pycam_rad_process_unbind_v1() bind(C, name='pycam_rad_process_unbind_v1') result(status)
    bound = .false.; shadow = .false.; plugin = c_null_funptr
    status = 0_c_int
  end function pycam_rad_process_unbind_v1

  integer(c_int) function pycam_rad_process_counts_v1(calls_out, seconds, first_seconds) &
       bind(C, name='pycam_rad_process_counts_v1') result(status)
    ! this rank's plugin calls, the wall seconds inside them, and the first call alone
    integer(c_int64_t), intent(out) :: calls_out
    real(c_double), intent(out) :: seconds, first_seconds
    integer(c_int64_t) :: rate
    call system_clock(count_rate=rate)
    calls_out = calls
    seconds = real(ticks, c_double) / real(rate, c_double)
    first_seconds = real(first_ticks, c_double) / real(rate, c_double)
    status = 0_c_int
  end function pycam_rad_process_counts_v1

end module pycam_rad_process
