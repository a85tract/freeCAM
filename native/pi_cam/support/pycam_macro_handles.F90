! Handles on what macrop_driver_tend keeps in derived types.
!
! A Python-driven macrophysics timestep needs the same working storage the
! Fortran driver declares as locals -- a copy of the state, two tendency
! records, the detrainment integrals, the water-tracer rate matrix -- and
! needs to hand that storage to the same routines the driver calls on it:
! physics_state_copy, physics_ptend_init, physics_ptend_sum, physics_update,
! cldfrc, wtrc_apply_rates, outfld.  None of those can be reached from Python
! directly: they take derived types, or the physics buffer, or a Fortran
! character.
!
! So the storage lives here, one slot per chunk of this rank, and each
! routine gets a bind(C) entry that takes the chunk index and does the call.
! Python never holds a derived type; it holds zero-copy views of the numeric
! components (`pycam_macro_view_v1`) and asks for the calls by name.  The
! allocation pattern is the driver's own: physics_state_copy and
! physics_ptend_init allocate, physics_update and physics_state_dealloc free,
! so a view is valid only between the call that allocates it and the call
! that frees it -- exactly as the Fortran local would have been.
!
! This module is an addition to the source tree.  It uses nothing from
! physpkg or cam_comp (both use it), and no production object is changed to
! call it.

module pycam_macro_handles

  use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int, c_int32_t, &
       c_int64_t, c_loc, c_null_ptr, c_ptr
  use shr_kind_mod,   only: r8 => shr_kind_r8
  use ppgrid,         only: pcols, pver, pverp, begchunk, endchunk
  use constituents,   only: pcnst
  use water_types,    only: pwtype
  use physics_types,  only: physics_state, physics_ptend, physics_state_copy, &
       physics_state_dealloc, physics_ptend_init, physics_ptend_sum, physics_update
  use physics_buffer, only: physics_buffer_desc, pbuf_get_chunk
  use cloud_fraction, only: cldfrc
  use water_tracers,  only: wtrc_apply_rates
  use cam_history,    only: outfld

  implicit none
  private

  public :: pycam_macro_bind_hosts, python_owns_tend
  public :: macro_ptend, macro_det_s, macro_det_ice

  ! Set by Python once its process is installed.  tphysbc's resume stage reads
  ! it: true means the tendencies are waiting in macro_ptend, false means
  ! nothing ran and the original Fortran driver has to.
  logical, save :: python_owns_tend = .false.

  ! The driver's locals, one per chunk.
  type(physics_state), allocatable, target, save :: macro_state_loc(:)
  type(physics_ptend), allocatable, target, save :: macro_ptend_loc(:)
  type(physics_ptend), allocatable, target, save :: macro_ptend(:)
  real(r8), allocatable, target, save :: macro_det_s(:,:)
  real(r8), allocatable, target, save :: macro_det_ice(:,:)
  real(r8), allocatable, target, save :: macro_process_rates(:,:,:,:,:,:)
  ! physics_state's array components are allocatable in the oracle build and
  ! pointers in the Python-owned-state build, so neither ALLOCATED nor
  ! ASSOCIATED is portable across the two.  Liveness is tracked here instead:
  ! set by the copy, cleared by the deallocation, as the driver's local was.
  logical, allocatable, save :: state_live(:)

  ! What the driver reached through its caller.
  type(physics_state), pointer, save :: host_state(:) => null()
  type(physics_buffer_desc), pointer, save :: host_pbuf2d(:,:) => null()

  ! View codes: which array pycam_macro_view_v1 hands out.  Python mirrors
  ! this table; a test keeps the two in step.
  integer(c_int), parameter, public :: view_state_t = 1
  integer(c_int), parameter, public :: view_state_q = 2
  integer(c_int), parameter, public :: view_state_pmid = 3
  integer(c_int), parameter, public :: view_state_pdel = 4
  integer(c_int), parameter, public :: view_state_pint = 5
  integer(c_int), parameter, public :: view_state_omega = 6
  integer(c_int), parameter, public :: view_state_phis = 7
  integer(c_int), parameter, public :: view_ptend_loc_s = 11
  integer(c_int), parameter, public :: view_ptend_loc_q = 12
  integer(c_int), parameter, public :: view_ptend_s = 21
  integer(c_int), parameter, public :: view_ptend_q = 22
  integer(c_int), parameter, public :: view_det_s = 31
  integer(c_int), parameter, public :: view_det_ice = 32
  integer(c_int), parameter, public :: view_process_rates = 33

  ! Which tendency record pycam_macro_ptend_init_v1 initialises.
  integer(c_int), parameter, public :: record_ptend_loc = 1
  integer(c_int), parameter, public :: record_ptend = 2

contains

  ! ------------------------------------------------------------------ !
  ! Host binding: called once from cam_comp, where the state array and
  ! the physics buffer live.
  ! ------------------------------------------------------------------ !
  subroutine pycam_macro_bind_hosts(state, pbuf2d)
    type(physics_state), pointer :: state(:)
    type(physics_buffer_desc), pointer :: pbuf2d(:,:)
    host_state => state
    host_pbuf2d => pbuf2d
    call ensure_allocated()
  end subroutine pycam_macro_bind_hosts

  subroutine ensure_allocated()
    if (allocated(macro_state_loc)) return
    allocate(macro_state_loc(begchunk:endchunk))
    allocate(macro_ptend_loc(begchunk:endchunk))
    allocate(macro_ptend(begchunk:endchunk))
    allocate(macro_det_s(pcols, begchunk:endchunk))
    allocate(macro_det_ice(pcols, begchunk:endchunk))
    allocate(macro_process_rates(pcols, pver, pwtype, pwtype, pwtype, begchunk:endchunk))
    allocate(state_live(begchunk:endchunk))
    state_live = .false.
    macro_det_s = 0._r8
    macro_det_ice = 0._r8
    macro_process_rates = 0._r8
  end subroutine ensure_allocated

  logical function chunk_ok(lchnk)
    integer(c_int), intent(in) :: lchnk
    chunk_ok = associated(host_state) .and. lchnk >= begchunk .and. lchnk <= endchunk
  end function chunk_ok

  ! ------------------------------------------------------------------ !
  ! Ownership
  ! ------------------------------------------------------------------ !
  integer(c_int) function pycam_macro_set_owner_v1(flag) &
       bind(C, name='pycam_macro_set_owner_v1') result(status)
    integer(c_int), value, intent(in) :: flag
    python_owns_tend = flag /= 0_c_int
    status = 0_c_int
  end function pycam_macro_set_owner_v1

  integer(c_int) function pycam_macro_owner_v1() &
       bind(C, name='pycam_macro_owner_v1') result(flag)
    flag = merge(1_c_int, 0_c_int, python_owns_tend)
  end function pycam_macro_owner_v1

  ! ------------------------------------------------------------------ !
  ! physics_state: the driver's local copy
  ! ------------------------------------------------------------------ !
  integer(c_int) function pycam_macro_state_copy_v1(lchnk) &
       bind(C, name='pycam_macro_state_copy_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    call physics_state_copy(host_state(lchnk), macro_state_loc(lchnk))
    state_live(lchnk) = .true.
    status = 0_c_int
  end function pycam_macro_state_copy_v1

  integer(c_int) function pycam_macro_state_dealloc_v1(lchnk) &
       bind(C, name='pycam_macro_state_dealloc_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    if (.not. state_live(lchnk)) then
       status = 2_c_int
       return
    end if
    call physics_state_dealloc(macro_state_loc(lchnk))
    state_live(lchnk) = .false.
    status = 0_c_int
  end function pycam_macro_state_dealloc_v1

  ! ------------------------------------------------------------------ !
  ! physics_ptend: the two tendency records
  ! ------------------------------------------------------------------ !
  integer(c_int) function pycam_macro_ptend_init_v1(lchnk, which, name, name_len, &
       with_flags, ls, lq) bind(C, name='pycam_macro_ptend_init_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk, which, name_len, with_flags, ls
    character(kind=c_char), intent(in) :: name(*)
    integer(c_int32_t), intent(in) :: lq(pcnst)
    character(len=64) :: fname
    logical :: lq_flags(pcnst)
    integer :: i
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    if (name_len < 1 .or. name_len > len(fname)) then
       status = 2_c_int
       return
    end if
    fname = ''
    do i = 1, name_len
       fname(i:i) = name(i)
    end do
    ! The driver calls this in two forms: with ls and lq, or with the name
    ! alone.  physics_ptend_init treats an absent flag differently from a
    ! false one, so the form has to be reproduced, not approximated.
    select case (which)
    case (record_ptend_loc)
       if (with_flags /= 0_c_int) then
          lq_flags = lq /= 0_c_int32_t
          call physics_ptend_init(macro_ptend_loc(lchnk), host_state(lchnk)%psetcols, &
               fname(1:name_len), ls=(ls /= 0_c_int), lq=lq_flags)
       else
          call physics_ptend_init(macro_ptend_loc(lchnk), host_state(lchnk)%psetcols, &
               fname(1:name_len))
       end if
    case (record_ptend)
       if (with_flags /= 0_c_int) then
          lq_flags = lq /= 0_c_int32_t
          call physics_ptend_init(macro_ptend(lchnk), host_state(lchnk)%psetcols, &
               fname(1:name_len), ls=(ls /= 0_c_int), lq=lq_flags)
       else
          call physics_ptend_init(macro_ptend(lchnk), host_state(lchnk)%psetcols, &
               fname(1:name_len))
       end if
    case default
       status = 3_c_int
       return
    end select
    status = 0_c_int
  end function pycam_macro_ptend_init_v1

  integer(c_int) function pycam_macro_ptend_sum_v1(lchnk, ncol) &
       bind(C, name='pycam_macro_ptend_sum_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk, ncol
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    call physics_ptend_sum(macro_ptend_loc(lchnk), macro_ptend(lchnk), ncol)
    status = 0_c_int
  end function pycam_macro_ptend_sum_v1

  integer(c_int) function pycam_macro_update_v1(lchnk, dt) &
       bind(C, name='pycam_macro_update_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk
    real(c_double), value, intent(in) :: dt
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    ! physics_update frees ptend at its end; every view of macro_ptend_loc
    ! is dead after this call, as the driver's would have been.
    call physics_update(macro_state_loc(lchnk), macro_ptend_loc(lchnk), dt)
    status = 0_c_int
  end function pycam_macro_update_v1

  ! ------------------------------------------------------------------ !
  ! Views: the address of one numeric component, never a copy
  ! ------------------------------------------------------------------ !
  integer(c_int) function pycam_macro_view_v1(lchnk, code, ptr, ndims, extents) &
       bind(C, name='pycam_macro_view_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk, code
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_null_ptr
    ndims = 0_c_int
    extents = 0_c_int64_t
    status = 1_c_int
    if (.not. chunk_ok(lchnk)) return
    status = 2_c_int
    select case (code)
    case (view_state_t)
       if (.not. state_live(lchnk)) return
       call view2(macro_state_loc(lchnk)%t, ptr, ndims, extents)
    case (view_state_q)
       if (.not. state_live(lchnk)) return
       call view3(macro_state_loc(lchnk)%q, ptr, ndims, extents)
    case (view_state_pmid)
       if (.not. state_live(lchnk)) return
       call view2(macro_state_loc(lchnk)%pmid, ptr, ndims, extents)
    case (view_state_pdel)
       if (.not. state_live(lchnk)) return
       call view2(macro_state_loc(lchnk)%pdel, ptr, ndims, extents)
    case (view_state_pint)
       if (.not. state_live(lchnk)) return
       call view2(macro_state_loc(lchnk)%pint, ptr, ndims, extents)
    case (view_state_omega)
       if (.not. state_live(lchnk)) return
       call view2(macro_state_loc(lchnk)%omega, ptr, ndims, extents)
    case (view_state_phis)
       if (.not. state_live(lchnk)) return
       call view1(macro_state_loc(lchnk)%phis, ptr, ndims, extents)
    case (view_ptend_loc_s)
       if (.not. allocated(macro_ptend_loc(lchnk)%s)) return
       call view2(macro_ptend_loc(lchnk)%s, ptr, ndims, extents)
    case (view_ptend_loc_q)
       if (.not. allocated(macro_ptend_loc(lchnk)%q)) return
       call view3(macro_ptend_loc(lchnk)%q, ptr, ndims, extents)
    case (view_ptend_s)
       if (.not. allocated(macro_ptend(lchnk)%s)) return
       call view2(macro_ptend(lchnk)%s, ptr, ndims, extents)
    case (view_ptend_q)
       if (.not. allocated(macro_ptend(lchnk)%q)) return
       call view3(macro_ptend(lchnk)%q, ptr, ndims, extents)
    case (view_det_s)
       call view1(macro_det_s(:,lchnk), ptr, ndims, extents)
    case (view_det_ice)
       call view1(macro_det_ice(:,lchnk), ptr, ndims, extents)
    case (view_process_rates)
       call view5(macro_process_rates(:,:,:,:,:,lchnk), ptr, ndims, extents)
    case default
       status = 3_c_int
       return
    end select
    status = 0_c_int
  end function pycam_macro_view_v1

  subroutine view1(array, ptr, ndims, extents)
    real(r8), target, intent(in) :: array(:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(array(1))
    ndims = 1_c_int
    extents(1) = int(size(array, 1), c_int64_t)
  end subroutine view1

  subroutine view2(array, ptr, ndims, extents)
    real(r8), target, intent(in) :: array(:,:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(array(1,1))
    ndims = 2_c_int
    extents(1) = int(size(array, 1), c_int64_t)
    extents(2) = int(size(array, 2), c_int64_t)
  end subroutine view2

  subroutine view3(array, ptr, ndims, extents)
    real(r8), target, intent(in) :: array(:,:,:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(array(1,1,1))
    ndims = 3_c_int
    extents(1) = int(size(array, 1), c_int64_t)
    extents(2) = int(size(array, 2), c_int64_t)
    extents(3) = int(size(array, 3), c_int64_t)
  end subroutine view3

  subroutine view5(array, ptr, ndims, extents)
    real(r8), target, intent(in) :: array(:,:,:,:,:)
    type(c_ptr), intent(out) :: ptr
    integer(c_int), intent(out) :: ndims
    integer(c_int64_t), intent(out) :: extents(5)
    ptr = c_loc(array(1,1,1,1,1))
    ndims = 5_c_int
    extents(1) = int(size(array, 1), c_int64_t)
    extents(2) = int(size(array, 2), c_int64_t)
    extents(3) = int(size(array, 3), c_int64_t)
    extents(4) = int(size(array, 4), c_int64_t)
    extents(5) = int(size(array, 5), c_int64_t)
  end subroutine view5

  ! ------------------------------------------------------------------ !
  ! cldfrc: takes the physics buffer, so it cannot be a direct kernel
  ! ------------------------------------------------------------------ !
  integer(c_int) function pycam_macro_cldfrc_v1(lchnk, ncol, &
       pmid, temp, q, omga, phis, shfrc, use_shfrc, cloud, rhcloud, clc, pdel, &
       cmfmc, cmfmc2, landfrac, snowh, concld, cldst, ts, sst, ps, zdu, ocnfrac, &
       rhu00, cldice, icecldf, liqcldf, relhum, dindex) &
       bind(C, name='pycam_macro_cldfrc_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk, ncol, use_shfrc, dindex
    real(c_double), intent(in)    :: pmid(pcols,pver), temp(pcols,pver), q(pcols,pver)
    real(c_double), intent(in)    :: omga(pcols,pver), phis(pcols), shfrc(pcols,pver)
    real(c_double), intent(inout) :: cloud(pcols,pver), rhcloud(pcols,pver), clc(pcols)
    real(c_double), intent(in)    :: pdel(pcols,pver), cmfmc(pcols,pverp), cmfmc2(pcols,pverp)
    real(c_double), intent(in)    :: landfrac(pcols), snowh(pcols)
    real(c_double), intent(inout) :: concld(pcols,pver), cldst(pcols,pver)
    real(c_double), intent(in)    :: ts(pcols), sst(pcols), ps(pcols), zdu(pcols,pver)
    real(c_double), intent(in)    :: ocnfrac(pcols)
    real(c_double), intent(inout) :: rhu00(pcols,pver)
    real(c_double), intent(in)    :: cldice(pcols,pver)
    real(c_double), intent(inout) :: icecldf(pcols,pver), liqcldf(pcols,pver), relhum(pcols,pver)
    type(physics_buffer_desc), pointer :: pbuf(:)
    status = 1_c_int
    if (.not. chunk_ok(lchnk) .or. .not. associated(host_pbuf2d)) return
    pbuf => pbuf_get_chunk(host_pbuf2d, lchnk)
    call cldfrc(lchnk, ncol, pbuf, &
         pmid, temp, q, omga, phis, shfrc, use_shfrc /= 0_c_int, &
         cloud, rhcloud, clc, pdel, &
         cmfmc, cmfmc2, landfrac, snowh, concld, cldst, &
         ts, sst, ps, zdu, ocnfrac, rhu00, &
         cldice, icecldf, liqcldf, relhum, dindex)
    status = 0_c_int
  end function pycam_macro_cldfrc_v1

  ! ------------------------------------------------------------------ !
  ! wtrc_apply_rates: derived types, the buffer, and the rate matrix
  ! ------------------------------------------------------------------ !
  integer(c_int) function pycam_macro_wtrc_apply_v1(lchnk, top_lev, dt, prelat) &
       bind(C, name='pycam_macro_wtrc_apply_v1') result(status)
    integer(c_int), value, intent(in) :: lchnk, top_lev
    real(c_double), value, intent(in) :: dt
    real(c_double), intent(in) :: prelat(pcols,pver)
    type(physics_buffer_desc), pointer :: pbuf(:)
    status = 1_c_int
    if (.not. chunk_ok(lchnk) .or. .not. associated(host_pbuf2d)) return
    if (.not. state_live(lchnk)) then
       status = 2_c_int
       return
    end if
    pbuf => pbuf_get_chunk(host_pbuf2d, lchnk)
    ! The driver's exact form: micro=.false., pre_rates and prelat only.
    call wtrc_apply_rates(macro_state_loc(lchnk), macro_ptend_loc(lchnk), pbuf, top_lev, dt, &
         .false., pre_rates=macro_process_rates(:,:,:,:,:,lchnk), prelat=prelat)
    status = 0_c_int
  end function pycam_macro_wtrc_apply_v1

  ! ------------------------------------------------------------------ !
  ! outfld: a Fortran character and an assumed-size array
  ! ------------------------------------------------------------------ !
  integer(c_int) function pycam_outfld_v1(name, name_len, field, idim, lchnk) &
       bind(C, name='pycam_outfld_v1') result(status)
    character(kind=c_char), intent(in) :: name(*)
    integer(c_int), value, intent(in) :: name_len, idim, lchnk
    real(c_double), intent(in) :: field(idim,*)
    character(len=32) :: fname
    integer :: i
    status = 1_c_int
    if (name_len < 1 .or. name_len > len(fname) .or. idim < 1) return
    fname = ''
    do i = 1, name_len
       fname(i:i) = name(i)
    end do
    call outfld(fname(1:name_len), field, idim, lchnk)
    status = 0_c_int
  end function pycam_outfld_v1

end module pycam_macro_handles
