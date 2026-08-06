module pycam_pi_cam_adapter
  use, intrinsic :: iso_c_binding, only: c_char, c_f_pointer, c_int, &
       c_int8_t, c_int32_t, c_int64_t, c_null_char, c_ptr
  use shr_kind_mod, only: r8 => shr_kind_r8, cs => shr_kind_cs
  use cam_comp, only: cam_init, cam_final, cam_run1, cam_run2, cam_run3, &
       cam_run4, cam_run1_dynamics, &
       cam_phys_run1_prepare, cam_phys_run1_scheme_action, &
       cam_phys_run2_prepare, cam_phys_run2_scheme_action, &
       cam_phys_run2_finish, cam_run2_dynamics, &
       cam_run4_history, cam_run4_restart, cam_run4_finish
  use camsrfexch, only: cam_in_t, cam_out_t
  use atm_import_export, only: atm_import, atm_export
  use atm_comp_mct, only: atm_python_post_cam_init_mct
  use atm_comp_mct, only: atm_python_write_srfrest_mct
  use cam_cpl_indices, only: cam_cpl_indices_set
  use cam_instance, only: cam_instance_init
  use cam_control_mod, only: nsrest, adiabatic, ideal_phys, aqua_planet
  use cam_control_mod, only: eccen, obliqr, lambm0, mvelpp
  use filenames, only: caseid
  use runtime_opts, only: read_namelist
  use scamMod, only: single_column, scmlat, scmlon
  use spmd_utils, only: spmdinit
  use time_manager, only: timemgr_init, advance_timestep, get_curr_date, get_nstep
  use shr_pio_mod, only: shr_pio_init1, shr_pio_init2, shr_pio_finalize
  use seq_comm_mct, only: seq_comm_init, seq_comm_setptrs, ATMID, GLOID
  use seq_flds_mod, only: seq_flds_set
  use shr_orb_mod, only: shr_orb_params, SHR_ORB_UNDEF_REAL
  use ESMF, only: ESMF_Initialize
  use water_tracer_vars, only: wtrc_srf_bucket_mode
  implicit none
  private

  public :: pycam_pi_cam_initialize_v1
  public :: pycam_pi_cam_action_v1
  public :: pycam_pi_cam_finalize_v1

  type(cam_in_t), pointer, save :: cam_in(:) => null()
  type(cam_out_t), pointer, save :: cam_out(:) => null()
  logical, save :: initialized = .false.
  logical, save :: finalized = .false.
  integer, save :: configured_stop_n = 0

  interface
     subroutine pycam_pi_cam_set_fp_environment_v1() bind(C, &
          name='pycam_pi_cam_set_fp_environment_v1')
     end subroutine pycam_pi_cam_set_fp_environment_v1
  end interface

contains

  subroutine clear_error(errmsg, errmsg_len)
    character(kind=c_char), intent(out) :: errmsg(*)
    integer(c_int), value, intent(in) :: errmsg_len
    integer :: index
    do index = 1, max(1, int(errmsg_len))
       errmsg(index) = c_null_char
    end do
  end subroutine clear_error

  integer(c_int) function pycam_pi_cam_initialize_v1(action_id, nfields, &
       pointers, ndims, shapes, max_rank, fortran_comm, errmsg, errmsg_len) &
       bind(C, name='pycam_pi_cam_initialize_v1') result(status)
    integer(c_int), value, intent(in) :: action_id, nfields, max_rank
    integer(c_int), value, intent(in) :: fortran_comm, errmsg_len
    type(c_ptr), intent(in) :: pointers(*)
    integer(c_int32_t), intent(in) :: ndims(*)
    integer(c_int64_t), intent(in) :: shapes(*)
    character(kind=c_char), intent(out) :: errmsg(*)
    integer :: global_comm, atm_comm, rank, ierr, local_surface_columns, chunk
    integer :: comp_id(1), comp_comm(1), comp_comm_iam(1)
    logical :: comp_iamin(1)
    character(len=8) :: comp_name(1)
    character(len=cs) :: calendar
    integer(c_int64_t), pointer :: stop_steps
    integer(c_int32_t), pointer :: orbital_year
    integer(c_int8_t), pointer :: case_name_bytes(:)
    integer :: index
    real(r8) :: orb_obliq, orb_mvelp

    call pycam_pi_cam_set_fp_environment_v1()
    call clear_error(errmsg, errmsg_len)
    status = 0_c_int
    if (initialized .or. finalized) then
       status = 1_c_int
       return
    endif
    if (action_id /= 0 .or. nfields /= 3 .or. ndims(1) /= 0 .or. &
         ndims(2) /= 1 .or. ndims(3) /= 0) then
       status = 2_c_int
       return
    endif
    call c_f_pointer(pointers(1), stop_steps)
    call c_f_pointer(pointers(2), case_name_bytes, (/ int(shapes(2)) /))
    call c_f_pointer(pointers(3), orbital_year)
    configured_stop_n = int(stop_steps)
    if (configured_stop_n < 1) then
       status = 4_c_int
       return
    endif
    caseid = ' '
    do index = 1, min(size(case_name_bytes), len(caseid) - 1)
       if (case_name_bytes(index) == 0_c_int8_t) exit
       caseid(index:index) = achar(int(case_name_bytes(index)))
    end do
    if (len_trim(caseid) == 0) then
       status = 5_c_int
       return
    endif
    orb_obliq = SHR_ORB_UNDEF_REAL
    eccen = SHR_ORB_UNDEF_REAL
    orb_mvelp = SHR_ORB_UNDEF_REAL
    call shr_orb_params(int(orbital_year), eccen, orb_obliq, orb_mvelp, &
         obliqr, lambm0, mvelpp, .false.)
    global_comm = fortran_comm
    call MPI_Comm_rank(global_comm, rank, ierr)
    if (ierr /= 0) then
       status = 3_c_int
       return
    endif

    ! CAM's field-index table is populated by the CESM driver before the
    ! atmosphere component starts.  The Python driver replaces that outer
    ! driver, so reproduce only the two metadata initializers needed by CAM:
    ! communicator layout first, then the exact drv_in field contract.
    call shr_pio_init1(1, 'drv_in', global_comm)
    call seq_comm_init(global_comm, 'drv_in')
    ! seq_comm_init recreates the source CESM communicator table, including
    ! the full-size but distinct ATM communicator.  CAM must use that ATM
    ! communicator rather than MPI_COMM_WORLD: HOMME communication setup is
    ! communicator-sensitive even when membership and rank order are equal.
    call seq_comm_setptrs(ATMID(1), mpicom=atm_comm)
    call MPI_Comm_rank(atm_comm, rank, ierr)
    if (ierr /= 0) then
       status = 6_c_int
       return
    endif
    comp_id(1) = 1
    comp_name(1) = 'atm'
    comp_iamin(1) = .true.
    comp_comm(1) = atm_comm
    comp_comm_iam(1) = rank
    call shr_pio_init2(comp_id, comp_name, comp_iamin, comp_comm, comp_comm_iam)
    ! The original cesm_driver initializes the WRF/ESMF calendar tables
    ! between its pre-init phases.  CAM's time manager relies on those tables.
    call ESMF_Initialize()
    call seq_flds_set('drv_in', GLOID)
    call cam_instance_init(1)
    call spmdinit(atm_comm)
    call cam_cpl_indices_set()
    call read_namelist(single_column_in=single_column, scmlat_in=scmlat, &
         scmlon_in=scmlon, nlfilename_in='atm_in')
    nsrest = 0
    adiabatic = .false.
    ideal_phys = .false.
    aqua_planet = .false.
    calendar = 'NO_LEAP'
    call timemgr_init(calendar_in=calendar, start_ymd=10101, start_tod=0, &
         ref_ymd=10101, ref_tod=0, stop_ymd=10102, stop_tod=3600, &
         perpetual_run=.false., perpetual_ymd=0)
    call cam_init(cam_out, cam_in, atm_comm, 10101, 0, 10101, 0, &
         10102, 3600, .false., 0, calendar)
    ! hub2atm_alloc initializes the simple-land bucket arrays only when the
    ! bucket mode is enabled.  In this PI configuration it is disabled, and
    ! the coupled executable happens to receive zero-filled fresh pages while
    ! a Python-main process reuses nonzero heap pages.  The fields are still
    ! read by legacy diagnostics/run1 code, so remove that undefined-memory
    ! dependence at the Python ownership boundary.
    if (.not. wtrc_srf_bucket_mode) then
       do chunk = lbound(cam_in, 1), ubound(cam_in, 1)
          cam_in(chunk)%buckH = 0.0_r8
          cam_in(chunk)%buck16 = 0.0_r8
          cam_in(chunk)%buckD = 0.0_r8
          cam_in(chunk)%buck18 = 0.0_r8
       end do
    endif
    call atm_python_post_cam_init_mct(atm_comm, 1, local_surface_columns)
    initialized = .true.
  end function pycam_pi_cam_initialize_v1

  integer(c_int) function pycam_pi_cam_action_v1(action_id, nfields, &
       pointers, ndims, shapes, max_rank, fortran_comm, errmsg, errmsg_len) &
       bind(C, name='pycam_pi_cam_action_v1') result(status)
    integer(c_int), value, intent(in) :: action_id, nfields, max_rank
    integer(c_int), value, intent(in) :: fortran_comm, errmsg_len
    type(c_ptr), intent(in) :: pointers(*)
    integer(c_int32_t), intent(in) :: ndims(*)
    integer(c_int64_t), intent(in) :: shapes(*)
    character(kind=c_char), intent(out) :: errmsg(*)
    real(r8), pointer :: exchange(:,:)
    real(r8), pointer :: initial_import(:,:), initial_export(:,:)
    integer :: local_status, year, month, day, seconds, native_step
    logical :: write_restart, end_run

    ! Python is the process main, so every Python -> Fortran transition must
    ! restore the floating-point control word installed by Intel Fortran's
    ! executable startup.  Setting it only during initialize is insufficient:
    ! Python/NumPy code may run before the next numerical action.
    call pycam_pi_cam_set_fp_environment_v1()
    call clear_error(errmsg, errmsg_len)
    status = 0_c_int
    if (.not. initialized .or. finalized) then
       status = 10_c_int
       return
    endif

    select case (action_id)
    case (200)
       ! The second source atm_init_mct call performs this complete sequence
       ! in one Fortran stack frame.  Keep the initialization-only boundary
       ! intact: returning through Python between these calls changes stack
       ! reuse and breaks source BFB for legacy CAM local temporaries.
       if (nfields /= 2 .or. ndims(1) /= 2 .or. ndims(2) /= 2) then
          status = 14_c_int
          return
       endif
       call c_f_pointer(pointers(1), initial_import, &
            (/ int(shapes(1)), int(shapes(2)) /))
       call c_f_pointer(pointers(2), initial_export, &
            (/ int(shapes(3)), int(shapes(4)) /))
       call atm_import(initial_import, cam_in, cam_out)
       call cam_run1(cam_in, cam_out)
       call atm_export(cam_out, initial_export)
    case (202)
       if (nfields /= 1 .or. ndims(1) /= 2) then
          status = 11_c_int
          return
       endif
       call c_f_pointer(pointers(1), exchange, &
            (/ int(shapes(1)), int(shapes(2)) /))
       call atm_import(exchange, cam_in, cam_out)
    case (500:501)
       ! BFB-safe complete CAM timestep.  The experimental 401:431 ABI
       ! exposes individual phases and schemes, but returning through Python
       ! between those calls changes the lifetime of hidden HOMME module
       ! state.  Keep the source numerical call boundary intact for the
       ! default scientific path while Python still owns step orchestration,
       ! boundary arrays, clocks, and the public action trace.
       if (nfields /= 2 .or. ndims(1) /= 2 .or. ndims(2) /= 2) then
          status = 15_c_int
          return
       endif
       call c_f_pointer(pointers(1), initial_import, &
            (/ int(shapes(1)), int(shapes(2)) /))
       call c_f_pointer(pointers(2), initial_export, &
            (/ int(shapes(3)), int(shapes(4)) /))
       if (action_id == 500) call atm_import(initial_import, cam_in, cam_out)
       call cam_run2(cam_out, cam_in)
       call cam_run3(cam_out)
       call get_curr_date(year, month, day, seconds)
       native_step = get_nstep()
       write_restart = native_step >= configured_stop_n
       end_run = native_step >= configured_stop_n
       call cam_run4(cam_out, cam_in, write_restart, end_run, &
            yr_spec=year, mon_spec=month, day_spec=day, sec_spec=seconds)
       call advance_timestep()
       call cam_run1(cam_in, cam_out)
       call atm_export(cam_out, initial_export)
       if (write_restart) then
          call atm_python_write_srfrest_mct(initial_import, initial_export, &
               year, month, day, seconds)
       endif
    case (401)
       call cam_phys_run2_prepare(cam_in, local_status)
       status = int(local_status, c_int)
    case (402:411)
       call cam_phys_run2_scheme_action(action_id - 401, cam_out, cam_in, &
            local_status)
       status = int(local_status, c_int)
    case (412)
       call cam_phys_run2_finish()
    case (413)
       call cam_run2_dynamics()
    case (414)
       ! Use CAM's original phase routine.  Keeping a second copy of the
       ! stepon_run3 call in the adapter changed Intel's floating-point code
       ! generation and introduced a one-ULP difference after two steps.
       call cam_run3(cam_out)
    case (415)
       call cam_run4_history()
    case (416)
       call get_curr_date(year, month, day, seconds)
       native_step = get_nstep()
       write_restart = native_step >= configured_stop_n
       call cam_run4_restart(cam_out, cam_in, write_restart, year, month, day, seconds)
    case (417)
       native_step = get_nstep()
       write_restart = native_step >= configured_stop_n
       end_run = native_step >= configured_stop_n
       call cam_run4_finish(write_restart, end_run)
    case (418)
       call advance_timestep()
    case (419)
       call cam_run1_dynamics(cam_in, cam_out, local_status)
       status = int(local_status, c_int)
    case (420)
       call cam_phys_run1_prepare(cam_in, cam_out, local_status)
       status = int(local_status, c_int)
    case (421:431)
       call cam_phys_run1_scheme_action(action_id - 420, cam_in, cam_out, &
            local_status)
       status = int(local_status, c_int)
    case (432)
       if (nfields /= 1 .or. ndims(1) /= 2) then
          status = 12_c_int
          return
       endif
       call c_f_pointer(pointers(1), exchange, &
            (/ int(shapes(1)), int(shapes(2)) /))
       call atm_export(cam_out, exchange)
    case default
       status = 13_c_int
    end select
  end function pycam_pi_cam_action_v1

  integer(c_int) function pycam_pi_cam_finalize_v1(action_id, nfields, &
       pointers, ndims, shapes, max_rank, fortran_comm, errmsg, errmsg_len) &
       bind(C, name='pycam_pi_cam_finalize_v1') result(status)
    integer(c_int), value, intent(in) :: action_id, nfields, max_rank
    integer(c_int), value, intent(in) :: fortran_comm, errmsg_len
    type(c_ptr), intent(in) :: pointers(*)
    integer(c_int32_t), intent(in) :: ndims(*)
    integer(c_int64_t), intent(in) :: shapes(*)
    character(kind=c_char), intent(out) :: errmsg(*)

    call pycam_pi_cam_set_fp_environment_v1()
    call clear_error(errmsg, errmsg_len)
    status = 0_c_int
    if (.not. initialized .or. finalized .or. action_id /= 0 .or. &
         nfields /= 0) then
       status = 20_c_int
       return
    endif
    call cam_final(cam_out, cam_in)
    call shr_pio_finalize()
    ! This component adapter does not own MPI.  In the WRF time-manager
    ! compatibility layer ESMF_Finalize calls MPI_Finalize directly, which
    ! would invalidate the mpi4py communicator before Python can gather
    ! results or continue serving a persistent session.  mpi4py finalizes
    ! the process-owned MPI world when the Python worker exits.
    finalized = .true.
  end function pycam_pi_cam_finalize_v1

end module pycam_pi_cam_adapter
