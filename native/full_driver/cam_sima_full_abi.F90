module cam_sima_full_abi
  use, intrinsic :: iso_c_binding, only : c_int, c_ptr, c_loc, c_null_ptr
  use mpi, only : MPI_Comm_rank, MPI_Comm_size
  use pio, only : pio_init, pio_finalize, PIO_REARR_BOX, PIO_IOTYPE_PNETCDF, &
       PIO_64BIT_DATA
  use ESMF, only : ESMF_Initialize, ESMF_LOGKIND_MULTI_ON_ERROR, &
       ESMF_CALKIND_GREGORIAN
  use shr_pio_mod, only : io_compname, pio_comp_settings, iosystems, io_compid
  use shr_orb_mod, only : shr_orb_params, SHR_ORB_UNDEF_REAL
  use spmd_utils, only : spmd_init
  use cam_instance, only : cam_instance_init
  use cam_comp, only : cam_init, cam_timestep_init, cam_run1, cam_run2, &
       cam_run3, cam_run4, cam_timestep_final, cam_final
  use time_manager, only : advance_timestep, get_nstep
  use physics_types, only : phys_state, phys_tend
  use cam_ccpp_cap, only : cam_constituents_array
  use ccpp_kinds, only : kind_phys
  implicit none
  private

  integer, parameter :: ABI_VERSION = 1
  integer, parameter :: ATM_ID = 2
  logical :: initialized = .false.

  public :: pycam_full_abi_version
  public :: pycam_full_initialize
  public :: pycam_full_timestep_init
  public :: pycam_full_run1
  public :: pycam_full_run2
  public :: pycam_full_run3
  public :: pycam_full_timestep_final
  public :: pycam_full_advance_timestep
  public :: pycam_full_finalize
  public :: pycam_full_get_nstep
  public :: pycam_full_get_field

contains

  integer(c_int) function pycam_full_abi_version() bind(C)
    pycam_full_abi_version = ABI_VERSION
  end function pycam_full_abi_version

  integer(c_int) function pycam_full_initialize(comm) bind(C)
    integer(c_int), value :: comm
    integer :: rank, npes, num_iotasks, rc
    real(kind_phys) :: eccen, obliq, mvelp, obliqr, lambm0, mvelpp
    character(len=256) :: caseid, ctitle, model_doi_url
    character(len=80) :: calendar

    pycam_full_initialize = 1_c_int
    if (initialized) return

    call MPI_Comm_rank(comm, rank, rc)
    if (rc /= 0) return
    call MPI_Comm_size(comm, npes, rc)
    if (rc /= 0) return

    call ESMF_Initialize(mpiCommunicator=comm, &
         logkindflag=ESMF_LOGKIND_MULTI_ON_ERROR, logappendflag=.false., &
         defaultCalkind=ESMF_CALKIND_GREGORIAN, ioUnitLBound=5001, &
         ioUnitUBound=5101, rc=rc)
    if (rc /= 0) return

    call spmd_init(comm)
    call initialize_pio(comm, rank, npes)
    call cam_instance_init(ATM_ID, 'ATM', 1, '')

    caseid = 'PYCAM_SIMA_FKESSLER'
    ctitle = 'Python controlled CAM-SIMA FKESSLER'
    model_doi_url = 'not_set'
    calendar = 'NO_LEAP'

    eccen = SHR_ORB_UNDEF_REAL
    obliq = SHR_ORB_UNDEF_REAL
    mvelp = SHR_ORB_UNDEF_REAL
    call shr_orb_params(2000, eccen, obliq, mvelp, obliqr, lambm0, mvelpp, rank == 0)

    call cam_init( &
         caseid=caseid, &
         ctitle=ctitle, &
         model_doi_url=model_doi_url, &
         initial_run_in=.true., restart_run_in=.false., branch_run_in=.false., &
         post_assim_in=.false., calendar=calendar, &
         brnch_retain_casename=.false., aqua_planet=.false., &
         single_column=.false., scmlat=-999._kind_phys, scmlon=-999._kind_phys, &
         eccen=eccen, obliqr=obliqr, lambm0=lambm0, mvelpp=mvelpp, &
         perpetual_run=.false., perpetual_ymd=0, dtime=1800, &
         start_ymd=10101, start_tod=0, ref_ymd=10101, ref_tod=0, &
         stop_ymd=99990101, stop_tod=0, curr_ymd=10101, curr_tod=0)

    initialized = .true.
    pycam_full_initialize = 0_c_int
  end function pycam_full_initialize

  subroutine initialize_pio(comm, rank, npes)
    integer, intent(in) :: comm, rank, npes
    integer :: num_iotasks, stride

    allocate(pio_comp_settings(1), io_compid(1), io_compname(1), iosystems(1))
    io_compid(1) = ATM_ID
    io_compname(1) = 'ATM'
    num_iotasks = max(1, min(npes, 4))
    stride = max(1, npes / num_iotasks)
    pio_comp_settings(1)%pio_root = 0
    pio_comp_settings(1)%pio_stride = stride
    pio_comp_settings(1)%pio_numiotasks = num_iotasks
    pio_comp_settings(1)%pio_iotype = PIO_IOTYPE_PNETCDF
    pio_comp_settings(1)%pio_rearranger = PIO_REARR_BOX
    pio_comp_settings(1)%pio_netcdf_ioformat = PIO_64BIT_DATA
    pio_comp_settings(1)%pio_async_interface = .false.
    call pio_init(rank, comm, num_iotasks, 0, stride, PIO_REARR_BOX, &
         iosystems(1), 0)
  end subroutine initialize_pio

  integer(c_int) function pycam_full_timestep_init() bind(C)
    pycam_full_timestep_init = require_initialized()
    if (pycam_full_timestep_init /= 0) return
    call cam_timestep_init()
  end function pycam_full_timestep_init

  integer(c_int) function pycam_full_run1() bind(C)
    pycam_full_run1 = require_initialized()
    if (pycam_full_run1 /= 0) return
    call cam_run1()
  end function pycam_full_run1

  integer(c_int) function pycam_full_run2() bind(C)
    pycam_full_run2 = require_initialized()
    if (pycam_full_run2 /= 0) return
    call cam_run2()
  end function pycam_full_run2

  integer(c_int) function pycam_full_run3() bind(C)
    pycam_full_run3 = require_initialized()
    if (pycam_full_run3 /= 0) return
    call cam_run3()
  end function pycam_full_run3

  integer(c_int) function pycam_full_timestep_final() bind(C)
    pycam_full_timestep_final = require_initialized()
    if (pycam_full_timestep_final /= 0) return
    call cam_run4(.false., .false.)
    call cam_timestep_final(.false., .false., do_ncdata_check=.false.)
  end function pycam_full_timestep_final

  integer(c_int) function pycam_full_advance_timestep() bind(C)
    pycam_full_advance_timestep = require_initialized()
    if (pycam_full_advance_timestep /= 0) return
    call advance_timestep()
  end function pycam_full_advance_timestep

  integer(c_int) function pycam_full_finalize() bind(C)
    pycam_full_finalize = require_initialized()
    if (pycam_full_finalize /= 0) return
    call cam_timestep_final(.false., .true., do_ncdata_check=.false., &
         do_history_write=.false.)
    call cam_final()
    call pio_finalize(iosystems(1), pycam_full_finalize)
    initialized = .false.
  end function pycam_full_finalize

  integer(c_int) function pycam_full_get_nstep() bind(C)
    if (initialized) then
      pycam_full_get_nstep = int(get_nstep(), c_int)
    else
      pycam_full_get_nstep = -1_c_int
    end if
  end function pycam_full_get_nstep

  integer(c_int) function require_initialized()
    if (initialized) then
      require_initialized = 0_c_int
    else
      require_initialized = 2_c_int
    end if
  end function require_initialized

  integer(c_int) function pycam_full_get_field(field_id, data, rank, dims) bind(C)
    integer(c_int), value :: field_id
    type(c_ptr), intent(out) :: data
    integer(c_int), intent(out) :: rank
    integer(c_int), intent(out) :: dims(4)
    real(kind_phys), pointer :: constituents(:,:,:)

    data = c_null_ptr
    rank = 0_c_int
    dims = 0_c_int
    pycam_full_get_field = require_initialized()
    if (pycam_full_get_field /= 0) return

    select case (field_id)
    case (1);  call export_2d(phys_state%T, data, rank, dims)
    case (2);  call export_2d(phys_state%u, data, rank, dims)
    case (3);  call export_2d(phys_state%v, data, rank, dims)
    case (4);  call export_1d(phys_state%ps, data, rank, dims)
    case (5);  call export_2d(phys_state%pdel, data, rank, dims)
    case (6);  call export_2d(phys_state%pdeldry, data, rank, dims)
    case (7);  call export_2d(phys_state%pmid, data, rank, dims)
    case (8);  call export_2d(phys_state%pmiddry, data, rank, dims)
    case (9);  call export_2d(phys_state%pint, data, rank, dims)
    case (10); call export_2d(phys_state%pintdry, data, rank, dims)
    case (11); call export_1d(phys_state%psdry, data, rank, dims)
    case (12); call export_1d(phys_state%phis, data, rank, dims)
    case (13); call export_2d(phys_state%zm, data, rank, dims)
    case (14); call export_2d(phys_state%zi, data, rank, dims)
    case (15); call export_2d(phys_state%omega, data, rank, dims)
    case (16); call export_2d(phys_state%exner, data, rank, dims)
    case (17); call export_2d(phys_state%dse, data, rank, dims)
    case (18); call export_2d(phys_tend%dTdt_total, data, rank, dims)
    case (19); call export_2d(phys_tend%dudt_total, data, rank, dims)
    case (20); call export_2d(phys_tend%dvdt_total, data, rank, dims)
    case (21)
      constituents => cam_constituents_array()
      call export_3d(constituents, data, rank, dims)
    case default
      pycam_full_get_field = 3_c_int
      return
    end select
    pycam_full_get_field = 0_c_int
  end function pycam_full_get_field

  subroutine export_1d(array, data, rank, dims)
    real(kind_phys), pointer, intent(in) :: array(:)
    type(c_ptr), intent(out) :: data
    integer(c_int), intent(out) :: rank, dims(4)
    dims = 0_c_int
    rank = 1_c_int
    dims(1) = int(size(array, 1), c_int)
    data = c_loc(array(1))
  end subroutine export_1d

  subroutine export_2d(array, data, rank, dims)
    real(kind_phys), pointer, intent(in) :: array(:,:)
    type(c_ptr), intent(out) :: data
    integer(c_int), intent(out) :: rank, dims(4)
    dims = 0_c_int
    rank = 2_c_int
    dims(1) = int(size(array, 1), c_int)
    dims(2) = int(size(array, 2), c_int)
    data = c_loc(array(1,1))
  end subroutine export_2d

  subroutine export_3d(array, data, rank, dims)
    real(kind_phys), pointer, intent(in) :: array(:,:,:)
    type(c_ptr), intent(out) :: data
    integer(c_int), intent(out) :: rank, dims(4)
    dims = 0_c_int
    rank = 3_c_int
    dims(1) = int(size(array, 1), c_int)
    dims(2) = int(size(array, 2), c_int)
    dims(3) = int(size(array, 3), c_int)
    data = c_loc(array(1,1,1))
  end subroutine export_3d

end module cam_sima_full_abi
