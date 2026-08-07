! Portable interface provider for source-preserving standalone devices.
!
! CAM's production cam_time_coord implementation owns PIO and time-manager
! services.  The CAM4 aquaplanet configuration used by the standalone runtime
! selects filename_of_solar_irradiance_data='NONE', so solar_irradiance_data
! never calls these methods.  Keep the original public type contract available
! to the compiler, but fail closed if a file-backed time coordinate is used
! without a declared Python host provider.
module cam_time_coord
  use ccpp_kinds, only: kind_phys
  implicit none
  private

  type, public :: time_coordinate
    integer :: ntimes = 0
    real(kind_phys) :: wghts(2) = -huge(1.0_kind_phys)
    integer :: indxs(2) = -huge(1)
    real(kind_phys), allocatable :: times(:)
    real(kind_phys), allocatable :: time_bnds(:,:)
    logical :: time_interp = .true.
    logical :: fixed = .false.
    integer :: fixed_ymd = -huge(1)
    integer :: fixed_tod = -huge(1)
    real(kind_phys) :: dtime = 0.0_kind_phys
    character(len=:), allocatable :: filename
  contains
    procedure :: initialize
    procedure :: advance
    procedure :: read_more
    procedure :: copy
    procedure :: destroy
  end type time_coordinate

contains

  subroutine initialize(this, filepath, fixed, fixed_ymd, fixed_tod, &
                        force_time_interp, set_weights, try_dates, delta_days)
    class(time_coordinate), intent(inout) :: this
    character(len=*), intent(in) :: filepath
    logical, optional, intent(in) :: fixed
    integer, optional, intent(in) :: fixed_ymd
    integer, optional, intent(in) :: fixed_tod
    logical, optional, intent(in) :: force_time_interp
    logical, optional, intent(in) :: set_weights
    logical, optional, intent(in) :: try_dates
    real(kind_phys), optional, intent(in) :: delta_days

    ! Touch every optional argument so strict compilers do not diagnose an
    ! accidental ABI mismatch.  File-backed time interpolation is intentionally
    ! unsupported until a Python callback provider supplies its model clock.
    this%filename = trim(filepath)
    if (present(fixed)) this%fixed = fixed
    if (present(fixed_ymd)) this%fixed_ymd = fixed_ymd
    if (present(fixed_tod)) this%fixed_tod = fixed_tod
    if (present(delta_days)) this%dtime = delta_days
    if (present(force_time_interp)) this%time_interp = force_time_interp
    if (present(set_weights)) this%time_interp = this%time_interp .or. set_weights
    if (present(try_dates)) this%time_interp = this%time_interp .or. try_dates
    error stop "freecam cam_time_coord: file-backed solar time coordinates require a Python host provider"
  end subroutine initialize

  subroutine advance(this)
    class(time_coordinate), intent(inout) :: this
    if (allocated(this%filename)) then
      error stop "freecam cam_time_coord: advance is unavailable for file-backed solar data"
    end if
  end subroutine advance

  logical function read_more(this) result(check)
    class(time_coordinate), intent(in) :: this
    check = .false.
    if (allocated(this%filename)) then
      error stop "freecam cam_time_coord: read_more is unavailable for file-backed solar data"
    end if
  end function read_more

  subroutine copy(this, obj)
    class(time_coordinate), intent(inout) :: this
    class(time_coordinate), intent(in) :: obj

    call this%destroy()
    this%ntimes = obj%ntimes
    this%wghts = obj%wghts
    this%indxs = obj%indxs
    this%time_interp = obj%time_interp
    this%fixed = obj%fixed
    this%fixed_ymd = obj%fixed_ymd
    this%fixed_tod = obj%fixed_tod
    this%dtime = obj%dtime
    if (allocated(obj%times)) then
      allocate(this%times(size(obj%times)))
      this%times = obj%times
    end if
    if (allocated(obj%time_bnds)) then
      allocate(this%time_bnds(size(obj%time_bnds, 1), size(obj%time_bnds, 2)))
      this%time_bnds = obj%time_bnds
    end if
    if (allocated(obj%filename)) this%filename = obj%filename
  end subroutine copy

  subroutine destroy(this)
    class(time_coordinate), intent(inout) :: this
    if (allocated(this%times)) deallocate(this%times)
    if (allocated(this%time_bnds)) deallocate(this%time_bnds)
    if (allocated(this%filename)) deallocate(this%filename)
    this%ntimes = 0
    this%wghts = -huge(1.0_kind_phys)
    this%indxs = -huge(1)
    this%time_interp = .true.
    this%fixed = .false.
    this%fixed_ymd = -huge(1)
    this%fixed_tod = -huge(1)
    this%dtime = 0.0_kind_phys
  end subroutine destroy

end module cam_time_coord
