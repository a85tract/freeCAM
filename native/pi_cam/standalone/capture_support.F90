! Capture one physics routine's actual arguments at its real call site.
!
! A standalone function image is only trustworthy if, given the arguments the
! model really passed, it returns what the model really got back.  This module
! records those arguments from inside a running CAM: a 'before' record with
! every actual argument as the call receives it, and an 'after' record with
! every argument as the call leaves it.  Whole chunks are written, so the lane
! layout, ncol, and every padding row are preserved exactly.
!
! The hook is inert unless PYCAM_FUNCTION_CAPTURE names an output prefix, and
! then writes only for timesteps in PYCAM_FUNCTION_CAPTURE_STEPS ('first,last').
! Nothing numerical is touched: the routine's own call sits untouched between
! the two records, and the capture executable is proven bit-for-bit against
! the oracle run with capture on and with it off.
!
! Stream layout per rank (`<prefix>.rank-NNNNNN.stream.bin`, production byte
! order): for each record, magic 'PYCAM_FUNCTION1 ', name (32 chars), phase
! (8 chars), int32 header (version, nstep, lchnk, ncol, rank, nargs), dt as
! real(8); then nargs entries of tag (32 chars), int32 kind, int32 rank, int32
! dims(2), an int32 associated flag for pointer kinds, and the payload.
module pycam_function_capture
  use shr_kind_mod, only: r8 => shr_kind_r8
  use spmd_utils, only: iam, npes
  use time_manager, only: get_nstep
  use shr_sys_mod, only: shr_sys_abort
  implicit none
  private
  public :: pycam_capture_begin, pycam_capture_end
  public :: pycam_capture_real0, pycam_capture_real1, pycam_capture_real2
  public :: pycam_capture_pointer2, pycam_capture_int, pycam_capture_logical

  integer, parameter :: kind_real = 1, kind_int = 2, kind_logical = 3, kind_pointer = 4
  character(len=16), parameter :: magic = 'PYCAM_FUNCTION1 '
  logical, save :: checked = .false., enabled = .false., active = .false.
  integer, save :: unit = 0, first_step = -1, last_step = -1
  character(len=1024), save :: prefix = ''
contains

  subroutine ensure_checked()
    character(len=64) :: steps
    character(len=1024) :: filename
    integer :: status, length, comma, ierr
    if (checked) return
    checked = .true.
    call get_environment_variable('PYCAM_FUNCTION_CAPTURE', prefix, length=length, status=status)
    enabled = status == 0 .and. length > 0
    if (.not. enabled) return
    steps = ''
    call get_environment_variable('PYCAM_FUNCTION_CAPTURE_STEPS', steps, length=length, status=status)
    if (status == 0 .and. length > 0) then
       comma = index(steps, ',')
       if (comma > 1) then
          read(steps(1:comma-1), *) first_step
          read(steps(comma+1:), *) last_step
       else
          read(steps, *) first_step
          last_step = first_step
       end if
    end if
    write(filename, '(a,".rank-",i6.6,".stream.bin")') trim(prefix), iam
    open(newunit=unit, file=trim(filename), access='stream', form='unformatted', &
         status='replace', action='write', iostat=ierr)
    if (ierr /= 0) call shr_sys_abort('pycam_function_capture: cannot open '//trim(filename))
  end subroutine ensure_checked

  subroutine pycam_capture_begin(name, phase, lchnk, ncol, dt, nargs)
    character(len=*), intent(in) :: name, phase
    integer, intent(in) :: lchnk, ncol, nargs
    real(r8), intent(in) :: dt
    character(len=32) :: padded_name
    character(len=8) :: padded_phase
    integer :: nstep, header(6)
    call ensure_checked()
    active = .false.
    if (.not. enabled) return
    nstep = get_nstep()
    if (first_step >= 0 .and. (nstep < first_step .or. nstep > last_step)) return
    active = .true.
    padded_name = name
    padded_phase = phase
    header = (/ 1, nstep, lchnk, ncol, iam, nargs /)
    write(unit) magic
    write(unit) padded_name
    write(unit) padded_phase
    write(unit) header
    write(unit) dt
  end subroutine pycam_capture_begin

  subroutine pycam_capture_end()
    if (active) flush(unit)
    active = .false.
  end subroutine pycam_capture_end

  subroutine write_entry_header(tag, kind, rank, dims, associated_flag)
    character(len=*), intent(in) :: tag
    integer, intent(in) :: kind, rank, dims(2), associated_flag
    character(len=32) :: padded
    padded = tag
    write(unit) padded
    write(unit) kind
    write(unit) rank
    write(unit) dims
    if (kind == kind_pointer) write(unit) associated_flag
  end subroutine write_entry_header

  subroutine pycam_capture_real0(tag, value)
    character(len=*), intent(in) :: tag
    real(r8), intent(in) :: value
    if (.not. active) return
    call write_entry_header(tag, kind_real, 0, (/ 1, 1 /), 0)
    write(unit) value
  end subroutine pycam_capture_real0

  subroutine pycam_capture_real1(tag, values)
    character(len=*), intent(in) :: tag
    real(r8), intent(in) :: values(:)
    if (.not. active) return
    call write_entry_header(tag, kind_real, 1, (/ size(values, 1), 1 /), 0)
    write(unit) values
  end subroutine pycam_capture_real1

  subroutine pycam_capture_real2(tag, values)
    character(len=*), intent(in) :: tag
    real(r8), intent(in) :: values(:,:)
    if (.not. active) return
    call write_entry_header(tag, kind_real, 2, (/ size(values, 1), size(values, 2) /), 0)
    write(unit) values
  end subroutine pycam_capture_real2

  subroutine pycam_capture_pointer2(tag, values)
    character(len=*), intent(in) :: tag
    real(r8), pointer :: values(:,:)
    if (.not. active) return
    if (associated(values)) then
       call write_entry_header(tag, kind_pointer, 2, (/ size(values, 1), size(values, 2) /), 1)
       write(unit) values
    else
       call write_entry_header(tag, kind_pointer, 2, (/ 0, 0 /), 0)
    end if
  end subroutine pycam_capture_pointer2

  subroutine pycam_capture_int(tag, value)
    character(len=*), intent(in) :: tag
    integer, intent(in) :: value
    if (.not. active) return
    call write_entry_header(tag, kind_int, 0, (/ 1, 1 /), 0)
    write(unit) value
  end subroutine pycam_capture_int

  subroutine pycam_capture_logical(tag, value)
    character(len=*), intent(in) :: tag
    logical, intent(in) :: value
    integer :: flag
    if (.not. active) return
    flag = 0
    if (value) flag = 1
    call write_entry_header(tag, kind_logical, 0, (/ 1, 1 /), 0)
    write(unit) flag
  end subroutine pycam_capture_logical

end module pycam_function_capture
