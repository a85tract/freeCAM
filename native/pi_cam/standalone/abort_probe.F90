! Probe the Fortran abort bridge of a standalone physics-function image.
!
! The image replaces shr_sys_abort with a C stub that reports the message and
! exits.  That stub relies on ifort's calling convention for a routine with
! optional character and integer dummies; this probe, compiled with the same
! compiler, calls shr_sys_abort with a message the build's smoke test then
! expects to see on stderr, proving the convention rather than assuming it.
module freecam_standalone_abort_probe
  use, intrinsic :: iso_c_binding, only: c_char, c_int
  use shr_sys_mod, only: shr_sys_abort
  implicit none
  private
  public :: freecam_standalone_abort_probe_v1
contains
  function freecam_standalone_abort_probe_v1(message, length) result(status) &
       bind(C, name='freecam_standalone_abort_probe_v1')
    character(kind=c_char), intent(in) :: message(*)
    integer(c_int), value, intent(in) :: length
    integer(c_int) :: status
    character(len=length) :: text
    integer :: index
    do index = 1, length
       text(index:index) = message(index)
    end do
    call shr_sys_abort(text)
    status = 1_c_int
  end function freecam_standalone_abort_probe_v1
end module freecam_standalone_abort_probe
