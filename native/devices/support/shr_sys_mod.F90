! Minimal fatal-error service for standalone devices.
module shr_sys_mod
  use iso_fortran_env, only: error_unit,real64
  implicit none
  private
  public :: shr_sys_abort,shr_sys_sleep
contains
  subroutine shr_sys_abort(message)
    character(len=*), intent(in) :: message
    write(error_unit, '(a)') trim(message)
    error stop 1
  end subroutine shr_sys_abort
  subroutine shr_sys_sleep(seconds)
    real(real64), intent(in) :: seconds
    ! Timing/backoff is a Python host responsibility for standalone devices.
  end subroutine shr_sys_sleep
end module shr_sys_mod
