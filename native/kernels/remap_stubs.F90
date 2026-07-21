! Minimal kind module for the selected remap source.
module shr_kind_mod
  use iso_c_binding, only: c_double
  implicit none
  integer, parameter :: shr_kind_r8 = c_double
end module shr_kind_mod

module cam_abortutils
  implicit none
contains
  subroutine endrun(message)
    character(len=*), intent(in) :: message
    error stop message
  end subroutine endrun
end module cam_abortutils
