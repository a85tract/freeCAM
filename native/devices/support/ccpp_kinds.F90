! Portable kind provider for standalone CCPP numerical devices.
module ccpp_kinds
  use iso_fortran_env, only: real64
  implicit none
  private
  integer, parameter, public :: kind_phys = real64
end module ccpp_kinds
