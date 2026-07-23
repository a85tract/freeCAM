module spmd_utils
  implicit none
  logical, public :: masterproc = .true.
  integer, public :: masterprocid = 0
  integer, public :: mpicom = 0
end module spmd_utils
