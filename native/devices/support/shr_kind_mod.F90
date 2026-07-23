! Portable kind-only host service for source-preserving numerical devices.
module shr_kind_mod
  use iso_fortran_env, only: int32, int64, real32, real64
  implicit none
  private
  integer, parameter, public :: shr_kind_r4 = real32
  integer, parameter, public :: shr_kind_r8 = real64
  integer, parameter, public :: shr_kind_in = int32
  integer, parameter, public :: shr_kind_i4 = int32
  integer, parameter, public :: shr_kind_i8 = int64
  integer, parameter, public :: shr_kind_cs = 128
  integer, parameter, public :: shr_kind_cl = 512
  integer, parameter, public :: shr_kind_cx = 512
end module shr_kind_mod
