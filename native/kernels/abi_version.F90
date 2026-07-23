module pycam_sima_abi
  use iso_c_binding, only: c_int
  implicit none
  private
  public :: pycam_sima_abi_version
contains
  integer(c_int) function pycam_sima_abi_version() &
       bind(C, name="pycam_sima_abi_version")
    pycam_sima_abi_version = 2_c_int
  end function pycam_sima_abi_version
end module pycam_sima_abi
