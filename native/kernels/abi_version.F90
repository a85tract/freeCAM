module pycam_sima_abi
  use iso_c_binding, only: c_int
  use pycam_sima_build_config, only: build_np, build_nc, build_nlev, build_ntrac
  implicit none
  private
  public :: pycam_sima_abi_version, pycam_sima_kernel_specialization_v1
contains
  integer(c_int) function pycam_sima_abi_version() &
       bind(C, name="pycam_sima_abi_version")
    pycam_sima_abi_version = 2_c_int
  end function pycam_sima_abi_version

  subroutine pycam_sima_kernel_specialization_v1(np,nc,nlev,ntrac) &
       bind(C, name="pycam_sima_kernel_specialization_v1")
    integer(c_int), intent(out) :: np,nc,nlev,ntrac
    np=build_np
    nc=build_nc
    nlev=build_nlev
    ntrac=build_ntrac
  end subroutine pycam_sima_kernel_specialization_v1
end module pycam_sima_abi
