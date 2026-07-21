! Stateless vertical-remap kernel for the Python model.
module pycam_sima_remap_kernel
  use iso_c_binding, only: c_bool, c_double, c_int
  use fv_mapz, only: map1_ppm
  implicit none
contains
  subroutine pycam_sima_remap_fv3_v1(nx, nlev, nq, identifier, mass_field, &
       kord, ptop, field, dp_source, dp_target) &
       bind(C, name="pycam_sima_remap_fv3_v1")
    integer(c_int), value, intent(in) :: nx, nlev, nq, identifier, kord
    logical(c_bool), value, intent(in) :: mass_field
    real(c_double), value, intent(in) :: ptop
    real(c_double), intent(inout) :: field(nx,nx,nlev,nq)
    real(c_double), intent(in) :: dp_source(nx,nx,nlev)
    real(c_double), intent(in) :: dp_target(nx,nx,nlev)
    real(c_double) :: pe_source(nx,nlev+1), pe_target(nx,nlev+1)
    real(c_double) :: inverse_dp(nx,nx,nlev), surface_value(nx)
    integer :: i, j, k, q

    if (mass_field) then
      inverse_dp = 1.0_c_double / dp_source
      field = field * spread(inverse_dp, dim=4, ncopies=nq)
    end if

    do j = 1, nx
      pe_source(:,1) = ptop
      pe_target(:,1) = ptop
      do k = 1, nlev
        do i = 1, nx
          pe_source(i,k+1) = pe_source(i,k) + dp_source(i,j,k)
          pe_target(i,k+1) = pe_target(i,k) + dp_target(i,j,k)
        end do
      end do
      pe_source(:,nlev+1) = pe_target(:,nlev+1)
      do q = 1, nq
        call map1_ppm(nlev, pe_source, field(:,:,:,q), surface_value, &
             nlev, pe_target, field(:,:,:,q), 1, nx, j, 1, nx, 1, nx, &
             identifier, abs(kord))
      end do
    end do

    if (mass_field) then
      field = field * spread(dp_target, dim=4, ncopies=nq)
    end if
  end subroutine pycam_sima_remap_fv3_v1
end module pycam_sima_remap_kernel
