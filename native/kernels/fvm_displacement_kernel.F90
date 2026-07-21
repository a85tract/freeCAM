! Rank-local displacement kernel for the Python model.
module pycam_sima_fvm_displacement_kernel
  use iso_c_binding, only: c_double, c_int
  use pycam_sima_build_config, only: build_nc,build_nlev,build_ntrac, &
       build_np,build_ngpc,build_irecons,build_nhe,build_nhr,build_nht, &
       build_ns,build_nhc
  use dimensions_mod, only: fvm_dimensions_c, configure_fvm_dimensions, &
       release_fvm_dimensions
  use fvm_control_volume_mod, only: fvm_struct
  use fvm_analytic_mod, only: gauss_points
  use fvm_consistent_se_cslam, only: compute_displacements_for_swept_areas
  implicit none
contains
  subroutine pycam_sima_fvm_displacement_v2(config,irecons_levels,pressure, &
       swept_flux,displacement_maximum,vertex_cartesian,errflg) &
       bind(C, name="pycam_sima_fvm_displacement_v2")
    type(fvm_dimensions_c), intent(in) :: config
    integer(c_int), intent(in) :: irecons_levels(build_nlev)
    real(c_double), intent(in) :: pressure(build_nc+2*build_nhc, &
         build_nc+2*build_nhc,build_nlev)
    real(c_double), intent(inout) :: swept_flux(build_nc,build_nc,4,build_nlev)
    real(c_double), intent(in) :: displacement_maximum(build_nc+2*build_nhc, &
         build_nc+2*build_nhc,4)
    real(c_double), intent(in) :: vertex_cartesian(4,2, &
         build_nc+2*build_nhc,build_nc+2*build_nhc)
    integer(c_int), intent(out) :: errflg
    real(c_double) :: weights(build_ngpc),points(build_ngpc)
    type(fvm_struct) :: fvm
    integer :: k,ierr

    call configure_fvm_dimensions(config%nc,config%nlev,config%ntrac, &
         config%np,config%ngpc,config%irecons,config%nhe,config%nhr, &
         config%nht,config%ns,config%nhc,config%kmin_jet, &
         config%kmax_jet,config%large_courant/=0,irecons_levels,ierr)
    errflg=ierr
    if (ierr/=0) return
    allocate(fvm%dp_fvm(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,build_nlev)); fvm%dp_fvm=pressure
    allocate(fvm%se_flux(0:build_nc+1,0:build_nc+1,4,build_nlev)); fvm%se_flux=0.0_c_double
    fvm%se_flux(1:build_nc,1:build_nc,:,:)=swept_flux
    allocate(fvm%displ_max(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,4)); fvm%displ_max=displacement_maximum
    allocate(fvm%vtx_cart(4,2,1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc)); fvm%vtx_cart=vertex_cartesian
    call gauss_points(build_ngpc,weights,points)
    points=0.5_c_double*(points+1.0_c_double)
    do k=config%level_begin,config%level_end
      call compute_displacements_for_swept_areas(fvm,fvm%dp_fvm(:,:,k),k,weights,points)
    end do
    swept_flux=fvm%se_flux(1:build_nc,1:build_nc,:,:)
    call release_fvm_dimensions()
  end subroutine pycam_sima_fvm_displacement_v2
end module pycam_sima_fvm_displacement_kernel
