! Rank-local FVM transport kernel for the Python model.
module pycam_sima_fvm_transport_kernel
  use iso_c_binding, only: c_double, c_int
  use pycam_sima_build_config, only: build_nc,build_nlev,build_ntrac, &
       build_np,build_ngpc,build_irecons,build_nhe,build_nhr,build_nht, &
       build_ns,build_nhc
  use dimensions_mod, only: fvm_dimensions_c, configure_fvm_dimensions, &
       release_fvm_dimensions
  use element_mod, only: element_t
  use fvm_control_volume_mod, only: fvm_struct
  use hybrid_mod, only: hybrid_t
  use time_mod, only: timelevel_t
  use hybvcoord_mod, only: hvcoord_t
  use edgetype_mod, only: edgebuffer_t
  use fvm_consistent_se_cslam, only: run_consistent_se_cslam, &
       large_courant_number_increment,pycam_transport_stage
  implicit none
contains
  subroutine pycam_sima_fvm_transport_v2(config,irecons_levels,dt, &
       subflux,tracer,dp,psc,se_flux,dp_ref,dp_ref_inverse,area, &
       inverse_area,cubeboundary,displ_max,flux_vec,vtx_cart, &
       flux_orient,ifct,rot_matrix,spherecentroid,recons_metrics, &
       recons_metrics_integral,jx_min,jx_max,jy_min,jy_max,ibase, &
       halo_weight,centroid_stretch,vertex_weights,errflg) &
       bind(C, name="pycam_sima_fvm_transport_v2")
    type(fvm_dimensions_c), intent(in) :: config
    integer(c_int), intent(in) :: irecons_levels(build_nlev)
    real(c_double), value, intent(in) :: dt
    real(c_double), intent(inout) :: subflux(build_nc,build_nc,4,build_nlev)
    real(c_double), intent(inout) :: tracer(build_nc+2*build_nhc, &
         build_nc+2*build_nhc,build_nlev,build_ntrac)
    real(c_double), intent(inout) :: dp(build_nc+2*build_nhc, &
         build_nc+2*build_nhc,build_nlev)
    real(c_double), intent(inout) :: psc(build_nc,build_nc)
    real(c_double), intent(inout) :: se_flux(build_nc+2,build_nc+2,4,build_nlev)
    real(c_double), intent(in) :: dp_ref(build_nlev),dp_ref_inverse(build_nlev)
    real(c_double), intent(in) :: area(build_nc,build_nc),inverse_area(build_nc,build_nc)
    integer(c_int), value, intent(in) :: cubeboundary
    real(c_double), intent(in) :: displ_max(build_nc+2*build_nhc, &
         build_nc+2*build_nhc,4)
    integer(c_int), intent(in) :: flux_vec(2,build_nc+2*build_nhc, &
         build_nc+2*build_nhc,4)
    real(c_double), intent(in) :: vtx_cart(4,2,build_nc+2*build_nhc, &
         build_nc+2*build_nhc)
    real(c_double), intent(in) :: flux_orient(2,build_nc+2*build_nhc, &
         build_nc+2*build_nhc)
    integer(c_int), intent(in) :: ifct(build_nc+2*build_nhc,build_nc+2*build_nhc)
    integer(c_int), intent(in) :: rot_matrix(2,2,build_nc+2*build_nhc, &
         build_nc+2*build_nhc)
    real(c_double), intent(in) :: spherecentroid(build_irecons-1, &
         build_nc+2*build_nhc,build_nc+2*build_nhc)
    real(c_double), intent(in) :: recons_metrics(build_irecons-3, &
         build_nc+2*build_nhe,build_nc+2*build_nhe)
    real(c_double), intent(in) :: recons_metrics_integral(build_irecons-3, &
         build_nc+2*build_nhe,build_nc+2*build_nhe)
    integer(c_int), intent(in) :: jx_min(build_nhr+1),jx_max(build_nhr+1)
    integer(c_int), intent(in) :: jy_min(build_nhr+1),jy_max(build_nhr+1)
    integer(c_int), intent(in) :: ibase(build_nc+2*build_nhr,2,build_nhr)
    real(c_double), intent(in) :: halo_weight(build_ns,build_nc+2*build_nhr,2,build_nhr)
    real(c_double), intent(in) :: centroid_stretch(build_nc+build_nht+1, &
         build_nc+2*build_nhe,build_nc+2*build_nhe)
    real(c_double), intent(in) :: vertex_weights(4,build_irecons-1, &
         build_nc+2*build_nhe,build_nc+2*build_nhe)
    integer(c_int), intent(out) :: errflg
    type(element_t) :: elem(1)
    type(fvm_struct) :: fvm(1)
    type(hybrid_t) :: hybrid
    type(timelevel_t) :: tl
    type(hvcoord_t) :: hvcoord
    type(edgebuffer_t) :: q_buffer,q1_buffer,flux_buffer
    integer :: ierr

    call configure_fvm_dimensions(config%nc,config%nlev,config%ntrac, &
         config%np,config%ngpc,config%irecons,config%nhe,config%nhr, &
         config%nht,config%ns,config%nhc,config%kmin_jet, &
         config%kmax_jet,config%large_courant/=0,irecons_levels,ierr)
    errflg=ierr
    if (ierr/=0) return

    allocate(elem(1)%sub_elem_mass_flux(build_nc,build_nc,4,build_nlev))
    elem(1)%sub_elem_mass_flux=subflux
    allocate(fvm(1)%c(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,build_nlev,build_ntrac)); fvm(1)%c=tracer
    allocate(fvm(1)%dp_fvm(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,build_nlev)); fvm(1)%dp_fvm=dp
    allocate(fvm(1)%psc(build_nc,build_nc)); fvm(1)%psc=psc
    allocate(fvm(1)%se_flux(0:build_nc+1,0:build_nc+1,4,build_nlev)); fvm(1)%se_flux=se_flux
    allocate(fvm(1)%dp_ref(build_nlev)); fvm(1)%dp_ref=dp_ref
    allocate(fvm(1)%dp_ref_inverse(build_nlev)); fvm(1)%dp_ref_inverse=dp_ref_inverse
    allocate(fvm(1)%area_sphere(build_nc,build_nc)); fvm(1)%area_sphere=area
    allocate(fvm(1)%inv_area_sphere(build_nc,build_nc)); fvm(1)%inv_area_sphere=inverse_area
    fvm(1)%cubeboundary=cubeboundary
    allocate(fvm(1)%displ_max(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,4)); fvm(1)%displ_max=displ_max
    allocate(fvm(1)%flux_vec(2,1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,4)); fvm(1)%flux_vec=flux_vec
    allocate(fvm(1)%vtx_cart(4,2,1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc)); fvm(1)%vtx_cart=vtx_cart
    allocate(fvm(1)%flux_orient(2,1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc)); fvm(1)%flux_orient=flux_orient
    allocate(fvm(1)%ifct(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc)); fvm(1)%ifct=ifct
    allocate(fvm(1)%rot_matrix(2,2,1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc)); fvm(1)%rot_matrix=rot_matrix
    allocate(fvm(1)%spherecentroid(build_irecons-1, &
         1-build_nhc:build_nc+build_nhc,1-build_nhc:build_nc+build_nhc)); &
         fvm(1)%spherecentroid=spherecentroid
    allocate(fvm(1)%recons_metrics(build_irecons-3,1-build_nhe:build_nc+build_nhe, &
         1-build_nhe:build_nc+build_nhe)); fvm(1)%recons_metrics=recons_metrics
    allocate(fvm(1)%recons_metrics_integral(build_irecons-3, &
         1-build_nhe:build_nc+build_nhe,1-build_nhe:build_nc+build_nhe)); &
         fvm(1)%recons_metrics_integral=recons_metrics_integral
    fvm(1)%jx_min=jx_min; fvm(1)%jx_max=jx_max
    fvm(1)%jy_min=jy_min; fvm(1)%jy_max=jy_max
    allocate(fvm(1)%ibase(1-build_nhr:build_nc+build_nhr,2,build_nhr)); fvm(1)%ibase=ibase
    allocate(fvm(1)%halo_interp_weight(build_ns,1-build_nhr:build_nc+build_nhr, &
         2,build_nhr)); fvm(1)%halo_interp_weight=halo_weight
    allocate(fvm(1)%centroid_stretch(build_nc+build_nht+1, &
         1-build_nhe:build_nc+build_nhe,1-build_nhe:build_nc+build_nhe)); &
         fvm(1)%centroid_stretch=centroid_stretch
    allocate(fvm(1)%vertex_recons_weights(4,build_irecons-1, &
         1-build_nhe:build_nc+build_nhe,1-build_nhe:build_nc+build_nhe)); &
         fvm(1)%vertex_recons_weights=vertex_weights

    pycam_transport_stage=1
    call run_consistent_se_cslam(elem,fvm,hybrid,dt,tl,1,1,hvcoord, &
         q_buffer,q1_buffer,flux_buffer,config%level_begin,config%level_end)
    pycam_transport_stage=0
    subflux=elem(1)%sub_elem_mass_flux
    tracer=fvm(1)%c
    dp=fvm(1)%dp_fvm
    psc=fvm(1)%psc
    se_flux=fvm(1)%se_flux
    call release_fvm_dimensions()
  end subroutine pycam_sima_fvm_transport_v2

  subroutine pycam_sima_fvm_large_courant_finalize_v1(config, &
       irecons_levels,tracer,dp,se_flux,dp_ref,inverse_area,psc, &
       pressure_top,errflg) &
       bind(C, name="pycam_sima_fvm_large_courant_finalize_v1")
    type(fvm_dimensions_c), intent(in) :: config
    integer(c_int), intent(in) :: irecons_levels(build_nlev)
    real(c_double), intent(inout) :: tracer(build_nc+2*build_nhc, &
         build_nc+2*build_nhc,build_nlev,build_ntrac)
    real(c_double), intent(inout) :: dp(build_nc+2*build_nhc, &
         build_nc+2*build_nhc,build_nlev)
    real(c_double), intent(inout) :: se_flux(build_nc+2, &
         build_nc+2,4,build_nlev)
    real(c_double), intent(in) :: dp_ref(build_nlev)
    real(c_double), intent(in) :: inverse_area(build_nc,build_nc)
    real(c_double), intent(out) :: psc(build_nc,build_nc)
    real(c_double), value, intent(in) :: pressure_top
    integer(c_int), intent(out) :: errflg
    type(fvm_struct) :: fvm
    real(c_double) :: inv_dp_area(build_nc,build_nc)
    integer :: ierr,i,j,k,itr

    call configure_fvm_dimensions(config%nc,config%nlev,config%ntrac, &
         config%np,config%ngpc,config%irecons,config%nhe,config%nhr, &
         config%nht,config%ns,config%nhc,config%kmin_jet, &
         config%kmax_jet,config%large_courant/=0,irecons_levels,ierr)
    errflg=ierr
    if (ierr/=0) return

    allocate(fvm%c(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,build_nlev,build_ntrac))
    fvm%c=tracer
    allocate(fvm%dp_fvm(1-build_nhc:build_nc+build_nhc, &
         1-build_nhc:build_nc+build_nhc,build_nlev))
    fvm%dp_fvm=dp
    allocate(fvm%se_flux(0:build_nc+1,0:build_nc+1,4,build_nlev))
    fvm%se_flux=se_flux
    allocate(fvm%dp_ref(build_nlev))
    fvm%dp_ref=dp_ref
    allocate(fvm%inv_area_sphere(build_nc,build_nc))
    fvm%inv_area_sphere=inverse_area

    if (config%large_courant/=0) then
      do k=config%kmin_jet,config%kmax_jet
        call large_courant_number_increment(fvm,k)
      enddo
    endif

    do k=config%level_begin,config%level_end
      do j=1,build_nc
        do i=1,build_nc
          inv_dp_area(i,j)=1.0_c_double/fvm%dp_fvm(i,j,k)
        enddo
      enddo
      do itr=1,build_ntrac
        do j=1,build_nc
          do i=1,build_nc
            fvm%c(i,j,k,itr)=fvm%c(i,j,k,itr)*inv_dp_area(i,j)
            fvm%c(i,j,k,itr)=max(fvm%c(i,j,k,itr),0.0_c_double)
          enddo
        enddo
      enddo
      fvm%dp_fvm(1:build_nc,1:build_nc,k)= &
           fvm%dp_fvm(1:build_nc,1:build_nc,k)*fvm%dp_ref(k)* &
           fvm%inv_area_sphere
    enddo
    do j=1,build_nc
      do i=1,build_nc
        psc(i,j)=sum(fvm%dp_fvm(i,j,:))+pressure_top
      enddo
    enddo

    tracer=fvm%c
    dp=fvm%dp_fvm
    se_flux=fvm%se_flux
    call release_fvm_dimensions()
  end subroutine pycam_sima_fvm_large_courant_finalize_v1
end module pycam_sima_fvm_transport_kernel
