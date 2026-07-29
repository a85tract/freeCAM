! Source-shaped hydrostatic energy service used by the clean kernel ABI.
module pycam_sima_hydrostatic_energy_reference
  use iso_c_binding, only: c_double, c_int
  implicit none
  private
  public :: hydrostatic_energy_reference
contains
  subroutine hydrostatic_energy_reference( &
       tracer,moist_mixing_ratio,pdel_in,cp_or_cv,U,V,T,vcoord,ptop,phis, &
       reciprocal_gravity,latent_vapor,latent_ice,water_vapor_index, &
       liquid_species,ice_species,te,se,po,ke,wv,H2O,liq,ice)
    real(c_double), intent(in) :: tracer(:,:,:)
    logical, intent(in) :: moist_mixing_ratio
    real(c_double), intent(in) :: pdel_in(:,:),cp_or_cv(:,:)
    real(c_double), intent(in) :: U(:,:),V(:,:),T(:,:)
    integer, intent(in) :: vcoord
    real(c_double), intent(in), optional :: ptop(:),phis(:)
    real(c_double), intent(in) :: reciprocal_gravity
    real(c_double), intent(in) :: latent_vapor,latent_ice
    integer, intent(in) :: water_vapor_index
    integer(c_int), intent(in) :: liquid_species(:),ice_species(:)
    real(c_double), intent(out), optional :: te(:),se(:),po(:),ke(:)
    real(c_double), intent(out), optional :: wv(:),H2O(:),liq(:),ice(:)
    real(c_double) :: ke_vint(size(tracer,1))
    real(c_double) :: se_vint(size(tracer,1))
    real(c_double) :: po_vint(size(tracer,1))
    real(c_double) :: wv_vint(size(tracer,1))
    real(c_double) :: liq_vint(size(tracer,1))
    real(c_double) :: ice_vint(size(tracer,1))
    real(c_double) :: pdel(size(tracer,1),size(tracer,2))
    real(c_double) :: latsub
    integer :: idx,kdx,qdx

    ! Keep a separate compilation boundary and the same assumed-shape and
    ! optional-output structure as CAM cam_thermo:get_hydrostatic_energy_1hd.
    ! The clean ABI passes CAM's module constants explicitly.
    pdel = pdel_in
    ke_vint = 0.0_c_double
    se_vint = 0.0_c_double
    select case(vcoord)
    case(2)
      po_vint = ptop
      do kdx=1,size(tracer,2)
        do idx=1,size(tracer,1)
          ke_vint(idx)=ke_vint(idx)+(pdel(idx,kdx)*0.5_c_double* &
               (U(idx,kdx)**2+V(idx,kdx)**2))*reciprocal_gravity
          se_vint(idx)=se_vint(idx)+(T(idx,kdx)*cp_or_cv(idx,kdx)* &
               pdel(idx,kdx)*reciprocal_gravity)
          po_vint(idx)=po_vint(idx)+pdel(idx,kdx)
        end do
      end do
      do idx=1,size(tracer,1)
        po_vint(idx)=phis(idx)*po_vint(idx)*reciprocal_gravity
      end do
    case default
      error stop "unsupported hydrostatic energy formula"
    end select
    if (present(te)) te=se_vint+po_vint+ke_vint
    if (present(se)) se=se_vint
    if (present(po)) po=po_vint
    if (present(ke)) ke=ke_vint

    wv_vint = 0.0_c_double
    do kdx=1,size(tracer,2)
      do idx=1,size(tracer,1)
        wv_vint(idx)=wv_vint(idx)+(tracer(idx,kdx,water_vapor_index)* &
             pdel(idx,kdx)*reciprocal_gravity)
      end do
    end do
    if (present(wv)) wv=wv_vint

    liq_vint = 0.0_c_double
    do qdx=1,size(tracer,3)
      if (liquid_species(qdx) == 0_c_int) cycle
      do kdx=1,size(tracer,2)
        do idx=1,size(tracer,1)
          liq_vint(idx)=liq_vint(idx)+(pdel(idx,kdx)* &
               tracer(idx,kdx,qdx)*reciprocal_gravity)
        end do
      end do
    end do
    if (present(liq)) liq=liq_vint

    ice_vint = 0.0_c_double
    do qdx=1,size(tracer,3)
      if (ice_species(qdx) == 0_c_int) cycle
      do kdx=1,size(tracer,2)
        do idx=1,size(tracer,1)
          ice_vint(idx)=ice_vint(idx)+(pdel(idx,kdx)* &
               tracer(idx,kdx,qdx)*reciprocal_gravity)
        end do
      end do
    end do
    if (present(ice)) ice=ice_vint
    if (present(H2O)) H2O=wv_vint+liq_vint+ice_vint

    latsub=latent_vapor+latent_ice
    if (present(te)) then
      te=te+(latsub*wv_vint)+(latent_ice*liq_vint)
    end if
  end subroutine hydrostatic_energy_reference
end module pycam_sima_hydrostatic_energy_reference
