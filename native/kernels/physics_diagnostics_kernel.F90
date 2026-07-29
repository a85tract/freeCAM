! Stateless physics diagnostics for the Python model.
module pycam_sima_physics_diagnostics_kernel
  use iso_c_binding, only: c_double, c_int
  use pycam_sima_build_config, only: build_np, build_nc, build_nlev
  use pycam_sima_hydrostatic_energy_reference, only: hydrostatic_energy_reference
  implicit none
contains
  subroutine pycam_sima_wet_to_dry_v1(ncol,nlev,nconst,q, &
       water_species,pdel,pdeldry) &
       bind(C,name="pycam_sima_wet_to_dry_v1")
    integer(c_int), value, intent(in) :: ncol,nlev,nconst
    real(c_double), intent(inout) :: q(ncol,nlev,nconst)
    integer(c_int), intent(in) :: water_species(nconst)
    real(c_double), intent(in) :: pdel(ncol,nlev),pdeldry(ncol,nlev)
    real(c_double) :: factor
    integer :: icol,klyr,cidx

    do klyr=1,nlev
      do icol=1,ncol
        factor=pdel(icol,klyr)/pdeldry(icol,klyr)
        do cidx=1,nconst
          if (water_species(cidx) /= 0_c_int) then
            q(icol,klyr,cidx)=factor*q(icol,klyr,cidx)
          end if
        end do
      end do
    end do
  end subroutine pycam_sima_wet_to_dry_v1

  subroutine pycam_sima_physics_diagnostics_v1(ncol,nlev,nconst, &
       water_vapor_index,update_dse,lagrangian_vertical,gravity,zvir, &
       temp,q,water_species,pdel,pmid,rpdel,pint,lnpint,rair,cpair, &
       phis,zi,zm,dse) &
       bind(C,name="pycam_sima_physics_diagnostics_v1")
    integer(c_int), value, intent(in) :: ncol,nlev,nconst
    integer(c_int), value, intent(in) :: water_vapor_index,update_dse
    integer(c_int), value, intent(in) :: lagrangian_vertical
    integer(c_int), intent(in) :: water_species(nconst)
    real(c_double), value, intent(in) :: gravity,zvir
    real(c_double), intent(in) :: temp(ncol,nlev),q(ncol,nlev,nconst)
    real(c_double), intent(in) :: pdel(ncol,nlev),pmid(ncol,nlev)
    real(c_double), intent(in) :: rpdel(ncol,nlev)
    real(c_double), intent(in) :: pint(ncol,nlev+1),lnpint(ncol,nlev+1)
    real(c_double), intent(in) :: rair(ncol,nlev),cpair(ncol,nlev),phis(ncol)
    real(c_double), intent(inout) :: zi(ncol,nlev+1),zm(ncol,nlev),dse(ncol,nlev)
    real(c_double) :: hkk(ncol),hkl(ncol),rog(ncol,nlev)
    real(c_double) :: qfac(ncol,nlev),sum_dry(ncol,nlev)
    real(c_double) :: tv,tvfac
    integer :: icol,klyr,cidx

    rog(:,:) = rair(:,:) / gravity
    zi(:,nlev+1) = 0.0_c_double
    qfac(:,:) = 1.0_c_double
    do cidx=1,nconst
      if (water_species(cidx) == 0_c_int) cycle
      do klyr=nlev,1,-1
        do icol=1,ncol
          qfac(icol,klyr)=qfac(icol,klyr)-q(icol,klyr,cidx)
        end do
      end do
    end do
    qfac(:,:)=1.0_c_double/qfac(:,:)

    sum_dry(:,:)=1.0_c_double
    do cidx=1,nconst
      if (water_species(cidx) == 0_c_int) cycle
      do klyr=nlev,1,-1
        do icol=1,ncol
          sum_dry(icol,klyr)=sum_dry(icol,klyr)+ &
               q(icol,klyr,cidx)*qfac(icol,klyr)
        end do
      end do
    end do
    sum_dry(:,:)=1.0_c_double/sum_dry(:,:)

    do klyr=nlev,1,-1
      do icol=1,ncol
        if (lagrangian_vertical /= 0_c_int) then
          hkl(icol)=lnpint(icol,klyr+1)-lnpint(icol,klyr)
          hkk(icol)=1.0_c_double- &
               pint(icol,klyr)*hkl(icol)*rpdel(icol,klyr)
        else
          hkl(icol)=pdel(icol,klyr)/pmid(icol,klyr)
          hkk(icol)=0.5_c_double*hkl(icol)
        end if
      end do
      do icol=1,ncol
        if (water_vapor_index > 0_c_int) then
          tvfac=(1.0_c_double+(zvir+1.0_c_double)* &
               q(icol,klyr,water_vapor_index)*qfac(icol,klyr))* &
               sum_dry(icol,klyr)
        else
          tvfac=sum_dry(icol,klyr)
        end if
        tv=temp(icol,klyr)*tvfac
        zm(icol,klyr)=zi(icol,klyr+1)+(rog(icol,klyr)*tv*hkk(icol))
        zi(icol,klyr)=zi(icol,klyr+1)+(rog(icol,klyr)*tv*hkl(icol))
      end do
    end do

    if (update_dse /= 0) then
      do klyr=1,nlev
        dse(:,klyr)=(temp(:,klyr)*cpair(:,klyr))+ &
             (gravity*zm(:,klyr))+phis(:)
      end do
    end if
  end subroutine pycam_sima_physics_diagnostics_v1

  subroutine pycam_sima_hydrostatic_energy_v1(ncol,nlev,nconst, &
       water_vapor_index,liquid_species,ice_species,rga,latvap,latice, &
       q,pdel,u,v,temp,temp_ini,cp_phys,cp_dyn,scaling,pintdry,phis, &
       te_phys,te_dyn,total_water) &
       bind(C,name="pycam_sima_hydrostatic_energy_v1")
    integer(c_int), value, intent(in) :: ncol,nlev,nconst
    integer(c_int), value, intent(in) :: water_vapor_index
    integer(c_int), intent(in) :: liquid_species(nconst),ice_species(nconst)
    real(c_double), value, intent(in) :: rga,latvap,latice
    real(c_double), intent(in) :: q(ncol,nlev,nconst)
    real(c_double), intent(in) :: pdel(ncol,nlev)
    real(c_double), intent(in) :: u(ncol,nlev),v(ncol,nlev)
    real(c_double), intent(in) :: temp(ncol,nlev),temp_ini(ncol,nlev)
    real(c_double), intent(in) :: cp_phys(ncol,nlev),cp_dyn(ncol,nlev)
    real(c_double), intent(in) :: scaling(ncol,nlev)
    real(c_double), intent(in) :: pintdry(ncol,nlev+1),phis(ncol)
    real(c_double), intent(out) :: te_phys(ncol),te_dyn(ncol)
    real(c_double), intent(out) :: total_water(ncol)
    real(c_double) :: temp_dyn(ncol,nlev)
    integer, parameter :: energy_formula_dycore_se = 2

    call hydrostatic_energy_reference( &
         tracer=q, moist_mixing_ratio=.true., pdel_in=pdel, &
         cp_or_cv=cp_phys, U=u, V=v, T=temp, &
         vcoord=energy_formula_dycore_se, ptop=pintdry(:,1), phis=phis, &
         reciprocal_gravity=rga,latent_vapor=latvap,latent_ice=latice, &
         water_vapor_index=water_vapor_index,liquid_species=liquid_species, &
         ice_species=ice_species, &
         te=te_phys, H2O=total_water)

    ! Preserve the array-expression boundary in check_energy_chng_run.
    temp_dyn(:,:) = temp_ini(:,:) + scaling(:,:) * (temp(:,:) - temp_ini(:,:))
    call hydrostatic_energy_reference( &
         tracer=q, moist_mixing_ratio=.true., pdel_in=pdel, &
         cp_or_cv=cp_dyn, U=u, V=v, T=temp_dyn, &
         vcoord=energy_formula_dycore_se, ptop=pintdry(:,1), phis=phis, &
         reciprocal_gravity=rga,latent_vapor=latvap,latent_ice=latice, &
         water_vapor_index=water_vapor_index,liquid_species=liquid_species, &
         ice_species=ice_species, &
         te=te_dyn)
  end subroutine pycam_sima_hydrostatic_energy_v1

  subroutine pycam_sima_dyn2phys_thermo_vector_v1(ngll,nphys,nlev,nelem, &
       temperature,pressure,zonal,meridional,metric,inverse_metric, &
       integration,interpolation,nodes,derivative, &
       temperature_physics,zonal_physics,meridional_physics, &
       pressure_physics) &
       bind(C,name="pycam_sima_dyn2phys_thermo_vector_v1")
    integer(c_int), value, intent(in) :: ngll,nphys,nlev,nelem
    ! The pinned CAM routine compiles NP, FV_NPHYS, and NLEV as module
    ! parameters.  Keep those extents compile-time constants here as well:
    ! gfortran otherwise lowers the small MATMUL expressions through a
    ! different dynamic-shape path and changes a few final bits.
    real(c_double), intent(in) :: temperature(build_np,build_np,build_nlev,nelem)
    real(c_double), intent(in) :: pressure(build_np,build_np,build_nlev,nelem)
    real(c_double), intent(in) :: zonal(build_np,build_np,build_nlev,nelem)
    real(c_double), intent(in) :: meridional(build_np,build_np,build_nlev,nelem)
    real(c_double), intent(in) :: metric(build_np,build_np,nelem)
    real(c_double), intent(in) :: inverse_metric(2,2,build_np,build_np,nelem)
    real(c_double), intent(in) :: integration(build_nc,build_np)
    real(c_double), intent(in) :: interpolation(build_np,build_np)
    real(c_double), intent(in) :: nodes(build_nc)
    real(c_double), intent(in) :: derivative(2,2,build_nc*build_nc,nelem)
    real(c_double), intent(out) :: temperature_physics(build_nc*build_nc,nelem,build_nlev)
    real(c_double), intent(out) :: zonal_physics(build_nc*build_nc,nelem,build_nlev)
    real(c_double), intent(out) :: meridional_physics(build_nc*build_nc,nelem,build_nlev)
    real(c_double), intent(out) :: pressure_physics(build_nc*build_nc,nelem,build_nlev)
    real(c_double) :: sampled(build_np,build_np)
    real(c_double) :: area(build_nc,build_nc),integrated(build_nc,build_nc)
    real(c_double) :: pressure_integrated(build_nc,build_nc)
    real(c_double) :: inverse_area(build_nc,build_nc)
    real(c_double) :: inverse_mass_area(build_nc,build_nc)
    real(c_double) :: contra1(build_np,build_np),contra2(build_np,build_np)
    real(c_double) :: v1,v2
    integer :: element,level,i,j,column

    sampled = 1.0_c_double
    do element=1,nelem
      area = dyn2phys_reference( &
           sampled,metric(:,:,element),integration)
      inverse_area = 1.0_c_double / area

      do level=1,build_nlev
        pressure_integrated = dyn2phys_reference( &
             pressure(:,:,level,element),metric(:,:,element), &
             integration,inverse_area)
        pressure_physics(:,element,level) = reshape( &
             pressure_integrated, &
             shape(pressure_physics(:,element,level)))
        inverse_mass_area = inverse_area / pressure_integrated

        integrated = dyn2phys_reference( &
             temperature(:,:,level,element)* &
             pressure(:,:,level,element),metric(:,:,element), &
             integration,inverse_mass_area)
        temperature_physics(:,element,level) = reshape( &
             integrated,shape(temperature_physics(:,element,level)))

        do j=1,build_np
          do i=1,build_np
            contra1(i,j)=inverse_metric(1,1,i,j,element)* &
                 zonal(i,j,level,element)+ &
                 inverse_metric(1,2,i,j,element)* &
                 meridional(i,j,level,element)
            contra2(i,j)=inverse_metric(2,1,i,j,element)* &
                 zonal(i,j,level,element)+ &
                 inverse_metric(2,2,i,j,element)* &
                 meridional(i,j,level,element)
          end do
        end do
        column=0
        do j=1,build_nc
          do i=1,build_nc
            column=column+1
            v1=interpolate_reference( &
                 contra1,nodes(i),nodes(j),interpolation)
            v2=interpolate_reference( &
                 contra2,nodes(i),nodes(j),interpolation)
            zonal_physics(column,element,level)= &
                 derivative(1,1,column,element)*v1+ &
                 derivative(1,2,column,element)*v2
            meridional_physics(column,element,level)= &
                 derivative(2,1,column,element)*v1+ &
                 derivative(2,2,column,element)*v2
          end do
        end do
      end do
    end do

  contains

    function dyn2phys_reference( &
         sampled_value,metric_value,integration_value,inverse_mass) &
         result(output)
      real(c_double), intent(in) :: sampled_value(:,:),metric_value(:,:)
      real(c_double), intent(in) :: integration_value(:,:)
      real(c_double), intent(in), optional :: inverse_mass(:,:)
      real(c_double) :: output(size(integration_value,1), &
           size(integration_value,1))
      real(c_double) :: weighted(size(sampled_value,1), &
           size(sampled_value,2))

      weighted=sampled_value*metric_value
      output=matmul(integration_value, &
           matmul(weighted,transpose(integration_value)))
      if (present(inverse_mass)) output=output*inverse_mass
    end function dyn2phys_reference

    function interpolate_reference(field,x,y,matrix) result(value)
      real(c_double), intent(in) :: field(build_np,build_np)
      real(c_double), value, intent(in) :: x,y
      real(c_double), intent(in) :: matrix(build_np,build_np)
      real(c_double) :: value
      real(c_double) :: vtemp(build_np),tmp1,tmp2,fk0,fk1,pk
      integer :: l,j,k

      do l=1,build_np,2
        pk=1.0_c_double
        fk0=0.0_c_double
        fk1=0.0_c_double
        do j=1,build_np
          fk0=fk0+matrix(j,1)*field(j,l)
          fk1=fk1+matrix(j,1)*field(j,l+1)
        end do
        vtemp(l)=pk*fk0
        vtemp(l+1)=pk*fk1

        tmp2=pk
        pk=x
        fk0=0.0_c_double
        fk1=0.0_c_double
        do j=1,build_np
          fk0=fk0+matrix(j,2)*field(j,l)
          fk1=fk1+matrix(j,2)*field(j,l+1)
        end do
        vtemp(l)=vtemp(l)+pk*fk0
        vtemp(l+1)=vtemp(l+1)+pk*fk1

        do k=2,build_np-1
          tmp1=tmp2
          tmp2=pk
          pk=((2*k-1)*x*tmp2-(k-1)*tmp1)* &
               (1.0_c_double/real(k,c_double))
          fk0=0.0_c_double
          fk1=0.0_c_double
          do j=1,build_np
            fk0=fk0+matrix(j,k+1)*field(j,l)
            fk1=fk1+matrix(j,k+1)*field(j,l+1)
          end do
          vtemp(l)=vtemp(l)+pk*fk0
          vtemp(l+1)=vtemp(l+1)+pk*fk1
        end do
      end do

      pk=1.0_c_double
      fk0=0.0_c_double
      do j=1,build_np
        fk0=fk0+matrix(j,1)*vtemp(j)
      end do
      value=pk*fk0
      tmp2=pk
      pk=y
      fk0=0.0_c_double
      do j=1,build_np
        fk0=fk0+matrix(j,2)*vtemp(j)
      end do
      value=value+pk*fk0
      do k=2,build_np-1
        tmp1=tmp2
        tmp2=pk
        pk=((2*k-1)*y*tmp2-(k-1)*tmp1)* &
             (1.0_c_double/real(k,c_double))
        fk0=0.0_c_double
        do j=1,build_np
          fk0=fk0+matrix(j,k+1)*vtemp(j)
        end do
        value=value+pk*fk0
      end do
    end function interpolate_reference
  end subroutine pycam_sima_dyn2phys_thermo_vector_v1

  subroutine pycam_sima_reference_pressure_thickness_v1(ngll,nlev,nelem, &
       hybrid_a_interface,hybrid_b_interface,reference_pressure, &
       source_pressure_thickness,surface_dry_air_pressure,pressure_thickness) &
       bind(C,name="pycam_sima_reference_pressure_thickness_v1")
    integer(c_int), value, intent(in) :: ngll,nlev,nelem
    real(c_double), intent(in) :: hybrid_a_interface(build_nlev+1)
    real(c_double), intent(in) :: hybrid_b_interface(build_nlev+1)
    real(c_double), value, intent(in) :: reference_pressure
    real(c_double), intent(in) :: &
         source_pressure_thickness(build_np,build_np,build_nlev,nelem)
    real(c_double), intent(out) :: &
         surface_dry_air_pressure(build_np,build_np,nelem)
    real(c_double), intent(out) :: &
         pressure_thickness(build_np,build_np,build_nlev,nelem)
    real(c_double) :: pressure_top
    integer :: element,level

    if (ngll /= build_np .or. nlev /= build_nlev) then
      surface_dry_air_pressure = 0.0_c_double
      pressure_thickness = 0.0_c_double
      return
    end if
    pressure_top=hybrid_a_interface(1)*reference_pressure
    do element=1,nelem
      surface_dry_air_pressure(:,:,element) = pressure_top + &
           sum(source_pressure_thickness(:,:,:,element),3)
      do level=1,build_nlev
        pressure_thickness(:,:,level,element) = &
             (hybrid_a_interface(level+1)-hybrid_a_interface(level))* &
             reference_pressure + &
             (hybrid_b_interface(level+1)-hybrid_b_interface(level))* &
             surface_dry_air_pressure(:,:,element)
      end do
    end do
  end subroutine pycam_sima_reference_pressure_thickness_v1
end module pycam_sima_physics_diagnostics_kernel
