! The CAM spectral-element tracer limiter, kept in its own compilation unit.
!
! Transcribed from the CAM SE dycore routine limiter_optim_iter_full.
! Copyright (c) 2017, University Corporation for Atmospheric Research (UCAR).
! Redistributed under the BSD 3-Clause license in
! LICENSES/UCAR-CESM-BSD-3-Clause.txt.  See NOTICE section 2.
!
! CAM compiles limiter_optim_iter_full separately from its advection caller.
! Keeping that call boundary is numerically significant: inlining the routine
! into the compact C ABI wrapper changes compiler reductions and can move Qdp
! by a few ulps.
module pycam_sima_se_limiter_reference
  use iso_c_binding, only: c_double
  use pycam_sima_build_config, only: build_np, build_ngp, build_nlev
  implicit none
  private
  public :: pycam_limiter_optim_iter_full

contains

  subroutine pycam_limiter_optim_iter_full( &
       ptens,sphweights,minp,maxp,dpmass,kbeg,kend)
    integer, intent(in) :: kbeg,kend
    real(c_double), intent(inout) :: minp(build_nlev),maxp(build_nlev)
    real(c_double), intent(inout) :: ptens(build_ngp,build_nlev)
    real(c_double), intent(in), optional :: dpmass(build_ngp,build_nlev)
    real(c_double), intent(in) :: sphweights(build_ngp)
    real(c_double) :: ptens_mass(build_np,build_np)
    integer :: k1,k,iter,weightsnum
    real(c_double) :: addmass,weightssum,mass,sumc
    real(c_double) :: x(build_ngp),c(build_ngp)
    integer :: maxiter=build_ngp-1
    real(c_double) :: tol_limiter=5.0e-14_c_double

    do k=kbeg,kend
      do k1=1,build_ngp
        c(k1)=sphweights(k1)*dpmass(k1,k)
        x(k1)=ptens(k1,k)/dpmass(k1,k)
      end do

      sumc=sum(c)
      if (sumc<=0.0_c_double) cycle
      mass=sum(c*x)

      if (mass<minp(k)*sumc) then
        minp(k)=mass/sumc
      end if
      if (mass>maxp(k)*sumc) then
        maxp(k)=mass/sumc
      end if

      do iter=1,maxiter
        addmass=0.0_c_double

        do k1=1,build_ngp
          if (x(k1)>maxp(k)) then
            addmass=addmass+(x(k1)-maxp(k))*c(k1)
            x(k1)=maxp(k)
          end if
          if (x(k1)<minp(k)) then
            addmass=addmass-(minp(k)-x(k1))*c(k1)
            x(k1)=minp(k)
          end if
        end do

        if (abs(addmass)<=tol_limiter*abs(mass)) exit

        weightssum=0.0_c_double
        if (addmass>0.0_c_double) then
          do k1=1,build_ngp
            if (x(k1)<maxp(k)) then
              weightssum=weightssum+c(k1)
            end if
          end do
          do k1=1,build_ngp
            if (x(k1)<maxp(k)) then
              x(k1)=x(k1)+addmass/weightssum
            end if
          end do
        else
          do k1=1,build_ngp
            if (x(k1)>minp(k)) then
              weightssum=weightssum+c(k1)
            end if
          end do
          do k1=1,build_ngp
            if (x(k1)>minp(k)) then
              x(k1)=x(k1)+addmass/weightssum
            end if
          end do
        end if
      end do

      do k1=1,build_ngp
        ptens(k1,k)=x(k1)
      end do
    end do

    do k=kbeg,kend
      do k1=1,build_ngp
        ptens(k1,k)=ptens(k1,k)*dpmass(k1,k)
      end do
    end do
  end subroutine pycam_limiter_optim_iter_full

end module pycam_sima_se_limiter_reference
