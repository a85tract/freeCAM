! Stateless physics diagnostics for the Python model.
module pycam_sima_physics_diagnostics_kernel
  use iso_c_binding, only: c_double, c_int
  implicit none
contains
  subroutine pycam_sima_physics_diagnostics_v1(ncol,nlev,nconst,update_dse, &
       gravity,zvir,temp,q,pdel,pmid,rair,cpair,phis,zi,zm,dse) &
       bind(C,name="pycam_sima_physics_diagnostics_v1")
    integer(c_int), value, intent(in) :: ncol,nlev,nconst,update_dse
    real(c_double), value, intent(in) :: gravity,zvir
    real(c_double), intent(in) :: temp(ncol,nlev),q(ncol,nlev,nconst)
    real(c_double), intent(in) :: pdel(ncol,nlev),pmid(ncol,nlev)
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
      do klyr=nlev,1,-1
        do icol=1,ncol
          qfac(icol,klyr)=qfac(icol,klyr)-q(icol,klyr,cidx)
        end do
      end do
    end do
    qfac(:,:)=1.0_c_double/qfac(:,:)

    sum_dry(:,:)=1.0_c_double
    do cidx=1,nconst
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
        hkl(icol)=pdel(icol,klyr)/pmid(icol,klyr)
        hkk(icol)=0.5_c_double*hkl(icol)
      end do
      do icol=1,ncol
        tvfac=(1.0_c_double+(zvir+1.0_c_double)*q(icol,klyr,3)* &
             qfac(icol,klyr))*sum_dry(icol,klyr)
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
end module pycam_sima_physics_diagnostics_kernel
