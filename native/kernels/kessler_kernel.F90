! Stateless Kessler kernel for the Python model.
module pycam_sima_kessler_kernel
  use iso_c_binding, only: c_double, c_int
  implicit none
  private
  public :: pycam_sima_abi_version, pycam_sima_kessler_v1, pycam_sima_kessler_update_v1
contains
  integer(c_int) function pycam_sima_abi_version() bind(C, name="pycam_sima_abi_version")
    pycam_sima_abi_version = 2_c_int
  end function

  subroutine pycam_sima_kessler_v1(ncol,nz,dt,lv,pref_pa,rhoqr,cpair,rair,rho,z,pk,theta,qv,qc,qr,precl,relhum,errflg) &
       bind(C, name="pycam_sima_kessler_v1")
    integer(c_int), value, intent(in) :: ncol,nz
    real(c_double), value, intent(in) :: dt,lv,pref_pa,rhoqr
    real(c_double), intent(in) :: cpair(ncol,nz),rair(ncol,nz),rho(ncol,nz),z(ncol,nz),pk(ncol,nz)
    real(c_double), intent(inout) :: theta(ncol,nz),qv(ncol,nz),qc(ncol,nz),qr(ncol,nz)
    real(c_double), intent(out) :: precl(ncol),relhum(ncol,nz)
    integer(c_int), intent(out) :: errflg
    real(c_double) :: r(nz),rhalf(nz),velqr(nz),sed(nz),pc(nz)
    real(c_double) :: f5,f2x,xk,ern,qrprod,prod,qvs,dt0,time_counter,precl_acc,pref
    integer :: col,klev
    errflg=0_c_int; precl=0.0_c_double
    if (dt <= 0.0_c_double) then; errflg=1_c_int; return; end if
    pref=pref_pa/100.0_c_double; f2x=17.27_c_double
    do col=1,ncol
      do klev=nz,1,-1
        f5=4093.0_c_double*lv/cpair(col,klev)
        xk=cpair(col,klev)/rair(col,klev)
        r(klev)=0.001_c_double*rho(col,klev)
        rhalf(klev)=sqrt(rho(col,nz)/rho(col,klev))
        pc(klev)=3.8_c_double/((pk(col,klev)**xk)*pref)
        qr(col,klev)=max(qr(col,klev),0.0_c_double)
        velqr(klev)=36.34_c_double*rhalf(klev)*(qr(col,klev)*r(klev))**0.1364_c_double
      end do
      dt0=dt
      do klev=nz,2,-1
        if (abs(velqr(klev)) > 1.0e-12_c_double) dt0=min(dt0,0.8_c_double*(z(col,klev-1)-z(col,klev))/velqr(klev))
      end do
      if (dt0 < 1.0e-12_c_double) then; errflg=2_c_int; return; end if
      time_counter=0.0_c_double; precl_acc=0.0_c_double
      do while (abs(dt-time_counter) > 1.0e-5_c_double)
        precl(col)=rho(col,nz)*qr(col,nz)*velqr(nz)/rhoqr
        precl_acc=precl_acc+precl(col)*dt0
        do klev=nz,2,-1
          sed(klev)=dt0*((r(klev-1)*qr(col,klev-1)*velqr(klev-1))-(r(klev)*qr(col,klev)*velqr(klev))) &
               /(r(klev)*(z(col,klev-1)-z(col,klev)))
        end do
        sed(1)=-dt0*qr(col,1)*velqr(1)/(0.5_c_double*(z(col,1)-z(col,2)))
        do klev=nz,1,-1
          qrprod=qc(col,klev)-(qc(col,klev)-dt0*max(0.001_c_double*(qc(col,klev)-0.001_c_double),0.0_c_double)) &
               /(1.0_c_double+dt0*2.2_c_double*qr(col,klev)**0.875_c_double)
          qc(col,klev)=max(qc(col,klev)-qrprod,0.0_c_double)
          qr(col,klev)=max(qr(col,klev)+qrprod+sed(klev),0.0_c_double)
          qvs=pc(klev)*exp(f2x*(pk(col,klev)*theta(col,klev)-273.0_c_double)/(pk(col,klev)*theta(col,klev)-36.0_c_double))
          prod=(qv(col,klev)-qvs)/(1.0_c_double+qvs*f5/(pk(col,klev)*theta(col,klev)-36.0_c_double)**2)
          ern=min(dt0*(((1.6_c_double+124.9_c_double*(r(klev)*qr(col,klev))**0.2046_c_double) &
               *(r(klev)*qr(col,klev))**0.525_c_double)/(2550000.0_c_double*pc(klev)/(3.8_c_double*qvs)+540000.0_c_double)) &
               *(max(qvs-qv(col,klev),0.0_c_double)/(r(klev)*qvs)),max(-prod-qc(col,klev),0.0_c_double),qr(col,klev))
          theta(col,klev)=theta(col,klev)+lv/(cpair(col,klev)*pk(col,klev))*(max(prod,-qc(col,klev))-ern)
          qv(col,klev)=max(qv(col,klev)-max(prod,-qc(col,klev))+ern,0.0_c_double)
          qc(col,klev)=qc(col,klev)+max(prod,-qc(col,klev))
          qr(col,klev)=max(qr(col,klev)-ern,0.0_c_double)
        end do
        time_counter=time_counter+dt0
        do klev=nz,1,-1
          velqr(klev)=36.34_c_double*rhalf(klev)*(qr(col,klev)*r(klev))**0.1364_c_double
        end do
        dt0=max(dt-time_counter,0.0_c_double)
        do klev=nz,2,-1
          if (abs(velqr(klev)) > 1.0e-12_c_double) dt0=min(dt0,0.8_c_double*(z(col,klev-1)-z(col,klev))/velqr(klev))
        end do
      end do
      precl(col)=precl_acc/dt
      do klev=nz,1,-1
        qvs=pc(klev)*exp(f2x*(pk(col,klev)*theta(col,klev)-273.0_c_double)/(pk(col,klev)*theta(col,klev)-36.0_c_double))
        relhum(col,klev)=qv(col,klev)/qvs*100.0_c_double
      end do
    end do
  end subroutine

  subroutine pycam_sima_kessler_update_v1(ncol,nz,dt,theta,exner,temp_prev,ttend_t) &
       bind(C,name="pycam_sima_kessler_update_v1")
    integer(c_int),value,intent(in)::ncol,nz
    real(c_double),value,intent(in)::dt
    real(c_double),intent(in)::theta(ncol,nz),exner(ncol,nz),temp_prev(ncol,nz)
    real(c_double),intent(inout)::ttend_t(ncol,nz)
    integer::klev
    do klev=1,nz
      ttend_t(:,klev)=ttend_t(:,klev)+(theta(:,klev)*exner(:,klev)-temp_prev(:,klev))/dt
    end do
  end subroutine
end module
