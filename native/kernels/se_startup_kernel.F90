! Stateless spectral-element kernels for the Python model.
module pycam_sima_se_startup_kernel
  use iso_c_binding, only: c_double, c_int
  use pycam_sima_build_config, only: build_np,build_ngp,build_nlev
  use pycam_sima_hypervis_reference_support, only: pycam_get_dp_ref
  use pycam_sima_se_limiter_reference, only: pycam_limiter_optim_iter_full
  implicit none
  private
  public :: pycam_sima_laplace_weak_v2
  public :: pycam_sima_divergence_sphere_v2
  public :: pycam_sima_tracer_flux_v2
  public :: pycam_sima_apply_tracer_forcing_v2
  public :: pycam_sima_scale_tracer_forcing_v2
  public :: pycam_sima_wind_tendency_v2
  public :: pycam_sima_vector_laplace_weak_v2
  public :: pycam_sima_hypervis_reference_v2
  public :: pycam_sima_limiter_optim_v2
  public :: pycam_sima_prepare_qwater_v1
  public :: pycam_sima_validate_se_dimensions_v2

contains

  subroutine pycam_sima_prepare_qwater_v1(np,nlev,nelem,nmass,nwater,qsize, &
       constituent_mass,pressure_thickness,qwater) &
       bind(C,name="pycam_sima_prepare_qwater_v1")
    integer(c_int), value, intent(in) :: np,nlev,nelem,nmass,nwater,qsize
    real(c_double), intent(in) :: constituent_mass(np,np,nlev,nelem,nmass)
    real(c_double), intent(in) :: pressure_thickness(np,np,nlev,nelem)
    real(c_double), intent(out) :: qwater(np,np,nlev,nelem,nwater)
    integer :: ie,nq

    qwater=0.0_c_double
    do ie=1,nelem
      do nq=1,qsize
        qwater(:,:,:,ie,nq)=constituent_mass(:,:,:,ie,nq) / &
             pressure_thickness(:,:,:,ie)
      end do
    end do
  end subroutine pycam_sima_prepare_qwater_v1

  subroutine pycam_sima_validate_se_dimensions_v2(np,ngp,errflg) &
       bind(C,name="pycam_sima_validate_se_dimensions_v2")
    integer(c_int), value, intent(in) :: np,ngp
    integer(c_int), intent(out) :: errflg
    errflg=0
    if (np/=build_np .or. ngp/=build_ngp) errflg=1
  end subroutine pycam_sima_validate_se_dimensions_v2

  subroutine gradient_sphere(s, dvv, dinv, ra, ds)
    real(c_double), intent(in) :: s(build_np,build_np),dvv(build_np,build_np)
    real(c_double), intent(in) :: dinv(2,2,build_np,build_np),ra
    real(c_double), intent(out) :: ds(build_np,build_np,2)
    real(c_double) :: dsdx00,dsdy00,v1(build_np,build_np),v2(build_np,build_np)
    integer :: i,j,l
    do j=1,build_np
      do l=1,build_np
        dsdx00=0.0_c_double
        dsdy00=0.0_c_double
        do i=1,build_np
          dsdx00=dsdx00+dvv(i,l)*s(i,j)
          dsdy00=dsdy00+dvv(i,l)*s(j,i)
        end do
        v1(l,j)=dsdx00*ra
        v2(j,l)=dsdy00*ra
      end do
    end do
    do j=1,build_np
      do i=1,build_np
        ds(i,j,1)=dinv(1,1,i,j)*v1(i,j)+dinv(2,1,i,j)*v2(i,j)
        ds(i,j,2)=dinv(1,2,i,j)*v1(i,j)+dinv(2,2,i,j)*v2(i,j)
      end do
    end do
  end subroutine gradient_sphere
  subroutine divergence_sphere_weak(v, dvv, dinv, spheremp, ra, div)
    real(c_double), intent(in) :: v(build_np,build_np,2),dvv(build_np,build_np)
    real(c_double), intent(in) :: dinv(2,2,build_np,build_np),spheremp(build_np,build_np),ra
    real(c_double), intent(out) :: div(build_np,build_np)
    real(c_double) :: vtemp(build_np,build_np,2)
    integer :: i,j,m,n
    do j=1,build_np
      do i=1,build_np
        vtemp(i,j,1)=dinv(1,1,i,j)*v(i,j,1)+dinv(1,2,i,j)*v(i,j,2)
        vtemp(i,j,2)=dinv(2,1,i,j)*v(i,j,1)+dinv(2,2,i,j)*v(i,j,2)
      end do
    end do
    do n=1,build_np
      do m=1,build_np
        div(m,n)=0.0_c_double
        do j=1,build_np
          div(m,n)=div(m,n)-(spheremp(j,n)*vtemp(j,n,1)*dvv(m,j) &
               +spheremp(m,j)*vtemp(m,j,2)*dvv(n,j))*ra
        end do
      end do
    end do
  end subroutine divergence_sphere_weak
  subroutine pycam_sima_laplace_weak_v2(nelem,nlev,np,ra,dvv,dinv,spheremp,scalar,laplace) &
       bind(C,name="pycam_sima_laplace_weak_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,np
    real(c_double), value, intent(in) :: ra
    real(c_double), intent(in) :: dvv(build_np,build_np)
    real(c_double), intent(in) :: dinv(2,2,build_np,build_np,nelem)
    real(c_double), intent(in) :: spheremp(build_np,build_np,nelem)
    real(c_double), intent(in) :: scalar(build_np,build_np,nlev,nelem)
    real(c_double), intent(out) :: laplace(build_np,build_np,nlev,nelem)
    real(c_double) :: grads(build_np,build_np,2)
    integer :: k,ie
    if (np/=build_np) return
    do ie=1,nelem
      do k=1,nlev
        call gradient_sphere(scalar(:,:,k,ie),dvv,dinv(:,:,:,:,ie),ra,grads)
        call divergence_sphere_weak(grads,dvv,dinv(:,:,:,:,ie),spheremp(:,:,ie),ra,laplace(:,:,k,ie))
      end do
    end do
  end subroutine pycam_sima_laplace_weak_v2

  subroutine divergence_sphere_rmet(v,dvv,dinv,metdet,rmetdet,ra,div)
    real(c_double), intent(in) :: v(build_np,build_np,2),dvv(build_np,build_np)
    real(c_double), intent(in) :: dinv(2,2,build_np,build_np)
    real(c_double), intent(in) :: metdet(build_np,build_np),rmetdet(build_np,build_np),ra
    real(c_double), intent(out) :: div(build_np,build_np)
    real(c_double) :: dudx00,dvdy00,gv(build_np,build_np,2),vvtemp(build_np,build_np)
    integer :: i,j,l
    do j=1,build_np
      do i=1,build_np
        gv(i,j,1)=metdet(i,j)*(dinv(1,1,i,j)*v(i,j,1)+dinv(1,2,i,j)*v(i,j,2))
        gv(i,j,2)=metdet(i,j)*(dinv(2,1,i,j)*v(i,j,1)+dinv(2,2,i,j)*v(i,j,2))
      end do
    end do
    do j=1,build_np
      do l=1,build_np
        dudx00=0.0_c_double
        dvdy00=0.0_c_double
        do i=1,build_np
          dudx00=dudx00+dvv(i,l)*gv(i,j,1)
          dvdy00=dvdy00+dvv(i,l)*gv(j,i,2)
        end do
        div(l,j)=dudx00
        vvtemp(j,l)=dvdy00
      end do
    end do
    do j=1,build_np
      do i=1,build_np
        div(i,j)=(div(i,j)+vvtemp(i,j))*(rmetdet(i,j)*ra)
      end do
    end do
  end subroutine divergence_sphere_rmet

  subroutine pycam_sima_divergence_sphere_v2( &
       nelem,nfield,np,ra,dvv,dinv,metdet,rmetdet,vector,divergence) &
       bind(C,name="pycam_sima_divergence_sphere_v2")
    integer(c_int), value, intent(in) :: nelem,nfield,np
    real(c_double), value, intent(in) :: ra
    real(c_double), intent(in) :: dvv(build_np,build_np)
    real(c_double), intent(in) :: dinv(2,2,build_np,build_np,nelem)
    real(c_double), intent(in) :: metdet(build_np,build_np,nelem)
    real(c_double), intent(in) :: rmetdet(build_np,build_np,nelem)
    real(c_double), intent(in) :: vector(build_np,build_np,2,nfield,nelem)
    real(c_double), intent(out) :: divergence(build_np,build_np,nfield,nelem)
    integer :: ie,field
    if (np/=build_np) return
    do ie=1,nelem
      do field=1,nfield
        call divergence_sphere_rmet( &
             vector(:,:,:,field,ie),dvv,dinv(:,:,:,:,ie), &
             metdet(:,:,ie),rmetdet(:,:,ie),ra,divergence(:,:,field,ie))
      end do
    end do
  end subroutine pycam_sima_divergence_sphere_v2

  subroutine pycam_sima_tracer_flux_v2( &
       nelem,nlev,nq,np,dt,rhs_multiplier,pressure_start, &
       projected_divergence,mean_mass_flux,source_qdp,tracer_flux) &
       bind(C,name="pycam_sima_tracer_flux_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,nq,np,rhs_multiplier
    real(c_double), value, intent(in) :: dt
    real(c_double), intent(in) :: pressure_start(build_np,build_np,nlev,nelem)
    real(c_double), intent(in) :: projected_divergence(build_np,build_np,nlev,nelem)
    real(c_double), intent(in) :: mean_mass_flux(build_np,build_np,2,nlev,nelem)
    real(c_double), intent(in) :: source_qdp(build_np,build_np,nlev,nq,nelem)
    real(c_double), intent(out) :: tracer_flux(build_np,build_np,2,nlev,nq,nelem)
    real(c_double) :: dp(build_np,build_np,nlev)
    real(c_double) :: vstar(build_np,build_np,2,nlev)
    integer :: ie,q,k,i,j
    if (np/=build_np) return
    do ie=1,nelem
      do k=1,nlev
        do j=1,build_np
          do i=1,build_np
            dp(i,j,k)=pressure_start(i,j,k,ie)- &
                 rhs_multiplier*dt*projected_divergence(i,j,k,ie)
            vstar(i,j,1,k)=mean_mass_flux(i,j,1,k,ie)/dp(i,j,k)
            vstar(i,j,2,k)=mean_mass_flux(i,j,2,k,ie)/dp(i,j,k)
          end do
        end do
      end do
      do q=1,nq
        do k=1,nlev
          do j=1,build_np
            do i=1,build_np
              tracer_flux(i,j,1,k,q,ie)= &
                   vstar(i,j,1,k)*source_qdp(i,j,k,q,ie)
              tracer_flux(i,j,2,k,q,ie)= &
                   vstar(i,j,2,k)*source_qdp(i,j,k,q,ie)
            end do
          end do
        end do
      end do
    end do
  end subroutine pycam_sima_tracer_flux_v2

  subroutine pycam_sima_apply_tracer_forcing_v2( &
       nelem,nlev,nq,np,dt,qdp,forcing) &
       bind(C,name="pycam_sima_apply_tracer_forcing_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,nq,np
    real(c_double), value, intent(in) :: dt
    real(c_double), intent(inout) :: qdp(build_np,build_np,nlev,nq,nelem)
    real(c_double), intent(in) :: forcing(build_np,build_np,nlev,nq,nelem)
    real(c_double) :: v1
    integer :: ie,q,k,i,j
    if (np/=build_np) return
    do ie=1,nelem
      do q=1,nq
        do k=1,nlev
          do j=1,build_np
            do i=1,build_np
              v1=dt*forcing(i,j,k,q,ie)
              if (qdp(i,j,k,q,ie)+v1<0.0_c_double .and. &
                  v1<0.0_c_double) then
                if (qdp(i,j,k,q,ie)<0.0_c_double) then
                  v1=0.0_c_double
                else
                  v1=-qdp(i,j,k,q,ie)
                end if
              end if
              qdp(i,j,k,q,ie)=qdp(i,j,k,q,ie)+v1
            end do
          end do
        end do
      end do
    end do
  end subroutine pycam_sima_apply_tracer_forcing_v2

  subroutine pycam_sima_scale_tracer_forcing_v2( &
       nelem,nlev,nq,np,reciprocal_timestep,forcing,pressure_thickness) &
       bind(C,name="pycam_sima_scale_tracer_forcing_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,nq,np
    real(c_double), value, intent(in) :: reciprocal_timestep
    real(c_double), intent(inout) :: forcing(build_np,build_np,nlev,nq,nelem)
    real(c_double), intent(in) :: pressure_thickness(build_np,build_np,nlev,nelem)
    integer :: ie,q,k,i,j
    if (np/=build_np) return
    do ie=1,nelem
      do q=1,nq
        do k=1,nlev
          do j=1,build_np
            do i=1,build_np
              forcing(i,j,k,q,ie)=forcing(i,j,k,q,ie)* &
                   reciprocal_timestep*pressure_thickness(i,j,k,ie)
            end do
          end do
        end do
      end do
    end do
  end subroutine pycam_sima_scale_tracer_forcing_v2

  subroutine vorticity_sphere_local(v,dvv,dmat,rmetdet,ra,vort)
    real(c_double), intent(in) :: v(build_np,build_np,2),dvv(build_np,build_np)
    real(c_double), intent(in) :: dmat(2,2,build_np,build_np),rmetdet(build_np,build_np),ra
    real(c_double), intent(out) :: vort(build_np,build_np)
    real(c_double) :: dvdx00,dudy00,vco(build_np,build_np,2),vtemp(build_np,build_np)
    integer :: i,j,l
    do j=1,build_np
      do i=1,build_np
        vco(i,j,1)=dmat(1,1,i,j)*v(i,j,1)+dmat(2,1,i,j)*v(i,j,2)
        vco(i,j,2)=dmat(1,2,i,j)*v(i,j,1)+dmat(2,2,i,j)*v(i,j,2)
      end do
    end do
    do j=1,build_np
      do l=1,build_np
        dudy00=0.0_c_double
        dvdx00=0.0_c_double
        do i=1,build_np
          dvdx00=dvdx00+dvv(i,l)*vco(i,j,2)
          dudy00=dudy00+dvv(i,l)*vco(j,i,1)
        end do
        vort(l,j)=dvdx00
        vtemp(j,l)=dudy00
      end do
    end do
    do j=1,build_np
      do i=1,build_np
        vort(i,j)=(vort(i,j)-vtemp(i,j))*(rmetdet(i,j)*ra)
      end do
    end do
  end subroutine vorticity_sphere_local

  subroutine gradient_wk_testcov(s,dvv,dmat,metinv,metdet,mp,ra,ds)
    real(c_double), intent(in) :: s(build_np,build_np),dvv(build_np,build_np)
    real(c_double), intent(in) :: dmat(2,2,build_np,build_np),metinv(2,2,build_np,build_np)
    real(c_double), intent(in) :: metdet(build_np,build_np),mp(build_np,build_np),ra
    real(c_double), intent(out) :: ds(build_np,build_np,2)
    real(c_double) :: dscontra(build_np,build_np,2)
    integer :: i,j,m,n
    dscontra=0.0_c_double
    do n=1,build_np
      do m=1,build_np
        do j=1,build_np
          dscontra(m,n,1)=dscontra(m,n,1)-( &
               mp(j,n)*metinv(1,1,m,n)*metdet(m,n)*s(j,n)*dvv(m,j) + &
               mp(m,j)*metinv(2,1,m,n)*metdet(m,n)*s(m,j)*dvv(n,j))*ra
          dscontra(m,n,2)=dscontra(m,n,2)-( &
               mp(j,n)*metinv(1,2,m,n)*metdet(m,n)*s(j,n)*dvv(m,j) + &
               mp(m,j)*metinv(2,2,m,n)*metdet(m,n)*s(m,j)*dvv(n,j))*ra
        end do
      end do
    end do
    do j=1,build_np
      do i=1,build_np
        ds(i,j,1)=dmat(1,1,i,j)*dscontra(i,j,1)+dmat(1,2,i,j)*dscontra(i,j,2)
        ds(i,j,2)=dmat(2,1,i,j)*dscontra(i,j,1)+dmat(2,2,i,j)*dscontra(i,j,2)
      end do
    end do
  end subroutine gradient_wk_testcov

  subroutine curl_wk_testcov(s,dvv,dmat,mp,ra,ds)
    real(c_double), intent(in) :: s(build_np,build_np),dvv(build_np,build_np)
    real(c_double), intent(in) :: dmat(2,2,build_np,build_np),mp(build_np,build_np),ra
    real(c_double), intent(out) :: ds(build_np,build_np,2)
    real(c_double) :: dscontra(build_np,build_np,2)
    integer :: i,j,m,n
    dscontra=0.0_c_double
    do n=1,build_np
      do m=1,build_np
        do j=1,build_np
          dscontra(m,n,1)=dscontra(m,n,1)-(mp(m,j)*s(m,j)*dvv(n,j))*ra
          dscontra(m,n,2)=dscontra(m,n,2)+(mp(j,n)*s(j,n)*dvv(m,j))*ra
        end do
      end do
    end do
    do j=1,build_np
      do i=1,build_np
        ds(i,j,1)=dmat(1,1,i,j)*dscontra(i,j,1)+dmat(1,2,i,j)*dscontra(i,j,2)
        ds(i,j,2)=dmat(2,1,i,j)*dscontra(i,j,1)+dmat(2,2,i,j)*dscontra(i,j,2)
      end do
    end do
  end subroutine curl_wk_testcov

  subroutine vector_laplace_weak_one(v,dvv,dmat,dinv,metinv,metdet,rmetdet,spheremp,mp,ra,nu_ratio,laplace)
    real(c_double), intent(in) :: v(build_np,build_np,2),dvv(build_np,build_np)
    real(c_double), intent(in) :: dmat(2,2,build_np,build_np),dinv(2,2,build_np,build_np)
    real(c_double), intent(in) :: metinv(2,2,build_np,build_np)
    real(c_double), intent(in) :: metdet(build_np,build_np),rmetdet(build_np,build_np)
    real(c_double), intent(in) :: spheremp(build_np,build_np),mp(build_np,build_np),ra,nu_ratio
    real(c_double), intent(out) :: laplace(build_np,build_np,2)
    real(c_double) :: vor(build_np,build_np),div(build_np,build_np)
    real(c_double) :: graddiv(build_np,build_np,2),curlvor(build_np,build_np,2),ra_sq
    integer :: m,n
    ra_sq=ra**2.0_c_double
    call divergence_sphere_rmet(v,dvv,dinv,metdet,rmetdet,ra,div)
    call vorticity_sphere_local(v,dvv,dmat,rmetdet,ra,vor)
    div=nu_ratio*div
    call gradient_wk_testcov(div,dvv,dmat,metinv,metdet,mp,ra,graddiv)
    call curl_wk_testcov(vor,dvv,dmat,mp,ra,curlvor)
    laplace=graddiv-curlvor
    do n=1,build_np
      do m=1,build_np
        laplace(m,n,1)=laplace(m,n,1)+2.0_c_double*spheremp(m,n)*v(m,n,1)*ra_sq
        laplace(m,n,2)=laplace(m,n,2)+2.0_c_double*spheremp(m,n)*v(m,n,2)*ra_sq
      end do
    end do
  end subroutine vector_laplace_weak_one

  subroutine pycam_sima_vector_laplace_weak_v2(nelem,nlev,np,ra,nu_ratio,dvv,dmat,dinv,metinv,metdet,rmetdet,spheremp,mp,vector,laplace) &
       bind(C,name="pycam_sima_vector_laplace_weak_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,np
    real(c_double), value, intent(in) :: ra,nu_ratio
    real(c_double), intent(in) :: dvv(build_np,build_np)
    real(c_double), intent(in) :: dmat(2,2,build_np,build_np,nelem),dinv(2,2,build_np,build_np,nelem)
    real(c_double), intent(in) :: metinv(2,2,build_np,build_np,nelem)
    real(c_double), intent(in) :: metdet(build_np,build_np,nelem),rmetdet(build_np,build_np,nelem)
    real(c_double), intent(in) :: spheremp(build_np,build_np,nelem),mp(build_np,build_np)
    real(c_double), intent(in) :: vector(build_np,build_np,2,nlev,nelem)
    real(c_double), intent(out) :: laplace(build_np,build_np,2,nlev,nelem)
    integer :: ie,k
    if (np/=build_np) return
    do ie=1,nelem
      do k=1,nlev
        call vector_laplace_weak_one(vector(:,:,:,k,ie),dvv,dmat(:,:,:,:,ie),dinv(:,:,:,:,ie),metinv(:,:,:,:,ie), &
             metdet(:,:,ie),rmetdet(:,:,ie),spheremp(:,:,ie),mp,ra,nu_ratio,laplace(:,:,:,k,ie))
      end do
    end do
  end subroutine pycam_sima_vector_laplace_weak_v2

  subroutine pycam_sima_hypervis_reference_v2(nelem,nlev,np,ps0,rair,cpair,gravit,tref,lapse,cappa, &
       hyai,hybi,hyam,hybm,phis,dp_ref,t_ref,ps_ref,nu_scale_top) bind(C,name="pycam_sima_hypervis_reference_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,np
    real(c_double), value, intent(in) :: ps0,rair,cpair,gravit,tref,lapse,cappa
    real(c_double), intent(in) :: hyai(nlev+1),hybi(nlev+1),hyam(nlev),hybm(nlev)
    real(c_double), intent(in) :: phis(build_np,build_np,nelem)
    real(c_double), intent(out) :: dp_ref(build_np,build_np,nlev,nelem)
    real(c_double), intent(out) :: t_ref(build_np,build_np,nlev,nelem)
    real(c_double), intent(out) :: ps_ref(build_np,build_np,nelem),nu_scale_top(nlev)
    real(c_double) :: t0,t1,tmp(build_np,build_np),tmp2(build_np,build_np),pressure,ptop
    integer :: ie,k
    if (np/=build_np) return
    t1=lapse*tref*cpair/gravit
    t0=tref-t1
    ptop=hyai(1)*ps0
    do ie=1,nelem
      ! CAM computes this through get_dp_ref_2hd -> get_dp_ref_1hd.  Preserve
      ! that assumed-shape call boundary so GNU uses the identical EXP path.
      call pycam_get_dp_ref( &
           hyai,hybi,ps0,phis(:,:,ie),rair,tref,dp_ref(:,:,:,ie),ps_ref(:,:,ie))
      do k=1,nlev
        dp_ref(:,:,k,ie)=(hyai(k+1)-hyai(k))*ps0+(hybi(k+1)-hybi(k))*ps_ref(:,:,ie)
        tmp=hyam(k)*ps0+hybm(k)*ps_ref(:,:,ie)
        tmp2=(tmp/ps0)**cappa
        t_ref(:,:,k,ie)=t0+t1*tmp2
      end do
    end do
    do k=1,nlev
      pressure=(hyam(k)+hybm(k))*ps0
      nu_scale_top(k)=8.0_c_double*(1.0_c_double+tanh(log(ptop/pressure)))
      if (nu_scale_top(k)<0.15_c_double) nu_scale_top(k)=0.0_c_double
    end do
  end subroutine pycam_sima_hypervis_reference_v2

  subroutine pycam_sima_limiter_optim_v2(nelem,nlev,nq,ptens,sphweights,minp,maxp,dpmass) &
       bind(C,name="pycam_sima_limiter_optim_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,nq
    real(c_double), intent(inout) :: ptens(build_ngp,nlev,nq,nelem)
    real(c_double), intent(in) :: sphweights(build_ngp,nelem),dpmass(build_ngp,nlev,nelem)
    real(c_double), intent(inout) :: minp(nlev,nq,nelem),maxp(nlev,nq,nelem)
    integer :: ie,q
    if (nlev/=build_nlev) return
    do ie=1,nelem
      do q=1,nq
        ! Preserve CAM's limiter call boundary as well as its operation order.
        ! GNU can otherwise optimize the inlined divide/multiply path
        ! differently and change Qdp in the least-significant bits.
        call pycam_limiter_optim_iter_full( &
             ptens(:,:,q,ie),sphweights(:,ie),minp(:,q,ie), &
             maxp(:,q,ie),dpmass(:,:,ie),1,build_nlev)
      end do
    end do
  end subroutine pycam_sima_limiter_optim_v2

  subroutine pycam_sima_wind_tendency_v2(nelem,nlev,np,ra,ps0,cpair,dvv,dinv,fcor,u,v,tv,pressure,phi,kappa,vort,vtens1,vtens2) &
       bind(C,name="pycam_sima_wind_tendency_v2")
    integer(c_int), value, intent(in) :: nelem,nlev,np
    real(c_double), value, intent(in) :: ra,ps0,cpair
    real(c_double), intent(in) :: dvv(build_np,build_np)
    real(c_double), intent(in) :: dinv(2,2,build_np,build_np,nelem),fcor(build_np,build_np,nelem)
    real(c_double), intent(in) :: u(build_np,build_np,nlev,nelem),v(build_np,build_np,nlev,nelem)
    real(c_double), intent(in) :: tv(build_np,build_np,nlev,nelem),pressure(build_np,build_np,nlev,nelem)
    real(c_double), intent(in) :: phi(build_np,build_np,nlev,nelem),kappa(build_np,build_np,nlev,nelem)
    real(c_double), intent(in) :: vort(build_np,build_np,nlev,nelem)
    real(c_double), intent(out) :: vtens1(build_np,build_np,nlev,nelem),vtens2(build_np,build_np,nlev,nelem)
    real(c_double) :: e,ephi(build_np,build_np),exner(build_np,build_np),theta_v(build_np,build_np)
    real(c_double) :: grad_energy(build_np,build_np,2),grad_exner(build_np,build_np,2)
    real(c_double) :: v1,v2,glnps1,glnps2
    integer :: ie,k,i,j
    if (np/=build_np) return
    do ie=1,nelem
      do k=1,nlev
        do j=1,build_np
          do i=1,build_np
            v1=u(i,j,k,ie)
            v2=v(i,j,k,ie)
            e=0.5_c_double*(v1*v1+v2*v2)
            ephi(i,j)=e+phi(i,j,k,ie)
          end do
        end do
        call gradient_sphere(ephi,dvv,dinv(:,:,:,:,ie),ra,grad_energy)
        exner(:,:)=(pressure(:,:,k,ie)/ps0)**kappa(:,:,k,ie)
        theta_v(:,:)=tv(:,:,k,ie)/exner(:,:)
        call gradient_sphere(exner,dvv,dinv(:,:,:,:,ie),ra,grad_exner)
        grad_exner(:,:,1)=cpair*theta_v(:,:)*grad_exner(:,:,1)
        grad_exner(:,:,2)=cpair*theta_v(:,:)*grad_exner(:,:,2)
        do j=1,build_np
          do i=1,build_np
            glnps1=grad_exner(i,j,1)
            glnps2=grad_exner(i,j,2)
            v1=u(i,j,k,ie)
            v2=v(i,j,k,ie)
            vtens1(i,j,k,ie)=v2*(fcor(i,j,ie)+vort(i,j,k,ie))-grad_energy(i,j,1)-glnps1
            vtens2(i,j,k,ie)=-v1*(fcor(i,j,ie)+vort(i,j,k,ie))-grad_energy(i,j,2)-glnps2
          end do
        end do
      end do
    end do
  end subroutine pycam_sima_wind_tendency_v2

end module pycam_sima_se_startup_kernel
