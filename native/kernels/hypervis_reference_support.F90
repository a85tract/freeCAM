! Source-order-compatible static reference pressure used by the SE dycore.
module pycam_sima_hypervis_reference_support
  use iso_c_binding, only: c_double
  implicit none
  private
  public :: pycam_get_dp_ref

  interface pycam_get_dp_ref
    module procedure pycam_get_dp_ref_2hd
  end interface

contains

  subroutine pycam_get_dp_ref_1hd(hyai,hybi,ps0,phis,rair,tref,dp_ref,ps_ref)
    real(c_double), intent(in) :: hyai(:),hybi(:),ps0,phis(:),rair,tref
    real(c_double), intent(out) :: dp_ref(:,:),ps_ref(:)
    integer :: k

    ps_ref(:)=ps0*exp(-phis(:)/(rair*tref))
    do k=1,size(dp_ref,2)
      dp_ref(:,k)=(hyai(k+1)-hyai(k))*ps0 + &
           (hybi(k+1)-hybi(k))*ps_ref(:)
    end do
  end subroutine pycam_get_dp_ref_1hd

  subroutine pycam_get_dp_ref_2hd(hyai,hybi,ps0,phis,rair,tref,dp_ref,ps_ref)
    real(c_double), intent(in) :: hyai(:),hybi(:),ps0,phis(:,:),rair,tref
    real(c_double), intent(out) :: dp_ref(:,:,:),ps_ref(:,:)
    integer :: j

    do j=1,size(dp_ref,2)
      call pycam_get_dp_ref_1hd( &
           hyai,hybi,ps0,phis(:,j),rair,tref,dp_ref(:,j,:),ps_ref(:,j))
    end do
  end subroutine pycam_get_dp_ref_2hd

end module pycam_sima_hypervis_reference_support
