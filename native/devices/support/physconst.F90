module physconst
  use iso_c_binding, only: c_double
  implicit none
  private
  real(c_double), public :: epsilo=0,latvap=0,latice=0,rh2o=0,cpair=0
  real(c_double), public :: tmelt=0,h2otrip=0,rearth=0,r_universal=0
  real(c_double), public :: avogad=0,boltz=0,pi=0,gravit=0
  real(c_double), public :: rair=0,rga=0
  public :: pycam_physconst_set_epsilo,pycam_physconst_set_latvap
  public :: pycam_physconst_set_latice,pycam_physconst_set_rh2o
  public :: pycam_physconst_set_cpair,pycam_physconst_set_tmelt
  public :: pycam_physconst_set_h2otrip,pycam_physconst_set_rearth
  public :: pycam_physconst_set_r_universal,pycam_physconst_set_avogad
  public :: pycam_physconst_set_boltz,pycam_physconst_set_pi
  public :: pycam_physconst_set_gravit
  public :: pycam_physconst_set_rair,pycam_physconst_set_rga
contains
  subroutine pycam_physconst_set_epsilo(value); real(c_double),intent(in)::value; epsilo=value; end
  subroutine pycam_physconst_set_latvap(value); real(c_double),intent(in)::value; latvap=value; end
  subroutine pycam_physconst_set_latice(value); real(c_double),intent(in)::value; latice=value; end
  subroutine pycam_physconst_set_rh2o(value); real(c_double),intent(in)::value; rh2o=value; end
  subroutine pycam_physconst_set_cpair(value); real(c_double),intent(in)::value; cpair=value; end
  subroutine pycam_physconst_set_tmelt(value); real(c_double),intent(in)::value; tmelt=value; end
  subroutine pycam_physconst_set_h2otrip(value); real(c_double),intent(in)::value; h2otrip=value; end
  subroutine pycam_physconst_set_rearth(value); real(c_double),intent(in)::value; rearth=value; end
  subroutine pycam_physconst_set_r_universal(value); real(c_double),intent(in)::value; r_universal=value; end
  subroutine pycam_physconst_set_avogad(value); real(c_double),intent(in)::value; avogad=value; end
  subroutine pycam_physconst_set_boltz(value); real(c_double),intent(in)::value; boltz=value; end
  subroutine pycam_physconst_set_pi(value); real(c_double),intent(in)::value; pi=value; end
  subroutine pycam_physconst_set_gravit(value); real(c_double),intent(in)::value; gravit=value; end
  subroutine pycam_physconst_set_rair(value); real(c_double),intent(in)::value; rair=value; end
  subroutine pycam_physconst_set_rga(value); real(c_double),intent(in)::value; rga=value; end
end module physconst
