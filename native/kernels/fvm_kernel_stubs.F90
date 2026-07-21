! Minimal dependencies for selected CAM FVM numerical sources.
module pycam_fvm_kinds
  use iso_c_binding, only: c_double
  implicit none
  integer, parameter :: r8 = c_double
end module pycam_fvm_kinds

module control_mod
  implicit none
  integer, parameter :: west=1, east=2, south=3, north=4
  integer, parameter :: swest=5, seast=6, nwest=7, neast=8
end module control_mod

module cam_logfile
  implicit none
  integer, parameter :: iulog=6
end module cam_logfile

module perf_mod
  implicit none
contains
  subroutine t_startf(name)
    character(len=*), intent(in) :: name
  end subroutine
  subroutine t_stopf(name)
    character(len=*), intent(in) :: name
  end subroutine
end module perf_mod

module dimensions_mod
  use iso_c_binding, only: c_int
  use pycam_sima_build_config, only: build_nc,build_nlev,build_ntrac, &
       build_np,build_ngpc,build_irecons,build_nhe,build_nhr,build_nht, &
       build_ns,build_nhc,build_kmin_jet,build_kmax_jet, &
       build_large_courant
  implicit none
  type, bind(C) :: fvm_dimensions_c
    integer(c_int) :: nc,nlev,ntrac,np,ngpc,irecons
    integer(c_int) :: nhe,nhr,nht,ns,nhc
    integer(c_int) :: kmin_jet,kmax_jet,large_courant
    integer(c_int) :: level_begin,level_end
  end type fvm_dimensions_c
  integer, parameter :: nc=build_nc,nlev=build_nlev,ntrac=build_ntrac,np=build_np
  integer, parameter :: ngpc=build_ngpc,irecons_tracer=build_irecons
  integer, parameter :: nhe=build_nhe,nhr=build_nhr,nht=build_nht
  integer, parameter :: ns=build_ns,nhc=build_nhc
  integer, parameter :: kmin_jet=build_kmin_jet,kmax_jet=build_kmax_jet
  logical, parameter :: large_Courant_incr=build_large_courant
  integer, dimension(nlev), parameter :: irecons_tracer_lev=build_irecons
contains
  subroutine configure_fvm_dimensions(nc_in,nlev_in,ntrac_in,np_in,ngpc_in, &
       irecons_in,nhe_in,nhr_in,nht_in,ns_in,nhc_in,kmin_in,kmax_in, &
       large_courant_in,irecons_levels,ierr)
    integer, intent(in) :: nc_in,nlev_in,ntrac_in,np_in,ngpc_in,irecons_in
    integer, intent(in) :: nhe_in,nhr_in,nht_in,ns_in,nhc_in,kmin_in,kmax_in
    logical, intent(in) :: large_courant_in
    integer, intent(in) :: irecons_levels(nlev_in)
    integer, intent(out) :: ierr
    ierr=0
    if (nc_in/=nc .or. nlev_in/=nlev .or. ntrac_in/=ntrac .or. &
         np_in/=np .or. ngpc_in/=ngpc .or. irecons_in/=irecons_tracer .or. &
         nhe_in/=nhe .or. nhr_in/=nhr .or. nht_in/=nht .or. ns_in/=ns .or. &
         nhc_in/=nhc) ierr=1
    if (kmin_in/=kmin_jet .or. kmax_in/=kmax_jet .or. &
         large_courant_in .neqv. large_Courant_incr) ierr=2
    if (any(irecons_levels/=irecons_tracer_lev)) ierr=3
  end subroutine configure_fvm_dimensions

  subroutine release_fvm_dimensions()
  end subroutine release_fvm_dimensions
end module dimensions_mod

module cube_mod
  use pycam_fvm_kinds, only: r8
  implicit none
  real(r8), parameter :: cube_xstart=-0.7853981633974483096156608458198757_r8
  real(r8), parameter :: cube_xend= 0.7853981633974483096156608458198757_r8
  real(r8), parameter :: cube_ystart=cube_xstart, cube_yend=cube_xend
end module cube_mod

module coordinate_systems_mod
  use pycam_fvm_kinds, only: r8
  implicit none
  type :: cartesian2D_t
    real(r8) :: x=0, y=0
  end type
  type :: cartesian3D_t
    real(r8) :: x=0, y=0, z=0
  end type
  type :: spherical_polar_t
    real(r8) :: lon=0, lat=0
  end type
  interface cart2spherical
    module procedure cart2spherical_cart, cart2spherical_xy
  end interface
contains
  function cart2spherical_cart(value) result(output)
    type(cartesian3D_t), intent(in) :: value
    type(spherical_polar_t) :: output
    output%lon=atan2(value%y,value%x)
    output%lat=atan2(value%z,sqrt(value%x*value%x+value%y*value%y))
  end function
  function cart2spherical_xy(x,y,face) result(output)
    real(r8), intent(in) :: x,y
    integer, intent(in) :: face
    type(spherical_polar_t) :: output
    output%lon=x
    output%lat=y
  end function
  function cubedsphere2cart(value, face) result(output)
    type(cartesian2D_t), intent(in) :: value
    integer, intent(in) :: face
    type(cartesian3D_t) :: output
    output%x=value%x; output%y=value%y; output%z=real(face,r8)
  end function
  function cart2cubedsphere(value, face) result(output)
    type(cartesian3D_t), intent(in) :: value
    integer, intent(in) :: face
    type(cartesian2D_t) :: output
    output%x=value%x; output%y=value%y
  end function
end module coordinate_systems_mod

module element_mod
  use pycam_fvm_kinds, only: r8
  use coordinate_systems_mod, only: cartesian2D_t
  implicit none
  type :: element_t
    integer :: FaceNum=1
    type(cartesian2D_t) :: corners(4)
    real(r8), allocatable :: sub_elem_mass_flux(:,:,:,:)
  end type
end module element_mod

module fvm_control_volume_mod
  use pycam_fvm_kinds, only: r8
  use pycam_sima_build_config, only: build_nhr
  implicit none
  type :: fvm_struct
    real(r8), allocatable :: c(:,:,:,:), se_flux(:,:,:,:), dp_fvm(:,:,:)
    real(r8), allocatable :: dp_ref(:), dp_ref_inverse(:), psc(:,:)
    real(r8), allocatable :: inv_area_sphere(:,:), area_sphere(:,:)
    integer :: cubeboundary=0
    real(r8), allocatable :: displ_max(:,:,:)
    integer, allocatable :: flux_vec(:,:,:,:)
    real(r8), allocatable :: vtx_cart(:,:,:,:), flux_orient(:,:,:)
    integer, allocatable :: ifct(:,:), rot_matrix(:,:,:,:)
    real(r8), allocatable :: spherecentroid(:,:,:)
    real(r8), allocatable :: recons_metrics(:,:,:), recons_metrics_integral(:,:,:)
    integer :: jx_min(build_nhr+1),jx_max(build_nhr+1)
    integer :: jy_min(build_nhr+1),jy_max(build_nhr+1)
    integer, allocatable :: ibase(:,:,:)
    real(r8), allocatable :: halo_interp_weight(:,:,:,:)
    real(r8), allocatable :: centroid_stretch(:,:,:)
    real(r8), allocatable :: vertex_recons_weights(:,:,:,:)
  end type
end module fvm_control_volume_mod

module time_mod
  implicit none
  type :: timelevel_t
    integer :: nstep=0
  end type
end module time_mod

module parallel_mod
  implicit none
  type :: parallel_t
    integer :: rank=0
  end type
end module parallel_mod

module hybrid_mod
  use parallel_mod, only: parallel_t
  use dimensions_mod, only: nlev
  implicit none
  type :: hybrid_t
    type(parallel_t) :: par
  end type
contains
  function config_thread_region(hybrid, name) result(output)
    type(hybrid_t), intent(in) :: hybrid
    character(len=*), intent(in) :: name
    type(hybrid_t) :: output
    output=hybrid
  end function
  subroutine get_loop_ranges(hybrid,kbeg,kend)
    type(hybrid_t), intent(in) :: hybrid
    integer, optional, intent(out) :: kbeg,kend
    if (present(kbeg)) kbeg=1
    if (present(kend)) kend=nlev
  end subroutine
  logical function threadOwnsVertLevel(hybrid,level)
    type(hybrid_t), intent(in) :: hybrid
    integer, intent(in) :: level
    threadOwnsVertLevel=.true.
  end function
end module hybrid_mod

module hybvcoord_mod
  implicit none
  type :: hvcoord_t
    integer :: unused=0
  end type
end module hybvcoord_mod

module thread_mod
  use hybrid_mod, only: hybrid_t
  implicit none
  integer, parameter :: vert_num_threads=1
contains
  subroutine omp_set_nested(value)
    logical, intent(in) :: value
  end subroutine
  logical function threadOwnsVertLevel(hybrid,level)
    type(hybrid_t), intent(in) :: hybrid
    integer, intent(in) :: level
    threadOwnsVertLevel=.true.
  end function
end module thread_mod

module edgetype_mod
  implicit none
  type :: edgebuffer_t
    integer :: unused=0
  end type
end module edgetype_mod

module edge_mod
  use pycam_fvm_kinds, only: r8
  use edgetype_mod, only: edgebuffer_t
  implicit none
contains
  subroutine ghostpack(buffer,field,kblk,kptr,ie)
    type(edgebuffer_t), intent(inout) :: buffer
    real(r8), intent(in) :: field(..)
    integer, intent(in) :: kblk,kptr,ie
  end subroutine
  subroutine ghostunpack(buffer,field,kblk,kptr,ie)
    type(edgebuffer_t), intent(inout) :: buffer
    real(r8), intent(inout) :: field(..)
    integer, intent(in) :: kblk,kptr,ie
  end subroutine
end module edge_mod

module bndry_mod
  use edgetype_mod, only: edgebuffer_t
  use hybrid_mod, only: hybrid_t
  implicit none
contains
  subroutine ghost_exchange(hybrid,buffer,location)
    type(hybrid_t), intent(in) :: hybrid
    type(edgebuffer_t), intent(inout) :: buffer
    character(len=*), optional, intent(in) :: location
  end subroutine
end module bndry_mod

module fvm_mod
  use element_mod, only: element_t
  use fvm_control_volume_mod, only: fvm_struct
  use hybrid_mod, only: hybrid_t
  use edgetype_mod, only: edgebuffer_t
  implicit none
  interface fill_halo_fvm
    module procedure fill_halo_fvm_stub
  end interface
contains
  subroutine fill_halo_fvm_stub(buffer,elem,fvm,hybrid,nets,nete,ndepth,kmin,kmax,ksize,active)
    type(edgebuffer_t), intent(inout) :: buffer
    type(element_t), intent(inout) :: elem(:)
    type(fvm_struct), intent(inout) :: fvm(:)
    type(hybrid_t), intent(in) :: hybrid
    integer, intent(in) :: nets,nete,ndepth,kmin,kmax,ksize
    logical, optional, intent(in) :: active
  end subroutine
end module fvm_mod
