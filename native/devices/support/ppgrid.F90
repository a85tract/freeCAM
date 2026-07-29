! Portable run-time dimensions for legacy CAM cloud utility routines.
!
! The original ppgrid module reaches through physics_grid and vert_coord,
! which in turn pull MPI/ESMF/PIO into an otherwise numerical device.  The
! values below are set from explicit C ABI dimension arguments immediately
! before every device entrypoint.  Nothing is fixed at compile time.
module ppgrid
  use iso_c_binding, only: c_int
  implicit none
  private

  integer, public, save :: pcols = 0
  integer, public, save :: pver = 0
  integer, public, save :: pverp = 0
  public :: pycam_ppgrid_set_dimensions

contains

  subroutine pycam_ppgrid_set_dimensions(columns, layers, interfaces)
    integer(c_int), intent(in) :: columns
    integer(c_int), intent(in) :: layers
    integer(c_int), intent(in) :: interfaces

    pcols = int(columns)
    pver = int(layers)
    pverp = int(interfaces)
  end subroutine pycam_ppgrid_set_dimensions

end module ppgrid
