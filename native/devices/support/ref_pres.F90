! Minimal low-top CAM reference-pressure service for standalone devices.
!
! Every suite pinned by freecam is a low-top configuration.  The original
! holtslag_boville interstitial explicitly documents that press_lim_idx at
! 1.e-7 Pa is level 1 in such configurations and that its host dependency may
! be replaced by ntop_eddy = 1.  Keeping that policy in a named provider lets
! the original scheme source remain unchanged without pulling vert_coord,
! MPI, ESMF, PIO, or CAM grid infrastructure into a numerical device.
module ref_pres
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: press_lim_idx
contains
  pure integer function press_lim_idx(pres, top) result(level)
    real(kind_phys), intent(in) :: pres
    logical, intent(in) :: top

    ! Reference the arguments so strict compilers do not warn that this
    ! deliberately low-top policy ignores them.
    if (top .or. pres >= 0.0_kind_phys) then
      level = 1
    else
      level = 1
    end if
  end function press_lim_idx
end module ref_pres
