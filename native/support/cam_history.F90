module cam_history
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: history_add_field, history_out_field

  interface history_out_field
    module procedure history_out_field_1d
    module procedure history_out_field_2d
  end interface history_out_field

contains

  subroutine history_add_field(name, long_name, dimensions, sampling, units, mixing_ratio)
    character(len=*), intent(in) :: name, long_name, dimensions, sampling, units
    character(len=*), intent(in), optional :: mixing_ratio
    ! pycam-sima observers replace CAM history files. Registration remains a
    ! real suite call, but no native history buffer owns or copies the state.
  end subroutine history_add_field

  subroutine history_out_field_1d(name, field)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: field(:)
  end subroutine history_out_field_1d

  subroutine history_out_field_2d(name, field)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: field(:,:)
  end subroutine history_out_field_2d

end module cam_history
