module ccpp_constituent_prop_mod
  implicit none
  private
  public :: ccpp_constituent_prop_ptr_t

  type :: ccpp_constituent_prop_ptr_t
    logical :: thermo_active = .true.
    logical :: water_species = .true.
    character(len=80) :: name = ''
  contains
    procedure :: is_thermo_active
    procedure :: is_water_species
    procedure :: standard_name
    procedure :: units
    procedure :: is_wet
    procedure :: long_name
  end type ccpp_constituent_prop_ptr_t

contains

  subroutine is_thermo_active(self, flag)
    class(ccpp_constituent_prop_ptr_t), intent(in) :: self
    logical, intent(out) :: flag
    flag = self%thermo_active
  end subroutine is_thermo_active

  subroutine is_water_species(self, flag)
    class(ccpp_constituent_prop_ptr_t), intent(in) :: self
    logical, intent(out) :: flag
    flag = self%water_species
  end subroutine is_water_species

  subroutine standard_name(self, name, errcode, errmsg)
    class(ccpp_constituent_prop_ptr_t), intent(in) :: self
    character(len=*), intent(out) :: name
    integer, intent(out), optional :: errcode
    character(len=*), intent(out), optional :: errmsg
    name = self%name
    if (present(errcode)) errcode = 0
    if (present(errmsg)) errmsg = ''
  end subroutine standard_name

  subroutine units(self, value, errcode, errmsg)
    class(ccpp_constituent_prop_ptr_t), intent(in) :: self
    character(len=*), intent(out) :: value
    integer, intent(out), optional :: errcode
    character(len=*), intent(out), optional :: errmsg
    value = 'kg kg-1'
    if (present(errcode)) errcode = 0
    if (present(errmsg)) errmsg = ''
  end subroutine units

  subroutine is_wet(self, value, errcode, errmsg)
    class(ccpp_constituent_prop_ptr_t), intent(in) :: self
    logical, intent(out) :: value
    integer, intent(out), optional :: errcode
    character(len=*), intent(out), optional :: errmsg
    value = .true.
    if (present(errcode)) errcode = 0
    if (present(errmsg)) errmsg = ''
  end subroutine is_wet

  subroutine long_name(self, value, errcode, errmsg)
    class(ccpp_constituent_prop_ptr_t), intent(in) :: self
    character(len=*), intent(out) :: value
    integer, intent(out), optional :: errcode
    character(len=*), intent(out), optional :: errmsg
    value = self%name
    if (present(errcode)) errcode = 0
    if (present(errmsg)) errmsg = ''
  end subroutine long_name

end module ccpp_constituent_prop_mod
