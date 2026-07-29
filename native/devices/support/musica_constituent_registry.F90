! C ABI ownership bridge for MUSICA's dynamically registered constituents.
module pycam_musica_constituent_registry
  use iso_c_binding, only: c_char, c_double, c_int, c_null_char, c_null_ptr, &
                           c_ptr, c_loc
  use ccpp_constituent_prop_mod, only: ccpp_constituent_properties_t, &
                                       ccpp_constituent_prop_ptr_t
  use musica_ccpp, only: musica_ccpp_register
  implicit none
  private

  type(ccpp_constituent_properties_t), allocatable, target, save :: properties(:)
  type(ccpp_constituent_prop_ptr_t), allocatable, target, save :: property_ptrs(:)

  public :: pycam_musica_register_v1, pycam_musica_metadata_v1
  public :: pycam_musica_release_v1

contains

  integer(c_int) function pycam_musica_register_v1( &
      address, count, error_buffer, capacity) result(status) &
      bind(C, name="pycam_musica_register_v1")
    type(c_ptr), intent(out) :: address
    integer(c_int), intent(out) :: count
    character(kind=c_char), intent(out) :: error_buffer(*)
    integer(c_int), value, intent(in) :: capacity
    character(len=512) :: errmsg
    type(ccpp_constituent_properties_t), pointer :: property
    integer :: errcode, index

    call release_registry()
    address = c_null_ptr
    count = 0_c_int
    errmsg = ""
    call musica_ccpp_register(properties, errmsg, errcode)
    if (errcode /= 0) then
      call copy_error_to_c(errmsg, error_buffer, capacity)
      status = int(errcode, c_int)
      return
    end if
    if (.not. allocated(properties) .or. size(properties) <= 0) then
      call copy_error_to_c( &
          "MUSICA register returned no constituent properties", &
          error_buffer, capacity)
      status = 1_c_int
      return
    end if

    allocate(property_ptrs(size(properties)))
    do index = 1, size(properties)
      property => properties(index)
      call property_ptrs(index)%set(property, errcode, errmsg)
      if (errcode /= 0) then
        call copy_error_to_c(errmsg, error_buffer, capacity)
        call release_registry()
        status = int(errcode, c_int)
        return
      end if
      call property_ptrs(index)%set_const_index(index, errcode, errmsg)
      if (errcode /= 0) then
        call copy_error_to_c(errmsg, error_buffer, capacity)
        call release_registry()
        status = int(errcode, c_int)
        return
      end if
    end do
    address = c_loc(property_ptrs(1))
    count = int(size(property_ptrs), c_int)
    if (capacity > 0_c_int) error_buffer(1) = c_null_char
    status = 0_c_int
  end function pycam_musica_register_v1

  integer(c_int) function pycam_musica_metadata_v1( &
      index, name_buffer, capacity, minimum_value, molar_mass_value, &
      error_buffer, error_capacity) result(status) &
      bind(C, name="pycam_musica_metadata_v1")
    integer(c_int), value, intent(in) :: index, capacity, error_capacity
    character(kind=c_char), intent(out) :: name_buffer(*), error_buffer(*)
    real(c_double), intent(out) :: minimum_value, molar_mass_value
    character(len=512) :: name, errmsg
    integer :: errcode

    if (.not. allocated(property_ptrs) .or. index < 1_c_int .or. &
        index > int(size(property_ptrs), c_int)) then
      call copy_error_to_c( &
          "MUSICA constituent metadata index is out of range", &
          error_buffer, error_capacity)
      status = 1_c_int
      return
    end if
    call property_ptrs(index)%standard_name(name, errcode, errmsg)
    if (errcode == 0) then
      call property_ptrs(index)%minimum(minimum_value, errcode, errmsg)
    end if
    if (errcode == 0) then
      call property_ptrs(index)%molar_mass(molar_mass_value, errcode, errmsg)
    end if
    if (errcode /= 0) then
      call copy_error_to_c(errmsg, error_buffer, error_capacity)
      status = int(errcode, c_int)
      return
    end if
    call copy_error_to_c(name, name_buffer, capacity)
    if (error_capacity > 0_c_int) error_buffer(1) = c_null_char
    status = 0_c_int
  end function pycam_musica_metadata_v1

  subroutine pycam_musica_release_v1(address) &
      bind(C, name="pycam_musica_release_v1")
    type(c_ptr), value, intent(in) :: address
    call release_registry()
  end subroutine pycam_musica_release_v1

  subroutine release_registry()
    if (allocated(property_ptrs)) deallocate(property_ptrs)
    if (allocated(properties)) deallocate(properties)
  end subroutine release_registry

  subroutine copy_error_to_c(message, buffer, capacity)
    character(len=*), intent(in) :: message
    character(kind=c_char), intent(out) :: buffer(*)
    integer(c_int), value, intent(in) :: capacity
    integer :: index, count

    if (capacity <= 0_c_int) return
    count = min(len_trim(message), int(capacity) - 1)
    do index = 1, count
      buffer(index) = achar(iachar(message(index:index)), kind=c_char)
    end do
    buffer(count + 1) = c_null_char
  end subroutine copy_error_to_c

end module pycam_musica_constituent_registry
