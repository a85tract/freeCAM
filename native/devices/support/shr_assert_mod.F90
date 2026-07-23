module shr_assert_mod
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  private
  public :: shr_assert_in_domain
contains
  subroutine shr_assert_in_domain(var,ge,gt,le,lt,varname,msg)
    real(r8), intent(in) :: var
    real(r8), intent(in), optional :: ge,gt,le,lt
    character(len=*), intent(in), optional :: varname,msg
    logical :: valid
    valid = .true.
    if (present(ge)) valid = valid .and. var >= ge
    if (present(gt)) valid = valid .and. var > gt
    if (present(le)) valid = valid .and. var <= le
    if (present(lt)) valid = valid .and. var < lt
    if (.not. valid) error stop "shr_assert_in_domain failed"
  end subroutine shr_assert_in_domain
end module shr_assert_mod
