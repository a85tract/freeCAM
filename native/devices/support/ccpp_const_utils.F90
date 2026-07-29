! Case-insensitive CCPP constituent lookup for generated standalone devices.
!
! CCPP stores standard names in lower case, while valid callers such as
! MUSICA use chemical symbols including "Cl", "O2", and "O3".  The original
! CAM-SIMA utility compares those strings case-sensitively.  This host-side
! compatibility provider preserves the original numerical scheme and supplies
! the lookup service expected by a standalone device.
module ccpp_const_utils
  implicit none
  private

  public :: ccpp_const_get_idx

contains

  subroutine ccpp_const_get_idx(constituent_props, name, cindex, errmsg, errflg)
    use ccpp_constituent_prop_mod, only: ccpp_constituent_prop_ptr_t

    type(ccpp_constituent_prop_ptr_t), intent(in)  :: constituent_props(:)
    character(len=*),                  intent(in)  :: name
    integer,                           intent(out) :: cindex
    character(len=512),                intent(out) :: errmsg
    integer,                           intent(out) :: errflg

    integer            :: t_cindex
    character(len=256) :: t_const_name

    errmsg = ''
    errflg = 0
    cindex = -1

    do t_cindex = lbound(constituent_props, 1), ubound(constituent_props, 1)
       call constituent_props(t_cindex)%standard_name(t_const_name, errflg, errmsg)
       if (errflg /= 0) return

       if (trim(to_lower(t_const_name)) == trim(to_lower(name))) then
          cindex = t_cindex
          exit
       end if
    end do
  end subroutine ccpp_const_get_idx

  pure function to_lower(input_string) result(lowercase_string)
    character(len=*), intent(in)     :: input_string
    character(len=len(input_string)) :: lowercase_string

    integer          :: i
    integer          :: code
    integer          :: upper_to_lower
    character(len=1) :: character_value

    upper_to_lower = iachar('a') - iachar('A')
    do i = 1, len(input_string)
       character_value = input_string(i:i)
       code = iachar(character_value)
       if (code >= iachar('A') .and. code <= iachar('Z')) then
          character_value = achar(code + upper_to_lower)
       end if
       lowercase_string(i:i) = character_value
    end do
  end function to_lower

end module ccpp_const_utils
