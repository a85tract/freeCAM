module error_messages
  use cam_abortutils, only: endrun
  implicit none
  private
  public :: alloc_err,handle_err,handle_errmsg
contains
  subroutine alloc_err(istat,routine,name,nelem)
    integer, intent(in) :: istat,nelem
    character(len=*), intent(in) :: routine,name
    if (istat /= 0) call endrun("allocation failure: "//trim(routine)//":"//trim(name))
  end subroutine alloc_err

  subroutine handle_err(istat,msg)
    integer, intent(in) :: istat
    character(len=*), intent(in) :: msg
    if (istat /= 0) call endrun(trim(msg))
  end subroutine handle_err

  subroutine handle_errmsg(errmsg,subname,extra_msg)
    character(len=*), intent(in) :: errmsg
    character(len=*), intent(in), optional :: subname,extra_msg
    if (len_trim(errmsg) > 0) call endrun(trim(errmsg))
  end subroutine handle_errmsg
end module error_messages
