module cam_abortutils
  implicit none
  private
  public :: endrun
contains
  subroutine endrun(message)
    character(len=*), intent(in), optional :: message
    if (present(message)) then
      error stop trim(message)
    else
      error stop "CAM numerical device aborted"
    end if
  end subroutine endrun
end module cam_abortutils
