! Minimal error-message service; normal model logging remains Python-owned.
module shr_log_mod
  implicit none
  private
  integer, public :: shr_log_Level = 0
  integer, public :: shr_log_Unit = 6
  public :: shr_log_errMsg
contains
  pure function shr_log_errMsg(file, line) result(message)
    character(len=*), intent(in) :: file
    integer, intent(in) :: line
    character(len=512) :: message
    character(len=32) :: line_text
    write(line_text, '(i0)') line
    message = 'ERROR in ' // trim(file) // ' at line ' // trim(line_text)
  end function shr_log_errMsg
end module shr_log_mod
