module runtime_temperature_offset
  use ccpp_kinds, only: kind_phys
  implicit none
contains
  !> \section arg_table_runtime_temperature_offset_run Argument Table
  !! \htmlinclude runtime_temperature_offset_run.html
  subroutine runtime_temperature_offset_run(field, increment, errmsg, errflg)
    real(kind_phys), intent(inout) :: field(:,:)
    real(kind_phys), intent(in) :: increment
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errflg

    field = field + increment
    errmsg = ''
    errflg = 0
  end subroutine runtime_temperature_offset_run
end module runtime_temperature_offset
