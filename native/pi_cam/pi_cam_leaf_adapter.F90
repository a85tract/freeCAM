module pycam_pi_cam_leaf_adapter
  use, intrinsic :: iso_c_binding, only: c_char, c_int, c_int32_t, &
       c_int64_t, c_null_char, c_ptr
  use cam_comp, only: cam_phys_run1_leaf_action, &
       cam_phys_run2_leaf_action, cam_run4_leaf_action
  use pycam_pi_cam_adapter, only: cam_in, cam_out, configured_stop_n
  use time_manager, only: get_nstep
  implicit none
  private

  public :: pycam_pi_cam_leaf_action_v1

  interface
     subroutine pycam_pi_cam_set_fp_environment_v1() bind(C, &
          name='pycam_pi_cam_set_fp_environment_v1')
     end subroutine pycam_pi_cam_set_fp_environment_v1
  end interface

contains

  integer(c_int) function pycam_pi_cam_leaf_action_v1(action_id, nfields, &
       pointers, ndims, shapes, max_rank, fortran_comm, errmsg, errmsg_len) &
       bind(C, name='pycam_pi_cam_leaf_action_v1') result(status)
    integer(c_int), value, intent(in) :: action_id, nfields, max_rank
    integer(c_int), value, intent(in) :: fortran_comm, errmsg_len
    type(c_ptr), intent(in) :: pointers(*)
    integer(c_int32_t), intent(in) :: ndims(*)
    integer(c_int64_t), intent(in) :: shapes(*)
    character(kind=c_char), intent(out) :: errmsg(*)
    integer :: index, local_status, native_step
    logical :: write_restart, end_run

    call pycam_pi_cam_set_fp_environment_v1()
    do index = 1, max(1, int(errmsg_len))
       errmsg(index) = c_null_char
    enddo
    status = 0_c_int
    if (.not. associated(cam_in) .or. .not. associated(cam_out)) then
       status = 1_c_int
       return
    endif
    if (nfields /= 0) then
       status = 2_c_int
       return
    endif
    select case (action_id)
    case (450:458)
       call cam_phys_run1_leaf_action(action_id - 438, cam_in, cam_out, &
            local_status)
    case (460:468)
       call cam_phys_run2_leaf_action(action_id - 449, cam_out, cam_in, &
            local_status)
    case (480:481)
       ! The two halves of the macrophysics stage: stop before the driver,
       ! resume after it.  They belong to cam_run1 but its 450:458 block is
       ! full, so they get a block of their own rather than a renumbering.
       call cam_phys_run1_leaf_action(action_id - 459, cam_in, cam_out, &
            local_status)
    case (470:472)
       native_step = get_nstep()
       write_restart = native_step >= configured_stop_n
       end_run = write_restart
       call cam_run4_leaf_action(action_id - 469, write_restart, end_run, &
            local_status)
    case default
       status = 2_c_int
       return
    end select
    status = int(local_status, c_int)
  end function pycam_pi_cam_leaf_action_v1

end module pycam_pi_cam_leaf_adapter
