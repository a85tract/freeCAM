module pycesm_full_driver_adapter
  use, intrinsic :: iso_c_binding, only: c_int, c_ptr
  use cesm_comp_mod, only: cesm_pre_init1, cesm_pre_init2, cesm_init, &
       cesm_run, cesm_final, cesm_action_call, cesm_nested_action_call, &
       cesm_cam_action_call, cesm_physics_action_call, cesm_step_begin, &
       cesm_step_end, cesm_run_finish, cesm_init_action_call, &
       cesm_final_action_call, cesm_external_atm_iteration, &
       cesm_exchange_buffer_query, cesm_init_atm_phase2_end
  use esmf, only: ESMF_Initialize
  implicit none

  logical, save :: initialized = .false.
  logical, save :: finalized = .false.
  logical, save :: initialization_active = .false.
  integer(c_int), save :: initialization_expected_action = 500_c_int

contains

  subroutine pycesm_full_initialize_v1(fortran_comm, status) &
       bind(C, name="pycesm_full_initialize_v1")
    integer(c_int), value, intent(in) :: fortran_comm
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (initialized .or. finalized) then
      status = 1_c_int
      return
    end if

    ! mpi4py has already initialized MPI.  The patched isolated CESM source
    ! accepts that communicator and therefore does not call MPI_Init twice.
    call cesm_pre_init1(fortran_comm)
    call ESMF_Initialize()
    call cesm_pre_init2()
    call cesm_init()
    initialized = .true.
  end subroutine pycesm_full_initialize_v1

  subroutine pycesm_full_initialize_action_v1(action_id, fortran_comm, status) &
       bind(C, name="pycesm_full_initialize_action_v1")
    integer(c_int), value, intent(in) :: action_id
    integer(c_int), value, intent(in) :: fortran_comm
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (initialized .or. finalized) then
      status = 1_c_int
      return
    end if
    if (action_id == 501_c_int) then
      if (initialization_active) then
        status = 1_c_int
        return
      end if
      initialization_active = .true.
      initialization_expected_action = 500_c_int
    end if
    if (.not. initialization_active .or. &
        action_id /= initialization_expected_action + 1_c_int) then
      status = 2_c_int
      return
    end if

    select case (action_id)
    case (501_c_int)
      call cesm_pre_init1(fortran_comm)
    case (502_c_int)
      call ESMF_Initialize()
    case (503_c_int)
      call cesm_pre_init2()
    case (504_c_int:532_c_int)
      call cesm_init_action_call(action_id, status)
      if (status /= 0_c_int) return
      if (action_id == 532_c_int) then
        initialized = .true.
        initialization_active = .false.
      end if
    case default
      status = 3_c_int
      return
    end select
    initialization_expected_action = action_id
  end subroutine pycesm_full_initialize_action_v1

  subroutine pycesm_full_step_v1(status) &
       bind(C, name="pycesm_full_step_v1")
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 1_c_int
      return
    end if
    call cesm_run(max_steps=1)
  end subroutine pycesm_full_step_v1

  subroutine pycesm_full_advance_v1(step_count, status) &
       bind(C, name="pycesm_full_advance_v1")
    integer(c_int), value, intent(in) :: step_count
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (.not. initialized .or. finalized .or. step_count < 0_c_int) then
      status = 1_c_int
      return
    end if
    if (step_count > 0_c_int) call cesm_run(max_steps=step_count)
  end subroutine pycesm_full_advance_v1

  subroutine pycesm_full_action_v1(action_id, status) &
       bind(C, name="pycesm_full_action_v1")
    integer(c_int), value, intent(in) :: action_id
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_action_call(action_id, status)
  end subroutine pycesm_full_action_v1

  subroutine pycesm_full_nested_action_v1(action_id, loop_complete, status) &
       bind(C, name="pycesm_full_nested_action_v1")
    integer(c_int), value, intent(in) :: action_id
    integer(c_int), intent(out) :: loop_complete
    integer(c_int), intent(out) :: status

    loop_complete = 0_c_int
    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_nested_action_call(action_id, loop_complete, status)
  end subroutine pycesm_full_nested_action_v1

  subroutine pycesm_full_external_atm_iteration_v1(loop_complete, status) &
       bind(C, name="pycesm_full_external_atm_iteration_v1")
    integer(c_int), intent(out) :: loop_complete
    integer(c_int), intent(out) :: status

    loop_complete = 0_c_int
    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_external_atm_iteration(loop_complete, status)
  end subroutine pycesm_full_external_atm_iteration_v1

  subroutine pycesm_full_exchange_buffer_v1(exchange_id, address, nattr, &
       npoint, status) bind(C, name="pycesm_full_exchange_buffer_v1")
    integer(c_int), value, intent(in) :: exchange_id
    type(c_ptr), intent(out) :: address
    integer(c_int), intent(out) :: nattr
    integer(c_int), intent(out) :: npoint
    integer(c_int), intent(out) :: status

    status = 0_c_int
    ! The ATM exchange vectors already exist immediately after initialization
    ! action 512.  Python must capture that exact pre-coupling state before
    ! later initialization actions prime a2x, so permit read-only pointer
    ! discovery while the explicit initialization sequence is active.
    if ((.not. initialized .and. .not. initialization_active) .or. &
         finalized) then
      status = 3_c_int
      return
    end if
    call cesm_exchange_buffer_query(exchange_id, address, nattr, npoint, &
         status)
  end subroutine pycesm_full_exchange_buffer_v1

  subroutine pycesm_full_initialize_atm_phase2_end_v1(status) &
       bind(C, name="pycesm_full_initialize_atm_phase2_end_v1")
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (.not. initialization_active .or. initialized .or. finalized .or. &
        initialization_expected_action /= 529_c_int) then
      status = 3_c_int
      return
    end if
    call cesm_init_atm_phase2_end(status)
  end subroutine pycesm_full_initialize_atm_phase2_end_v1

  subroutine pycesm_full_cam_action_v1(action_id, loop_complete, status) &
       bind(C, name="pycesm_full_cam_action_v1")
    integer(c_int), value, intent(in) :: action_id
    integer(c_int), intent(out) :: loop_complete
    integer(c_int), intent(out) :: status

    loop_complete = 0_c_int
    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_cam_action_call(action_id, loop_complete, status)
  end subroutine pycesm_full_cam_action_v1

  subroutine pycesm_full_physics_action_v1(action_id, loop_complete, status) &
       bind(C, name="pycesm_full_physics_action_v1")
    integer(c_int), value, intent(in) :: action_id
    integer(c_int), intent(out) :: loop_complete
    integer(c_int), intent(out) :: status

    loop_complete = 0_c_int
    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_physics_action_call(action_id, loop_complete, status)
  end subroutine pycesm_full_physics_action_v1

  subroutine pycesm_full_step_begin_v1(stepno, ymd, tod, alarm_mask, status) &
       bind(C, name="pycesm_full_step_begin_v1")
    integer(c_int), intent(out) :: stepno
    integer(c_int), intent(out) :: ymd
    integer(c_int), intent(out) :: tod
    integer(c_int), intent(out) :: alarm_mask
    integer(c_int), intent(out) :: status

    stepno = 0_c_int
    ymd = 0_c_int
    tod = 0_c_int
    alarm_mask = 0_c_int
    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_step_begin(stepno, ymd, tod, alarm_mask, status)
  end subroutine pycesm_full_step_begin_v1

  subroutine pycesm_full_step_begin_python_v1(expected_step, expected_ymd, &
       expected_tod, python_alarm_mask, native_alarm_mask, status) &
       bind(C, name="pycesm_full_step_begin_python_v1")
    integer(c_int), value, intent(in) :: expected_step
    integer(c_int), value, intent(in) :: expected_ymd
    integer(c_int), value, intent(in) :: expected_tod
    integer(c_int), value, intent(in) :: python_alarm_mask
    integer(c_int), intent(out) :: native_alarm_mask
    integer(c_int), intent(out) :: status
    integer(c_int) :: native_step
    integer(c_int) :: native_ymd
    integer(c_int) :: native_tod

    native_alarm_mask = 0_c_int
    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_step_begin(native_step, native_ymd, native_tod, &
         native_alarm_mask, status, expected_step, expected_ymd, &
         expected_tod, python_alarm_mask)
  end subroutine pycesm_full_step_begin_python_v1

  subroutine pycesm_full_step_end_v1(status) &
       bind(C, name="pycesm_full_step_end_v1")
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 3_c_int
      return
    end if
    call cesm_step_end(status)
  end subroutine pycesm_full_step_end_v1

  subroutine pycesm_full_finalize_v1(status) &
       bind(C, name="pycesm_full_finalize_v1")
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 1_c_int
      return
    end if
    call cesm_run_finish(status)
    if (status /= 0_c_int) return
    call cesm_final()
    ! The bundled esmf_wrf_timemgr stub implements ESMF_Finalize solely as
    ! MPI_Finalize and ignores ESMF_END_KEEPMPI.  MPI is owned by mpi4py in
    ! this embedding, so Python must remain able to perform its final
    ! collective and let mpi4py finalize MPI at process exit.
    finalized = .true.
  end subroutine pycesm_full_finalize_v1

  subroutine pycesm_full_finalize_action_v1(action_id, status) &
       bind(C, name="pycesm_full_finalize_action_v1")
    integer(c_int), value, intent(in) :: action_id
    integer(c_int), intent(out) :: status

    status = 0_c_int
    if (.not. initialized .or. finalized) then
      status = 1_c_int
      return
    end if
    if (action_id == 601_c_int) then
      call cesm_run_finish(status)
      if (status /= 0_c_int) return
    end if
    call cesm_final_action_call(action_id, status)
    if (status == 0_c_int .and. action_id == 610_c_int) then
      finalized = .true.
    end if
  end subroutine pycesm_full_finalize_action_v1

end module pycesm_full_driver_adapter
