! Every chunk's compute_uwshcu_inv inputs, gathered before the shallow convection stage runs, so
! the model bound at the kernel's hook answers the rank's chunks in one forward.
!
! convect_shallow_tend computes none of the kernel's inputs: it hands on the chunk's state and
! the physics-buffer fields it fetched (convect_shallow.F90 576-693, the UW branch).  This module
! fetches the same fields the same way for every chunk -- the state the stage is given, the
! buffer's CLD and CONCLD at the old time level, pblh, cush at the old time level and tke -- and
! passes them to the hook's batch in the call's own argument order.  No arithmetic: the hook then
! checks, at every call, that what the call passes is what was gathered for its chunk, bit for
! bit, and stops the run if it is not.
module pycam_shcu_batch
  use, intrinsic :: iso_c_binding, only: c_int
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver, begchunk, endchunk
  use constituents, only: pcnst, cnst_get_ind
  use time_manager, only: get_step_size
  use phys_control, only: phys_getopts
  use physics_types, only: physics_state
  use physics_buffer, only: physics_buffer_desc, pbuf_get_chunk, pbuf_get_field, pbuf_get_index, &
                            pbuf_old_tim_idx
  use pycam_stage_hosts, only: host_state, host_pbuf2d
  use pycam_hooks, only: pycam_hooks_batch_begin_compute_uwshcu_inv, pycam_hooks_batch_put_compute_uwshcu_inv, &
                         pycam_hooks_batch_forward_compute_uwshcu_inv
  implicit none
  private
  public :: pycam_shcu_batch_v1

  integer, save :: cld_idx = -1, concld_idx = -1, tke_idx = -1, pblh_idx = -1, cush_idx = -1
  integer, save :: ixcldliq = -1, ixcldice = -1

contains

  integer(c_int) function pycam_shcu_batch_v1() bind(C, name='pycam_shcu_batch_v1') result(status)
    ! gather every chunk's inputs and answer them at once; 0, or 2 when the stage hosts are not
    ! bound, 3 when the shallow scheme is not UW, 4 when a buffer field is not registered
    type(physics_state), pointer :: state
    type(physics_buffer_desc), pointer :: pbuf(:)
    real(r8), pointer :: cld(:,:), concld(:,:), tke(:,:), pblh(:), cush(:)
    character(len=16) :: shallow_scheme
    real(r8) :: ztodt
    integer :: lchnk, itim_old, rows
    status = 0_c_int
    if (.not. associated(host_state)) then
      status = 2_c_int; return
    end if
    call phys_getopts(shallow_scheme_out=shallow_scheme)
    if (trim(shallow_scheme) /= 'UW') then
      status = 3_c_int; return
    end if
    if (cush_idx < 0) then
      cld_idx = pbuf_get_index('CLD')
      concld_idx = pbuf_get_index('CONCLD')
      tke_idx = pbuf_get_index('tke')
      pblh_idx = pbuf_get_index('pblh')
      cush_idx = pbuf_get_index('cush')
      call cnst_get_ind('CLDLIQ', ixcldliq)
      call cnst_get_ind('CLDICE', ixcldice)
    end if
    if (min(cld_idx, concld_idx, tke_idx, pblh_idx, cush_idx) < 1) then
      status = 4_c_int; return
    end if
    ztodt = real(get_step_size(), r8)             ! as tphysbc's glue passes it
    rows = 0
    do lchnk = begchunk, endchunk
      rows = rows + host_state(lchnk)%ncol
    end do
    call pycam_hooks_batch_begin_compute_uwshcu_inv(endchunk - begchunk + 1, rows)
    do lchnk = begchunk, endchunk
      state => host_state(lchnk)
      pbuf => pbuf_get_chunk(host_pbuf2d, lchnk)
      itim_old = pbuf_old_tim_idx()
      call pbuf_get_field(pbuf, cld_idx, cld, start=(/1,1,itim_old/), kount=(/pcols,pver,1/))
      call pbuf_get_field(pbuf, concld_idx, concld, start=(/1,1,itim_old/), kount=(/pcols,pver,1/))
      call pbuf_get_field(pbuf, pblh_idx, pblh)
      call pbuf_get_field(pbuf, cush_idx, cush, (/1,itim_old/), (/pcols,1/))
      call pbuf_get_field(pbuf, tke_idx, tke)
      ! convect_shallow.F90 683-693: the actual arguments, less the outputs
      call pycam_hooks_batch_put_compute_uwshcu_inv(pcols, pver, state%ncol, pcnst, ztodt, &
           state%pint, state%zi, state%pmid, state%zm, state%pdel, &
           state%u, state%v, state%q(:,:,1), state%q(:,:,ixcldliq), state%q(:,:,ixcldice), &
           state%t, state%s, state%q(:,:,:), &
           tke, cld, concld, pblh, cush, &
           state%lchnk, state%pdeldry)
    end do
    call pycam_hooks_batch_forward_compute_uwshcu_inv()
  end function pycam_shcu_batch_v1

end module pycam_shcu_batch
