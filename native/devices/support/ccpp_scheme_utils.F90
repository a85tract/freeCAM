module ccpp_scheme_utils
   ! Python-owned CCPP constituent-name registry.
   !
   ! CAM's generated cap normally constructs ccpp_model_constituents_t and
   ! gives ccpp_scheme_utils a pointer to that object.  freeCAM owns the
   ! constituent axis in StatePool instead, so this portable provider stores
   ! the exact Python axis order.  Original schemes continue to call the
   ! unmodified ccpp_constituent_index interface.

   use, intrinsic :: iso_c_binding, only: c_char, c_int, c_null_char

   implicit none
   private

   public :: ccpp_constituent_index
   public :: ccpp_constituent_indices
   public :: pycam_ccpp_scheme_registry_initialize_v1

   logical :: initialized = .false.
   character(len=:), allocatable :: constituent_names(:)

contains

   integer(c_int) function pycam_ccpp_scheme_registry_initialize_v1( &
        count, name_width, names, errmsg, errmsg_len) result(status) bind(C)
      integer(c_int), value, intent(in) :: count
      integer(c_int), value, intent(in) :: name_width
      character(kind=c_char), intent(in) :: names(*)
      character(kind=c_char), intent(out) :: errmsg(*)
      integer(c_int), value, intent(in) :: errmsg_len

      integer :: char_index
      integer :: constituent_index
      integer :: offset

      call set_c_error('', errmsg, errmsg_len)
      status = 1_c_int
      initialized = .false.
      if (allocated(constituent_names)) deallocate(constituent_names)
      if (count <= 0_c_int) then
         call set_c_error('constituent count must be positive', errmsg, errmsg_len)
         return
      end if
      if (name_width <= 1_c_int) then
         call set_c_error('constituent name width must exceed one', errmsg, errmsg_len)
         return
      end if

      allocate(character(len=int(name_width) - 1) :: constituent_names(int(count)))
      constituent_names = ''
      do constituent_index = 1, int(count)
         offset = (constituent_index - 1) * int(name_width)
         do char_index = 1, int(name_width) - 1
            if (names(offset + char_index) == c_null_char) exit
            constituent_names(constituent_index)(char_index:char_index) = &
                 achar(iachar(names(offset + char_index)))
         end do
         if (len_trim(constituent_names(constituent_index)) == 0) then
            call set_c_error('constituent names cannot be empty', errmsg, errmsg_len)
            deallocate(constituent_names)
            return
         end if
      end do
      initialized = .true.
      status = 0_c_int
   end function pycam_ccpp_scheme_registry_initialize_v1

   subroutine ccpp_constituent_index(standard_name, const_index, errcode, errmsg)
      character(len=*), intent(in) :: standard_name
      integer, intent(out) :: const_index
      integer, optional, intent(out) :: errcode
      character(len=*), optional, intent(out) :: errmsg

      integer :: index

      const_index = -huge(1)
      if (present(errcode)) errcode = 0
      if (present(errmsg)) errmsg = ''
      if (.not. initialized) then
         if (present(errcode)) errcode = 1
         if (present(errmsg)) then
            errmsg = 'ccpp_constituent_index FAILED, module not initialized'
         end if
         return
      end if

      do index = 1, size(constituent_names)
         if (lowercase(trim(standard_name)) == &
              lowercase(trim(constituent_names(index)))) then
            const_index = index
            return
         end if
      end do
      ! Match ccpp_model_constituents_t: a constituent that is not declared
      ! by the active suite is a valid query and returns int_unassigned.
      ! Schemes such as prescribe_radiative_gas_concentrations use that
      ! result to skip optional gases.
   end subroutine ccpp_constituent_index

   subroutine ccpp_constituent_indices(standard_names, const_inds, errcode, errmsg)
      character(len=*), intent(in) :: standard_names(:)
      integer, intent(out) :: const_inds(:)
      integer, optional, intent(out) :: errcode
      character(len=*), optional, intent(out) :: errmsg

      character(len=512) :: local_error
      integer :: index
      integer :: local_status

      const_inds = -huge(1)
      if (present(errcode)) errcode = 0
      if (present(errmsg)) errmsg = ''
      if (size(const_inds) < size(standard_names)) then
         if (present(errcode)) errcode = 1
         if (present(errmsg)) then
            errmsg = 'ccpp_constituent_indices: output array is too small'
         end if
         return
      end if
      do index = 1, size(standard_names)
         call ccpp_constituent_index(standard_names(index), const_inds(index), &
              local_status, local_error)
         if (local_status /= 0) then
            if (present(errcode)) errcode = local_status
            if (present(errmsg)) errmsg = trim(local_error)
            return
         end if
      end do
   end subroutine ccpp_constituent_indices

   pure function lowercase(value) result(lower)
      character(len=*), intent(in) :: value
      character(len=len(value)) :: lower
      integer :: index
      integer :: code

      lower = value
      do index = 1, len(value)
         code = iachar(value(index:index))
         if (code >= iachar('A') .and. code <= iachar('Z')) then
            lower(index:index) = achar(code + iachar('a') - iachar('A'))
         end if
      end do
   end function lowercase

   subroutine set_c_error(message, errmsg, errmsg_len)
      character(len=*), intent(in) :: message
      character(kind=c_char), intent(out) :: errmsg(*)
      integer(c_int), value, intent(in) :: errmsg_len
      integer :: index
      integer :: limit

      if (errmsg_len <= 0_c_int) return
      limit = min(len_trim(message), int(errmsg_len) - 1)
      do index = 1, limit
         errmsg(index) = achar(iachar(message(index:index)), kind=c_char)
      end do
      errmsg(limit + 1) = c_null_char
   end subroutine set_c_error

end module ccpp_scheme_utils
