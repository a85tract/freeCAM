#!/usr/bin/env python3
"""Generate the portable CCPP NetCDF-reader callback provider.

The CCPP schemes keep their original ``abstract_netcdf_reader_t`` API.  This
provider implements it with C callbacks registered by the Python host, so a
device can read NetCDF input without linking PIO, MPI, NetCDF, or HDF5.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/phys_utils/"
    "ccpp_io_reader.F90"
)
OUTPUT = ROOT / "native/devices/support/ccpp_io_reader.F90"


CALLBACK_MODULE_HEAD = r"""

module pycam_netcdf_callback_reader
  use iso_c_binding, only: c_char,c_double,c_f_procpointer,c_funptr,c_int, &
       c_loc,c_ptr,c_size_t
  use ccpp_io_reader, only: abstract_netcdf_reader_t
  use ccpp_kinds, only: kind_phys
  implicit none
  private
  public :: callback_reader_t
  public :: pycam_register_netcdf_reader_callbacks

  abstract interface
    integer(c_int) function open_callback_t(path,path_len,handle,errmsg,errmsg_len) bind(C)
      import c_char,c_int
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), value :: path_len,errmsg_len
      integer(c_int), intent(out) :: handle
      character(kind=c_char), intent(out) :: errmsg(*)
    end function
    integer(c_int) function close_callback_t(handle,errmsg,errmsg_len) bind(C)
      import c_char,c_int
      integer(c_int), value :: handle,errmsg_len
      character(kind=c_char), intent(out) :: errmsg(*)
    end function
    integer(c_int) function shape_callback_t(handle,name,name_len,rank,subset,start,count,dims,char_len,errmsg,errmsg_len) bind(C)
      import c_char,c_int
      integer(c_int), value :: handle,name_len,rank,subset,errmsg_len
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), intent(in) :: start(*),count(*)
      integer(c_int), intent(out) :: dims(*),char_len
      character(kind=c_char), intent(out) :: errmsg(*)
    end function
    integer(c_int) function read_callback_t(handle,name,name_len,rank,subset,start,count,data,elements,errmsg,errmsg_len) bind(C)
      import c_char,c_int,c_ptr,c_size_t
      integer(c_int), value :: handle,name_len,rank,subset,errmsg_len
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), intent(in) :: start(*),count(*)
      type(c_ptr), value :: data
      integer(c_size_t), value :: elements
      character(kind=c_char), intent(out) :: errmsg(*)
    end function
  end interface

  procedure(open_callback_t), pointer :: open_callback => null()
  procedure(close_callback_t), pointer :: close_callback => null()
  procedure(shape_callback_t), pointer :: shape_callback => null()
  procedure(read_callback_t), pointer :: read_int_callback => null()
  procedure(read_callback_t), pointer :: read_real_callback => null()
  procedure(read_callback_t), pointer :: read_char_callback => null()

  type, extends(abstract_netcdf_reader_t) :: callback_reader_t
    integer(c_int) :: handle = 0_c_int
  contains
    procedure :: open_file => callback_open_file
    procedure :: close_file => callback_close_file
"""


CALLBACK_MODULE_MIDDLE = r"""
  end type callback_reader_t

contains

  subroutine pycam_register_netcdf_reader_callbacks(open_fn,close_fn,shape_fn,read_int_fn,read_real_fn,read_char_fn) &
       bind(C,name="pycam_register_netcdf_reader_callbacks")
    type(c_funptr), value :: open_fn,close_fn,shape_fn
    type(c_funptr), value :: read_int_fn,read_real_fn,read_char_fn
    call c_f_procpointer(open_fn,open_callback)
    call c_f_procpointer(close_fn,close_callback)
    call c_f_procpointer(shape_fn,shape_callback)
    call c_f_procpointer(read_int_fn,read_int_callback)
    call c_f_procpointer(read_real_fn,read_real_callback)
    call c_f_procpointer(read_char_fn,read_char_callback)
  end subroutine pycam_register_netcdf_reader_callbacks

  subroutine callback_open_file(this,file_path,errmsg,errcode)
    class(callback_reader_t), intent(inout) :: this
    character(len=*), intent(in) :: file_path
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errcode
    errmsg = ""
    if (.not.associated(open_callback)) then
      errcode = 90
      errmsg = "Python NetCDF open callback is not registered"
      return
    end if
    errcode = open_callback(file_path,len_trim(file_path),this%handle,errmsg,len(errmsg))
  end subroutine callback_open_file

  subroutine callback_close_file(this,errmsg,errcode)
    class(callback_reader_t), intent(inout) :: this
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errcode
    errmsg = ""
    if (this%handle == 0_c_int) then
      errcode = 0
    else if (.not.associated(close_callback)) then
      errcode = 91
      errmsg = "Python NetCDF close callback is not registered"
    else
      errcode = close_callback(this%handle,errmsg,len(errmsg))
      if (errcode == 0) this%handle = 0_c_int
    end if
  end subroutine callback_close_file

  subroutine query_shape(this,varname,rank,dims,char_len,errmsg,errcode,start,count)
    class(callback_reader_t), intent(in) :: this
    character(len=*), intent(in) :: varname
    integer, intent(in) :: rank
    integer(c_int), intent(out) :: dims(6),char_len
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errcode
    integer, optional, intent(in) :: start(:),count(:)
    integer(c_int) :: c_start(6),c_count(6),subset
    dims = 1_c_int
    char_len = 1_c_int
    c_start = 0_c_int
    c_count = 0_c_int
    subset = 0_c_int
    errmsg = ""
    if (present(start) .neqv. present(count)) then
      errcode = 92
      errmsg = "NetCDF start and count must either both be present or absent"
      return
    end if
    if (present(start)) then
      if (size(start) /= rank .or. size(count) /= rank) then
        errcode = 93
        errmsg = "NetCDF start/count rank does not match requested variable"
        return
      end if
      subset = 1_c_int
      c_start(:rank) = int(start,c_int)
      c_count(:rank) = int(count,c_int)
    end if
    if (.not.associated(shape_callback)) then
      errcode = 94
      errmsg = "Python NetCDF shape callback is not registered"
      return
    end if
    errcode = shape_callback(this%handle,varname,len_trim(varname),rank,subset, &
         c_start,c_count,dims,char_len,errmsg,len(errmsg))
  end subroutine query_shape
"""


CALLBACK_MODULE_TAIL = r"""
end module pycam_netcdf_callback_reader

submodule(ccpp_io_reader) pycam_ccpp_io_reader_factory
  use pycam_netcdf_callback_reader, only: callback_reader_t
contains
  module procedure create_netcdf_reader_t
    allocate(callback_reader_t :: r)
  end procedure create_netcdf_reader_t
end submodule pycam_ccpp_io_reader_factory
"""


def _deferred_bindings() -> str:
    lines = []
    for category in ("int", "real", "char"):
        for rank in range(6):
            lines.append(
                f"    procedure :: get_var_{category}_{rank}d => "
                f"callback_get_var_{category}_{rank}d"
            )
    return "\n".join(lines) + "\n"


def _shape(rank: int, *, offset: int = 0) -> str:
    if rank == 0:
        return ""
    return "(" + ",".join(f"dims({index + offset})" for index in range(1, rank + 1)) + ")"


def _dummy_shape(rank: int) -> str:
    return "" if rank == 0 else "(" + ",".join(":" for _ in range(rank)) + ")"


def _numeric_wrapper(category: str, rank: int) -> str:
    declaration = "integer" if category == "int" else "real(kind_phys)"
    callback = "read_int_callback" if category == "int" else "read_real_callback"
    scalar_or_shape = _shape(rank)
    dummy = _dummy_shape(rank)
    elements = "1" if rank == 0 else "size(var)"
    return f"""
  subroutine callback_get_var_{category}_{rank}d(this,varname,var,errmsg,errcode,start,count)
    class(callback_reader_t), intent(in) :: this
    character(len=*), intent(in) :: varname
    {declaration}, allocatable, target, intent(out) :: var{dummy}
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errcode
    integer, optional, intent(in) :: start(:),count(:)
    integer(c_int) :: dims(6),char_len,c_start(6),c_count(6),subset
    call query_shape(this,varname,{rank},dims,char_len,errmsg,errcode,start,count)
    if (errcode /= 0) return
    allocate(var{scalar_or_shape},stat=errcode,errmsg=errmsg)
    if (errcode /= 0) return
    c_start=0_c_int; c_count=0_c_int; subset=0_c_int
    if (present(start)) then
      subset=1_c_int; c_start(:{max(rank, 1)})=int(start,c_int); c_count(:{max(rank, 1)})=int(count,c_int)
    end if
    if (.not.associated({callback})) then
      errcode=95; errmsg="Python NetCDF data callback is not registered"; return
    end if
    errcode={callback}(this%handle,varname,len_trim(varname),{rank},subset, &
         c_start,c_count,c_loc(var),int({elements},c_size_t),errmsg,len(errmsg))
  end subroutine callback_get_var_{category}_{rank}d
"""


def _character_wrapper(rank: int) -> str:
    dummy = _dummy_shape(rank)
    allocation = (
        "allocate(character(len=char_len) :: var"
        + _shape(rank, offset=1)
        + ",stat=errcode,errmsg=errmsg)"
    )
    file_rank = rank + 1
    elements = "len(var)" if rank == 0 else "len(var)*size(var)"
    return f"""
  subroutine callback_get_var_char_{rank}d(this,varname,var,errmsg,errcode,start,count)
    class(callback_reader_t), intent(in) :: this
    character(len=*), intent(in) :: varname
    character(len=:), allocatable, target, intent(out) :: var{dummy}
    character(len=*), intent(out) :: errmsg
    integer, intent(out) :: errcode
    integer, optional, intent(in) :: start(:),count(:)
    integer(c_int) :: dims(6),char_len,c_start(6),c_count(6),subset
    call query_shape(this,varname,{file_rank},dims,char_len,errmsg,errcode,start,count)
    if (errcode /= 0) return
    {allocation}
    if (errcode /= 0) return
    c_start=0_c_int; c_count=0_c_int; subset=0_c_int
    if (present(start)) then
      subset=1_c_int; c_start(:{file_rank})=int(start,c_int); c_count(:{file_rank})=int(count,c_int)
    end if
    if (.not.associated(read_char_callback)) then
      errcode=95; errmsg="Python NetCDF character callback is not registered"; return
    end if
    errcode=read_char_callback(this%handle,varname,len_trim(varname),{file_rank},subset, &
         c_start,c_count,c_loc(var),int({elements},c_size_t),errmsg,len(errmsg))
  end subroutine callback_get_var_char_{rank}d
"""


def generate() -> str:
    original = SOURCE.read_text().replace(
        "allocatable,", "allocatable, target,"
    )
    wrappers = []
    for category in ("int", "real"):
        wrappers.extend(_numeric_wrapper(category, rank) for rank in range(6))
    wrappers.extend(_character_wrapper(rank) for rank in range(6))
    return (
        "! Generated by tools/generate_ccpp_io_reader_provider.py; do not edit.\n"
        + original.rstrip()
        + "\n"
        + CALLBACK_MODULE_HEAD
        + _deferred_bindings()
        + CALLBACK_MODULE_MIDDLE
        + "\n".join(wrappers)
        + CALLBACK_MODULE_TAIL
    )


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(generate())
    print(OUTPUT)
