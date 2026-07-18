module mpi
  use iso_c_binding, only: c_double
  implicit none
  integer, parameter :: MPI_MIN=1, MPI_SUM=2, MPI_INTEGER=3, MPI_REAL8=4
  interface MPI_REDUCE
    module procedure mpi_reduce_int
    module procedure mpi_reduce_real
  end interface MPI_REDUCE
contains
  subroutine mpi_reduce_int(sendbuf, recvbuf, count, datatype, op, root, comm, ierr)
    integer, intent(in) :: sendbuf(*), count, datatype, op, root, comm
    integer, intent(out) :: recvbuf(*), ierr
    recvbuf(1:count)=sendbuf(1:count)
    ierr=0
  end subroutine mpi_reduce_int
  subroutine mpi_reduce_real(sendbuf, recvbuf, count, datatype, op, root, comm, ierr)
    real(c_double), intent(in) :: sendbuf(*)
    real(c_double), intent(out) :: recvbuf(*)
    integer, intent(in) :: count, datatype, op, root, comm
    integer, intent(out) :: ierr
    recvbuf(1:count)=sendbuf(1:count)
    ierr=0
  end subroutine mpi_reduce_real
end module mpi

subroutine MPI_REDUCE(sendbuf, recvbuf, count, datatype, op, root, comm, ierr)
  ! qneg's source imports MPI constants with an ONLY list but calls MPI_REDUCE
  ! as an external. Statistics are disabled in pycam-sima; this local symbol
  ! exists only to make that dormant branch self-contained.
  integer, intent(in) :: sendbuf(*), count, datatype, op, root, comm
  integer, intent(out) :: recvbuf(*), ierr
  recvbuf(1:count)=sendbuf(1:count)
  ierr=0
end subroutine MPI_REDUCE
