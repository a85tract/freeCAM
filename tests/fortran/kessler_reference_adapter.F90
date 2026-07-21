module kessler_reference_adapter
  use iso_c_binding, only: c_double, c_int
  use kessler, only: kessler_init, kessler_run
  implicit none
contains
  subroutine kessler_reference_v1(ncol, nz, dt, lv, pref, rhoqr, cpair, &
       rair, rho, z, pk, theta, qv, qc, qr, precl, relhum, errflg) bind(C)
    integer(c_int), value :: ncol, nz
    real(c_double), value :: dt, lv, pref, rhoqr
    real(c_double), intent(in) :: cpair(ncol, nz), rair(ncol, nz)
    real(c_double), intent(in) :: rho(ncol, nz), z(ncol, nz), pk(ncol, nz)
    real(c_double), intent(inout) :: theta(ncol, nz), qv(ncol, nz)
    real(c_double), intent(inout) :: qc(ncol, nz), qr(ncol, nz)
    real(c_double), intent(out) :: precl(ncol), relhum(ncol, nz)
    integer(c_int), intent(out) :: errflg
    character(len=512) :: errmsg
    character(len=64) :: scheme

    call kessler_init(lv, pref, rhoqr, errmsg, errflg)
    if (errflg /= 0) return
    call kessler_run(ncol, nz, dt, nz, 1, cpair, rair, rho, z, pk, &
         theta, qv, qc, qr, precl, relhum, scheme, errmsg, errflg)
  end subroutine kessler_reference_v1
end module kessler_reference_adapter
