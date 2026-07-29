! C ABI bridge to the pinned CESM share-library orbital implementation.
module pycam_orbital_host_adapter
  use iso_c_binding, only: c_bool, c_double, c_int
  use shr_orb_mod, only: shr_orb_cosz, shr_orb_decl, shr_orb_params, &
                         SHR_ORB_UNDEF_REAL
  implicit none
  private
  public :: pycam_orbital_advance_v1

contains

  integer(c_int) function pycam_orbital_advance_v1( &
      orbital_year, number_of_columns, calendar_day, latitudes, longitudes, &
      averaging_seconds, use_uniform_angle, uniform_angle, &
      solar_declination, earth_sun_distance, solar_zenith_angle, &
      cosine_zenith_for_radiation) result(status) &
      bind(C, name="pycam_orbital_advance_v1")
    integer(c_int), value, intent(in) :: orbital_year
    integer(c_int), value, intent(in) :: number_of_columns
    real(c_double), value, intent(in) :: calendar_day
    real(c_double), intent(in) :: latitudes(number_of_columns)
    real(c_double), intent(in) :: longitudes(number_of_columns)
    real(c_double), value, intent(in) :: averaging_seconds
    logical(c_bool), value, intent(in) :: use_uniform_angle
    real(c_double), value, intent(in) :: uniform_angle
    real(c_double), intent(out) :: solar_declination
    real(c_double), intent(out) :: earth_sun_distance
    real(c_double), intent(out) :: solar_zenith_angle(number_of_columns)
    real(c_double), intent(out) :: cosine_zenith_for_radiation(number_of_columns)

    integer :: column
    real(c_double) :: eccentricity
    real(c_double) :: obliquity_degrees
    real(c_double) :: moving_vernal_equinox_degrees
    real(c_double) :: obliquity_radians
    real(c_double) :: mean_longitude_at_vernal_equinox
    real(c_double) :: moving_vernal_equinox_radians

    eccentricity = SHR_ORB_UNDEF_REAL
    obliquity_degrees = SHR_ORB_UNDEF_REAL
    moving_vernal_equinox_degrees = SHR_ORB_UNDEF_REAL
    call shr_orb_params(orbital_year, eccentricity, obliquity_degrees, &
         moving_vernal_equinox_degrees, obliquity_radians, &
         mean_longitude_at_vernal_equinox, &
         moving_vernal_equinox_radians, .false.)
    call shr_orb_decl(calendar_day, eccentricity, &
         moving_vernal_equinox_radians, &
         mean_longitude_at_vernal_equinox, obliquity_radians, &
         solar_declination, earth_sun_distance)

    do column = 1, number_of_columns
      solar_zenith_angle(column) = acos( &
           shr_orb_cosz(calendar_day, latitudes(column), &
                        longitudes(column), solar_declination))
      if (use_uniform_angle) then
        cosine_zenith_for_radiation(column) = shr_orb_cosz( &
             calendar_day, latitudes(column), longitudes(column), &
             solar_declination, averaging_seconds, &
             uniform_angle=uniform_angle)
      else
        cosine_zenith_for_radiation(column) = shr_orb_cosz( &
             calendar_day, latitudes(column), longitudes(column), &
             solar_declination, averaging_seconds)
      end if
    end do
    status = 0_c_int
  end function pycam_orbital_advance_v1

end module pycam_orbital_host_adapter
