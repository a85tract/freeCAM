cmake_minimum_required(VERSION 3.24)
project(pycam_sima_native VERSION 0.1.0 LANGUAGES C Fortran)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/modules")

set(CAM_SIMA "${CMAKE_CURRENT_LIST_DIR}/../external/CAM-SIMA" CACHE PATH "Pinned CAM-SIMA source")
set(PHYS "${CAM_SIMA}/src/physics/ncar_ccpp/schemes")

add_library(pycam_sima_kessler SHARED
  support/ccpp_kinds.F90
  support/ccpp_constituent_prop_mod.F90
  support/mpi.F90
  support/cam_thermo_formula.F90
  support/cam_thermo.F90
  support/cam_history_support.F90
  support/cam_history.F90
  support/thermo_water_local.F90
  "${PHYS}/utilities/state_converters.F90"
  "${PHYS}/kessler/kessler.F90"
  "${PHYS}/kessler/kessler_update.F90"
  "${PHYS}/conservation_adjust/check_energy/check_energy_zero_fluxes.F90"
  "${PHYS}/conservation_adjust/check_energy/check_energy_scaling.F90"
  "${PHYS}/conservation_adjust/check_energy/check_energy_chng.F90"
  "${PHYS}/conservation_adjust/check_energy/dycore_energy_consistency_adjust.F90"
  "${PHYS}/utilities/physics_tendency_updaters.F90"
  "${PHYS}/utilities/qneg.F90"
  "${PHYS}/utilities/geopotential_temp.F90"
  "${PHYS}/sima_diagnostics/kessler_diagnostics.F90"
  "${PHYS}/sima_diagnostics/sima_state_diagnostics.F90"
  "${PHYS}/sima_diagnostics/sima_tend_diagnostics.F90"
  kessler_wrappers/kessler_abi.F90
)

set_target_properties(pycam_sima_kessler PROPERTIES
  OUTPUT_NAME pycam_sima_kessler
  Fortran_MODULE_DIRECTORY "${CMAKE_Fortran_MODULE_DIRECTORY}"
)

if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(pycam_sima_kessler PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-ffree-line-length-none>
    $<$<COMPILE_LANGUAGE:Fortran>:-fallow-argument-mismatch>
  )
endif()

enable_testing()
