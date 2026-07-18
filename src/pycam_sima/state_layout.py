from __future__ import annotations

import numpy as np

from .state_pool import FieldSpec, StatePool


Q_WET = (
    "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water",
    "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
    "rain_mixing_ratio_wrt_moist_air_and_condensed_water",
)
Q_DRY = (
    "water_vapor_mixing_ratio_wrt_dry_air",
    "cloud_liquid_water_mixing_ratio_wrt_dry_air",
    "rain_mixing_ratio_wrt_dry_air",
)


def allocate_fkessler_kernel_state(
    pool: StatePool, *, ncol: int, pver: int, dt_seconds: float
) -> None:
    """Allocate a complete Python-owned FKESSLER column state.

    The deterministic profile is a kernel smoke/inspection state.  A BFB run
    must replace its values with a CAM-SIMA reference capture or the eventual
    SE analytic-initial-condition adapter.
    """
    if len(pool):
        raise ValueError("FKESSLER state allocation requires an empty pool")
    if ncol < 1 or pver < 2:
        raise ValueError("ncol must be positive and pver must be at least two")

    def alloc(name: str, shape: tuple[int, ...], fill: float = 0.0) -> np.ndarray:
        return pool.allocate(FieldSpec(name, np.float64, ()), shape, fill=fill)

    def scalar(name: str, value: float, dtype: np.dtype = np.float64) -> np.ndarray:
        return pool.allocate(FieldSpec(name, dtype, ()), (1,), fill=value)

    scalar("horizontal_loop_extent", ncol, np.int32)
    scalar("vertical_layer_dimension", pver, np.int32)
    scalar("vertical_interface_dimension", pver + 1, np.int32)
    scalar("number_of_ccpp_constituents", 3, np.int32)
    scalar("timestep_for_physics", dt_seconds)
    scalar("vertical_index_at_surface_adjacent_layer", pver, np.int32)
    scalar("vertical_index_at_top_adjacent_layer", 1, np.int32)
    scalar("vertical_index_at_surface_interface", pver + 1, np.int32)
    scalar("vertical_index_at_top_interface", 1, np.int32)
    scalar("surface_reference_pressure", 100000.0)
    scalar("standard_gravitational_acceleration", 9.80616)
    scalar("latent_heat_of_fusion_of_water_at_0c", 3.337e5)
    scalar("latent_heat_of_vaporization_of_water_at_0c", 2.501e6)
    scalar("total_energy_formula_for_physics", 0, np.int32)
    scalar("total_energy_formula_for_dycore", 1, np.int32)
    scalar("flag_for_dycore_energy_consistency_adjustment", 1, np.int32)
    scalar("do_lagrangian_vertical_coordinate", 1, np.int32)
    scalar("number_of_atmosphere_columns_with_significant_energy_or_water_imbalances", 0, np.int32)
    scalar("is_first_timestep", 1, np.int32)

    pint_1d = np.geomspace(100.0, 100000.0, pver + 1, dtype=np.float64)
    pint = alloc("air_pressure_at_interface", (ncol, pver + 1))
    pint[:] = pint_1d[None, :]
    pintdry = alloc("air_pressure_of_dry_air_at_interface", (ncol, pver + 1))
    pintdry[:] = 0.99 * pint
    pdel = alloc("air_pressure_thickness", (ncol, pver))
    pdel[:] = np.diff(pint_1d)[None, :]
    pdeldry = alloc("air_pressure_thickness_of_dry_air", (ncol, pver))
    pdeldry[:] = np.diff(0.99 * pint_1d)[None, :]
    pmid = alloc("air_pressure", (ncol, pver))
    pmid[:] = 0.5 * (pint[:, :-1] + pint[:, 1:])
    pmiddry = alloc("air_pressure_of_dry_air", (ncol, pver))
    pmiddry[:] = 0.5 * (pintdry[:, :-1] + pintdry[:, 1:])

    alloc("ln_air_pressure_at_interface", (ncol, pver + 1))[:] = np.log(pint)
    alloc("ln_air_pressure_of_dry_air_at_interface", (ncol, pver + 1))[:] = np.log(pintdry)
    alloc("ln_air_pressure", (ncol, pver))[:] = np.log(pmid)
    alloc("ln_air_pressure_of_dry_air", (ncol, pver))[:] = np.log(pmiddry)
    alloc("reciprocal_of_air_pressure_thickness", (ncol, pver))[:] = 1.0 / pdel
    alloc("reciprocal_of_air_pressure_thickness_of_dry_air", (ncol, pver))[:] = 1.0 / pdeldry

    sigma = pmid / 100000.0
    temp = alloc("air_temperature", (ncol, pver))
    temp[:] = 205.0 + 85.0 * sigma**0.18
    alloc("air_temperature_on_previous_timestep", (ncol, pver))[:] = temp
    alloc("air_temperature_at_start_of_physics_timestep", (ncol, pver))[:] = temp
    alloc("eastward_wind", (ncol, pver), 10.0)
    alloc("northward_wind", (ncol, pver), 0.0)
    alloc("surface_air_pressure", (ncol,))[:] = pint[:, -1]
    alloc("surface_pressure_of_dry_air", (ncol,))[:] = pintdry[:, -1]
    alloc("surface_geopotential", (ncol,), 0.0)

    cpair = alloc("composition_dependent_specific_heat_of_dry_air_at_constant_pressure", (ncol, pver), 1004.64)
    rair = alloc("composition_dependent_gas_constant_of_dry_air", (ncol, pver), 287.0)
    alloc("specific_heat_of_air_used_in_dycore", (ncol, pver))[:] = cpair
    alloc("ratio_of_water_vapor_gas_constant_to_composition_dependent_dry_air_gas_constant_minus_one", (ncol, pver), 0.608)

    constituents = alloc("ccpp_constituents", (ncol, pver, 3))
    constituents[:, :, 0] = 0.012 * sigma**1.5
    constituents[:, :, 1] = 2.0e-4 * np.exp(-((sigma - 0.65) / 0.18) ** 2)
    constituents[:, :, 2] = 0.0
    for index, name in enumerate(Q_WET):
        pool.register(FieldSpec(name, np.float64, ()), constituents[:, :, index])
    for name in Q_DRY:
        alloc(name, (ncol, pver))
    qmin = alloc("ccpp_constituent_minimum_values", (3,))
    qmin[:] = (1.0e-12, 0.0, 0.0)

    alloc("dimensionless_exner_function", (ncol, pver), 1.0)
    alloc("reciprocal_of_dimensionless_exner_function_wrt_surface_air_pressure", (ncol, pver), 1.0)
    alloc("air_potential_temperature", (ncol, pver))
    alloc("dry_air_density", (ncol, pver))
    alloc("relative_humidity", (ncol, pver))
    alloc("total_precipitation_rate_at_surface", (ncol,))

    zi = alloc("geopotential_height_wrt_surface_at_interface", (ncol, pver + 1))
    zm = alloc("geopotential_height_wrt_surface", (ncol, pver))
    zi[:] = np.maximum(0.0, -7000.0 * np.log(pint / pint[:, -1:]))
    zm[:] = 0.5 * (zi[:, :-1] + zi[:, 1:])
    alloc("geopotential_height_wrt_surface_at_start_of_physics_timestep", (ncol, pver))[:] = zm
    alloc("dry_static_energy", (ncol, pver))[:] = cpair * temp + 9.80616 * zm
    alloc("lagrangian_tendency_of_air_pressure", (ncol, pver))

    alloc("tendency_of_air_temperature_due_to_model_physics", (ncol, pver))
    alloc("tendency_of_air_temperature", (ncol, pver))
    alloc("tendency_of_eastward_wind_due_to_model_physics", (ncol, pver))
    alloc("tendency_of_northward_wind_due_to_model_physics", (ncol, pver))
    alloc("static_energy", (ncol, pver))
    alloc("ratio_of_specific_heat_of_air_used_in_physics_energy_formula_to_specific_heat_of_air_used_in_dycore_energy_formula", (ncol, pver), 1.0)

    for name in (
        "vertically_integrated_total_energy_using_physics_energy_formula_at_start_of_physics_timestep",
        "vertically_integrated_total_energy_using_dycore_energy_formula_at_start_of_physics_timestep",
        "vertically_integrated_total_water_at_start_of_physics_timestep",
        "vertically_integrated_total_energy_using_physics_energy_formula",
        "vertically_integrated_total_energy_using_dycore_energy_formula",
        "vertically_integrated_total_water",
        "cumulative_total_energy_boundary_flux_using_physics_energy_formula",
        "cumulative_total_water_boundary_flux",
        "total_energy_for_global_fixer",
        "net_water_vapor_fluxes_through_top_and_bottom_of_atmosphere_column",
        "net_liquid_and_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column",
        "net_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column",
        "net_sensible_heat_flux_through_top_and_bottom_of_atmosphere_column",
    ):
        alloc(name, (ncol,))

    pool.validate()
