from __future__ import annotations

from collections import Counter
from pathlib import Path

from .state_pool import StatePool


class RecordingBackend:
    """Deterministic backend used to verify Python control and observers."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.counts: Counter[str] = Counter()

    def lifecycle(self, name: str, pool: StatePool) -> None:
        token = f"lifecycle:{name}"
        self.calls.append(token)
        self.counts[token] += 1

    def call(self, name: str, pool: StatePool) -> None:
        self.calls.append(name)
        self.counts[name] += 1


class NativeKesslerBackend:
    """CFFI loader for the generated explicit per-scheme ABI."""

    def __init__(self, library: str | Path) -> None:
        from cffi import FFI

        self.library = Path(library).resolve()
        if not self.library.is_file():
            raise FileNotFoundError(f"native Kessler library not found: {self.library}")
        self.ffi = FFI()
        self.ffi.cdef(
            """
            int pycam_kessler_abi_version(void);
            int pycam_kessler_lifecycle(const char *phase, char *errmsg, int errmsg_len);
            int pycam_kessler_has_scheme(const char *name);
            int pycam_kessler_timestep_initial(int,int,int,int,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,int,int,void*,void*);
            int pycam_kessler_timestep_final(int,int,void*,void*,void*,void*,void*);
            int pycam_calc_exner_run(int,int,void*,void*,double,void*,void*);
            int pycam_temp_to_potential_temp_run(int,int,void*,void*,void*);
            int pycam_calc_dry_air_ideal_gas_density_run(int,int,void*,void*,void*,void*);
            int pycam_wet_to_dry_water_vapor_run(int,int,void*,void*,void*,void*);
            int pycam_wet_to_dry_cloud_liquid_water_run(int,int,void*,void*,void*,void*);
            int pycam_wet_to_dry_rain_run(int,int,void*,void*,void*,void*);
            int pycam_kessler_run(int,int,double,int,int,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*,void*);
            int pycam_potential_temp_to_temp_run(int,int,void*,void*,void*);
            int pycam_dry_to_wet_water_vapor_run(int,int,void*,void*,void*,void*);
            int pycam_dry_to_wet_cloud_liquid_water_run(int,int,void*,void*,void*,void*);
            int pycam_dry_to_wet_rain_run(int,int,void*,void*,void*,void*);
            int pycam_kessler_update_run(int,int,double,void*,void*,void*,void*);
            int pycam_qneg_run(int,int,int,void*,void*);
            int pycam_geopotential_temp_run(int,int,int,int,int,int,int,int,void*,void*,void*,void*,void*,void*,void*,void*,void*,double,void*,void*,void*);
            int pycam_check_energy_zero_fluxes_run(int,void*,void*,void*,void*);
            int pycam_check_energy_scaling_run(int,int,void*,void*,void*);
            int pycam_check_energy_chng_run(int ncol,int nz,int ncnst,
                void *q,void *pdel,void *u,void *v,void *temp,void *pintdry,void *phis,void *zm,
                void *cp_phys,void *cp_dy,void *scaling,void *te_phys,void *te_dyn,void *tw,
                void *tend_te,void *tend_tw,void *temp_ini,void *z_ini,void *count,
                double dt,double latice,double latvap,int energy_phys,int energy_dyn,
                void *flx_vap,void *flx_cnd,void *flx_ice,void *flx_sen);
            int pycam_sima_state_diagnostics_run(int ncol,int nz,int ncnst,
                void *ps,void *psdry,void *phis,void *temp,void *u,void *v,void *dse,void *omega,
                void *pmid,void *pmiddry,void *pdel,void *pdeldry,void *rpdel,void *rpdeldry,
                void *lnpmid,void *lnpmiddry,void *inv_exner,void *zm,void *pint,void *pintdry,
                void *lnpint,void *lnpintdry,void *zi,void *const_array);
            int pycam_kessler_diagnostics_run(int,void*);
            int pycam_thermo_water_update_run(int,int,int,void*,void*,void*,void*,void*);
            int pycam_dycore_energy_consistency_adjust_run(int,int,int,void*,void*,void*);
            int pycam_apply_tendency_of_air_temperature_run(int,int,void*,void*,void*,double);
            int pycam_sima_tend_diagnostics_run(int,int,void*,void*,void*);
            """
        )
        self.lib = self.ffi.dlopen(str(self.library))
        version = int(self.lib.pycam_kessler_abi_version())
        if version != 1:
            raise RuntimeError(f"unsupported Kessler ABI version {version}")

    def lifecycle(self, name: str, pool: StatePool) -> None:
        if name == "timestep_initial":
            self._timestep_initial(pool)
            return
        if name == "timestep_final":
            self._timestep_final(pool)
            return
        errmsg = self.ffi.new("char[]", 1024)
        ierr = self.lib.pycam_kessler_lifecycle(name.encode(), errmsg, 1024)
        if ierr:
            raise RuntimeError(self.ffi.string(errmsg).decode(errors="replace"))

    def call(self, name: str, pool: StatePool) -> None:
        if not self.lib.pycam_kessler_has_scheme(name.encode()):
            raise RuntimeError(f"native library does not export scheme {name}")
        ncol, nz, ncnst = self._dims(pool)
        p = lambda field: self._ptr(pool, field)
        dispatch = {
            "calc_exner": lambda: self.lib.pycam_calc_exner_run(
                ncol, nz, p("composition_dependent_specific_heat_of_dry_air_at_constant_pressure"),
                p("composition_dependent_gas_constant_of_dry_air"), self._float(pool, "surface_reference_pressure"),
                p("air_pressure"), p("dimensionless_exner_function")
            ),
            "temp_to_potential_temp": lambda: self.lib.pycam_temp_to_potential_temp_run(
                ncol, nz, p("air_temperature"), p("dimensionless_exner_function"), p("air_potential_temperature")
            ),
            "calc_dry_air_ideal_gas_density": lambda: self.lib.pycam_calc_dry_air_ideal_gas_density_run(
                ncol, nz, p("composition_dependent_gas_constant_of_dry_air"), p("air_pressure_of_dry_air"),
                p("air_temperature"), p("dry_air_density")
            ),
            "wet_to_dry_water_vapor": lambda: self._convert(
                "pycam_wet_to_dry_water_vapor_run", pool,
                "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water",
                "water_vapor_mixing_ratio_wrt_dry_air",
            ),
            "wet_to_dry_cloud_liquid_water": lambda: self._convert(
                "pycam_wet_to_dry_cloud_liquid_water_run", pool,
                "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
                "cloud_liquid_water_mixing_ratio_wrt_dry_air",
            ),
            "wet_to_dry_rain": lambda: self._convert(
                "pycam_wet_to_dry_rain_run", pool,
                "rain_mixing_ratio_wrt_moist_air_and_condensed_water", "rain_mixing_ratio_wrt_dry_air",
            ),
            "kessler": lambda: self.lib.pycam_kessler_run(
                ncol, nz, self._float(pool, "timestep_for_physics"),
                self._int(pool, "vertical_index_at_surface_adjacent_layer"),
                self._int(pool, "vertical_index_at_top_adjacent_layer"),
                p("composition_dependent_specific_heat_of_dry_air_at_constant_pressure"),
                p("composition_dependent_gas_constant_of_dry_air"), p("dry_air_density"),
                p("geopotential_height_wrt_surface"), p("dimensionless_exner_function"),
                p("air_potential_temperature"), p("water_vapor_mixing_ratio_wrt_dry_air"),
                p("cloud_liquid_water_mixing_ratio_wrt_dry_air"), p("rain_mixing_ratio_wrt_dry_air"),
                p("total_precipitation_rate_at_surface"), p("relative_humidity")
            ),
            "potential_temp_to_temp": lambda: self.lib.pycam_potential_temp_to_temp_run(
                ncol, nz, p("air_potential_temperature"), p("dimensionless_exner_function"), p("air_temperature")
            ),
            "dry_to_wet_water_vapor": lambda: self._convert(
                "pycam_dry_to_wet_water_vapor_run", pool,
                "water_vapor_mixing_ratio_wrt_dry_air",
                "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water",
            ),
            "dry_to_wet_cloud_liquid_water": lambda: self._convert(
                "pycam_dry_to_wet_cloud_liquid_water_run", pool,
                "cloud_liquid_water_mixing_ratio_wrt_dry_air",
                "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
            ),
            "dry_to_wet_rain": lambda: self._convert(
                "pycam_dry_to_wet_rain_run", pool, "rain_mixing_ratio_wrt_dry_air",
                "rain_mixing_ratio_wrt_moist_air_and_condensed_water",
            ),
            "kessler_update": lambda: self.lib.pycam_kessler_update_run(
                ncol, nz, self._float(pool, "timestep_for_physics"), p("air_potential_temperature"),
                p("dimensionless_exner_function"), p("air_temperature_on_previous_timestep"),
                p("tendency_of_air_temperature_due_to_model_physics")
            ),
            "qneg": lambda: self.lib.pycam_qneg_run(
                ncol, nz, ncnst, p("ccpp_constituent_minimum_values"), p("ccpp_constituents")
            ),
            "geopotential_temp": lambda: self.lib.pycam_geopotential_temp_run(
                ncol, nz, ncnst, self._int(pool, "do_lagrangian_vertical_coordinate"),
                self._int(pool, "vertical_index_at_surface_adjacent_layer"),
                self._int(pool, "vertical_index_at_top_adjacent_layer"),
                self._int(pool, "vertical_index_at_surface_interface"),
                self._int(pool, "vertical_index_at_top_interface"),
                p("ln_air_pressure_at_interface"), p("air_pressure_at_interface"), p("air_pressure"),
                p("air_pressure_thickness"), p("reciprocal_of_air_pressure_thickness"), p("air_temperature"),
                p("water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water"), p("ccpp_constituents"),
                p("composition_dependent_gas_constant_of_dry_air"), self._float(pool, "standard_gravitational_acceleration"),
                p("ratio_of_water_vapor_gas_constant_to_composition_dependent_dry_air_gas_constant_minus_one"),
                p("geopotential_height_wrt_surface_at_interface"), p("geopotential_height_wrt_surface")
            ),
            "check_energy_zero_fluxes": lambda: self.lib.pycam_check_energy_zero_fluxes_run(
                ncol, p("net_water_vapor_fluxes_through_top_and_bottom_of_atmosphere_column"),
                p("net_liquid_and_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column"),
                p("net_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column"),
                p("net_sensible_heat_flux_through_top_and_bottom_of_atmosphere_column")
            ),
            "check_energy_scaling": lambda: self.lib.pycam_check_energy_scaling_run(
                ncol, nz, p("specific_heat_of_air_used_in_dycore"),
                p("composition_dependent_specific_heat_of_dry_air_at_constant_pressure"),
                p("ratio_of_specific_heat_of_air_used_in_physics_energy_formula_to_specific_heat_of_air_used_in_dycore_energy_formula")
            ),
            "check_energy_chng": lambda: self._check_energy_chng(pool),
            "sima_state_diagnostics": lambda: self._state_diagnostics(pool),
            "kessler_diagnostics": lambda: self.lib.pycam_kessler_diagnostics_run(
                ncol, p("total_precipitation_rate_at_surface")
            ),
            "thermo_water_update": lambda: self.lib.pycam_thermo_water_update_run(
                ncol, nz, ncnst, p("ccpp_constituents"), p("air_pressure_thickness"),
                p("air_pressure_thickness_of_dry_air"),
                p("composition_dependent_specific_heat_of_dry_air_at_constant_pressure"),
                p("specific_heat_of_air_used_in_dycore")
            ),
            "dycore_energy_consistency_adjust": lambda: self.lib.pycam_dycore_energy_consistency_adjust_run(
                ncol, nz, self._int(pool, "flag_for_dycore_energy_consistency_adjustment"),
                p("ratio_of_specific_heat_of_air_used_in_physics_energy_formula_to_specific_heat_of_air_used_in_dycore_energy_formula"),
                p("tendency_of_air_temperature_due_to_model_physics"), p("tendency_of_air_temperature")
            ),
            "apply_tendency_of_air_temperature": lambda: self.lib.pycam_apply_tendency_of_air_temperature_run(
                ncol, nz, p("tendency_of_air_temperature"), p("air_temperature"),
                p("tendency_of_air_temperature_due_to_model_physics"), self._float(pool, "timestep_for_physics")
            ),
            "sima_tend_diagnostics": lambda: self.lib.pycam_sima_tend_diagnostics_run(
                ncol, nz, p("tendency_of_air_temperature_due_to_model_physics"),
                p("tendency_of_eastward_wind_due_to_model_physics"),
                p("tendency_of_northward_wind_due_to_model_physics")
            ),
        }
        try:
            ierr = int(dispatch[name]())
        except KeyError as exc:
            raise RuntimeError(f"Python binding is missing scheme {name}") from exc
        if ierr:
            raise RuntimeError(f"native scheme {name} failed with error code {ierr}")

    def _dims(self, pool: StatePool) -> tuple[int, int, int]:
        return (
            self._int(pool, "horizontal_loop_extent"),
            self._int(pool, "vertical_layer_dimension"),
            self._int(pool, "number_of_ccpp_constituents"),
        )

    def _ptr(self, pool: StatePool, name: str):
        return self.ffi.cast("void *", pool.pointer(name))

    @staticmethod
    def _int(pool: StatePool, name: str) -> int:
        return int(pool.require(name).reshape(-1)[0])

    @staticmethod
    def _float(pool: StatePool, name: str) -> float:
        return float(pool.require(name).reshape(-1)[0])

    def _convert(self, symbol: str, pool: StatePool, source: str, target: str) -> int:
        ncol, nz, _ = self._dims(pool)
        function = getattr(self.lib, symbol)
        return int(function(ncol, nz, self._ptr(pool, "air_pressure_thickness"),
                            self._ptr(pool, "air_pressure_thickness_of_dry_air"),
                            self._ptr(pool, source), self._ptr(pool, target)))

    def _timestep_initial(self, pool: StatePool) -> None:
        ncol, nz, ncnst = self._dims(pool)
        p = lambda field: self._ptr(pool, field)
        ierr = self.lib.pycam_kessler_timestep_initial(
            ncol, nz, ncnst, self._int(pool, "is_first_timestep"), p("ccpp_constituents"),
            p("air_pressure_thickness"), p("eastward_wind"), p("northward_wind"), p("air_temperature"),
            p("air_pressure_of_dry_air_at_interface"), p("surface_geopotential"),
            p("geopotential_height_wrt_surface"),
            p("composition_dependent_specific_heat_of_dry_air_at_constant_pressure"),
            p("specific_heat_of_air_used_in_dycore"),
            p("vertically_integrated_total_energy_using_physics_energy_formula_at_start_of_physics_timestep"),
            p("vertically_integrated_total_energy_using_dycore_energy_formula_at_start_of_physics_timestep"),
            p("vertically_integrated_total_water_at_start_of_physics_timestep"),
            p("vertically_integrated_total_energy_using_physics_energy_formula"),
            p("vertically_integrated_total_energy_using_dycore_energy_formula"),
            p("vertically_integrated_total_water"),
            p("cumulative_total_energy_boundary_flux_using_physics_energy_formula"),
            p("cumulative_total_water_boundary_flux"), p("air_temperature_at_start_of_physics_timestep"),
            p("geopotential_height_wrt_surface_at_start_of_physics_timestep"),
            p("number_of_atmosphere_columns_with_significant_energy_or_water_imbalances"),
            p("total_energy_for_global_fixer"), self._int(pool, "total_energy_formula_for_physics"),
            self._int(pool, "total_energy_formula_for_dycore"), p("air_temperature_on_previous_timestep"),
            p("tendency_of_air_temperature_due_to_model_physics"),
        )
        if ierr:
            raise RuntimeError(f"native timestep_initial failed with error code {ierr}")
        pool["is_first_timestep"][0] = 0

    def _timestep_final(self, pool: StatePool) -> None:
        ncol, nz, _ = self._dims(pool)
        ierr = self.lib.pycam_kessler_timestep_final(
            ncol, nz, self._ptr(pool, "composition_dependent_specific_heat_of_dry_air_at_constant_pressure"),
            self._ptr(pool, "air_temperature"), self._ptr(pool, "geopotential_height_wrt_surface"),
            self._ptr(pool, "surface_geopotential"), self._ptr(pool, "dry_static_energy"),
        )
        if ierr:
            raise RuntimeError(f"native timestep_final failed with error code {ierr}")

    def _check_energy_chng(self, pool: StatePool) -> int:
        ncol, nz, ncnst = self._dims(pool)
        p = lambda field: self._ptr(pool, field)
        return int(self.lib.pycam_check_energy_chng_run(
            ncol, nz, ncnst, p("ccpp_constituents"), p("air_pressure_thickness"),
            p("eastward_wind"), p("northward_wind"), p("air_temperature"),
            p("air_pressure_of_dry_air_at_interface"), p("surface_geopotential"),
            p("geopotential_height_wrt_surface"),
            p("composition_dependent_specific_heat_of_dry_air_at_constant_pressure"),
            p("specific_heat_of_air_used_in_dycore"),
            p("ratio_of_specific_heat_of_air_used_in_physics_energy_formula_to_specific_heat_of_air_used_in_dycore_energy_formula"),
            p("vertically_integrated_total_energy_using_physics_energy_formula"),
            p("vertically_integrated_total_energy_using_dycore_energy_formula"),
            p("vertically_integrated_total_water"),
            p("cumulative_total_energy_boundary_flux_using_physics_energy_formula"),
            p("cumulative_total_water_boundary_flux"), p("air_temperature_at_start_of_physics_timestep"),
            p("geopotential_height_wrt_surface_at_start_of_physics_timestep"),
            p("number_of_atmosphere_columns_with_significant_energy_or_water_imbalances"),
            self._float(pool, "timestep_for_physics"), self._float(pool, "latent_heat_of_fusion_of_water_at_0c"),
            self._float(pool, "latent_heat_of_vaporization_of_water_at_0c"),
            self._int(pool, "total_energy_formula_for_physics"), self._int(pool, "total_energy_formula_for_dycore"),
            p("net_water_vapor_fluxes_through_top_and_bottom_of_atmosphere_column"),
            p("net_liquid_and_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column"),
            p("net_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column"),
            p("net_sensible_heat_flux_through_top_and_bottom_of_atmosphere_column"),
        ))

    def _state_diagnostics(self, pool: StatePool) -> int:
        ncol, nz, ncnst = self._dims(pool)
        p = lambda field: self._ptr(pool, field)
        names = (
            "surface_air_pressure", "surface_pressure_of_dry_air", "surface_geopotential", "air_temperature",
            "eastward_wind", "northward_wind", "dry_static_energy", "lagrangian_tendency_of_air_pressure",
            "air_pressure", "air_pressure_of_dry_air", "air_pressure_thickness", "air_pressure_thickness_of_dry_air",
            "reciprocal_of_air_pressure_thickness", "reciprocal_of_air_pressure_thickness_of_dry_air",
            "ln_air_pressure", "ln_air_pressure_of_dry_air",
            "reciprocal_of_dimensionless_exner_function_wrt_surface_air_pressure", "geopotential_height_wrt_surface",
            "air_pressure_at_interface", "air_pressure_of_dry_air_at_interface", "ln_air_pressure_at_interface",
            "ln_air_pressure_of_dry_air_at_interface", "geopotential_height_wrt_surface_at_interface", "ccpp_constituents",
        )
        return int(self.lib.pycam_sima_state_diagnostics_run(ncol, nz, ncnst, *(p(name) for name in names)))
