"""Machine-readable persistent-state contract for the model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


@dataclass(frozen=True, slots=True)
class FieldContract:
    standard_name: str
    dtype: str
    dimensions: tuple[str, ...]
    intent: str
    category: str
    units: str = "1"
    aliases: tuple[str, ...] = ()
    ccpp_standard_name: str | None = None
    owner: str = "python"
    lifetime: str = "persistent"
    history: bool = False
    restart: bool = False
    writable: bool = True

    @property
    def shape_expression(self) -> str:
        return "(" + ", ".join(self.dimensions) + ")"

    def shape(self, dimensions: Mapping[str, int]) -> tuple[int, ...]:
        return tuple(
            int(name) if str(name).isdigit() else int(dimensions[name])
            for name in self.dimensions
        )

    def machine_record(self) -> dict[str, Any]:
        result = asdict(self)
        result["dimensions"] = list(self.dimensions)
        result["aliases"] = list(self.aliases)
        result["shape_expression"] = self.shape_expression
        return result

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FieldContract":
        """Restore one contract from a checkpoint or public variable spec."""

        return cls(
            standard_name=str(values["standard_name"]),
            dtype=str(values["dtype"]),
            dimensions=tuple(str(item) for item in values.get("dimensions", ())),
            intent=str(values["intent"]),
            category=str(values["category"]),
            units=str(values.get("units", "1")),
            aliases=tuple(str(item) for item in values.get("aliases", ())),
            ccpp_standard_name=(
                None
                if values.get("ccpp_standard_name") is None
                else str(values["ccpp_standard_name"])
            ),
            owner=str(values.get("owner", "python")),
            lifetime=str(values.get("lifetime", "persistent")),
            history=bool(values.get("history", False)),
            restart=bool(values.get("restart", False)),
            writable=bool(values.get("writable", True)),
        )


@dataclass(frozen=True, slots=True)
class AliasRule:
    alias: str
    target: str
    axis: int | None = None
    index: int | None = None
    ccpp_standard_name: str | None = None


def _field(
    name: str,
    dtype: str,
    dims: Iterable[str],
    intent: str,
    category: str,
    units: str = "1",
    *,
    aliases: Iterable[str] = (),
    ccpp_standard_name: str | None = None,
    history: bool = False,
    restart: bool = False,
    writable: bool = True,
) -> FieldContract:
    return FieldContract(
        standard_name=name,
        dtype=dtype,
        dimensions=tuple(dims),
        intent=intent,
        category=category,
        units=units,
        aliases=tuple(aliases),
        ccpp_standard_name=ccpp_standard_name,
        history=history,
        restart=restart,
        writable=writable,
    )


def default_contracts() -> tuple[FieldContract, ...]:
    """Return the complete legacy reference schema.

    New model workers use :func:`component_contracts` plus the contracts
    selected by ``CCPPStateSchema``.  Keeping this complete schema as the
    StatePool default preserves the standalone low-level API.
    """

    static = False
    return (
        # Configuration, constants, time, and indices.
        _field(
            "model_timestep",
            "float64",
            (),
            "in",
            "configuration",
            "s",
            aliases=("dt",),
            ccpp_standard_name="timestep_for_physics",
            writable=static,
        ),
        _field("dynamics_timestep", "float64", (), "in", "configuration", "s", writable=static),
        _field("vertical_remap_timestep", "float64", (), "in", "configuration", "s", writable=static),
        _field("hyperviscosity_subcycles", "int32", (), "in", "configuration", writable=static),
        _field("dynamics_nsplit", "int32", (), "in", "configuration", writable=static),
        _field("dynamics_qsplit", "int32", (), "in", "configuration", writable=static),
        _field("dynamics_rsplit", "int32", (), "in", "configuration", writable=static),
        _field("dynamics_timestep_type", "int32", (), "in", "configuration", writable=static),
        _field("dynamics_forcing_type", "int32", (), "in", "configuration", writable=static),
        _field("pressure_hyperviscosity", "float64", (), "in", "configuration", "m4 s-1", aliases=("nu_p",), writable=static),
        _field("velocity_hyperviscosity", "float64", (), "in", "configuration", "m4 s-1", aliases=("nu",), writable=static),
        _field("divergence_hyperviscosity", "float64", (), "in", "configuration", "m4 s-1", aliases=("nu_div",), writable=static),
        _field("temperature_hyperviscosity", "float64", (), "in", "configuration", "m4 s-1", aliases=("nu_s",), writable=static),
        _field("tracer_hyperviscosity", "float64", (), "in", "configuration", "m4 s-1", aliases=("nu_q",), writable=static),
        _field("sponge_top_viscosity", "float64", (), "in", "configuration", "m2 s-1", aliases=("nu_top",), writable=static),
        _field("sponge_level_count", "int32", (), "in", "configuration", aliases=("ksponge_end",), writable=static),
        _field("sponge_viscosity_scale", "float64", ("pver",), "in", "configuration", writable=static),
        _field("mpi_rank", "int32", (), "in", "configuration", writable=static),
        _field("mpi_size", "int32", (), "in", "configuration", writable=static),
        _field("spectral_element_count", "int32", (), "in", "configuration", writable=static),
        _field("vertical_level_count", "int32", (), "in", "configuration", writable=static),
        _field("physics_column_count", "int32", (), "in", "configuration", writable=static),
        _field("model_step", "int64", (), "inout", "time", aliases=("nstep",), restart=True),
        _field("current_date", "int32", (), "inout", "time", aliases=("date",), restart=True),
        _field("current_seconds_of_day", "int32", (), "inout", "time", "s", aliases=("datesec",), restart=True),
        _field("dynamics_time_level_nm1", "int32", (), "inout", "time", restart=True),
        _field("dynamics_time_level_n0", "int32", (), "inout", "time", restart=True),
        _field("dynamics_time_level_np1", "int32", (), "inout", "time", restart=True),
        _field("dynamics_internal_step", "int64", (), "inout", "time", restart=True),
        _field(
            "reference_pressure",
            "float64",
            (),
            "in",
            "constants",
            "Pa",
            aliases=("ps0",),
            ccpp_standard_name="surface_reference_pressure",
            writable=static,
        ),
        _field(
            "gravitational_acceleration",
            "float64",
            (),
            "in",
            "constants",
            "m s-2",
            aliases=("gravit",),
            ccpp_standard_name="standard_gravitational_acceleration",
            writable=static,
        ),
        _field(
            "reciprocal_gravitational_acceleration",
            "float64",
            (),
            "in",
            "constants",
            "s2 m-1",
            writable=static,
        ),
        _field("dry_air_gas_constant", "float64", (), "in", "constants", "J kg-1 K-1", aliases=("rair",), writable=static),
        _field("water_vapor_gas_constant", "float64", (), "in", "constants", "J kg-1 K-1", aliases=("rh2o",), writable=static),
        _field("virtual_temperature_coefficient", "float64", (), "in", "constants", aliases=("zvir",), writable=static),
        _field("dry_air_specific_heat", "float64", (), "in", "constants", "J kg-1 K-1", aliases=("cpair",), writable=static),
        _field("dry_air_kappa", "float64", (), "in", "constants", writable=static),
        _field("earth_radius", "float64", (), "in", "constants", "m", aliases=("rearth",), writable=static),
        _field("earth_angular_velocity", "float64", (), "in", "constants", "s-1", aliases=("omega_earth",), writable=static),
        _field("orbital_eccentricity", "float64", (), "in", "constants", writable=static),
        _field("orbital_obliquity", "float64", (), "in", "constants", "radian", writable=static),
        _field("orbital_longitude_of_perihelion", "float64", (), "in", "constants", "radian", writable=static),
        _field("water_to_dry_molecular_weight_ratio", "float64", (), "in", "constants", writable=static),
        _field(
            "latent_heat_of_vaporization",
            "float64",
            (),
            "in",
            "constants",
            "J kg-1",
            ccpp_standard_name="latent_heat_of_vaporization_of_water_at_0c",
            writable=static,
        ),
        _field(
            "latent_heat_of_fusion",
            "float64",
            (),
            "in",
            "constants",
            "J kg-1",
            writable=static,
        ),
        _field(
            "water_freezing_temperature",
            "float64",
            (),
            "in",
            "constants",
            "K",
            writable=static,
        ),
        _field(
            "water_triple_point_temperature",
            "float64",
            (),
            "in",
            "constants",
            "K",
            writable=static,
        ),
        _field(
            "universal_gas_constant",
            "float64",
            (),
            "in",
            "constants",
            "J K-1 kmol-1",
            writable=static,
        ),
        _field(
            "avogadro_constant",
            "float64",
            (),
            "in",
            "constants",
            "molecule kmol-1",
            writable=static,
        ),
        _field(
            "boltzmann_constant",
            "float64",
            (),
            "in",
            "constants",
            "J K-1 molecule-1",
            writable=static,
        ),
        _field(
            "circle_constant",
            "float64",
            (),
            "in",
            "constants",
            writable=static,
        ),
        _field(
            "liquid_water_density",
            "float64",
            (),
            "in",
            "constants",
            "kg m-3",
            ccpp_standard_name="fresh_liquid_water_density_at_0c",
            writable=static,
        ),
        _field("water_vapor_specific_heat", "float64", (), "in", "constants", "J kg-1 K-1", aliases=("cpwv",), writable=static),
        _field("liquid_water_specific_heat", "float64", (), "in", "constants", "J kg-1 K-1", aliases=("cpliq",), writable=static),
        # Hybrid vertical coordinate.
        _field("hybrid_a_interface", "float64", ("pverp",), "in", "vertical_coordinate", aliases=("hyai",), writable=static),
        _field("hybrid_b_interface", "float64", ("pverp",), "in", "vertical_coordinate", aliases=("hybi",), writable=static),
        _field("hybrid_a_midpoint", "float64", ("pver",), "in", "vertical_coordinate", aliases=("hyam",), writable=static),
        _field("hybrid_b_midpoint", "float64", ("pver",), "in", "vertical_coordinate", aliases=("hybm",), writable=static),
        _field("reference_interface_pressure", "float64", ("pverp",), "in", "vertical_coordinate", "Pa", writable=static),
        _field("reference_midpoint_pressure", "float64", ("pver",), "in", "vertical_coordinate", "Pa", writable=static),
        # Element/rank decomposition, topology, global/local DOFs, and halo schedule.
        _field("global_element_id", "int32", ("nelem_local",), "in", "topology", aliases=("element_id",), writable=static),
        _field("cube_face", "int32", ("nelem_local",), "in", "topology", writable=static),
        _field("cube_element_i", "int32", ("nelem_local",), "in", "topology", writable=static),
        _field("cube_element_j", "int32", ("nelem_local",), "in", "topology", writable=static),
        _field("cube_corner_angle", "float64", ("metric_i", "edge_count", "nelem_local"), "in", "topology", "radian", writable=static),
        _field("mapping_uniform_to_quadrilateral", "float64", ("edge_count", "metric_i", "nelem_local"), "in", "mapping", writable=static),
        _field("space_filling_curve_index", "int32", ("nelem_local",), "in", "topology", aliases=("sfc_index",), writable=static),
        _field("element_owner_rank", "int32", ("nelem_local",), "in", "topology", writable=static),
        _field("gll_global_dof", "int64", ("np", "np", "nelem_local"), "in", "topology", writable=static),
        _field("physics_global_column", "int64", ("fv_nphys", "fv_nphys", "nelem_local"), "in", "topology", writable=static),
        _field("halo_peer_rank", "int32", ("nhalo_peer",), "in", "communication", writable=static),
        _field("halo_shared_dof_count", "int32", ("nhalo_peer",), "in", "communication", writable=static),
        _field("halo_shared_dof", "int64", ("nhalo_dof",), "in", "communication", writable=static),
        _field("halo_shared_dof_offset", "int32", ("nhalo_peerp",), "in", "communication", writable=static),
        _field("pg3_halo_global_column", "int64", ("fvm_halo", "fvm_halo", "nelem_local"), "in", "communication", writable=static),
        # GLL/PG3 grid, metrics, operators, and mass matrices.
        _field("gll_node", "float64", ("np",), "in", "grid", writable=static),
        _field("gll_weight", "float64", ("np",), "in", "grid", writable=static),
        _field("gll_derivative", "float64", ("np", "np"), "in", "grid", aliases=("derivative_matrix",), writable=static),
        _field("gll_longitude", "float64", ("np", "np", "nelem_local"), "in", "grid", "radian", writable=static),
        _field("gll_latitude", "float64", ("np", "np", "nelem_local"), "in", "grid", "radian", writable=static),
        _field("gll_cartesian", "float64", ("cartesian", "np", "np", "nelem_local"), "in", "grid", writable=static),
        _field("metric_jacobian", "float64", ("np", "np", "nelem_local"), "in", "grid", aliases=("metdet",), writable=static),
        _field("inverse_metric_jacobian", "float64", ("np", "np", "nelem_local"), "in", "grid", aliases=("rmetdet",), writable=static),
        _field("metric_derivative", "float64", ("metric_i", "metric_j", "np", "np", "nelem_local"), "in", "grid", aliases=("D",), writable=static),
        _field("inverse_metric", "float64", ("metric_i", "metric_j", "np", "np", "nelem_local"), "in", "grid", aliases=("Dinv",), writable=static),
        _field("inverse_metric_tensor", "float64", ("metric_i", "metric_j", "np", "np", "nelem_local"), "in", "grid", aliases=("metinv",), writable=static),
        _field("spectral_mass_matrix", "float64", ("np", "np", "nelem_local"), "in", "grid", aliases=("spheremp",), writable=static),
        _field("inverse_spectral_mass_matrix", "float64", ("np", "np", "nelem_local"), "in", "grid", aliases=("rspheremp",), writable=static),
        _field("physics_longitude", "float64", ("nphys_local",), "in", "grid", "radian", aliases=("lon",), history=True, writable=static),
        _field("physics_latitude", "float64", ("nphys_local",), "in", "grid", "radian", aliases=("lat",), history=True, writable=static),
        _field("physics_cell_area", "float64", ("nphys_local",), "in", "grid", "steradian", aliases=("area",), history=True, writable=static),
        _field("coriolis_parameter", "float64", ("np", "np", "nelem_local"), "in", "grid", "s-1", aliases=("fcor",), writable=static),
        _field("fvm_cube_boundary", "int32", ("nelem_local",), "in", "topology", writable=static),
        _field("fvm_normalized_element_coordinate", "float64", ("metric_i", "fvm_halo", "fvm_halo", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_inverse_metric_physgrid", "float64", ("metric_i", "metric_j", "fvm_halo", "fvm_halo", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_reference_pressure_thickness", "float64", ("pver", "nelem_local"), "in", "grid", "Pa", writable=static),
        _field("fvm_inverse_reference_pressure_thickness", "float64", ("pver", "nelem_local"), "in", "grid", "Pa-1", writable=static),
        _field("fvm_cell_area", "float64", ("fv_nphys", "fv_nphys", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_inverse_cell_area", "float64", ("fv_nphys", "fv_nphys", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_displacement_maximum", "float64", ("fvm_halo", "fvm_halo", "edge_count", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_flux_vector", "int32", ("metric_i", "fvm_halo", "fvm_halo", "edge_count", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_vertex_cartesian", "float64", ("edge_count", "metric_i", "fvm_halo", "fvm_halo", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_flux_orientation", "float64", ("metric_i", "fvm_halo", "fvm_halo", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_cell_indicator", "int32", ("fvm_halo", "fvm_halo", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_rotation_matrix", "int32", ("metric_i", "metric_j", "fvm_halo", "fvm_halo", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_sphere_centroid", "float64", ("fvm_reconstruction", "fvm_halo", "fvm_halo", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_reconstruction_metric", "float64", ("fvm_reconstruction_terms", "fvm_internal", "fvm_internal", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_reconstruction_metric_integral", "float64", ("fvm_reconstruction_terms", "fvm_internal", "fvm_internal", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_jx_min", "int32", ("fvm_halo_range", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_jx_max", "int32", ("fvm_halo_range", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_jy_min", "int32", ("fvm_halo_range", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_jy_max", "int32", ("fvm_halo_range", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_interpolation_base", "int32", ("fvm_interp_span", "metric_i", "metric_j", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_halo_interpolation_weight", "float64", ("fv_nphys", "fvm_interp_span", "metric_i", "metric_j", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_centroid_stretch", "float64", ("fvm_stretch", "fvm_internal", "fvm_internal", "nelem_local"), "in", "grid", writable=static),
        _field("fvm_vertex_reconstruction_weight", "float64", ("edge_count", "fvm_reconstruction", "fvm_internal", "fvm_internal", "nelem_local"), "in", "grid", writable=static),
        # SE prognostic state and multiple time levels.
        _field("zonal_wind", "float64", ("np", "np", "pver", "nelem_local", "ntime"), "inout", "se_state", "m s-1", aliases=("U",), restart=True),
        _field("meridional_wind", "float64", ("np", "np", "pver", "nelem_local", "ntime"), "inout", "se_state", "m s-1", aliases=("V",), restart=True),
        _field("air_temperature", "float64", ("np", "np", "pver", "nelem_local", "ntime"), "inout", "se_state", "K", aliases=("T",), restart=True),
        _field("surface_pressure", "float64", ("np", "np", "nelem_local", "ntime"), "inout", "se_state", "Pa", aliases=("PS",), restart=True),
        _field("surface_dry_air_pressure", "float64", ("np", "np", "nelem_local"), "inout", "se_state", "Pa", aliases=("psdry",), restart=True),
        _field("layer_pressure_thickness", "float64", ("np", "np", "pver", "nelem_local", "ntime"), "inout", "se_state", "Pa", aliases=("DP",), restart=True),
        _field("constituent_mixing_ratio", "float64", ("np", "np", "pver", "nelem_local", "nconst", "ntime"), "inout", "constituents", "kg kg-1", aliases=("Q",), restart=True),
        _field("constituent_mass", "float64", ("np", "np", "pver", "nelem_local", "qsize_storage", "ntracer_time"), "inout", "fvm_state", "Pa kg kg-1", aliases=("Qdp",), restart=True),
        _field("fvm_layer_pressure_thickness", "float64", ("fvm_halo", "fvm_halo", "pver", "nelem_local"), "inout", "fvm_state", "Pa", aliases=("dp_fvm",), restart=True),
        _field("fvm_surface_dry_air_pressure", "float64", ("fv_nphys", "fv_nphys", "nelem_local"), "inout", "fvm_state", "Pa", aliases=("ps_fvm",), restart=True),
        _field("fvm_surface_geopotential", "float64", ("fv_nphys", "fv_nphys", "nelem_local"), "inout", "fvm_state", "m2 s-2", aliases=("phis_fvm",), restart=True),
        _field("fvm_tracer", "float64", ("fvm_halo", "fvm_halo", "pver", "nelem_local", "nconst"), "inout", "fvm_state", "kg kg-1", restart=True),
        _field("fvm_swept_flux", "float64", ("fvm_internal", "fvm_internal", "edge_count", "pver", "nelem_local"), "inout", "fvm_process", "1"),
        _field("constituent_index", "int32", ("nconst",), "in", "configuration", writable=static),
        _field("constituent_minimum", "float64", ("nconst",), "in", "configuration", "kg kg-1", writable=static),
        _field("constituent_molecular_weight", "float64", ("nconst",), "in", "configuration", "kg kmol-1", writable=static),
        # Dynamics derived state and forcing.
        _field("virtual_temperature", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "K"),
        _field("exner_function_gll", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived"),
        _field("pressure_midpoint_gll", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa"),
        _field("vertical_pressure_velocity", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa s-1", aliases=("OMEGA",)),
        _field("vertical_pressure_velocity_raw", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "Pa s-1"),
        _field("vertical_pressure_velocity_after_dss", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "Pa s-1"),
        _field("omega_biharmonic_stage", "float64", ("np", "np", "pver", "nelem_local", "nhypervis"), "out", "dynamics_diagnostic", "Pa m-4 s-1"),
        _field("omega_after_hypervis_stage", "float64", ("np", "np", "pver", "nelem_local", "nhypervis"), "out", "dynamics_diagnostic", "Pa s-1"),
        _field("surface_geopotential_gll", "float64", ("np", "np", "nelem_local"), "inout", "dynamics_derived", "m2 s-2", aliases=("PHIS",), restart=True),
        _field("mean_horizontal_mass_flux", "float64", ("np", "np", "metric_i", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa m s-1", aliases=("vn0",)),
        _field("pressure_at_step_start", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa"),
        _field("mass_flux_divergence", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa s-1", aliases=("divdp",)),
        _field("projected_mass_flux_divergence", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa s-1", aliases=("divdp_proj",)),
        _field("pressure_dissipation_average", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa s-1", aliases=("dpdiss_ave",)),
        _field("pressure_dissipation_biharmonic", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa s-1", aliases=("dpdiss_biharmonic",)),
        _field("tracer_stage_minimum", "float64", ("pver", "qsize", "nelem_local"), "inout", "tracer_process", "kg kg-1", aliases=("qmin",)),
        _field("tracer_stage_maximum", "float64", ("pver", "qsize", "nelem_local"), "inout", "tracer_process", "kg kg-1", aliases=("qmax",)),
        _field("subelement_mass_flux", "float64", ("fv_nphys", "fv_nphys", "edge_count", "pver", "nelem_local"), "inout", "dynamics_derived", "Pa s-1"),
        _field("rk_water_mixing_ratio", "float64", ("np", "np", "pver", "nelem_local", "nconst"), "inout", "dynamics_derived", "kg kg-1"),
        _field("rk_inverse_heat_capacity", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived", "kg K J-1"),
        _field("rk_kappa", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_derived"),
        _field("rk_geopotential", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "m2 s-2"),
        _field("rk_vorticity", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "s-1"),
        _field("rk_dry_mass_flux_divergence", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "Pa s-1"),
        _field("rk_full_mass_flux_divergence", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "Pa s-1"),
        _field("rk_horizontal_pressure_advection", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "Pa s-1"),
        _field("rk_zonal_wind_tendency", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "m s-2"),
        _field("rk_meridional_wind_tendency", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "m s-2"),
        _field("rk_temperature_tendency", "float64", ("np", "np", "pver", "nelem_local"), "out", "dynamics_diagnostic", "K s-1"),
        _field("zonal_wind_forcing", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_forcing", "m s-2"),
        _field("meridional_wind_forcing", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_forcing", "m s-2"),
        _field("temperature_forcing", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_forcing", "K s-1"),
        _field("constituent_forcing", "float64", ("np", "np", "pver", "nelem_local", "nconst"), "inout", "dynamics_forcing", "kg kg-1 s-1"),
        _field("forcing_full_layer_pressure_thickness", "float64", ("np", "np", "pver", "nelem_local"), "inout", "dynamics_forcing", "Pa", aliases=("FDP",)),
        _field("fvm_temperature_forcing", "float64", ("fv_nphys", "fv_nphys", "pver", "nelem_local"), "inout", "dynamics_forcing", "K s-1"),
        _field("fvm_momentum_forcing", "float64", ("fv_nphys", "fv_nphys", "metric_i", "pver", "nelem_local"), "inout", "dynamics_forcing", "m s-2"),
        _field("fvm_constituent_adjustment", "float64", ("fv_nphys", "fv_nphys", "pver", "nelem_local", "nconst"), "inout", "dynamics_forcing", "kg kg-1"),
        _field("fvm_constituent_mass_forcing", "float64", ("fv_nphys", "fv_nphys", "pver", "nelem_local", "qsize_storage"), "inout", "dynamics_forcing", "Pa kg kg-1 s-1", aliases=("fc",)),
        _field("fvm_dry_pressure_from_physics", "float64", ("fv_nphys", "fv_nphys", "pver", "nelem_local"), "inout", "dynamics_forcing", "Pa"),
        # Physics state and tendencies on PG3 columns.
        _field("physics_zonal_wind", "float64", ("nphys_local", "pver"), "inout", "physics_state", "m s-1", aliases=("phys_u",), history=True),
        _field("physics_meridional_wind", "float64", ("nphys_local", "pver"), "inout", "physics_state", "m s-1", aliases=("phys_v",), history=True),
        _field(
            "physics_air_temperature",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_state",
            "K",
            aliases=("phys_t",),
            ccpp_standard_name="air_temperature",
            history=True,
        ),
        _field("physics_surface_pressure", "float64", ("nphys_local",), "inout", "physics_state", "Pa", aliases=("phys_ps",), history=True),
        _field("physics_surface_dry_air_pressure", "float64", ("nphys_local",), "inout", "physics_state", "Pa", aliases=("phys_psdry",), history=True),
        _field(
            "physics_surface_geopotential",
            "float64",
            ("nphys_local",),
            "inout",
            "physics_state",
            "m2 s-2",
            aliases=("phys_phis",),
            ccpp_standard_name="surface_geopotential",
            history=True,
        ),
        _field("physics_layer_pressure_thickness", "float64", ("nphys_local", "pver"), "inout", "physics_state", "Pa", aliases=("phys_dp",), history=True),
        _field("physics_dry_layer_pressure_thickness", "float64", ("nphys_local", "pver"), "inout", "physics_state", "Pa", aliases=("phys_dpdry",), history=True),
        _field("physics_midpoint_pressure", "float64", ("nphys_local", "pver"), "inout", "physics_state", "Pa", aliases=("phys_pmid",), history=True),
        _field("physics_dry_midpoint_pressure", "float64", ("nphys_local", "pver"), "inout", "physics_state", "Pa", aliases=("phys_pmiddry",), history=True),
        _field("physics_interface_pressure", "float64", ("nphys_local", "pverp"), "inout", "physics_state", "Pa", aliases=("phys_pint",), history=True),
        _field("physics_dry_interface_pressure", "float64", ("nphys_local", "pverp"), "inout", "physics_state", "Pa", aliases=("phys_pintdry",), history=True),
        _field("physics_reciprocal_layer_pressure_thickness", "float64", ("nphys_local", "pver"), "inout", "physics_state", "Pa-1", aliases=("phys_rpdel",), history=True),
        _field("physics_reciprocal_dry_layer_pressure_thickness", "float64", ("nphys_local", "pver"), "inout", "physics_state", "Pa-1", aliases=("phys_rpdeldry",), history=True),
        _field("physics_log_midpoint_pressure", "float64", ("nphys_local", "pver"), "inout", "physics_state", aliases=("phys_lnpmid",), history=True),
        _field("physics_log_dry_midpoint_pressure", "float64", ("nphys_local", "pver"), "inout", "physics_state", aliases=("phys_lnpmiddry",), history=True),
        _field("physics_log_interface_pressure", "float64", ("nphys_local", "pverp"), "inout", "physics_state", aliases=("phys_lnpint",), history=True),
        _field("physics_log_dry_interface_pressure", "float64", ("nphys_local", "pverp"), "inout", "physics_state", aliases=("phys_lnpintdry",), history=True),
        _field("physics_inverse_surface_exner", "float64", ("nphys_local", "pver"), "inout", "physics_state", aliases=("phys_inv_exner",), history=True),
        _field("physics_vertical_pressure_velocity", "float64", ("nphys_local", "pver"), "inout", "physics_state", "Pa s-1", aliases=("phys_omega",), history=True),
        _field("physics_interface_geopotential_height", "float64", ("nphys_local", "pverp"), "inout", "physics_state", "m", aliases=("phys_zi",), history=True),
        _field("physics_constituent_mixing_ratio", "float64", ("nphys_local", "pver", "nconst"), "inout", "physics_state", "kg kg-1", aliases=("phys_q",), history=True),
        _field(
            "physics_air_temperature_tendency",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_tendency",
            "K s-1",
            aliases=("ttend_t",),
            ccpp_standard_name=(
                "tendency_of_air_temperature_due_to_model_physics"
            ),
        ),
        _field("physics_zonal_wind_tendency", "float64", ("nphys_local", "pver"), "inout", "physics_tendency", "m s-2"),
        _field("physics_meridional_wind_tendency", "float64", ("nphys_local", "pver"), "inout", "physics_tendency", "m s-2"),
        _field("physics_constituent_tendency", "float64", ("nphys_local", "pver", "nconst"), "inout", "physics_tendency", "kg kg-1 s-1"),
        _field("physics_constituent_previous", "float64", ("nphys_local", "pver", "nconst"), "inout", "coupling", "kg kg-1"),
        # Metadata-selected physics process state.
        _field(
            "potential_temperature",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "K",
            aliases=("theta",),
            ccpp_standard_name="air_potential_temperature",
        ),
        _field(
            "exner_function",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            aliases=("exner",),
            ccpp_standard_name="dimensionless_exner_function",
        ),
        _field(
            "dry_air_density",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "kg m-3",
            aliases=("rho",),
            ccpp_standard_name="dry_air_density",
        ),
        _field(
            "thermodynamic_level_height",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "m",
            aliases=("z",),
            ccpp_standard_name="geopotential_height_wrt_surface",
        ),
        _field(
            "column_dry_air_specific_heat",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "J kg-1 K-1",
            aliases=("cpair_column",),
            ccpp_standard_name=(
                "composition_dependent_specific_heat_of_dry_air_at_"
                "constant_pressure"
            ),
        ),
        _field(
            "column_dry_air_gas_constant",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "J kg-1 K-1",
            aliases=("rair_column",),
            ccpp_standard_name="composition_dependent_gas_constant_of_dry_air",
        ),
        _field(
            "column_dry_air_kappa",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            ccpp_standard_name=(
                "composition_dependent_ratio_of_dry_air_gas_constant_to_"
                "specific_heat_of_dry_air_at_constant_pressure"
            ),
        ),
        _field(
            "air_temperature_previous_timestep",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "K",
            aliases=("temp_prev", "temperature_before_kessler"),
            ccpp_standard_name="air_temperature_on_previous_timestep",
        ),
        _field("dycore_heat_capacity", "float64", ("nphys_local", "pver"), "inout", "coupling", "J kg-1 K-1", aliases=("cp_or_cv_dycore",)),
        _field("dycore_energy_scaling", "float64", ("nphys_local", "pver"), "inout", "coupling", aliases=("scaling_dycore",)),
        _field("temperature_consistency_tendency", "float64", ("nphys_local", "pver"), "inout", "physics_tendency", "K s-1"),
        _field(
            "large_scale_precipitation_rate",
            "float64",
            ("nphys_local",),
            "inout",
            "physics_process",
            "m s-1",
            aliases=("PRECL",),
            ccpp_standard_name="total_precipitation_rate_at_surface",
            history=True,
        ),
        _field(
            "relative_humidity",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "%",
            aliases=("RELHUM",),
            ccpp_standard_name="relative_humidity",
            history=True,
        ),
        _field(
            "static_energy",
            "float64",
            ("nphys_local", "pver"),
            "inout",
            "physics_process",
            "J kg-1",
            ccpp_standard_name="dry_static_energy",
        ),
        # Coupling/mapping buffers and history accumulators.
        _field("dynamics_to_physics_buffer", "float64", ("nphys_local", "pver", "mapping_fields"), "inout", "mapping"),
        _field("physics_to_dynamics_buffer", "float64", ("np", "np", "pver", "nelem_local", "mapping_fields"), "inout", "mapping"),
        _field("mapping_weights_gll_to_pg3", "float64", ("fv_nphys", "np"), "in", "mapping", writable=static),
        _field("mapping_weights_pg3_to_gll", "float64", ("np", "fv_nphys"), "in", "mapping", writable=static),
        _field("mapping_subcell_integration", "float64", ("fv_nphys", "np"), "in", "mapping", writable=static),
        _field("mapping_boundary_interpolation", "float64", ("fv_nphys", "metric_i", "np"), "in", "mapping", writable=static),
        _field("mapping_interpolation_matrix", "float64", ("np", "np"), "in", "mapping", writable=static),
        _field("mapping_derivative_pg3", "float64", ("metric_i", "metric_j", "nphys_local"), "in", "mapping", aliases=("D_phys",), writable=static),
        _field("history_sample_count", "int64", (), "inout", "history"),
        _field("history_precipitation_accumulator", "float64", ("nphys_local",), "inout", "history", "m"),
        _field("coupler_import_buffer", "float64", ("nphys_local", "coupler_fields"), "inout", "coupling"),
        _field("coupler_export_buffer", "float64", ("nphys_local", "coupler_fields"), "inout", "coupling"),
    )


def component_contracts() -> tuple[FieldContract, ...]:
    """Return suite-independent CAM SE/FVM state."""

    return tuple(
        contract
        for contract in default_contracts()
        if contract.category != "physics_process"
    )


def process_contract_templates() -> tuple[FieldContract, ...]:
    """Return named process fields selected only when metadata requires them."""

    return tuple(
        contract
        for contract in default_contracts()
        if contract.category == "physics_process"
    )


def default_alias_rules() -> tuple[AliasRule, ...]:
    """Return aliases for the historical three-constituent Kessler layout."""

    return model_alias_rules(
        ("cloud_liquid_water", "rain_water", "water_vapor")
    )


def model_alias_rules(
    constituent_names: Iterable[str],
) -> tuple[AliasRule, ...]:
    """Build zero-copy species views from the configured constituent order.

    CAM suites do not all use the Kessler ``cloud, rain, vapor`` layout.  In
    particular, the dry idealized suites carry only water vapor.  The alias
    index must therefore come from ``ModelConfig.constituent_names`` instead
    of being embedded in the host model.
    """

    normalized = tuple(
        str(name).strip().lower() for name in constituent_names
    )
    aliases: list[AliasRule] = []
    species = {
        "cloud_ice": "physics_cloud_ice",
        "cloud_liquid_water": "physics_cloud_liquid_water",
        "rain_water": "physics_rain_water",
        "water_vapor": "physics_water_vapor",
    }
    for constituent, physics_alias in species.items():
        if constituent not in normalized:
            continue
        index = normalized.index(constituent)
        aliases.append(
            AliasRule(
                constituent,
                "constituent_mixing_ratio",
                -2,
                index,
            )
        )
        aliases.append(
            AliasRule(
                physics_alias,
                "physics_constituent_mixing_ratio",
                -1,
                index,
            )
        )
    return tuple(aliases)


def model_ccpp_field_aliases(
    constituent_names: Iterable[str],
) -> dict[str, str]:
    """Map CCPP standard names onto the Python-owned component arrays.

    Values are canonical StatePool fields or zero-copy aliases created by
    :func:`model_alias_rules`.  These are host-model bindings: they prevent
    the metadata compiler from allocating a second, disconnected copy of a
    field that CAM already owns.
    """

    aliases = {
        # Prognostic/diagnostic physics state.
        "eastward_wind": "physics_zonal_wind",
        "northward_wind": "physics_meridional_wind",
        "surface_air_pressure": "physics_surface_pressure",
        "surface_pressure_of_dry_air": "physics_surface_dry_air_pressure",
        "air_pressure": "physics_midpoint_pressure",
        "air_pressure_at_interface": "physics_interface_pressure",
        "air_pressure_of_dry_air": "physics_dry_midpoint_pressure",
        "air_pressure_of_dry_air_at_interface": (
            "physics_dry_interface_pressure"
        ),
        "air_pressure_thickness": "physics_layer_pressure_thickness",
        "air_pressure_thickness_of_dry_air": (
            "physics_dry_layer_pressure_thickness"
        ),
        "reciprocal_of_air_pressure_thickness": (
            "physics_reciprocal_layer_pressure_thickness"
        ),
        "reciprocal_of_air_pressure_thickness_of_dry_air": (
            "physics_reciprocal_dry_layer_pressure_thickness"
        ),
        "ln_air_pressure": "physics_log_midpoint_pressure",
        "ln_air_pressure_at_interface": "physics_log_interface_pressure",
        "ln_air_pressure_of_dry_air": "physics_log_dry_midpoint_pressure",
        "ln_air_pressure_of_dry_air_at_interface": (
            "physics_log_dry_interface_pressure"
        ),
        "reciprocal_of_dimensionless_exner_function_wrt_surface_air_pressure": (
            "physics_inverse_surface_exner"
        ),
        "lagrangian_tendency_of_air_pressure": (
            "physics_vertical_pressure_velocity"
        ),
        "geopotential_height_wrt_surface_at_interface": (
            "physics_interface_geopotential_height"
        ),
        "latitude": "physics_latitude",
        "longitude": "physics_longitude",
        "latitude_degrees_north": "physics_latitude",
        "longitude_degrees_east": "physics_longitude",
        "cell_area": "physics_cell_area",
        # Accumulated model-physics tendencies.
        "tendency_of_air_temperature_due_to_model_physics": (
            "physics_air_temperature_tendency"
        ),
        "tendency_of_eastward_wind_due_to_model_physics": (
            "physics_zonal_wind_tendency"
        ),
        "tendency_of_northward_wind_due_to_model_physics": (
            "physics_meridional_wind_tendency"
        ),
        # Constituent registry and shared component work arrays.
        "ccpp_constituents": "physics_constituent_mixing_ratio",
        "specific_heat_of_air_used_in_dycore": "dycore_heat_capacity",
        "ratio_of_specific_heat_of_air_used_in_physics_energy_formula_to_"
        "specific_heat_of_air_used_in_dycore_energy_formula": (
            "dycore_energy_scaling"
        ),
        # Constants already owned and initialized by Python.
        "gas_constant_of_water_vapor": "water_vapor_gas_constant",
        "gas_constant_of_dry_air": "dry_air_gas_constant",
        "ratio_of_water_vapor_to_dry_air_molecular_weights": (
            "water_to_dry_molecular_weight_ratio"
        ),
        "latent_heat_of_fusion": "latent_heat_of_fusion",
        "latent_heat_of_fusion_of_water_at_0c": "latent_heat_of_fusion",
        "latent_heat_of_vaporization": "latent_heat_of_vaporization",
        "pi_constant": "circle_constant",
        "specific_heat_of_dry_air_at_constant_pressure": (
            "dry_air_specific_heat"
        ),
        "specific_heat_of_liquid_water_at_constant_pressure": (
            "liquid_water_specific_heat"
        ),
        "specific_heat_of_water_vapor_at_constant_pressure": (
            "water_vapor_specific_heat"
        ),
        "ratio_of_dry_air_gas_constant_to_specific_heat_of_dry_air_at_"
        "constant_pressure": "dry_air_kappa",
        "freezing_point_of_water": "water_freezing_temperature",
        "water_freezing_temperature": "water_freezing_temperature",
        "water_triple_point_temperature": (
            "water_triple_point_temperature"
        ),
    }
    normalized = tuple(
        str(name).strip().lower() for name in constituent_names
    )
    species_standards = {
        "cloud_ice": (
            "cloud_ice_mixing_ratio_wrt_dry_air",
            "cloud_ice_mixing_ratio_wrt_moist_air_and_condensed_water",
        ),
        "cloud_liquid_water": (
            "cloud_liquid_water_mixing_ratio_wrt_dry_air",
            "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
        ),
        "rain_water": (
            "rain_mixing_ratio_wrt_dry_air",
            "rain_mixing_ratio_wrt_moist_air_and_condensed_water",
        ),
        "water_vapor": (
            "water_vapor_mixing_ratio_wrt_dry_air",
            "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water",
        ),
    }
    for species, standards in species_standards.items():
        if species not in normalized:
            continue
        target = f"physics_{species}"
        for standard_name in standards:
            aliases[standard_name] = target
    return aliases


def export_contract(path: str | Path, contracts: Iterable[FieldContract] | None = None) -> None:
    records = [item.machine_record() for item in (contracts or default_contracts())]
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump({"version": 2, "fields": records}, stream, sort_keys=False)
