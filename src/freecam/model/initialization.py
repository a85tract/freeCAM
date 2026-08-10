"""Describable and pausable Python initialization plan.

Some routines in this module are transcribed from CESM/CAM Fortran sources
and preserve their upstream expression order; each is marked with a ``Port
...`` docstring naming its upstream routine.  Those routines are
Copyright (c) 2017, University Corporation for Atmospheric Research (UCAR)
and are redistributed under the BSD 3-Clause license in
LICENSES/UCAR-CESM-BSD-3-Clause.txt.  See NOTICE section 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Callable, Mapping
import xml.etree.ElementTree as ET

import numpy as np
from taskflow import engines, task
from taskflow.patterns import linear_flow

from .clock import ModelClock
from .constituents import water_constituent_indices
from .ccpp_state import NamelistBinding
from .config import ModelConfig
from .contracts import AliasRule, FieldContract
from .errors import ConfigurationError, ValidationError
from .fvm_geometry import generate_fvm_geometry
from .grid import dimensions_for_rank, populate_grid
from .initial_conditions import populate_initial_state
from .dimension_service import infer_suite_dimensions
from .namelist import read_atm_in
from .scientific_data import (
    read_musica_placeholder_data,
    read_ridge_gravity_wave_data,
    read_tropopause_climatology,
    stage_musica_tuvx_configuration,
)
from .state import StatePool
from .vertical import load_vertical_coordinate


_FORTRAN_KIND_SUFFIX = re.compile(r"_(?:kind_phys|r8|r4|i4|i8)\b", re.I)


def _registry_initial_value(text: str, pool: StatePool):
    """Evaluate the small literal vocabulary used by CAM registry.xml."""

    value = _FORTRAN_KIND_SUFFIX.sub("", text.strip())
    lower = value.lower()
    constants = {
        "cpair": pool.get("dry_air_specific_heat").item(),
        "rair": pool.get("dry_air_gas_constant").item(),
        "rair/cpair": pool.get("dry_air_kappa").item(),
        "mwdry": np.float64(28.966),
        "zvir": pool.get("virtual_temperature_coefficient").item(),
        "unset_real": np.finfo(np.float64).max,
    }
    if lower in constants:
        return constants[lower]
    if lower == ".true.":
        return True
    if lower == ".false.":
        return False
    if value == "''":
        return ""
    normalized = re.sub(r"(?<=\d)[dD](?=[+-]?\d)", "e", value)
    try:
        return int(normalized)
    except ValueError:
        try:
            return float(normalized)
        except ValueError as exc:
            raise ConfigurationError(
                f"unsupported CAM registry initial_value {text!r}"
            ) from exc


def _cam_registry_defaults(source_root: Path, pool: StatePool):
    """Yield exact host-field defaults declared by pinned CAM registry.xml."""

    source_root = Path(source_root)
    registry = source_root / "src/data/registry.xml"
    if not registry.is_file():
        raise ConfigurationError(f"missing CAM registry: {registry}")
    for variable in ET.parse(registry).getroot().iter("variable"):
        standard_name = variable.attrib.get("standard_name", "").strip()
        initial = variable.findtext("initial_value")
        if standard_name and initial is not None:
            yield standard_name, _registry_initial_value(initial, pool)


@dataclass(slots=True)
class InitializationContext:
    config: ModelConfig
    run_dir: Path
    comm: object
    configured_contracts: tuple[FieldContract, ...] | None = None
    generated_contracts: tuple[FieldContract, ...] = ()
    alias_rules: tuple[AliasRule, ...] | None = None
    ccpp_aliases: dict[str, str] = field(default_factory=dict)
    namelist_bindings: dict[str, tuple[NamelistBinding, ...]] = field(
        default_factory=dict
    )
    required_dimensions: frozenset[str] = frozenset()
    fixed_dimensions: dict[str, int] = field(default_factory=dict)
    atm: dict | None = None
    pool: StatePool | None = None
    clock: ModelClock | None = None
    completed: list[str] = field(default_factory=list)


class _InitTask(task.Task):
    def __init__(self, name: str, callback: Callable[[InitializationContext], None], context: InitializationContext):
        super().__init__(name=name)
        self.callback, self.context = callback, context

    def execute(self):
        self.callback(self.context)
        self.context.completed.append(self.name)


class InitializationPlan:
    STEP_NAMES = (
        "parse_configuration", "establish_mpi_layout", "allocate_state_pool",
        "load_vertical_coordinate", "initialize_constants_and_clock", "build_grid_and_topology",
        "register_constituents", "generate_initial_state", "initialize_derived_buffers",
        "validate_python_owned_state",
    )

    def __init__(
        self,
        config: ModelConfig,
        run_dir: str | Path,
        comm,
        *,
        contracts: tuple[FieldContract, ...] | None = None,
        generated_contracts: tuple[FieldContract, ...] = (),
        alias_rules: tuple[AliasRule, ...] | None = None,
        ccpp_aliases: dict[str, str] | None = None,
        namelist_bindings: Mapping[
            str, tuple[NamelistBinding, ...]
        ] | None = None,
        required_dimensions: frozenset[str] = frozenset(),
        fixed_dimensions: Mapping[str, int] | None = None,
    ):
        self.context = InitializationContext(
            config,
            Path(run_dir).resolve(),
            comm,
            configured_contracts=contracts,
            generated_contracts=generated_contracts,
            alias_rules=alias_rules,
            ccpp_aliases=dict(ccpp_aliases or {}),
            namelist_bindings={
                str(name).lower(): tuple(bindings)
                for name, bindings in (namelist_bindings or {}).items()
            },
            required_dimensions=frozenset(required_dimensions),
            fixed_dimensions={
                str(name).lower(): int(value)
                for name, value in (fixed_dimensions or {}).items()
            },
        )
        callbacks = (
            self._parse, self._mpi, self._allocate, self._vertical, self._constants,
            self._grid, self._constituents, self._initial_state, self._buffers, self._validate,
        )
        self._callbacks = dict(zip(self.STEP_NAMES, callbacks))

    def describe(self) -> list[dict[str, object]]:
        return [{"order": index + 1, "name": name, "completed": name in self.context.completed} for index, name in enumerate(self.STEP_NAMES)]

    def run(self, through: str | None = None) -> InitializationContext:
        if through is not None and through not in self.STEP_NAMES:
            raise ValueError(f"unknown initialization task {through!r}")
        start = len(self.context.completed)
        end = self.STEP_NAMES.index(through) + 1 if through else len(self.STEP_NAMES)
        if end < start:
            return self.context
        flow = linear_flow.Flow("python_initialization")
        for name in self.STEP_NAMES[start:end]:
            flow.add(_InitTask(name, self._callbacks[name], self.context))
        engines.run(flow, engine="serial")
        return self.context

    @staticmethod
    def _parse(ctx):
        ctx.config.validate()
        ctx.config.verify_source_revision()
        ctx.atm = read_atm_in(
            ctx.config.resolve_atm_in(ctx.run_dir),
            ctx.config,
        )

    @staticmethod
    def _mpi(ctx):
        if ctx.comm.size != ctx.config.mpi_size:
            raise ConfigurationError(
                f"the model requires {ctx.config.mpi_size} MPI ranks, got {ctx.comm.size}"
            )
        if not 0 <= ctx.comm.rank < ctx.comm.size:
            raise ConfigurationError(f"invalid MPI rank {ctx.comm.rank}/{ctx.comm.size}")

    @staticmethod
    def _allocate(ctx):
        dimensions = dimensions_for_rank(
            ctx.comm.rank,
            ctx.comm.size,
            pver=ctx.config.pver,
            np_value=ctx.config.np,
            fv_nphys=ctx.config.fv_nphys,
            constituent_count=ctx.config.constituent_count,
            advected_constituent_count=(
                ctx.config.advected_constituent_count
            ),
            thermodynamic_constituent_count=len(
                water_constituent_indices(
                    ctx.config.constituent_names
                )
            ),
            ne=ctx.config.ne,
            extra_dimensions={
                **ctx.fixed_dimensions,
                **ctx.config.dimension_overrides,
            },
        )
        inferred_dimensions = infer_suite_dimensions(
            required=ctx.required_dimensions,
            existing=dimensions,
            namelist=ctx.atm["namelist"],
            namelist_bindings=ctx.namelist_bindings,
            source_root=ctx.config.source_root,
        )
        for name, value in inferred_dimensions.items():
            if name in dimensions and dimensions[name] != value:
                raise ConfigurationError(
                    f"inferred dimension {name!r}={value} conflicts with "
                    f"configured value {dimensions[name]}"
                )
            dimensions[name] = value
        missing_dimensions = sorted(
            ctx.required_dimensions - set(dimensions)
        )
        if missing_dimensions:
            raise ConfigurationError(
                f"suite {ctx.config.physics_suite!r} requires runtime "
                f"dimensions {missing_dimensions}; define them under "
                "ModelConfig.dimension_overrides"
            )
        ctx.pool = StatePool(
            dimensions,
            contracts=ctx.configured_contracts,
            alias_rules=ctx.alias_rules,
            ccpp_aliases=ctx.ccpp_aliases,
            constituent_names=ctx.config.constituent_names,
            advected_constituent_indices=(
                ctx.config.advected_constituent_indices
            ),
        )
        for contract in ctx.generated_contracts:
            ctx.pool.register_field(
                contract,
                initialized=False,
                dynamic=False,
            )
        pool = ctx.pool
        pool.set("mpi_rank", ctx.comm.rank); pool.set("mpi_size", ctx.comm.size)
        pool.set("spectral_element_count", 6 * ctx.config.ne * ctx.config.ne)
        pool.set("vertical_level_count", ctx.config.pver)
        pool.set("physics_column_count", pool.dimensions["nphys_local"])
        pool.set("model_timestep", ctx.config.dt_seconds)
        pool.set("dynamics_timestep", np.float64(ctx.config.dt_seconds) / np.float64(6.0))
        pool.set("vertical_remap_timestep", np.float64(ctx.config.dt_seconds) / np.float64(2.0))
        pool.set("hyperviscosity_subcycles", 3)
        pool.set("dynamics_nsplit", 2)
        pool.set("dynamics_qsplit", 1)
        pool.set("dynamics_rsplit", 3)
        pool.set("dynamics_timestep_type", 4)
        pool.set("dynamics_forcing_type", 2)

    @staticmethod
    def _vertical(ctx):
        load_vertical_coordinate(
            ctx.pool,
            ctx.atm["ncdata"],
            ctx.comm,
            expected_levels=ctx.config.pver,
        )

    @staticmethod
    def _constants(ctx):
        pool = ctx.pool
        avogad, boltzmann, mwdry, mwh2o = 6.02214e26, 1.38065e-23, 28.966, 18.016
        universal_gas_constant = np.float64(avogad) * np.float64(boltzmann)
        dry_air_gas_constant = universal_gas_constant / np.float64(mwdry)
        water_vapor_gas_constant = universal_gas_constant / np.float64(mwh2o)
        values = {
            "gravitational_acceleration": 9.80616,
            "standard_gravitational_acceleration": 9.80616,
            "reciprocal_gravitational_acceleration": (
                np.float64(1.0) / np.float64(9.80616)
            ),
            "reciprocal_of_gravitational_acceleration": (
                np.float64(1.0) / np.float64(9.80616)
            ),
            "dry_air_gas_constant": dry_air_gas_constant,
            "gas_constant_of_dry_air": dry_air_gas_constant,
            "water_vapor_gas_constant": water_vapor_gas_constant,
            "gas_constant_of_water_vapor": water_vapor_gas_constant,
            "virtual_temperature_coefficient": np.float64(water_vapor_gas_constant / dry_air_gas_constant) - np.float64(1.0),
            "ratio_of_water_vapor_to_dry_air_gas_constants_minus_one": (
                np.float64(
                    water_vapor_gas_constant / dry_air_gas_constant
                )
                - np.float64(1.0)
            ),
            "dry_air_specific_heat": 1.00464e3,
            "specific_heat_of_dry_air_at_constant_pressure": 1.00464e3,
            "dry_air_kappa": (
                np.float64(dry_air_gas_constant) / np.float64(1.00464e3)
            ),
            "earth_radius": 6.37122e6,
            "radius_of_earth": 6.37122e6,
            "reciprocal_of_radius_of_earth": (
                np.float64(1.0) / np.float64(6.37122e6)
            ),
            "earth_angular_velocity": 2.0 * np.float64(3.14159265358979323846) / 86164.0,
            "angular_velocity_of_earth_rotation": (
                np.float64(2.0)
                * np.float64(3.14159265358979323846)
                / np.float64(86164.0)
            ),
            "water_to_dry_molecular_weight_ratio": mwh2o / mwdry,
            "ratio_of_water_vapor_to_dry_air_molecular_weights": (
                mwh2o / mwdry
            ),
            "latent_heat_of_vaporization": 2.501e6,
            "latent_heat_of_vaporization_of_water_at_0c": 2.501e6,
            "latent_heat_of_fusion": 3.337e5,
            "latent_heat_of_fusion_of_water_at_0c": 3.337e5,
            "water_freezing_temperature": 273.15,
            "freezing_point_of_water": 273.15,
            "water_triple_point_temperature": 273.16,
            "triple_point_temperature_of_water": 273.16,
            "density_of_dry_air_at_stp": (
                np.float64(101325.0)
                / (
                    np.float64(dry_air_gas_constant)
                    * np.float64(273.15)
                )
            ),
            "universal_gas_constant": universal_gas_constant,
            "avogadro_constant": avogad,
            "avogadro_number": avogad,
            "boltzmann_constant": boltzmann,
            "circle_constant": np.float64(3.14159265358979323846),
            "pi_constant": np.float64(3.14159265358979323846),
            "liquid_water_density": 1000.0,
            "fresh_liquid_water_density_at_0c": 1000.0,
            "water_vapor_specific_heat": 1.810e3,
            "specific_heat_of_water_vapor_at_constant_pressure": 1.810e3,
            "liquid_water_specific_heat": 4.188e3,
            "specific_heat_of_liquid_water_at_constant_pressure": 4.188e3,
            "specific_heat_of_fresh_ice": 2.11727e3,
            "seconds_in_calendar_day": 86400.0,
            "seconds_in_sidereal_day": 86164.0,
            "von_karman_constant": 0.4,
            "us_standard_atmospheric_pressure_at_sea_level": 101325.0,
            "surface_reference_pressure": 100000.0,
            "reference_temperature_at_sea_level": 288.0,
            "reference_temperature_lapse_rate": 0.0065,
            "stefan_boltzmanns_constant": 5.67e-8,
            "speed_of_light_in_vacuum": 2.99792458e8,
            "plancks_constant": 6.6260755e-34,
            "molecular_weight_of_co2": 44.0,
            "molecular_weight_of_n2o": 44.0,
            "molecular_weight_of_ch4": 16.0,
            "molecular_weight_of_cfc11": 136.0,
            "molecular_weight_of_cfc12": 120.0,
            "molecular_weight_of_o3": 48.0,
            "molecular_weight_of_so2": 64.0,
            "molecular_weight_of_so4": 96.0,
            "molecular_weight_of_h2o2": 34.0,
            "molecular_weight_of_dms": 62.0,
            "molecular_weight_of_nh4": 18.0,
            "molecular_weight_of_h2o": mwh2o,
            "molecular_weight_of_dry_air": mwdry,
            # rrtmgp_variables.unset_real is initialized from CAM's
            # ``unset_real = huge(1.0_r8)`` host definition.
            "definition_of_unset_for_real_variables": np.finfo(
                np.float64
            ).max,
            # Fixed year-0 orbital values are explicit state, not Fortran module data.
            "orbital_eccentricity": 0.016715,
            "orbital_obliquity": np.deg2rad(23.4441),
            "orbital_longitude_of_perihelion": np.deg2rad(102.7),
        }
        # The StatePool contract is suite-derived: dry idealized suites do not
        # allocate every moist/full-physics constant.  Populate only constants
        # requested by the selected suite instead of making one suite's host
        # requirements mandatory for all models.
        for name, value in values.items():
            if name in pool.contracts:
                field_name = name
            else:
                try:
                    field_name = pool.ccpp_field_name(name)
                except KeyError:
                    continue
            pool.set(field_name, value)
        exponent = np.float64(1.0) / np.log10(np.float64(2.0))
        nu_factor = np.float64(1.0e15) / np.float64(np.float64(110000.0) ** exponent)
        nu_p = nu_factor * np.float64(
            (
                np.float64(30.0)
                / np.float64(ctx.config.ne)
                * np.float64(110000.0)
            )
            ** exponent
        )
        pool.set("pressure_hyperviscosity", nu_p)
        pool.set("velocity_hyperviscosity", np.float64(0.5) * nu_p)
        pool.set("divergence_hyperviscosity", np.float64(2.5) * nu_p)
        pool.set("temperature_hyperviscosity", nu_p)
        pool.set("tracer_hyperviscosity", nu_p)
        pool.set("sponge_top_viscosity", np.float64(5.0e5))
        pool.set("sponge_level_count", pool.dimensions["nhypervis"])
        sponge_scale = np.empty(pool.dimensions["pver"], dtype=np.float64)
        ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(
            pool.get("reference_pressure")
        )
        for level in range(pool.dimensions["pver"]):
            pressure = np.float64(
                pool.get("hybrid_a_midpoint")[level]
                + pool.get("hybrid_b_midpoint")[level]
            ) * np.float64(pool.get("reference_pressure"))
            value = np.float64(8.0) * np.float64(
                np.float64(1.0)
                + np.float64(
                    math.tanh(
                        math.log(float(np.float64(ptop / pressure)))
                    )
                )
            )
            sponge_scale[level] = value if value >= np.float64(0.15) else 0.0
        pool.set("sponge_viscosity_scale", sponge_scale)
        start_year, start_month, start_day = (
            int(part) for part in ctx.config.start_date.split("-")
        )
        ctx.clock = ModelClock(
            year=start_year,
            month=start_month,
            day=start_day,
            seconds=ctx.config.start_seconds,
            dt_seconds=ctx.config.dt_seconds,
            calendar=ctx.config.calendar,
        )
        pool.set("model_step", 0)
        pool.set("current_date", ctx.clock.yyyymmdd)
        pool.set("current_seconds_of_day", ctx.clock.seconds)
        pool.set("dynamics_time_level_nm1", 0)
        pool.set("dynamics_time_level_n0", 1)
        pool.set("dynamics_time_level_np1", 2)
        pool.set("dynamics_internal_step", 0)

    @staticmethod
    def _grid(ctx):
        populate_grid(
            ctx.pool,
            ctx.comm.rank,
            ctx.comm.size,
            ne=ctx.config.ne,
        )
        InitializationPlan._generate_fvm_geometry(ctx)
        omega = np.float64(ctx.pool.get("earth_angular_velocity"))
        latitude = ctx.pool.get("gll_latitude")
        fcor = ctx.pool.get("coriolis_parameter")
        for le in range(ctx.pool.dimensions["nelem_local"]):
            for j in range(ctx.pool.dimensions["np"]):
                for i in range(ctx.pool.dimensions["np"]):
                    fcor[i, j, le] = np.float64(2.0) * omega * np.sin(latitude[i, j, le])

    @staticmethod
    def _generate_fvm_geometry(ctx):
        payload = generate_fvm_geometry(
            ctx.pool.get("hybrid_a_interface"),
            ctx.pool.get("hybrid_b_interface"),
            ctx.pool.get("reference_pressure"),
            ne=ctx.config.ne,
            nc=ctx.config.fv_nphys,
        )
        gids = np.asarray(payload["global_element_id"], dtype=np.int32)
        wanted = ctx.pool.get("global_element_id")
        indices = np.array([int(np.flatnonzero(gids == gid)[0]) for gid in wanted])
        mapping = {
            "cube_boundary": "fvm_cube_boundary",
            "dp_ref": "fvm_reference_pressure_thickness",
            "dp_ref_inverse": "fvm_inverse_reference_pressure_thickness",
            "area_sphere": "fvm_cell_area",
            "inverse_area_sphere": "fvm_inverse_cell_area",
            "displacement_maximum": "fvm_displacement_maximum",
            "flux_vector": "fvm_flux_vector",
            "vertex_cartesian": "fvm_vertex_cartesian",
            "flux_orientation": "fvm_flux_orientation",
            "cell_indicator": "fvm_cell_indicator",
            "rotation_matrix": "fvm_rotation_matrix",
            "sphere_centroid": "fvm_sphere_centroid",
            "reconstruction_metric": "fvm_reconstruction_metric",
            "reconstruction_metric_integral": "fvm_reconstruction_metric_integral",
            "jx_min": "fvm_jx_min",
            "jx_max": "fvm_jx_max",
            "jy_min": "fvm_jy_min",
            "jy_max": "fvm_jy_max",
            "interpolation_base": "fvm_interpolation_base",
            "halo_interpolation_weight": "fvm_halo_interpolation_weight",
            "centroid_stretch": "fvm_centroid_stretch",
            "vertex_reconstruction_weight": "fvm_vertex_reconstruction_weight",
        }
        for source, target in mapping.items():
            ctx.pool.set(target, np.asarray(payload[source])[..., indices])

    @staticmethod
    def _constituents(ctx):
        count = ctx.config.constituent_count
        ctx.pool.set(
            "constituent_index",
            np.arange(1, count + 1, dtype=np.int32),
        )
        ctx.pool.set(
            "constituent_minimum",
            np.asarray(ctx.config.constituent_minima, dtype=np.float64),
        )
        ctx.pool.set(
            "constituent_molecular_weight",
            np.asarray(
                ctx.config.constituent_molecular_weights,
                dtype=np.float64,
            ),
        )

    @staticmethod
    def _initial_state(ctx):
        populate_initial_state(
            ctx.pool,
            ctx.comm,
            kind=ctx.config.analytic_ic_type,
            temperature=ctx.config.initial_temperature,
            surface_pressure=ctx.config.initial_surface_pressure,
            constituent_names=ctx.config.constituent_names,
        )
        InitializationPlan._scale_full_physics_dry_mass(ctx)

    @staticmethod
    def _scale_full_physics_dry_mass(ctx):
        """Port CAM-SE ``prim_set_dry_mass`` for full-physics startup.

        CAM deliberately rescales startup dry surface pressure for non-simple
        physics suites.  Its analytic Held-Suarez state starts at 100000 Pa,
        then CAM-SE sets the global mean to 98288 Pa when a topography file is
        attached (101080 Pa otherwise).  The simple adiabatic, HS94, Kessler
        and TJ2016 suites explicitly skip this operation.
        """

        if ctx.config.physics_suite in {
            "adiabatic",
            "held_suarez_1994",
            "kessler",
            "tj2016",
            "grayrad",
        }:
            return

        pool = ctx.pool
        ps = pool.get("surface_pressure")
        initial = ps[..., 0]
        local_min = np.min(initial).item()
        local_max = np.max(initial).item()
        extrema = tuple(ctx.comm.allgather((local_min, local_max)))
        global_min = min(item[0] for item in extrema)
        global_max = max(item[1] for item in extrema)
        if global_min != global_max:
            raise ConfigurationError(
                "full-physics startup dry-mass scaling currently requires "
                "a spatially uniform analytic surface pressure"
            )

        # prim_set_dry_mass does not divide by the mathematically exact
        # uniform value.  It first calls SE global_integral, whose
        # element-local loop and fixed-point reproducible reduction can place
        # the result a few ulps away from that value.  Those ulps enter the
        # pressure scaling and then the mass-weighted GLL-to-PG3 temperature
        # map, so reproduce the source reduction here instead of using
        # global_min.
        from .phases import _fixed_point_reproducible_sums

        mass = pool.get("spectral_mass_matrix")
        local_integrals = np.empty(
            (pool.dimensions["nelem_local"], 1),
            dtype=np.float64,
            order="F",
        )
        for element in range(pool.dimensions["nelem_local"]):
            value = np.float64(0.0)
            for j in range(pool.dimensions["np"]):
                for i in range(pool.dimensions["np"]):
                    value = np.float64(
                        value
                        + np.float64(
                            mass[i, j, element]
                            * initial[i, j, element]
                        )
                    )
            local_integrals[element, 0] = value
        gathered = tuple(ctx.comm.allgather(local_integrals))
        global_sum = _fixed_point_reproducible_sums(gathered)[0]
        global_average = np.float64(
            global_sum
            / np.float64(
                np.float64(4.0) * pool.get("circle_constant").item()
            )
        )

        topo_file = ctx.atm["namelist"].get(
            "cam_initfiles_nl", {}
        ).get("bnd_topo")
        has_topography = bool(
            topo_file and str(topo_file).upper() != "UNSET_PATH"
        )
        target = np.float64(98288.0 if has_topography else 101080.0)
        scale = np.float64(target / global_average)
        ps[...] = ps * scale

        ps0 = np.float64(pool.get("reference_pressure"))
        hyai = pool.get("hybrid_a_interface")
        hybi = pool.get("hybrid_b_interface")
        dp = pool.get("layer_pressure_thickness")
        q = pool.get("constituent_mixing_ratio")
        for time_level in range(pool.dimensions["ntime"]):
            for level in range(pool.dimensions["pver"]):
                previous_dp = dp[:, :, level, :, time_level].copy(
                    order="F"
                )
                current_dp = (
                    np.float64(hyai[level + 1] - hyai[level]) * ps0
                    + np.float64(hybi[level + 1] - hybi[level])
                    * ps[:, :, :, time_level]
                )
                dp[:, :, level, :, time_level] = current_dp
                factor = previous_dp / current_dp
                for constituent in range(pool.dimensions["nconst"]):
                    values = q[
                        :, :, level, :, constituent, time_level
                    ]
                    values[...] = np.maximum(
                        np.float64(0.0), values * factor
                    )

        # qdp is conserved by prim_set_dry_mass: q is multiplied by
        # old_dp/new_dp and subsequently multiplied by new_dp.

    @staticmethod
    def _buffers(ctx):
        # All arrays were zeroed at allocation. Populate tracer aliases and reference derived state.
        from .fvm_mapping import physgrid_to_gll, synchronize_min_owned_gll
        from .phases import initialize_fvm_state, thermo_water_update
        from .scientific_data import read_physics_topography

        initialize_fvm_state(ctx.pool)
        # CAM-SE initializes ``cp_or_cv_dycore`` in ``d_p_coupling`` from
        # the actual water composition before the first CCPP group runs.
        # This is observably different from copying dry-air cp even for the
        # 1.e-12 water-vapor floor used by analytic initial conditions.
        thermo_water_update(ctx.pool, constituents_are_dry=True)
        topo_file = ctx.atm["namelist"].get(
            "cam_initfiles_nl", {}
        ).get("bnd_topo")
        if topo_file and str(topo_file).upper() != "UNSET_PATH":
            physics_phis = read_physics_topography(
                topo_file,
                global_columns=ctx.pool.get("physics_global_column"),
            )
            ctx.pool.get("physics_surface_geopotential")[...] = physics_phis
            nc = ctx.pool.dimensions["fv_nphys"]
            nelem = ctx.pool.dimensions["nelem_local"]
            local_phis = physics_phis.reshape(
                (nc, nc, nelem), order="F"
            )
            ctx.pool.get("fvm_surface_geopotential")[...] = local_phis
            mapped_phis = physgrid_to_gll(
                ctx.pool,
                ctx.comm,
                local_phis[:, :, None, None, :],
                limiter=True,
            )
            ctx.pool.get("surface_geopotential_gll")[...] = (
                synchronize_min_owned_gll(
                    ctx.pool,
                    ctx.comm,
                    mapped_phis[:, :, 0, 0, :],
                )
            )
        else:
            ctx.pool.get("physics_surface_geopotential")[...] = 0.0
            ctx.pool.get("fvm_surface_geopotential")[...] = 0.0
            ctx.pool.get("surface_geopotential_gll")[...] = 0.0
        # CAM's analytic-IC branch treats the input surface pressure as dry
        # pressure and retains the value produced by prim_set_dry_mass.  Do
        # not reconstruct it by summing dp3d here: although mathematically
        # equivalent, that changes the final low bit for full-physics suites
        # and perturbs the first SE dynamics advance.
        ctx.pool.get("surface_dry_air_pressure")[...] = ctx.pool.get(
            "surface_pressure"
        )[..., 0]
        # HOMME leaves the three FVM halo layers untouched until the first
        # explicit tracer ghost exchange.  StatePool deterministically owns
        # those locations as zeros instead of native uninitialized storage.
        # The unit-test communicator deliberately implements only metadata
        # collectives. Production workers always provide mpi4py's Allreduce.
        if hasattr(ctx.comm, "Allreduce"):
            from .dynamics import assemble_inverse_spectral_mass, initialize_vertical_pressure_velocity
            assemble_inverse_spectral_mass(ctx.pool, ctx.comm)
            initialize_vertical_pressure_velocity(ctx.pool, ctx.comm)
        ctx.pool.get("history_sample_count")[...] = 0
        InitializationPlan._initialize_ccpp_host_fields(ctx)

    @staticmethod
    def _initialize_ccpp_host_fields(ctx):
        """Initialize explicit CAM host controls selected by suite metadata."""

        pool = ctx.pool

        def set_standard(standard_name, value):
            try:
                field_name = pool.ccpp_field_name(standard_name)
            except KeyError:
                return
            target = pool.get(field_name, unsafe=True)
            if target.dtype.kind == "S" and isinstance(value, str):
                value = value.encode("utf-8")
            pool.set(field_name, value)

        def has_standard(standard_name):
            try:
                pool.ccpp_field_name(standard_name)
            except KeyError:
                return False
            return True

        # CAM's source registry is the authoritative initial-state contract
        # for component and coupler fields that do not yet have a scheme
        # producer (for example CAM7's temporary PBL height).  Apply only to
        # fields not already populated by the Python IC reader.
        registry_defaults = dict(
            _cam_registry_defaults(ctx.config.source_root, pool)
        )
        for standard_name, value in registry_defaults.items():
            try:
                field_name = pool.ccpp_field_name(standard_name)
            except KeyError:
                continue
            if not pool.is_initialized(field_name):
                set_standard(standard_name, value)

        # Values established by CAM's host-model initialization rather than a
        # numerical scheme.  They remain ordinary Python-owned StatePool data.
        communicator = (
            int(ctx.comm.py2f())
            if hasattr(ctx.comm, "py2f")
            else 0
        )
        controls = {
            "control_for_negative_constituent_warning": "off",
            "do_lagrangian_vertical_coordinate": False,
            "flag_for_dycore_energy_consistency_adjustment": True,
            "flag_for_energy_conservation_warning": False,
            "flag_for_energy_global_means_output": False,
            "flag_for_mpi_root": int(ctx.comm.rank) == 0,
            "fractional_calendar_days_on_end_of_current_timestep": (
                ctx.clock.fractional_calendar_day()
            ),
            "fractional_calendar_days_on_end_of_next_timestep": (
                ctx.clock.fractional_calendar_day(
                    offset_seconds=ctx.config.dt_seconds
                )
            ),
            "is_first_restart_timestep": False,
            "is_first_timestep": True,
            "log_output_unit": 6,
            "mpi_communicator": communicator,
            "mpi_root": 0,
            "next_calendar_day_to_perform_shortwave_radiation_for_surface_models": (
                ctx.clock.fractional_calendar_day(
                    offset_seconds=ctx.config.dt_seconds
                )
            ),
            "number_of_seconds_until_next_shortwave_radiation_timestep": 0,
            "number_of_atmosphere_columns_with_significant_energy_or_water_imbalances": 0,
            "total_energy_formula_for_physics": 0,
            "total_energy_formula_for_dycore": 1,
            "vertical_index_at_surface_adjacent_layer": ctx.config.pver,
            "vertical_index_at_surface_interface": ctx.config.pver + 1,
            "vertical_index_at_top_adjacent_layer": 1,
            "vertical_index_at_top_interface": 1,
        }
        radiation = ctx.atm["namelist"].get("radiation_nl", {})

        def radiation_steps(local_name, default):
            value = int(radiation.get(local_name, default))
            if value < 0:
                return int(
                    np.rint(
                        np.float64(-value * 3600)
                        / np.float64(ctx.config.dt_seconds)
                    )
                )
            return value

        controls.update(
            {
                "frequency_of_shortwave_radiation_calculation": (
                    radiation_steps("iradsw", -1)
                ),
                "frequency_of_longwave_radiation_calculation": (
                    radiation_steps("iradlw", -1)
                ),
                (
                    "number_of_timesteps_to_force_radiation_calculation_"
                    "after_initialization_namelist_parameter"
                ): radiation_steps("irad_always", 0),
                (
                    "use_radiation_uniform_angle_in_solar_zenith_angle_"
                    "calculation"
                ): bool(radiation.get("use_rad_uniform_angle", False)),
                (
                    "radiation_uniform_angle_in_solar_zenith_angle_"
                    "calculation"
                ): np.float64(radiation.get("rad_uniform_angle", -99.0)),
            }
        )
        # CAM's CCPP constituent object owns a distinct array, but its values
        # are populated from the same registered constituent properties.
        # Keep the storage independent while preserving the component minima
        # used by qneg and vertical diffusion.
        set_standard(
            "ccpp_constituent_minimum_values",
            np.array(
                pool.get("constituent_minimum"),
                dtype=np.float64,
                order="F",
                copy=True,
            ),
        )
        for standard_name, value in controls.items():
            set_standard(standard_name, value)

        # The pinned upstream MUSICA suite has an explicit placeholder host
        # provider for four pure inputs.  Parse those values from the original
        # CAM-SIMA source so the Python host service does not duplicate its
        # wavelength table.
        if ctx.config.physics_suite == "musica":
            musica_data = read_musica_placeholder_data(
                ctx.config.source_root,
                horizontal_dimension=pool.dimensions["nphys_local"],
                wavelength_interface_dimension=pool.dimensions[
                    "photolysis_wavelength_grid_interface_dimension"
                ],
                wavelength_section_dimension=pool.dimensions[
                    "photolysis_wavelength_grid_section_dimension"
                ],
            )
            for standard_name, value in musica_data.items():
                set_standard(standard_name, value)

        # The original CCPP namelist XML is the contract between each scheme
        # and CAM's generated atm_in.  Copy those values into Python-owned
        # fields by standard name, so adding a scheme to a suite also adds its
        # configuration without another hand-written Python initializer.
        namelist = ctx.atm["namelist"]
        for standard_name, bindings in ctx.namelist_bindings.items():
            value = None
            found = False
            for binding in bindings:
                group = namelist.get(binding.group, {})
                if binding.local_name in group:
                    value = group[binding.local_name]
                    found = True
                    break
            if not found:
                defaults = {
                    binding.default_value
                    for binding in bindings
                    if binding.default_value is not None
                }
                if len(defaults) != 1:
                    continue
                value = defaults.pop()
            if isinstance(value, str):
                project_root = Path(__file__).resolve().parents[3]
                value = (
                    value.replace("${RUNDIR}", str(ctx.run_dir))
                    .replace("${PROJECT_ROOT}", str(project_root))
                    .replace(
                        "${SOURCE_ROOT}", str(ctx.config.source_root)
                    )
                )
            try:
                field_name = pool.ccpp_field_name(standard_name)
            except KeyError:
                continue
            target = pool.get(field_name, unsafe=True)
            if target.dtype.kind == "S":
                encoded = np.asarray(value, dtype=target.dtype)
                if encoded.size == target.size and encoded.shape != target.shape:
                    encoded = encoded.reshape(target.shape, order="F")
                pool.set(field_name, encoded)
                continue
            if target.dtype.kind == "b" and isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {".true.", "true", "t", "1"}:
                    value = True
                elif normalized in {".false.", "false", "f", "0"}:
                    value = False
            elif isinstance(value, str):
                value = _registry_initial_value(value, pool)
            converted = np.asarray(value, dtype=target.dtype)
            if converted.size == target.size and converted.shape != target.shape:
                converted = converted.reshape(target.shape, order="F")
            pool.set(field_name, converted)

        # TUV-x's pinned JSON uses paths relative to CIME's staged
        # ``musica_configurations`` directory.  A Python-owned run has no CIME
        # staging phase, so generate an equivalent rank-local JSON whose data
        # paths are absolute.  Numerical chemistry remains in the original
        # Fortran library.
        if ctx.config.physics_suite == "musica":
            configuration_name = "filename_of_tuvx_configuration"
            if has_standard(configuration_name):
                field_name = pool.ccpp_field_name(configuration_name)
                raw = pool.get(field_name, unsafe=True)
                source_path = (
                    raw.tobytes()
                    .split(b"\0", 1)[0]
                    .decode("utf-8")
                    .strip()
                )
                staged_path = stage_musica_tuvx_configuration(
                    source_path,
                    run_dir=ctx.run_dir,
                    rank=int(ctx.comm.rank),
                )
                set_standard(configuration_name, str(staged_path))

        # CAM's cam-only import path derives the first-step upward longwave
        # surface flux from the registry-initialized blackbody temperature.
        # Without a mediator there is no coupler call that would otherwise
        # populate this field before RRTMGP.
        try:
            upward_longwave = pool.get_ccpp(
                "longwave_upward_radiative_flux_at_surface_from_coupler"
            )
        except KeyError:
            pass
        else:
            try:
                surface_temperature = pool.get_ccpp(
                    "blackbody_temperature_at_surface_from_coupler"
                )
            except KeyError:
                surface_temperature = np.float64(
                    registry_defaults[
                        "blackbody_temperature_at_surface_from_coupler"
                    ]
                )
            temperature_squared = np.multiply(
                surface_temperature, surface_temperature
            )
            temperature_fourth = np.multiply(
                temperature_squared, temperature_squared
            )
            np.multiply(
                np.float64(
                    pool.get_ccpp("stefan_boltzmanns_constant")
                ),
                temperature_fourth,
                out=upward_longwave,
            )
            pool.mark_initialized(
                pool.ccpp_field_name(
                    "longwave_upward_radiative_flux_at_surface_from_coupler"
                )
            )

        normalized_pressure = (
            pool.get("hybrid_a_midpoint")
            + pool.get("hybrid_b_midpoint")
        )
        reference_pressure = np.float64(pool.get("reference_pressure"))
        reference_pressure_midpoint = (
            pool.get("hybrid_a_midpoint")
            + pool.get("hybrid_b_midpoint")
        ) * reference_pressure
        reference_pressure_interface = (
            pool.get("hybrid_a_interface")
            + pool.get("hybrid_b_interface")
        ) * reference_pressure
        set_standard(
            "reference_pressure_at_interface",
            reference_pressure_interface,
        )
        set_standard(
            "reference_pressure_in_atmosphere_layer",
            reference_pressure_midpoint,
        )
        set_standard(
            "reference_pressure_in_atmosphere_layer_normalized_by_surface_reference_pressure",
            normalized_pressure,
        )
        set_standard(
            "air_pressure_at_top_of_atmosphere_model",
            reference_pressure_interface[0],
        )
        nonzero_b = np.flatnonzero(
            pool.get("hybrid_b_interface")[:-1] != np.float64(0.0)
        )
        pure_pressure_levels = (
            int(nonzero_b[0])
            if nonzero_b.size
            else ctx.config.pver + 2
        )
        set_standard(
            "number_of_pure_pressure_levels_at_top",
            pure_pressure_levels,
        )

        def pressure_limit_index(pressure, *, top):
            pressure = np.float64(pressure)
            if top:
                indices = np.flatnonzero(
                    reference_pressure_midpoint > pressure
                )
                return (
                    int(indices[0]) + 1
                    if indices.size
                    else ctx.config.pver + 1
                )
            indices = np.flatnonzero(
                reference_pressure_midpoint < pressure
            )
            return int(indices[-1]) + 1 if indices.size else 0

        reference_controls = ctx.atm["namelist"].get("ref_pres_nl", {})
        tropopause_cloud_pressure = reference_controls.get(
            "trop_cloud_top_press",
            100.0,
        )
        aerosol_top_pressure = reference_controls.get(
            "clim_modal_aero_top_press",
            1.0e-4,
        )
        molecular_top_pressure = reference_controls.get(
            "do_molec_press",
            0.1,
        )
        molecular_bottom_pressure = reference_controls.get(
            "molec_diff_bot_press",
            50.0,
        )
        gravity_wave_taper_bottom_pressure = np.float64(0.6e-2)
        do_molecular_diffusion = (
            reference_pressure_interface[0] < molecular_top_pressure
        )
        set_standard(
            "vertical_layer_index_of_troposphere_cloud_physics_top",
            pressure_limit_index(tropopause_cloud_pressure, top=True),
        )
        set_standard(
            "index_of_air_pressure_at_top_of_aerosol_model",
            pressure_limit_index(aerosol_top_pressure, top=True),
        )
        set_standard(
            "largest_model_top_pressure_that_allows_molecular_diffusion",
            molecular_top_pressure,
        )
        set_standard(
            "pressure_at_bottom_of_molecular_diffusion",
            molecular_bottom_pressure,
        )
        set_standard(
            "do_molecular_diffusion",
            do_molecular_diffusion,
        )
        set_standard(
            "vertical_layer_index_at_bottom_of_molecular_diffusion",
            (
                pressure_limit_index(
                    molecular_bottom_pressure,
                    top=False,
                )
                if do_molecular_diffusion
                else 0
            ),
        )
        set_standard(
            "largest_model_top_pressure_that_allows_tapering_gravity_wave_drag_at_model_top",
            gravity_wave_taper_bottom_pressure,
        )
        set_standard(
            "vertical_index_of_bottom_limit_for_tapering_gravity_wave_drag_at_model_top",
            pressure_limit_index(
                gravity_wave_taper_bottom_pressure,
                top=True,
            ),
        )
        set_standard(
            "sum_of_sigma_pressure_hybrid_coordinate_a_coefficient_and_sigma_pressure_hybrid_coordinate_b_coefficient",
            normalized_pressure,
        )
        kappa_standard_name = (
            "composition_dependent_ratio_of_dry_air_gas_constant_to_"
            "specific_heat_of_dry_air_at_constant_pressure"
        )
        if has_standard(kappa_standard_name):
            set_standard(
                kappa_standard_name,
                pool.get("column_dry_air_gas_constant")
                / pool.get("column_dry_air_specific_heat"),
            )
        set_standard(
            "ratio_of_water_vapor_gas_constant_to_composition_dependent_dry_air_gas_constant_minus_one",
            pool.get("virtual_temperature_coefficient"),
        )

        tropopause_file = namelist.get("tropopause_nl", {}).get(
            "tropopause_climo_file"
        )
        if tropopause_file:
            try:
                pool.ccpp_field_name(
                    "tropopause_air_pressure_from_tropopause_climatology_dataset"
                )
            except KeyError:
                pass
            else:
                pressure, calendar_days = read_tropopause_climatology(
                    tropopause_file,
                    target_longitude=pool.get("physics_longitude"),
                    target_latitude=pool.get("physics_latitude"),
                )
                set_standard(
                    "tropopause_air_pressure_from_tropopause_climatology_dataset",
                    pressure,
                )
                set_standard(
                    "tropopause_calendar_days_from_tropopause_climatology",
                    calendar_days,
                )

        ridge_file = namelist.get("cam_initfiles_nl", {}).get("bnd_topo")
        if ridge_file and str(ridge_file).upper() != "UNSET_PATH":
            try:
                pool.ccpp_field_name(
                    "grid_box_area_for_beta_ridge_gravity_wave_drag"
                )
            except KeyError:
                pass
            else:
                ridge_fields = read_ridge_gravity_wave_data(
                    ridge_file,
                    global_columns=pool.get("physics_global_column"),
                    earth_radius=float(pool.get("earth_radius")),
                    ridge_count=pool.dimensions[
                        "number_of_ridges_in_ridge_gravity_wave_drag"
                    ],
                )
                for standard_name, values in ridge_fields.items():
                    set_standard(standard_name, values)

        # CAM allocates registry work arrays with initialization enabled.
        # Only metadata-declared inout fields that still lack a producer are
        # zero-initialized here; pure inputs continue to fail closed.
        for contract in ctx.generated_contracts:
            if (
                contract.intent == "inout"
                and not pool.is_initialized(contract.standard_name)
            ):
                pool.set(
                    contract.standard_name,
                    np.zeros_like(
                        pool.get(contract.standard_name, unsafe=True)
                    ),
                )

    @staticmethod
    def _validate(ctx):
        pool = ctx.pool
        pool.validate(finite=True)
        if not np.array_equal(np.sort(pool.get("global_element_id")), np.unique(pool.get("global_element_id"))):
            raise ValidationError("rank owns duplicate elements")
        if np.any(pool.get("physics_layer_pressure_thickness") <= 0.0):
            raise ValidationError("initial state contains non-positive layer pressure thickness")
        normalized_constituents = tuple(
            name.strip().lower()
            for name in ctx.config.constituent_names
        )
        if "water_vapor" in normalized_constituents:
            if np.any(
                pool.get("physics_water_vapor") < 0.0
            ):
                raise ValidationError(
                    "initial water vapor must be nonnegative"
                )
            # CAM analytic dry initial conditions intentionally begin with
            # exactly zero water, even when the registered constituent
            # minimum is positive.  The initial before-coupler qneg call
            # applies that registered floor before the first history sample.
        inventory = ctx.comm.allgather(pool.get("global_element_id").tolist())
        if len(inventory) == ctx.comm.size:
            flat = sorted(
                item for rank_items in inventory for item in rank_items
            )
            expected = list(range(1, 6 * ctx.config.ne * ctx.config.ne + 1))
            if flat != expected:
                raise ValidationError(
                    f"{ctx.comm.size}-rank element ownership does not cover "
                    "each global element once"
                )
        pool.seal_static()
