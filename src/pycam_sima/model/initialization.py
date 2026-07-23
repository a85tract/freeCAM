"""Describable and pausable Python initialization plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from taskflow import engines, task
from taskflow.patterns import linear_flow

from .clock import NoLeapClock
from .config import ModelConfig
from .errors import ConfigurationError, ValidationError
from .fvm_geometry import generate_fvm_geometry
from .grid import dimensions_for_rank, populate_grid
from .initial_conditions import populate_initial_state
from .namelist import read_atm_in
from .state import StatePool
from .vertical import load_vertical_coordinate


@dataclass(slots=True)
class InitializationContext:
    config: ModelConfig
    run_dir: Path
    comm: object
    atm: dict | None = None
    pool: StatePool | None = None
    clock: NoLeapClock | None = None
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
        "register_constituents", "generate_dcmip2016_initial_state", "initialize_derived_buffers",
        "validate_python_owned_state",
    )

    def __init__(self, config: ModelConfig, run_dir: str | Path, comm):
        self.context = InitializationContext(config, Path(run_dir).resolve(), comm)
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
        ctx.atm = read_atm_in(ctx.config.resolve_atm_in(ctx.run_dir))

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
        ctx.pool = StatePool(dimensions_for_rank(ctx.comm.rank, ctx.comm.size))
        pool = ctx.pool
        pool.set("mpi_rank", ctx.comm.rank); pool.set("mpi_size", ctx.comm.size)
        pool.set("spectral_element_count", 54); pool.set("vertical_level_count", 30)
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
        load_vertical_coordinate(ctx.pool, ctx.atm["ncdata"], ctx.comm)

    @staticmethod
    def _constants(ctx):
        pool = ctx.pool
        avogad, boltzmann, mwdry, mwh2o = 6.02214e26, 1.38065e-23, 28.966, 18.016
        universal_gas_constant = np.float64(avogad) * np.float64(boltzmann)
        dry_air_gas_constant = universal_gas_constant / np.float64(mwdry)
        water_vapor_gas_constant = universal_gas_constant / np.float64(mwh2o)
        values = {
            "gravitational_acceleration": 9.80616,
            "reciprocal_gravitational_acceleration": (
                np.float64(1.0) / np.float64(9.80616)
            ),
            "dry_air_gas_constant": dry_air_gas_constant,
            "water_vapor_gas_constant": water_vapor_gas_constant,
            "virtual_temperature_coefficient": np.float64(water_vapor_gas_constant / dry_air_gas_constant) - np.float64(1.0),
            "dry_air_specific_heat": 1.00464e3,
            "earth_radius": 6.37122e6,
            "earth_angular_velocity": 2.0 * np.float64(3.14159265358979323846) / 86164.0,
            "water_to_dry_molecular_weight_ratio": mwh2o / mwdry,
            "latent_heat_of_vaporization": 2.501e6,
            "latent_heat_of_fusion": 3.337e5,
            "water_freezing_temperature": 273.15,
            "water_triple_point_temperature": 273.16,
            "universal_gas_constant": universal_gas_constant,
            "avogadro_constant": avogad,
            "boltzmann_constant": boltzmann,
            "circle_constant": np.float64(3.14159265358979323846),
            "liquid_water_density": 1000.0,
            "water_vapor_specific_heat": 1.810e3,
            "liquid_water_specific_heat": 4.188e3,
            # Fixed year-0 orbital values are explicit state, not Fortran module data.
            "orbital_eccentricity": 0.016715,
            "orbital_obliquity": np.deg2rad(23.4441),
            "orbital_longitude_of_perihelion": np.deg2rad(102.7),
        }
        for name, value in values.items(): pool.set(name, value)
        exponent = np.float64(1.0) / np.log10(np.float64(2.0))
        nu_factor = np.float64(1.0e15) / np.float64(np.float64(110000.0) ** exponent)
        nu_p = nu_factor * np.float64((np.float64(30.0) / np.float64(3.0) * np.float64(110000.0)) ** exponent)
        pool.set("pressure_hyperviscosity", nu_p)
        pool.set("velocity_hyperviscosity", np.float64(0.5) * nu_p)
        pool.set("divergence_hyperviscosity", np.float64(2.5) * nu_p)
        pool.set("temperature_hyperviscosity", nu_p)
        pool.set("tracer_hyperviscosity", nu_p)
        pool.set("sponge_top_viscosity", np.float64(5.0e5))
        pool.set("sponge_level_count", 3)
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
                np.float64(1.0) + np.tanh(np.log(np.float64(ptop / pressure)))
            )
            sponge_scale[level] = value if value >= np.float64(0.15) else 0.0
        pool.set("sponge_viscosity_scale", sponge_scale)
        ctx.clock = NoLeapClock(dt_seconds=ctx.config.dt_seconds)
        pool.set("model_step", 0); pool.set("current_date", ctx.clock.yyyymmdd); pool.set("current_seconds_of_day", 0)
        pool.set("dynamics_time_level_nm1", 0)
        pool.set("dynamics_time_level_n0", 1)
        pool.set("dynamics_time_level_np1", 2)
        pool.set("dynamics_internal_step", 0)

    @staticmethod
    def _grid(ctx):
        populate_grid(ctx.pool, ctx.comm.rank, ctx.comm.size)
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
        ctx.pool.set("constituent_index", np.array((1, 2, 3), dtype=np.int32))
        ctx.pool.set("constituent_minimum", np.array((0.0, 0.0, QV_MIN := 1.0e-12)))
        ctx.pool.set("constituent_molecular_weight", np.array((18.016, 18.016, 18.016)))

    @staticmethod
    def _initial_state(ctx):
        populate_initial_state(ctx.pool, ctx.comm)

    @staticmethod
    def _buffers(ctx):
        # All arrays were zeroed at allocation. Populate tracer aliases and reference derived state.
        from .phases import initialize_fvm_state
        initialize_fvm_state(ctx.pool)
        ptop = np.float64(ctx.pool.get("hybrid_a_interface")[0]) * np.float64(
            ctx.pool.get("reference_pressure")
        )
        for le in range(ctx.pool.dimensions["nelem_local"]):
            for j in range(ctx.pool.dimensions["np"]):
                for i in range(ctx.pool.dimensions["np"]):
                    value = ptop
                    for k in range(ctx.pool.dimensions["pver"]):
                        value = np.float64(
                            value
                            + ctx.pool.get("layer_pressure_thickness")[i, j, k, le, 0]
                        )
                    ctx.pool.get("surface_dry_air_pressure")[i, j, le] = value
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
        ctx.pool.get("surface_geopotential_gll")[...] = 0.0

    @staticmethod
    def _validate(ctx):
        pool = ctx.pool
        pool.validate(finite=True)
        if not np.array_equal(np.sort(pool.get("global_element_id")), np.unique(pool.get("global_element_id"))):
            raise ValidationError("rank owns duplicate elements")
        if np.any(pool.get("physics_layer_pressure_thickness") <= 0.0):
            raise ValidationError("initial state contains non-positive layer pressure thickness")
        if np.any(pool.get("physics_water_vapor") < pool.get("constituent_minimum")[2]):
            raise ValidationError("water vapor is below its registered minimum")
        inventory = ctx.comm.allgather(pool.get("global_element_id").tolist())
        if len(inventory) == 24:
            flat = sorted(item for rank_items in inventory for item in rank_items)
            if flat != list(range(1, 55)):
                raise ValidationError("24-rank element ownership does not cover each global element once")
        pool.seal_static()
