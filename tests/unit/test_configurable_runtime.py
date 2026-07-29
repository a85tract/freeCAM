from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset
import pytest

from pycam_sima.model.clock import ModelClock
from pycam_sima.model.config import ModelConfig
from pycam_sima.model.dimension_service import infer_suite_dimensions
from pycam_sima.model.grid import dimensions_for_rank
from pycam_sima.model.history import HistoryWriter
from pycam_sima.model.initialization import InitializationPlan
from pycam_sima.model.state import StatePool


ROOT = Path(__file__).resolve().parents[2]


class _SingleRankComm:
    rank = 0
    size = 1

    @staticmethod
    def bcast(value, root=0):
        del root
        return value

    @staticmethod
    def allgather(value):
        return [value]

    @staticmethod
    def gather(value, root=0):
        del root
        return [value]


@pytest.mark.parametrize(
    ("filename", "suite"),
    (
        ("adiabatic_model.yaml", "adiabatic"),
        ("held_suarez_1994_model.yaml", "held_suarez_1994"),
        ("tj2016_model.yaml", "tj2016"),
    ),
)
def test_idealized_oracle_profiles_are_generic_model_configs(
    filename: str,
    suite: str,
) -> None:
    config = ModelConfig.from_yaml(ROOT / "configs" / filename)

    assert config.physics_suite == suite
    expected_initial_state = (
        "moist_baroclinic_wave_dcmip2016"
        if suite == "tj2016"
        else "held_suarez_1994"
    )
    assert config.analytic_ic_type == expected_initial_state
    assert config.constituent_names == ("water_vapor",)
    assert config.stop_n == 50
    assert config.mpi_size == 24


def test_suite_specific_dimensions_remain_runtime_configuration() -> None:
    config = ModelConfig.from_mapping(
        {
            **ModelConfig().as_dict(),
            "dimension_overrides": {
                "number_of_bands_for_longwave_radiation": 16,
                "number_of_bands_for_shortwave_radiation": 14,
            },
        }
    )
    dimensions = dimensions_for_rank(
        0,
        config.mpi_size,
        pver=config.pver,
        np_value=config.np,
        fv_nphys=config.fv_nphys,
        constituent_count=config.constituent_count,
        ne=config.ne,
        extra_dimensions=config.dimension_overrides,
    )

    assert dimensions["number_of_bands_for_longwave_radiation"] == 16
    assert dimensions["number_of_bands_for_shortwave_radiation"] == 14


def test_suite_dimensions_are_inferred_from_namelist_files_and_source(
    tmp_path: Path,
) -> None:
    solar = tmp_path / "solar.nc"
    lw = tmp_path / "lw.nc"
    sw = tmp_path / "sw.nc"
    tropopause = tmp_path / "tropopause.nc"
    for path, dimension, extent in (
        (solar, "wlen", 17),
        (lw, "gpt", 11),
        (sw, "gpt", 13),
        (tropopause, "time", 12),
    ):
        with Dataset(path, "w") as dataset:
            dataset.createDimension(dimension, extent)
    source = tmp_path / "source"
    utilities = source / "src/physics/utils"
    utilities.mkdir(parents=True)
    (utilities / "gravity_wave_drag_ridge_read.F90").write_text(
        "integer, parameter :: prdg = 16\n"
    )
    (utilities / "musica_ccpp_dependencies.F90").write_text(
        "integer :: photolysis_wavelength_grid_section_dimension = 102\n"
        "integer :: photolysis_wavelength_grid_interface_dimension = 103\n"
    )
    namelist = {
        "rrtmgp": {
            "nradgas": 8,
            "nlwbands": 16,
            "nswbands": 14,
        },
        "rrtmgp_constituents": {"ndiags": 1},
        "rrtmgp_lw_gas_optics": {
            "rrtmgp_coefs_lw_file": str(lw)
        },
        "rrtmgp_sw_gas_optics": {
            "rrtmgp_coefs_sw_file": str(sw)
        },
        "solar_data": {"solar_irrad_data_file": str(solar)},
        "tropopause_nl": {
            "tropopause_climo_file": str(tropopause)
        },
    }
    binding = type("Binding", (), {})

    def bound(group: str, local_name: str):
        value = binding()
        value.group = group
        value.local_name = local_name
        return (value,)

    bindings = {
        "number_of_active_gases_for_rrtmgp": bound(
            "rrtmgp", "nradgas"
        ),
        "number_of_bands_for_longwave_radiation": bound(
            "rrtmgp", "nlwbands"
        ),
        "number_of_bands_for_shortwave_radiation": bound(
            "rrtmgp", "nswbands"
        ),
        "number_of_diagnostic_subcycles": bound(
            "rrtmgp_constituents", "ndiags"
        ),
    }
    required = {
        *bindings,
        "daytime_columns_dimension",
        "number_of_vertical_layers_in_rrtmgp",
        "number_of_vertical_interfaces_in_rrtmgp",
        "number_of_longwave_g_point_intervals",
        "number_of_shortwave_g_point_intervals",
        "number_of_wavelength_samples_of_spectrum",
        "number_of_wavelength_samples_of_spectrum_plus_one",
        "number_of_time_slices_in_tropopause_climatology_dataset",
        "number_of_ridges_in_ridge_gravity_wave_drag",
        "photolysis_wavelength_grid_section_dimension",
        "photolysis_wavelength_grid_interface_dimension",
    }

    inferred = infer_suite_dimensions(
        required=required,
        existing={"nphys_local": 27, "pver": 30, "pverp": 31},
        namelist=namelist,
        namelist_bindings=bindings,
        source_root=source,
    )

    assert inferred == {
        "daytime_columns_dimension": 27,
        "number_of_active_gases_for_rrtmgp": 8,
        "number_of_bands_for_longwave_radiation": 16,
        "number_of_bands_for_shortwave_radiation": 14,
        "number_of_diagnostic_subcycles": 1,
        "number_of_longwave_g_point_intervals": 11,
        "number_of_ridges_in_ridge_gravity_wave_drag": 16,
        "number_of_shortwave_g_point_intervals": 13,
        "number_of_time_slices_in_tropopause_climatology_dataset": 12,
        "number_of_vertical_interfaces_in_rrtmgp": 32,
        "number_of_vertical_layers_in_rrtmgp": 31,
        "number_of_wavelength_samples_of_spectrum": 17,
        "number_of_wavelength_samples_of_spectrum_plus_one": 18,
        "photolysis_wavelength_grid_interface_dimension": 103,
        "photolysis_wavelength_grid_section_dimension": 102,
    }


def _write_vertical_coordinate(path: Path, level_count: int) -> None:
    hyai = np.zeros(level_count + 1)
    hyai[0] = 0.01
    hybi = np.linspace(0.0, 1.0, level_count + 1)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("lev", level_count)
        dataset.createDimension("ilev", level_count + 1)
        dataset.createVariable("hyai", "f8", ("ilev",))[:] = hyai
        dataset.createVariable("hybi", "f8", ("ilev",))[:] = hybi
        dataset.createVariable("hyam", "f8", ("lev",))[:] = (
            hyai[:-1] + hyai[1:]
        ) / 2.0
        dataset.createVariable("hybm", "f8", ("lev",))[:] = (
            hybi[:-1] + hybi[1:]
        ) / 2.0
        dataset.createVariable("P0", "f8").assignValue(100000.0)


def test_calendar_progression_is_not_locked_to_no_leap() -> None:
    gregorian = ModelClock(
        year=2000,
        month=2,
        day=28,
        seconds=23 * 3600,
        dt_seconds=3600,
        calendar="GREGORIAN",
    )
    gregorian.advance()
    assert (gregorian.year, gregorian.month, gregorian.day) == (2000, 2, 29)

    model_360 = ModelClock(
        year=2001,
        month=2,
        day=30,
        seconds=23 * 3600,
        dt_seconds=3600,
        calendar="360_DAY",
    )
    model_360.advance()
    assert (model_360.year, model_360.month, model_360.day) == (2001, 3, 1)


def test_fractional_calendar_day_matches_esmf_operation_order() -> None:
    clock = ModelClock(
        year=1,
        month=1,
        day=2,
        seconds=0,
        dt_seconds=1800,
        calendar="NO_LEAP",
    )

    assert np.float64(clock.fractional_calendar_day(1800)).view(
        np.uint64
    ) == np.uint64(0x40002AAAAAAAAAAA)
    assert np.float64(clock.fractional_calendar_day(3600)).view(
        np.uint64
    ) == np.uint64(0x4000555555555556)


def test_full_python_initialization_uses_nonreference_configuration(
    tmp_path: Path,
) -> None:
    vertical = tmp_path / "vertical_l12.nc"
    _write_vertical_coordinate(vertical, 12)
    (tmp_path / "atm_in").write_text(
        "\n".join(
            (
                "&analytic_ic_nl "
                "analytic_ic_type='resting_isothermal' /",
                "&cam_initfiles_nl "
                f"pertlim=0.0, ncdata='{vertical}' /",
                "&dyn_se_nl se_ne=2, se_fv_nphys=4 /",
                "&physics_nl physics_suite='kessler' /",
                "&vert_coord_nl pver=12 /",
            )
        )
        + "\n"
    )
    config = ModelConfig.from_yaml(
        ROOT / "configs/configurable_ne2np5_pg4_l12.yaml"
    ).with_overrides(mpi_size=1, atm_in="atm_in")

    context = InitializationPlan(
        config,
        tmp_path,
        _SingleRankComm(),
    ).run()

    assert context.clock.calendar == "360_DAY"
    assert context.clock.iso_stamp == "2000-02-30-00000"
    assert context.pool.dimensions["np"] == 5
    assert context.pool.dimensions["fv_nphys"] == 4
    assert context.pool.dimensions["pver"] == 12
    assert context.pool.dimensions["nconst"] == 5
    assert context.pool.get("air_temperature").shape == (5, 5, 12, 24, 3)
    assert context.pool.get("physics_air_temperature").shape == (384, 12)
    assert np.all(context.pool.get("air_temperature") == 300.0)


def test_history_inventory_follows_a_reduced_constituent_pool(
    tmp_path: Path,
) -> None:
    config = ModelConfig(
        grid="ne1np3.pg2",
        ne=1,
        np=3,
        fv_nphys=2,
        pver=4,
        constituent_count=1,
        mpi_size=1,
        analytic_ic_type="resting_isothermal",
        constituent_names=("cloud_liquid_water",),
        constituent_minima=(0.0,),
        constituent_molecular_weights=(18.016,),
        case_name="one_constituent",
    )
    config.validate()
    pool = StatePool(
        dimensions_for_rank(
            0,
            1,
            ne=1,
            np_value=3,
            fv_nphys=2,
            pver=4,
            constituent_count=1,
        )
    )
    pool.get("physics_global_column")[...] = np.arange(
        pool.dimensions["nphys_local"]
    ).reshape(pool.get("physics_global_column").shape, order="F")
    writer = HistoryWriter(
        tmp_path,
        config.case_name,
        _SingleRankComm(),
        config=config,
    )
    path = writer.write(
        pool,
        ModelClock(dt_seconds=60, calendar="GREGORIAN"),
    )

    assert path is not None
    with Dataset(path) as dataset:
        assert "CLDLIQ" in dataset.variables
        assert "RAINQM" not in dataset.variables
        assert "Q" not in dataset.variables
        assert dataset.getncattr("constituent_count") == 1
