from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pycam_sima.model.clock import ModelClock
from pycam_sima.model.config import ModelConfig
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
