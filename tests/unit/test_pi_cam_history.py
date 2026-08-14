from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from freecam.pi_cam.history import PICAMOutputView


class _Driver:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir


def _write_output(path: Path, value: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("lev", 2)
        time = dataset.createVariable("time", "f8", ("time",))
        temperature = dataset.createVariable("T", "f8", ("time", "lev"))
        time[:] = (value,)
        temperature[:] = ((value, value + 1.0),)


def _write_spatial_output(path: Path, step: int, offset: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("lev", 2)
        dataset.createDimension("ncol", 2)
        steps = dataset.createVariable("nsteph", "i4", ("time",))
        area = dataset.createVariable("area", "f8", ("ncol",))
        temperature = dataset.createVariable(
            "T", "f8", ("time", "lev", "ncol")
        )
        temperature.units = "K"
        temperature.long_name = "Temperature"
        steps[:] = (step,)
        area[:] = (1.0, 3.0)
        temperature[:] = (
            (
                (1.0 + offset, 3.0 + offset),
                (5.0 + offset, 7.0 + offset),
            ),
        )


def test_output_view_discovers_streams_and_opens_history_with_xarray(
    tmp_path: Path,
) -> None:
    first = tmp_path / "case.cam.h0.0001-01-01-00000.nc"
    second = tmp_path / "case.cam.h0.0001-01-01-01800.nc"
    auxiliary = tmp_path / "case.cam.h1.0001-01-01-01800.nc"
    restart = tmp_path / "case.cam.r.0001-01-01-01800.nc"
    for path, value in (
        (first, 0.0),
        (second, 1.0),
        (auxiliary, 2.0),
        (restart, 3.0),
    ):
        _write_output(path, value)

    driver = _Driver(tmp_path)
    history = PICAMOutputView(driver, "history")
    restarts = PICAMOutputView(driver, "restart")

    assert tuple(history.streams) == ("h0", "h1")
    assert history.latest("h0") == second
    assert restarts.files == (restart,)
    with history.open("h0") as dataset:
        assert dataset.sizes == {"time": 2, "lev": 2}
        assert np.array_equal(dataset["T"].values, ((0.0, 1.0), (1.0, 2.0)))
    with restarts.open() as dataset:
        assert np.array_equal(dataset["T"].values, ((3.0, 4.0),))


def test_output_view_reports_empty_run_directory(tmp_path: Path) -> None:
    view = PICAMOutputView(_Driver(tmp_path), "history")

    assert view.files == ()
    assert view.streams == {}
    try:
        view.latest()
    except FileNotFoundError as error:
        assert "no CAM history files" in str(error)
    else:
        raise AssertionError("empty history unexpectedly returned a file")


def test_output_view_builds_area_weighted_step_series(tmp_path: Path) -> None:
    _write_spatial_output(
        tmp_path / "case.cam.h0.0001-01-01-00000.nc", step=10, offset=0.0
    )
    _write_spatial_output(
        tmp_path / "case.cam.h0.0001-01-01-01800.nc", step=11, offset=2.0
    )
    history = PICAMOutputView(_Driver(tmp_path), "history")

    series = history.step_series("T", statistic="global_mean")
    lowest_level = history.step_series("T", statistic="global_mean", level=-1)

    assert np.array_equal(series.coords["model_step"], (10, 11))
    assert np.array_equal(series.values, (4.5, 6.5))
    assert np.array_equal(lowest_level.values, (6.5, 8.5))
    assert series.attrs["source_variable"] == "T"
    assert lowest_level.attrs["selected_level_index"] == -1


def test_output_view_plots_reduced_field_against_step(tmp_path: Path) -> None:
    _write_spatial_output(
        tmp_path / "case.cam.h0.0001-01-01-00000.nc", step=3, offset=0.0
    )
    _write_spatial_output(
        tmp_path / "case.cam.h0.0001-01-01-01800.nc", step=4, offset=1.0
    )
    history = PICAMOutputView(_Driver(tmp_path), "history")

    axis = history.plot_steps(
        "T", statistic="global_mean", level=-1, label="PI-CAM"
    )

    assert np.array_equal(axis.lines[0].get_xdata(), (3, 4))
    assert np.array_equal(axis.lines[0].get_ydata(), (6.5, 7.5))
    assert axis.get_xlabel() == "model step"
    assert axis.get_ylabel() == "global mean Temperature (K)"
    assert axis.get_legend().get_texts()[0].get_text() == "PI-CAM"
