from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    PICAMHistoryVariable,
    RecordingCAMBackend,
)
from freecam.pi_cam.errors import PICAMConfigurationError, PICAMStateError
from freecam.pi_cam.history_output import _elapsed_days

netCDF4 = pytest.importorskip("netCDF4")


PVER = 30
PCOLS = 4
CHUNKS = 2
NCOLS = (3, 2)
TOTAL_COLUMNS = sum(NCOLS)


def _driver(tmp_path: Path, *, rank: int = 0, size: int = 1) -> PICAMDriver:
    config = PICAMConfig(
        case_name="unit-history",
        source_root=Path("/tmp/source"),
        mpi_size=size,
        stop_n=2,
        pver=PVER,
        pcols=PCOLS,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "atm_in").write_text("&atm_in\n/\n")
    boundary = InMemoryBoundaryProvider(
        {
            (step, rank): {"sst": np.full((2,), 280.0 + step)}
            for step in range(4)
        }
    )
    driver = PICAMDriver(
        config,
        boundary,
        RecordingCAMBackend(),
        rank=rank,
        size=size,
        run_dir=run_dir,
    )
    pool = driver.pool
    pool.create_from_array(
        "phys_state.cid",
        np.asarray([[1, 4], [2, 5], [3, 0], [0, 0]], dtype=np.float64, order="F"),
    )
    pool.create_from_array(
        "phys_state.ngrdcol", np.asarray(NCOLS, dtype=np.float64, order="F")
    )
    return driver


def _column_field(driver: PICAMDriver, name: str) -> np.ndarray:
    values = driver.pool.create_from_array(
        name, np.zeros((PCOLS, CHUNKS), dtype=np.float64, order="F")
    )
    # Give each valid column its global column id so ordering is verifiable.
    values[0, 0], values[1, 0], values[2, 0] = 1.0, 2.0, 3.0
    values[0, 1], values[1, 1] = 4.0, 5.0
    return values


def _set_column_field(values: np.ndarray, factor: float) -> None:
    """Rewrite the valid columns in place, keeping their global-id pattern."""

    values[0, 0], values[1, 0], values[2, 0] = 1.0 * factor, 2.0 * factor, 3.0 * factor
    values[0, 1], values[1, 1] = 4.0 * factor, 5.0 * factor


def _level_field(driver: PICAMDriver, name: str) -> np.ndarray:
    values = driver.pool.create_from_array(
        name, np.zeros((PCOLS, PVER, CHUNKS), dtype=np.float64, order="F")
    )
    for column, identifier in ((0, 1.0), (1, 2.0), (2, 3.0)):
        values[column, :, 0] = identifier
    for column, identifier in ((0, 4.0), (1, 5.0)):
        values[column, :, 1] = identifier
    return values


def _advance_clock(driver: PICAMDriver, steps: int) -> None:
    for _ in range(steps):
        driver.clock.advance()


def test_elapsed_days_follows_the_no_leap_calendar() -> None:
    assert _elapsed_days(1, 1, 1, 0) == 0.0
    assert _elapsed_days(1, 2, 1, 0) == 31.0
    assert _elapsed_days(2, 1, 1, 0) == 365.0
    assert _elapsed_days(1, 1, 1, 43200) == 0.5
    # The supervisor's reference November mean spans days 6874 to 6904.
    assert _elapsed_days(19, 11, 1, 0) == 6874.0
    assert _elapsed_days(19, 12, 1, 0) == 6904.0


def test_monthly_mean_matches_the_original_cam_layout(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    values = _column_field(driver, "diag.rain")
    stream = driver.history_streams.install(
        "python_history",
        fields=[
            PICAMHistoryVariable(field="diag.rain", units="mm", long_name="Rain")
        ],
    )

    # One NO_LEAP January of half-hour steps, with a changing field so the
    # written sample must be a real time mean rather than a snapshot.
    for step in range(31 * 48):
        _set_column_field(values, 1.0 if step == 0 else 3.0)
        assert stream.step() is None
        driver.clock.advance()
    path = stream.step()

    assert path is not None and path.is_file()
    # CAM stamps a monthly mean with the month it covers, not the next month.
    assert path.name == "unit-history.cam.h9.0001-01.nc"
    with netCDF4.Dataset(path) as dataset:
        assert dataset.dimensions["ncol"].size == TOTAL_COLUMNS
        assert dataset.dimensions["time"].isunlimited()
        assert dataset.dimensions["time"].size == 1
        assert dataset.dimensions["lev"].size == PVER
        assert dataset.dimensions["ilev"].size == PVER + 1
        assert dataset.dimensions["nbnd"].size == 2
        assert dataset.dimensions["chars"].size == 8
        assert dataset.getncattr("Conventions") == "CF-1.0"
        assert dataset.getncattr("source") == "CAM"
        assert dataset.getncattr("case") == "unit-history"
        time = dataset.variables["time"]
        assert time.units == "days since 0001-01-01 00:00:00"
        assert time.calendar == "noleap"
        assert time.bounds == "time_bnds"
        # January mean: window [0, 31), stamped at its end like the original.
        assert float(time[0]) == pytest.approx(31.0)
        assert np.allclose(dataset.variables["time_bnds"][0, :], [0.0, 31.0])
        assert int(dataset.variables["date"][0]) == 10201
        assert int(dataset.variables["datesec"][0]) == 0
        assert int(dataset.variables["ndcur"][0]) == 31
        assert int(dataset.variables["nscur"][0]) == 0
        assert int(dataset.variables["nsteph"][0]) == 31 * 48
        assert b"".join(dataset.variables["date_written"][0, :]).strip()
        rain = dataset.variables["rain"]
        assert rain.dimensions == ("time", "ncol")
        assert rain.dtype == np.float32
        assert rain.units == "mm"
        assert rain.long_name == "Rain"
        assert rain.cell_methods == "time: mean"
        # 1487 steps at 3x and one step at 1x, averaged over 1488 samples.
        expected = np.asarray([1, 2, 3, 4, 5], dtype=np.float64)
        expected = expected * (1.0 + 3.0 * (31 * 48 - 1)) / (31 * 48)
        # Columns land at their CAM global column id, not rank-local order.
        assert np.allclose(rain[0, :], expected.astype("f4"), rtol=1e-6)


def test_step_frequency_writes_instantaneous_samples(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    _column_field(driver, "diag.rain")
    stream = driver.history_streams.install(
        "python_history",
        fields=["diag.rain"],
        nhtfrq=2,
        mfilt=2,
        time_period="instantaneous",
    )

    first = None
    for _ in range(4):
        written = stream.step()
        if written is not None:
            first = first or written
        driver.clock.advance()

    assert first is not None
    # Two half-hour steps accumulate, so the sample is stamped at 1800 s.
    assert first.name == "unit-history.cam.h9.0001-01-01-01800.nc"
    with netCDF4.Dataset(first) as dataset:
        assert dataset.dimensions["time"].size == 2
        rain = dataset.variables["rain"]
        assert rain.cell_methods == "time: instantaneous"
        # An instantaneous sample carries a zero-width interval.
        assert np.allclose(
            dataset.variables["time_bnds"][0, :], [1.0 / 48, 1.0 / 48]
        )
        assert np.allclose(rain[0, :], np.asarray([1, 2, 3, 4, 5], dtype="f4"))


def test_hourly_frequency_uses_negative_nhtfrq(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    _column_field(driver, "diag.rain")
    stream = driver.history_streams.install(
        "python_history", fields=["diag.rain"], nhtfrq=-3
    )

    written: list[Path] = []
    for _ in range(13):
        sample = stream.step()
        if sample is not None:
            written.append(sample)
        driver.clock.advance()

    # Half-hour steps: a three-hour window closes on the seventh sample, and
    # the next window opens where the previous one closed.
    assert [item.name for item in written] == [
        "unit-history.cam.h9.0001-01-01-10800.nc",
        "unit-history.cam.h9.0001-01-01-21600.nc",
    ]
    assert stream.interval_start == (1, 1, 1, 21600)
    # Consecutive windows abut, exactly like the original monthly means.
    with netCDF4.Dataset(written[0]) as dataset:
        assert np.allclose(dataset.variables["time_bnds"][0, :], [0.0, 0.125])
    with netCDF4.Dataset(written[1]) as dataset:
        assert np.allclose(dataset.variables["time_bnds"][0, :], [0.125, 0.25])


def test_level_field_uses_the_lev_dimension(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    _level_field(driver, "diag.heating")
    stream = driver.history_streams.install(
        "levels", fields=["diag.heating"], stream="h8", nhtfrq=1
    )

    driver.clock.advance()
    path = stream.step()

    with netCDF4.Dataset(path) as dataset:
        heating = dataset.variables["heating"]
        assert heating.dimensions == ("time", "lev", "ncol")
        assert heating.mdims == 1
        assert np.allclose(
            heating[0, 0, :], np.asarray([1, 2, 3, 4, 5], dtype="f4")
        )


def test_mfilt_opens_a_new_file_when_the_current_one_is_full(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    _column_field(driver, "diag.rain")
    stream = driver.history_streams.install(
        "python_history", fields=["diag.rain"], nhtfrq=1, mfilt=2
    )

    written: list[Path] = []
    for _ in range(3):
        driver.clock.advance()
        written.append(stream.step())

    assert written[0] == written[1]
    assert written[2] != written[0]
    with netCDF4.Dataset(written[0]) as dataset:
        assert dataset.dimensions["time"].size == 2
    with netCDF4.Dataset(written[2]) as dataset:
        assert dataset.dimensions["time"].size == 1


def test_static_grid_metadata_comes_from_the_runs_cam_output(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    _column_field(driver, "diag.rain")
    template = Path(driver.run_dir) / "unit-history.cam.h0.0001-01.nc"
    with netCDF4.Dataset(template, "w") as dataset:
        dataset.createDimension("ncol", TOTAL_COLUMNS)
        dataset.createDimension("lev", PVER)
        dataset.setncatts({"ne": 16, "np": 4, "title": "oracle"})
        latitude = dataset.createVariable("lat", "f8", ("ncol",))
        latitude.units = "degrees_north"
        latitude[:] = np.linspace(-45.0, 45.0, TOTAL_COLUMNS)
        area = dataset.createVariable("area", "f8", ("ncol",))
        area[:] = np.full(TOTAL_COLUMNS, 0.25)
        coefficient = dataset.createVariable("hyam", "f8", ("lev",))
        coefficient[:] = np.linspace(0.0, 1.0, PVER)

    stream = driver.history_streams.install(
        "python_history", fields=["diag.rain"], nhtfrq=1
    )
    driver.clock.advance()
    path = stream.step()

    with netCDF4.Dataset(path) as dataset:
        assert int(dataset.getncattr("ne")) == 16
        assert int(dataset.getncattr("np")) == 4
        assert dataset.getncattr("title") == "oracle"
        assert dataset.variables["lat"].units == "degrees_north"
        assert np.allclose(
            dataset.variables["lat"][:], np.linspace(-45.0, 45.0, TOTAL_COLUMNS)
        )
        assert np.allclose(dataset.variables["area"][:], 0.25)
        assert np.allclose(
            dataset.variables["hyam"][:], np.linspace(0.0, 1.0, PVER)
        )


def test_history_stream_installation_fails_closed(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    _column_field(driver, "diag.rain")

    with pytest.raises(PICAMStateError, match="unknown field"):
        driver.history_streams.install("bad", fields=["diag.missing"])
    with pytest.raises(PICAMConfigurationError, match="at least one field"):
        driver.history_streams.install("empty", fields=[])
    with pytest.raises(PICAMConfigurationError, match="mfilt"):
        driver.history_streams.install("zero", fields=["diag.rain"], mfilt=0)
    with pytest.raises(PICAMConfigurationError, match="float32 or float64"):
        driver.history_streams.install(
            "precision", fields=["diag.rain"], precision="float16"
        )
    with pytest.raises(PICAMConfigurationError, match="mean.*instantaneous"):
        driver.history_streams.install(
            "period", fields=["diag.rain"], time_period="maximum"
        )

    driver.history_streams.install("python_history", fields=["diag.rain"])
    with pytest.raises(PICAMConfigurationError, match="already installed"):
        driver.history_streams.install("python_history", fields=["diag.rain"])
    with pytest.raises(PICAMConfigurationError, match="identifier"):
        driver.history_streams.install("other", fields=["diag.rain"], stream="h9")
    with pytest.raises(PICAMConfigurationError, match="not installed"):
        driver.history_streams.stream("absent")


def test_flush_without_accumulated_samples_fails_closed(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    _column_field(driver, "diag.rain")
    stream = driver.history_streams.install("python_history", fields=["diag.rain"])

    with pytest.raises(PICAMStateError, match="nothing accumulated"):
        stream.flush()


def test_unsupported_field_rank_and_missing_grid_fail_closed(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    driver.pool.create_from_array(
        "diag.scalar", np.zeros((PCOLS,), dtype=np.float64, order="F")
    )
    scalar = driver.history_streams.install("scalar", fields=["diag.scalar"])
    with pytest.raises(PICAMStateError, match="dimensions"):
        scalar.step()

    _column_field(driver, "diag.rain")
    rain = driver.history_streams.install(
        "rain", fields=["diag.rain"], stream="h8"
    )
    driver.pool.remove("phys_state.cid")
    with pytest.raises(PICAMStateError, match="phys_state.cid"):
        rain.step()


def test_history_stream_joins_the_workflow_and_runs_with_each_step(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    _column_field(driver, "diag.rain")

    action = driver.install_history_stream(
        "python_history", fields=["diag.rain"], nhtfrq=1, after="wshist"
    )

    assert action.kind == "python_history"
    names = [item["name"] for item in driver.step_plan.describe()]
    assert names.index("python_history") == names.index("history") + 1

    rows = driver.step()
    assert any(row.operation == "python_history" for row in rows)
    stream = driver.history_streams.stream("python_history")
    assert stream.writes == 1
    assert len(tuple(Path(driver.run_dir).glob("*.cam.h9.*.nc"))) == 1

    driver.remove_history_stream("python_history")
    assert "python_history" not in driver.history_streams
    assert all(
        item["name"] != "python_history" for item in driver.step_plan.describe()
    )


def test_describe_reports_installed_state(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    _column_field(driver, "diag.rain")
    stream = driver.history_streams.install(
        "python_history",
        fields=[{"field": "diag.rain", "name": "RAIN", "units": "mm"}],
        nhtfrq=-6,
        mfilt=4,
    )
    stream.accumulate()

    described = driver.history_streams.describe()
    assert len(described) == 1
    assert described[0]["name"] == "python_history"
    assert described[0]["stream"] == "h9"
    assert described[0]["nhtfrq"] == -6
    assert described[0]["mfilt"] == 4
    assert described[0]["time_period"] == "mean"
    assert described[0]["writes"] == 0
    assert described[0]["accumulated"] == 1
    assert described[0]["variables"] == [
        {"field": "diag.rain", "name": "RAIN", "units": "mm", "long_name": None}
    ]
