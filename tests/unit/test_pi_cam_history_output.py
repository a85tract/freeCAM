"""Python-owned fields reach CAM's own history files, not a separate stream."""

from pathlib import Path
import warnings

import numpy as np
import pytest

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    PICAMVariableSpec,
    RecordingCAMBackend,
)
from freecam.pi_cam.state import PICAMFieldContract
from freecam.pi_cam.errors import PICAMConfigurationError, PICAMStateError
from freecam.pi_cam.history_output import elapsed_days

netCDF4 = pytest.importorskip("netCDF4")

PVER = 30
PCOLS = 4
CHUNKS = 2
NCOLS = (3, 2)
TOTAL_COLUMNS = sum(NCOLS)


def _driver(
    tmp_path: Path, *, nhtfrq: int = 0, backend: object | None = None, **kwargs
) -> PICAMDriver:
    config = PICAMConfig(
        case_name="unit-history",
        source_root=Path("/tmp/source"),
        mpi_size=1,
        stop_n=2,
        pver=PVER,
        pcols=PCOLS,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "atm_in").write_text(f"&atm_in\n mfilt = 1\n nhtfrq = {nhtfrq}\n/\n")
    boundary = InMemoryBoundaryProvider(
        {(step, 0): {"sst": np.full((2,), 280.0 + step)} for step in range(6)}
    )
    driver = PICAMDriver(
        config,
        boundary,
        backend or RecordingCAMBackend(),
        rank=0,
        size=1,
        run_dir=run_dir,
        **kwargs,
    )
    pool = driver.pool
    pool.dimensions.setdefault("chunks", CHUNKS)
    _attach_native(
        pool,
        "phys_state.cid",
        ("pcols", "chunks"),
        np.asarray([[1, 4], [2, 5], [3, 0], [0, 0]], dtype=np.float64, order="F"),
    )
    _attach_native(
        pool,
        "phys_state.ngrdcol",
        ("chunks",),
        np.asarray(NCOLS, dtype=np.float64, order="F"),
    )
    return driver


def _attach_native(pool, name, dimensions, values) -> None:
    """Attach a CAM-owned field the way the generated state bridge does."""

    pool.attach(
        PICAMFieldContract(
            name=name,
            dimensions=dimensions,
            category="native_cam_state",
            owner="native",
        ),
        values,
    )


def _python_field(driver: PICAMDriver, name: str, *, levels: bool = False,
                  output: bool = True, units: str = "1") -> np.ndarray:
    """Create a Python-owned column field the way a notebook user would."""

    dimensions = ("pcols", "pver", "chunks") if levels else ("pcols", "chunks")
    driver.pool.dimensions.setdefault("chunks", CHUNKS)
    values = driver.pool.create(
        PICAMVariableSpec(
            name=name, dimensions=dimensions, units=units, output=output
        ).contract()
    )
    if levels:
        for column, value in ((0, 1.0), (1, 2.0), (2, 3.0)):
            values[column, :, 0] = value
        for column, value in ((0, 4.0), (1, 5.0)):
            values[column, :, 1] = value
    else:
        values[0, 0], values[1, 0], values[2, 0] = 1.0, 2.0, 3.0
        values[0, 1], values[1, 1] = 4.0, 5.0
    return values


def _cam_history_file(driver: PICAMDriver, samples) -> Path:
    """Write a CAM-style h0 file the way the original writer would."""

    first = samples[0]
    path = Path(driver.run_dir) / (
        f"unit-history.cam.h0.{first[0]:04d}-{first[1]:02d}.nc"
    )
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", None)
        dataset.createDimension("ncol", TOTAL_COLUMNS)
        dataset.createDimension("lev", PVER)
        dataset.createDimension("ilev", PVER + 1)
        dataset.setncatts({"Conventions": "CF-1.0", "source": "CAM"})
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "days since 0001-01-01 00:00:00"
        time.calendar = "noleap"
        date = dataset.createVariable("date", "i4", ("time",))
        datesec = dataset.createVariable("datesec", "i4", ("time",))
        native = dataset.createVariable("T", "f4", ("time", "lev", "ncol"))
        native.units = "K"
        for index, (year, month, day, seconds) in enumerate(samples):
            time[index] = elapsed_days(year, month, day, seconds)
            date[index] = year * 10000 + month * 100 + day
            datesec[index] = seconds
            native[index, :, :] = 250.0
    return path


def test_elapsed_days_follows_the_no_leap_calendar() -> None:
    assert elapsed_days(1, 1, 1, 0) == 0.0
    assert elapsed_days(1, 2, 1, 0) == 31.0
    assert elapsed_days(2, 1, 1, 0) == 365.0
    # The production November mean spans days 6874 to 6904.
    assert elapsed_days(19, 11, 1, 0) == 6874.0
    assert elapsed_days(19, 12, 1, 0) == 6904.0


def test_a_run_without_python_fields_adds_nothing(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    path = _cam_history_file(driver, [(1, 2, 1, 0)])
    before = path.read_bytes()

    # A default stream is installed, but it has nothing to contribute.
    assert "python_fields" in driver.history_streams
    stream = driver.history_streams.stream("python_fields")
    assert stream.fields() == ()
    driver.clock.month = 2
    assert stream.step() == 0
    driver.finalize()

    assert path.read_bytes() == before
    assert sorted(p.name for p in Path(driver.run_dir).glob("*.nc")) == [path.name]


def test_python_field_lands_in_the_cam_history_file(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    values = _python_field(driver, "heating_rate", units="K s-1")
    path = _cam_history_file(driver, [(1, 2, 1, 0)])
    stream = driver.history_streams.stream("python_fields")

    assert [item.output_name for item in stream.fields()] == ["heating_rate"]
    stream.accumulate()
    values[:] = values * 3.0
    stream.accumulate()
    driver.clock.month = 2
    assert stream.step() == 1

    with netCDF4.Dataset(path) as dataset:
        # The file keeps its own inventory and gains one variable.
        assert "T" in dataset.variables
        added = dataset.variables["heating_rate"]
        assert added.dimensions == ("time", "ncol")
        assert added.units == "K s-1"
        assert added.cell_methods == "time: mean"
        # Three accumulated samples: one at 1x and two at 3x.
        expected = np.asarray([1, 2, 3, 4, 5]) * (1.0 + 3.0 + 3.0) / 3.0
        assert np.allclose(added[0, :], expected.astype("f4"), rtol=1e-6)
        assert float(dataset.variables["T"][0, 0, 0]) == 250.0


def test_level_field_uses_the_files_own_vertical_axis(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    _python_field(driver, "heating", levels=True)
    path = _cam_history_file(driver, [(1, 2, 1, 0)])
    stream = driver.history_streams.stream("python_fields")

    stream.accumulate()
    driver.clock.month = 2
    stream.step()

    with netCDF4.Dataset(path) as dataset:
        added = dataset.variables["heating"]
        assert added.dimensions == ("time", "lev", "ncol")
        assert added.mdims == 1
        assert np.allclose(added[0, 0, :], np.asarray([1, 2, 3, 4, 5], dtype="f4"))


def test_samples_match_their_own_time_index(tmp_path: Path) -> None:
    driver = _driver(tmp_path, nhtfrq=1)
    driver.initialize()
    values = _python_field(driver, "rain")
    path = _cam_history_file(driver, [(1, 1, 1, 1800), (1, 1, 1, 3600)])
    stream = driver.history_streams.stream("python_fields")

    driver.clock.advance()
    assert stream.step() == 1
    values[:] = values * 10.0
    driver.clock.advance()
    assert stream.step() == 1

    with netCDF4.Dataset(path) as dataset:
        rain = dataset.variables["rain"]
        assert np.allclose(rain[0, :], np.asarray([1, 2, 3, 4, 5], dtype="f4"))
        assert np.allclose(rain[1, :], np.asarray([10, 20, 30, 40, 50], dtype="f4"))


def test_windows_wait_until_cam_writes_their_file(tmp_path: Path) -> None:
    driver = _driver(tmp_path, nhtfrq=1)
    driver.initialize()
    _python_field(driver, "rain")
    stream = driver.history_streams.stream("python_fields")

    # CAM has not written anything yet, so the window queues instead of failing.
    driver.clock.advance()
    assert stream.step() == 0
    assert stream.pending == 1

    path = _cam_history_file(driver, [(1, 1, 1, 1800)])
    assert stream.drain() == 1
    assert stream.pending == 0
    with netCDF4.Dataset(path) as dataset:
        assert "rain" in dataset.variables


def test_a_tape_cam_never_writes_bounds_the_queue(tmp_path: Path) -> None:
    """A stream pointed at a tape the case never writes must not grow forever."""

    driver = _driver(tmp_path, nhtfrq=1)
    driver.initialize()
    _python_field(driver, "rain")
    stream = driver.history_streams.stream("python_fields")
    limit = stream.pending_limit

    for _ in range(limit):
        driver.clock.advance()
        stream.step()
    assert stream.pending == limit
    assert stream.dropped == 0

    # The next window has nowhere to go; the oldest is discarded, loudly.
    driver.clock.advance()
    with pytest.warns(RuntimeWarning, match="no 'h0' history file"):
        stream.step()
    assert stream.pending == limit
    assert stream.dropped == 1

    # Only the first eviction warns, but every loss is still counted.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        driver.clock.advance()
        stream.step()
    assert stream.pending == limit
    assert stream.dropped == 2


def test_finalize_drains_queued_windows(tmp_path: Path) -> None:
    driver = _driver(tmp_path, nhtfrq=1)
    driver.initialize()
    _python_field(driver, "rain")
    stream = driver.history_streams.stream("python_fields")
    driver.clock.advance()
    stream.step()
    assert stream.pending == 1

    path = _cam_history_file(driver, [(1, 1, 1, 1800)])
    driver.finalize()

    assert stream.pending == 0
    with netCDF4.Dataset(path) as dataset:
        assert "rain" in dataset.variables


def test_finalize_drains_the_tape_cam_closes_last(tmp_path: Path) -> None:
    """A run that stops on a history boundary still writes its last window.

    CAM owns the file it is writing until ``cam_final`` closes it, so the
    window that closed on the run's own last step can only be appended
    after the native finalization, not before it.
    """

    live: dict[str, PICAMDriver] = {}

    class _LateTapeBackend(RecordingCAMBackend):
        def finalize(self, pool, *, fcomm: int) -> None:
            _cam_history_file(live["driver"], [(1, 1, 1, 1800)])
            super().finalize(pool, fcomm=fcomm)

    driver = _driver(tmp_path, nhtfrq=1, backend=_LateTapeBackend())
    live["driver"] = driver
    driver.initialize()
    _python_field(driver, "rain")
    stream = driver.history_streams.stream("python_fields")

    driver.clock.advance()
    assert stream.step() == 0
    assert stream.pending == 1

    driver.finalize()

    assert stream.pending == 0
    path = next(Path(driver.run_dir).glob("*.cam.h0.*.nc"))
    with netCDF4.Dataset(path) as dataset:
        assert "rain" in dataset.variables


def test_finalize_writes_a_window_cam_has_already_sampled(tmp_path: Path) -> None:
    """The run's last CAM sample must still carry its Python-owned fields.

    An hourly tape at a half-hour timestep closes this stream's window one
    step after CAM writes the sample that window belongs to, so a run that
    stops on a CAM write leaves that window open.
    """

    driver = _driver(tmp_path, nhtfrq=-1)
    driver.initialize()
    _python_field(driver, "rain")
    stream = driver.history_streams.stream("python_fields")

    driver.clock.advance()
    assert stream.step() == 0
    assert stream.accumulated == 1
    assert stream.pending == 0

    path = _cam_history_file(driver, [(1, 1, 1, 1800)])
    driver.finalize()

    assert stream.accumulated == 0
    assert stream.pending == 0
    with netCDF4.Dataset(path) as dataset:
        added = dataset.variables["rain"]
        assert added.dimensions == ("time", "ncol")
        # The partial window is reduced over the one sample it holds.
        expected = np.asarray([1, 2, 3, 4, 5], dtype="f4")
        assert np.allclose(added[0, :], expected, rtol=1e-6)


def test_output_false_keeps_a_field_out_of_history(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    _python_field(driver, "kept")
    _python_field(driver, "scratch", output=False)
    stream = driver.history_streams.stream("python_fields")

    assert [item.output_name for item in stream.fields()] == ["kept"]


def test_non_column_fields_are_skipped_by_automatic_selection(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    _python_field(driver, "rain")
    driver.pool.create_from_array(
        "tuning", np.zeros((3,), dtype=np.float64, order="F")
    )
    stream = driver.history_streams.stream("python_fields")

    assert [item.output_name for item in stream.fields()] == ["rain"]


def test_explicit_fields_fail_closed(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    _python_field(driver, "rain")

    with pytest.raises(PICAMStateError, match="unknown field"):
        driver.history_streams.install("bad", fields=["missing"])
    with pytest.raises(PICAMConfigurationError, match="mean.*instantaneous"):
        driver.history_streams.install("period", fields=["rain"], time_period="max")
    with pytest.raises(PICAMConfigurationError, match="already installed"):
        driver.history_streams.install("python_fields", fields=["rain"])
    with pytest.raises(PICAMConfigurationError, match="not installed"):
        driver.history_streams.stream("absent")


def test_default_stream_reads_nhtfrq_from_the_case_namelist(
    tmp_path: Path,
) -> None:
    hourly = _driver(tmp_path / "a", nhtfrq=-6)
    hourly.initialize()
    assert hourly.history_streams.stream("python_fields").spec.nhtfrq == -6

    stepwise = _driver(tmp_path / "b", nhtfrq=48)
    stepwise.initialize()
    assert stepwise.history_streams.stream("python_fields").spec.nhtfrq == 48


def test_default_stream_can_be_switched_off(tmp_path: Path) -> None:
    driver = _driver(tmp_path, default_history_stream=False)
    driver.initialize()

    assert "python_fields" not in driver.history_streams
    assert all(
        item["name"] != "python_fields" for item in driver.step_plan.describe()
    )


def test_default_stream_runs_inside_the_workflow(tmp_path: Path) -> None:
    driver = _driver(tmp_path, nhtfrq=1)
    driver.initialize()
    _python_field(driver, "rain")
    names = [item["name"] for item in driver.step_plan.describe()]
    assert names.index("python_fields") == names.index("history") + 1

    rows = driver.step()

    assert any(row.operation == "python_fields" for row in rows)
    assert driver.history_streams.stream("python_fields").pending == 1
