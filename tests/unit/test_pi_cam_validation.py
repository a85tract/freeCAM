from netCDF4 import Dataset
import numpy as np

from freecam.pi_cam import compare_pi_cam_directories


def _write(path, value, *, timestamp="12:00:00", run_path="/first/run"):
    with Dataset(path, "w") as dataset:
        dataset.createDimension("column", 2)
        dataset.createDimension("chars", 8)
        variable = dataset.createVariable("T", "f8", ("column",))
        variable[:] = np.asarray(value, dtype="f8")
        written = dataset.createVariable("time_written", "S1", ("chars",))
        written[:] = np.asarray(tuple(timestamp), dtype="S1")
        stored_path = dataset.createVariable("cpath", "S1", ("chars",))
        stored_path[:] = np.asarray(tuple(run_path[:8].ljust(8)), dtype="S1")


def test_pi_cam_validation_ignores_case_prefix_but_not_one_bit(tmp_path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write(reference / "oracle.cam.h0.0001-01-01-00000.nc", [1.0, 2.0])
    _write(
        candidate / "python.cam.h0.0001-01-01-00000.nc",
        [1.0, 2.0],
        timestamp="13:00:00",
        run_path="/second/run",
    )

    assert compare_pi_cam_directories(reference, candidate).bfb

    _write(
        candidate / "python.cam.h0.0001-01-01-00000.nc",
        [1.0, np.nextafter(2.0, 3.0)],
    )
    result = compare_pi_cam_directories(reference, candidate)
    assert not result.bfb
    assert result.first_difference["variable"] == "T"
    assert result.first_difference["index"] == (1,)


def test_a_nan_both_runs_store_with_the_same_bits_is_no_difference(tmp_path) -> None:
    # A restart file stores NaN where a field is undefined.  It compared unequal
    # to itself: NaN != NaN found it before the bits were looked at.
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write(reference / "oracle.cam.r.0001-01-02-03600.nc", [np.nan, 2.0])
    _write(candidate / "python.cam.r.0001-01-02-03600.nc", [np.nan, 2.0])

    assert compare_pi_cam_directories(reference, candidate).bfb

    other_nan = np.array([0x7FF8000000000001], dtype="u8").view("f8")[0]
    _write(candidate / "python.cam.r.0001-01-02-03600.nc", [other_nan, 2.0])
    result = compare_pi_cam_directories(reference, candidate)
    assert not result.bfb
    assert result.first_difference["index"] == (0,)


def test_other_components_are_compared_when_named(tmp_path) -> None:
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for root, value in ((reference, 1.0), (candidate, np.nextafter(1.0, 2.0))):
        _write(root / f"{root.name}.cam.h0.0001-01-01-00000.nc", [1.0, 2.0])
        _write(root / f"{root.name}.clm2.r.0001-01-02-03600.nc", [value, 2.0])

    assert compare_pi_cam_directories(reference, candidate).bfb
    result = compare_pi_cam_directories(reference, candidate, components=("cam", "clm2"))
    assert not result.bfb
    assert result.compared_files == 2
    assert result.first_difference["file"] == "clm2.r.0001-01-02-03600.nc"
