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
