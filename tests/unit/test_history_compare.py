from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pycam_sima.history_compare import compare_history


def _write(path: Path, value: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("lev", 2)
        dataset.createDimension("ncol", 3)
        for field in ("T", "Q", "U", "V"):
            variable = dataset.createVariable(field, "f8", ("time", "lev", "ncol"))
            variable[:] = np.full((1, 2, 3), value)
        pressure = dataset.createVariable("PS", "f8", ("time", "ncol"))
        pressure[:] = np.full((1, 3), value)


def test_history_comparison_is_bitwise(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    name = "case.cam.h1i.0001-01-01-00000.nc"
    _write(reference / name, 1.0)
    _write(candidate / name, 1.0)
    assert compare_history(reference, candidate).bfb

    _write(candidate / name, np.nextafter(1.0, 2.0))
    result = compare_history(reference, candidate)
    assert not result.bfb
    assert result.first_difference is not None
    assert result.first_difference.differing_values > 0
