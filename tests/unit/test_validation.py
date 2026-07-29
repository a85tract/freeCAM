from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest

from pycam_sima.model.errors import ValidationError
from pycam_sima.model.validation import compare_history_directories


def _write_history(path: Path, temperature: float) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("ncol", 1)
        dataset.createDimension("lev", 1)
        dataset.createDimension("ilev", 2)
        for name, dtype, value in (
            ("time", "f8", 0.0),
            ("date", "i4", 10101),
            ("datesec", "i4", 0),
            ("nsteph", "i4", 0),
        ):
            dataset.createVariable(name, dtype, ("time",))[:] = value
        for name, dims, values in (
            ("lat", ("ncol",), [0.0]),
            ("lon", ("ncol",), [0.0]),
            ("area", ("ncol",), [1.0]),
            ("lev", ("lev",), [500.0]),
            ("ilev", ("ilev",), [0.0, 1000.0]),
            ("hyam", ("lev",), [0.5]),
            ("hybm", ("lev",), [0.5]),
            ("hyai", ("ilev",), [0.0, 1.0]),
            ("hybi", ("ilev",), [1.0, 0.0]),
        ):
            dataset.createVariable(name, "f8", dims)[:] = np.asarray(values)
        dataset.createVariable("T", "f8", ("time", "lev", "ncol"))[:] = (
            temperature
        )


def test_history_comparison_pairs_different_stream_names_by_timestamp(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_history(
        reference / "original.cam.h1i.0001-01-01-00000.nc",
        250.0,
    )
    _write_history(
        candidate / "python.cam.h0.0001-01-01-00000i.nc",
        250.0,
    )

    compare_history_directories(
        reference,
        candidate,
        expected_files=1,
        expected_numeric_variables=1,
        fields=("T",),
    )


def test_history_comparison_still_fails_closed_for_selected_fields(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_history(
        reference / "original.cam.h1i.0001-01-01-00000.nc",
        250.0,
    )
    _write_history(
        candidate / "python.cam.h0.0001-01-01-00000i.nc",
        251.0,
    )

    with pytest.raises(ValidationError, match="field=T"):
        compare_history_directories(
            reference,
            candidate,
            expected_files=1,
            expected_numeric_variables=1,
            fields=("T",),
        )
