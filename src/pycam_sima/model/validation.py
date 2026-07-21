"""Fail-closed history validation against fixed oracle output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from .errors import ValidationError
from .history import HISTORY_FIELDS

HISTORY_FIELD_NAMES = tuple(name for name, _state_name in HISTORY_FIELDS)
TIME_FIELDS = ("time", "date", "datesec", "nsteph")
GRID_FIELDS = ("lat", "lon", "area", "lev", "ilev", "hyam", "hybm", "hyai", "hybi")


def _compare_variable(filename: str, field: str, lhs: Dataset, rhs: Dataset, *, step: int) -> None:
    if field not in lhs.variables or field not in rhs.variables:
        raise ValidationError(f"{filename}: required variable {field!r} is absent from oracle or model history")
    a, b = np.asarray(lhs[field][...]), np.asarray(rhs[field][...])
    if a.dtype != b.dtype or a.shape != b.shape:
        raise ValidationError(f"file={filename} field={field}: expected dtype/shape {a.dtype}/{a.shape}, actual {b.dtype}/{b.shape}")
    equal = a.view(np.uint64) == b.view(np.uint64) if a.dtype == np.float64 else a == b
    if bool(np.all(equal)):
        return
    index = tuple(int(value) for value in np.argwhere(~equal)[0])
    if a.dtype == np.float64:
        expected_bits = f"0x{int(a[index].view(np.uint64)):016x}"
        actual_bits = f"0x{int(b[index].view(np.uint64)):016x}"
    else:
        expected_bits, actual_bits = repr(a[index]), repr(b[index])
    raise ValidationError(
        f"file={filename} rank=global step={step} phase=history field={field} "
        f"global/local_index={index} expected_bits={expected_bits} actual_bits={actual_bits} "
        "first_differing_operation_or_kernel=unknown"
    )


def compare_history_files(expected_file: str | Path, actual_file: str | Path) -> None:
    expected_file, actual_file = Path(expected_file), Path(actual_file)
    with Dataset(expected_file) as lhs, Dataset(actual_file) as rhs:
        step = int(np.asarray(lhs["nsteph"][...]).reshape(-1)[0])
        for field in TIME_FIELDS + GRID_FIELDS + HISTORY_FIELD_NAMES:
            _compare_variable(expected_file.name, field, lhs, rhs, step=step)


def compare_history_directories(expected_dir: str | Path, actual_dir: str | Path, *, expected_files: int = 51, expected_numeric_variables: int = 26) -> None:
    expected = {item.name: item for item in Path(expected_dir).glob("*.cam.h0.*.nc")}
    actual = {item.name: item for item in Path(actual_dir).glob("*.cam.h0.*.nc")}
    if len(expected) != expected_files:
        raise ValidationError(f"oracle history has {len(expected)} files, required {expected_files}")
    if set(expected) != set(actual):
        raise ValidationError(f"history file set mismatch expected_only={sorted(set(expected)-set(actual))} actual_only={sorted(set(actual)-set(expected))}")
    for filename in sorted(expected):
        if len(HISTORY_FIELD_NAMES) != expected_numeric_variables:
            raise ValidationError(f"validator contract has {len(HISTORY_FIELD_NAMES)} diagnostic fields, required {expected_numeric_variables}")
        compare_history_files(expected[filename], actual[filename])
