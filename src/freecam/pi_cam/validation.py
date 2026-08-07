"""Fail-closed bitwise validation for PI-CAM history and restart fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from netCDF4 import Dataset
import numpy as np


# CESM restart files contain path, timestamp, and file-list character arrays.
# The PI-CAM gate is deliberately over every *numeric* CAM variable, not these
# execution-location strings.  Numeric inventories still have to match exactly.
_NUMERIC_KINDS = frozenset({"b", "i", "u", "f", "c"})


@dataclass(frozen=True, slots=True)
class PICAMBFBResult:
    bfb: bool
    reference_files: int
    candidate_files: int
    compared_files: int
    missing_in_candidate: tuple[str, ...]
    extra_in_candidate: tuple[str, ...]
    first_difference: dict[str, Any] | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def _logical_name(path: Path) -> str | None:
    marker = path.name.find(".cam.")
    return None if marker < 0 else path.name[marker + 1 :]


def _cam_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.glob("*.cam.*.nc")):
        name = _logical_name(path)
        if name is not None:
            result[name] = path
    return result


def _value_record(value: Any, dtype: np.dtype[Any]) -> object:
    scalar = np.asarray(value, dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        unsigned = np.dtype(f"u{dtype.itemsize}")
        return {
            "value": float(scalar),
            "bits": f"0x{int(scalar.view(unsigned)):0{dtype.itemsize * 2}x}",
        }
    return scalar.item()


def _compare_netcdf(reference: Path, candidate: Path) -> dict[str, Any] | None:
    with Dataset(reference) as expected, Dataset(candidate) as actual:
        expected.set_auto_mask(False)
        expected.set_auto_scale(False)
        actual.set_auto_mask(False)
        actual.set_auto_scale(False)
        expected_names = {
            name
            for name, variable in expected.variables.items()
            if np.dtype(variable.dtype).kind in _NUMERIC_KINDS
        }
        actual_names = {
            name
            for name, variable in actual.variables.items()
            if np.dtype(variable.dtype).kind in _NUMERIC_KINDS
        }
        if expected_names != actual_names:
            return {
                "kind": "variable_inventory",
                "reference_only": sorted(expected_names - actual_names),
                "candidate_only": sorted(actual_names - expected_names),
            }
        # Keep the first reported mismatch stable across Python processes.
        for name in sorted(expected_names):
            left_variable = expected.variables[name]
            right_variable = actual.variables[name]
            if left_variable.dtype != right_variable.dtype or left_variable.shape != right_variable.shape:
                return {
                    "kind": "contract",
                    "variable": name,
                    "reference_shape": left_variable.shape,
                    "candidate_shape": right_variable.shape,
                    "reference_dtype": str(left_variable.dtype),
                    "candidate_dtype": str(right_variable.dtype),
                }
            left = np.asarray(left_variable[:])
            right = np.asarray(right_variable[:])
            if np.array_equal(left, right):
                continue
            different = np.argwhere(left != right)
            if different.size:
                index = tuple(int(item) for item in different[0])
            else:
                # NaNs compare unequal even when their payload bits match.
                left_bits = left.view(np.dtype(f"u{left.dtype.itemsize}"))
                right_bits = right.view(np.dtype(f"u{right.dtype.itemsize}"))
                bit_difference = np.argwhere(left_bits != right_bits)
                if not bit_difference.size:
                    continue
                index = tuple(int(item) for item in bit_difference[0])
            return {
                "kind": "value",
                "variable": name,
                "index": index,
                "reference": _value_record(left[index], left.dtype),
                "candidate": _value_record(right[index], right.dtype),
            }
    return None


def compare_pi_cam_directories(
    reference_root: str | Path, candidate_root: str | Path
) -> PICAMBFBResult:
    reference = _cam_files(Path(reference_root))
    candidate = _cam_files(Path(candidate_root))
    common = sorted(set(reference) & set(candidate))
    first_difference = None
    for name in common:
        difference = _compare_netcdf(reference[name], candidate[name])
        if difference is not None:
            first_difference = {"file": name, **difference}
            break
    missing = tuple(sorted(set(reference) - set(candidate)))
    extra = tuple(sorted(set(candidate) - set(reference)))
    return PICAMBFBResult(
        bfb=not missing and not extra and first_difference is None,
        reference_files=len(reference),
        candidate_files=len(candidate),
        compared_files=len(common),
        missing_in_candidate=missing,
        extra_in_candidate=extra,
        first_difference=first_difference,
    )
