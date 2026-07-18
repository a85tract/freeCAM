from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from netCDF4 import Dataset


DEFAULT_STATE_FIELDS = ("T", "Q", "U", "V", "PS")


@dataclass(frozen=True)
class FieldComparison:
    timestamp: str
    field: str
    equal: bool
    differing_values: int
    max_abs_difference: float


@dataclass(frozen=True)
class HistoryComparison:
    bfb: bool
    reference_files: int
    candidate_files: int
    compared_files: int
    missing_in_candidate: tuple[str, ...]
    extra_in_candidate: tuple[str, ...]
    first_difference: FieldComparison | None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        return result


def discover_history_files(directory: str | Path) -> dict[str, Path]:
    root = Path(directory).resolve()
    result: dict[str, Path] = {}
    for path in root.glob("*.cam.h1i.*.nc"):
        try:
            timestamp = path.name.split(".h1i.", 1)[1]
        except IndexError:
            continue
        if timestamp in result:
            raise ValueError(f"duplicate history timestamp {timestamp} in {root}")
        result[timestamp] = path
    return result


def compare_history(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    *,
    fields: Iterable[str] = DEFAULT_STATE_FIELDS,
) -> HistoryComparison:
    fields = tuple(fields)
    reference = discover_history_files(reference_dir)
    candidate = discover_history_files(candidate_dir)
    missing = tuple(sorted(reference.keys() - candidate.keys()))
    extra = tuple(sorted(candidate.keys() - reference.keys()))
    common = sorted(reference.keys() & candidate.keys())
    first_difference: FieldComparison | None = None

    for timestamp in common:
        with Dataset(reference[timestamp]) as ref, Dataset(candidate[timestamp]) as cand:
            for field in fields:
                if field not in ref.variables or field not in cand.variables:
                    first_difference = FieldComparison(timestamp, field, False, -1, float("inf"))
                    break
                left = np.asarray(ref[field][:])
                right = np.asarray(cand[field][:])
                if left.shape != right.shape or left.dtype != right.dtype:
                    first_difference = FieldComparison(timestamp, field, False, -1, float("inf"))
                    break
                if not np.array_equal(left, right):
                    delta = np.abs(left - right)
                    first_difference = FieldComparison(
                        timestamp,
                        field,
                        False,
                        int(np.count_nonzero(left != right)),
                        float(np.max(delta)),
                    )
                    break
        if first_difference is not None:
            break

    bfb = not missing and not extra and first_difference is None
    return HistoryComparison(
        bfb=bfb,
        reference_files=len(reference),
        candidate_files=len(candidate),
        compared_files=len(common),
        missing_in_candidate=missing,
        extra_in_candidate=extra,
        first_difference=first_difference,
    )


def history_manifest(
    directory: str | Path,
    *,
    fields: Iterable[str] = DEFAULT_STATE_FIELDS,
) -> dict[str, object]:
    fields = tuple(fields)
    files = discover_history_files(directory)
    records: list[dict[str, object]] = []
    for timestamp, path in sorted(files.items()):
        field_records: dict[str, object] = {}
        with Dataset(path) as dataset:
            for field in fields:
                array = np.asarray(dataset[field][:])
                field_records[field] = {
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                }
        records.append(
            {
                "timestamp": timestamp,
                "file": path.name,
                "fields": field_records,
            }
        )
    return {"file_count": len(records), "fields": list(fields), "records": records}
