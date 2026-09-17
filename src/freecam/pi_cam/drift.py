"""State drift of a run from a reference: the size of the difference, not its bits.

The bit-for-bit comparison (`compare_pi_cam_directories`) says whether two runs
are the same; a replaced process makes them differ, and the question becomes by
how much.  This module measures that on one CAM file of each run (the restart
by default), field by field, in the field's own unit scaled to something a
reader can weigh (temperature in K, humidity in g/kg, pressure in hPa).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from freecam.pi_cam.validation import _logical_name

#: Field, scale applied to the difference, and the unit the scaled value is in.
DEFAULT_VARIABLES: tuple[tuple[str, float, str], ...] = (
    ("T", 1.0, "K"),
    ("Q", 1e3, "g/kg"),
    ("U", 1.0, "m/s"),
    ("V", 1.0, "m/s"),
    ("PS", 1e-2, "hPa"),
    ("CLDLIQ", 1e6, "mg/kg"),
    ("CLDICE", 1e6, "mg/kg"),
)


@dataclass(frozen=True)
class DriftRow:
    """One field's drift: root-mean-square and largest absolute difference, scaled."""

    name: str
    unit: str
    scale: float
    rms: float | None
    largest: float | None
    mean: float | None
    reference_rms: float | None
    count: int
    note: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name, "unit": self.unit, "scale": self.scale,
            "rms": self.rms, "max": self.largest, "mean": self.mean,
            "reference_rms": self.reference_rms, "count": self.count, "note": self.note,
        }


def cam_file(root: Path, kind: str = "r", pick: str = "only") -> Path:
    """The ``*.cam.<kind>.*.nc`` file under ``root``.

    ``pick`` says which when several exist: ``only`` refuses ambiguity, ``last``
    and ``first`` take the latest or earliest by name (CAM names carry the date).
    """

    matches = sorted(Path(root).glob(f"*.cam.{kind}.*.nc"))
    if not matches:
        raise FileNotFoundError(f"{root}: expected a *.cam.{kind}.*.nc file, found none")
    if pick == "last":
        return matches[-1]
    if pick == "first":
        return matches[0]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{root}: expected one *.cam.{kind}.*.nc file, found {len(matches)}"
            f" ({', '.join(m.name for m in matches)}); pass pick='last' or 'first'")
    return matches[0]


def _load(path: Path, name: str) -> np.ndarray | None:
    from netCDF4 import Dataset

    with Dataset(path) as dataset:
        if name not in dataset.variables:
            return None
        values = dataset.variables[name][:]
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)


def variable_drift(reference: Path, candidate: Path,
                   variables: Iterable[tuple[str, float, str]] = DEFAULT_VARIABLES) -> list[DriftRow]:
    """Scaled rms, largest and mean difference of each field, candidate minus reference."""

    rows: list[DriftRow] = []
    for name, scale, unit in variables:
        before, after = _load(reference, name), _load(candidate, name)
        if before is None or after is None:
            missing = " and ".join(w for w, v in (("reference", before), ("candidate", after)) if v is None)
            rows.append(DriftRow(name, unit, scale, None, None, None, None, 0, f"absent from the {missing}"))
            continue
        if before.shape != after.shape:
            rows.append(DriftRow(name, unit, scale, None, None, None, None, 0,
                                 f"shapes differ: {before.shape} vs {after.shape}"))
            continue
        delta = (after - before) * scale
        finite = np.isfinite(delta)
        if not finite.any():
            rows.append(DriftRow(name, unit, scale, None, None, None, None, 0, "no finite values"))
            continue
        delta = delta[finite]
        rows.append(DriftRow(
            name, unit, scale,
            rms=float(math.sqrt(float(np.mean(delta * delta)))),
            largest=float(np.abs(delta).max()),
            mean=float(delta.mean()),
            reference_rms=float(math.sqrt(float(np.mean((before[finite] * scale) ** 2)))),
            count=int(delta.size)))
    return rows


def report(reference_dir: Path, candidate_dir: Path, kind: str = "r",
           variables: Iterable[tuple[str, float, str]] = DEFAULT_VARIABLES, pick: str = "only") -> dict[str, Any]:
    """The drift of ``candidate_dir`` from ``reference_dir`` on their ``cam.<kind>`` files.

    Names only the files' logical names, never their directories: a record
    written from this names no site.
    """

    reference, candidate = cam_file(reference_dir, kind, pick), cam_file(candidate_dir, kind, pick)
    rows = variable_drift(reference, candidate, variables)
    return {
        "schema_version": 1,
        "what": "state drift of the candidate from the reference, candidate minus reference, "
                "per field: root-mean-square, largest absolute and mean difference in the scaled unit",
        "file_kind": kind,
        "reference_file": _logical_name(reference),
        "candidate_file": _logical_name(candidate),
        "identical": all(row.rms == 0.0 for row in rows if row.rms is not None) and any(row.rms is not None for row in rows),
        "fields": [row.to_payload() for row in rows],
    }


def format_table(rows: Sequence[DriftRow]) -> str:
    """The rows as fixed-width text: field, rms, max, mean, unit, and the reference's own rms."""

    lines = [f"{'field':8s} {'rms':>10s} {'max':>10s} {'mean':>11s} {'unit':6s} {'ref rms':>10s}"]
    for row in rows:
        if row.rms is None:
            lines.append(f"{row.name:8s} {'-':>10s} {'-':>10s} {'-':>11s} {row.unit:6s} {'-':>10s}  {row.note}")
            continue
        lines.append(f"{row.name:8s} {row.rms:10.4g} {row.largest:10.4g} {row.mean:11.4g} {row.unit:6s} "
                     f"{row.reference_rms:10.4g}")
    return "\n".join(lines)


def parse_variables(items: Iterable[str]) -> tuple[tuple[str, float, str], ...]:
    """``NAME[:SCALE[:UNIT]]`` items into the (name, scale, unit) triples the report takes."""

    defaults = {name: (scale, unit) for name, scale, unit in DEFAULT_VARIABLES}
    result = []
    for item in items:
        parts = item.split(":")
        name = parts[0]
        scale = float(parts[1]) if len(parts) > 1 and parts[1] else defaults.get(name, (1.0, ""))[0]
        unit = parts[2] if len(parts) > 2 else (defaults[name][1] if name in defaults and len(parts) < 2 else "")
        result.append((name, scale, unit))
    return tuple(result)
