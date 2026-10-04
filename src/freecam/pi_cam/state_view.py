"""The globe viewer of a PI-CAM run's state snapshots (``freecam globe DIR``).

Reads a directory :class:`freecam.pi_cam.state_record.StateRecorder` writes and
serves one page: a globe with the continents, a field at one level, a step
slider (the state at the end of each recorded step) and, at the steps recorded
after every action, a process bar (the state after each plan action, or what
that action changed).  A clicked column gives its vertical profile and the
meridian through it a height-latitude section.

    freecam globe DIR                     # serve on 127.0.0.1 (forward the port)
    freecam globe DIR --html page.html    # one self-contained page, a subset of the data

The server reads only what a view asks for; a run still being recorded is
re-read as its files grow.  The self-contained page embeds chosen fields and
levels, quantized to 8 bits (a colour map has no more), compressed.
"""

from __future__ import annotations

import argparse
import base64
import json
import secrets
import threading
import time
import webbrowser
import zlib
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

import numpy as np

from .state_record import START_OF_STEP

#: a trace's change below this fraction of the column's largest value is float noise
NOISE = 1e-12
KINDS = ("steps", "actions")

#: What counts as an anomaly in a recorded field, column by column: a value that is not finite, for
#: any field; a value below zero, for water and every constituent (the original physics leaves none
#: negative between its actions); a value outside RANGES (viewer units) where one is given.
NON_NEGATIVE = frozenset({"Q", "CLDLIQ", "CLDICE", "NUMLIQ", "NUMICE", "PS"})
RANGES = {"T": (100.0, 400.0)}
#: a column's anomaly code: 0 none, then these, the first that applies at any of its levels
ANOMALY_KINDS = ("non-finite", "negative", "out of range")


def anomaly_rule(name: str) -> str:
    """The anomaly rule applied to a field, in words."""

    if name in RANGES:
        lo, hi = RANGES[name]
        return f"finite and within {lo:g} to {hi:g}"
    if name in NON_NEGATIVE or name.startswith("phys_state.q:"):
        return "finite and not negative"
    return "finite"


def anomaly_codes(values: np.ndarray, name: str, scale: float = 1.0) -> np.ndarray:
    """Each column's anomaly code (uint8) from its values, (levels, columns) or (frames, levels, columns)."""

    finite = np.isfinite(values)
    code = np.where((~finite).any(axis=-2), 1, 0).astype(np.uint8)
    safe = np.where(finite, values, 0.0) * float(scale)
    if name in NON_NEGATIVE or name.startswith("phys_state.q:"):
        code = np.where((code == 0) & (safe < 0).any(axis=-2), 2, code).astype(np.uint8)
    if name in RANGES:
        lo, hi = RANGES[name]
        outside = (finite & ((safe < lo) | (safe > hi))).any(axis=-2)
        code = np.where((code == 0) & outside, 3, code).astype(np.uint8)
    return code


def _num(value: Any) -> float | None:
    """A float for JSON: None where it is not finite (JSON has no NaN)."""

    value = float(value)
    return value if np.isfinite(value) else None


def _finite(payload: Any) -> Any:
    """``payload`` with every non-finite float replaced by None, for strict JSON."""

    if isinstance(payload, float):
        return payload if np.isfinite(payload) else None
    if isinstance(payload, dict):
        return {key: _finite(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_finite(value) for value in payload]
    return payload


class StateData:
    """One state directory: the manifest, the columns, and any frame on demand."""

    def __init__(self, directory: str | Path, *, cache_frames: int = 48) -> None:
        self.directory = Path(directory)
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_frames = int(cache_frames)
        self._lock = threading.Lock()
        self._changes: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._changes_lock = threading.Lock()      # one computation at a time; the others wait for it
        self.reload()

    # -- loading -----------------------------------------------------------
    def _signature(self) -> tuple:
        manifest = self.directory / "manifest.json"
        return (manifest.stat().st_mtime_ns if manifest.exists() else 0,)

    def changed(self) -> bool:
        return self._signature() != self._loaded

    def reload(self) -> None:
        manifest_file = self.directory / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"{self.directory} has no manifest.json: not a state directory")
        self.manifest = json.loads(manifest_file.read_text())
        columns = np.load(self.directory / "columns.npz")
        self.rank = np.asarray(columns["rank"], dtype=np.int64)
        self.lat = np.asarray(columns["lat"], dtype=np.float64)
        self.lon = np.asarray(columns["lon"], dtype=np.float64)
        self.level_pressure = np.asarray(columns["level_pressure_pa"], dtype=np.float64)
        self.nranks = int(self.manifest["ranks"])
        self.rank_columns = np.bincount(self.rank, minlength=self.nranks)
        surface = self.directory / "surface.npz"
        self.landfrac = (np.asarray(np.load(surface)["landfrac"], dtype=np.float64)
                         if surface.is_file() else np.zeros(len(self.lat)))
        if self.landfrac.shape != self.lat.shape:
            self.landfrac = np.zeros(len(self.lat))
        stored = self.manifest.get("dtype", {})
        self.dtypes = stored if isinstance(stored, dict) else {"steps": stored, "actions": stored}
        self.fields = {field["name"]: field for field in self.manifest["fields"]}
        self.field_order = [field["name"] for field in self.manifest["fields"]]
        levels = [int(field["levels"]) for field in self.manifest["fields"]]
        self.levels_total = sum(levels)
        self.level_offset = {name: sum(levels[:i]) for i, name in enumerate(self.field_order)}
        self.step_frames = [int(step) for step in self.manifest.get("step_frames", [])]
        frames = self.manifest.get("action_frames", [])
        self.action_frames = [(int(frame[0]), str(frame[1])) for frame in frames]
        #: what computed each action frame (None where the run did not record it)
        self.action_owners = [frame[2] if len(frame) > 2 else None for frame in frames]
        self.action_steps: dict[int, list[int]] = {}
        for index, (step, _) in enumerate(self.action_frames):
            self.action_steps.setdefault(step, []).append(index)
        with self._lock:
            self._cache.clear()
            self._changes.clear()
            self._anomalies: dict[Any, dict[str, np.ndarray]] = {}
            self._seconds = None
        self._loaded = self._signature()

    def frames(self, kind: str) -> int:
        return len(self.step_frames) if kind == "steps" else len(self.action_frames)

    # -- frames ------------------------------------------------------------
    def frame(self, kind: str, index: int, field: str) -> np.ndarray:
        """One field of one snapshot over every column, (levels, columns), in stored units."""

        if kind not in KINDS:
            raise ValueError(f"kind is steps or actions, not {kind!r}")
        if field not in self.fields:
            raise KeyError(f"no field {field!r}; the run recorded {self.field_order}")
        index = int(index)
        if not 0 <= index < self.frames(kind):
            raise ValueError(f"{kind} frame {index} is not recorded (0..{self.frames(kind) - 1})")
        key = (kind, index, field)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        levels = int(self.fields[field]["levels"])
        parts = []
        for rank in range(self.nranks):
            n = int(self.rank_columns[rank])
            if n == 0:
                continue
            dtype = np.dtype(self.dtypes[kind])
            start = (index * n * self.levels_total + n * self.level_offset[field]) * dtype.itemsize
            values = np.fromfile(self.directory / "ranks" / f"rank-{rank:04d}.{kind}.bin", dtype=dtype,
                                 count=levels * n, offset=start)
            if values.size != levels * n:
                raise ValueError(f"rank {rank}'s {kind} file ends before frame {index}")
            parts.append(values.reshape(levels, n))
        block = np.concatenate(parts, axis=1) if parts else np.zeros((levels, 0), np.dtype(self.dtypes[kind]))
        with self._lock:
            self._cache[key] = block
            while len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
        return block

    def previous(self, kind: str, index: int) -> int | None:
        """The frame a change is measured from: the previous recorded step, or the previous
        action frame of the same step (None for the first)."""

        if index <= 0:
            return None
        if kind == "actions" and self.action_frames[index - 1][0] != self.action_frames[index][0]:
            return None
        return index - 1

    def values(self, kind: str, index: int, field: str, mode: str = "value") -> np.ndarray:
        """The shown values, (levels, columns), float32: the state, or its change since the
        previous frame, in the viewer's units."""

        scale = float(self.fields[field]["scale"])
        now = self.frame(kind, index, field).astype(np.float64)
        if mode == "value":
            return (now * scale).astype(np.float32)
        if mode != "change":
            raise ValueError(f"mode is value or change, not {mode!r}")
        before = self.previous(kind, index)
        if before is None:
            return np.zeros(now.shape, np.float32)
        # the difference of the stored values (float64 for action frames), then scaled: what the
        # step or the action did
        return ((now - self.frame(kind, before, field)) * scale).astype(np.float32)

    # -- views -------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        steps: dict[str, list[str]] = {}
        for step, name in self.action_frames:
            steps.setdefault(str(step), []).append(name)
        return {
            "directory": self.directory.name,
            # the run directory's name only: a page is shared, and a path names the site
            "run": Path(str(self.manifest.get("run", ""))).name,
            "complete": bool(self.manifest.get("complete")),
            "fields": [{key: self.fields[name][key] for key in ("name", "units", "scale", "label", "levels")}
                       for name in self.field_order],
            "every": self.manifest.get("every", 1),
            "step_frames": self.step_frames,
            "action_steps": {step: names for step, names in steps.items()},
            "action_frame_index": {str(step): indices for step, indices in self.action_steps.items()},
            "action_owners": {str(step): [self.action_owners[i] for i in indices]
                              for step, indices in self.action_steps.items()},
            "level_pressure_hpa": [round(float(p) / 100.0, 2) for p in self.level_pressure],
            "columns": len(self.lat),
            "lat": [round(float(v), 3) for v in self.lat],
            "lon": [round(float(v), 3) for v in self.lon],
            "land": [round(float(v), 2) for v in self.landfrac],
            "start_of_step": START_OF_STEP,
            "anomaly_rules": {name: anomaly_rule(name) for name in self.field_order},
            "anomaly_kinds": list(ANOMALY_KINDS),
        }

    def level(self, kind: str, index: int, field: str, level: int, mode: str = "value") -> np.ndarray:
        values = self.values(kind, index, field, mode)
        level = int(level)
        if not 0 <= level < values.shape[0]:
            raise ValueError(f"{field} has levels 0..{values.shape[0] - 1}")
        return np.ascontiguousarray(values[level], dtype="<f4")

    def profile(self, kind: str, index: int, field: str, column: int) -> dict[str, Any]:
        """One column's values at every level, now and at the frame before."""

        column = int(column)
        if not 0 <= column < len(self.lat):
            raise ValueError(f"column {column} out of range")
        scale = float(self.fields[field]["scale"])
        now = self.frame(kind, index, field)[:, column].astype(np.float64) * scale
        before_index = self.previous(kind, index)
        before = (None if before_index is None
                  else self.frame(kind, before_index, field)[:, column].astype(np.float64) * scale)
        return {"column": column, "lat": float(self.lat[column]), "lon": float(self.lon[column]),
                "rank": int(self.rank[column]), "land": float(self.landfrac[column]),
                "values": [float(v) for v in now],
                "before": None if before is None else [float(v) for v in before]}

    def section(self, kind: str, index: int, field: str, lon: float, mode: str = "value",
                half_width_degrees: float = 1.0) -> dict[str, Any]:
        """Every column within ``half_width_degrees`` of great-circle distance of the meridian
        through ``lon``, sorted by latitude: (levels, columns) of shown values."""

        dlon = (self.lon - float(lon) + 180.0) % 360.0 - 180.0
        near = np.abs(dlon) * np.cos(np.radians(self.lat)) <= half_width_degrees
        order = np.flatnonzero(near)[np.argsort(self.lat[near], kind="stable")]
        values = self.values(kind, index, field, mode)[:, order]
        return {"lon": float(lon), "lat": [round(float(v), 3) for v in self.lat[order]],
                "values": [[float(v) for v in row] for row in values]}

    def action_seconds(self) -> np.ndarray | None:
        """Every action frame's wall time on every rank, (ranks, frames); None when the run
        recorded none (older directories)."""

        with self._lock:
            if self._seconds is not None:
                return self._seconds
        if not self.manifest.get("action_seconds"):
            return None
        count = len(self.action_frames)
        seconds = np.zeros((self.nranks, count))
        for rank in range(self.nranks):
            if int(self.rank_columns[rank]) == 0:
                continue
            values = np.fromfile(self.directory / "ranks" / f"rank-{rank:04d}.action_seconds.bin", dtype="<f8",
                                 count=count)
            if values.size != count:
                return None
            seconds[rank] = values
        with self._lock:
            self._seconds = seconds
        return seconds

    def changes(self, step: int, field: str) -> list[dict[str, Any]]:
        """Per action frame of ``step``: how much the action changed ``field`` (largest and
        root-mean-square change over every column and level, in the viewer's units).  Computed
        for every field of the step at once, from one read of each rank's frames, and kept."""

        if field not in self.fields:
            raise KeyError(f"no field {field!r}; the run recorded {self.field_order}")
        step = int(step)
        if step not in self.action_steps:
            raise ValueError(f"step {step} has no action frames (recorded: {sorted(self.action_steps)})")
        key = (step, field)
        with self._lock:
            cached = self._changes.get(key)
        if cached is not None:
            return cached
        with self._changes_lock:
            with self._lock:
                cached = self._changes.get(key)
            if cached is not None:
                return cached
            rows = self._step_changes(step)
            with self._lock:
                self._changes.update({(step, name): value for name, value in rows.items()})
            return rows[field]

    def nearest(self, lat: float, lon: float) -> int:
        """The column nearest a place (degrees), by great-circle distance."""

        la, lo = np.radians(self.lat), np.radians(self.lon)
        pla, plo = np.radians(float(lat)), np.radians(float(lon))
        cosine = np.sin(la) * np.sin(pla) + np.cos(la) * np.cos(pla) * np.cos(lo - plo)
        return int(np.argmax(cosine))

    def trace(self, step: int, column: int, level: int | None = None,
              fields: list[str] | None = None) -> dict[str, Any]:
        """One column through one action step: every recorded field at ``level`` after each action,
        what that action changed there, the largest change it made anywhere in the column, its
        time and what computed it.  With no level, the level where the first field changed most
        over the step."""

        step, column = int(step), int(column)
        if step not in self.action_steps:
            raise ValueError(f"step {step} has no action frames (recorded: {sorted(self.action_steps)})")
        if not 0 <= column < len(self.lat):
            raise ValueError(f"column {column} out of range")
        fields = list(fields or self.field_order)
        for field in fields:
            if field not in self.fields:
                raise KeyError(f"no field {field!r}; the run recorded {self.field_order}")
        indices = self.action_steps[step]
        # (actions, levels) of each field in this column, in the viewer's units
        stored = self._column_frames("actions", indices, column)
        columns = {field: stored[:, self.level_offset[field]:self.level_offset[field] + int(self.fields[field]["levels"])]
                   * float(self.fields[field]["scale"]) for field in fields}
        first = columns[fields[0]]
        codes = self.action_codes(step)
        if level is None:
            level = self._anomalous_level(fields, columns, codes, column)
        if level is None:
            # where the first field changed most over the step
            moved = np.nan_to_num(np.abs(first[-1] - first[0]), nan=0.0, posinf=0.0)
            level = int(np.argmax(moved)) if first.shape[1] > 1 else 0
        level = int(level)
        if not 0 <= level < first.shape[1]:
            raise ValueError(f"{fields[0]} has levels 0..{first.shape[1] - 1}")
        seconds = self.action_seconds()
        ranks = self.rank_columns > 0
        rows = []
        for k, index in enumerate(indices):
            row: dict[str, Any] = {"index": index, "name": self.action_frames[index][1],
                                   "owner": self.action_owners[index], "values": {}, "change": {},
                                   "column_max_change": {}, "anomaly": {}, "became": {}}
            for field in fields:
                values = columns[field]
                lev = min(level, values.shape[1] - 1)
                row["values"][field] = _num(values[k, lev])
                code = int(codes[field][k, column])
                if code:
                    # the column (at any level) is anomalous after this action; "became": it was not before
                    row["anomaly"][field] = ANOMALY_KINDS[code - 1]
                    if not k or int(codes[field][k - 1, column]) != code:
                        row["became"][field] = ANOMALY_KINDS[code - 1]
                if k:
                    row["change"][field] = _num(values[k, lev] - values[k - 1, lev])
                    moved = np.abs(values[k] - values[k - 1])
                    row["column_max_change"][field] = _num(np.nanmax(moved)) if np.isfinite(moved).any() else None
            if seconds is not None and k:
                row["seconds_mean"] = float(seconds[ranks, index].mean())
                row["seconds_max"] = float(seconds[ranks, index].max())
            rows.append(row)
        pressure = self.level_pressure[level] / 100.0 if level < len(self.level_pressure) else None
        return {"step": step, "column": column, "lat": float(self.lat[column]), "lon": float(self.lon[column]),
                "land": float(self.landfrac[column]), "level": level,
                "pressure_hpa": None if pressure is None else float(pressure),
                # a change below this is float noise (a denormal, a round-off): 1e-12 of the largest
                # value anywhere in the column over the step
                "fields": [{"name": f, "units": self.fields[f]["units"], "rule": anomaly_rule(f),
                            "noise": float(NOISE * np.nanmax(np.abs(columns[f]))) if np.isfinite(columns[f]).any() else 0.0}
                           for f in fields], "rows": rows}

    @staticmethod
    def _anomalous_level(fields: list[str], columns: dict[str, np.ndarray], codes: dict[str, np.ndarray],
                         column: int) -> int | None:
        """The level where the column first went anomalous in the step, in the first field that did."""

        for field in fields:
            path = codes[field][:, column]
            if not path.any():
                continue
            k = int(np.flatnonzero(path)[0])
            # this field's levels after action k (viewer units), each judged as a column of one level
            per_level = np.array([anomaly_codes(np.array([[v]]), field)[0] != 0 for v in columns[field][k]])
            if per_level.any():
                return int(np.flatnonzero(per_level)[0])
        return None

    def _column_frames(self, kind: str, indices: list[int], column: int) -> np.ndarray:
        """Every recorded level of one column in the given frames, (frames, levels), stored units
        as float64: read from the one rank file that holds the column, not the whole globe."""

        # frames concatenate the ranks' columns in rank order, so the column is a slice of one file
        starts = np.concatenate([[0], np.cumsum(self.rank_columns)])
        rank = int(np.searchsorted(starts, column, side="right") - 1)
        n, local = int(self.rank_columns[rank]), column - int(starts[rank])
        dtype = np.dtype(self.dtypes[kind])
        per = n * self.levels_total
        rows = []
        with open(self.directory / "ranks" / f"rank-{rank:04d}.{kind}.bin", "rb") as handle:
            for index in indices:
                handle.seek(int(index) * per * dtype.itemsize)
                values = np.fromfile(handle, dtype=dtype, count=per)
                if values.size != per:
                    raise ValueError(f"rank {rank}'s {kind} file ends before frame {index}")
                rows.append(values.reshape(self.levels_total, n)[:, local])
        return np.stack(rows).astype(np.float64)

    # -- anomalies -----------------------------------------------------------
    def _codes(self, kind: str, indices: list[int]) -> dict[str, np.ndarray]:
        """Each field's anomaly code per frame and column, (frames, columns) uint8, for ``indices``
        of ``kind`` (contiguous): one read of each rank's file."""

        first, count = indices[0], len(indices)
        if indices != list(range(first, first + count)):
            raise ValueError(f"the {kind} frames {indices[:3]}... are not contiguous")
        dtype = np.dtype(self.dtypes[kind])
        parts: dict[str, list[np.ndarray]] = {name: [] for name in self.field_order}
        for rank in range(self.nranks):
            n = int(self.rank_columns[rank])
            if n == 0:
                continue
            per = n * self.levels_total
            block = np.fromfile(self.directory / "ranks" / f"rank-{rank:04d}.{kind}.bin", dtype=dtype,
                                count=count * per, offset=first * per * dtype.itemsize)
            if block.size != count * per:
                raise ValueError(f"rank {rank}'s {kind} file ends before frame {first + count - 1}")
            block = block.reshape(count, self.levels_total, n)
            for name in self.field_order:
                offset, levels = self.level_offset[name], int(self.fields[name]["levels"])
                parts[name].append(anomaly_codes(block[:, offset:offset + levels, :], name,
                                                 float(self.fields[name]["scale"])))
        return {name: np.concatenate(parts[name], axis=1) if parts[name] else np.zeros((count, 0), np.uint8)
                for name in self.field_order}

    def _cached_codes(self, key: Any, kind: str, indices: list[int]) -> dict[str, np.ndarray]:
        with self._changes_lock:
            codes = self._anomalies.get(key)
            if codes is None:
                codes = self._anomalies[key] = self._codes(kind, indices)
        return codes

    def action_codes(self, step: int) -> dict[str, np.ndarray]:
        """Each field's anomaly code after every action of an action step, (actions, columns)."""

        step = int(step)
        if step not in self.action_steps:
            raise ValueError(f"step {step} has no action frames (recorded: {sorted(self.action_steps)})")
        return self._cached_codes(("actions", step), "actions", self.action_steps[step])

    def step_codes(self) -> dict[str, np.ndarray]:
        """Each field's anomaly code at the end of every recorded step, (steps, columns)."""

        if not self.step_frames:
            return {name: np.zeros((0, len(self.lat)), np.uint8) for name in self.field_order}
        return self._cached_codes(("steps",), "steps", list(range(len(self.step_frames))))

    def anomaly_mask(self, kind: str, index: int, field: str) -> np.ndarray:
        """One frame's anomaly code per column, for the globe."""

        if field not in self.fields:
            raise KeyError(f"no field {field!r}; the run recorded {self.field_order}")
        index = int(index)
        if kind == "steps":
            return self.step_codes()[field][index]
        if kind != "actions" or not 0 <= index < len(self.action_frames):
            raise ValueError(f"no {kind} frame {index}")
        step = self.action_frames[index][0]
        return self.action_codes(step)[field][self.action_steps[step].index(index)]

    def anomalies(self, step: int) -> dict[str, Any]:
        """What went wrong in an action step: for each field and action, how many columns are
        anomalous after it and how many of them it made so (they were not before it), and the first
        action that made each kind of anomaly -- the process to look at."""

        step = int(step)
        codes = self.action_codes(step)
        indices = self.action_steps[step]
        fields, first = {}, []
        for name in self.field_order:
            c = codes[name]
            frames = []
            for k, index in enumerate(indices):
                now = c[k] != 0
                new = now & (c[k - 1] == 0) if k else now
                frames.append({"columns": int(now.sum()), "new": int(new.sum()),
                               "kinds": {kind: int((c[k] == code).sum()) for code, kind in enumerate(ANOMALY_KINDS, 1)
                                         if (c[k] == code).any()}})
                for code, kind in enumerate(ANOMALY_KINDS, 1):
                    made = (c[k] == code) & ((c[k - 1] != code) if k else True)
                    if made.any() and not any(f["field"] == name and f["kind"] == kind for f in first):
                        first.append({"field": name, "kind": kind, "k": k, "index": index,
                                      "name": self.action_frames[index][1], "owner": self.action_owners[index],
                                      "columns": int(made.sum()), "at_start": k == 0,
                                      "example_columns": [int(v) for v in np.flatnonzero(made)[:5]]})
            fields[name] = {"rule": anomaly_rule(name), "frames": frames}
        first.sort(key=lambda f: (f["k"], f["field"]))
        return {"step": step, "names": [self.action_frames[i][1] for i in indices], "fields": fields, "first": first}

    def step_anomalies(self) -> dict[str, Any]:
        """Anomalous columns at the end of every recorded step, and the first step each field had any."""

        codes = self.step_codes()
        fields, first = {}, []
        for name in self.field_order:
            counts = [int((row != 0).sum()) for row in codes[name]]
            fields[name] = {"rule": anomaly_rule(name), "columns": counts}
            bad = [p for p, n in enumerate(counts) if n]
            if bad:
                p = bad[0]
                kinds = sorted({ANOMALY_KINDS[int(v) - 1] for v in np.unique(codes[name][p]) if v})
                everywhere = next((q for q, n in enumerate(counts) if n == len(self.lat)), None)
                first.append({"field": name, "p": p, "step": self.step_frames[p], "columns": counts[p], "kinds": kinds,
                              "every_column_from": None if everywhere is None else self.step_frames[everywhere]})
        first.sort(key=lambda f: (f["p"], f["field"]))
        return {"steps": self.step_frames, "fields": fields, "first": first}

    def _step_changes(self, step: int) -> dict[str, list[dict[str, Any]]]:
        indices = self.action_steps[step]
        first, count = indices[0], len(indices)
        if indices != list(range(first, first + count)):
            raise ValueError(f"the action frames of step {step} are not contiguous")
        dtype = np.dtype(self.dtypes["actions"])
        largest = {name: np.zeros(count) for name in self.field_order}
        squares = {name: np.zeros(count) for name in self.field_order}
        values = {name: 0 for name in self.field_order}
        for rank in range(self.nranks):
            n = int(self.rank_columns[rank])
            if n == 0:
                continue
            per = n * self.levels_total
            block = np.fromfile(self.directory / "ranks" / f"rank-{rank:04d}.actions.bin", dtype=dtype,
                                count=count * per, offset=first * per * dtype.itemsize)
            if block.size != count * per:
                raise ValueError(f"rank {rank}'s actions file ends before step {step}'s frames")
            block = block.reshape(count, self.levels_total, n)
            for name in self.field_order:
                offset, levels = self.level_offset[name], int(self.fields[name]["levels"])
                change = np.diff(block[:, offset:offset + levels, :].astype(np.float64), axis=0)
                if change.size:
                    largest[name][1:] = np.maximum(largest[name][1:], np.abs(change).max(axis=(1, 2)))
                    squares[name][1:] += (change ** 2).sum(axis=(1, 2))
                values[name] += levels * n
        rows = {}
        for name in self.field_order:
            scale = abs(float(self.fields[name]["scale"]))
            rms = np.sqrt(squares[name] / max(values[name], 1)) * scale
            rows[name] = [{"index": index, "name": self.action_frames[index][1],
                           "max": float(largest[name][k] * scale), "rms": float(rms[k])}
                          for k, index in enumerate(indices)]
        seconds = self.action_seconds()
        if seconds is not None:
            # how long each action took: the mean over the ranks (their share of the step) and
            # the slowest rank's time
            ranks = self.rank_columns > 0
            for name in self.field_order:
                for row in rows[name]:
                    column = seconds[ranks, row["index"]]
                    row["seconds_mean"] = float(column.mean()) if column.size else 0.0
                    row["seconds_max"] = float(column.max()) if column.size else 0.0
        return rows


# -- the self-contained page ---------------------------------------------------
def _page() -> str:
    page = resources.files("freecam.pi_cam").joinpath("state_static/globe.html").read_text()
    coastline = resources.files("freecam.pi_cam").joinpath("state_static/coastline_110m.json").read_text()
    return page.replace("/*COASTLINE*/null", coastline.strip())


def _pack(values: np.ndarray, dtype: str) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(values, dtype=dtype).tobytes(), 9)).decode()


def _quantize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = hi - lo if hi > lo else 1.0
    # a non-finite value is carried as the lowest code: the page marks it from the anomaly codes
    values = np.nan_to_num(values, nan=lo, posinf=hi, neginf=lo)
    return np.clip(np.rint((values - lo) / span * 255.0), 0, 255).astype(np.uint8)


def _signed(values: np.ndarray) -> tuple[np.ndarray, float]:
    """``values`` as int8 over a symmetric scale: (codes, scale); value = code / 127 * scale."""

    finite = values[np.isfinite(values)]
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    if scale == 0.0 or not np.isfinite(scale):
        return np.zeros(values.shape, np.int8), 0.0
    # the scale of the finite changes; a non-finite one is carried as 0 (the anomaly codes say where)
    values = np.nan_to_num(values, nan=0.0, posinf=scale, neginf=-scale)
    return np.clip(np.rint(values / scale * 127.0), -127, 127).astype(np.int8), scale


def snapshot_data(data: StateData, *, fields: list[str] | None = None,
                  levels: list[int] | dict[str, list[int] | None] | None = None,
                  step_stride: int = 1, action_steps: list[int] | None = None) -> dict[str, Any]:
    """The data one self-contained page embeds: for each field and level, every ``step_stride``-th
    step frame (8-bit over the field-level's range) and, at ``action_steps``, the state as the
    step began and what each action changed (8-bit, symmetric, each action's own scale).
    ``levels`` is one list for every field, or a list (None: all) per field name."""

    fields = list(fields or data.field_order)
    for field in fields:
        if field not in data.fields:
            raise KeyError(f"no field {field!r}; the run recorded {data.field_order}")
    step_indices = list(range(0, len(data.step_frames), max(int(step_stride), 1)))
    wanted_steps = sorted(data.action_steps) if action_steps is None else [int(s) for s in action_steps]
    missing = [s for s in wanted_steps if s not in data.action_steps]
    if missing:
        raise ValueError(f"steps {missing} have no action frames (recorded: {sorted(data.action_steps)})")
    embedded: dict[str, Any] = {"step_indices": step_indices, "fields": {}, "action_steps": {}}
    for field in fields:
        nlev = int(data.fields[field]["levels"])
        wanted = levels.get(field) if isinstance(levels, dict) else levels
        chosen = list(range(nlev)) if wanted is None else [lev for lev in wanted if 0 <= lev < nlev] or [nlev - 1]
        # every frame read once: (frames, chosen levels, columns)
        stack = (np.stack([data.values("steps", i, field)[chosen] for i in step_indices], axis=0)
                 if step_indices else np.zeros((0, len(chosen), len(data.lat)), np.float32))
        per_level = {}
        for j, lev in enumerate(chosen):
            block = stack[:, j, :]
            finite = block[np.isfinite(block)]
            lo = float(np.percentile(finite, 1)) if finite.size else 0.0
            hi = float(np.percentile(finite, 99)) if finite.size else 1.0
            per_level[str(lev)] = {"lo": lo, "hi": hi, "steps": _pack(_quantize(block, lo, hi), "u1")}
        embedded["fields"][field] = {"levels": chosen, "data": per_level}
    for step in wanted_steps:
        indices = data.action_steps[step]
        block: dict[str, Any] = {"names": [data.action_frames[i][1] for i in indices],
                                 "owners": [data.action_owners[i] for i in indices], "fields": {}}
        for field in fields:
            chosen = embedded["fields"][field]["levels"]
            frames = np.stack([data.frame("actions", i, field)[chosen].astype(np.float64) for i in indices], axis=0)
            scale = float(data.fields[field]["scale"])
            start = (frames[0] * scale).astype(np.float32)
            changes = np.diff(frames, axis=0, prepend=frames[:1]) * scale      # the first frame changes nothing
            levels_block = {}
            for j, lev in enumerate(chosen):
                codes, scales = zip(*(_signed(changes[k, j]) for k in range(len(indices))))
                # the state as the step began, exactly; the state after action k is it plus the
                # first k changes (each 8-bit over its own scale)
                levels_block[str(lev)] = {"start": _pack(start[j], "<f4"), "scales": list(scales),
                                          "changes": _pack(np.stack(codes), "i1")}
            block["fields"][field] = {"levels": levels_block, "totals": data.changes(step, field)}
        embedded["action_steps"][str(step)] = block
    # anomalies: each embedded field's code per column at the carried step frames and after every
    # action of the carried action steps (0 none, 1 non-finite, 2 negative, 3 out of range), with the
    # summaries that name the action that made them
    step_codes = data.step_codes()
    embedded["anomalies"] = {
        "steps": {"summary": data.step_anomalies(),
                  "codes": {f: _pack(step_codes[f][step_indices], "u1") for f in fields}},
        "action_steps": {str(step): {"summary": data.anomalies(step),
                                     "codes": {f: _pack(data.action_codes(step)[f], "u1") for f in fields}}
                         for step in wanted_steps}}
    return embedded


def snapshot_html(data: StateData, *, label: str | None = None, **options: Any) -> str:
    meta = data.meta()
    if label:
        meta["label"] = str(label)
    payload = _finite({"meta": meta, "snapshot": snapshot_data(data, **options)})
    return _page().replace("/*STATE_DATA*/null", json.dumps(payload, separators=(",", ":"), allow_nan=False))


# -- the server ---------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    data: StateData
    token: str
    lock: threading.Lock
    checked = 0.0

    def log_message(self, *args: Any) -> None:     # quiet
        pass

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _refresh(self) -> None:
        cls = type(self)
        with cls.lock:
            now = time.monotonic()
            if now - cls.checked > 2.0:
                cls.checked = now
                if cls.data.changed():
                    cls.data.reload()

    def do_GET(self) -> None:  # noqa: N802 - the http.server interface
        url = urlparse(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        if query.get("token", "") != self.token:
            self._send(b"forbidden: open the address the command printed", "text/plain", HTTPStatus.FORBIDDEN)
            return
        if url.path == "/":
            self._send(_page().encode(), "text/html; charset=utf-8")
            return
        self._refresh()
        status, body, content_type = respond(type(self).data, url.path, query)
        self._send(body, content_type, status)


def respond(data: StateData, path: str, query: Mapping[str, str]) -> tuple[HTTPStatus, bytes, str]:
    """One request of the page's API (``/api/meta``, ``/api/level`` ...) as (status, body,
    content type): what ``freecam globe`` serves, and what the Workflow Builder serves under
    ``/globe/`` for the run it started."""

    def as_json(payload: Any) -> tuple[HTTPStatus, bytes, str]:
        return (HTTPStatus.OK, json.dumps(_finite(payload), separators=(",", ":"), allow_nan=False).encode(),
                "application/json")

    try:
        kind, field = query.get("kind", "steps"), query.get("field", "")
        index = int(query.get("i", 0))
        if path == "/api/meta":
            return as_json(data.meta())
        if path == "/api/level":
            values = data.level(kind, index, field, int(query["lev"]), query.get("mode", "value"))
            return HTTPStatus.OK, values.tobytes(), "application/octet-stream"
        if path == "/api/profile":
            return as_json(data.profile(kind, index, field, int(query["col"])))
        if path == "/api/section":
            return as_json(data.section(kind, index, field, float(query["lon"]), query.get("mode", "value")))
        if path == "/api/changes":
            return as_json(data.changes(int(query["step"]), field))
        if path == "/api/anomalies":
            return as_json(data.anomalies(int(query["step"])) if query.get("step") not in (None, "")
                           else data.step_anomalies())
        if path == "/api/anomaly_mask":
            return (HTTPStatus.OK, data.anomaly_mask(kind, index, field).astype(np.uint8).tobytes(),
                    "application/octet-stream")
        if path == "/api/trace":
            return as_json(data.trace(int(query["step"]), int(query["col"]),
                                      None if query.get("lev") in (None, "") else int(query["lev"])))
        return HTTPStatus.NOT_FOUND, b"not found", "text/plain"
    except (KeyError, ValueError) as error:
        return HTTPStatus.BAD_REQUEST, str(error).encode(), "text/plain"


def serve(directory: str | Path, *, host: str = "127.0.0.1", port: int = 0, open_browser: bool = False) -> None:
    data = StateData(directory)
    token = secrets.token_urlsafe(16)
    handler = type("StateHandler", (_Handler,), {"data": data, "token": token, "lock": threading.Lock()})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{server.server_address[1]}/?token={token}"
    print(f"freeCAM globe of {data.directory} ({len(data.step_frames)} step frames, "
          f"{len(data.action_frames)} action frames, fields {', '.join(data.field_order)}): {url}", flush=True)
    print("forward the port to your machine if this is a remote host (ssh -L, or the editor's port forwarding)",
          flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _ints(text: str | None) -> list[int] | None:
    if text is None or text.strip() in ("", "all"):
        return None
    return [int(item) for item in text.split(",") if item.strip()]


def _levels(text: str | None) -> list[int] | dict[str, list[int] | None] | None:
    """``14,20,23`` for every field, or ``T=14,20,23;CLDLIQ=all`` field by field (others: all)."""

    if text is None or "=" not in text:
        return _ints(text)
    chosen: dict[str, list[int] | None] = {}
    for part in text.split(";"):
        if part.strip():
            name, _, value = part.partition("=")
            chosen[name.strip()] = _ints(value)
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freecam globe", description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="a state directory (--state-dir of a run)")
    parser.add_argument("--html", type=Path, help="write one self-contained page here instead of serving")
    parser.add_argument("--fields", help="with --html: the fields to embed (default every recorded field)")
    parser.add_argument("--levels", help="with --html: level indices to embed, from 0 at the top (default all); "
                                         "one list for every field, or T=14,20,23;CLDLIQ=all")
    parser.add_argument("--step-stride", type=int, default=1, help="with --html: embed every Nth step frame")
    parser.add_argument("--action-steps", help="with --html: the action steps to embed (default all recorded)")
    parser.add_argument("--label", help="with --html: what the page's header calls the run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--open", action="store_true", help="open a browser")
    arguments = parser.parse_args(argv)
    if arguments.html is not None:
        data = StateData(arguments.directory)
        fields = None if arguments.fields is None else [f.strip() for f in arguments.fields.split(",") if f.strip()]
        page = snapshot_html(data, fields=fields, levels=_levels(arguments.levels),
                             step_stride=arguments.step_stride, action_steps=_ints(arguments.action_steps),
                             label=arguments.label)
        arguments.html.write_text(page)
        print(f"wrote {arguments.html} ({len(page) / 1e6:.1f} MB)")
        return 0
    serve(arguments.directory, host=arguments.host, port=arguments.port, open_browser=arguments.open)
    return 0


def owner_text(owner: Mapping[str, Any] | None) -> str:
    """What computed an action, in words."""

    if not owner:
        return "not recorded"
    by = owner.get("by")
    if by == "fortran":
        return "original Fortran"
    if by == "coupler":
        return "coupler exchange"
    if by in ("io", "clock", "service"):
        return {"io": "output", "clock": "clock", "service": "state service"}[by]
    if by == "python":
        return "Python process"
    if by == "python-stage":
        slots = owner.get("kernels") or {}
        if not slots:
            return "original Fortran, whole, through its Python stage class"
        parts = []
        for kernel, slot in slots.items():
            kind = slot.get("by")
            if kind == "ml":
                parts.append(f"{kernel} by an ML model ({slot.get('file')}, {slot.get('device')})")
            elif kind == "plugin":
                parts.append(f"{kernel} by a compiled plugin")
            elif kind == "fortran-through-python":
                parts.append(f"{kernel} by the original, through Python")
            else:
                parts.append(f"{kernel} by a Python function")
        return "original Fortran, except " + "; ".join(parts)
    return str(by)


def trace_text(trace: Mapping[str, Any], *, show_all: bool = False, relative: float = 1e-3) -> str:
    """A column's trace as text: the state as the step begins, then each action that changed it.

    An action is listed when it changed some field at this level by at least ``relative`` of that
    field's largest change there over the step (every action with ``show_all``); the others are
    counted, apart from the ones that changed nothing here at all.  A change no larger than the
    field's ``noise`` (float round-off, a denormal) counts as none.  An action after which the column
    became anomalous (non-finite, negative water, out of range) is always listed and marked."""

    lat, lon = trace["lat"], ((trace["lon"] + 180.0) % 360.0) - 180.0
    place = f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
    where = "land" if trace["land"] > 0.5 else "sea"
    pressure = f", about {trace['pressure_hpa']:.0f} hPa" if trace.get("pressure_hpa") else ""
    fields = [f["name"] for f in trace["fields"]]
    units = {f["name"]: f["units"] for f in trace["fields"]}
    noise = {f["name"]: float(f.get("noise", 0.0)) for f in trace["fields"]}
    rows = trace["rows"]
    # decimals by each field's magnitude over the step, and each field's largest change here
    decimals = {}
    largest = {}
    for name in fields:
        top = max((abs(row["values"][name]) for row in rows if row["values"][name] is not None), default=0.0)
        decimals[name] = 2 if top >= 100 else 3 if top >= 1 else 4 if top >= 1e-3 else None   # None: 4 significant digits
        largest[name] = max((abs(row["change"][name]) for row in rows[1:] if row["change"][name] is not None), default=0.0)
        if largest[name] <= noise[name]:
            largest[name] = 0.0                                          # nothing but noise changed it

    def state(row: Mapping[str, Any]) -> str:
        def shown(name: str) -> str:
            value = row["values"][name]
            if value is None:
                return "non-finite"
            return (f"{value:.4g}" if decimals[name] is None else f"{value:.{decimals[name]}f}") + f" {units[name]}"

        return "   ".join(f"{name} = {shown(name)}" for name in fields)

    def change(row: Mapping[str, Any], still: set[str] = frozenset()) -> str:
        parts = []
        for name in fields:
            value = row["change"][name]
            shown = ("still non-finite" if name in still else "non-finite" if value is None else "0" if value == 0.0
                     else "≈0" if abs(value) <= noise[name] or abs(value) < relative * largest[name]
                     else f"{value:+.3g}")
            parts.append(f"Δ{name} {shown}" + ("" if "non-finite" in shown else f" {units[name]}"))
        return "   ".join(parts)

    lines = [f"step {trace['step']}, the column at {place} ({where}), level {trace['level'] + 1}{pressure}", "",
             f"  {state(rows[0])}"]
    small = untouched = elsewhere = 0
    if rows[0].get("anomaly"):
        lines.append("  ⚠ already as the step begins: " + ", ".join(f"{f} {k}" for f, k in rows[0]["anomaly"].items()))
    for k, row in enumerate(rows[1:], start=1):
        became = row.get("became") or {}
        # a value non-finite before and after the action: still non-finite, no reason to list it
        still = {name for name in fields if row["values"][name] is None and rows[k - 1]["values"][name] is None}
        listed = show_all or bool(became) or any(
            name not in still and (row["change"][name] is None
                                   or (largest[name] > 0.0 and abs(row["change"][name]) >= relative * largest[name]))
            for name in fields)
        if not listed:
            if any(row["change"][name] != 0.0 for name in fields if name not in still):
                small += 1
            else:
                untouched += 1
                elsewhere += any((row["column_max_change"][name] or 0.0) > 0.0 for name in fields)
            continue
        took = f", {row['seconds_mean'] * 1e3:.1f} ms" if "seconds_mean" in row else ""
        lines += ["", f"        ↓ {row['name']}   ({owner_text(row.get('owner'))}{took})",
                  f"  {state(row)}", f"      → {change(row, still)}"]
        if became:
            lines.append("      ⚠ the column became " + ", ".join(f"{kind} in {field}" for field, kind in became.items())
                         + " here (at some level)")
    notes = []
    if small:
        notes.append(f"{small} more changed this level by less than {relative:.1%} of its largest change")
    if untouched:
        extra = f", {elsewhere} of them changing other levels of the column" if elsewhere else ""
        notes.append(f"{untouched} left it unchanged{extra}")
    if notes:
        lines += ["", "(" + "; ".join(notes) + ")"]
    first, last = rows[0], rows[-1]
    def total(name: str) -> str:
        if last["values"][name] is None or first["values"][name] is None:
            return "non-finite"
        value = last["values"][name] - first["values"][name]
        return ("0" if value == 0.0 else "≈0" if abs(value) <= noise[name] or abs(value) < relative * largest[name]
                else f"{value:+.3g}")

    lines.append("over the step: " + "   ".join(f"Δ{name} {total(name)}" + ("" if total(name) == "non-finite" else f" {units[name]}")
                                              for name in fields))
    return "\n".join(lines)


def trace_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freecam trace",
                                     description="One column through one recorded action step: what each action changed.")
    parser.add_argument("directory", type=Path, help="a state directory (--state-dir of a run)")
    parser.add_argument("--lat", type=float, required=True, help="degrees north (south negative)")
    parser.add_argument("--lon", type=float, required=True, help="degrees east (west negative)")
    parser.add_argument("--step", type=int, help="an action step of the run (default the first recorded)")
    placing = parser.add_mutually_exclusive_group()
    placing.add_argument("--level", type=int, help="the model level, 1 at the top (default: where the first field "
                                                  "changed most over the step)")
    placing.add_argument("--hpa", type=float, help="the level nearest this mean pressure")
    parser.add_argument("--fields", help="the fields to trace (default every recorded field)")
    parser.add_argument("--all", action="store_true", help="list every action, the ones that changed nothing too")
    parser.add_argument("--relative", type=float, default=1e-3,
                        help="list an action when it changed a field at this level by at least this share of the "
                             "field's largest change over the step (default 0.001)")
    parser.add_argument("--json", action="store_true", help="the trace as JSON")
    arguments = parser.parse_args(argv)
    data = StateData(arguments.directory)
    if not data.action_steps:
        raise SystemExit(f"{arguments.directory} recorded no action steps (--state-action-steps)")
    step = sorted(data.action_steps)[0] if arguments.step is None else arguments.step
    level = None
    if arguments.level is not None:
        level = arguments.level - 1
    elif arguments.hpa is not None:
        level = int(np.argmin(np.abs(data.level_pressure / 100.0 - arguments.hpa)))
    fields = None if arguments.fields is None else [f.strip() for f in arguments.fields.split(",") if f.strip()]
    trace = data.trace(step, data.nearest(arguments.lat, arguments.lon), level, fields)
    print(json.dumps(_finite(trace), indent=1) if arguments.json
          else trace_text(trace, show_all=arguments.all, relative=arguments.relative))
    return 0


def anomalies_text(data: StateData, steps: list[int] | None = None) -> str:
    """Where the recorded fields went wrong: the first step frame each field had an anomalous
    column, and in every action step the first action that made each kind of anomaly."""

    rules = "; ".join(f"{name}: {anomaly_rule(name)}" for name in data.field_order)
    lines = [f"anomaly rules -- {rules}", ""]
    summary = data.step_anomalies()
    if summary["first"]:
        for f in summary["first"]:
            everywhere = (f", every column from step {f['every_column_from']}" if f["every_column_from"] is not None else "")
            lines.append(f"steps: {f['field']} {' and '.join(f['kinds'])} first at the end of step {f['step']} "
                         f"({f['columns']} columns{everywhere})")
    else:
        lines.append(f"steps: no anomalous column at the end of any of the {len(data.step_frames)} recorded steps")
    # consecutive steps that only begin anomalous, the same fields the same way, are one line
    carried: list[tuple[int, int, str]] = []

    def flush() -> None:
        if carried:
            first_step, last_step, what = carried[0][0], carried[-1][0], carried[0][2]
            span = f"{first_step}" if first_step == last_step else f"{first_step}-{last_step}"
            lines.append(f"action step{'s' if first_step != last_step else ''} {span}: {what} already as the step begins"
                         f" (up to {max(n for _, n, _ in carried)} columns); no action made a new kind of anomaly")
            carried.clear()

    for step in (sorted(data.action_steps) if steps is None else steps):
        report = data.anomalies(step)
        if report["first"] and all(f["at_start"] for f in report["first"]):
            what = ", ".join(f"{f['field']} {f['kind']}" for f in report["first"])
            if carried and carried[-1][2] != what:
                flush()
            carried.append((step, max(f["columns"] for f in report["first"]), what))
            continue
        flush()
        if not report["first"]:
            lines.append(f"action step {step}: no action made a column anomalous")
            continue
        for f in report["first"]:
            example = ", ".join(f"{c} ({data.lat[c]:.1f}, {data.lon[c]:.1f})" for c in f["example_columns"][:3])
            if f["at_start"]:
                lines.append(f"action step {step}: {f['field']} {f['kind']} already as the step begins: {f['columns']} columns")
            else:
                lines.append(f"action step {step}: {f['field']} became {f['kind']} after {f['name']} "
                             f"({owner_text(f['owner'])}): {f['columns']} columns, e.g. {example}")
    flush()
    return "\n".join(lines)


def anomalies_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freecam anomalies",
                                     description="Where a recorded run went wrong: non-finite values, negative water or "
                                                 "constituents, temperature out of range, and the action that made them.")
    parser.add_argument("directory", type=Path, help="a state directory (--state-dir of a run)")
    parser.add_argument("--step", type=int, action="append", help="an action step to report (default every one)")
    parser.add_argument("--json", action="store_true", help="the reports as JSON")
    arguments = parser.parse_args(argv)
    data = StateData(arguments.directory)
    if arguments.json:
        steps = sorted(data.action_steps) if arguments.step is None else arguments.step
        print(json.dumps(_finite({"rules": {n: anomaly_rule(n) for n in data.field_order},
                                  "steps": data.step_anomalies(), "action_steps": [data.anomalies(s) for s in steps]}),
                         indent=1))
    else:
        print(anomalies_text(data, arguments.step))
    return 0


__all__ = ["ANOMALY_KINDS", "StateData", "anomalies_main", "anomalies_text", "anomaly_codes", "anomaly_rule", "main",
           "owner_text", "serve", "snapshot_data", "snapshot_html", "trace_main", "trace_text"]
