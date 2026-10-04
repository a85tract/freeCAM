"""Snapshots of chosen physics-state fields of a PI-CAM run, for the globe viewer.

Off unless asked for (``--state-dir``).  When on, every rank copies the chosen
fields of its own columns -- temperature, water vapour, cloud liquid and ice by
default -- at the end of every ``every``-th step, and, at the steps named in
``action_steps``, also at the start of the step and after every plan action, so
the viewer can show what each process did.  The copies are read-only (float32
views of this rank's real columns); nothing the model computes is touched.
Each rank appends its snapshots to its own files every ``flush_every``
snapshots; nothing is sent between ranks while the model steps.  The only
collectives are one barrier at the start, one gather of the column
coordinates, one reduction of the level pressures after initialization, and one
gather of the land fraction after the first step.

Layout of a state directory::

    manifest.json               rank 0: fields (name, units, levels), frames written, complete
    columns.npz                 rank 0: each column's rank, lat, lon (degrees); level pressures
    surface.npz                 rank 0: each column's land fraction (after the first step)
    ranks/rank-NNNN.steps.bin   step snapshots, float32, appended
    ranks/rank-NNNN.actions.bin action snapshots, float64, appended
    ranks/rank-NNNN.action_seconds.bin  each action frame's wall time on the rank, float64
                                (0 for the start of a step; the copy is not counted)

Each action frame of the manifest is ``[step, action, owner]``: owner says what computed the
action (the original Fortran, the coupler exchange, output, a Python process, or a Python stage
class with what stood in its replaced kernel slots), as rank 0 saw it.

One snapshot of one rank is every field in manifest order, each ``(levels,
columns)`` in C order (a level of a rank is contiguous), so snapshot ``i`` of a
rank with ``n`` columns starts at value ``i * n * sum(levels)``.  Action
snapshots keep the model's float64: what one process changes in one step can be
smaller than float32 resolves at the field's magnitude (2e-5 K at 280 K).  The
frames' steps (and, for action snapshots, their action names) are in the
manifest; every rank records the same frames.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1

START_OF_STEP = "start of step"

#: the stored precision of each kind of snapshot
DTYPES = {"steps": np.float32, "actions": np.float64}


@dataclass(frozen=True)
class FieldSpec:
    """One recordable field: where it lives in the state pool and how to show it."""

    name: str                       # the viewer's name (T, Q, CLDLIQ, ...)
    field: str                      # the state-pool field
    constituent: str | int | None   # an index into the constituent axis of phys_state.q, by name or number
    units: str                      # the units the viewer shows
    scale: float                    # stored value x scale = shown value
    label: str


#: the fields a short name stands for; any ``phys_state.<name>`` (2-D or 3-D) and any
#: ``phys_state.q:<constituent>`` may be named as well
KNOWN_FIELDS: dict[str, FieldSpec] = {
    "T": FieldSpec("T", "phys_state.t", None, "K", 1.0, "temperature"),
    "Q": FieldSpec("Q", "phys_state.q", "Q", "g/kg", 1.0e3, "water vapour"),
    "CLDLIQ": FieldSpec("CLDLIQ", "phys_state.q", "CLDLIQ", "mg/kg", 1.0e6, "cloud liquid water"),
    "CLDICE": FieldSpec("CLDICE", "phys_state.q", "CLDICE", "mg/kg", 1.0e6, "cloud ice"),
    "U": FieldSpec("U", "phys_state.u", None, "m/s", 1.0, "zonal wind"),
    "V": FieldSpec("V", "phys_state.v", None, "m/s", 1.0, "meridional wind"),
    "OMEGA": FieldSpec("OMEGA", "phys_state.omega", None, "Pa/s", 1.0, "vertical pressure velocity"),
    "PS": FieldSpec("PS", "phys_state.ps", None, "hPa", 1.0e-2, "surface pressure"),
}

DEFAULT_FIELDS = ("T", "Q", "CLDLIQ", "CLDICE")


def field_spec(name: str) -> FieldSpec:
    """``T``, ``phys_state.omega`` or ``phys_state.q:CLDLIQ`` -> the field's spec."""

    name = str(name).strip()
    if name in KNOWN_FIELDS:
        return KNOWN_FIELDS[name]
    field, _, constituent = name.partition(":")
    if not field.startswith("phys_state."):
        raise ValueError(f"cannot record {name!r}: name one of {sorted(KNOWN_FIELDS)}, a phys_state.<field>, "
                         f"or phys_state.q:<constituent>")
    if constituent:
        index: str | int = int(constituent) if constituent.isdigit() else constituent
        return FieldSpec(name, field, index, "kg/kg", 1.0, f"{field} {constituent}")
    return FieldSpec(name, field, None, "", 1.0, field)


def constituent_names(library: Any, count: int) -> list[str]:
    """The image's constituent names (``constituents::cnst_name``), or an empty list when the image
    has no such symbol."""

    if library is None or count <= 0:
        return []
    try:
        from ..physics.image import module_view

        names = module_view(library, "constituents_mp_cnst_name_", "S16", (int(count),))
    except Exception:
        return []
    return [bytes(item).decode("ascii", errors="replace").strip() for item in np.asarray(names).reshape(-1)]


def _real_columns(pool: Mapping[str, Any]) -> list[int] | None:
    for name in ("phys_state.ncol", "grid.chunk_ncols"):
        try:
            return [int(n) for n in np.asarray(pool[name], dtype=np.int64).reshape(-1)]
        except (KeyError, TypeError):
            continue
    return None


def _take(array: np.ndarray, ncol: Sequence[int]) -> np.ndarray:
    """A (pcols, chunks) field over this rank's real columns: each chunk's first ncol, in chunk order."""

    parts = [array[: int(n), c] for c, n in enumerate(ncol) if int(n) > 0]
    return np.concatenate(parts) if parts else np.zeros(0)


class _Resolved:
    """A field spec bound to this rank's pool: which array, which constituent, how many levels."""

    def __init__(self, spec: FieldSpec, pool: Mapping[str, Any], constituents: Sequence[str]) -> None:
        self.spec = spec
        try:
            array = np.asarray(pool[spec.field])
        except KeyError:
            raise ValueError(f"cannot record {spec.name!r}: the state pool has no {spec.field!r}") from None
        self.index: int | None = None
        if spec.constituent is not None:
            if array.ndim != 4:
                raise ValueError(f"cannot record {spec.name!r}: {spec.field} has no constituent axis")
            if isinstance(spec.constituent, int):
                self.index = spec.constituent
            elif spec.constituent in constituents:
                self.index = list(constituents).index(spec.constituent)
            else:
                raise ValueError(f"cannot record {spec.name!r}: the image names no constituent "
                                 f"{spec.constituent!r} (it has {list(constituents)[:12]}...)")
            if not 0 <= self.index < array.shape[2]:
                raise ValueError(f"cannot record {spec.name!r}: constituent {self.index} is out of range")
            self.levels = int(array.shape[1])
        elif array.ndim == 3:
            self.levels = int(array.shape[1])
        elif array.ndim == 2:
            self.levels = 1
        else:
            raise ValueError(f"cannot record {spec.name!r}: {spec.field} is {array.ndim}-D; a column field "
                             f"is (pcols, chunks) or (pcols, levels, chunks)")

    def values(self, pool: Mapping[str, Any], ncol: Sequence[int], dtype: Any = np.float32) -> np.ndarray:
        """This rank's real columns, (levels, columns), as ``dtype`` (a copy)."""

        array = np.asarray(pool[self.spec.field])
        if self.index is not None:
            array = array[:, :, self.index, :]
        if array.ndim == 2:
            array = array[:, None, :]
        # (pcols, levels, chunks) -> (levels, columns): each chunk's first ncol columns, in order
        parts = [array[: int(n), :, c] for c, n in enumerate(ncol) if int(n) > 0]
        block = np.concatenate(parts, axis=0) if parts else np.zeros((0, array.shape[1]))
        return np.ascontiguousarray(block.T, dtype=dtype)


class StateRecorder:
    """One rank's state snapshots, appended to ``directory/ranks/rank-NNNN.*.bin``."""

    def __init__(self, directory: str | Path, *, rank: int, size: int, comm: Any | None = None,
                 fields: Iterable[str] = DEFAULT_FIELDS, every: int = 1, action_steps: Iterable[int] = (),
                 flush_every: int = 24, run_label: str = "") -> None:
        if every < 1:
            raise ValueError("every must be at least 1")
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.directory = Path(directory)
        self.rank = int(rank)
        self.size = int(size)
        self.comm = comm
        self.specs = [field_spec(name) for name in fields]
        if not self.specs:
            raise ValueError("name at least one field to record")
        names = [spec.name for spec in self.specs]
        if len(set(names)) != len(names):
            raise ValueError(f"a field is named twice: {names}")
        self.every = int(every)
        self.action_steps = frozenset(int(step) for step in action_steps)
        if any(step < 0 for step in self.action_steps):
            raise ValueError("action steps count from 0")
        self.flush_every = int(flush_every)
        self.run_label = str(run_label)
        self._resolved: list[_Resolved] = []
        self._ncol: list[int] = []
        self._pending = {"steps": [], "actions": []}
        self._pending_seconds: list[float] = []
        self._frames: dict[str, list] = {"steps": [], "actions": []}
        self._written = {"steps": 0, "actions": 0}
        self._last_step = -1
        self._started = False
        self._described = False
        self._closed = False

    # -- paths -------------------------------------------------------------
    def rank_file(self, kind: str) -> Path:
        return self.directory / "ranks" / f"rank-{self.rank:04d}.{kind}.bin"

    @property
    def manifest_file(self) -> Path:
        return self.directory / "manifest.json"

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Create the directory and truncate this rank's files (collectively: all ranks, then on)."""

        (self.directory / "ranks").mkdir(parents=True, exist_ok=True)
        for kind in ("steps", "actions", "action_seconds"):
            self.rank_file(kind).write_bytes(b"")
        barrier = getattr(self.comm, "Barrier", None)
        if callable(barrier):
            barrier()
        self._started = True

    def describe(self, pool: Mapping[str, Any], library: Any = None) -> None:
        """Bind the fields to this rank's pool, gather the column coordinates and the level
        pressures to rank 0 (after initialization, collectively) and write the manifest."""

        q = pool.get("phys_state.q", np.zeros((0, 0, 0, 0))) if hasattr(pool, "get") else np.zeros((0, 0, 0, 0))
        pcnst = int(np.asarray(q).shape[2]) if np.asarray(q).ndim == 4 else 0
        constituents = constituent_names(library, pcnst)
        self._resolved = [_Resolved(spec, pool, constituents) for spec in self.specs]
        ncol = _real_columns(pool)
        if ncol is None:
            raise ValueError("the state pool has no phys_state.ncol: cannot tell real columns from padding")
        self._ncol = ncol
        # the coordinates by the same slicing as every snapshot's values: column i of a rank's
        # snapshot is column i here
        lat = _take(np.asarray(pool["phys_state.lat"], dtype=np.float64), ncol)
        lon = _take(np.asarray(pool["phys_state.lon"], dtype=np.float64), ncol)
        pressure = self._level_pressure(pool, ncol)
        gather = getattr(self.comm, "gather", None)
        gathered = gather((lat, lon), root=0) if callable(gather) else [(lat, lon)]
        self._described = True
        if self.rank != 0:
            return
        ranks = [np.full(len(rank_lat), r, dtype=np.int32) for r, (rank_lat, _) in enumerate(gathered)]
        lat_all = np.concatenate([np.asarray(v[0], dtype=np.float64) for v in gathered])
        lon_all = np.concatenate([np.asarray(v[1], dtype=np.float64) for v in gathered])
        if lat_all.size and float(np.nanmax(np.abs(lat_all))) <= np.pi / 2 + 1e-6:
            lat_all, lon_all = np.degrees(lat_all), np.degrees(lon_all)
        np.savez(self.directory / "columns.npz", rank=np.concatenate(ranks), lat=lat_all, lon=lon_all,
                 level_pressure_pa=pressure)
        self._constituents = constituents
        self._write_manifest(complete=False)

    def _level_pressure(self, pool: Mapping[str, Any], ncol: Sequence[int]) -> np.ndarray:
        """The mean midpoint pressure of each level over every column (one reduction)."""

        try:
            pmid = np.asarray(pool["phys_state.pmid"], dtype=np.float64)
        except KeyError:
            return np.zeros(0)
        if pmid.ndim != 3:
            return np.zeros(0)
        parts = [pmid[: int(n), :, c] for c, n in enumerate(ncol) if int(n) > 0]
        block = np.concatenate(parts, axis=0) if parts else np.zeros((0, pmid.shape[1]))
        local = np.concatenate([block.sum(axis=0), [float(block.shape[0])]])
        allreduce = getattr(self.comm, "allreduce", None)
        total = allreduce(local) if callable(allreduce) else local
        total = np.asarray(total, dtype=np.float64)
        return total[:-1] / max(total[-1], 1.0)

    # -- recording ---------------------------------------------------------
    def wants_actions(self, step: int) -> bool:
        return self._described and int(step) in self.action_steps

    def _snapshot(self, pool: Mapping[str, Any], kind: str = "steps") -> np.ndarray:
        dtype = DTYPES[kind]
        return np.concatenate([field.values(pool, self._ncol, dtype).reshape(-1) for field in self._resolved])

    def step_start(self, step: int, pool: Mapping[str, Any]) -> None:
        """The state as the step begins (at an action step): what the first process starts from."""

        if self.wants_actions(step):
            self._pending["actions"].append(self._snapshot(pool, "actions"))
            self._pending_seconds.append(0.0)
            self._frames["actions"].append([int(step), START_OF_STEP, None])

    def after_action(self, step: int, name: str, pool: Mapping[str, Any], seconds: float = 0.0,
                     owner: Mapping[str, Any] | None = None) -> None:
        """The state after one plan action (at an action step), the action's wall time, and what
        computed it."""

        if self.wants_actions(step):
            self._pending["actions"].append(self._snapshot(pool, "actions"))
            self._pending_seconds.append(float(seconds))
            self._frames["actions"].append([int(step), str(name), None if owner is None else dict(owner)])

    def step_done(self, step: int, pool: Mapping[str, Any]) -> None:
        """The state at the end of a step: recorded every ``every`` steps."""

        if not self._described:
            return
        if int(step) % self.every == 0:
            self._pending["steps"].append(self._snapshot(pool))
            self._frames["steps"].append(int(step))
        self._last_step = max(self._last_step, int(step))
        if int(step) == 0:
            self.write_surface(pool)
        if len(self._pending["steps"]) >= self.flush_every or len(self._pending["actions"]) >= 4 * self.flush_every:
            self.flush()

    def write_surface(self, pool: Mapping[str, Any]) -> None:
        """Gather every column's land fraction to rank 0 (one collective, once the first
        boundary import has filled it) and write ``surface.npz``."""

        try:
            landfrac = np.asarray(pool["cam_in.landfrac"], dtype=np.float64)
        except KeyError:
            landfrac = None
        if landfrac is not None and landfrac.ndim == 2 and landfrac.shape[1] == len(self._ncol):
            values = _take(landfrac, self._ncol)
        else:
            values = np.zeros(0)
        gather = getattr(self.comm, "gather", None)
        gathered = gather(values, root=0) if callable(gather) else [values]
        if self.rank != 0:
            return
        land = np.concatenate([np.asarray(v, dtype=np.float64) for v in gathered]) if gathered else np.zeros(0)
        np.savez(self.directory / "surface.npz", landfrac=np.where(np.isfinite(land), land, 0.0))

    def flush(self, *, collective: bool = True) -> None:
        """Append the pending snapshots to this rank's files; rank 0 refreshes the manifest.

        Every rank flushes at the same steps, so the flush waits for all of them before
        rank 0 advertises the new frames: a page following the run never reads a frame some
        rank has not written yet.  ``collective=False`` is for a rank failing on its own: it
        saves what it has and touches no other rank and no manifest.
        """

        if not self._started:
            return
        for kind, pending in self._pending.items():
            if pending:
                with self.rank_file(kind).open("ab") as handle:
                    np.concatenate(pending).astype(DTYPES[kind], copy=False).tofile(handle)
                self._written[kind] += len(pending)
                pending.clear()
        if self._pending_seconds:
            with self.rank_file("action_seconds").open("ab") as handle:
                np.asarray(self._pending_seconds, dtype="<f8").tofile(handle)
            self._pending_seconds.clear()
        if not collective:
            return
        barrier = getattr(self.comm, "Barrier", None)
        if callable(barrier) and self.size > 1:
            barrier()
        if self.rank == 0 and self._described:
            self._write_manifest(complete=False)

    def abandon(self) -> None:
        """This rank failed: keep what it recorded, wait for no other rank, mark nothing complete."""

        if self._closed or not self._started:
            return
        self.flush(collective=False)
        self._closed = True

    def close(self) -> None:
        if self._closed or not self._started:
            return
        self.flush()
        self._closed = True
        if self.rank == 0 and self._described:
            self._write_manifest(complete=True)

    # -- metadata ----------------------------------------------------------
    def describe_run(self) -> dict[str, Any]:
        """What this rank recorded, for the run summary."""

        return {"fields": [spec.name for spec in self.specs], "every": self.every,
                "action_steps": sorted(self.action_steps),
                "step_snapshots": self._written["steps"] + len(self._pending["steps"]),
                "action_snapshots": self._written["actions"] + len(self._pending["actions"]),
                "read_only": True}

    def _write_manifest(self, *, complete: bool) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "what": "snapshots of physics-state fields of a PI-CAM run (freecam.pi_cam.state_record)",
            "run": self.run_label,
            "ranks": self.size,
            "dtype": {kind: np.dtype(dtype).str for kind, dtype in DTYPES.items()},
            "layout": "per rank and snapshot: every field in order, each (levels, columns) in C order",
            "fields": [{"name": field.spec.name, "field": field.spec.field,
                        "constituent": field.spec.constituent, "levels": field.levels,
                        "units": field.spec.units, "scale": field.spec.scale, "label": field.spec.label}
                       for field in self._resolved],
            "constituents": getattr(self, "_constituents", []),
            "every": self.every,
            "action_steps": sorted(self.action_steps),
            "action_seconds": True,
            "step_frames": self._frames["steps"][: self._written["steps"]],
            "action_frames": self._frames["actions"][: self._written["actions"]],
            "last_step": self._last_step,
            "complete": bool(complete),
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temporary = self.manifest_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=1) + "\n")
        temporary.replace(self.manifest_file)


def parse_steps(text: str) -> tuple[int, ...]:
    """``"0,24,40-42"`` -> (0, 24, 40, 41, 42); ``"12-48/12"`` -> (12, 24, 36, 48).  Items are
    separated by commas or semicolons (a job's ``qsub -v`` cannot carry a comma)."""

    steps: list[int] = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        span, slash, stride = item.partition("/")
        first, dash, last = span.partition("-")
        every = int(stride) if slash else 1
        if every < 1:
            raise ValueError(f"a step stride is at least 1, not {stride!r}")
        if dash:
            steps.extend(range(int(first), int(last) + 1, every))
        elif slash:
            raise ValueError(f"{item!r}: a stride needs a range, first-last/stride")
        else:
            steps.append(int(item))
    return tuple(sorted(set(steps)))


__all__ = ["DEFAULT_FIELDS", "FieldSpec", "KNOWN_FIELDS", "SCHEMA_VERSION", "START_OF_STEP", "StateRecorder",
           "field_spec", "parse_steps"]
