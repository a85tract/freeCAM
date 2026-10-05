"""Snapshots of chosen physics-state fields of a PI-CAM run, for the globe viewer.

Off unless asked for (``--state-dir``).  When on, every rank copies the chosen
fields of its own columns -- temperature, water vapour, cloud liquid and ice by
default -- at the end of every ``every``-th step, and, at the steps named in
``action_steps``, also at the start of the step and after every plan action, so
the viewer can show what each process did.  The copies are read-only (float32
views of this rank's real columns); nothing the model computes is touched.
Every ``flush_every`` snapshots the ranks send theirs to rank 0, which appends
them, every column in rank order, to one file per kind: one writer, so a page
following the run reads one file a frame instead of one per rank, and the ranks
never touch the file system for a snapshot.  The other collectives are one
barrier at the start, one gather of the column coordinates, one reduction of the
level pressures after initialization, and one gather of the land fraction after
the first step.

Without a directory the snapshots stay in memory, each rank keeping its own columns of the
newest ``keep_steps`` step snapshots and ``keep_actions`` action snapshots: nothing is written
and nothing is sent while the model steps.  A viewer asks for what it shows with
:meth:`StateRecorder.query`, which every rank answers together between two steps (the
Workflow Builder's Globe tab, through the notebook session): one field of one snapshot is
gathered to rank 0, a sum over every column is reduced there.

Layout of a state directory::

    manifest.json        rank 0: fields (name, units, levels), frames written, complete
    columns.npz          rank 0: each column's rank, lat, lon (degrees); level pressures
    surface.npz          rank 0: each column's land fraction (after the first step)
    steps.bin            step snapshots, float32, appended
    actions.bin          action snapshots, float64, appended
    action_seconds.bin   each action frame's wall time on every rank, (frames, ranks), float64
                         (0 for the start of a step; the copy is not counted)

Each action frame of the manifest is ``[step, action, owner]``: owner says what computed the
action (the original Fortran, the coupler exchange, output, a Python process, or a Python stage
class with what stood in its replaced kernel slots), as rank 0 saw it.

One snapshot is every field in manifest order, each ``(levels, columns)`` in C
order over every column (a level is contiguous), so snapshot ``i`` starts at
value ``i * columns * sum(levels)``; the columns are in rank order, as
``columns.npz`` lists them.  Action
snapshots keep the model's float64: what one process changes in one step can be
smaller than float32 resolves at the field's magnitude (2e-5 K at 280 K).  The
frames' steps (and, for action snapshots, their action names) are in the
manifest; every rank records the same frames.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 2

#: snapshots gathered to rank 0 at once, so a write of many keeps its memory bounded
GATHER_CHUNK = 8

START_OF_STEP = "start of step"

#: the stored precision of each kind of snapshot
DTYPES = {"steps": np.float32, "actions": np.float64}

#: in memory: the snapshots each rank keeps by default (4 fields of 30 levels: 13 KB a step
#: snapshot, 26 KB an action snapshot, of a rank's 27 columns at 512 ranks)
KEEP_STEPS = 1000
KEEP_ACTIONS = 2000


@dataclass(frozen=True)
class FieldSpec:
    """One recordable field: where it lives in the state pool and how to show it."""

    name: str                       # the viewer's name (T, Q, CLDLIQ, ...)
    field: str                      # the state-pool field
    constituent: str | int | None   # an index into the field's constituent axis, by name or number
    units: str                      # the units the viewer shows
    scale: float                    # stored value x scale = shown value
    label: str
    group: str = "atmosphere"       # where a picker lists it: atmosphere, surface or tendency


#: the state-pool structures a field may come from: the physics state, what the surface
#: components hand the atmosphere and what it hands them, and the physics tendencies
OWNERS = ("phys_state", "cam_in", "cam_out", "phys_tend")

#: seconds in a day: a rate per second shown per day
DAY = 86400.0


def _surface(name: str, field: str, units: str, scale: float, label: str) -> FieldSpec:
    return FieldSpec(name, field, None, units, scale, label, "surface")


#: the fields a short name stands for (named as CAM's history names them where it has one; each is
#: the state-pool field it names, at the moment of the snapshot); any ``<owner>.<field>`` of
#: OWNERS (2-D or 3-D) and any ``<owner>.<field>:<constituent>`` may be named as well
KNOWN_FIELDS: dict[str, FieldSpec] = {
    "T": FieldSpec("T", "phys_state.t", None, "K", 1.0, "temperature"),
    "Q": FieldSpec("Q", "phys_state.q", "Q", "g/kg", 1.0e3, "water vapour"),
    "CLDLIQ": FieldSpec("CLDLIQ", "phys_state.q", "CLDLIQ", "mg/kg", 1.0e6, "cloud liquid water"),
    "CLDICE": FieldSpec("CLDICE", "phys_state.q", "CLDICE", "mg/kg", 1.0e6, "cloud ice"),
    "NUMLIQ": FieldSpec("NUMLIQ", "phys_state.q", "NUMLIQ", "1/mg", 1.0e-6, "cloud droplet number"),
    "NUMICE": FieldSpec("NUMICE", "phys_state.q", "NUMICE", "1/mg", 1.0e-6, "cloud ice number"),
    "U": FieldSpec("U", "phys_state.u", None, "m/s", 1.0, "zonal wind"),
    "V": FieldSpec("V", "phys_state.v", None, "m/s", 1.0, "meridional wind"),
    "OMEGA": FieldSpec("OMEGA", "phys_state.omega", None, "Pa/s", 1.0, "vertical pressure velocity"),
    "ZM": FieldSpec("ZM", "phys_state.zm", None, "m", 1.0, "height above the surface"),
    "PS": _surface("PS", "phys_state.ps", "hPa", 1.0e-2, "surface pressure"),
    "PSL": _surface("PSL", "cam_out.psl", "hPa", 1.0e-2, "sea-level pressure"),
    "TS": _surface("TS", "cam_in.ts", "K", 1.0, "surface temperature"),
    "TREFHT": _surface("TREFHT", "cam_in.tref", "K", 1.0, "2 m temperature"),
    "QREFHT": _surface("QREFHT", "cam_in.qref", "g/kg", 1.0e3, "2 m specific humidity"),
    "U10": _surface("U10", "cam_in.u10", "m/s", 1.0, "10 m wind speed"),
    "PRECC": _surface("PRECC", "cam_out.precc", "mm/day", 1.0e3 * DAY, "convective precipitation"),
    "PRECL": _surface("PRECL", "cam_out.precl", "mm/day", 1.0e3 * DAY, "large-scale precipitation"),
    "PRECSC": _surface("PRECSC", "cam_out.precsc", "mm/day", 1.0e3 * DAY, "convective snowfall (water)"),
    "PRECSL": _surface("PRECSL", "cam_out.precsl", "mm/day", 1.0e3 * DAY, "large-scale snowfall (water)"),
    "SHFLX": _surface("SHFLX", "cam_in.shf", "W/m2", 1.0, "sensible heat flux, upward"),
    "LHFLX": _surface("LHFLX", "cam_in.lhf", "W/m2", 1.0, "latent heat flux, upward"),
    "FLDS": _surface("FLDS", "cam_out.flwds", "W/m2", 1.0, "longwave down at the surface"),
    "FSNS": _surface("FSNS", "cam_out.netsw", "W/m2", 1.0, "net shortwave at the surface"),
    "TAUX": _surface("TAUX", "cam_in.wsx", "N/m2", 1.0, "zonal surface stress"),
    "TAUY": _surface("TAUY", "cam_in.wsy", "N/m2", 1.0, "meridional surface stress"),
    "ICEFRAC": _surface("ICEFRAC", "cam_in.icefrac", "", 1.0, "sea-ice fraction"),
    "OCNFRAC": _surface("OCNFRAC", "cam_in.ocnfrac", "", 1.0, "ocean fraction"),
    "SNOWHLND": _surface("SNOWHLND", "cam_in.snowhland", "m", 1.0, "snow depth over land (water)"),
    "DTDT": FieldSpec("DTDT", "phys_tend.dtdt", None, "K/day", DAY, "physics temperature tendency", "tendency"),
    "DUDT": FieldSpec("DUDT", "phys_tend.dudt", None, "m/s/day", DAY, "physics zonal wind tendency", "tendency"),
    "DVDT": FieldSpec("DVDT", "phys_tend.dvdt", None, "m/s/day", DAY, "physics meridional wind tendency",
                      "tendency"),
}

DEFAULT_FIELDS = ("T", "Q", "CLDLIQ", "CLDICE")

#: what the Workflow Builder's globe keeps by default: in memory a field costs each rank a few KB a
#: step, so the winds, the surface pressure and the surface's temperature, fluxes and rain as well
BUILDER_FIELDS = ("T", "Q", "CLDLIQ", "CLDICE", "U", "V", "OMEGA", "PS", "TS", "PRECC", "PRECL", "SHFLX", "LHFLX")


def field_spec(name: str) -> FieldSpec:
    """``T``, ``phys_state.omega``, ``cam_in.shf`` or ``phys_state.q:CLDLIQ`` -> the field's spec."""

    name = str(name).strip()
    if name in KNOWN_FIELDS:
        return KNOWN_FIELDS[name]
    field, _, constituent = name.partition(":")
    owner, dot, member = field.partition(".")
    if owner not in OWNERS or not dot or not member.isidentifier():
        raise ValueError(f"cannot record {name!r}: name one of {sorted(KNOWN_FIELDS)}, an <owner>.<field> of "
                         f"{', '.join(OWNERS)}, or <owner>.<field>:<constituent>")
    group = {"cam_in": "surface", "cam_out": "surface", "phys_tend": "tendency"}.get(owner, "atmosphere")
    if constituent:
        index: str | int = int(constituent) if constituent.isdigit() else constituent
        units = "kg/kg" if field == "phys_state.q" else ""
        return FieldSpec(name, field, index, units, 1.0, f"{field} {constituent}", group)
    return FieldSpec(name, field, None, "", 1.0, field, group)


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

    def __init__(self, spec: FieldSpec, pool: Mapping[str, Any], constituents: Sequence[str],
                 pcnst: int = 0) -> None:
        self.spec = spec
        try:
            array = np.asarray(pool[spec.field])
        except KeyError:
            raise ValueError(f"cannot record {spec.name!r}: the state pool has no {spec.field!r}") from None
        self.index: int | None = None
        #: the constituent axis: 2 of (pcols, levels, pcnst, chunks), 1 of a surface field's (pcols, pcnst, chunks)
        self.axis = 2 if array.ndim == 4 else 1
        if spec.constituent is not None:
            if not (array.ndim == 4 or (array.ndim == 3 and pcnst > 0 and array.shape[1] == pcnst)):
                raise ValueError(f"cannot record {spec.name!r}: {spec.field} has no constituent axis")
            if isinstance(spec.constituent, int):
                self.index = spec.constituent
            elif spec.constituent in constituents:
                self.index = list(constituents).index(spec.constituent)
            else:
                raise ValueError(f"cannot record {spec.name!r}: the image names no constituent "
                                 f"{spec.constituent!r} (it has {list(constituents)[:12]}...)")
            if not 0 <= self.index < array.shape[self.axis]:
                raise ValueError(f"cannot record {spec.name!r}: constituent {self.index} is out of range")
            self.levels = int(array.shape[1]) if self.axis == 2 else 1
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
            array = array[:, :, self.index, :] if self.axis == 2 else array[:, self.index, :]
        if array.ndim == 2:
            array = array[:, None, :]
        # (pcols, levels, chunks) -> (levels, columns): each chunk's first ncol columns, in order
        parts = [array[: int(n), :, c] for c, n in enumerate(ncol) if int(n) > 0]
        block = np.concatenate(parts, axis=0) if parts else np.zeros((0, array.shape[1]))
        return np.ascontiguousarray(block.T, dtype=dtype)


class StateRecorder:
    """One rank's state snapshots, gathered to rank 0 and appended to ``directory/*.bin``, or,
    without a directory, kept in this rank's memory for :meth:`query`."""

    def __init__(self, directory: str | Path | None, *, rank: int, size: int, comm: Any | None = None,
                 fields: Iterable[str] = DEFAULT_FIELDS, every: int = 1, action_steps: Iterable[int] = (),
                 flush_every: int = 24, run_label: str = "", keep_steps: int = KEEP_STEPS,
                 keep_actions: int = KEEP_ACTIONS) -> None:
        if every < 1:
            raise ValueError("every must be at least 1")
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        if keep_steps < 1 or keep_actions < 1:
            raise ValueError("keep_steps and keep_actions must be at least 1")
        self.memory = directory is None
        self.directory = None if directory is None else Path(directory)
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
        # in memory: this rank's snapshots, oldest first, and how many were dropped before them
        self.keep_steps, self.keep_actions = int(keep_steps), int(keep_actions)
        self._kept: dict[str, deque] = {"steps": deque(), "actions": deque()}
        self._kept_seconds: deque[float] = deque()
        self._dropped = {"steps": 0, "actions": 0}
        self._columns: dict[str, np.ndarray] | None = None
        self._landfrac: np.ndarray | None = None
        self._complete = False
        self._started = False
        self._described = False
        self._surface = False
        self._closed = False

    # -- paths -------------------------------------------------------------
    def data_file(self, kind: str) -> Path:
        return self.directory / f"{kind}.bin"

    @property
    def manifest_file(self) -> Path:
        return self.directory / "manifest.json"

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Rank 0 creates the directory and empties the files (collectively: all ranks, then on)."""

        if self.memory:
            self._started = True
            return
        if self.rank == 0:
            self.directory.mkdir(parents=True, exist_ok=True)
            for kind in ("steps", "actions", "action_seconds"):
                self.data_file(kind).write_bytes(b"")
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
        self._resolved = [_Resolved(spec, pool, constituents, pcnst) for spec in self.specs]
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
        self._constituents = constituents
        if self.memory:
            self._columns = {"rank": np.concatenate(ranks), "lat": lat_all, "lon": lon_all,
                             "level_pressure_pa": pressure}
            return
        np.savez(self.directory / "columns.npz", rank=np.concatenate(ranks), lat=lat_all, lon=lon_all,
                 level_pressure_pa=pressure)
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
        if not self._surface:
            # the first step this rank records (0 for a notebook run, 1 for a run started from
            # CESM's own clock): every rank gets here at the same step
            self._surface = True
            self.write_surface(pool)
        if self.memory:
            self._keep()
        elif len(self._pending["steps"]) >= self.flush_every or len(self._pending["actions"]) >= 4 * self.flush_every:
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
        land = np.where(np.isfinite(land), land, 0.0)
        if self.memory:
            self._landfrac = land
            return
        np.savez(self.directory / "surface.npz", landfrac=land)

    def flush(self, *, collective: bool = True) -> None:
        """Gather the pending snapshots to rank 0, which appends them and names them in the
        manifest.

        Every rank flushes at the same steps with the same snapshots pending, so the gathers
        line up; rank 0 names a frame only after every rank's part of it is written, and a
        page following the run never reads a frame some rank has not sent.  ``collective=False``
        is for a rank failing on its own: it waits for no other rank, and its pending snapshots
        are dropped (what was written stays).
        """

        if not self._started:
            return
        if self.memory:
            self._keep()
            return
        if not collective:
            for pending in self._pending.values():
                pending.clear()
            self._pending_seconds.clear()
            return
        for kind, pending in self._pending.items():
            for begin in range(0, len(pending), GATHER_CHUNK):
                chunk = np.stack(pending[begin:begin + GATHER_CHUNK]).astype(DTYPES[kind], copy=False)
                parts = self._gather(chunk)
                if self.rank == 0:
                    # each rank's (snapshots, sum(levels) * its columns) -> (snapshots, sum(levels), columns)
                    levels = sum(field.levels for field in self._resolved)
                    blocks = [np.asarray(part).reshape(len(chunk), levels, -1) for part in parts]
                    with self.data_file(kind).open("ab") as handle:
                        np.concatenate(blocks, axis=2).astype(DTYPES[kind], copy=False).tofile(handle)
            self._written[kind] += len(pending)
            pending.clear()
        if self._pending_seconds:
            parts = self._gather(np.asarray(self._pending_seconds, dtype="<f8"))
            if self.rank == 0:
                with self.data_file("action_seconds").open("ab") as handle:
                    np.stack([np.asarray(part, dtype="<f8") for part in parts], axis=1).tofile(handle)
            self._pending_seconds.clear()
        if self.rank == 0 and self._described:
            self._write_manifest(complete=False)

    def _gather(self, values: Any) -> list[Any]:
        gather = getattr(self.comm, "gather", None)
        if callable(gather) and self.size > 1:
            return gather(values, root=0) or []
        return [values]

    # -- in memory -----------------------------------------------------------
    def _keep(self) -> None:
        """Move the pending snapshots into this rank's memory, dropping the oldest beyond the
        bounds (action snapshots a whole step at a time)."""

        for kind, pending in self._pending.items():
            self._kept[kind].extend(pending)
            self._written[kind] += len(pending)
            pending.clear()
        self._kept_seconds.extend(self._pending_seconds)
        self._pending_seconds.clear()
        while len(self._kept["steps"]) > self.keep_steps:
            self._kept["steps"].popleft()
            self._dropped["steps"] += 1
        actions, frames = self._kept["actions"], self._frames["actions"]
        while len(actions) > self.keep_actions:
            oldest = frames[self._dropped["actions"]][0]
            if oldest == frames[-1][0]:
                break                                   # never the step just recorded
            while actions and frames[self._dropped["actions"]][0] == oldest:
                actions.popleft()
                self._kept_seconds.popleft()
                self._dropped["actions"] += 1

    def _local(self, kind: str, index: int) -> np.ndarray:
        """This rank's part of a kept snapshot, (sum(levels), columns)."""

        if kind not in DTYPES:
            raise ValueError(f"kind is steps or actions, not {kind!r}")
        index, first = int(index), self._dropped[kind]
        if index < first:
            keep = self.keep_steps if kind == "steps" else self.keep_actions
            raise ValueError(f"{kind} frame {index} is no longer kept: the model keeps the newest {keep}")
        if index >= first + len(self._kept[kind]):
            raise ValueError(f"{kind} frame {index} is not recorded yet")
        levels = sum(field.levels for field in self._resolved)
        return self._kept[kind][index - first].reshape(levels, -1)

    def _field_rows(self, name: str) -> slice:
        offset = 0
        for field in self._resolved:
            if field.spec.name == name:
                return slice(offset, offset + field.levels)
            offset += field.levels
        raise ValueError(f"no field {name!r}; the run records {[f.spec.name for f in self._resolved]}")

    def query(self, request: Mapping[str, Any]) -> Any:
        """Answer a viewer's request from the snapshots kept in memory.  Collective: every rank
        is asked the same request at the same point (between two steps); the answer is rank 0's,
        None on the others (``column``: the owning rank's).

        ``layout``: the fields, the columns and the level pressures.  ``frames``: the frames
        recorded after the first ``since_steps`` and ``since_actions``, and how many are no longer
        kept.  ``frame``: one field (or one ``level`` of it) of one snapshot over every column,
        (levels, columns).  ``codes``: each field's anomaly code per frame and column.
        ``anomaly_counts``: per frame, field and code, how many columns have it.
        ``change_sums``: per action frame, each field's largest change and sum of squared
        changes over every column.  ``column``: one column through some frames.  ``seconds``:
        every kept action frame's time on every rank, (ranks, frames)."""

        if not self.memory:
            raise ValueError("this recorder writes its snapshots to files: read its directory")
        if not self._described:
            raise ValueError("no step recorded yet")
        what = request.get("q")
        root = self.rank == 0
        if what == "layout":
            return self._layout_payload() if root else None
        if what == "frames":
            if not root:
                return None
            since_steps, since_actions = int(request.get("since_steps", 0)), int(request.get("since_actions", 0))
            return {"step_frames": self._frames["steps"][since_steps:self._written["steps"]],
                    "action_frames": self._frames["actions"][since_actions:self._written["actions"]],
                    "kept_from": dict(self._dropped), "last_step": self._last_step,
                    "complete": self._complete,
                    # the land fraction (gathered at the first step) only to a viewer still without it
                    "landfrac": self._landfrac if request.get("landfrac") else None}
        if what == "frame":
            block = self._local(request["kind"], request["index"])[self._field_rows(str(request["field"]))]
            if request.get("level") is not None:
                level = int(request["level"])
                if not 0 <= level < len(block):
                    raise ValueError(f"{request['field']} has levels 0..{len(block) - 1}")
                block = block[level:level + 1]
            parts = self._gather(np.ascontiguousarray(block))
            return np.concatenate(parts, axis=1) if root else None
        if what == "codes":
            from .state_view import anomaly_codes

            kind, indices = request["kind"], [int(i) for i in request["indices"]]
            local = {field.spec.name: np.zeros((len(indices), len(self._local(kind, indices[0])[0]) if indices else 0),
                                               np.uint8) for field in self._resolved}
            for k, index in enumerate(indices):
                block = self._local(kind, index)
                for field in self._resolved:
                    name = field.spec.name
                    local[name][k] = anomaly_codes(block[self._field_rows(name)], name, field.spec.scale)
            parts = self._gather(local)
            if not root:
                return None
            return {name: np.concatenate([part[name] for part in parts], axis=1) for name in local}
        if what == "anomaly_counts":
            from .state_view import anomaly_codes

            kind, indices = request["kind"], [int(i) for i in request["indices"]]
            local = np.zeros((len(indices), len(self._resolved), 4), np.int64)
            for k, index in enumerate(indices):
                block = self._local(kind, index)
                for f, field in enumerate(self._resolved):
                    codes = anomaly_codes(block[self._field_rows(field.spec.name)], field.spec.name, field.spec.scale)
                    local[k, f] = np.bincount(codes, minlength=4)[:4]
            parts = self._gather(local)
            return np.sum(parts, axis=0) if root else None
        if what == "change_sums":
            indices = [int(i) for i in request["indices"]]
            names = [field.spec.name for field in self._resolved]
            largest = np.zeros((len(names), len(indices)))
            squares = np.zeros((len(names), len(indices)))
            before = None
            for k, index in enumerate(indices):
                block = self._local("actions", index).astype(np.float64)
                if before is not None:
                    for f, name in enumerate(names):
                        rows = self._field_rows(name)
                        change = block[rows] - before[rows]
                        if change.size:
                            largest[f, k] = np.abs(change).max()
                            squares[f, k] = (change ** 2).sum()
                before = block
            parts = self._gather((largest, squares))
            if not root:
                return None
            # NaN where a change was not finite: the largest of any rank's, as one file would give
            most = np.stack([part[0] for part in parts]).max(axis=0)
            total = np.stack([part[1] for part in parts]).sum(axis=0)
            return {"largest": {name: most[f] for f, name in enumerate(names)},
                    "squares": {name: total[f] for f, name in enumerate(names)}}
        if what == "column":
            if self.rank != int(request["rank"]):
                return None
            local = int(request["local"])
            return np.stack([self._local(request["kind"], index)[:, local] for index in request["indices"]]
                            ).astype(np.float64)
        if what == "seconds":
            parts = self._gather(np.asarray(self._kept_seconds, dtype=np.float64))
            return np.stack(parts) if root else None
        raise ValueError(f"no state query {what!r}")

    def _layout_payload(self) -> dict[str, Any]:
        assert self._columns is not None
        return {"run": self.run_label, "ranks": self.size,
                "dtype": {kind: np.dtype(dtype).str for kind, dtype in DTYPES.items()},
                "fields": self._field_payload(), "constituents": getattr(self, "_constituents", []),
                "every": self.every, "action_steps": sorted(self.action_steps), "action_seconds": True,
                "keep_steps": self.keep_steps, "keep_actions": self.keep_actions, **self._columns}

    def _field_payload(self) -> list[dict[str, Any]]:
        return [{"name": field.spec.name, "field": field.spec.field, "constituent": field.spec.constituent,
                 "levels": field.levels, "units": field.spec.units, "scale": field.spec.scale,
                 "label": field.spec.label} for field in self._resolved]

    def abandon(self) -> None:
        """This rank failed: wait for no other rank, mark nothing complete."""

        if self._closed or not self._started:
            return
        self.flush(collective=False)
        self._closed = True

    def close(self) -> None:
        if self._closed or not self._started:
            return
        self.flush()
        self._closed = True
        self._complete = True
        if self.rank == 0 and self._described and not self.memory:
            self._write_manifest(complete=True)

    # -- metadata ----------------------------------------------------------
    def describe_run(self) -> dict[str, Any]:
        """What this rank recorded, for the run summary."""

        summary = {"fields": [spec.name for spec in self.specs], "every": self.every,
                   "action_steps": sorted(self.action_steps),
                   "step_snapshots": self._written["steps"] + len(self._pending["steps"]),
                   "action_snapshots": self._written["actions"] + len(self._pending["actions"]),
                   "store": "memory" if self.memory else "files", "read_only": True}
        if self.memory:
            summary["kept"] = {kind: len(kept) for kind, kept in self._kept.items()}
        return summary

    def _write_manifest(self, *, complete: bool) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "what": "snapshots of physics-state fields of a PI-CAM run (freecam.pi_cam.state_record)",
            "run": self.run_label,
            "ranks": self.size,
            "dtype": {kind: np.dtype(dtype).str for kind, dtype in DTYPES.items()},
            "layout": "per snapshot: every field in order, each (levels, columns) in C order, the columns in rank order",
            "fields": self._field_payload(),
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


__all__ = ["BUILDER_FIELDS", "DEFAULT_FIELDS", "FieldSpec", "KEEP_ACTIONS", "KEEP_STEPS", "KNOWN_FIELDS", "SCHEMA_VERSION",
           "START_OF_STEP", "StateRecorder", "field_spec", "parse_steps"]
