"""Per-rank action timeline of a PI-CAM run, for the timeline viewer.

Off unless asked for (``--timeline-dir``).  When on, every rank keeps, in
memory, one record per executed action (step, action, kind, start, end) plus
the time it spent inside the driver's collective agreement calls, and appends
them to its own binary file every ``flush_every`` steps; rank 0 keeps a small
manifest current.  Nothing is sent between ranks while the model steps: the
only collectives are one barrier that sets a common time origin and one
gather, after initialization, of each rank's host and column coordinates.

Layout of a timeline directory::

    manifest.json              rank 0: schema, ranks, kinds, steps flushed, complete
    columns.npz                rank 0: each rank's host and node-local rank; each column's rank,
                               lat, lon (degrees) and surface geopotential (land where nonzero)
    ranks/rank-NNNN.bin        records, RECORD_DTYPE, appended
    ranks/rank-NNNN.names.json this rank's action names, index = record ``action``

Times are seconds from the common origin, on each rank's monotonic clock; the
origin barrier aligns the ranks to within its exit skew (tens of microseconds).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

SCHEMA_VERSION = 1

RECORD_DTYPE = np.dtype([
    ("step", "<i4"),        # coupling step (0 = the first step); -1 = initialization
    ("action", "<i2"),      # index into this rank's names table
    ("kind", "<i1"),        # KIND_*
    ("pad", "<i1"),
    ("t0", "<f8"),          # seconds from the common origin
    ("t1", "<f8"),
])

KIND_ACTION = 0             # one plan action, start to end
KIND_WAIT = 1               # time inside a collective agreement call, within the enclosing action
KIND_STEP = 2               # one whole step
KIND_PHASE = 3              # a lifecycle phase outside the step loop (initialization, finalization)
KINDS = {"action": KIND_ACTION, "wait": KIND_WAIT, "step": KIND_STEP, "phase": KIND_PHASE}

STEP_NAME = "step"
INIT_NAME = "initialize"


def _node_local_rank() -> int:
    for name in ("PMI_LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID"):
        value = os.environ.get(name)
        if value is not None and value.strip().lstrip("-").isdigit():
            return int(value)
    return -1


class TimelineRecorder:
    """One rank's action timeline, appended to ``directory/ranks/rank-NNNN.bin``."""

    def __init__(self, directory: str | Path, *, rank: int, size: int, comm: Any | None = None,
                 flush_every: int = 100, run_label: str = "") -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least 1")
        self.directory = Path(directory)
        self.rank = int(rank)
        self.size = int(size)
        self.comm = comm
        self.flush_every = int(flush_every)
        self.run_label = str(run_label)
        self.clock = time.perf_counter
        self._origin = 0.0
        self._origin_utc = ""
        self._names: dict[str, int] = {}
        self._names_written = 0
        self._pending: list[tuple[int, int, int, int, float, float]] = []
        self._steps_since_flush = 0
        self._last_step = -1
        self._records_written = 0
        self._started = False
        self._closed = False
        #: the action whose collectives are being timed (set by the driver around each action)
        self.current_action = ""

    # -- paths -------------------------------------------------------------
    @property
    def rank_file(self) -> Path:
        return self.directory / "ranks" / f"rank-{self.rank:04d}.bin"

    @property
    def names_file(self) -> Path:
        return self.directory / "ranks" / f"rank-{self.rank:04d}.names.json"

    @property
    def manifest_file(self) -> Path:
        return self.directory / "manifest.json"

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Create the directory, truncate this rank's file and set the common time origin."""

        (self.directory / "ranks").mkdir(parents=True, exist_ok=True)
        self.rank_file.write_bytes(b"")
        barrier = getattr(self.comm, "Barrier", None)
        if callable(barrier):
            barrier()
        self._origin = self.clock()
        self._origin_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._started = True
        if self.rank == 0:
            self._write_manifest(complete=False)

    def now(self) -> float:
        return self.clock() - self._origin

    def _id(self, name: str) -> int:
        index = self._names.get(name)
        if index is None:
            index = self._names[name] = len(self._names)
            if index > np.iinfo(np.int16).max:
                raise ValueError("too many distinct timeline names")
        return index

    def record(self, step: int, name: str, kind: int, t0: float, t1: float) -> None:
        """Append one record; ``t0``/``t1`` are raw ``clock()`` readings."""

        self._pending.append((int(step), self._id(name), int(kind), 0, t0 - self._origin, t1 - self._origin))

    def action(self, step: int, name: str, started: float) -> None:
        self.record(step, name, KIND_ACTION, started, self.clock())

    def wait(self, step: int, started: float) -> None:
        self.record(step, self.current_action or "outside the plan", KIND_WAIT, started, self.clock())

    def phase(self, name: str, started: float) -> None:
        self.record(-1, name, KIND_PHASE, started, self.clock())

    def step_done(self, step: int, started: float) -> None:
        self.record(step, STEP_NAME, KIND_STEP, started, self.clock())
        self._last_step = max(self._last_step, int(step))
        self._steps_since_flush += 1
        if self._steps_since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Append the pending records to this rank's file; rank 0 refreshes the manifest."""

        if not self._started:
            return
        if self._pending:
            block = np.array(self._pending, dtype=RECORD_DTYPE)
            with self.rank_file.open("ab") as handle:
                block.tofile(handle)
            self._records_written += len(block)
            self._pending.clear()
        if len(self._names) != self._names_written:
            names = [None] * len(self._names)
            for name, index in self._names.items():
                names[index] = name
            self.names_file.write_text(json.dumps(names))
            self._names_written = len(self._names)
        self._steps_since_flush = 0
        if self.rank == 0:
            self._write_manifest(complete=False)

    def close(self) -> None:
        if self._closed or not self._started:
            return
        self.flush()
        self._closed = True
        if self.rank == 0:
            self._write_manifest(complete=True)

    # -- metadata ----------------------------------------------------------
    def _write_manifest(self, *, complete: bool) -> None:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "what": "per-rank action timeline of a PI-CAM run (freecam.pi_cam.timeline)",
            "run": self.run_label,
            "ranks": self.size,
            "origin_utc": self._origin_utc,
            "flush_every": self.flush_every,
            "last_step_flushed": self._last_step,
            "complete": bool(complete),
            "kinds": KINDS,
            "record_dtype": [[name, RECORD_DTYPE[name].str] for name in RECORD_DTYPE.names],
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        temporary = self.manifest_file.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=1) + "\n")
        temporary.replace(self.manifest_file)

    def write_columns(self, pool: Mapping[str, Any]) -> None:
        """Gather every rank's host, node-local rank and column coordinates to rank 0 (one
        collective, after initialization) and write ``columns.npz``."""

        lat, lon, phis = _columns(pool)
        local = (socket.gethostname().split(".")[0], _node_local_rank(), lat, lon, phis)
        gather = getattr(self.comm, "gather", None)
        gathered = gather(local, root=0) if callable(gather) else [local]
        if self.rank != 0:
            return
        ranks, lats, lons, phiss, hosts, local_ranks = [], [], [], [], [], []
        for rank, (host, local_rank, rank_lat, rank_lon, rank_phis) in enumerate(gathered):
            hosts.append(host)
            local_ranks.append(local_rank)
            ranks.append(np.full(len(rank_lat), rank, dtype=np.int32))
            lats.append(np.asarray(rank_lat, dtype=np.float64))
            lons.append(np.asarray(rank_lon, dtype=np.float64))
            phiss.append(np.asarray(rank_phis, dtype=np.float64))
        lat_all = np.concatenate(lats) if lats else np.zeros(0)
        lon_all = np.concatenate(lons) if lons else np.zeros(0)
        radians = lat_all.size > 0 and float(np.nanmax(np.abs(lat_all))) <= np.pi / 2 + 1e-6
        if radians:
            lat_all, lon_all = np.degrees(lat_all), np.degrees(lon_all)
        np.savez(self.directory / "columns.npz", rank=np.concatenate(ranks) if ranks else np.zeros(0, np.int32),
                 lat=lat_all, lon=lon_all, phis=np.concatenate(phiss) if phiss else np.zeros(0),
                 host=np.array(hosts), local_rank=np.array(local_ranks, dtype=np.int32))


def _columns(pool: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This rank's physics columns (lat, lon, surface geopotential) from the state pool:
    ``phys_state.lat``/``lon``/``phis`` are (pcols, chunks), and each chunk holds ``ncol``
    real columns.  The geopotential lets the viewer tell land (above sea level) from sea."""

    def field(name: str) -> np.ndarray | None:
        try:
            return np.asarray(pool[name], dtype=np.float64)
        except (KeyError, TypeError):
            return None

    lat, lon, phis = field("phys_state.lat"), field("phys_state.lon"), field("phys_state.phis")
    if lat is None or lon is None:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if phis is None or phis.shape != lat.shape:
        phis = np.zeros_like(lat)
    ncol = None
    for name in ("phys_state.ncol", "grid.chunk_ncols"):
        try:
            ncol = np.asarray(pool[name], dtype=np.int64).reshape(-1)
            break
        except (KeyError, TypeError):
            continue
    if lat.ndim == 2 and ncol is not None and ncol.size == lat.shape[1]:
        take = [slice(0, int(n)) for n in ncol]
        lat = np.concatenate([lat[t, c] for c, t in enumerate(take)])
        lon = np.concatenate([lon[t, c] for c, t in enumerate(take)])
        phis = np.concatenate([phis[t, c] for c, t in enumerate(take)])
    else:
        lat, lon, phis = lat.reshape(-1), lon.reshape(-1), phis.reshape(-1)
    keep = np.isfinite(lat) & np.isfinite(lon)
    return lat[keep], lon[keep], np.where(np.isfinite(phis[keep]), phis[keep], 0.0)


__all__ = ["KINDS", "KIND_ACTION", "KIND_PHASE", "KIND_STEP", "KIND_WAIT", "RECORD_DTYPE", "SCHEMA_VERSION",
           "TimelineRecorder"]
