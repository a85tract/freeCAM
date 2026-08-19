"""Write Python-owned StatePool fields into CAM's own history files.

The original iCESM writer only knows the fields CAM registered at build time,
so Notebook-defined StatePool variables and Python physics results never
appear in any output file.  This module closes that gap the way CAM does for
a newly registered field: the value lands **in the model's own history file**,
beside ``T``, ``PS``, and the rest, rather than in a separate stream.

The invariant is that freeCAM's run directory stays indistinguishable from
the original model's.  A run with no Python-owned fields writes no extra
variables and no extra files at all; a run that creates them adds those
variables to the history files CAM already produced, with the same column
ordering, the same time samples, and CAM's own metadata conventions.

Numerics stay in the original source.  This writer only reads StatePool
arrays CAM already produced and appends them to a closed NetCDF file.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence
import warnings

import numpy as np

from .errors import PICAMConfigurationError, PICAMStateError

_DAYS_PER_MONTH: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_COLUMN_DIMENSION = "ncol"
_TIME_DIMENSION = "time"
_STREAM_PATTERN = re.compile(r"\.cam\.(h\d+)\.")
# A completed window waits only until CAM closes the file it belongs to,
# so a healthy stream holds at most one or two.  A queue that keeps growing
# means the target tape is never written, and the reduced fields it holds
# are large enough that an unbounded queue would exhaust rank 0.
DEFAULT_PENDING_LIMIT: int = 64


@dataclass(frozen=True, slots=True)
class PICAMHistoryVariable:
    """One StatePool field added to CAM's history output."""

    field: str
    name: str | None = None
    units: str | None = None
    long_name: str | None = None

    @property
    def output_name(self) -> str:
        return self.name or self.field.rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True)
class PICAMHistorySpec:
    """How Python-owned fields join one CAM history stream."""

    name: str = "python_fields"
    # Which CAM tape to extend.  ``h0`` is the model's default history file.
    stream: str = "h0"
    # ``None`` selects every Python-owned StatePool field that is marked for
    # output, mirroring CAM's registered fields joining the default tape.
    variables: tuple[PICAMHistoryVariable, ...] | None = None
    # CAM's ``nhtfrq``: 0 accumulates monthly means, a positive value writes
    # every N steps, a negative value every N hours.
    nhtfrq: int = 0
    time_period: str = "mean"
    precision: str = "float32"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise PICAMConfigurationError("history stream name cannot be empty")
        if not _STREAM_PATTERN.search(f".cam.{self.stream}."):
            raise PICAMConfigurationError(
                "stream must name a CAM history tape such as 'h0'"
            )
        if self.time_period not in {"mean", "instantaneous"}:
            raise PICAMConfigurationError(
                "time_period must be 'mean' or 'instantaneous'"
            )
        if self.precision not in {"float32", "float64"}:
            raise PICAMConfigurationError(
                "history precision must be float32 or float64"
            )
        if self.variables is not None:
            names = [item.output_name for item in self.variables]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise PICAMConfigurationError(
                    f"history stream {self.name!r} repeats output names: "
                    + ", ".join(duplicates)
                )

    @property
    def monthly(self) -> bool:
        return int(self.nhtfrq) == 0

    @property
    def cell_methods(self) -> str:
        return f"time: {self.time_period}"

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stream": self.stream,
            "nhtfrq": int(self.nhtfrq),
            "time_period": self.time_period,
            "precision": self.precision,
            "variables": (
                None
                if self.variables is None
                else [
                    {
                        "field": item.field,
                        "name": item.output_name,
                        "units": item.units,
                        "long_name": item.long_name,
                    }
                    for item in self.variables
                ]
            ),
        }


@dataclass(slots=True)
class _ColumnPlan:
    """Rank-local valid columns and their CAM global column identifiers."""

    chunks: np.ndarray
    columns: np.ndarray
    global_ids: np.ndarray
    global_count: int
    shape: tuple[int, int]


@dataclass(slots=True)
class _PendingSample:
    """One completed window waiting for CAM to finish its history file."""

    date: int
    datesec: int
    values: dict[str, np.ndarray]
    metadata: dict[str, tuple[str | None, str | None]]


def _as_variable(item: Any) -> PICAMHistoryVariable:
    if isinstance(item, PICAMHistoryVariable):
        return item
    if isinstance(item, str):
        return PICAMHistoryVariable(field=item)
    if isinstance(item, Mapping):
        try:
            return PICAMHistoryVariable(
                field=str(item["field"]),
                name=None if item.get("name") is None else str(item["name"]),
                units=None if item.get("units") is None else str(item["units"]),
                long_name=(
                    None
                    if item.get("long_name") is None
                    else str(item["long_name"])
                ),
            )
        except KeyError as exc:
            raise PICAMConfigurationError(
                "history variable mappings require a 'field' key"
            ) from exc
    raise PICAMConfigurationError(
        "history variables must be field names, mappings, or "
        "PICAMHistoryVariable instances"
    )


def elapsed_days(year: int, month: int, day: int, seconds: int) -> float:
    """Days since 0001-01-01 00:00:00 on CAM's NO_LEAP calendar."""

    days = (int(year) - 1) * 365
    days += sum(_DAYS_PER_MONTH[: int(month) - 1])
    days += int(day) - 1
    return float(days) + float(seconds) / 86400.0


class PICAMHistoryStream:
    """Accumulate Python-owned fields and add them to CAM's history files."""

    def __init__(self, driver: Any, spec: PICAMHistorySpec) -> None:
        self.driver = driver
        self.spec = spec
        self._sums: dict[str, np.ndarray] = {}
        self._samples = 0
        self._interval_start: tuple[int, int, int, int] | None = None
        self._pending: deque[_PendingSample] = deque(maxlen=DEFAULT_PENDING_LIMIT)
        self._written = 0
        self._dropped = 0
        self._plan: _ColumnPlan | None = None

    # -- state -----------------------------------------------------------

    @property
    def written(self) -> int:
        """Time samples this stream has added to CAM history files."""

        return self._written

    @property
    def pending(self) -> int:
        """Completed windows still waiting for their CAM history file."""

        return len(self._pending)

    @property
    def pending_limit(self) -> int:
        """How many completed windows may wait before the oldest is lost."""

        return int(self._pending.maxlen)

    @property
    def dropped(self) -> int:
        """Windows discarded because no CAM history file ever claimed them."""

        return self._dropped

    @property
    def accumulated(self) -> int:
        return self._samples

    def fields(self) -> tuple[PICAMHistoryVariable, ...]:
        """Resolve the Python-owned fields this stream writes right now."""

        if self.spec.variables is not None:
            return self.spec.variables
        pool = self.driver.pool
        selected: list[PICAMHistoryVariable] = []
        for name in sorted(pool.dynamic_fields):
            contract = pool.contract(name)
            if not getattr(contract, "output", True):
                continue
            if not self._is_column_field(np.asarray(pool[name])):
                continue
            selected.append(
                PICAMHistoryVariable(field=name, units=contract.units)
            )
        return tuple(selected)

    # -- workflow entry point --------------------------------------------

    def step(self) -> int:
        """Accumulate this step and, when a window closes, extend CAM output."""

        self.accumulate()
        if not self.window_complete():
            return 0
        self.close_window()
        return self.drain()

    def accumulate(self) -> None:
        """Add this step's rank-local values to the open averaging window."""

        variables = self.fields()
        if not variables:
            return
        clock = self.driver.clock
        if self._interval_start is None:
            self._interval_start = self._clock_stamp()
        plan = self._column_plan()
        for variable in variables:
            values = self._local_values(variable, plan)
            if values is None:
                continue
            if self.spec.time_period == "instantaneous":
                self._sums[variable.output_name] = values
            else:
                total = self._sums.get(variable.output_name)
                self._sums[variable.output_name] = (
                    values if total is None else total + values
                )
        self._samples += 1
        del clock

    def window_complete(self) -> bool:
        """Return CAM's ``nhtfrq`` alarm for the open averaging window."""

        if self._samples == 0 or self._interval_start is None:
            return False
        clock = self.driver.clock
        if self.spec.monthly:
            return int(clock.month) != self._interval_start[1]
        frequency = int(self.spec.nhtfrq)
        if frequency > 0:
            return self._samples >= frequency
        elapsed = elapsed_days(*self._clock_stamp()) - elapsed_days(
            *self._interval_start
        )
        return elapsed * 24.0 >= float(-frequency) - 1.0e-9

    def close_window(self) -> None:
        """Reduce the open window and queue it for CAM's history file."""

        if self._samples == 0 or self._interval_start is None:
            raise PICAMStateError(
                f"history stream {self.spec.name!r} has nothing accumulated"
            )
        divisor = 1.0 if self.spec.time_period == "instantaneous" else self._samples
        variables = {item.output_name: item for item in self.fields()}
        local = {
            name: (values / divisor).astype(self.spec.precision, copy=False)
            for name, values in self._sums.items()
        }
        plan = self._column_plan()
        gathered = {
            name: self.driver.comm.gather(values, root=0)
            for name, values in local.items()
        }
        indices = self.driver.comm.gather(plan.global_ids, root=0)
        year, month, day, seconds = self._clock_stamp()
        self._sums.clear()
        self._samples = 0
        # CAM's windows abut: the next one opens where this one closed.
        self._interval_start = (year, month, day, seconds)
        if int(self.driver.comm.rank) != 0:
            return
        placement = np.concatenate(
            [np.asarray(item, dtype=np.int64) for item in indices]
        )
        if placement.size != plan.global_count:
            raise PICAMStateError(
                "gathered column count does not match the collective total"
            )
        if np.unique(placement).size != placement.size:
            raise PICAMStateError(
                "CAM global column identifiers repeat across MPI ranks"
            )
        values = {
            name: self._order(np.concatenate(parts, axis=0), placement)
            for name, parts in gathered.items()
        }
        metadata = {
            name: (
                variables[name].units,
                variables[name].long_name,
            )
            for name in values
        }
        if len(self._pending) == self._pending.maxlen:
            # Losing model output silently is worse than the memory it costs,
            # so say so.  Reaching this point means the configured tape is
            # never written, not that the queue is merely busy.
            self._dropped += 1
            if self._dropped == 1:
                warnings.warn(
                    f"history stream {self.spec.name!r} has "
                    f"{self.pending_limit} completed windows waiting and no "
                    f"'{self.spec.stream}' history file to write them to; the "
                    "oldest window is being discarded. Check that the case "
                    f"writes a {self.spec.stream} tape.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._pending.append(
            _PendingSample(
                date=year * 10000 + month * 100 + day,
                datesec=seconds,
                values=values,
                metadata=metadata,
            )
        )

    def drain(self) -> int:
        """Add every queued window whose CAM history file is complete."""

        if int(self.driver.comm.rank) != 0 or not self._pending:
            return 0
        run_dir = self.driver.run_dir
        if run_dir is None:
            raise PICAMStateError(
                "Python-owned history output requires a run directory"
            )
        written = 0
        for path in sorted(Path(run_dir).glob(f"*.cam.{self.spec.stream}.*.nc")):
            if not self._pending:
                break
            written += self._extend(path)
        self._written += written
        return written

    # -- CAM history file extension --------------------------------------

    def _extend(self, path: Path) -> int:
        from netCDF4 import Dataset

        try:
            with Dataset(path) as dataset:
                if _COLUMN_DIMENSION not in dataset.dimensions:
                    return 0
                stamps = {
                    (int(date), int(seconds)): index
                    for index, (date, seconds) in enumerate(
                        zip(
                            dataset.variables["date"][:],
                            dataset.variables["datesec"][:],
                        )
                    )
                }
                columns = len(dataset.dimensions[_COLUMN_DIMENSION])
                levels = {
                    name: len(dimension)
                    for name, dimension in dataset.dimensions.items()
                    if name in {"lev", "ilev"}
                }
        except OSError:
            # CAM still owns the file; try again once it closes.
            return 0
        matched = [
            sample
            for sample in self._pending
            if (sample.date, sample.datesec) in stamps
        ]
        if not matched:
            return 0
        with Dataset(path, "a") as dataset:
            for sample in matched:
                index = stamps[(sample.date, sample.datesec)]
                for name, values in sample.values.items():
                    self._write(
                        dataset, name, values, index, sample, columns, levels
                    )
        for sample in matched:
            self._pending.remove(sample)
        return len(matched)

    def _write(
        self,
        dataset: Any,
        name: str,
        values: np.ndarray,
        index: int,
        sample: _PendingSample,
        columns: int,
        levels: Mapping[str, int],
    ) -> None:
        if values.shape[-1] != columns:
            raise PICAMStateError(
                f"history field {name!r} spans {values.shape[-1]} columns; "
                f"the CAM history file has {columns}"
            )
        if values.ndim == 1:
            dimensions = (_TIME_DIMENSION, _COLUMN_DIMENSION)
        else:
            level = self._level_name(values.shape[0], levels)
            dimensions = (_TIME_DIMENSION, level, _COLUMN_DIMENSION)
        if name not in dataset.variables:
            units, long_name = sample.metadata.get(name, (None, None))
            created = dataset.createVariable(
                name,
                "f4" if self.spec.precision == "float32" else "f8",
                dimensions,
            )
            if len(dimensions) == 3:
                created.mdims = 1
            created.units = units or "1"
            created.long_name = long_name or name
            created.cell_methods = self.spec.cell_methods
        elif dataset.variables[name].dimensions != dimensions:
            raise PICAMStateError(
                f"CAM history file already defines {name!r} with different "
                "dimensions"
            )
        dataset.variables[name][index, ...] = values

    def _level_name(self, size: int, levels: Mapping[str, int]) -> str:
        for name in ("lev", "ilev"):
            if levels.get(name) == size:
                return name
        raise PICAMStateError(
            f"history output cannot label a vertical dimension of {size} "
            "levels; the CAM history file uses lev and ilev"
        )

    @staticmethod
    def _order(values: np.ndarray, placement: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            ordered = np.empty(placement.size, dtype=values.dtype)
            ordered[placement] = values
            return ordered
        ordered = np.empty(
            (values.shape[1], placement.size), dtype=values.dtype
        )
        ordered[:, placement] = values.T
        return ordered

    # -- rank-local collection -------------------------------------------

    def _clock_stamp(self) -> tuple[int, int, int, int]:
        clock = self.driver.clock
        return (
            int(clock.year),
            int(clock.month),
            int(clock.day),
            int(clock.seconds),
        )

    def _column_plan(self) -> _ColumnPlan:
        # CAM's physics decomposition is fixed once the grid exists, so the
        # column map is built once per stream instead of at every step.
        if self._plan is None:
            self._plan = self._build_column_plan()
        return self._plan

    def _build_column_plan(self) -> _ColumnPlan:
        pool = self.driver.pool
        for name in ("phys_state.cid", "phys_state.ngrdcol"):
            if name not in pool:
                raise PICAMStateError(
                    f"Python-owned history output requires {name!r}; it is "
                    "available once CAM has initialized its physics grid"
                )
        identifiers = np.asarray(pool["phys_state.cid"])
        valid = np.asarray(pool["phys_state.ngrdcol"]).astype(int).ravel()
        if identifiers.ndim != 2:
            raise PICAMStateError(
                "phys_state.cid must be a (columns, chunks) array"
            )
        chunk_indices: list[int] = []
        column_indices: list[int] = []
        global_ids: list[int] = []
        for chunk in range(identifiers.shape[1]):
            count = int(valid[chunk]) if chunk < valid.size else 0
            for column in range(count):
                identifier = int(identifiers[column, chunk])
                if identifier <= 0:
                    raise PICAMStateError(
                        "phys_state.cid holds a non-positive column id; CAM "
                        "assigns one-based unique column identifiers"
                    )
                chunk_indices.append(chunk)
                column_indices.append(column)
                global_ids.append(identifier - 1)
        local = np.asarray(global_ids, dtype=np.int64)
        counts = self.driver.comm.allgather(int(local.size))
        return _ColumnPlan(
            chunks=np.asarray(chunk_indices, dtype=np.int64),
            columns=np.asarray(column_indices, dtype=np.int64),
            global_ids=local,
            global_count=int(sum(int(item) for item in counts)),
            shape=(int(identifiers.shape[0]), int(identifiers.shape[1])),
        )

    def _is_column_field(self, values: np.ndarray) -> bool:
        plan = self._column_plan()
        pcols, chunks = plan.shape
        if values.ndim == 2:
            return values.shape == (pcols, chunks)
        if values.ndim == 3:
            return values.shape[0] == pcols and values.shape[2] == chunks
        return False

    def _local_values(
        self, variable: PICAMHistoryVariable, plan: _ColumnPlan
    ) -> np.ndarray | None:
        pool = self.driver.pool
        try:
            canonical = pool.canonical_name(variable.field)
        except KeyError as exc:
            raise PICAMStateError(
                f"history stream {self.spec.name!r} references unknown field "
                f"{variable.field!r}"
            ) from exc
        values = np.asarray(pool[canonical])
        if not self._is_column_field(values):
            if self.spec.variables is None:
                # Automatic selection simply skips fields that do not live on
                # CAM's physics columns.
                return None
            raise PICAMStateError(
                f"history field {variable.field!r} has shape {values.shape}; "
                "CAM history output covers (columns, chunks) and "
                "(columns, levels, chunks) fields"
            )
        if values.ndim == 2:
            selected = values[plan.columns, plan.chunks]
        else:
            selected = values[plan.columns, :, plan.chunks]
        return np.ascontiguousarray(selected, dtype=np.float64)


class PICAMHistoryStreamRegistry:
    """Own the Python-defined history extensions of one live model."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self._streams: dict[str, PICAMHistoryStream] = {}

    def __contains__(self, name: object) -> bool:
        return str(name) in self._streams

    def __iter__(self) -> Iterator[str]:
        return iter(tuple(self._streams))

    def __len__(self) -> int:
        return len(self._streams)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._streams)

    def stream(self, name: str) -> PICAMHistoryStream:
        try:
            return self._streams[str(name)]
        except KeyError as exc:
            raise PICAMConfigurationError(
                f"history stream {name!r} is not installed"
            ) from exc

    def install(
        self,
        name: str = "python_fields",
        *,
        fields: Sequence[Any] | None = None,
        stream: str = "h0",
        nhtfrq: int = 0,
        time_period: str = "mean",
        precision: str = "float32",
    ) -> PICAMHistoryStream:
        """Add Python-owned fields to one CAM history stream on every rank."""

        spec = PICAMHistorySpec(
            name=str(name),
            stream=str(stream),
            variables=(
                None
                if fields is None
                else tuple(_as_variable(item) for item in fields)
            ),
            nhtfrq=int(nhtfrq),
            time_period=str(time_period),
            precision=str(precision),
        )
        if spec.name in self._streams:
            raise PICAMConfigurationError(
                f"history stream {spec.name!r} is already installed"
            )
        if spec.variables is not None:
            for variable in spec.variables:
                self._require_field(variable.field)
        installed = PICAMHistoryStream(self.driver, spec)
        self._streams[spec.name] = installed
        return installed

    def remove(self, name: str) -> PICAMHistorySpec:
        stream = self.stream(name)
        del self._streams[stream.spec.name]
        return stream.spec

    def step(self, name: str) -> int:
        return self.stream(name).step()

    def drain(self) -> int:
        """Add every queued window across streams, at finalization."""

        return sum(stream.drain() for stream in self._streams.values())

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **stream.spec.to_payload(),
                "resolved_fields": [item.output_name for item in stream.fields()],
                "written": stream.written,
                "pending": stream.pending,
                "dropped": stream.dropped,
                "accumulated": stream.accumulated,
            }
            for stream in self._streams.values()
        )

    def _require_field(self, field: str) -> None:
        pool = self.driver.pool
        try:
            pool.canonical_name(field)
        except KeyError as exc:
            raise PICAMStateError(
                f"history output references unknown field {field!r}"
            ) from exc
