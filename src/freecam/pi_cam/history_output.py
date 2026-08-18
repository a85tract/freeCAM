"""Python-owned CAM-format history output for rank-local StatePool fields.

The original iCESM history writer only knows the fields CAM registered at
build time, so Notebook-defined StatePool variables and Python physics results
never reach a history file.  This module adds one Python-owned output step
that writes those fields with the layout, naming, and time semantics the
original model produces:

* global ``ncol`` columns ordered by CAM's unique column id;
* CAM's ``nhtfrq`` accumulation window (``0`` monthly, ``>0`` every N steps,
  ``<0`` every N hours) and ``mfilt`` samples per file;
* ``time`` at the end of the accumulation window with ``time_bnds`` spanning
  it, ``date``/``datesec``/``ndcur``/``nscur``/``nsteph`` on the NO_LEAP
  calendar, and ``cell_methods = "time: mean"`` for averaged fields;
* file names ``<case>.cam.<stream>.<YYYY>-<MM>.nc`` for monthly output and
  ``<case>.cam.<stream>.<YYYY>-<MM>-<DD>-<SSSSS>.nc`` otherwise;
* the static grid and vertical-coordinate variables carried over from the
  run's own CAM history stream, so both writers describe one identical grid.

Numerics stay in the original model.  This writer only reads StatePool arrays
that CAM already produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import socket
import time as time_module
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .errors import PICAMConfigurationError, PICAMStateError


# Static coordinate and vertical-grid variables the original CAM history file
# defines.  They describe the grid rather than a timestep, so a Python-owned
# stream copies them verbatim from the run's own CAM output instead of
# recomputing values that must agree with the model bit for bit.
TEMPLATE_VARIABLES: tuple[str, ...] = (
    "lev",
    "ilev",
    "hyam",
    "hybm",
    "hyai",
    "hybi",
    "P0",
    "area",
    "lat",
    "lon",
    "gw",
    "ntrm",
    "ntrn",
    "ntrk",
    "ndbase",
    "nsbase",
    "nbdate",
    "nbsec",
    "mdt",
)

TEMPLATE_ATTRIBUTES: tuple[str, ...] = (
    "np",
    "ne",
    "Conventions",
    "source",
    "case",
    "title",
    "initial_file",
    "topography_file",
)

_DAYS_PER_MONTH: tuple[int, ...] = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

_COLUMN_DIMENSION = "ncol"
_TIME_DIMENSION = "time"
_CHARS = 8


@dataclass(frozen=True, slots=True)
class PICAMHistoryVariable:
    """One StatePool field selected for a Python-owned history stream."""

    field: str
    name: str | None = None
    units: str | None = None
    long_name: str | None = None

    @property
    def output_name(self) -> str:
        return self.name or self.field.rsplit(".", 1)[-1]


@dataclass(frozen=True, slots=True)
class PICAMHistorySpec:
    """Declarative description of one Python-owned history stream."""

    name: str
    stream: str = "h9"
    variables: tuple[PICAMHistoryVariable, ...] = ()
    # CAM's ``nhtfrq``: 0 writes monthly means, a positive value writes every
    # N model steps, a negative value writes every N hours.
    nhtfrq: int = 0
    # CAM's ``mfilt``: time samples accumulated in one file.
    mfilt: int = 1
    time_period: str = "mean"
    template: str | Path | None = None
    precision: str = "float32"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise PICAMConfigurationError("history stream name cannot be empty")
        if not self.variables:
            raise PICAMConfigurationError(
                f"history stream {self.name!r} needs at least one field"
            )
        if int(self.mfilt) < 1:
            raise PICAMConfigurationError(
                "mfilt must be a positive number of time samples per file"
            )
        if self.time_period not in {"mean", "instantaneous"}:
            raise PICAMConfigurationError(
                "time_period must be 'mean' or 'instantaneous'"
            )
        if self.precision not in {"float32", "float64"}:
            raise PICAMConfigurationError(
                "history precision must be float32 or float64"
            )
        duplicates = sorted(
            name
            for name in {item.output_name for item in self.variables}
            if sum(item.output_name == name for item in self.variables) > 1
        )
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
            "mfilt": int(self.mfilt),
            "time_period": self.time_period,
            "precision": self.precision,
            "variables": [
                {
                    "field": item.field,
                    "name": item.output_name,
                    "units": item.units,
                    "long_name": item.long_name,
                }
                for item in self.variables
            ],
        }


@dataclass(slots=True)
class _ColumnPlan:
    """Rank-local valid columns and their CAM global column identifiers."""

    chunks: np.ndarray
    columns: np.ndarray
    global_ids: np.ndarray
    global_count: int


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


def _elapsed_days(year: int, month: int, day: int, seconds: int) -> float:
    """Days since 0001-01-01 00:00:00 on CAM's NO_LEAP calendar."""

    days = (int(year) - 1) * 365
    days += sum(_DAYS_PER_MONTH[: int(month) - 1])
    days += int(day) - 1
    return float(days) + float(seconds) / 86400.0


def _login_name() -> str:
    import os

    return os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"


def _template_attributes(path: Path) -> dict[str, Any]:
    from netCDF4 import Dataset

    with Dataset(path) as source:
        return {
            name: source.getncattr(name)
            for name in TEMPLATE_ATTRIBUTES
            if name in source.ncattrs()
        }


class PICAMHistoryStream:
    """Accumulate and write StatePool fields as one CAM-format history stream."""

    def __init__(self, driver: Any, spec: PICAMHistorySpec) -> None:
        self.driver = driver
        self.spec = spec
        self._sums: dict[str, np.ndarray] = {}
        self._samples = 0
        self._writes = 0
        self._interval_start: tuple[int, int, int, int] | None = None
        self._template: Path | None = None
        self._active_path: Path | None = None
        self._records = 0

    # -- state -----------------------------------------------------------

    @property
    def writes(self) -> int:
        """Number of time samples this stream has written."""

        return self._writes

    @property
    def accumulated(self) -> int:
        """Steps accumulated into the open averaging window."""

        return self._samples

    @property
    def interval_start(self) -> tuple[int, int, int, int] | None:
        return self._interval_start

    # -- workflow entry point --------------------------------------------

    def step(self) -> Path | None:
        """Accumulate this step and write once the CAM window completes."""

        self.accumulate()
        if not self.window_complete():
            return None
        return self.flush()

    def accumulate(self) -> None:
        """Add this step's rank-local values to the open averaging window."""

        clock = self.driver.clock
        if self._interval_start is None:
            self._interval_start = (
                int(clock.year),
                int(clock.month),
                int(clock.day),
                int(clock.seconds),
            )
        plan = self._column_plan()
        for variable in self.spec.variables:
            values = self._local_values(variable, plan)
            if self.spec.time_period == "instantaneous":
                self._sums[variable.output_name] = values
            else:
                total = self._sums.get(variable.output_name)
                self._sums[variable.output_name] = (
                    values if total is None else total + values
                )
        self._samples += 1

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
        elapsed = _elapsed_days(
            int(clock.year), int(clock.month), int(clock.day), int(clock.seconds)
        ) - _elapsed_days(*self._interval_start)
        return elapsed * 24.0 >= float(-frequency) - 1.0e-9

    def flush(self) -> Path | None:
        """Write the open window as one CAM-format time sample."""

        if self._samples == 0 or self._interval_start is None:
            raise PICAMStateError(
                f"history stream {self.spec.name!r} has nothing accumulated"
            )
        plan = self._column_plan()
        divisor = 1.0 if self.spec.time_period == "instantaneous" else self._samples
        local = {
            name: (values / divisor).astype(self.spec.precision, copy=False)
            for name, values in self._sums.items()
        }
        gathered = tuple(
            (variable, self.driver.comm.gather(local[variable.output_name], root=0))
            for variable in self.spec.variables
        )
        indices = self.driver.comm.gather(plan.global_ids, root=0)
        bounds = (self._interval_start, self._clock_stamp())
        self._sums.clear()
        self._samples = 0
        # CAM's averaging windows abut: the next one starts where this one
        # ended, so consecutive samples carry contiguous ``time_bnds``.
        self._interval_start = bounds[1]
        self._writes += 1
        path = self._resolve_path(bounds[1])
        if path != self._active_path:
            self._active_path = path
            self._records = 0
        self._records += 1
        if int(self.driver.comm.rank) != 0:
            return None
        self._append(path, plan, indices, gathered, bounds)
        return path

    # -- naming ----------------------------------------------------------

    def path(self) -> Path:
        """Path of the file the next time sample will land in."""

        return self._resolve_path(self._clock_stamp())

    def _clock_stamp(self) -> tuple[int, int, int, int]:
        clock = self.driver.clock
        return (
            int(clock.year),
            int(clock.month),
            int(clock.day),
            int(clock.seconds),
        )

    def _resolve_path(self, stamp: tuple[int, int, int, int]) -> Path:
        if self._active_path is not None and self._records < int(self.spec.mfilt):
            return self._active_path
        run_dir = self.driver.run_dir
        if run_dir is None:
            raise PICAMStateError(
                "Python-owned history output requires a run directory"
            )
        year, month, day, seconds = stamp
        if self.spec.monthly:
            # CAM stamps a completed monthly mean with the month it covers,
            # not with the first date of the following month.
            if month == 1:
                year, month = year - 1, 12
            else:
                month -= 1
            label = f"{year:04d}-{month:02d}"
        else:
            label = f"{year:04d}-{month:02d}-{day:02d}-{seconds:05d}"
        case = str(self.driver.config.case_name)
        return Path(run_dir) / f"{case}.cam.{self.spec.stream}.{label}.nc"

    # -- rank-local collection -------------------------------------------

    def _column_plan(self) -> _ColumnPlan:
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
        )

    def _local_values(
        self, variable: PICAMHistoryVariable, plan: _ColumnPlan
    ) -> np.ndarray:
        pool = self.driver.pool
        try:
            canonical = pool.canonical_name(variable.field)
        except KeyError as exc:
            raise PICAMStateError(
                f"history stream {self.spec.name!r} references unknown field "
                f"{variable.field!r}"
            ) from exc
        values = np.asarray(pool[canonical])
        if values.ndim == 2:
            selected = values[plan.columns, plan.chunks]
        elif values.ndim == 3:
            selected = values[plan.columns, :, plan.chunks]
        else:
            raise PICAMStateError(
                f"history field {variable.field!r} has {values.ndim} dimensions; "
                "CAM history output covers (columns, chunks) and "
                "(columns, levels, chunks) fields"
            )
        return np.ascontiguousarray(selected, dtype=np.float64)

    # -- rank-zero writing -----------------------------------------------

    def _append(
        self,
        path: Path,
        plan: _ColumnPlan,
        indices: Sequence[np.ndarray],
        gathered: Sequence[tuple[PICAMHistoryVariable, Sequence[np.ndarray]]],
        bounds: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    ) -> None:
        from netCDF4 import Dataset

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
        created = not path.is_file()
        with Dataset(path, "w" if created else "a") as dataset:
            if created:
                self._define(dataset, plan)
            record = dataset.dimensions[_TIME_DIMENSION].size
            self._write_time(dataset, record, bounds)
            for variable, parts in gathered:
                values = np.concatenate(
                    [np.asarray(item) for item in parts], axis=0
                )
                self._write_variable(dataset, variable, values, placement, record)

    def _define(self, dataset: Any, plan: _ColumnPlan) -> None:
        dataset.createDimension(_TIME_DIMENSION, None)
        dataset.createDimension(_COLUMN_DIMENSION, plan.global_count)
        dataset.createDimension("nbnd", 2)
        dataset.createDimension("chars", _CHARS)
        # CAM history files always carry both vertical axes, so a Python-owned
        # stream declares them even when this sample holds column-only fields.
        levels = int(self.driver.config.pver)
        dataset.createDimension("lev", levels)
        dataset.createDimension("ilev", levels + 1)
        dataset.setncatts(self._global_attributes())
        time = dataset.createVariable("time", "f8", (_TIME_DIMENSION,))
        time.long_name = "time"
        time.units = "days since 0001-01-01 00:00:00"
        time.calendar = "noleap"
        time.bounds = "time_bnds"
        bounds = dataset.createVariable(
            "time_bnds", "f8", (_TIME_DIMENSION, "nbnd")
        )
        bounds.long_name = "time interval endpoints"
        for name, long_name in (
            ("date", "current date (YYYYMMDD)"),
            ("datesec", "current seconds of current date"),
            ("ndcur", "current day (from base day)"),
            ("nscur", "current seconds of current day"),
            ("nsteph", "current timestep"),
        ):
            created = dataset.createVariable(name, "i4", (_TIME_DIMENSION,))
            created.long_name = long_name
        for name, long_name in (
            ("date_written", "date the sample was written"),
            ("time_written", "time the sample was written"),
        ):
            created = dataset.createVariable(
                name, "S1", (_TIME_DIMENSION, "chars")
            )
            created.long_name = long_name
        self._copy_template(dataset)

    def _write_time(
        self,
        dataset: Any,
        record: int,
        bounds: tuple[tuple[int, int, int, int], tuple[int, int, int, int]],
    ) -> None:
        start, end = bounds
        start_days = _elapsed_days(*start)
        end_days = _elapsed_days(*end)
        # CAM timestamps an averaged sample at the end of its window and an
        # instantaneous sample at the moment it was taken.
        dataset.variables["time"][record] = end_days
        dataset.variables["time_bnds"][record, :] = (
            (start_days, end_days)
            if self.spec.time_period == "mean"
            else (end_days, end_days)
        )
        year, month, day, seconds = end
        dataset.variables["date"][record] = year * 10000 + month * 100 + day
        dataset.variables["datesec"][record] = seconds
        dataset.variables["ndcur"][record] = int(end_days)
        dataset.variables["nscur"][record] = seconds
        dataset.variables["nsteph"][record] = int(self.driver.clock.nstep)
        stamp = time_module.strftime("%m/%d/%y")
        clock = time_module.strftime("%H:%M:%S")
        dataset.variables["date_written"][record, :] = _characters(stamp)
        dataset.variables["time_written"][record, :] = _characters(clock)

    def _write_variable(
        self,
        dataset: Any,
        variable: PICAMHistoryVariable,
        values: np.ndarray,
        placement: np.ndarray,
        record: int,
    ) -> None:
        name = variable.output_name
        if values.ndim == 1:
            dimensions = (_TIME_DIMENSION, _COLUMN_DIMENSION)
            ordered = np.empty(placement.size, dtype=values.dtype)
            ordered[placement] = values
        elif values.ndim == 2:
            level = self._level_name(values.shape[1])
            dimensions = (_TIME_DIMENSION, level, _COLUMN_DIMENSION)
            ordered = np.empty(
                (values.shape[1], placement.size), dtype=values.dtype
            )
            ordered[:, placement] = values.T
        else:
            raise PICAMStateError(
                f"history field {variable.field!r} gathered {values.ndim} "
                "dimensions; expected columns or columns and levels"
            )
        if name not in dataset.variables:
            created = dataset.createVariable(
                name,
                "f4" if self.spec.precision == "float32" else "f8",
                dimensions,
            )
            contract = self._contract(variable.field)
            if len(dimensions) == 3:
                created.mdims = 1
            created.units = variable.units or (
                "1" if contract is None else contract.units
            )
            created.long_name = variable.long_name or name
            created.cell_methods = self.spec.cell_methods
        dataset.variables[name][record, ...] = ordered

    # -- metadata --------------------------------------------------------

    def _level_name(self, size: int) -> str:
        if size == int(self.driver.config.pver):
            return "lev"
        if size == int(self.driver.config.pver) + 1:
            return "ilev"
        raise PICAMStateError(
            f"history output cannot label a vertical dimension of {size} "
            "levels; CAM history files use lev and ilev"
        )

    def _contract(self, field: str) -> Any:
        pool = self.driver.pool
        try:
            return pool.contract(pool.canonical_name(field))
        except (KeyError, PICAMStateError):
            return None

    def _global_attributes(self) -> dict[str, Any]:
        attributes: dict[str, Any] = {
            "Conventions": "CF-1.0",
            "source": "CAM",
            "case": str(self.driver.config.case_name),
            "title": "UNSET",
            "logname": _login_name(),
            "host": socket.gethostname(),
            "freecam_history_stream": self.spec.name,
            "freecam_control": "python",
        }
        template = self._template_path()
        if template is not None:
            attributes.update(_template_attributes(template))
        return attributes

    def _template_path(self) -> Path | None:
        if self._template is not None:
            return self._template
        declared = self.spec.template
        if declared is not None:
            path = Path(declared)
            if not path.is_file():
                raise PICAMConfigurationError(
                    f"history template {path} does not exist"
                )
            self._template = path
            return path
        run_dir = self.driver.run_dir
        if run_dir is None:
            return None
        case = str(self.driver.config.case_name)
        for candidate in sorted(Path(run_dir).glob(f"{case}.cam.h?.*.nc")):
            if f".cam.{self.spec.stream}." in candidate.name:
                continue
            self._template = candidate
            return candidate
        return None

    def _copy_template(self, dataset: Any) -> None:
        template = self._template_path()
        if template is None:
            return
        from netCDF4 import Dataset

        with Dataset(template) as source:
            for name in TEMPLATE_VARIABLES:
                variable = source.variables.get(name)
                if variable is None or name in dataset.variables:
                    continue
                if any(
                    dimension not in dataset.dimensions
                    or dataset.dimensions[dimension].size
                    != source.dimensions[dimension].size
                    for dimension in variable.dimensions
                ):
                    continue
                created = dataset.createVariable(
                    name, variable.dtype, variable.dimensions
                )
                created.setncatts(
                    {key: variable.getncattr(key) for key in variable.ncattrs()}
                )
                created[...] = variable[...]


def _characters(text: str) -> np.ndarray:
    padded = f"{text:<{_CHARS}}"[:_CHARS]
    return np.asarray(list(padded), dtype="S1")


class PICAMHistoryStreamRegistry:
    """Own the Python-defined history streams of one live model."""

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
        name: str,
        *,
        fields: Sequence[Any],
        stream: str = "h9",
        nhtfrq: int = 0,
        mfilt: int = 1,
        time_period: str = "mean",
        template: str | Path | None = None,
        precision: str = "float32",
    ) -> PICAMHistoryStream:
        """Install one Python-owned history stream on every MPI rank."""

        spec = PICAMHistorySpec(
            name=str(name),
            stream=str(stream),
            variables=tuple(_as_variable(item) for item in fields),
            nhtfrq=int(nhtfrq),
            mfilt=int(mfilt),
            time_period=str(time_period),
            template=template,
            precision=str(precision),
        )
        if spec.name in self._streams:
            raise PICAMConfigurationError(
                f"history stream {spec.name!r} is already installed"
            )
        used = {item.spec.stream for item in self._streams.values()}
        if spec.stream in used:
            raise PICAMConfigurationError(
                f"history stream identifier {spec.stream!r} is already in use"
            )
        for variable in spec.variables:
            self._require_field(variable.field)
        installed = PICAMHistoryStream(self.driver, spec)
        self._streams[spec.name] = installed
        return installed

    def remove(self, name: str) -> PICAMHistorySpec:
        stream = self.stream(name)
        del self._streams[stream.spec.name]
        return stream.spec

    def step(self, name: str) -> Path | None:
        return self.stream(name).step()

    def flush(self, name: str) -> Path | None:
        return self.stream(name).flush()

    def describe(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                **stream.spec.to_payload(),
                "writes": stream.writes,
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
