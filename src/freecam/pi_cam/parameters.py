"""Runtime-writable CAM physics tunables, bound from Fortran module storage.

CAM copies namelist tunables into Fortran module variables during
initialization and never consults the file again, so changing a parameter
on a running model means writing those module variables directly.  This
module binds a hand-audited set of them -- variables proven to be re-read
by a run-phase routine on every timestep -- as zero-copy 0-d StatePool
views, and applies changes with the same collective discipline as Python
process parameters: all ranks at the same step boundary, value agreement
verified across ranks, every storage location of a parameter written
together.

Admission is data, not code: ``native/pi_cam/runtime_parameters.yaml``
names each parameter's namelist id, its audited module symbols, and the
workflow action it tunes.  Every binding self-verifies at initialization
by comparing the value read through the symbol against the value parsed
from the run directory's ``atm_in``; any mismatch disables that parameter
instead of risking a write through the wrong address.

Two behaviours to know: after initialization each rank's module copy is
independent (CAM never re-communicates them), so writes are collective by
construction here; and these values are not part of any restart file, so
a run restarted from CAM restart files silently reverts to the namelist
values -- re-apply runtime changes after a restart.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .errors import PICAMConfigurationError, PICAMStateError
from .state import PICAMFieldContract

_TABLE_RELATIVE_PATH = Path("native/pi_cam/runtime_parameters.yaml")
_ADMITTED_DTYPES = {"float64": ctypes.c_double}
_FIELD_PREFIX = "parameters."


@dataclass(frozen=True, slots=True)
class PICAMRuntimeParameter:
    """One audited runtime-writable tunable from the curated table."""

    name: str
    dtype: str
    symbols: tuple[str, ...]
    workflow_action: str
    readers: str
    notes: str


@dataclass(slots=True)
class _BoundParameter:
    parameter: PICAMRuntimeParameter
    views: tuple[np.ndarray, ...]
    # The ctypes objects anchor the views' lifetimes.
    anchors: tuple[Any, ...]
    baseline: float


def default_table_path(project_root: str | Path | None = None) -> Path:
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return root / _TABLE_RELATIVE_PATH


def load_runtime_parameters(
    path: str | Path | None = None,
) -> dict[str, PICAMRuntimeParameter]:
    """Load and validate the curated runtime-parameter table, fail closed."""

    table_path = Path(path) if path is not None else default_table_path()
    if not table_path.is_file():
        raise PICAMConfigurationError(
            f"runtime parameter table not found: {table_path}"
        )
    data = yaml.safe_load(table_path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise PICAMConfigurationError(
            f"{table_path} must declare schema_version: 1"
        )
    entries = data.get("parameters")
    if not isinstance(entries, dict) or not entries:
        raise PICAMConfigurationError(f"{table_path} declares no parameters")
    admitted = {"dtype", "symbols", "workflow_action", "readers", "notes"}
    table: dict[str, PICAMRuntimeParameter] = {}
    for name, entry in entries.items():
        key = str(name).strip().lower()
        if not isinstance(entry, dict):
            raise PICAMConfigurationError(
                f"runtime parameter {key!r} must be a mapping"
            )
        unknown = sorted(set(entry) - admitted)
        if unknown:
            raise PICAMConfigurationError(
                f"runtime parameter {key!r} has unsupported keys: "
                + ", ".join(unknown)
            )
        dtype = str(entry.get("dtype", ""))
        if dtype not in _ADMITTED_DTYPES:
            raise PICAMConfigurationError(
                f"runtime parameter {key!r} has unsupported dtype {dtype!r}"
            )
        symbols = tuple(str(item) for item in entry.get("symbols", ()))
        if not symbols:
            raise PICAMConfigurationError(
                f"runtime parameter {key!r} names no symbols"
            )
        table[key] = PICAMRuntimeParameter(
            name=key,
            dtype=dtype,
            symbols=symbols,
            workflow_action=str(entry.get("workflow_action", "")),
            readers=str(entry.get("readers", "")).strip(),
            notes=str(entry.get("notes", "")).strip(),
        )
    return table


def _fortran_real(text: str) -> float:
    return float(text.strip().replace("D", "e").replace("d", "e"))


def _coerce_real(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PICAMConfigurationError(
            f"runtime parameter {name!r} takes a real number, got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise PICAMConfigurationError(
            f"runtime parameter {name!r} must be finite, got {result!r}"
        )
    return result


class PICAMModuleParameterRegistry:
    """Rank-local bindings of the audited tunables, with collective writes."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self._bound: dict[str, _BoundParameter] = {}
        self._unavailable: dict[str, str] = {}

    # -- binding ----------------------------------------------------------

    def bind(
        self, table: Mapping[str, PICAMRuntimeParameter] | None = None
    ) -> None:
        """Bind every admitted parameter whose storage self-verifies.

        Called once, after native initialization has read ``atm_in`` into
        the module variables.  A parameter whose symbols cannot be
        resolved, or whose stored value does not match the namelist file
        bit for bit, is recorded as unavailable rather than bound.
        """

        if table is None:
            try:
                table = load_runtime_parameters()
            except PICAMConfigurationError as error:
                self._unavailable["*"] = str(error)
                return
        resolver = getattr(self.driver.backend, "module_symbol", None)
        if not callable(resolver):
            self._unavailable["*"] = (
                "the numerical backend exposes no module storage"
            )
            return
        expected = self._namelist_values()
        for name, parameter in table.items():
            reason = self._bind_one(parameter, resolver, expected)
            if reason is not None:
                self._unavailable[name] = reason

    def _namelist_values(self) -> dict[str, str]:
        run_dir = getattr(self.driver, "run_dir", None)
        if run_dir is None:
            return {}
        path = Path(run_dir) / "atm_in"
        if not path.is_file():
            return {}
        from .namelist import read_values

        try:
            return read_values(path)
        except PICAMConfigurationError:
            return {}

    def _bind_one(
        self,
        parameter: PICAMRuntimeParameter,
        resolver: Any,
        expected: Mapping[str, str],
    ) -> str | None:
        raw = expected.get(parameter.name)
        if raw is None:
            return "not set in this run's atm_in; no value to verify against"
        try:
            reference = _fortran_real(raw)
        except ValueError:
            return f"unparseable atm_in value {raw!r}"
        anchors: list[Any] = []
        views: list[np.ndarray] = []
        for symbol in parameter.symbols:
            anchor = resolver(symbol, parameter.dtype)
            if anchor is None:
                return f"symbol {symbol!r} is not present in the loaded model"
            view = np.ctypeslib.as_array(anchor).reshape(())
            value = float(view[()])
            if value != reference:
                return (
                    f"storage {symbol!r} holds {value!r} but atm_in says "
                    f"{reference!r}; refusing to bind an unverified address"
                )
            anchors.append(anchor)
            views.append(view)
        contract = PICAMFieldContract(
            _FIELD_PREFIX + parameter.name,
            (),
            parameter.dtype,
            category="native_module_parameter",
            writable=False,
            restart=False,
            output=False,
            owner="native",
        )
        self.driver.pool.attach(contract, views[0])
        self._bound[parameter.name] = _BoundParameter(
            parameter=parameter,
            views=tuple(views),
            anchors=tuple(anchors),
            baseline=reference,
        )
        return None

    # -- inspection -------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return str(name).strip().lower() in self._bound

    def __len__(self) -> int:
        return len(self._bound)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._bound))

    @property
    def unavailable(self) -> Mapping[str, str]:
        return dict(self._unavailable)

    def value(self, name: str) -> float:
        return float(self._record(name).views[0][()])

    def values(self) -> dict[str, float]:
        return {name: float(record.views[0][()]) for name, record in
                sorted(self._bound.items())}

    def overrides(self) -> dict[str, tuple[float, float]]:
        """Parameters whose current value differs from the namelist value."""

        report: dict[str, tuple[float, float]] = {}
        for name, record in sorted(self._bound.items()):
            current = float(record.views[0][()])
            if current != record.baseline:
                report[name] = (record.baseline, current)
        return report

    def describe(self) -> dict[str, Any]:
        return {
            "parameters": {
                name: {
                    "value": float(record.views[0][()]),
                    "baseline": record.baseline,
                    "workflow_action": record.parameter.workflow_action,
                    "symbols": list(record.parameter.symbols),
                }
                for name, record in sorted(self._bound.items())
            },
            "unavailable": dict(self._unavailable),
        }

    def for_action(self, qualified_action: str) -> tuple[str, ...]:
        """Bound parameter names tuning one workflow action."""

        return tuple(
            name
            for name, record in sorted(self._bound.items())
            if record.parameter.workflow_action == str(qualified_action)
        )

    def _record(self, name: str) -> _BoundParameter:
        key = str(name).strip().lower()
        record = self._bound.get(key)
        if record is None:
            reason = self._unavailable.get(key) or self._unavailable.get("*")
            detail = f": {reason}" if reason else ""
            raise PICAMConfigurationError(
                f"{key!r} is not a runtime-writable parameter{detail}"
            )
        return record

    # -- collective write -------------------------------------------------

    def set(self, name: str, value: Any) -> dict[str, Any]:
        """Collectively write one parameter's every storage location.

        Takes effect the next time the owning physics routine runs; CAM
        re-reads these module variables on every call.
        """

        self.driver._require_collective_boundary(
            "set a runtime physics parameter"
        )
        comm = self.driver.comm
        record: _BoundParameter | None = None
        coerced: float | None = None
        local_error: str | None = None
        try:
            record = self._record(name)
            coerced = _coerce_real(record.parameter.name, value)
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        errors = comm.allgather(local_error)
        if any(error is not None for error in errors):
            from freecam.model.collective import collective_error_message

            message = collective_error_message(
                "runtime parameter update", errors
            )
            raise PICAMConfigurationError(
                message or "runtime parameter update failed"
            )
        assert record is not None and coerced is not None
        agreed = comm.allgather(coerced.hex())
        if len(set(agreed)) != 1:
            raise PICAMStateError(
                f"runtime parameter values differ across MPI ranks: {agreed}"
            )
        previous = float(record.views[0][()])
        for view in record.views:
            view[()] = coerced
        for symbol, view in zip(record.parameter.symbols, record.views):
            written = float(view[()])
            if written != coerced:
                raise PICAMStateError(
                    f"storage {symbol!r} reads back {written!r} after "
                    f"writing {coerced!r}"
                )
        comm.barrier()
        return {
            "name": record.parameter.name,
            "previous": previous,
            "value": coerced,
            "symbols_written": list(record.parameter.symbols),
        }
