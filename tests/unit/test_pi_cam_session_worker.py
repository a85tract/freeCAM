"""Worker command dispatch against a bounded driver, without live MPI."""

from __future__ import annotations

import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

try:  # pragma: no cover - exercised only where an MPI runtime exists
    from mpi4py import MPI  # noqa: F401

    _STUBBED = False
except ImportError:
    # The worker's command dispatch never touches MPI symbols for the
    # operations tested here; only ``main()`` and the ``stats`` op do.
    # Stub mpi4py just long enough to import the worker module, then restore
    # ``sys.modules`` so the rest of the test session sees reality.
    _STUBBED = True
    _saved = {name: sys.modules.get(name) for name in ("mpi4py", "mpi4py.MPI")}
    _package = types.ModuleType("mpi4py")
    _package.MPI = types.SimpleNamespace()
    sys.modules["mpi4py"] = _package
    sys.modules["mpi4py.MPI"] = types.ModuleType("mpi4py.MPI")

from freecam.pi_cam import session_worker as session_worker_module  # noqa: E402

if _STUBBED:
    for _name, _module in _saved.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    RecordingCAMBackend,
)
_command = session_worker_module._command
_parse_trace_limit = session_worker_module._parse_trace_limit
_status = session_worker_module._status


class FakeComm:
    def __init__(self, rank: int = 0, size: int = 1) -> None:
        self.rank = rank
        self.size = size


def _bounded_driver(trace_limit: int | None = 3) -> PICAMDriver:
    config = PICAMConfig(
        case_name="unit-pi-cam-worker",
        source_root=Path("/tmp/source"),
        mpi_size=1,
        stop_n=2,
    )
    boundary = InMemoryBoundaryProvider(
        {
            (0, 0): {"sst": np.full((2,), 280.0)},
            (1, 0): {"sst": np.full((2,), 281.0)},
            (2, 0): {"sst": np.full((2,), 282.0)},
        }
    )
    driver = PICAMDriver(
        config,
        boundary,
        RecordingCAMBackend(),
        rank=0,
        size=1,
        trace_limit=trace_limit,
    )
    driver.initialize()
    driver.step()
    return driver


def test_worker_status_reports_bounded_trace_counters() -> None:
    driver = _bounded_driver(trace_limit=3)

    status = _status(driver)

    assert status["actions"] == driver.trace_count
    assert status["trace_retained"] == 3
    assert status["trace_first_sequence"] == driver.trace_count - 3
    assert status["trace_limit"] == 3


def test_worker_trace_reply_carries_header() -> None:
    driver = _bounded_driver(trace_limit=3)

    reply = _command({"op": "trace", "since": 0}, driver, FakeComm(rank=0))

    assert reply["total"] == driver.trace_count
    assert reply["first_sequence"] == driver.trace_first_sequence
    assert reply["first_sequence"] > 0
    assert reply["records"] == [asdict(record) for record in driver.trace]

    empty = _command(
        {"op": "trace", "since": driver.trace_count}, driver, FakeComm(rank=0)
    )
    assert empty["records"] == []
    assert empty["first_sequence"] == driver.trace_count

    with pytest.raises(ValueError, match="trace cursor"):
        _command(
            {"op": "trace", "since": driver.trace_count + 1},
            driver,
            FakeComm(rank=0),
        )


def test_worker_trace_returns_none_off_rank_zero() -> None:
    driver = _bounded_driver(trace_limit=3)

    reply = _command(
        {"op": "trace", "since": 10**6}, driver, FakeComm(rank=1, size=2)
    )

    assert reply is None


def test_parse_trace_limit_accepts_none_and_integers() -> None:
    assert _parse_trace_limit("none") is None
    assert _parse_trace_limit("None") is None
    assert _parse_trace_limit("4096") == 4096
