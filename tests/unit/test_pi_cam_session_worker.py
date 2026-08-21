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
    PICAMVariableSpec,
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


def test_python_parameter_ops_round_trip_through_the_dispatcher() -> None:
    driver = _bounded_driver()
    driver.define_variable(
        PICAMVariableSpec("marker", ("pcols",), initial=0.0)
    )

    def stamp(fields, context, *, properties=None):
        del context
        fields["marker"][...] = float(
            (properties or {}).get("value", -1.0)
        )

    driver.physics.install_python(
        stamp,
        name="stamper",
        after="dadadj",
        writes=("marker",),
        parameters={"properties": {"value": 1.0}},
    )

    result = _command(
        {"op": "get_python_parameters", "name": "stamper"},
        driver,
        FakeComm(rank=0),
    )
    assert result == {"properties": {"value": 1.0}}

    result = _command(
        {
            "op": "set_python_parameters",
            "name": "stamper",
            "parameters": {"properties": {"value": 7.0}},
        },
        driver,
        FakeComm(rank=0),
    )
    assert result["parameters"] == {"properties": {"value": 7.0}}

    driver.physics.process("stamper").run()
    assert np.array_equal(
        driver.pool["marker"], np.full_like(driver.pool["marker"], 7.0)
    )


def test_python_parameter_ops_reply_only_from_rank_zero() -> None:
    driver = _bounded_driver()
    driver.define_variable(
        PICAMVariableSpec("marker", ("pcols",), initial=0.0)
    )

    def stamp(fields, context, *, properties=None):
        del context, properties
        fields["marker"][...] = 1.0

    driver.physics.install_python(
        stamp,
        name="stamper",
        after="dadadj",
        writes=("marker",),
        parameters={"properties": {}},
    )
    assert (
        _command(
            {"op": "get_python_parameters", "name": "stamper"},
            driver,
            FakeComm(rank=1),
        )
        is None
    )
    assert (
        _command(
            {
                "op": "set_python_parameters",
                "name": "stamper",
                "parameters": {"properties": {"value": 2.0}},
            },
            driver,
            FakeComm(rank=1),
        )
        is None
    )


def test_module_parameter_ops_route_to_the_registry() -> None:
    driver = _bounded_driver()
    calls = []

    class RegistryStub:
        def set(self, name, value):
            calls.append(("set", name, value))
            return {"name": name, "previous": 1.0, "value": value}

        def describe(self):
            calls.append(("describe",))
            return {"parameters": {}, "unavailable": {}}

    driver.module_parameters = RegistryStub()
    result = _command(
        {"op": "set_module_parameter", "name": "zmconv_ke", "value": 2e-6},
        driver,
        FakeComm(rank=0),
    )
    assert result == {"name": "zmconv_ke", "previous": 1.0, "value": 2e-6}
    assert (
        _command(
            {"op": "get_module_parameters"}, driver, FakeComm(rank=1)
        )
        is None
    )
    assert calls == [("set", "zmconv_ke", 2e-6), ("describe",)]


class _CountingStreams:
    """Stand-in registry that records which ranks resolve stream metadata."""

    def __init__(self) -> None:
        self.described = 0

    def describe(self):
        self.described += 1
        return ({"name": "python_fields", "resolved_fields": []},)


def test_history_stream_description_reaches_every_rank() -> None:
    """Resolving a stream's fields is collective, so no rank may skip it."""

    driver = _bounded_driver()
    streams = _CountingStreams()
    driver.history_streams = streams
    driver.remove_history_stream = lambda name: None

    assert _command({"op": "history_streams"}, driver, FakeComm(rank=3)) is None
    assert streams.described == 1

    reply = _command({"op": "history_streams"}, driver, FakeComm(rank=0))
    assert reply == ({"name": "python_fields", "resolved_fields": []},)
    assert streams.described == 2

    removed = _command(
        {"op": "remove_history_stream", "name": "python_fields"},
        driver,
        FakeComm(rank=3),
    )
    assert removed is None
    assert streams.described == 3


def test_assign_expression_rejects_read_only_fields() -> None:
    driver = _bounded_driver()
    # model_timestep is a configuration scalar declared writable=False.
    with pytest.raises(ValueError, match="read-only"):
        _command(
            {
                "op": "assign_expression",
                "name": "model_timestep",
                "expression": None,
            },
            driver,
            FakeComm(rank=0),
        )
