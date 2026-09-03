"""The device's kernel dispatch, with stub adapters in the libraries' places.

Every stage test drives a fake native and never reaches
``NativeCAMDevice._adapter_for``; a refactor of it once shipped a NameError
that 512 ranks found together (job 7302899).  This exercises the real
dispatch -- one-shot and bound -- against adapters that only record.
"""

from __future__ import annotations

import numpy as np
import pytest

from freecam.core.fortran_adapter import FortranAdapterError
from freecam.pi_cam.errors import NativeCAMError
from freecam.pi_cam.native import NativeCAMDevice


class _Adapter:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name, self.fail = name, fail
        self.calls: list[tuple] = []
        self.binds: list[tuple] = []
        self.runs: list[str] = []

    def call(self, operation, pool, *, fcomm):
        if self.fail:
            raise FortranAdapterError(f"{self.name} refused {operation}")
        self.calls.append((operation, tuple(pool), fcomm))

    def bind(self, operation, pool, *, fcomm):
        if self.fail:
            raise FortranAdapterError(f"{self.name} refused to bind {operation}")
        self.binds.append((operation, tuple(pool), fcomm))

        def run():
            self.runs.append(operation)
        return run


def _device(*, promoted: _Adapter, main: _Adapter) -> NativeCAMDevice:
    device = object.__new__(NativeCAMDevice)
    device._leaf_operation_names = frozenset()
    device._promoted_kernel_operation_names = frozenset({"direct_kernel.promoted"})
    device._native_initialized = True
    device._promoted_kernel_abi = promoted
    device._abi = main
    device.direct_kernels = ("promoted", "plain")
    return device


def test_a_kernel_is_dispatched_to_the_adapter_that_holds_it() -> None:
    promoted, main = _Adapter("promoted"), _Adapter("main")
    device = _device(promoted=promoted, main=main)
    pool = {"x": np.zeros((2, 1), order="F")}

    device.execute_kernel("promoted", pool, fcomm=7)
    device.execute_kernel("plain", pool, fcomm=7)

    assert promoted.calls == [("direct_kernel.promoted", ("x",), 7)]
    assert main.calls == [("direct_kernel.plain", ("x",), 7)]


def test_a_bound_kernel_binds_once_and_runs_on_each_call() -> None:
    promoted, main = _Adapter("promoted"), _Adapter("main")
    device = _device(promoted=promoted, main=main)
    pool = {"x": np.zeros((2, 1), order="F")}

    run = device.bind_kernel("promoted", pool, fcomm=3)
    run()
    run()

    assert promoted.binds == [("direct_kernel.promoted", ("x",), 3)]
    assert promoted.runs == ["direct_kernel.promoted"] * 2
    assert promoted.calls == []                       # bound, never one-shot


def test_an_unknown_kernel_and_an_adapter_refusal_are_native_errors() -> None:
    device = _device(promoted=_Adapter("promoted", fail=True), main=_Adapter("main"))
    pool = {"x": np.zeros((2, 1), order="F")}

    with pytest.raises(NativeCAMError, match="unknown direct CAM kernel"):
        device.bind_kernel("nowhere", pool, fcomm=0)
    with pytest.raises(NativeCAMError, match="refused to bind"):
        device.bind_kernel("promoted", pool, fcomm=0)
    with pytest.raises(NativeCAMError, match="refused direct_kernel.promoted"):
        device.execute_kernel("promoted", pool, fcomm=0)
