"""A native Python process may reach the image; an ordinary one may not."""

from __future__ import annotations

import numpy as np
import pytest

from freecam.model.python_processes import (
    NativeAccess, PythonProcessContext, PythonProcessSpec, PythonProcessContractError,
)


def _process(fields, context):
    return None


def test_the_flag_is_off_by_default_and_survives_the_wire() -> None:
    plain = PythonProcessSpec.from_callable(_process, name="plain")
    assert plain.native is False
    assert PythonProcessSpec.from_mapping(plain.as_dict()).native is False
    native = PythonProcessSpec.from_callable(_process, name="native", native=True)
    assert native.native is True
    assert native.as_dict()["native"] is True
    assert PythonProcessSpec.from_mapping(native.as_dict()).native is True
    # a record written before the flag existed reads as an ordinary process
    legacy = {k: v for k, v in plain.as_dict().items() if k != "native"}
    assert PythonProcessSpec.from_mapping(legacy).native is False


def test_the_context_carries_no_native_handle_unless_asked() -> None:
    context = PythonProcessContext(
        process_name="p", group="g", rank=0, size=1, step=0, timestep_seconds=1800,
        year=1, month=1, day=1, seconds=0, calendar="NO_LEAP",
    )
    assert context.native is None


class _Backend:
    def __init__(self) -> None:
        self._library = object()
        self._operations = {
            "direct_kernel.demo": {"symbol": "s", "action_id": 0,
                                   "arguments": [{"field": "macro.x", "rank": 3}]},
        }


class _Driver:
    def __init__(self) -> None:
        self.backend = _Backend()
        self.fcomm = 7
        self.pool = {
            "grid.chunk_id": np.array([1540, 1541], dtype=np.int32),
            "grid.chunk_ncols": np.array([14, 13], dtype=np.int32),
        }
        self.calls: list[tuple[str, object]] = []

    def run_kernel(self, name, *, experimental=False, pool=None):
        assert experimental
        self.calls.append((name, pool))


def test_native_access_is_a_thin_window_on_the_driver() -> None:
    driver = _Driver()
    native = NativeAccess(driver)
    assert native.library is driver.backend._library
    assert native.fcomm == 7
    lchnk, ncol = native.chunks
    assert lchnk.tolist() == [1540, 1541] and ncol.tolist() == [14, 13]
    arrays = {"macro.x": np.zeros((2, 3, 1), order="F")}
    native.run_kernel("demo", arrays)
    assert driver.calls == [("demo", arrays)]
    assert native.kernel_arguments("demo")[0]["field"] == "macro.x"
    with pytest.raises(PythonProcessContractError, match="no direct kernel"):
        native.kernel_arguments("absent")


def test_native_access_refuses_a_backend_without_an_image() -> None:
    driver = _Driver()
    driver.backend = object()
    with pytest.raises(PythonProcessContractError, match="no loaded image"):
        NativeAccess(driver).library
