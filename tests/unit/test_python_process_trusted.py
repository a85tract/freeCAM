"""A trusted native process: freeCAM's own stage classes, run lean by the registry."""

from __future__ import annotations

import numpy as np
import pytest

from freecam.model.errors import PythonProcessContractError, PythonProcessTaintedError
from freecam.model.python_processes import PythonProcessSpec
from freecam.pi_cam import InMemoryBoundaryProvider, PICAMConfig, PICAMDriver, RecordingCAMBackend
from pathlib import Path


def _stage(fields, context):
    """Stands in for a stage: runs the original Fortran action once, like native-whole."""

    assert fields is None                      # a trusted process gets no field view
    context.native.run_action("cam_run1.cloud_macro_microphysics")


def test_trusted_native_requires_a_native_non_transactional_fieldless_process() -> None:
    with pytest.raises(PythonProcessContractError, match="trusted_native requires"):
        PythonProcessSpec.from_callable(_stage, name="s", trusted_native=True)
    with pytest.raises(PythonProcessContractError, match="trusted_native requires"):
        PythonProcessSpec.from_callable(_stage, name="s", native=True, trusted_native=True)
    with pytest.raises(PythonProcessContractError, match="trusted_native requires"):
        PythonProcessSpec.from_callable(_stage, name="s", native=True, transactional=False,
                                        writes=("q",), trusted_native=True)
    spec = PythonProcessSpec.from_callable(_stage, name="s", native=True, transactional=False,
                                           trusted_native=True)
    assert spec.trusted_native is True
    assert PythonProcessSpec.from_mapping(spec.as_dict()).trusted_native is True
    assert PythonProcessSpec.from_mapping({**spec.as_dict(), "trusted_native": False}).trusted_native is False


def _driver(comm=None, size=1):
    config = PICAMConfig(case_name="trusted", source_root=Path("/tmp/source"), mpi_size=size, stop_n=4)
    boundary = InMemoryBoundaryProvider({(step, 0): {"sst": np.full((2,), 280.0 + step)} for step in range(6)})
    backend = RecordingCAMBackend()
    kwargs = {"communicator": comm} if comm is not None else {}
    driver = PICAMDriver(config, boundary, backend, rank=0, size=size, **kwargs)
    driver.initialize()
    driver.step_plan.set_enabled("cloud_macro_microphysics", False, phase="cam_run1", experimental=True)
    return driver, backend


class _CountingComm:
    rank, size = 0, 1

    def __init__(self) -> None:
        self.gathers: list = []

    def allgather(self, value):
        self.gathers.append(value)
        return [value]

    def gather(self, value, root=0):
        return [value]

    def barrier(self) -> None:
        return None


def test_a_trusted_process_adds_no_pointer_scan_and_no_collective_to_the_step() -> None:
    comm = _CountingComm()
    driver, backend = _driver(comm)
    scans = {"n": 0}
    original = driver.pool.pointer_records

    def counted():
        scans["n"] += 1
        return original()

    driver.pool.pointer_records = counted
    comm.gathers.clear()                       # initialization gathered too; count the step alone
    driver.step()
    scans_without, gathers_without = scans["n"], len(comm.gathers)
    calls_without = backend.calls.count("macro_microphysics")

    driver.physics.install_python(_stage, name="stage_like", after="dadadj", native=True,
                                  trusted_native=True, transactional=False, unsafe=True)
    scans["n"] = 0
    comm.gathers.clear()
    driver.step()
    assert backend.calls.count("macro_microphysics") == calls_without + 1     # the stage ran the action
    assert scans["n"] == scans_without                                       # no pointer scan of its own
    assert len(comm.gathers) == gathers_without                              # no collective of its own
    export = [g for g in comm.gathers if isinstance(g, tuple) and g[1]]
    assert export and export[-1][1] == [("stage_like", None)]               # its outcome rode on the export


def test_a_trusted_process_failure_is_raised_at_the_export_on_every_rank() -> None:
    class RemoteFailure(_CountingComm):
        size = 2

        def allgather(self, value):
            self.gathers.append(value)
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], list) and value[1]:
                other = [(name, "Traceback: rank one failed") for name, _ in value[1]]
                return [value, (None, other)]
            return [value, value]

    driver, _ = _driver(RemoteFailure(), size=2)
    driver.physics.install_python(_stage, name="stage_like", after="dadadj", native=True,
                                  trusted_native=True, transactional=False, unsafe=True)
    with pytest.raises(PythonProcessTaintedError, match="failed collectively on 1/2") as failure:
        driver.step()
    assert "rank one failed" in str(failure.value)


def test_a_trusted_process_run_outside_a_step_settles_at_once() -> None:
    driver, backend = _driver()

    def broken(fields, context):
        raise RuntimeError("the stage refused")

    process = driver.physics.install_python(broken, name="broken_stage", after="dadadj", native=True,
                                            trusted_native=True, transactional=False, unsafe=True)
    with pytest.raises(PythonProcessTaintedError, match="the stage refused"):
        process.run()
    good = driver.physics.install_python(_stage, name="stage_like", after="dadadj", native=True,
                                         trusted_native=True, transactional=False, unsafe=True)
    before = backend.calls.count("macro_microphysics")
    good.run()
    assert backend.calls.count("macro_microphysics") == before + 1


def test_a_succeeding_trusted_process_costs_the_export_one_integer_reduction_not_a_gather() -> None:
    """The fast path: the outcome (name, None) is not a failure, so the flag stays zero."""

    class Reducing(_CountingComm):
        def __init__(self) -> None:
            super().__init__()
            self.reductions = 0

        def Allreduce(self, send, recv):
            self.reductions += 1
            recv[0] = send[0]

    comm = Reducing()
    driver, backend = _driver(comm)
    driver.physics.install_python(_stage, name="stage_like", after="dadadj", native=True,
                                  trusted_native=True, transactional=False, unsafe=True)
    before_gathers, before_reductions = len(comm.gathers), comm.reductions
    driver.step()
    assert backend.calls.count("macro_microphysics") >= 1
    assert comm.reductions > before_reductions                              # the export reduced one flag
    assert not [g for g in comm.gathers[before_gathers:] if isinstance(g, tuple) and g[1]]   # and gathered nothing


def test_a_failing_trusted_process_still_raises_through_the_reduction_path() -> None:
    class ReducingFailure(_CountingComm):
        size = 2

        def Allreduce(self, send, recv):
            recv[0] = send[0]

        def allgather(self, value):
            self.gathers.append(value)
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], list) and value[1]:
                return [value, value]
            return [value, value]

    driver, _ = _driver(ReducingFailure(), size=2)

    def broken(fields, context):
        raise RuntimeError("the stage refused")

    driver.physics.install_python(broken, name="broken_stage", after="dadadj", native=True,
                                  trusted_native=True, transactional=False, unsafe=True)
    with pytest.raises(PythonProcessTaintedError, match="the stage refused"):
        driver.step()
