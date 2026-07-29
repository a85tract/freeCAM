from pathlib import Path

import pytest

from pycam_sima.model import CCPPSuitePlan, ProcessRouter
from pycam_sima.model.errors import MissingKernelError


ROOT = Path(__file__).resolve().parents[2]
KESSLER_SUITE = (
    ROOT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


class _Devices:
    def __init__(self, processes=()):
        self.process_names = frozenset(processes)

    def has_process(self, process):
        return process in self.process_names


class _Services:
    def __init__(self, processes=()):
        self.process_names = frozenset(processes)
        self.calls = []

    def invoke(self, process, pool):
        self.calls.append((process, pool))


def _scheme(name):
    return CCPPSuitePlan.from_xml(KESSLER_SUITE).scheme(name)


def test_router_prefers_explicit_component_provider() -> None:
    calls = []
    pool = object()
    router = ProcessRouter(
        devices=_Devices(("calc_exner",)),
        native_invoke=lambda name, state: calls.append(("device", name, state)),
        host_handlers={
            "physics_before_coupler.calc_exner": (
                lambda state: calls.append(("host", state))
            ),
        },
    )

    assert router.invoke(_scheme("calc_exner"), pool) == "python-host-process"
    assert calls == [("host", pool)]


def test_router_uses_device_then_declared_host_service() -> None:
    calls = []
    pool = object()
    services = _Services(("kessler_diagnostics",))
    router = ProcessRouter(
        devices=_Devices(("kessler",)),
        native_invoke=lambda name, state: calls.append((name, state)),
        host_services=services,
    )

    assert router.invoke(_scheme("kessler"), pool) == "fortran-device"
    assert calls == [("kessler", pool)]
    assert (
        router.invoke(_scheme("kessler_diagnostics"), pool)
        == "python-host-service"
    )
    assert services.calls == [("kessler_diagnostics", pool)]
    coverage = router.describe(
        (_scheme("kessler"), _scheme("kessler_diagnostics"))
    )
    assert [row["provider"] for row in coverage] == [
        "fortran-device",
        "python-host-service",
    ]


def test_router_fails_closed_for_an_unprovided_suite_process() -> None:
    router = ProcessRouter(
        devices=_Devices(),
        native_invoke=lambda _name, _pool: None,
    )
    with pytest.raises(MissingKernelError, match="no generated device"):
        router.invoke(_scheme("kessler"), object())


def test_router_noops_run_node_with_host_only_lifecycle_provider() -> None:
    services = _Services(("convect_shallow_diagnostics:initialize",))
    router = ProcessRouter(
        devices=_Devices(),
        native_invoke=lambda _name, _pool: None,
        host_services=services,
    )

    assert (
        router.invoke_process("convect_shallow_diagnostics", object())
        == "lifecycle-only-noop"
    )
    assert (
        router.provider_for_process("convect_shallow_diagnostics")
        == "lifecycle-only-noop"
    )
    assert services.calls == []
