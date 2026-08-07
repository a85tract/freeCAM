from pathlib import Path

import pytest

from freecam import CCPPDeviceHost, CCPPSuitePlan, DeviceCatalog
from freecam.model.errors import MissingKernelError


ROOT = Path(__file__).resolve().parents[2]
SUITES = ROOT / "external/CAM-SIMA/src/physics/ncar_ccpp/suites"


def test_every_pinned_suite_compiles_to_a_lossless_plan():
    catalog = DeviceCatalog.discover(ROOT)
    for suite_file in catalog.suites:
        plan = CCPPSuitePlan.from_xml(suite_file)
        expected = sum(
            sum(
                occurrence.suite == plan.name
                for occurrence in entry.occurrences
            )
            for entry in catalog.entries.values()
        )
        assert len(plan.schemes) == expected
        assert plan.sequence_safe


def test_cam4_subcycles_expand_as_complete_blocks():
    plan = CCPPSuitePlan.from_xml(SUITES / "suite_cam4.xml")
    plain = sum(
        1
        for row in plan.describe("physics_before_coupler")
        if not row["controls"]
    )
    looped = sum(
        1
        for row in plan.describe("physics_before_coupler")
        if row["controls"]
    )
    expanded = plan.expanded(
        "physics_before_coupler",
        {"number_of_diagnostic_subcycles": 3},
    )
    assert len(expanded) == plain + 3 * looped
    restored = CCPPSuitePlan.from_payload(plan.to_payload())
    assert [
        item.key
        for item in restored.expanded(
            "physics_before_coupler",
            {"number_of_diagnostic_subcycles": 3},
        )
    ] == [item.key for item in expanded]


def test_generic_plan_supports_cross_group_move_and_disable():
    plan = CCPPSuitePlan.from_xml(SUITES / "suite_kessler.xml")
    kessler = plan.scheme("kessler", group="physics_before_coupler")
    plan.move(
        kessler.key,
        to_group="physics_after_coupler",
        unsafe=True,
    )
    assert plan.execution_group(kessler.key) == "physics_after_coupler"
    plan.disable(kessler.key, unsafe=True)
    assert not plan.scheme(kessler.key).enabled
    assert not plan.sequence_safe


class _Pool:
    dimensions = {"number_of_diagnostic_subcycles": 1}

    def pointer_records(self):
        return {}

    def assert_pointer_stability(self, _before):
        return None


class _Registry:
    def __init__(self, names):
        self.process_names = frozenset(names)
        self.calls = []

    def invoke(self, process, _pool):
        self.calls.append(process)


def test_device_host_routes_scheme_and_lifecycle_processes():
    plan = CCPPSuitePlan.from_xml(SUITES / "suite_musica.xml")
    names = {item.name for item in plan.schemes} | {
        f"{item.name}:initialize" for item in plan.schemes
    }
    registry = _Registry(names)
    host = CCPPDeviceHost(_Pool(), registry, plan)
    initialized = host.run_lifecycle("initialize")
    executed = host.run_group("physics_after_coupler")
    assert initialized
    assert len(executed) == 3
    assert registry.calls[-3:] == [item.name for item in plan.schemes]


def test_device_host_fails_closed_when_a_kernel_is_not_built():
    plan = CCPPSuitePlan.from_xml(SUITES / "suite_musica.xml")
    host = CCPPDeviceHost(_Pool(), _Registry(()), plan)
    with pytest.raises(MissingKernelError, match="requires process"):
        host.run_group("physics_after_coupler")
