from pathlib import Path
from xml.etree import ElementTree

import pytest

from pycam_sima.model import (
    CCPPSuitePlan,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
)


PROJECT = Path(__file__).resolve().parents[2]
SUITE = (
    PROJECT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


def _plan() -> CCPPSuitePlan:
    return CCPPSuitePlan.from_xml(SUITE)


def test_default_plan_exactly_matches_pinned_ccpp_suite() -> None:
    root = ElementTree.parse(SUITE).getroot()
    expected = {
        group.attrib["name"]: tuple(
            scheme.text.strip() for scheme in group.findall("scheme")
        )
        for group in root.findall("group")
    }
    plan = _plan()
    actual = {
        group: tuple(scheme.name for scheme in plan.active(group))
        for group in (PHYSICS_BEFORE_COUPLER, PHYSICS_AFTER_COUPLER)
    }
    assert actual == expected
    assert len(plan.keys) == 24
    assert plan.sequence_safe


def test_required_scheme_changes_must_be_explicitly_unsafe() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="unsafe=True"):
        plan.disable("kessler")
    plan.disable("kessler", unsafe=True)
    assert not plan.scheme("kessler").enabled
    assert not plan.sequence_safe
    plan.enable("kessler")
    assert plan.sequence_safe


def test_scheme_can_move_between_coupler_groups() -> None:
    plan = _plan()
    source_key = plan.scheme("kessler").key
    plan.move(
        "kessler",
        to_group=PHYSICS_AFTER_COUPLER,
        unsafe=True,
    )
    assert "kessler" not in {
        scheme.name for scheme in plan.active(PHYSICS_BEFORE_COUPLER)
    }
    assert plan.active(PHYSICS_AFTER_COUPLER)[-1].key == source_key
    moved = plan.scheme(source_key)
    assert moved.source_group == PHYSICS_BEFORE_COUPLER
    assert moved.group == PHYSICS_AFTER_COUPLER
    described = plan.describe(PHYSICS_AFTER_COUPLER)[-1]
    assert described["source_group"] == PHYSICS_BEFORE_COUPLER
    assert described["execution_group"] == PHYSICS_AFTER_COUPLER
    assert not plan.sequence_safe

    restored = CCPPSuitePlan.from_payload(plan.to_payload())
    assert restored.to_payload() == plan.to_payload()
    plan.reset()
    assert plan.scheme("kessler").group == PHYSICS_BEFORE_COUPLER
    assert plan.sequence_safe

    plan.move("kessler", before="thermo_water_update", unsafe=True)
    assert plan.active(PHYSICS_AFTER_COUPLER)[0].key == source_key

    with pytest.raises(ValueError, match="unsafe=True"):
        plan.move("kessler", after="kessler_update")
    plan.move("kessler", after="kessler_update", unsafe=True)
    assert not plan.sequence_safe
    plan.reset()
    assert plan.sequence_safe


def test_duplicate_scheme_name_requires_group_and_payload_round_trips() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="ambiguous"):
        plan.scheme("check_energy_scaling")
    assert (
        plan.scheme(
            "check_energy_scaling", group=PHYSICS_AFTER_COUPLER
        ).group
        == PHYSICS_AFTER_COUPLER
    )
    restored = CCPPSuitePlan.from_payload(plan.to_payload())
    assert restored.keys == plan.keys
    assert restored.to_payload() == plan.to_payload()

    before_key = plan.scheme(
        "check_energy_scaling", group=PHYSICS_BEFORE_COUPLER
    ).key
    plan.move(
        before_key,
        to_group=PHYSICS_AFTER_COUPLER,
        unsafe=True,
    )
    with pytest.raises(ValueError, match="ambiguous"):
        plan.scheme("check_energy_scaling", group=PHYSICS_AFTER_COUPLER)
    assert plan.scheme(before_key).group == PHYSICS_AFTER_COUPLER
