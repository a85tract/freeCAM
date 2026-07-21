from pathlib import Path
from xml.etree import ElementTree

import pytest

from pycam_sima.model import (
    KesslerSchemePlan,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
)


PROJECT = Path(__file__).resolve().parents[2]
SUITE = (
    PROJECT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


def test_default_plan_exactly_matches_pinned_ccpp_suite() -> None:
    root = ElementTree.parse(SUITE).getroot()
    expected = {
        group.attrib["name"]: tuple(
            scheme.text.strip() for scheme in group.findall("scheme")
        )
        for group in root.findall("group")
    }
    plan = KesslerSchemePlan.default()
    actual = {
        group: tuple(scheme.name for scheme in plan.active(group))
        for group in (PHYSICS_BEFORE_COUPLER, PHYSICS_AFTER_COUPLER)
    }
    assert actual == expected
    assert len(plan.keys) == 24
    assert plan.sequence_safe


def test_required_scheme_changes_must_be_explicitly_unsafe() -> None:
    plan = KesslerSchemePlan.default()
    with pytest.raises(ValueError, match="unsafe=True"):
        plan.disable("kessler")
    plan.disable("kessler", unsafe=True)
    assert not plan.scheme("kessler").enabled
    assert not plan.sequence_safe
    plan.enable("kessler")
    assert plan.sequence_safe

    with pytest.raises(ValueError, match="unsafe=True"):
        plan.move("kessler", after="kessler_update")
    plan.move("kessler", after="kessler_update", unsafe=True)
    assert not plan.sequence_safe
    plan.reset()
    assert plan.sequence_safe


def test_duplicate_scheme_name_requires_group_and_payload_round_trips() -> None:
    plan = KesslerSchemePlan.default()
    with pytest.raises(ValueError, match="ambiguous"):
        plan.scheme("check_energy_scaling")
    assert (
        plan.scheme(
            "check_energy_scaling", group=PHYSICS_AFTER_COUPLER
        ).group
        == PHYSICS_AFTER_COUPLER
    )
    restored = KesslerSchemePlan.from_payload(plan.to_payload())
    assert restored.keys == plan.keys
    assert restored.to_payload() == plan.to_payload()
