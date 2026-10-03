"""The library and the snapshot come from the model's own records."""

import json
from pathlib import Path

import pytest

from freecam.pi_cam.plan import PICAMStepPlan
from freecam.pi_cam.segment_runner import bindable_kernels
from freecam.pi_cam.workflow_builder import (
    WorkflowDocument,
    build_snapshot,
    catalog_entries,
    default_document,
    kernel_capabilities,
    load_catalog,
    validate_document,
)
from freecam.pi_cam.workflow_builder.catalog import runtime_parameters


def test_the_default_workflow_is_the_current_step_plan() -> None:
    document = default_document()
    plan = [f"{a.phase}.{a.name}" for a in PICAMStepPlan.default().actions]
    assert list(document.ids) == plan
    assert [n.enabled for n in document.nodes] == [a.enabled for a in PICAMStepPlan.default().actions]
    assert [n.parent_stage for n in document.nodes] == [a.parent_stage for a in PICAMStepPlan.default().actions]


def test_the_library_lists_every_catalog_process_with_a_reason_when_it_cannot_be_added() -> None:
    document, entries, _ = load_catalog()
    from_catalog = [e for e in entries.values() if e.node.origin == "catalog"]
    assert from_catalog, "the physics catalog contributes entries"
    for entry in from_catalog:
        assert entry.addable or entry.reason, entry.id
        assert entry.category == "Catalog process"
    for node in document.nodes:
        assert node.id in entries
        assert entries[node.id].addable == node.scientific


def test_kernel_capabilities_follow_the_runner_not_the_catalog() -> None:
    capabilities = {c.kernel: c for c in kernel_capabilities()}
    assert set(capabilities) >= {"mmacro_pcond", "micro_mg_tend", "rad_rrtmg_sw", "rad_rrtmg_lw"}
    runner_kernels = set(bindable_kernels())
    for name, capability in capabilities.items():
        assert capability.bindable == (name in runner_kernels), name
        if not capability.bindable:
            assert capability.reason and "runner" in capability.reason
    assert capabilities["mmacro_pcond"].validated
    assert capabilities["mmacro_pcond"].evidence
    assert capabilities["mmacro_pcond"].stage_action == "cam_run1.cloud_macro_microphysics"


def test_the_stage_node_carries_its_kernel_slots_and_tunables() -> None:
    document = default_document()
    stage = document.node("cam_run1.cloud_macro_microphysics")
    assert set(stage.configuration.kernels) >= {"mmacro_pcond", "micro_mg_tend"}
    assert all(not binding.replaces for binding in stage.configuration.kernels.values())
    names = {p["name"] for p in stage.metadata["parameters"]}
    assert "cldfrc_rhminl" in names
    deep = document.node("cam_run1.deep_convection")
    assert {p["name"] for p in deep.metadata["parameters"]} >= {"zmconv_c0_lnd", "zmconv_ke"}


def test_runtime_parameters_are_grouped_by_the_action_that_reads_them() -> None:
    grouped = runtime_parameters()
    assert "cam_run1.deep_convection" in grouped
    for action, parameters in grouped.items():
        assert action.count(".") == 1
        for parameter in parameters:
            assert set(parameter) == {"name", "dtype", "notes"}


def test_the_snapshot_is_reproducible_and_carries_no_paths_or_accounts() -> None:
    first = build_snapshot(stamp=False)
    second = build_snapshot(stamp=False)
    assert first["catalog_hash"] == second["catalog_hash"]
    assert first == second
    text = json.dumps(first)
    import getpass

    site = Path(__file__).resolve().parents[2] / "site.env"
    # the login name is a site fact: forbidden by name only where a site is configured
    for forbidden in ("/glade/", "UCUB", "$HOME") + ((getpass.getuser(),) if site.exists() else ()):
        assert forbidden not in text, forbidden
    document = WorkflowDocument.from_payload(first["default_document"])
    assert document.catalog_version == first["catalog_hash"]
    assert first["rules"]["control_skeleton"] == ["boundary_import", "advance_timestep", "boundary_export"]
    assert "cam_run1.cloud_macro_microphysics" in first["rules"]["parent_leaf_groups"]


def test_the_stamped_snapshot_names_its_commit_and_time() -> None:
    snapshot = build_snapshot()
    assert "generated_at" in snapshot and snapshot["generated_at"].endswith("+00:00")
    assert "commit" in snapshot


def test_the_default_document_passes_its_own_check_at_both_levels() -> None:
    document, entries, snapshot = load_catalog()
    for level in ("browser", "local"):
        report = validate_document(document, default=document, catalog=entries, level=level,
                                   catalog_version=snapshot["catalog_hash"])
        assert report.status == "valid", report.to_payload()
        assert report.checks["not_verified"] == []


@pytest.mark.parametrize("case", ["PI-atm", "PI-atm-replay", "PI-atm-1month", "PI-atm-online"])
def test_every_case_a_driver_accepts_has_a_default_document(case) -> None:
    assert default_document(case, 5).case == case


def test_an_unknown_case_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown case"):
        default_document("PI-atm-other")


def test_every_scientific_process_has_its_science_and_every_phase_a_name() -> None:
    snapshot = build_snapshot(stamp=False)
    entries = {entry["id"]: entry for entry in snapshot["entries"]}
    for node in snapshot["default_nodes"]:
        assert node["phase"] in snapshot["phases"], node["phase"]
        if not node["scientific"]:
            continue
        science = entries[node["id"]]["science"]
        assert science and science["title"] and science["summary"], node["id"]
        for equation in science["equations"]:
            assert equation["tex"].strip(), node["id"]
        for reference in science["references"]:
            assert reference["citation"] and reference["url"].startswith("https://"), reference
    labels = {phase: info["label"] for phase, info in snapshot["phases"].items()}
    assert labels["cam_run1"] == "Physics before surface coupling"
    assert labels["cam_run2"] == "Physics after surface coupling"
    assert labels["cam_run3"] == "Dynamics"
    # a catalogued sub-process shows its stage's science on the page, not a copy of it
    assert all(entry["science"] is None for entry in snapshot["entries"] if entry["origin"] == "catalog")


def test_a_process_inert_in_this_configuration_says_so() -> None:
    _, entries, _ = load_catalog()
    for inert in ("cam_run2.rayleigh_friction", "cam_run2.qbo_relaxation", "cam_run2.ion_drag"):
        assert entries[inert].science["active"] is False, inert
    for working in ("cam_run1.deep_convection", "cam_run1.radiation", "cam_run3.dynamics"):
        assert entries[working].science["active"] is True, working


@pytest.mark.parametrize("change, message", [
    (lambda r: r["processes"].update({"cam_run9.nothing": {"title": "x"}}), "not an action of the step plan"),
    (lambda r: r["processes"]["cam_run3.dynamics"].update({"colour": "blue"}), "unknown fields"),
    (lambda r: r["processes"]["cam_run3.dynamics"].update({"references": ["nobody1999"]}), "no reference"),
    (lambda r: r["phases"].pop("cam_run1"), "phases without a label"),
])
def test_the_science_record_fails_closed(monkeypatch, change, message) -> None:
    import copy

    from freecam.pi_cam.workflow_builder import catalog

    record = copy.deepcopy(catalog._load_science())
    change(record)
    monkeypatch.setattr(catalog, "_load_science", lambda: record)
    with pytest.raises(ValueError, match=message):
        catalog.process_science(default_document().nodes)
