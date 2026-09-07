"""The library and the snapshot come from the model's own records."""

import json

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
    for forbidden in ("/glade/", "UCUB", "ruitong", "$HOME"):
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
