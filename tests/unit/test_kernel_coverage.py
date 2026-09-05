"""The decoupling inventory is built from the repository's records, closes, and is committed current."""

from __future__ import annotations

import json
from pathlib import Path

from freecam.pi_cam import kernel_coverage

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "validation/physics_kernel_decoupling.json"


def test_the_inventory_closes_over_the_step_plan_and_the_catalog() -> None:
    record = kernel_coverage.build_coverage()
    assert kernel_coverage.check_closure(record) == []
    summary = record["summary"]
    assert summary["actions"] == 58 and summary["enabled"] == 47
    by_id = {a["id"]: a for a in record["actions"]}
    # the eleven disabled actions are each the other form of enabled work, never a hole
    disabled = [a for a in record["actions"] if not a["enabled"]]
    assert len(disabled) == 11 and all(a["alternate_of"] for a in disabled)
    assert by_id["cam_run1.wet_deposition"]["alternate_of"] == [
        "cam_run1.modal_aerosol_preparation_leaf", "cam_run1.aerosol_wet_deposition_leaf",
        "cam_run1.carma_wet_deposition_leaf", "cam_run1.convective_tracer_transport_leaf"]
    assert by_id["cam_run1.macro_tend_pre_leaf"]["alternate_of"] == ["cam_run1.cloud_macro_microphysics"]
    # non-physics actions are out of scope and say so
    assert by_id["clock.advance_timestep"]["classification"] == "clock"
    assert by_id["cam_run3.dynamics"]["classification"] == "dynamics"
    assert by_id["cam_run4.history"]["classification"] == "io"
    assert by_id["cam_run2.physics_buffer_deallocate_leaf"]["classification"] == "host_service"
    assert by_id["cam_run1.prepare"]["classification"] == "process_control"
    # the two stages Python drives, and what they expose
    stage7 = by_id["cam_run1.cloud_macro_microphysics"]
    assert stage7["python_class"].endswith("CloudMacroMicrophysics")
    assert stage7["kernels"] == ["mmacro_pcond", "micro_mg_tend"] and stage7["coverage"] == "partial"
    assert stage7["performance"] == ["performance_overhead.md", "pi_cam_native_whole_1month_median.json"]
    radiation = by_id["cam_run1.radiation"]
    assert radiation["kernels"] == ["rad_rrtmg_sw", "rad_rrtmg_lw"] and radiation["coverage"] == "partial"
    # a leaf's execution is counted from the recorded runs, at both lengths
    assert by_id["cam_run1.aerosol_wet_deposition_leaf"]["execution"] == {"online_50step": 51, "month_1488step": 1489}
    assert by_id["cam_run1.macro_tend_pre_leaf"]["execution"] == {"online_50step": 0, "month_1488step": 0}
    # an inert-by-configuration action is not counted as covered, and is listed as unresolved
    assert by_id["cam_run2.rayleigh_friction"]["activity"] == "inert-by-configuration"
    assert by_id["cam_run2.rayleigh_friction"]["coverage"] == "not-required-in-this-configuration"
    assert any(u["what"] == "cam_run2.rayleigh_friction" for u in record["unresolved"])


def test_every_kernel_row_is_a_kernel_a_stage_class_describes_and_only_mmacro_pcond_has_closed_the_loop() -> None:
    record = kernel_coverage.build_coverage()
    rows = {k["kernel"]: k for k in record["kernels"]}
    assert set(rows) == {"mmacro_pcond", "micro_mg_tend", "rad_rrtmg_sw", "rad_rrtmg_lw"}
    pcond = rows["mmacro_pcond"]
    assert pcond["status"] == "complete" and pcond["missing"] == []
    assert pcond["contract"] == "reviewed" and pcond["bindable"] and pcond["validated_through_runner"]
    assert all(pcond["evidence"][step] for step in kernel_coverage.EVIDENCE_PATTERNS)
    assert pcond["in_model_gates"][0]["bfb"] is True and pcond["in_model_gates"][0]["path"] == "segmented"
    micro = rows["micro_mg_tend"]
    assert micro["status"] == "open" and "segment_runner" in micro["missing"] and micro["contract"] == "reviewed"
    assert micro["in_model_gates"][0]["bfb"] is True          # the walk with the core through its image
    for name in ("rad_rrtmg_sw", "rad_rrtmg_lw"):
        assert rows[name]["contract"] == "draft" and "reviewed_contract" in rows[name]["missing"]
    assert record["summary"]["kernels_by_status"] == {"complete": 1, "open": 3}


def test_the_committed_record_is_what_the_builder_writes_now() -> None:
    committed = json.loads(RECORD.read_text())
    current = kernel_coverage.build_coverage()
    assert committed["coverage_hash"] == current["coverage_hash"], \
        "validation/physics_kernel_decoupling.json is stale; run tools/build_physics_kernel_coverage.py"
    assert kernel_coverage.coverage_hash(committed) == committed["coverage_hash"]


def test_the_record_names_no_site_path_and_no_person() -> None:
    text = RECORD.read_text()
    assert "/glade" not in text and "/home/" not in text
    for row in json.loads(text)["kernels"]:
        for files in row["evidence"].values():
            assert all(not name.startswith("/") for name in files)
