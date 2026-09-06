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
    # an inert action is confirmed by the inertness gate (7333956: the eleven disabled, bit-for-bit)
    # and is then neither counted as covered nor listed as unresolved
    rayleigh = by_id["cam_run2.rayleigh_friction"]
    assert rayleigh["activity"] == "inert-confirmed" and "7333956" in rayleigh["activity_basis"]
    assert rayleigh["coverage"] == "not-required-in-this-configuration"
    assert not any(u["what"] == "cam_run2.rayleigh_friction" for u in record["unresolved"])
    assert sum(a["activity"] == "inert-confirmed" for a in record["actions"]) == 11


def test_an_inert_action_stays_unconfirmed_without_the_gate_s_record(monkeypatch) -> None:
    monkeypatch.setattr(kernel_coverage, "INERT_GATE", ("missing_summary.json", "missing_bfb.json"))
    record = kernel_coverage.build_coverage()
    by_id = {a["id"]: a for a in record["actions"]}
    assert by_id["cam_run2.rayleigh_friction"]["activity"] == "inert-by-configuration"
    assert any(u["what"] == "cam_run2.rayleigh_friction" for u in record["unresolved"])


def test_every_kernel_row_is_a_kernel_a_stage_class_describes_and_two_have_closed_the_loop() -> None:
    record = kernel_coverage.build_coverage()
    rows = {k["kernel"]: k for k in record["kernels"]}
    assert set(rows) == {"mmacro_pcond", "micro_mg_tend", "rad_rrtmg_sw", "rad_rrtmg_lw",
                         "dadadj", "compute_uwshcu_inv",
                         "zm_convr", "zm_conv_evap", "momtran", "convtran",
                         "compute_tms", "compute_eddy_diff", "compute_vdiff", "gw_drag_prof",
                         "wetdepa_v2", "modal_aero_depvel_part", "gas_phase_chemdr"}
    # the pausable stages: dadadj has a reviewed contract and the runner pauses at it
    assert rows["dadadj"]["bindable"] and rows["dadadj"]["contract"] == "reviewed"
    pcond = rows["mmacro_pcond"]
    assert pcond["status"] == "complete" and pcond["missing"] == []
    assert pcond["contract"] == "reviewed" and pcond["bindable"] and pcond["validated_through_runner"]
    assert all(pcond["evidence"][step] for step in kernel_coverage.EVIDENCE_PATTERNS)
    assert pcond["in_model_gates"][0]["bfb"] is True and pcond["in_model_gates"][0]["path"] == "segmented"
    micro = rows["micro_mg_tend"]
    assert micro["status"] == "open" and micro["contract"] == "reviewed" and micro["bindable"]
    assert micro["validated_through_runner"]                    # gate 7331040
    assert "segment_runner" not in micro["missing"] and "in_model_replacement_bfb" not in micro["missing"]
    assert "capture" in micro["missing"]                        # no captured calls replayed through its image yet
    assert record["summary"]["kernels_validated_through_runner"] == 17     # every exposed kernel, through 7335681
    assert micro["in_model_gates"][0]["bfb"] is True          # the walk with the core through its image
    # the pause gates the manifest names are in-model evidence too (7331040, 7331041)
    assert [g["record"] for g in micro["in_model_gates"][1:]] == [
        "pi_cam_stage7_segmented_micro_50step.json", "pi_cam_stage7_segmented_both_50step.json"]
    # the pausable classes' kernels: dadadj has closed the loop (7333952, 7333955), uwshcu lacks capture/replay
    dadadj = rows["dadadj"]
    assert dadadj["status"] == "complete" and dadadj["validated_through_runner"]
    assert all(g["bfb"] is True and g["path"].startswith("segmented") for g in dadadj["in_model_gates"])
    uwshcu = rows["compute_uwshcu_inv"]
    assert uwshcu["status"] == "open" and uwshcu["validated_through_runner"]
    assert "in_model_replacement_bfb" not in uwshcu["missing"] and "capture" in uwshcu["missing"]
    for name in ("rad_rrtmg_sw", "rad_rrtmg_lw"):
        # the frame descriptor is the contract of a kernel taking derived types, and the
        # radt runner pauses at both cores; the in-model gate is the walk's until the pause gates run
        assert rows[name]["contract"] == "frame" and rows[name]["contract_path"] == "native/pi_cam/segment_frames.yaml"
        assert rows[name]["bindable"] and rows[name]["validated_through_runner"]      # gates 7334070-7334073
        assert "reviewed_contract" not in rows[name]["missing"] and "segment_runner" not in rows[name]["missing"]
        assert "in_model_replacement_bfb" not in rows[name]["missing"]
        assert [g["record"] for g in rows[name]["in_model_gates"][1:]][-1] == "pi_cam_pausable_all-pause_50step.json"
    assert record["summary"]["kernels_by_status"] == {"complete": 2, "open": 15}     # P3-P5 kernels await capture and replay


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
