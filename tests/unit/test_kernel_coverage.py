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
    assert stage7["kernels"] == ["mmacro_pcond", "cldfrc_fice", "instratus_condensate", "micro_mg_tend"]
    # every kernel of the cloud stage is captured and replayed bit-for-bit through its image
    # (instratus_condensate and micro_mg_tend closed by 7630946-7630955)
    assert stage7["coverage"] == "complete"
    assert stage7["performance"] == ["performance_overhead.md", "pi_cam_native_whole_1month_median.json",
                                     "pi_cam_faster_than_fortran.json"]
    # every class-owned action points at the all-class pairs; an action without a class has no performance record
    assert by_id["cam_run1.deep_convection"]["performance"] == ["pi_cam_faster_than_fortran.json"]
    assert by_id["cam_run2.rayleigh_friction"]["performance"] == ["pi_cam_faster_than_fortran.json"]
    assert by_id["cam_run3.dynamics"]["performance"] == []
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
                         "wetdepa_v2", "modal_aero_depvel_part", "gas_phase_chemdr",
                         # the kernel-API closure's first entries: a function inside an assignment, and
                         # two kernels inside compiled kernels reached through hooks
                         "virtem", "cldfrc_fice", "fluxbelowinv",
                         # the saturation-adjustment core inside mmacro_pcond, reached by a hook
                         "instratus_condensate"}
    # virtem: gates 7343257 and 7343260 answered it through its assignment pause; cldfrc_fice's
    # pause runs a hoisted copy of zm_conv_evap, which gate 7343258 showed is not bit-for-bit
    # virtem has closed the loop: contract, image, snapshot, pause gates, and every captured frame
    # replayed bit-for-bit through the standalone function (7343396, pi_cam_virtem_frame_replay.json)
    assert rows["virtem"]["bindable"] and rows["virtem"]["validated_through_runner"]
    assert rows["virtem"]["status"] == "complete" and rows["virtem"]["missing"] == []
    # cldfrc_fice is tracked once per stage context: complete at the deep-convection
    # hook, and complete at the cloud stage's transcribed call site through its own
    # pause gate (7359461: every call answered at the pause, bit-for-bit)
    by_stage = {(k["kernel"], k["stage_action"]): k for k in record["kernels"]}
    assert len(by_stage) == len(record["kernels"])          # (kernel, stage) is the row's identity
    fice_deep = by_stage[("cldfrc_fice", "cam_run1.deep_convection")]
    fice_cloud = by_stage[("cldfrc_fice", "cam_run1.cloud_macro_microphysics")]
    assert fice_cloud["status"] == "complete" and fice_cloud["missing"] == []
    assert fice_cloud["validated_through_runner"]
    assert [g["record"] for g in fice_cloud["in_model_gates"]] == ["pi_cam_pausable_cloud-fice_50step.json"]
    # the hooked kernels: answered by the original at their hooks, bit-for-bit (7343708, 7343709)
    for row in (fice_deep, rows["fluxbelowinv"]):
        assert row["bindable"] and row["validated_through_runner"]
        assert "in_model_replacement_bfb" not in row["missing"]
    # cldfrc_fice closed the deep-convection loop: every frame captured at its hook (7343811, run tag
    # fice-capture, found through the replay record) replayed bit-for-bit through the standalone function
    assert fice_deep["status"] == "complete" and fice_deep["missing"] == []
    assert fice_deep["evidence"]["capture"] == ["pi_cam_pausable_fice-capture_50step.json"]
    # fluxbelowinv too: 36733580 frames captured at its hook (7343922) replayed bit-for-bit through the
    # standalone function with the model's snapshot of uwshcu's g (7344823); the first replay, without
    # that module state, is kept as a failure record
    assert rows["fluxbelowinv"]["status"] == "complete" and rows["fluxbelowinv"]["missing"] == []
    assert rows["fluxbelowinv"]["evidence"]["module_state"] == ["pi_cam_fluxbelowinv_module_state.json"]
    assert "7343258" in (fice_deep["note"] or "")
    # the pausable stages: dadadj has a reviewed contract and the runner pauses at it
    assert rows["dadadj"]["bindable"] and rows["dadadj"]["contract"] == "reviewed"
    pcond = rows["mmacro_pcond"]
    assert pcond["status"] == "complete" and pcond["missing"] == []
    assert pcond["contract"] == "reviewed" and pcond["bindable"] and pcond["validated_through_runner"]
    assert all(pcond["evidence"][step] for step in kernel_coverage.EVIDENCE_PATTERNS)
    assert pcond["in_model_gates"][0]["bfb"] is True and pcond["in_model_gates"][0]["path"] == "segmented"
    micro = rows["micro_mg_tend"]
    assert micro["contract"] == "reviewed" and micro["bindable"]
    assert micro["validated_through_runner"]                    # gate 7331040
    # captured at its pause (7622015, kernels-a-capture) and every frame replayed bit-for-bit through its
    # image, lane by lane (7630946) and a call at a time (7630949)
    assert micro["status"] == "complete" and micro["missing"] == []
    assert micro["evidence"]["capture"] == ["pi_cam_pausable_kernels-a-capture_50step.json"]
    assert record["summary"]["kernels_validated_through_runner"] == 22     # every exposed row; fice in both its contexts, instratus at its hook
    assert micro["in_model_gates"][0]["bfb"] is True          # the walk with the core through its image
    # the pause gates the manifest names are in-model evidence too (7331040, 7331041)
    assert [g["record"] for g in micro["in_model_gates"][1:]] == [
        "pi_cam_stage7_segmented_micro_50step.json", "pi_cam_stage7_segmented_both_50step.json",
        "pi_cam_pausable_everything_50step.json"]                  # every class installed, all kernels paused
    # the pausable classes' kernels: dadadj has closed the loop (7333952, 7333955), uwshcu lacks capture/replay
    dadadj = rows["dadadj"]
    assert dadadj["status"] == "complete" and dadadj["validated_through_runner"]
    assert all(g["bfb"] is True and g["path"].startswith("segmented") for g in dadadj["in_model_gates"])
    uwshcu = rows["compute_uwshcu_inv"]
    assert uwshcu["status"] == "complete" and uwshcu["validated_through_runner"]
    # its spec, image and snapshot are named uwshcu: the rows find records by either name
    assert uwshcu["evidence"]["standalone_build"] == ["pi_cam_uwshcu_standalone_build.json"]
    assert uwshcu["evidence"]["module_state"] == ["pi_cam_uwshcu_module_state.json"]
    for name in ("rad_rrtmg_sw", "rad_rrtmg_lw"):
        # the frame descriptor is the contract of a kernel taking derived types, and the
        # radt runner pauses at both cores; the in-model gate is the walk's until the pause gates run
        assert rows[name]["contract"] == "frame" and rows[name]["contract_path"] == "native/pi_cam/segment_frames.yaml"
        assert rows[name]["bindable"] and rows[name]["validated_through_runner"]      # gates 7334070-7334073
        assert "reviewed_contract" not in rows[name]["missing"] and "segment_runner" not in rows[name]["missing"]
        assert "in_model_replacement_bfb" not in rows[name]["missing"]
        assert [g["record"] for g in rows[name]["in_model_gates"][1:]][-1] == "pi_cam_pausable_everything_50step.json"
    # the ZM routines gather the columns that convect: one column is not replayed alone, and the
    # whole-call replays (7630951, 7630952) are the proof; the steps that replay a column are
    # not applicable, and say why
    for name in ("zm_convr", "momtran"):
        assert rows[name]["status"] == "complete" and rows[name]["missing"] == [], (name, rows[name]["missing"])
        assert set(rows[name]["not_applicable"]) == {"replay_single_column", "replay_public_api"}
        assert rows[name]["evidence"]["replay_full_chunk"] == [f"pi_cam_{name}_frame_replay_chunk.json"]
    # zm_conv_evap's columns stand alone but round by lane: its lane replay (7630947) is not
    # bit-for-bit and is no evidence, so the column steps stay open
    evap = rows["zm_conv_evap"]
    assert evap["status"] == "open" and evap["missing"] == ["replay_single_column", "replay_public_api"]
    assert evap["evidence"]["replay_full_chunk"] == ["pi_cam_zm_conv_evap_frame_replay_chunk.json"]
    assert rows["instratus_condensate"]["status"] == "complete" and rows["compute_tms"]["status"] == "complete"
    # the P3-P5 kernels await capture and replay; cldfrc_fice is complete in
    # both of its stage contexts
    assert record["summary"]["kernels_by_status"] == {"complete": 12, "open": 10}


def test_a_replay_record_that_did_not_pass_is_no_evidence(monkeypatch, tmp_path: Path) -> None:
    for name, payload in (("pi_cam_k_frame_replay.json", {"bfb": False}), ("pi_cam_k_frame_replay_chunk.json", {"bfb": True}),
                          ("pi_cam_k_full_chunk_vs_capture.json", {"passed": True})):
        (tmp_path / name).write_text(json.dumps(payload))
    monkeypatch.setattr(kernel_coverage, "VALIDATION", tmp_path)
    assert not kernel_coverage._passed("pi_cam_k_frame_replay.json")
    assert kernel_coverage._passed("pi_cam_k_frame_replay_chunk.json")
    assert kernel_coverage._passed("pi_cam_k_full_chunk_vs_capture.json")
    assert not kernel_coverage._passed("pi_cam_k_absent.json")


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
