"""The progress exporter: joins, status normalization, scoping, and publication safety."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from freecam.pi_cam.progress import (ProgressExportError, build_progress_snapshot,
                                     process_edges, resolve_tracked_kernel, summarize_evidence)

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO / "web" / "progress" / "public" / "progress.json"


# --------------------------------------------------------------------------- #
# A minimal repository the exporter can run against
# --------------------------------------------------------------------------- #

def _procedure(qualified: str, *, category="numeric_kernel", parents=(), callers=(), callees=(),
               action_root=False, host=None, public=True, inert=False):
    module, name = (qualified.split("::", 1) + [None])[:2]
    return {
        "qualified": qualified, "name": name or qualified, "module": module, "kind": "subroutine",
        "host": host, "public": public, "source": f"components/{module}.F90",
        "line_start": 1, "line_end": 9, "category": category, "in_configuration": True,
        "inert_in_configuration": inert, "parent_actions": list(parents), "action_root": action_root,
        "callers": list(callers), "callees": list(callees), "dummies": [], "result": None,
        "runtime_evidence": {"gated_kernel": {"status": "stale-cached-status-do-not-read"}},
    }


def _ledger_kernel(name, stage, *, status="open", missing=(), evidence=None, gates=(), contract="reviewed",
                   routine=None, contract_path=None):
    return {
        "kernel": name, "routine": routine, "stage_action": stage,
        "owner_class": f"freecam.physics.pausable.{name.title()}",
        "contract": contract, "contract_path": contract_path or f"native/pi_cam/functions/{name}.yaml",
        "bindable": True, "validated_through_runner": bool(gates),
        "evidence": evidence or {"capture": [], "standalone_build": [], "replay_full_chunk": [],
                                 "replay_single_column": [], "replay_public_api": [], "module_state": []},
        "in_model_gates": list(gates), "status": status, "missing": list(missing), "note": None,
    }


def _write(root: Path, rel: str, payload) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith(".yaml"):
        path.write_text(yaml.safe_dump(payload))
    else:
        path.write_text(json.dumps(payload))


def mini_repo(tmp_path: Path, *, gate_stage_counts=None, replay_record=None, ambiguous=False) -> Path:
    """Two processes sharing one kernel; kernel `alpha` gated inside process A only."""

    root = tmp_path / "repo"
    procedures = [
        _procedure("drv::a_tend", parents=["cam_run1.a"], action_root=True,
                   callees=["m::alpha", "m::shared"], category="internal_service"),
        _procedure("drv2::b_tend", parents=["cam_run1.b"], action_root=True,
                   callees=["m::alpha", "m::shared"], category="internal_service"),
        _procedure("m::alpha", parents=["cam_run1.a", "cam_run1.b"],
                   callers=["drv::a_tend", "drv2::b_tend"], callees=["m::shared"]),
        _procedure("m::shared", parents=["cam_run1.a", "cam_run1.b"],
                   callers=["drv::a_tend", "drv2::b_tend", "m::alpha"]),
        _procedure("m::loner", parents=[]),                       # unmapped candidate
        _procedure("m2::alpha", parents=["cam_run1.b"]) if ambiguous else
        _procedure("m2::other", parents=["cam_run1.b"]),
    ]
    _write(root, "validation/pi_cam_kernel_api_closure.json", {
        "schema_version": 1, "procedures": procedures,
        "actions": [{"id": "cam_run1.a", "procedures": ["drv::a_tend"]},
                    {"id": "cam_run1.b", "procedures": ["drv2::b_tend"]}],
    })
    gates = [{"path": "segmented", "record": "gate_a.json", "bfb_record": "gate_a_bfb.json",
              "present": True, "bfb": True, "compared_files": 4}]
    evidence = {"capture": [], "standalone_build": ["alpha_build.json"],
                "replay_full_chunk": ["alpha_replay.json"] if replay_record else [],
                "replay_single_column": [], "replay_public_api": [], "module_state": []}
    _write(root, "validation/physics_kernel_decoupling.json", {
        "schema_version": 1, "cam_source_revision": "rev", "configuration": {"case": "PI-atm"},
        "summary": {}, "closure": {}, "unresolved": [],
        "actions": [
            {"id": "cam_run1.a", "native_id": 1, "operation": "a_tend", "phase": "cam_run1",
             "kind": "scheme", "granularity": "stage", "parent_stage": None, "enabled": True,
             "classification": "numeric_scheme", "activity": "active", "activity_basis": "enabled in the plan",
             "python_class": "freecam.physics.pausable.A", "kernels": ["alpha"],
             "kernel_candidates": [{"qualified_name": "m::shared", "adapter_status": "context_required",
                                    "blockers": ["derived_or_unknown_type"]}],
             "coverage": "partial", "alternate_of": [], "note": None},
            {"id": "cam_run1.b", "native_id": 2, "operation": "b_tend", "phase": "cam_run1",
             "kind": "scheme", "granularity": "stage", "parent_stage": None, "enabled": False,
             "classification": "numeric_scheme", "activity": "alternate", "activity_basis": "disabled",
             "python_class": None, "kernels": [], "kernel_candidates": [],
             "coverage": "alternate", "alternate_of": ["cam_run1.a"], "note": None},
        ],
        "kernels": [_ledger_kernel("alpha", "cam_run1.a", gates=gates, evidence=evidence,
                                   status="open", missing=["capture"])],
    })
    _write(root, "validation/pi_cam_kernel_api_redirectable_calls.json", {
        "schema_version": 1,
        "procedures": [
            {"qualified": "m::alpha", "classification": "rename-references", "redirectable": True,
             "host": None, "kind": "subroutine", "public": True, "source": "components/m.F90",
             "source_callers": 2, "callers_in_same_source": 0, "symbols": []},
            {"qualified": "m::shared", "classification": "no-call-relocation", "redirectable": False,
             "host": None, "kind": "subroutine", "public": True, "source": "components/m.F90",
             "source_callers": 3, "callers_in_same_source": 3, "symbols": []},
        ],
    })
    _write(root, "native/pi_cam/segment_runners.yaml", {
        "schema_version": 1,
        "runners": [{"stage": "cam_run1.a", "prefix": "pycam_a", "module": "native/a.F90",
                     "generator": "g", "descriptors": "d",
                     "kernels": [{"name": "alpha", "owner": "freecam.physics.pausable.A",
                                  "contract": "native/pi_cam/functions/alpha.yaml",
                                  "validated_by": ["gate_a.json", "gate_a_bfb.json"]}]}],
    })
    _write(root, "native/pi_cam/hooks.yaml", {"schema_version": 1, "hooks": []})
    _write(root, "web/public/catalog.json", {
        "schema_version": 1,
        "entries": [
            {"id": "cam_run1.a", "kind": "scheme", "display_name": "A", "description": "Process A.",
             "default_index": 0, "in_default": True, "name": "a_tend"},
            {"id": "cam_run1.b", "kind": "scheme", "display_name": "B", "description": "Process B.",
             "default_index": 1, "in_default": True, "name": "b_tend"},
            {"id": "catalog:x", "kind": "runtime_catalog_process", "name": "x", "display_name": "x",
             "qualified_name": "mx::x", "description": "mx::x", "present": False, "addable": False,
             "reason": "not independently runnable (context_required): derived_or_unknown_type"},
        ],
    })
    _write(root, "native/pi_cam/functions/alpha.yaml", {"qualified_name": "m::alpha"})
    counts = gate_stage_counts if gate_stage_counts is not None else {"a_python": {"alpha": 100}}
    _write(root, "validation/gate_a.json", {
        "schema_version": 1, "run_status": "passed", "steps": 50, "mpi_ranks": 4,
        "pbs_job_id": "1234.server", "timing": {"advance_seconds": 1.5},
        "python_stages": ["a_tend"], "native_manifest": "build/pi_cam_promoted_x9/native_cam_manifest.json",
        "native_library_sha256": "c0ffee", "stage_execution": {
            key: {"execution_mode": "segmented", "python_model_calls_by_kernel": by}
            for key, by in counts.items()},
    })
    _write(root, "validation/gate_a_bfb.json", {"bfb": True, "compared_files": 4, "reference_files": 4,
                                                "candidate_files": 4, "first_difference": None,
                                                "missing_in_candidate": [], "extra_in_candidate": []})
    _write(root, "validation/alpha_build.json", {"schema_version": 1, "library_sha256": "beef",
                                                 "spec_sha256": "s", "members": ["m.o"], "wrapper": "w.f90",
                                                 "original_call_proof": "call m_mp_alpha_"})
    if replay_record:
        _write(root, "validation/alpha_replay.json", replay_record)
    return root


def _kernel(snapshot, kid):
    return snapshot["kernels"][kid]


def test_the_export_is_deterministic_and_detects_staleness(tmp_path: Path) -> None:
    root = mini_repo(tmp_path)
    first = build_progress_snapshot(root)
    second = build_progress_snapshot(root)
    assert first["content_hash"] == second["content_hash"]
    # volatile metadata stays outside the hash
    assert "volatile" in first and "generated_at" in first["volatile"]
    hashed = {k: v for k, v in first.items() if k not in ("content_hash", "volatile")}
    import hashlib
    assert first["content_hash"] == hashlib.sha256(
        json.dumps(hashed, sort_keys=True).encode()).hexdigest()
    # a source change changes the hash: that is what --check detects
    ledger = json.loads((root / "validation/physics_kernel_decoupling.json").read_text())
    ledger["kernels"][0]["status"] = "complete"
    _write(root, "validation/physics_kernel_decoupling.json", ledger)
    assert build_progress_snapshot(root)["content_hash"] != first["content_hash"]


def test_shared_kernels_are_counted_once_and_appear_in_both_processes(tmp_path: Path) -> None:
    snapshot = build_progress_snapshot(mini_repo(tmp_path))
    assert snapshot["totals"]["candidate_kernels"] == 4          # alpha, shared, loner, other -- shared once
    shared = _kernel(snapshot, "m::shared")
    assert shared["processes"] == ["cam_run1.a", "cam_run1.b"]
    for pid in ("cam_run1.a", "cam_run1.b"):
        assert "m::shared" in snapshot["process_membership"][pid]["kernels"]
    # the disabled alternate stays discoverable, not a failure
    b = next(p for p in snapshot["processes"] if p["id"] == "cam_run1.b")
    assert not b["enabled"] and b["activity"] == "alternate" and b["alternate_of"] == ["cam_run1.a"]
    # the class's core kernels are a separate, smaller list than the recursive candidates
    a = next(p for p in snapshot["processes"] if p["id"] == "cam_run1.a")
    assert a["core_kernels"] == [{"routine": "alpha", "id": "m::alpha",
                                  "owner_class": "freecam.physics.pausable.Alpha", "status": "open"}]
    assert b["core_kernels"] == []
    assert len(snapshot["process_membership"]["cam_run1.a"]["kernels"]) > len(a["core_kernels"])


def test_replacement_evidence_is_scoped_to_the_tested_process(tmp_path: Path) -> None:
    """A gate under process A never marks the same kernel's other callers verified."""

    snapshot = build_progress_snapshot(mini_repo(tmp_path))
    rows = {r["process"]: r for r in snapshot["replacements"] if r["kernel_routine"] == "alpha"}
    assert rows["cam_run1.a"]["state"] == "verified"
    assert rows["cam_run1.a"]["gates"][0]["replacement_calls"] == 100
    assert "cam_run1.b" not in rows                              # no runner reaches alpha inside B
    alpha = _kernel(snapshot, "m::alpha")
    assert alpha["capabilities"]["original_replacement_bfb"]["contexts"] == ["cam_run1.a"]
    assert snapshot["notes"]["replacement_scope"].startswith("A replacement gate is scoped")


def test_a_gate_that_ran_the_kernel_in_another_process_is_historical_not_verified(tmp_path: Path) -> None:
    root = mini_repo(tmp_path, gate_stage_counts={"b_python": {"alpha": 100}})
    snapshot = build_progress_snapshot(root)
    row = next(r for r in snapshot["replacements"] if r["kernel_routine"] == "alpha")
    assert row["state"] == "historical-evidence"
    assert row["gates"][0]["replacement_calls"] is None and "limitation" in row["gates"][0]


def test_build_only_versus_callable_versus_replay_verified(tmp_path: Path) -> None:
    build_only = build_progress_snapshot(mini_repo(tmp_path))
    caps = _kernel(build_only, "m::alpha")["capabilities"]
    assert caps["adapter_build"]["state"] == "available"
    assert caps["independently_callable"]["state"] == "not-verified"      # a build is not an execution
    assert caps["standalone_replay"]["state"] == "not-assessed"

    replayed = build_progress_snapshot(mini_repo(tmp_path, replay_record={
        "schema_version": 1, "kernel": "alpha", "function": "m::alpha", "layout": "column",
        "binding": "module", "calls": 10, "samples": 10, "compared_values": 100,
        "statuses": {"ok": 10}, "bfb": True, "image_sha256": "beef", "mismatches": []}))
    caps = _kernel(replayed, "m::alpha")["capabilities"]
    assert caps["independently_callable"]["state"] == "verified"
    assert caps["standalone_replay"]["state"] == "verified"

    mismatched = build_progress_snapshot(mini_repo(tmp_path, replay_record={
        "schema_version": 1, "kernel": "alpha", "function": "m::alpha", "layout": "column",
        "binding": "module", "calls": 10, "samples": 10, "compared_values": 100,
        "statuses": {"ok": 10}, "bfb": False, "image_sha256": "beef", "mismatches": [{"call": 1}]}))
    caps = _kernel(mismatched, "m::alpha")["capabilities"]
    assert caps["independently_callable"]["state"] == "verified"          # it ran; it did not match
    assert caps["standalone_replay"]["state"] == "not-verified"


def test_missing_evidence_never_becomes_success(tmp_path: Path) -> None:
    root = mini_repo(tmp_path)
    (root / "validation/alpha_build.json").unlink()               # the named record is absent
    snapshot = build_progress_snapshot(root)
    caps = _kernel(snapshot, "m::alpha")["capabilities"]
    assert caps["adapter_build"]["state"] == "not-implemented"
    assert caps["independently_callable"]["state"] == "not-assessed"


def test_untracked_candidates_are_unassessed_or_need_binding_and_unmapped_stay_visible(tmp_path: Path) -> None:
    snapshot = build_progress_snapshot(mini_repo(tmp_path))
    shared = _kernel(snapshot, "m::shared")["capabilities"]
    assert shared["independently_callable"]["state"] == "needs-binding"
    assert "derived_or_unknown_type" in shared["independently_callable"]["blockers"]
    assert shared["standalone_replay"]["state"] == "not-assessed"
    assert _kernel(snapshot, "m::shared")["redirect"]["blocker"].startswith("inlined")
    assert snapshot["unmapped_kernels"] == ["m::loner"]
    assert snapshot["totals"]["unmapped_kernels"] == 1


def test_the_stale_status_cached_in_the_call_tree_is_never_read(tmp_path: Path) -> None:
    root = mini_repo(tmp_path)
    ledger = json.loads((root / "validation/physics_kernel_decoupling.json").read_text())
    ledger["kernels"][0]["status"] = "complete"
    ledger["kernels"][0]["missing"] = []
    _write(root, "validation/physics_kernel_decoupling.json", ledger)
    snapshot = build_progress_snapshot(root)
    # the closure fixture's cached status says otherwise; the ledger wins
    assert _kernel(snapshot, "m::alpha")["status"] == "complete"
    assert "stale-cached-status-do-not-read" not in json.dumps(snapshot)


def test_ambiguous_name_joins_are_rejected_not_guessed(tmp_path: Path) -> None:
    root = mini_repo(tmp_path, ambiguous=True)
    ledger = json.loads((root / "validation/physics_kernel_decoupling.json").read_text())
    ledger["kernels"][0]["stage_action"] = "cam_run1.b"           # both m::alpha and m2::alpha live in B
    ledger["kernels"][0]["contract_path"] = "native/pi_cam/functions/missing.yaml"
    _write(root, "validation/physics_kernel_decoupling.json", ledger)
    with pytest.raises(ProgressExportError, match="ambiguously"):
        build_progress_snapshot(root)


def test_unsupported_schemas_are_rejected(tmp_path: Path) -> None:
    root = mini_repo(tmp_path)
    ledger = json.loads((root / "validation/physics_kernel_decoupling.json").read_text())
    ledger["schema_version"] = 99
    _write(root, "validation/physics_kernel_decoupling.json", ledger)
    with pytest.raises(ProgressExportError, match="schema_version"):
        build_progress_snapshot(root)


def test_nothing_published_names_a_path_a_server_or_this_user(tmp_path: Path) -> None:
    snapshot = build_progress_snapshot(mini_repo(tmp_path))
    text = json.dumps(snapshot)
    for fragment in ("/glade", "desched", "scratch", "/home/"):
        assert fragment not in text
    assert "1234.server" not in text and '"pbs_job": "1234"' in text
    image = snapshot["evidence"]["gate_a.json"]["image"]
    assert image == {"build": "pi_cam_promoted_x9", "role": "development", "sha256": "c0ffee"}


def test_process_edges_nest_by_nearest_numeric_ancestor_and_survive_recursion() -> None:
    procedures = {
        "d::root": _procedure("d::root", parents=["p"], action_root=True, category="internal_service",
                              callees=["s::mid"]),
        "s::mid": _procedure("s::mid", parents=["p"], category="internal_service",
                             callers=["d::root"], callees=["k::inner"]),
        "k::outer": _procedure("k::outer", parents=["p"], callers=["d::root"], callees=["k::inner"]),
        "k::inner": _procedure("k::inner", parents=["p"], callers=["s::mid", "k::outer", "k::inner"],
                               callees=["k::inner"]),  # self-recursive
    }
    candidates = {q: p for q, p in procedures.items() if p["category"] == "numeric_kernel"}
    edges = process_edges("p", sorted(candidates), procedures, candidates)
    by_kernel: dict[str, list] = {}
    for edge in edges:
        by_kernel.setdefault(edge["kernel"], []).append(edge)
    # inner nests under outer directly, and under the root through the named intermediary
    parents = {(e["parent"], tuple(e["via"])) for e in by_kernel["k::inner"]}
    assert ("k::outer", ()) in parents
    # the second path reaches the action root without crossing a kernel: a top-level
    # entry whose via names the non-numerical intermediary and ends at the root
    assert (None, ("s::mid", "d::root")) in parents
    assert by_kernel["k::outer"][0]["parent"] is None
    assert all(e["kernel"] != e["parent"] for e in edges)         # recursion did not loop


def _coverage_record(label: str) -> dict:
    return {
        "schema_version": 1,
        "run": {"label": label, "bfb": True, "run_status": "passed", "steps": 50, "pbs_job": "77",
                "final_date": "0001-01-03", "mpi_ranks": 4, "summary_record": "s.json", "bfb_record": "b.json"},
        "image": {"native_library_sha256": "cafe", "kernel_counts": {"scope": "all"}},
        "slots": {"0": "initialization"},
        "kernels": [
            {"qualified": "m::alpha", "index": 0, "entry": "trampoline", "coverage": "full",
             "blind_spots": [], "processes": ["cam_run1.a", "cam_run1.b"], "status": "observed",
             "calls_total": 100, "calls_by_context": {"cam_run1.a": 98, "initialization": 2},
             "first_step": 1, "last_step": 50, "ranks_with_calls": 4,
             "rank_calls_min": 25, "rank_calls_max": 25},
            {"qualified": "m::shared", "index": 1, "entry": "trampoline", "coverage": "full",
             "blind_spots": [], "processes": ["cam_run1.a", "cam_run1.b"],
             "status": "not-observed-in-this-run", "calls_total": 0, "calls_by_context": {},
             "first_step": -1, "last_step": -1, "ranks_with_calls": 0,
             "rank_calls_min": 0, "rank_calls_max": 0},
        ],
        "process_calls": {"cam_run1.a": {"m::alpha": {"calls": 98, "first_step": 1, "last_step": 50}}},
        "summary": {"candidates": 4, "instrumented": 2, "observed": 1,
                    "not-observed-in-this-run": 1, "unknown": 2, "all_ranks_reported": True},
    }


def test_observation_runs_join_per_process_and_absence_stays_honest(tmp_path: Path) -> None:
    root = mini_repo(tmp_path)
    bare = build_progress_snapshot(root)
    assert bare["observation_runs"] == [] and bare["totals"]["observed_by_run"] == {}
    assert bare["kernels"]["m::alpha"]["observation"] == {}

    _write(root, "validation/pi_cam_kernel_runtime_coverage_50step.json", _coverage_record("50step"))
    snapshot = build_progress_snapshot(root)
    runs = snapshot["observation_runs"]
    assert [r["key"] for r in runs] == ["50step"] and runs[0]["validated"]
    alpha = snapshot["kernels"]["m::alpha"]["observation"]["50step"]
    assert alpha["status"] == "observed" and alpha["calls_by_context"]["cam_run1.a"] == 98
    shared = snapshot["kernels"]["m::shared"]["observation"]["50step"]
    assert shared["status"] == "not-observed-in-this-run"
    # observation is attributed per process: alpha was observed in A only, and the
    # per-process map carries exactly that -- nothing global marks it observed in B
    assert snapshot["process_observation"]["50step"]["cam_run1.a"]["m::alpha"]["calls"] == 98
    assert "cam_run1.b" not in snapshot["process_observation"]["50step"]
    assert snapshot["totals"]["observed_by_run"] == {"50step": 1}
    # a failed observation run is listed but never validated
    failed = _coverage_record("50step")
    failed["run"]["bfb"] = False
    _write(root, "validation/pi_cam_kernel_runtime_coverage_50step.json", failed)
    unvalidated = build_progress_snapshot(root)
    assert unvalidated["observation_runs"][0]["validated"] is False


# --------------------------------------------------------------------------- #
# The committed snapshot is a regression fixture of the real records
# --------------------------------------------------------------------------- #

def test_the_committed_snapshot_matches_the_checkout() -> None:
    committed = json.loads(SNAPSHOT.read_text())
    fresh = build_progress_snapshot(REPO)
    assert committed["content_hash"] == fresh["content_hash"], \
        "web/progress/public/progress.json is stale; run tools/export_progress_snapshot.py"


def test_the_inspected_records_produce_the_known_counts() -> None:
    snapshot = json.loads(SNAPSHOT.read_text())
    ledger = json.loads((REPO / "validation/physics_kernel_decoupling.json").read_text())
    totals = snapshot["totals"]
    # derived from data, and pinned as the regression fixture of this checkout
    assert totals["candidate_kernels"] == 601
    assert totals["tracked_kernels"] == ledger["summary"]["kernels"] == 20
    assert totals["tracked_by_status"] == ledger["summary"]["kernels_by_status"] == {"complete": 5, "open": 15}
    assert totals["unmapped_kernels"] == len(snapshot["unmapped_kernels"]) == 6
    assert totals["processes"] == ledger["summary"]["actions"] == 58
    # five complete tracked records do not mean five kernels replaceable in every caller
    fice = snapshot["kernels"]["cloud_fraction::cldfrc_fice"]
    assert len(fice["processes"]) > 1
    verified = fice["capabilities"]["original_replacement_bfb"]["contexts"]
    assert verified == ["cam_run1.deep_convection"]


def test_core_kernels_and_recursive_candidates_are_two_levels_in_the_real_records() -> None:
    """CloudMacroMicrophysics exposes two core kernels; its candidate tree holds ~a hundred."""

    snapshot = json.loads(SNAPSHOT.read_text())
    stage7 = next(p for p in snapshot["processes"] if p["id"] == "cam_run1.cloud_macro_microphysics")
    core = {c["routine"]: c for c in stage7["core_kernels"]}
    assert sorted(core) == ["micro_mg_tend", "mmacro_pcond"]
    # the core kernels are owned by the composed sub-classes, not the stage class itself
    assert core["mmacro_pcond"]["owner_class"] == "freecam.physics.macrophysics.Macrophysics"
    assert core["micro_mg_tend"]["owner_class"] == "freecam.physics.microphysics.Microphysics"
    candidates = snapshot["process_membership"]["cam_run1.cloud_macro_microphysics"]["kernels"]
    assert len(candidates) > 50 and {c["id"] for c in core.values()} <= set(candidates)
    assert "Exposing a process does not expose every candidate" in snapshot["notes"]["core_vs_candidates"]


def test_historical_failures_precede_the_scoped_success_in_the_real_records() -> None:
    snapshot = json.loads(SNAPSHOT.read_text())
    fice = snapshot["kernels"]["cloud_fraction::cldfrc_fice"]
    assert fice["capabilities"]["original_replacement_bfb"]["state"] == "verified"
    assert any("failure" in record for record in fice["failures"])
    flux = snapshot["kernels"]["uwshcu::fluxbelowinv"]
    assert flux["capabilities"]["standalone_replay"]["state"] == "verified"
    assert "pi_cam_fluxbelowinv_frame_replay_no_module_state_failure.json" in flux["failures"]


def test_the_real_snapshot_is_publishable() -> None:
    text = SNAPSHOT.read_text()
    for fragment in ("/glade", "desched", "/home/", "scratch"):
        assert fragment not in text.lower()
    # the account name must not appear as a path segment; a bare-word search would
    # trip over legitimate vocabulary (on GitHub runners USER is "runner", which
    # the records use for the segment runners)
    user = os.environ.get("USER")
    if user and len(user) > 3:
        assert f"/{user}" not in text and f"{user}@" not in text
