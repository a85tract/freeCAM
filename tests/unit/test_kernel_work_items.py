"""The work-item ledger: claims, single ownership, honest closure, schema checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("kernel_work_items", REPO / "tools/kernel_work_items.py")
wi = importlib.util.module_from_spec(_spec)
sys.modules["kernel_work_items"] = wi
_spec.loader.exec_module(wi)

KERNELS = {"m::alpha", "m::beta"}
PROCESSES = {"cam_run1.a", "cam_run1.b"}


@pytest.fixture()
def ledger_path(tmp_path, monkeypatch):
    path = tmp_path / "kernel_work_items.yaml"
    monkeypatch.setattr(wi, "LEDGER_PATH", path)
    monkeypatch.setattr(wi, "_known_kernels", lambda: set(KERNELS))
    monkeypatch.setattr(wi, "_known_processes", lambda: set(PROCESSES))
    return path


def _claim(kernel="m::alpha", stage="contract"):
    return ["claim", "--kernel", kernel, "--process", "cam_run1.a",
            "--owner-class", "freecam.physics.pausable.A", "--owner", "claude",
            "--branch", "dev", "--stage", stage, "--next-gate", "50-step BFB"]


def test_claim_update_block_and_the_single_active_claim_rule(ledger_path) -> None:
    assert wi.main(_claim()) == 0
    with pytest.raises(SystemExit, match="already has an active claim"):
        wi.main(_claim())                                   # a second claim is refused
    wi.main(["update", "--kernel", "m::alpha", "--stage", "adapter", "--next-gate", "compile check"])
    wi.main(["block", "--kernel", "m::alpha", "--blocker", "derived-type boundary unsafe"])
    payload = yaml.safe_load(ledger_path.read_text())
    item = payload["items"][0]
    assert item["state"] == "blocked" and item["blocker"].startswith("derived-type")
    assert item["stage"] == "adapter"
    with pytest.raises(SystemExit, match="already has an active claim"):
        wi.main(_claim())                                   # blocked still owns the kernel
    # updating clears the block and resumes
    wi.main(["update", "--kernel", "m::alpha"])
    assert yaml.safe_load(ledger_path.read_text())["items"][0]["state"] == "in_progress"


def test_close_needs_scientific_evidence_never_a_bare_claim(ledger_path, monkeypatch, tmp_path) -> None:
    wi.main(_claim(stage="contract"))
    with pytest.raises(SystemExit, match="may not close"):
        wi.main(["close", "--kernel", "m::alpha"])          # no evidence named
    with pytest.raises(SystemExit, match="may not close"):
        wi.main(["close", "--kernel", "m::alpha", "--evidence", "missing.json"])
    evidence = tmp_path / "validation" / "alpha_contract_review.json"
    evidence.parent.mkdir()
    evidence.write_text("{}")
    monkeypatch.setattr(wi, "VALIDATION", evidence.parent)
    wi.main(["close", "--kernel", "m::alpha", "--evidence", evidence.name])
    payload = yaml.safe_load(ledger_path.read_text())
    assert payload["items"][0]["state"] == "closed" and payload["items"][0]["closed_at"]
    # a closed item releases the kernel for the next claim
    wi.main(_claim(stage="adapter"))


def test_gate_stages_close_only_through_the_decoupling_ledger(ledger_path, monkeypatch) -> None:
    wi.main(_claim(stage="in_model_replacement"))
    monkeypatch.setattr(wi, "_tracked", lambda kernel: None)
    with pytest.raises(SystemExit, match="may not close"):
        wi.main(["close", "--kernel", "m::alpha", "--evidence", "anything.json"])
    monkeypatch.setattr(wi, "_tracked", lambda kernel: {
        "validated_through_runner": True,
        "evidence": {},
        "in_model_gates": [{"present": True, "bfb": True}],
    })
    wi.main(["close", "--kernel", "m::alpha"])
    assert yaml.safe_load(ledger_path.read_text())["items"][0]["state"] == "closed"


def test_check_reports_schema_violations(ledger_path) -> None:
    ledger_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "items": [
            {"kernel": "m::alpha", "state": "in_progress", "stage": "contract",
             "target_processes": ["cam_run1.a"], "owner_class": "A", "owner": "claude",
             "branch": "dev", "started_at": "2026-09-09", "updated_at": "2026-09-09",
             "next_gate": "g"},
            {"kernel": "m::alpha", "state": "blocked", "stage": "adapter",
             "target_processes": ["cam_run1.zzz"], "owner_class": "A", "owner": "x",
             "branch": "dev", "started_at": "not-a-date", "updated_at": "2026-09-09",
             "next_gate": "g"},
            {"kernel": "m::ghost", "state": "closed", "stage": "bfb_gate"},
        ],
    }))
    problems = wi.check_ledger(wi.load_ledger(ledger_path),
                               known_kernels=KERNELS, known_processes=PROCESSES)
    text = "\n".join(problems)
    assert "2 active claims" in text
    assert "unknown target process 'cam_run1.zzz'" in text
    assert "must name its blocker" in text
    assert "not an ISO date" in text
    assert "not a canonical candidate id" in text
    assert "needs closed_at" in text


def test_the_committed_ledger_is_valid() -> None:
    problems = wi.check_ledger(wi.load_ledger())
    assert problems == []
