"""The carve manifest RecastEngine reads, and the run configuration written for it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import apply_pi_cam_source_patches as patches  # noqa: E402
import export_recast_carve  # noqa: E402
import prepare_pi_cam_source as prepare  # noqa: E402
import recast_run_config  # noqa: E402


def test_the_committed_manifest_is_current() -> None:
    assert export_recast_carve.main(["--check"]) == 0


def test_the_manifest_is_the_patch_and_support_lists_in_their_order() -> None:
    manifest = json.loads((REPO / "recast" / "carve.json").read_text())
    assert manifest["schema"] == "recast.carve.v1"
    assert manifest["patches"] == list(patches.PATCHES)
    assert [(f["source"], f["target"]) for f in manifest["files"]] == list(patches.SUPPORT_SOURCES)
    assert manifest["upstream"]["revision"] == prepare.PINNED_REVISIONS["."]
    for relative in (*manifest["patches"], *(f["source"] for f in manifest["files"])):
        assert (REPO / relative).is_file(), relative


def test_every_declared_statement_says_why_and_names_no_site() -> None:
    manifest = json.loads((REPO / "recast" / "carve.json").read_text())
    declared = manifest["numerics"]["declared"]
    assert declared and all(d["reason"].strip() for d in declared)
    targets = {f["target"] for f in manifest["files"]} | {
        line[6:].strip()
        for relative in manifest["patches"]
        for line in (REPO / relative).read_text().splitlines()
        if line.startswith("+++ b/")
    }
    assert {d["file"] for d in declared} <= targets
    text = json.dumps(manifest)
    assert "/glade" not in text and "/home/" not in text


def test_the_run_configuration_builds_the_candidates_image_then_runs_the_gate(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(recast_run_config, "site", lambda repo: {
        "FREECAM_SCRATCH": str(tmp_path / "scratch"),
        "FREECAM_ACCOUNT": "ACCT",
        "FREECAM_REFERENCE_RUN": str(tmp_path / "oracle/run"),
    })
    config = recast_run_config.build(REPO, tmp_path / "ws", "recast/carve.json")
    assert config["executor"] == "pbs" and config["stages"]["pbs"]["account"] == "ACCT"
    assert config["stages"]["pinned-run"]["output"] == str(tmp_path / "oracle/run")
    build, run = config["stages"]["fullmodel.bitwise"]["job"]
    assert build["env"]["FREECAM_IMAGE_ROOT"] == "{workspace}/image"
    assert run["env"]["PYCAM_NATIVE_MANIFEST"] == "{workspace}/image/native_cam_manifest.json"
    assert run["env"]["FREECAM_REFERENCE_RUN"] == "{reference}"
    for job in (build, run):
        assert (REPO / job["argv"][0]).is_file()
