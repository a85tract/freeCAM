"""The CESM source a PI-atm case is built from: the pinned submodule and the recipe's patches.

The oracle, python-state and pyCESM cases were made by hand from checkouts that
carried local commits and edits on no remote.  The recipe is those, kept here,
and a comparison against each checkout as it was when its case was built is the
proof the recipe rebuilds it.  These tests hold the recipe, the patching and the
comparison to that.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import prepare_cesm_source as cesm  # noqa: E402

SUBMODULE = REPO / "external/iCESM1.3.1_fzhu"
PATCH = """diff --git a/f.txt b/f.txt
--- a/f.txt
+++ b/f.txt
@@ -1,2 +1,2 @@
 a
-b
+c
"""


def _git(*arguments: str, cwd: Path) -> None:
    subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True)


def _recipe(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump({"schema_version": 1, "base": "unused", "patches": [],
                                    "variants": {"cases": {"patches": []}}, **extra}))
    return path


def _old(path: Path) -> None:
    os.utime(path, (1_000_000_000, 1_000_000_000), follow_symlinks=False)


def test_every_patch_the_recipe_names_exists_and_says_why() -> None:
    recipe = cesm.load_recipe()

    for variant in recipe["variants"]:
        for patch in cesm._patches(recipe, variant):
            assert (cesm.RECIPE.parent / patch["file"]).is_file(), patch["file"]
            assert patch["component"] in cesm.COMPONENTS, patch["component"]
            assert patch["why"].strip(), patch["file"]
    for why in cesm.known_differences(recipe, "state").values():
        assert why.strip()


def test_the_state_variant_is_the_cases_source_with_pic_objects() -> None:
    recipe = cesm.load_recipe()
    cases = [patch["file"] for patch in cesm._patches(recipe, "cases")]
    state = [patch["file"] for patch in cesm._patches(recipe, "state")]

    assert state[:len(cases)] == cases
    assert state[len(cases):] == ["patches/cime-pic-objects.patch"]
    # a variant's known differences add to the recipe's, never replace them
    assert set(cesm.known_differences(recipe, "cases")) < set(cesm.known_differences(recipe, "state"))


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(SystemExit, match="unknown variant"):
        cesm._patches(cesm.load_recipe(), "pcols32")


def test_the_patches_apply_to_the_pinned_submodule() -> None:
    recipe = cesm.load_recipe()
    if not (SUBMODULE / "cime").is_dir():
        pytest.skip("the iCESM submodule is not checked out")
    for patch in cesm._patches(recipe, "state"):
        result = subprocess.run(["git", "apply", "--check", str(cesm.RECIPE.parent / patch["file"])],
                                cwd=SUBMODULE / patch["component"], capture_output=True, text=True)
        assert result.returncode == 0, (patch["file"], result.stderr)


def test_a_patch_lands_inside_another_git_work_tree(tmp_path: Path) -> None:
    # Plain `git apply` in a subdirectory of a work tree takes the patch's paths
    # from that tree's top and skips them with status 0: a prepared tree under
    # build/ would have silently gone unpatched.
    _git("init", "-q", cwd=tmp_path)
    component = tmp_path / "prepared/cime"
    component.mkdir(parents=True)
    (component / "f.txt").write_text("a\nb\n")
    patch = tmp_path / "p.patch"
    patch.write_text(PATCH)

    cesm.apply_patch(patch, component)

    assert (component / "f.txt").read_text() == "a\nc\n"


def test_a_patch_that_does_not_apply_is_refused(tmp_path: Path) -> None:
    component = tmp_path / "cime"
    component.mkdir()
    (component / "f.txt").write_text("a\nx\n")
    patch = tmp_path / "p.patch"
    patch.write_text(PATCH)

    with pytest.raises(SystemExit, match="does not apply"):
        cesm.apply_patch(patch, component)


def test_prepare_copies_the_base_patches_it_and_records_what_it_did(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base"
    (base / "cime/.git").mkdir(parents=True)
    (base / "cime/f.txt").write_text("a\nb\n")
    (base / "cime/__pycache__").mkdir()
    (tmp_path / "patches").mkdir()
    (tmp_path / "patches/p.patch").write_text(PATCH)
    recipe = _recipe(tmp_path, base="base", variants={
        "cases": {"patches": []},
        "state": {"patches": [{"component": "cime", "file": "patches/p.patch", "why": "test"}]}})
    monkeypatch.setattr(cesm, "REPO", tmp_path)
    monkeypatch.setattr(cesm, "ensure_pinned_externals", lambda source: {"cime": "abc"})

    output = cesm.prepare(tmp_path / "out", variant="state", recipe_path=recipe)

    assert (output / "cime/f.txt").read_text() == "a\nc\n"
    assert not (output / "cime/.git").exists() and not (output / "cime/__pycache__").exists()
    record = json.loads((output / ".cesm-source.json").read_text())
    assert record["variant"] == "state" and record["revisions"] == {"cime": "abc"}
    assert record["patches"][0]["file"] == "patches/p.patch" and len(record["patches"][0]["sha256"]) == 64
    with pytest.raises(SystemExit, match="exists"):
        cesm.prepare(output, variant="state", recipe_path=recipe)


def _trees(tmp_path: Path) -> tuple[Path, Path]:
    checkout, prepared = tmp_path / "checkout", tmp_path / "prepared"
    for root in (checkout, prepared):
        (root / "cime").mkdir(parents=True)
        (root / "cime/same.txt").write_text("same\n")
        (root / "cime/known.txt").write_text(f"{root.name}\n")
        os.symlink("same.txt", root / "cime/link")
    (checkout / "cime/__pycache__").mkdir()
    (checkout / "cime/__pycache__/x.pyc").write_bytes(b"\0")
    for path in checkout.rglob("*"):
        _old(path)
    return checkout, prepared


def test_trees_with_the_same_bytes_compare_identical(tmp_path: Path) -> None:
    checkout, prepared = _trees(tmp_path)
    recipe = _recipe(tmp_path, known_differences=[{"path": "cime/known.txt", "why": "test"}])

    result = cesm.compare(prepared, checkout, datetime(2020, 1, 1), recipe_path=recipe)

    assert result["identical"], result["problems"]
    assert (result["same"], result["known_differences"], result["taken_from_git"]) == (2, 1, 0)
    assert result["compared_with"] == "checkout"


def test_a_difference_the_recipe_does_not_explain_is_a_problem(tmp_path: Path) -> None:
    checkout, prepared = _trees(tmp_path)
    (prepared / "cime/extra.txt").write_text("\n")
    os.remove(prepared / "cime/link")
    os.symlink("known.txt", prepared / "cime/link")

    result = cesm.compare(prepared, checkout, datetime(2020, 1, 1), recipe_path=_recipe(tmp_path))

    assert not result["identical"]
    assert set(result["problems"]) == {"differs: cime/known.txt", "link differs: cime/link",
                                       "only in the prepared tree: cime/extra.txt"}


def test_a_file_edited_after_the_build_is_read_from_git(tmp_path: Path) -> None:
    checkout, prepared = _trees(tmp_path)
    recipe = _recipe(tmp_path, known_differences=[{"path": "cime/known.txt", "why": "test"}])
    _git("init", "-q", cwd=checkout / "cime")
    _git("add", "same.txt", cwd=checkout / "cime")
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=checkout / "cime")
    (checkout / "cime/same.txt").write_text("edited after the build\n")
    (checkout / "cime/new.txt").write_text("written after the build\n")

    result = cesm.compare(prepared, checkout, datetime(2020, 1, 1), recipe_path=recipe)

    assert result["taken_from_git_files"] == ["cime/same.txt"]
    assert result["problems"] == ["changed after 2020-01-01T00:00:00 and not in git: cime/new.txt"]


@pytest.mark.parametrize("record", sorted((REPO / "validation").glob("pi_cam_cesm_source_*_case.json")),
                         ids=lambda path: path.stem)
def test_the_recorded_reconstructions_are_of_the_patches_in_the_recipe(record: Path) -> None:
    # An edited patch is a different source: its reconstruction has to be shown again.
    payload = json.loads(record.read_text())
    recipe = cesm.load_recipe()
    expected = [{"component": patch["component"], "file": patch["file"],
                 "sha256": cesm._sha256((cesm.RECIPE.parent / patch["file"]).read_bytes())}
                for patch in cesm._patches(recipe, payload["variant"])]

    assert payload["identical"] and not payload["problems"]
    assert payload["source"]["patches"] == expected
    assert payload["known"] == cesm.known_differences(recipe, payload["variant"])
