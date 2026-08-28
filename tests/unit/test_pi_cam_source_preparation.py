"""The patched source tree the native image is built from, and its record.

The image is compiled from a tree this repository prepares: the pinned
submodule, twelve patches, and ten modules this repository owns.  What the
prepared tree says was done to it has to be what was done to it -- a build
record that under-reports is worse than none, because it is believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import apply_pi_cam_source_patches as applier  # noqa: E402
import prepare_pi_cam_source as preparer  # noqa: E402


def test_every_applied_patch_and_support_module_exists() -> None:
    for relative in applier.PATCHES:
        assert (REPO / relative).is_file(), relative
    for relative, destination in applier.SUPPORT_SOURCES:
        assert (REPO / relative).is_file(), relative
        assert destination.startswith("src/"), destination


def test_the_report_is_what_the_applier_applies() -> None:
    report = applier.report()

    assert report["patches"] == tuple(applier.PATCHES)
    assert report["support_sources"] == tuple(
        source for source, _ in applier.SUPPORT_SOURCES
    )


def test_preparation_keeps_no_second_copy_of_the_patch_list() -> None:
    # It used to.  The copy named ten patches while twelve were applied, so
    # every prepared tree's provenance omitted the macrophysics and radiation
    # stage boundaries -- the two the current work rests on.
    text = Path(preparer.__file__).read_text()

    assert ".patch" not in text, (
        "prepare_pi_cam_source names a patch again; take the list from the "
        "applier's --report so the record cannot drift from the action"
    )


def test_the_stage_boundary_patches_are_applied() -> None:
    # 0039 and 0041 open the macrophysics and radiation stages to Python.
    # Without them the Python stages have nowhere to attach.
    applied = set(applier.PATCHES)

    assert "native/pi_cam/control_patches/0039-macro-tend-boundary.patch" in applied
    assert "native/pi_cam/control_patches/0041-rad-tend-boundary.patch" in applied


def test_the_pinned_revisions_are_declared_for_every_component() -> None:
    # A source tree prepared from a different revision is a different model.
    assert set(preparer.PINNED_REVISIONS) == {
        ".",
        "cime",
        "components/cam",
        "components/cice",
        "components/clm",
        "components/pop",
        "components/rtm",
    }
    for revision in preparer.PINNED_REVISIONS.values():
        assert len(revision) == 40 and int(revision, 16) >= 0
