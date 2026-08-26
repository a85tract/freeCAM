"""The mmacro_pcond kernel boundary: generated artefacts, record, and patch."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import generate_pi_cam_macro_split as generator  # noqa: E402
from apply_pi_cam_source_patches import PATCHES, SUPPORT_SOURCES  # noqa: E402

from freecam.pi_cam.state_codegen import load_state_bridge  # noqa: E402

SPEC = yaml.safe_load(generator.SPEC.read_text())
ARGUMENTS = SPEC["arguments"]
INPUTS = tuple(a["name"] for a in ARGUMENTS if a["role"] in ("input", "inout"))
RETURNED = tuple(
    [a["name"] for a in ARGUMENTS if a["role"] == "output"]
    + [a["name"] for a in ARGUMENTS if a["role"] == "inout"]
)

pinned = pytest.mark.skipif(
    not generator.PINNED.is_file(),
    reason="the pinned iCESM submodule is not checked out",
)


def _record_components() -> dict[str, list[str]]:
    """The generated record's components, grouped by their prefix."""

    text = generator.MODULE.read_text()
    body = text.split("type pycam_macro_record_t", 1)[1].split("end type", 1)[0]
    groups: dict[str, list[str]] = {"in_": [], "out_": [], "ref_": []}
    for match in re.finditer(r"::\s*(in_|out_|ref_)(\w+)", body):
        groups[match.group(1)].append(match.group(2))
    return groups


@pinned
def test_the_generated_artefacts_are_what_the_specification_produces() -> None:
    assert generator.MODULE.read_text() == generator.render_module()
    assert generator.PATCH.read_text() == generator.render_patch()


def test_the_record_covers_the_kernel_boundary_and_nothing_else() -> None:
    groups = _record_components()
    assert tuple(groups["in_"]) == INPUTS
    assert tuple(groups["out_"]) == RETURNED
    assert tuple(groups["ref_"]) == RETURNED
    # Workspace arguments are the routine's own scratch; a surrogate must not
    # see them, and the six pointer dummies are unassociated on this path.
    workspace = {a["name"] for a in ARGUMENTS if a["role"] == "workspace"}
    assert workspace and not workspace & set(groups["in_"] + groups["out_"])


def test_the_state_bridge_exposes_every_boundary_component(tmp_path: Path) -> None:
    cam = tmp_path / "components/cam/src"
    (cam / "physics/cam").mkdir(parents=True)
    (cam / "control").mkdir(parents=True)
    prepared = REPO / "build/iCESM1.3.1_PI_cam_only/components/cam/src"
    if not prepared.is_dir():
        pytest.skip("the prepared PI-CAM source tree is not built")
    shutil.copy2(prepared / "physics/cam/physics_types.F90", cam / "physics/cam")
    shutil.copy2(prepared / "control/camsrfexch.F90", cam / "control")
    shutil.copy2(generator.MODULE, cam / "physics/cam/pycam_macro_split.F90")

    bridge = load_state_bridge(REPO / "native/pi_cam/state_bridge.yaml", tmp_path)
    fields = {f.name: f for f in bridge.fields if f.owner.name == "macro_split"}
    assert len(fields) == 2 + len(INPUTS) + 2 * len(RETURNED)
    assert fields["macro_split.in_t0"].python_dimensions == ("pcols", "pver")
    assert fields["macro_split.out_cld"].python_dimensions == ("pcols", "pver")
    assert fields["macro_split.in_landfrac"].python_dimensions == ("pcols",)
    assert fields["macro_split.in_dt"].python_dimensions == ()
    assert fields["macro_split.kernel_mode"].dtype == "integer"


@pinned
def test_the_patch_applies_to_the_pinned_source_and_ships_in_production(tmp_path: Path) -> None:
    target = tmp_path / "src/physics/cam"
    target.mkdir(parents=True)
    shutil.copy2(generator.PINNED, target / "macrop_driver.F90")
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", str(generator.PATCH)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    # The split changes a numerical object, so it belongs in the production
    # patch set and has to earn the bit-for-bit gate -- not in the leaf add-on
    # set, which exists precisely so control code never moves numerical code.
    assert str(generator.PATCH.relative_to(REPO)) in PATCHES
    assert any(source.endswith("pycam_macro_split.F90") for source, _ in SUPPORT_SOURCES)


@pinned
def test_the_monolithic_path_still_reaches_the_original_kernel() -> None:
    """stage 0 must call mmacro_pcond with the arguments it always had."""

    patched = generator.render_patch()
    assert "call pycam_macro_before(macro_stage_local, &" in patched
    assert "if (macro_stage_local == 2) go to 1000" in patched
    # The kernel call moved into the module, and the module calls it unchanged
    # on the monolithic path.
    module = generator.MODULE.read_text()
    assert module.count("call mmacro_pcond( &") == 2
    assert "if (stage == 0) then" in module


def test_the_call_site_mapping_agrees_with_the_capture_patch() -> None:
    """The 60 argument bindings must match the reviewed capture hook.

    ``0002-capture-mmacro-pcond.patch`` records the same specification-name to
    Fortran-local mapping for a completely different purpose.  Two independent
    readings of one call site are worth more than one.
    """

    capture = (REPO / "native/pi_cam/patches/0002-capture-mmacro-pcond.patch").read_text()
    block = capture.split("'before'", 1)[1].split("pycam_capture_end", 1)[0]
    recorded = dict(re.findall(r"pycam_capture_\w+\('(\w+)', ([\w%]+)\)", block))
    assert recorded, "the capture patch's before-record could not be read"
    for name, local in recorded.items():
        assert generator.ACTUAL[name] == local, f"{name} binds to a different local"
    assert set(recorded) == set(generator.ACTUAL)
