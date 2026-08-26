"""The carved macrophysics kernels are the driver's own arithmetic, moved."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import generate_pi_cam_macro_kernels as carve  # noqa: E402
from apply_pi_cam_source_patches import PATCHES  # noqa: E402

pinned = pytest.mark.skipif(
    not carve.PINNED.is_file(),
    reason="the pinned iCESM submodule is not checked out",
)


@pinned
def test_the_generated_artefacts_are_what_the_pinned_source_produces() -> None:
    assert carve.MODULE.read_text() == carve.render_module()
    assert carve.PATCH.read_text() == carve.render_patch()


@pinned
def test_every_carved_body_is_the_original_text_renamed_and_nothing_else() -> None:
    """The whole claim of this patch, checked line for line.

    If a body were retyped rather than lifted, an expression could differ by a
    parenthesis or an operand order and still compile -- and the bit-for-bit
    gate would fail with nothing to point at.  So compare the text.
    """

    source = carve.PINNED.read_text().splitlines()
    module = carve.MODULE.read_text()
    for block in carve.BLOCKS:
        expected = "\n".join(block.body(source))
        assert expected.strip(), block.name
        assert expected in module, f"{block.name} body is not the pinned text"
        # and every rename actually removed the host-associated name
        for original in block.renames:
            if original.endswith(")") or " " in original:
                continue
            assert original not in expected, f"{block.name} still refers to {original}"


@pinned
def test_the_driver_no_longer_computes_what_the_kernels_compute() -> None:
    """Carved, not copied: the arithmetic must exist in exactly one place."""

    patched = carve.render_patch()
    removed = [line[1:] for line in patched.splitlines() if line.startswith("-")]
    added = "\n".join(line[1:] for line in patched.splitlines() if line.startswith("+"))
    signatures = (
        "dpdlfliq(i,k) = ( dlf(i,k)",
        "clrw_old(i,k) = max(",
        "lmitend(:ncol,top_lev:pver) =",
        "ptend_loc%q(i,k,ixnumice) = niten(i,k)",
        "nqctn(i,k) = qcten(i,k)",
        "mr_lsliq(i,k) = state_loc%q(i,k,ixcldliq)",
        "cldsice(:ncol,k) = lcwat(:ncol,k)",
    )
    joined = "\n".join(removed)
    for signature in signatures:
        assert signature in joined, f"the patch never removed {signature!r}"
        assert signature not in added, f"the patch re-adds {signature!r}"


@pinned
def test_each_kernel_is_called_once_with_the_caller_s_own_names() -> None:
    added = "\n".join(
        line[1:] for line in carve.render_patch().splitlines() if line.startswith("+")
    )
    for block in carve.BLOCKS:
        assert added.count(f"call {block.name}(") == 1, block.name
    # The bodies speak of `t` and `ptend_s`; the call site must still speak of
    # the derived types the driver actually holds.
    assert "state_loc%t" in added and "ptend_loc%q" in added
    assert "wtrc_iatype(:,iwtliq)" in added
    assert "get_nstep()" in added


@pinned
def test_the_refusals_became_a_status_the_caller_still_stops_on() -> None:
    module = carve.MODULE.read_text()
    body = module.split("subroutine macrop_kernel_to_ptend", 1)[1]
    body = body.split("end subroutine", 1)[0]
    assert "call endrun" not in body, "a carved kernel must not abort the model itself"
    assert body.count("status = 1") + body.count("status = 2") == 4
    added = "\n".join(
        line[1:] for line in carve.render_patch().splitlines() if line.startswith("+")
    )
    assert "if (macro_status /= 0) call endrun" in added


@pinned
def test_the_patch_applies_to_the_pinned_source_and_ships_in_production(tmp_path: Path) -> None:
    target = tmp_path / "src/physics/cam"
    target.mkdir(parents=True)
    shutil.copy2(carve.PINNED, target / "macrop_driver.F90")
    subprocess.run(
        ["git", "apply", "--unidiff-zero", "--check", str(carve.PATCH)],
        cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    # It moves numerical code, so it answers to the bit-for-bit gate.
    assert str(carve.PATCH.relative_to(REPO)) in PATCHES


def test_the_kernels_touch_no_host_service() -> None:
    """No derived type, no pbuf, no clock, no history -- that is the point."""

    module = carve.MODULE.read_text()
    body = module.split("contains", 1)[1]
    for forbidden in ("pbuf", "outfld", "get_nstep", "physics_state", "physics_ptend",
                      "state_loc", "ptend_loc", "%"):
        assert forbidden not in body, f"a carved kernel reaches for {forbidden}"
    uses = re.findall(r"^\s*use\s+(\w+)", module, re.M)
    assert set(uses) == {"shr_kind_mod", "ppgrid", "constituents"}


@pinned
def test_the_two_macrop_driver_patches_compose_in_production_order(tmp_path: Path) -> None:
    """0035 and 0038 edit the same file; zero-context hunks can drift.

    git apply searches when a hunk's line number no longer matches, so a stale
    offset shows up as a wrong-place edit rather than an error.  Apply both in
    the order production uses and check the result by content.
    """

    target = tmp_path / "src/physics/cam"
    target.mkdir(parents=True)
    shutil.copy2(carve.PINNED, target / "macrop_driver.F90")
    ours = [name for name in PATCHES if "macro-split-actions" in name or "macro-carve" in name]
    assert len(ours) == 2 and ours.index(
        "native/pi_cam/control_patches/0035-macro-split-actions.patch"
    ) < ours.index("native/pi_cam/control_patches/0038-macro-carve-arithmetic.patch")
    for name in ours:
        subprocess.run(
            ["git", "apply", "--unidiff-zero", "--verbose", str(REPO / name)],
            cwd=tmp_path, check=True, capture_output=True, text=True,
        )

    result = (target / "macrop_driver.F90").read_text()
    assert result.count("call pycam_macro_before(macro_stage_local, &") == 1
    assert result.count("call pycam_macro_after(macro_stage_local, &") == 1
    assert "call mmacro_pcond(" not in result, "the kernel call should have moved out"
    for block in carve.BLOCKS:
        assert result.count(f"call {block.name}( &") == 1, block.name
    # The carved arithmetic is gone from the driver, exactly once each.
    for signature in ("dpdlfliq(i,k) = ( dlf(i,k)", "clrw_old(i,k) = max(",
                      "cldsice(:ncol,k) = lcwat(:ncol,k)"):
        assert signature not in result, f"{signature!r} survived in the driver"
