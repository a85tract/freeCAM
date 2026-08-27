"""The stage glue's arithmetic, lifted verbatim from tphysbc."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/physpkg.F90"
MODULE = REPO / "native/pi_cam/support/pycam_mm_kernels.F90"
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_mm.yaml"
sys.path.insert(0, str(REPO / "tools"))

pinned = pytest.mark.skipif(not PINNED.is_file(),
                            reason="the pinned iCESM submodule is not checked out")


@pinned
def test_the_committed_module_and_descriptors_are_what_the_generator_writes() -> None:
    import generate_pi_cam_mm_kernels as gen

    assert MODULE.read_text() == gen.render_module()
    assert DESCRIPTORS.read_text() == gen.render_descriptors()


@pinned
def test_every_body_is_the_pinned_text_character_for_character() -> None:
    """No renames at all here: tphysbc's locals keep their names as dummies."""

    import generate_pi_cam_mm_kernels as gen

    lines = PINNED.read_text().splitlines()
    module = MODULE.read_text()
    for block in gen.BLOCKS:
        assert not block.renames, block.name
        body = block.body(lines)
        assert "\n".join(body) in module, block.name
        for carved, number in zip(body, range(block.first, block.last + 1)):
            assert carved.strip() == lines[number - 1].strip(), (block.name, number)


@pinned
def test_the_twelve_glue_statements_and_the_substep_length_are_all_covered() -> None:
    """The stage's arithmetic between the two drivers, by line, and nothing
    that belongs to a dead branch (the CLUBB and CARMA sums)."""

    import generate_pi_cam_mm_kernels as gen

    lines = PINNED.read_text().splitlines()
    covered = {n for b in gen.BLOCKS for n in range(b.first, b.last + 1)}
    expected = {2210, 2254, 2255, 2369, 2370, 2371, 2372, 2376, 2377, 2378, 2379, 2380, 2381}
    assert covered == expected, sorted(covered ^ expected)
    # the lines are what the plan says they are
    assert "cld_macmic_ztodt = ztodt/cld_macmic_num_steps" in lines[2209]
    assert "flx_cnd(:ncol) = -1._r8*rliq(:ncol)" in lines[2253]
    assert "prec_str(:ncol) = prec_pcw(:ncol) + prec_sed(:ncol)" in lines[2379]
    # the CLUBB flux terms and the CARMA sums are dead here and not lifted
    assert "cam_in%shf(:ncol) + det_s(:ncol)" in lines[2280]
    assert 2281 not in covered and 2387 not in covered


def test_the_module_touches_nothing_but_arrays() -> None:
    code = "\n".join(line.split("!")[0] for line in MODULE.read_text().splitlines())
    for forbidden in ("physics_state", "physics_ptend", "pbuf", "outfld", "cam_in", "call "):
        assert forbidden not in code, forbidden
    assert code.count("subroutine ") == 8           # four routines, opened and closed
