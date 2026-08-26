"""The carved macrophysics kernels are the driver's own arithmetic, lifted."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import generate_pi_cam_macro_kernels as carve  # noqa: E402
from apply_pi_cam_source_patches import PATCHES, SUPPORT_SOURCES  # noqa: E402

pinned = pytest.mark.skipif(
    not carve.PINNED.is_file(),
    reason="the pinned iCESM submodule is not checked out",
)


@pinned
def test_the_module_is_what_the_pinned_source_produces() -> None:
    assert carve.MODULE.read_text() == carve.render_module()


@pinned
def test_every_carved_body_is_the_original_text_renamed_and_nothing_else() -> None:
    """The whole claim of this module, checked line for line.

    The arithmetic now exists twice -- in the oracle's driver and here -- and
    only one of the two is validated by machine code.  If a body here were
    retyped rather than lifted, an expression could differ by a parenthesis
    or an operand order and still compile; the Python-driven run would then
    fail its bit-for-bit gate with nothing to point at.  So compare the text.
    """

    source = carve.PINNED.read_text().splitlines()
    module = carve.MODULE.read_text()
    for block in carve.BLOCKS:
        expected = "\n".join(block.body(source))
        assert expected.strip(), block.name
        assert expected in module, f"{block.name} body is not the pinned text"
        for original in block.renames:
            if original.endswith(")") or " " in original:
                continue
            assert original not in expected, f"{block.name} still refers to {original}"


@pinned
def test_the_blocks_cover_every_arithmetic_statement_of_the_driver() -> None:
    """Nothing that computes may be left for Python to reproduce."""

    lines = carve.PINNED.read_text().splitlines()
    covered = set()
    for block in carve.BLOCKS:
        covered.update(range(block.first, block.last + 1))
    # executable part of macrop_driver_tend: after the declarations, before end
    arithmetic = []
    for number in range(612, 1224):
        text = lines[number - 1].split("!")[0].strip()
        if not text or text.startswith(("call ", "if", "do ", "end", "else", "enddo", "endif")):
            continue
        if "=" not in text or text.startswith(("real", "integer", "logical", "type")):
            continue
        rhs = text.split("=", 1)[1]
        if re.search(r"[*/]|[+-]\s*[a-zA-Z(]", rhs) and not re.fullmatch(r"\s*0\._r8\s*", rhs):
            arithmetic.append(number)
    uncovered = [n for n in arithmetic if n not in covered]
    # `rdtime = 1._r8/dtime` and `latsub = latvap + latice` are scalars the
    # caller computes and passes; `det_ice(:ncol) = det_ice(:ncol)/1000._r8`
    # sits between two blocks and is one array statement Python asks the
    # image for as well.  Anything else uncovered is a hole.
    allowed = {n for n in uncovered if any(
        token in lines[n - 1] for token in ("rdtime = 1._r8/dtime", "latsub = latvap + latice",
                                            "det_ice(:ncol) = det_ice(:ncol)/1000._r8")
    )}
    assert set(uncovered) == allowed, [lines[n - 1].strip() for n in uncovered if n not in allowed]


@pinned
def test_the_refusals_became_a_status_not_an_abort() -> None:
    module = carve.MODULE.read_text()
    body = module.split("subroutine macrop_kernel_to_ptend", 1)[1].split("end subroutine", 1)[0]
    assert "call endrun" not in body, "a carved kernel must not abort the model itself"
    assert body.count("status = 1") + body.count("status = 2") == 4
    assert "status = 0" in body


def test_the_kernels_touch_no_host_service() -> None:
    """No derived type, no pbuf, no clock, no history -- that is the point."""

    module = carve.MODULE.read_text()
    # Code only: a comment is allowed to say what the original did.
    body = "\n".join(line.split("!")[0] for line in module.split("contains", 1)[1].splitlines())
    for forbidden in ("pbuf", "outfld", "get_nstep", "physics_state", "physics_ptend",
                      "state_loc", "ptend_loc", "%", "endrun"):
        assert forbidden not in body, f"a carved kernel reaches for {forbidden}"
    uses = re.findall(r"^\s*use\s+(\w+)", module, re.M)
    assert set(uses) == {"shr_kind_mod", "ppgrid", "constituents"}


def test_the_module_is_an_addition_and_never_a_replacement() -> None:
    """The oracle's macrop_driver.o must stay byte for byte what the gate ran.

    Recompiling a numerical object -- even from unchanged source -- has
    produced ULP differences in this repository before, so no patch may edit
    macrop_driver.F90 and no numerical object may be replaced.  The module is
    copied in beside the source and reached only from Python.
    """

    assert any(source.endswith("pycam_macro_kernels.F90") for source, _ in SUPPORT_SOURCES)
    for name in PATCHES:
        text = (REPO / name).read_text()
        assert "macrop_driver.F90" not in text, f"{name} edits the macrophysics driver"
        assert "pycam_macro_kernels" not in text, f"{name} wires the kernels into Fortran"
