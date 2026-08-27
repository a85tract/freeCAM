"""The carved radiation kernels are the driver's own arithmetic, lifted.

The point of the lift is that no expression is rewritten.  These tests hold
the generated module against the pinned source line by line, and hold the
set of carved lines against every live arithmetic statement of
``radiation_tend``, so a body that drifts or a statement that is quietly
left for Python to compute fails here rather than in a 512-rank gate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/rrtmg/radiation.F90"
MODULE = REPO / "native/pi_cam/support/pycam_rad_kernels.F90"
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_radiation.yaml"
sys.path.insert(0, str(REPO / "tools"))

pinned = pytest.mark.skipif(not PINNED.is_file(),
                            reason="the pinned iCESM submodule is not checked out")

#: Branches the admitted configuration never enters: oldcldoptics is .false.
#: and icecldoptics/liqcldoptics are 'mitchell'/'gammadist' in atm_in, docosp
#: and dohirs are .false. by default, spectralflux is .false., and this is
#: not a single-column run.  ``Radiation.refuse_unsupported`` asserts each
#: of these against the image before a timestep runs.
DEAD = ((862, 866), (869, 874), (891, 893), (896, 897), (904, 905),
        (949, 951), (954, 955), (962, 963), (1201, 1236), (1259, 1273))

FIRST, LAST = 848, 1320       # radiation_tend's executable body


def _blocks():
    import generate_pi_cam_rad_kernels as gen

    return gen.blocks(PINNED.read_text().splitlines())


def _statements(first: int, last: int):
    """(first_line, text) for every statement, continuations joined."""

    lines = PINNED.read_text().splitlines()
    out, buffer, start = [], "", None
    for number in range(first, last + 1):
        text = lines[number - 1].split("!")[0].rstrip()
        if not text.strip():
            continue
        if start is None:
            start = number
        text = (buffer + " " + text.strip()) if buffer else text
        if text.rstrip().endswith("&"):
            buffer = text.rstrip()[:-1]
            continue
        out.append((start, text.strip()))
        buffer, start = "", None
    return out


def _dead(number: int) -> bool:
    return any(a <= number <= b for a, b in DEAD)


ARITH = re.compile(r"[-+*/]|\bmax\(|\bmin\(|\bsqrt\(|\bexp\(|\*\*")
DECL = re.compile(r"^(subroutine|use |implicit|real|integer|logical|character|type|intent|"
                  r"end |contains|#)", re.I)
CTRL = re.compile(r"^(if|else|elseif|end\s*if|endif|do\b|end\s*do|enddo|select|case|"
                  r"end\s*select|where|elsewhere|end\s*where|return|cycle|exit)", re.I)


def _live_arithmetic() -> set[int]:
    """Every live statement that computes a floating-point number."""

    found = set()
    for number, text in _statements(FIRST, LAST):
        if _dead(number) or DECL.match(text) or CTRL.match(text):
            continue
        if re.match(r"call\s+\w+", text, re.I):
            continue
        if "=" not in text:
            continue
        if ARITH.search(text.split("=", 1)[1]):
            found.add(number)
    return found


# -- the module is generated, and generated from the pinned source ---------------


@pinned
def test_the_committed_module_and_descriptors_are_what_the_generator_writes() -> None:
    import generate_pi_cam_rad_kernels as gen

    assert MODULE.read_text() == gen.render_module()
    assert DESCRIPTORS.read_text() == gen.render_descriptors()


@pinned
def test_every_carved_body_is_the_pinned_text_with_names_substituted() -> None:
    """Names may be substituted; structure may not change.

    Comparing the carved line with the original after undoing the renames is
    circular -- that is what the generator did.  What is worth asserting is
    that nothing but an identifier differs: strip every identifier and
    number and the operator skeleton of the two lines must be identical, so
    a rewritten expression, a reordered term or a changed bound fails here.
    """

    lines = PINNED.read_text().splitlines()
    module = MODULE.read_text()
    for block in _blocks():
        body = block.body(lines)
        assert "\n".join(body) in module, block.name
        source = [lines[n - 1] for n in range(block.first, block.last + 1)
                  if n not in block.skip]
        assert len(body) == len(source), block.name
        for carved, original in zip(body, source):
            if _rewritten(original):
                continue
            substituted = original
            for old, new in block.renames.items():
                substituted = substituted.replace(old, new)
            assert _skeleton(carved) == _skeleton(substituted), \
                f"{block.name}: {original!r} became {carved!r}"


def _skeleton(line: str) -> str:
    """The line's operators and punctuation, with identifiers and numbers gone."""

    text = line.split("!")[0]
    text = re.sub(r"\b\d+\.?\d*(_r8)?\b", "#", text)          # numeric literals
    text = re.sub(r"[A-Za-z_][A-Za-z_0-9%]*", "@", text)       # identifiers, components
    text = re.sub(r"(@\s*)+", "@", text)                       # a renamed name may split
    return re.sub(r"\s+", "", text)


def _rewritten(original: str) -> bool:
    """A line the generator replaced rather than renamed, listed so it is visible."""

    return "lwupcgs(i) = 1000*stebol*tground(1)**4" in original


@pinned
def test_every_substitution_is_a_host_association_or_a_guard_becoming_an_argument() -> None:
    """The renames are the contract the structure test leans on, so they are
    held to a shape: a derived-type component the routine now receives by
    name, or a module-state guard that becomes a logical argument."""

    import generate_pi_cam_rad_kernels as gen

    guards = {"cldfsnow_idx > 0": "has_snow",
              "single_column.and.scm_crm_mode.and.have_tg": "refused_scm",
              "conserve_energy": "conserve_energy"}
    for block in _blocks():
        for old, new in block.renames.items():
            if guards.get(old) == new:
                continue
            if re.fullmatch(r"\w+%\w+", old) and re.fullmatch(r"\w+", new):
                continue                                   # state%t -> t
            if re.fullmatch(r"\w+\(:ncol,:pver\)", old):
                continue                                   # qrs(:ncol,:pver) -> field(...)
            if old == new:
                continue
            assert old == "lwupcgs(i) = 1000*stebol*tground(1)**4", \
                f"{block.name} substitutes {old!r} -> {new!r}, which is neither"
    assert set(gen.COMMON) - set(guards) == {
        "state%t", "state%pmid", "state%pdel", "state%pint", "state%lnpint",
        "state%lnpmid", "cam_in%lwup"}


@pinned
def test_a_skipped_line_is_only_a_call_python_makes_or_a_guard_outside_the_range() -> None:
    lines = PINNED.read_text().splitlines()
    for block in _blocks():
        for number in sorted(block.skip):
            text = lines[number - 1].split("!")[0].strip()
            assert re.match(r"call\s+(get_snow_optics_sw|snow_cloud_get_rad_props_lw)\b", text) \
                or re.fullmatch(r"end\s*if|endif", text, re.I), \
                f"{block.name} drops {number}: {text!r}"


# -- coverage: nothing is left for Python to compute -----------------------------


@pinned
def test_the_carved_routines_cover_every_live_arithmetic_statement() -> None:
    carved = set()
    for block in _blocks():
        if block.name == "rad_inp":
            continue                      # radinp is a routine, not driver-body lines
        carved |= (set(range(block.first, block.last + 1)) - block.skip) | block.covers
    missing = sorted(_live_arithmetic() - carved)
    assert not missing, (
        "these live statements compute a number and no carved routine covers them: "
        + "; ".join(f"{n}: {t}" for n, t in _statements(FIRST, LAST) if n in missing))


@pinned
def test_no_live_call_hides_arithmetic_in_its_argument_list() -> None:
    """Python cannot evaluate an expression passed straight to a routine.

    The two ``outfld(..., qrl(:ncol,:)/cpair, ncol, ...)`` calls are the whole
    of this class in radiation_tend; they are reproduced by
    ``pycam_rad_outfld_scaled_v1`` in the handles module rather than split.
    """

    known = {1170, 1171}
    offenders = set()
    for number, text in _statements(FIRST, LAST):
        if _dead(number):
            continue
        match = re.match(r"call\s+\w+\s*\((.*)\)\s*$", text, re.I)
        if not match:
            continue
        arguments = re.sub(r"'[^']*'", "", match.group(1))
        arguments = re.sub(r"//\s*diag\(icall\)", "", arguments)
        if re.search(r"[\w)]\s*[-+*/]\s*[\w(]", arguments):
            offenders.add(number)
    assert offenders == known, sorted(offenders ^ known)


@pinned
def test_no_live_branch_condition_needs_a_floating_point_comparison() -> None:
    """Python branches on these, so each must be boolean or integer."""

    carved = set()
    for block in _blocks():
        carved |= set(range(block.first, block.last + 1))
    for number, text in _statements(FIRST, LAST):
        if _dead(number) or number in carved:
            continue
        match = re.match(r"(?:else\s*)?if\s*\((.*)\)\s*then\s*$", text, re.I)
        if not match:
            continue
        condition = match.group(1)
        assert not re.search(r"[-+*/]|\.gt\.|\.lt\.|\.ge\.|\.le\.|[<>]\s*[-\d.]*\d\.",
                             condition), f"{number}: {condition!r}"


# -- the module is an addition, not a patch --------------------------------------


def test_the_generator_never_edits_the_radiation_driver() -> None:
    text = (REPO / "tools/generate_pi_cam_rad_kernels.py").read_text()
    assert "radiation.F90" in text                      # it reads the pinned source
    assert ".write_text" in text
    written = re.findall(r"for path, rendered in \(\((\w+), .*?\((\w+),", text, re.S)
    assert written == [("MODULE", "DESCRIPTORS")], written


def test_the_module_touches_no_derived_type_no_buffer_and_no_history() -> None:
    code = "\n".join(line.split("!")[0] for line in MODULE.read_text().splitlines())
    for forbidden in ("physics_state", "physics_ptend", "physics_buffer", "pbuf",
                      "cam_history", "outfld", "rrtmg_state"):
        assert forbidden not in code, forbidden
    # the one call the lift keeps, and why
    assert code.count("call ") == 1 and "shr_orb_decl" in code
