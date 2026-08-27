"""The carved microphysics kernels are the driver's own arithmetic, lifted.

micro_mg_cam_tend is 1490 statements with sixty-three live arithmetic
statements outside the routines it calls.  These tests hold the generated
module to the pinned source line by line, and hold the set of carved lines
against every live arithmetic statement, so a rewritten expression or a
statement quietly left for Python fails here rather than in a 512-rank gate.

Dead code is decided the way the driver decides it: by the namelist and
module flags of the admitted configuration -- MG version 1.0, no subcolumns,
prognostic cloud liquid and ice -- tracked through the nesting of `if` and
`select case`, not by a table of line ranges.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/micro_mg_cam.F90"
MODULE = REPO / "native/pi_cam/support/pycam_micro_kernels.F90"
DESCRIPTORS = REPO / "native/pi_cam/direct_kernels_microphysics.yaml"
sys.path.insert(0, str(REPO / "tools"))

pinned = pytest.mark.skipif(not PINNED.is_file(),
                            reason="the pinned iCESM submodule is not checked out")

FIRST, LAST = 1554, 3184           # the routine's executable part


def _blocks():
    import generate_pi_cam_micro_kernels as gen

    return gen.BLOCKS


def _statements():
    """(first, last, text) for every statement, continuations joined."""

    lines = PINNED.read_text().splitlines()
    out, buffer, start = [], "", None
    for number in range(FIRST, LAST + 1):
        text = lines[number - 1].split("!")[0].rstrip()
        if not text.strip():
            continue
        if start is None:
            start = number
        text = (buffer + " " + text.strip()) if buffer else text
        if text.rstrip().endswith("&"):
            buffer = text.rstrip()[:-1]
            continue
        out.append((start, number, text.strip()))
        buffer, start = "", None
    return out


def _norm(condition: str) -> str:
    return re.sub(r"\s+", "", condition.lower())


#: The configuration, as the conditions the driver tests.
DEAD_WHEN_TRUE = {
    "use_subcol_microp", "micro_mg_version>1",
    "micro_mg_version/=1.or.micro_mg_sub_version/=0",
    ".not.(micro_mg_version==1.and.micro_mg_sub_version==0)",
    ".not.do_cldice", ".not.do_cldliq",
}
LIVE_WHEN_TRUE = {
    "micro_mg_version==1.and.micro_mg_sub_version==0", "do_cldice", "do_cldliq",
    "micro_mg_sub_version==0", ".not.use_subcol_microp",
}
#: `select case (micro_mg_version)` / `(micro_mg_sub_version)`: the live case
CASE_LIVE = {"micro_mg_version": "1", "micro_mg_sub_version": "0"}


def _liveness():
    """statement -> live?, following if/else/select nesting as the driver does."""

    result = {}
    stack: list[str] = []            # "dead" | "live" | "neutral" per open block
    selects: list[str | None] = []   # the variable of each open select, or None

    def live() -> bool:
        return all(frame != "dead" for frame in stack)

    for first, last, text in _statements():
        match = re.match(r"(?:\w+\s*:\s*)?if\s*\((.*)\)\s*then\s*$", text, re.I)
        if match:
            result[first] = live()
            condition = _norm(match.group(1))
            stack.append("dead" if condition in DEAD_WHEN_TRUE
                         else "live" if condition in LIVE_WHEN_TRUE else "neutral")
            continue
        if re.match(r"else\s*if\b", text, re.I):
            result[first] = live()
            continue
        if re.match(r"else\s*$", text, re.I):
            if stack:
                stack[-1] = {"dead": "live", "live": "dead"}.get(stack[-1], "neutral")
            result[first] = live()
            continue
        if re.match(r"end\s*if\b|endif\b", text, re.I):
            if stack:
                stack.pop()
            result[first] = live()
            continue
        match = re.match(r"select\s+case\s*\((\w+)\)", text, re.I)
        if match:
            variable = match.group(1).lower()
            selects.append(variable if variable in CASE_LIVE else None)
            stack.append("neutral")
            result[first] = live()
            continue
        match = re.match(r"case\s*(?:\((.*)\)|default)", text, re.I)
        if match and selects:
            variable = selects[-1]
            if variable is not None:
                values = {v.strip() for v in (match.group(1) or "").split(",")}
                stack[-1] = "live" if CASE_LIVE[variable] in values else "dead"
            result[first] = live()
            continue
        if re.match(r"end\s*select", text, re.I):
            if selects:
                selects.pop()
                stack.pop()
            result[first] = live()
            continue
        result[first] = live()
    return result


ARITH = re.compile(r"[-+*/]|\bmax\(|\bmin\(|\bsqrt\(|\bexp\(|\*\*|\bsum\(")
DECL = re.compile(r"^(subroutine|use |implicit|real|integer|logical|character|type\b|intent|"
                  r"end subroutine|contains|#|class)", re.I)
CTRL = re.compile(r"^(if|else|end\s*if|endif|do\b|end\s*do|enddo|select|case|end\s*select|"
                  r"where|elsewhere|end\s*where|return|cycle|exit)", re.I)


def _live_arithmetic() -> dict[int, str]:
    alive = _liveness()
    found = {}
    for first, last, text in _statements():
        if not alive.get(first, True) or DECL.match(text) or CTRL.match(text):
            continue
        if re.match(r"call\s+\w+", text, re.I) or "=" not in text:
            continue
        if ARITH.search(text.split("=", 1)[1]):
            found[first] = text
    return found


# -- the module is generated, and generated from the pinned source ---------------


@pinned
def test_the_committed_module_and_descriptors_are_what_the_generator_writes() -> None:
    import generate_pi_cam_micro_kernels as gen

    assert MODULE.read_text() == gen.render_module()
    assert DESCRIPTORS.read_text() == gen.render_descriptors()


def _skeleton(line: str) -> str:
    text = line.split("!")[0]
    text = re.sub(r"\b\d+\.?\d*(e[-+]?\d+)?(_r8)?\b", "#", text, flags=re.I)
    text = re.sub(r"[A-Za-z_][A-Za-z_0-9%]*", "@", text)
    text = re.sub(r"(@\s*)+", "@", text)
    return re.sub(r"\s+", "", text)


@pinned
def test_every_carved_body_is_the_pinned_text_with_names_substituted() -> None:
    import generate_pi_cam_micro_kernels as gen

    lines = PINNED.read_text().splitlines()
    module = MODULE.read_text()
    for block in _blocks():
        body = block.body(lines)
        assert "\n".join(body) in module, block.name
        expected = []
        for number in range(block.first, block.last + 1):
            if number in block.skip:
                continue
            if number in block.replace:
                expected.extend(block.replace[number])
                continue
            original = lines[number - 1]
            for old, new in gen.COMMON.items():
                original = original.replace(old, new)
            for old, new in block.line_renames.get(number, {}).items():
                original = original.replace(old, new)
            expected.append(original)
        assert len(body) == len(expected), block.name
        for carved, original in zip(body, expected):
            assert _skeleton(carved) == _skeleton(original), f"{block.name}: {original!r}"


@pinned
def test_a_skipped_line_is_a_dead_branch_boundary_or_a_history_write() -> None:
    """The only lines a lift may drop: the dead `if (micro_mg_version > 1)`
    branch and its closing, or an outfld Python makes itself."""

    lines = PINNED.read_text().splitlines()
    for block in _blocks():
        for number in sorted(block.skip):
            text = lines[number - 1].split("!")[0].strip()
            assert (re.match(r"call outfld\(", text)
                    or re.fullmatch(r"end\s*if|endif|else", text, re.I)
                    or text == ""
                    or re.match(r"if \(micro_mg_version > 1\) then", text)
                    or 2768 <= number <= 2794), f"{block.name} drops {number}: {text!r}"
    # the dropped branch is dead: its condition is in the configuration's table
    assert "micro_mg_version>1" in DEAD_WHEN_TRUE


@pinned
def test_every_substitution_is_a_host_association_or_a_named_output() -> None:
    import generate_pi_cam_micro_kernels as gen

    for old, new in gen.COMMON.items():
        assert re.fullmatch(r"state(_loc)?%\w+", old) and re.fullmatch(r"\w+", new), (old, new)
    for block in _blocks():
        for number, renames in block.line_renames.items():
            assert set(renames) == {"ftem_grid"}, (block.name, number)
        for number, added in block.replace.items():
            assert number == 3037 and all(re.fullmatch(r"\w+ = 0\._r8", a) for a in added)


@pinned
def test_the_parameters_are_the_driver_s_own() -> None:
    import generate_pi_cam_micro_kernels as gen

    lines = PINNED.read_text().splitlines()
    for parameter in gen.PARAMETERS:
        name, value = re.match(r"\s*real\(r8\), parameter :: (\w+)\s*=\s*(\S+)", parameter).groups()
        source = next(l for l in lines[1490:1500] if re.search(rf"parameter :: {name}\s*=", l))
        assert re.search(rf"{name}\s*=\s*{re.escape(value)}", source), (name, value)


# -- coverage: nothing is left for Python to compute -----------------------------


@pinned
def test_the_carved_routines_cover_every_live_arithmetic_statement() -> None:
    carved: set[int] = set()
    for block in _blocks():
        carved |= set(range(block.first, block.last + 1)) - block.skip
    live = _live_arithmetic()
    # what the packer-coupled handles lift instead (M-B), and integer-only work
    elsewhere = {
        1554,            # nlev = pver - top_lev + 1: integer arithmetic, exact in Python
        2214, 2215, 2216, 2217, 2218, 2221,   # packer%unpack(...) with an expression
    }
    missing = sorted(n for n in live if n not in carved and n not in elsewhere)
    assert not missing, [(n, live[n][:70]) for n in missing]
    assert len(live) >= 60, len(live)


@pinned
def test_arithmetic_inside_a_call_s_arguments_is_only_where_the_handles_keep_it() -> None:
    """Python cannot evaluate an expression passed straight to a routine."""

    alive = _liveness()
    offenders = {}
    for first, last, text in _statements():
        if not alive.get(first, True):
            continue
        match = re.match(r"call\s+(\w+)\s*\((.*)\)\s*$", text, re.I)
        if not match:
            continue
        arguments = re.sub(r"'[^']*'", "", match.group(2))
        if re.search(r"[\w)]\s*[-+*/]\s*[\w(]", arguments):
            offenders[first] = match.group(1)
    # each of these is kept whole inside its handle wrapper (M-B), which
    # receives dtime and num_steps and forms the expression in Fortran:
    # the substep length the core and physics_update take, the 1/num_steps
    # scale, and the water-tracer rate sums
    assert set(offenders.values()) <= {
        "wtrc_add_rates", "micro_mg_tend1_0", "physics_update", "physics_ptend_scale"}, offenders


# -- the module is an addition, not a patch --------------------------------------


def test_the_module_touches_no_derived_type_no_buffer_and_no_history() -> None:
    code = "\n".join(line.split("!")[0] for line in MODULE.read_text().splitlines())
    for forbidden in ("physics_state", "physics_ptend", "physics_buffer", "pbuf", "outfld",
                      "cam_history", "MGPacker", "MGPostProc", "packer%", "post_proc%"):
        assert forbidden not in code, forbidden
    calls = re.findall(r"^\s*call\s+(\w+)", code, re.M)
    assert set(calls) == {"size_dist_param_liq", "size_dist_param_basic"}, calls
    assert "avg_diameter(" in code
