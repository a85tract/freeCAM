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


# -- descriptors: the same declarations, seen as direct kernels ---------------

from freecam.pi_cam.kernel_codegen import load_direct_kernels  # noqa: E402
import yaml  # noqa: E402


def _descriptors():
    return {k.name: k for k in load_direct_kernels(carve.DESCRIPTORS)}


@pinned
def test_the_descriptors_are_what_the_generator_produces() -> None:
    assert carve.DESCRIPTORS.read_text() == carve.render_descriptors()


def test_every_lifted_kernel_has_a_descriptor_whose_arguments_are_its_dummies() -> None:
    kernels = _descriptors()
    for block in carve.BLOCKS:
        kernel = kernels[block.name]
        fields = tuple(a.field.removeprefix("macro.") for a in kernel.arguments)
        assert fields == block.arguments, block.name
        assert kernel.routine == block.name
        assert dict(kernel.modules)["pycam_macro_kernels"] == (block.name,)
        # one chunk per slice, chunk axis last, every extent a name the wrapper can see
        for argument in kernel.arguments:
            assert argument.chunk_axis == argument.rank
            assert argument.extents[-1] == "chunks"
            assert set(argument.extents[:-1]) <= {"pcols", "pver", "pcnst", "pwtype", "wtrc_nwset"}


def test_mmacro_pcond_descriptor_is_the_reviewed_specification_in_model() -> None:
    spec = yaml.safe_load(carve.SPEC.read_text())["arguments"]
    kernel = _descriptors()["mmacro_pcond"]
    assert len(kernel.arguments) == len(spec) == 60
    for item, argument in zip(spec, kernel.arguments):
        assert argument.field == f"macro.{item['name']}"
        assert argument.rank == len(item["native_shape"]) + 1
        assert argument.pointer == bool(item.get("pointer"))
        if item.get("carrier") == "logical":
            assert argument.fortran_type == "logical"
        expected = {"structural": "in", "input": "in", "inout": "inout",
                    "workspace": "inout", "output": "out"}[item["role"]]
        if item.get("pointer"):
            expected = "inout"
        assert argument.intent == expected, item["name"]
    assert dict(kernel.modules)["cldwat2m_macro"] == ("mmacro_pcond",)


def test_the_water_tracer_rate_routines_never_get_the_optional_argument() -> None:
    kernels = _descriptors()
    assert [a.field for a in kernels["wtrc_init_rates"].arguments] == ["macro.top_lev", "macro.process_rates"]
    add = [a.field for a in kernels["wtrc_add_rates"].arguments]
    assert add[-1] == "macro.rate" and "macro.do_reverse" not in add


def test_the_descriptors_reach_the_promoted_set_additively() -> None:
    promoted = {k.name: k for k in load_direct_kernels(REPO / "native/pi_cam/direct_kernels_promoted.yaml")}
    ours = _descriptors()
    for name, kernel in ours.items():
        assert name in promoted, name
        assert promoted[name].symbol == kernel.symbol
        assert len(promoted[name].arguments) == len(kernel.arguments)
    # the reviewed base still comes first and dadadj is still there
    assert "dadadj" in promoted
    assert all(not k.symbol.startswith("freecam_pi_cam_promoted_") for k in ours.values()), \
        "our kernels belong in the fixed image, not the promoted add-on"


def test_the_promoted_round_trip_keeps_pointer_and_logical_dummies() -> None:
    """The image build failed once because it did not.

    build_pi_cam_promoted_kernels.py re-serializes every reviewed kernel into
    direct_kernels_promoted.yaml, and the serializer used to write neither
    `pointer` nor `fortran_type`.  The wrapper generated from that file then
    passed mmacro_pcond's six pointer dummies as plain sections and its
    logical as an integer, and the compiler refused both.
    """

    from freecam.pi_cam.kernel_codegen import DirectKernel
    from freecam.pi_cam.process_codegen import direct_kernel_payload

    ours = tuple(load_direct_kernels(carve.DESCRIPTORS))
    payload = direct_kernel_payload(ours)
    again = tuple(DirectKernel.from_payload(item) for item in payload["kernels"])
    for before, after in zip(ours, again):
        for a, b in zip(before.arguments, after.arguments):
            assert a.pointer == b.pointer, (before.name, a.field)
            assert a.fortran_type == b.fortran_type, (before.name, a.field)
    # scoped to mmacro_pcond: other stages contribute logical dummies of
    # their own, and a global count would drift with every one added
    promoted = {k.name: k for k in load_direct_kernels(
        REPO / "native/pi_cam/direct_kernels_promoted.yaml")}
    kernel = promoted["mmacro_pcond"]
    assert sum(1 for a in kernel.arguments if a.pointer) == 6
    assert sum(1 for a in kernel.arguments if a.fortran_type == "logical") == 1
    # and every reviewed kernel's flags survive the round trip into that file
    for name, mine in ((k.name, k) for k in ours):
        for a, b in zip(mine.arguments, promoted[name].arguments):
            assert a.pointer == b.pointer and a.fortran_type == b.fortran_type, (name, a.field)
