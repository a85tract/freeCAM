"""The microphysics handles module: the packer section verbatim, the rest by entry."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/micro_mg_cam.F90"
MODULE = REPO / "native/pi_cam/support/pycam_micro_handles.F90"
VIEWS = REPO / "native/pi_cam/micro_views.yaml"
sys.path.insert(0, str(REPO / "tools"))

pinned = pytest.mark.skipif(not PINNED.is_file(),
                            reason="the pinned iCESM submodule is not checked out")

ENTRIES = (
    "pycam_micro_set_owner_v1", "pycam_micro_configure_v1", "pycam_micro_nstep_v1",
    "pycam_micro_dt_v1", "pycam_micro_view_v1",
    "pycam_micro_begin_v1", "pycam_micro_ptend_init_v1", "pycam_micro_bind_input_v1",
    "pycam_micro_pack_prelude_v1", "pycam_micro_substep_pack_v1", "pycam_micro_core_v1",
    "pycam_micro_substep_unpack_v1", "pycam_micro_post_proc_v1",
    "pycam_micro_wtrc_apply_v1", "pycam_micro_wtrc_add_sum_v1",
    "pycam_micro_output_precip_v1", "pycam_micro_end_v1",
)


def _code() -> str:
    return "\n".join(line.split("!")[0] for line in MODULE.read_text().splitlines())


@pinned
def test_the_committed_module_and_view_table_are_what_the_generator_writes() -> None:
    import generate_pi_cam_micro_handles as gen

    assert MODULE.read_text() == gen.render_module()
    assert VIEWS.read_text() == gen.render_views()


def test_every_promised_entry_is_bound_and_nothing_else_is() -> None:
    code = _code()
    bound = set(re.findall(r"bind\(C, name='([a-z_0-9]+)'\)", code))
    assert bound == set(ENTRIES), sorted(bound ^ set(ENTRIES))
    for entry in ENTRIES:
        assert re.search(rf"function {entry}\(", code), entry


@pinned
def test_the_five_procedures_and_the_helpers_are_the_pinned_text_verbatim() -> None:
    import generate_pi_cam_micro_handles as gen

    lines = PINNED.read_text().splitlines()
    module = MODULE.read_text()
    for name, first, last in gen.VERBATIM:
        body = module[module.index(f"  subroutine {name}()"):]
        body = body[:body.index(f"  end subroutine {name}")]
        def code_only(items):
            return [s for s in (x.strip() for x in items) if s and not s.startswith("!")]

        expected = code_only(lines[n - 1] for n in range(first, last + 1))
        carried = code_only(body.splitlines())
        carried = [s for s in carried if not s.startswith(("subroutine", "integer :: i"))]
        assert carried == expected, name
    # the lq build and the cldwat ptend, 1743-1766, inside ptend_init
    entry = module[module.index("function pycam_micro_ptend_init_v1("):]
    entry = entry[:entry.index("end function pycam_micro_ptend_init_v1")]
    for n in range(1743, 1767):
        if lines[n - 1].strip():
            assert lines[n - 1].strip() in entry, n
    # the pointer helpers add_field takes
    for n in range(3186, 3197):
        if lines[n - 1].strip():
            assert lines[n - 1].strip() in module, n


@pinned
def test_the_procedures_hold_nothing_python_walks_itself() -> None:
    """No buffer read, no history write, no subcolumn averaging in the
    verbatim ranges: those stay with Python, statement for statement."""

    lines = PINNED.read_text().splitlines()
    import generate_pi_cam_micro_handles as gen

    for name, first, last in gen.VERBATIM:
        text = "\n".join(lines[first - 1:last])
        for forbidden in ("pbuf_get_field", "outfld", "subcol_field_avg"):
            assert forbidden not in text, (name, forbidden)


def test_the_configuration_is_module_state_set_from_python_never_used_from_the_driver() -> None:
    """micro_mg_cam's flags are private to it; Python reads them off the image
    and passes them in, and the verbatim text tests the copies."""

    code = _code()
    assert "use micro_mg_cam" not in code
    import generate_pi_cam_micro_handles as gen

    for name, kind in gen.CONFIGURATION:
        assert re.search(rf"^\s*{kind}, save :: {name}\s*$", code, re.M), name
        assert f"{name}_in" in code, name


def test_the_view_table_and_the_module_agree_and_codes_are_unique() -> None:
    import yaml

    table = yaml.safe_load(VIEWS.read_text())
    code = _code()
    codes = [row["code"] for row in table["views"]]
    assert len(codes) == len(set(codes)) and codes == sorted(codes)
    for row in table["views"]:
        assert f"view_{row['name']} = {row['code']}" in code, row["name"]
        assert f"call view{row['rank']}({row['expression']}, ptr, ndims, extents)" in code, row["name"]
    assert len(table["views"]) > 200
    inputs = table["inputs"]
    assert [r["code"] for r in inputs] == list(range(1, len(inputs) + 1))
    for row in inputs:
        assert f"call c_f_pointer(ptr, {row['name']}," in code, row["name"]


def test_the_core_is_skipped_when_a_model_owns_it() -> None:
    code = _code()
    body = code[code.index("function pycam_micro_core_v1("):]
    body = body[:body.index("end function pycam_micro_core_v1")]
    assert "if (.not. python_owns_core) call micro_core()" in body


def test_the_module_sits_below_the_control_layer() -> None:
    used = set(re.findall(r"^\s*use\s+(\w+)", _code(), re.M))
    for control in ("physpkg", "cam_comp", "atm_comp_mct", "micro_mg_cam",
                    "pycam_macro_handles", "pycam_rad_handles"):
        assert control not in used, control
