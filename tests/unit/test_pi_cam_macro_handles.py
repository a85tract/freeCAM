"""The handles module: derived types stay in Fortran, Python gets views."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "native/pi_cam/support/pycam_macro_handles.F90"

#: Every C entry the module promises.  Python's macrophysics class binds
#: exactly these; a rename on either side fails here first.
ENTRIES = (
    "pycam_macro_set_owner_v1",
    "pycam_macro_owner_v1",
    "pycam_macro_state_copy_v1",
    "pycam_macro_state_dealloc_v1",
    "pycam_macro_ptend_init_v1",
    "pycam_macro_ptend_sum_v1",
    "pycam_macro_update_v1",
    "pycam_macro_view_v1",
    "pycam_macro_cldfrc_v1",
    "pycam_macro_wtrc_apply_v1",
    "pycam_outfld_v1",
)


def _text() -> str:
    return MODULE.read_text()


def _code() -> str:
    return "\n".join(line.split("!")[0] for line in _text().splitlines())


def test_every_promised_entry_is_bound_under_its_own_name() -> None:
    code = _code()
    for entry in ENTRIES:
        assert re.search(rf"function {entry}\(", code), entry
        assert f"bind(C, name='{entry}')" in code, entry


def test_the_module_sits_below_the_control_layer() -> None:
    """physpkg and cam_comp use it; it must not use them back."""

    uses = {m.lower() for m in re.findall(r"^\s*use\s+(\w+)", _code(), re.M)}
    assert "physpkg" not in uses and "cam_comp" not in uses
    # and nothing numerical of its own: every arithmetic call is the original
    assert uses >= {"physics_types", "physics_buffer", "cloud_fraction", "water_tracers", "cam_history"}


def test_view_codes_are_a_single_table_python_can_mirror() -> None:
    codes = dict(re.findall(r"integer\(c_int\), parameter, public :: (view_\w+) = (\d+)", _text()))
    assert len(codes) == 14
    assert len(set(codes.values())) == len(codes), "two views share a code"
    assert codes["view_state_t"] == "1" and codes["view_ptend_q"] == "22"
    assert codes["view_process_rates"] == "33"
    # every code the table declares is served by the dispatcher
    body = _code().split("function pycam_macro_view_v1", 1)[1].split("end function", 1)[0]
    for name in codes:
        assert f"case ({name})" in body, name


def test_liveness_is_tracked_rather_than_inquired() -> None:
    """ALLOCATED is wrong for the pointer-shell build; ASSOCIATED is wrong for
    the oracle build.  The module must use neither on state components."""

    code = _code()
    assert not re.search(r"allocated\(macro_state_loc\(lchnk\)%", code)
    assert not re.search(r"associated\(macro_state_loc\(lchnk\)%", code)
    assert "state_live(lchnk) = .true." in code
    assert "state_live(lchnk) = .false." in code
    # the routines that read the state copy refuse a dead one
    for entry in ("pycam_macro_state_dealloc_v1", "pycam_macro_wtrc_apply_v1"):
        body = code.split(f"function {entry}", 1)[1].split("end function", 1)[0]
        assert "state_live(lchnk)" in body, entry


def test_ptend_init_reproduces_both_forms_the_driver_uses() -> None:
    body = _code().split("function pycam_macro_ptend_init_v1", 1)[1].split("end function", 1)[0]
    # with ls and lq, and with the name alone -- physics_ptend_init treats an
    # absent flag differently from a false one
    assert body.count("ls=(ls /= 0_c_int), lq=lq_flags") == 2
    assert body.count("fname(1:name_len))") == 2
    assert "host_state(lchnk)%psetcols" in body


def test_wtrc_apply_is_called_in_the_driver_s_exact_form() -> None:
    body = _code().split("function pycam_macro_wtrc_apply_v1", 1)[1].split("end function", 1)[0]
    assert ".false., pre_rates=macro_process_rates(:,:,:,:,:,lchnk), prelat=prelat)" in body


def test_outfld_receives_a_fortran_string_of_the_caller_s_length() -> None:
    body = _code().split("function pycam_outfld_v1", 1)[1].split("end function", 1)[0]
    assert "call outfld(fname(1:name_len), field, idim, lchnk)" in body
    assert "field(idim,*)" in body


def test_the_resume_stage_reads_only_what_the_module_publishes() -> None:
    public = set()
    for line in _text().splitlines():
        m = re.match(r"\s*public :: (.*)", line)
        if m:
            public.update(x.strip() for x in m.group(1).split(","))
    for name in ("macro_ptend", "macro_det_s", "macro_det_ice", "python_owns_tend", "pycam_macro_bind_hosts"):
        assert name in public, name
    # the working copies stay private: Python reaches them through views
    for name in ("macro_state_loc", "macro_ptend_loc", "macro_process_rates"):
        assert name not in public, name
