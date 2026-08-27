"""The radiation handles module: derived types stay in Fortran, Python gets views.

radiation_tend's arithmetic is lifted into ``pycam_rad_kernels``; everything
else it does is a call taking a derived type, and each of those gets one
``bind(C)`` wrapper here.  These tests hold the module to three promises: it
exposes exactly the entries Python binds, it sits below the control layer so
nothing it uses can use it back, and it hands out no address for storage that
is not alive.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "native/pi_cam/support/pycam_rad_handles.F90"
sys.path.insert(0, str(REPO / "tools"))

#: Every C entry the module promises.  Python's Radiation class binds exactly
#: these; a rename on either side fails here first.
ENTRIES = (
    # the shared core every NativeStage expects
    "pycam_rad_set_owner_v1", "pycam_rad_bind_hosts_v1", "pycam_rad_view_v1",
    "pycam_rad_nstep_v1", "pycam_rad_dt_v1",
    # radiation's own queries: the driver's predicates, never re-derived
    "pycam_rad_calday_v1", "pycam_rad_do_v1", "pycam_rad_latlon_v1",
    # zenith is a bare external subroutine, not a module procedure, so it
    # cannot be a generated direct kernel and lives here instead
    "pycam_rad_zenith_v1",
    "pycam_rad_options_v1", "pycam_rad_hist_active_v1",
    # the RRTMG state's chunk-local lifetime
    "pycam_rad_rstate_create_v1", "pycam_rad_rstate_update_v1",
    "pycam_rad_rstate_destroy_v1",
    # cloud optics, the branches this configuration takes
    "pycam_rad_ice_optics_sw_v1", "pycam_rad_liquid_optics_sw_v1",
    "pycam_rad_snow_optics_sw_v1", "pycam_rad_ice_props_lw_v1",
    "pycam_rad_liquid_props_lw_v1", "pycam_rad_snow_props_lw_v1",
    # aerosol optics
    "pycam_rad_aer_props_sw_v1", "pycam_rad_aer_props_lw_v1",
    # the two numerical cores a model may replace
    "pycam_rad_rrtmg_sw_v1", "pycam_rad_rrtmg_lw_v1",
    # the rest of the driver's derived-type calls
    "pycam_rad_tropopause_find_v1", "pycam_rad_cnst_out_v1",
    "pycam_rad_data_write_v1", "pycam_rad_radheat_v1",
    # the one history call that carries arithmetic
    "pycam_rad_outfld_scaled_v1",
)


def _text() -> str:
    return MODULE.read_text()


def _code() -> str:
    return "\n".join(line.split("!")[0] for line in _text().splitlines())


# -- what the module promises ----------------------------------------------------


def test_the_committed_module_is_what_the_generator_writes() -> None:
    import generate_pi_cam_rad_handles as gen

    assert MODULE.read_text() == gen.render_module()


def test_every_promised_entry_is_bound_under_its_own_name() -> None:
    code = _code()
    for entry in ENTRIES:
        assert re.search(rf"function {entry}\(", code), entry
        assert f"bind(C, name='{entry}')" in code, entry


def test_the_module_promises_nothing_beyond_the_listed_entries() -> None:
    bound = set(re.findall(r"bind\(C, name='([a-z_0-9]+)'\)", _code()))
    assert bound == set(ENTRIES), sorted(bound ^ set(ENTRIES))


# -- where it sits ---------------------------------------------------------------


def test_the_module_sits_below_the_control_layer() -> None:
    """physpkg and the generated cam_comp use it; it must not use them back."""

    used = set(re.findall(r"^\s*use\s+(\w+)", _code(), re.M))
    for control in ("physpkg", "cam_comp", "atm_comp_mct", "pycam_macro_handles"):
        assert control not in used, control


def test_it_replaces_no_numerical_routine_and_only_calls_the_originals() -> None:
    code = _code()
    # every routine it calls is one the oracle already provides
    # the lookbehind matters: without it `icall` followed by a `type(...)`
    # declaration reads as a call to `type`
    called = set(re.findall(r"(?<![\w%])call\s+(\w+)\s*\(", code))
    own = set(re.findall(r"subroutine\s+(\w+)", code)) | {"view1", "view2"}
    outside = called - own
    assert outside == {
        "get_ice_optics_sw", "get_liquid_optics_sw", "get_snow_optics_sw",
        "ice_cloud_get_rad_props_lw", "liquid_cloud_get_rad_props_lw",
        "snow_cloud_get_rad_props_lw", "aer_rad_props_sw", "aer_rad_props_lw",
        "rrtmg_state_update", "rrtmg_state_destroy", "rad_rrtmg_sw", "rad_rrtmg_lw",
        "tropopause_find", "rad_cnst_out", "rad_data_write", "radheat_tend",
        "rad_cnst_get_call_list", "get_rlat_all_p", "get_rlon_all_p", "outfld",
        "zenith", "view1", "view2",
    } - own, sorted(outside)
    # rrtmg_state_create is a function, so it is referenced rather than called
    assert "rrtmg_state_create(" in code


def test_the_module_computes_nothing_a_gate_would_notice() -> None:
    """The one arithmetic expression here is the outfld line kept whole."""

    arithmetic = []
    for line in _code().splitlines():
        text = line.strip()
        if not text or text.startswith(("real", "integer", "logical", "type", "use ")):
            continue
        assignment = re.match(r"[\w%()\s,:]+?=(?!=|>)(.*)$", text)
        if not assignment:
            continue
        right = assignment.group(1)
        right = re.sub(r"/=|=>|_c_int|_c_int64_t|_r8", "", right)   # not operators
        right = re.sub(r"(?<![\w)])-(?=\d)", "", right)             # a negative literal
        if re.search(r"[-+*/]", right):
            arithmetic.append(text)
    # codes(4) = codes(4) + 1 counts how many radiation calls are active, an
    # integer tally.  Nothing else here computes a model value: the one
    # floating-point expression is radiation.F90:1170-1171 kept whole, which
    # lives inside a call argument and so is not an assignment.
    assert [a for a in arithmetic if "codes(4)" not in a] == []
    assert "field(:ncol,:)/cpair_in" in _code()


# -- liveness --------------------------------------------------------------------


def test_no_view_hands_out_an_address_for_storage_that_is_not_alive() -> None:
    import generate_pi_cam_rad_handles as gen

    body = _code()[_code().index("function pycam_rad_view_v1"):]
    body = body[:body.index("end function pycam_rad_view_v1")]
    for name, (code, rank, expression, guard) in gen.VIEWS.items():
        case = body[body.index(f"case (view_{name})"):]
        case = case[:case.index("call view")]
        if guard:
            assert f"if (.not. {guard}) return" in case, name
        assert f"call view{rank}({expression}" in body, name


def test_the_rrtmg_state_views_are_all_guarded_by_its_lifetime() -> None:
    import generate_pi_cam_rad_handles as gen

    for name, (_, _, _, guard) in gen.VIEWS.items():
        if name.startswith("rstate_"):
            assert guard == "rstate_live", name


def test_the_cores_refuse_to_run_without_an_rrtmg_state() -> None:
    code = _code()
    for entry in ("pycam_rad_rrtmg_sw_v1", "pycam_rad_rrtmg_lw_v1"):
        body = code[code.index(f"function {entry}("):]
        body = body[:body.index(f"end function {entry}")]
        assert "if (.not. rstate_live) return" in body, entry
        assert "status = 2_c_int" in body, entry


def test_every_wrapper_refuses_a_chunk_it_does_not_own() -> None:
    code = _code()
    skip = {"pycam_rad_set_owner_v1", "pycam_rad_bind_hosts_v1", "pycam_rad_nstep_v1",
            "pycam_rad_dt_v1", "pycam_rad_calday_v1", "pycam_rad_do_v1",
            "pycam_rad_options_v1", "pycam_rad_hist_active_v1",
            "pycam_rad_rstate_destroy_v1"}
    for entry in ENTRIES:
        if entry in skip:
            continue
        body = code[code.index(f"function {entry}("):]
        body = body[:body.index(f"end function {entry}")]
        assert "if (.not. chunk_ok(lchnk)) return" in body, entry
        assert "status = 1_c_int" in body, entry


# -- the calls are the driver's, argument for argument ---------------------------


def test_the_shortwave_core_is_called_the_way_the_driver_calls_it() -> None:
    code = _code()
    body = code[code.index("function pycam_rad_rrtmg_sw_v1("):]
    body = body[:body.index("end function pycam_rad_rrtmg_sw_v1")]
    # the optional cloud optics are passed by keyword, as the driver does
    for keyword in ("E_cld_tau=", "E_cld_tau_w=", "E_cld_tau_w_g=", "E_cld_tau_w_f="):
        assert keyword in body, keyword
    assert "old_convert = .false." in body
    # spectralflux is refused, so the spectral fluxes stay the driver's nulls
    assert "null_su" in body and "null_sd" in body


def test_radheat_fills_the_held_ptend_and_the_held_net_flux() -> None:
    code = _code()
    body = code[code.index("function pycam_rad_radheat_v1("):]
    body = body[:body.index("end function pycam_rad_radheat_v1")]
    assert "rad_ptend(lchnk)" in body
    assert "rad_net_flx(:,lchnk)" in body
    # the surface fluxes come from comsrf, the same storage tphysbc was passed
    for name in ("fsns(:,lchnk)", "fsnt(:,lchnk)", "flns(:,lchnk)", "flnt(:,lchnk)"):
        assert name in body, name


def test_the_chunk_binding_takes_addresses_and_owns_nothing() -> None:
    code = _code()
    body = code[code.index("subroutine pycam_rad_bind_chunk("):]
    body = body[:body.index("end subroutine pycam_rad_bind_chunk")]
    assert "target :: chunk_cam_in" in body and "target :: chunk_cam_out" in body
    assert "host_cam_in(lchnk)%p => chunk_cam_in" in body
    assert "host_cam_out(lchnk)%p => chunk_cam_out" in body
    assert not re.search(r"\ballocate\s*\(", body)     # it stores pointers, not copies
