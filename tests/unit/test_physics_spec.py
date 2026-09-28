"""The reviewed function specs: loading, validation, and build-time checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from freecam.physics import PhysicsSpecError, load_function_spec
from freecam.physics.spec import default_functions_dir, parse_function_spec
from freecam.physics.column import InvalidInput, coerce_inputs
from freecam.physics.verify import (
    VerificationReport,
    declaration_units,
    verify_against_inventory,
    verify_against_source,
    verify_overwritten_on_entry,
)

PROJECT = Path(__file__).resolve().parents[2]


def _document(name: str) -> dict:
    return yaml.safe_load((default_functions_dir() / f"{name}.yaml").read_text())


def _inventory_record(qualified_name: str) -> dict:
    inventory = json.loads((PROJECT / "validation/pi_cam_kernel_inventory.json").read_text())
    return next(
        item for item in inventory["procedures"] if item["qualified_name"] == qualified_name
    )


def test_shipped_specs_load_and_partition_every_argument() -> None:
    dadadj = load_function_spec("dadadj")
    assert [item.name for item in dadadj.arguments] == [
        "lchnk", "ncol", "pmid", "pint", "pdel", "t", "q",
    ]
    assert [item.name for item in dadadj.inouts] == ["t", "q"]
    assert dadadj.outputs == ()
    assert list(dadadj.parameters) == ["nlvdry"]
    assert dadadj.parameters["nlvdry"].range == (1, 29)

    macro = load_function_spec("mmacro_pcond")
    roles = {role: len(getattr(macro, role)) for role in ("inputs", "inouts", "outputs", "workspace", "structural")}
    assert roles == {"inputs": 29, "inouts": 6, "outputs": 17, "workspace": 6, "structural": 2}
    assert sum(roles.values()) == 60 == len(macro.arguments)
    # Every argument the user sees has a public shape without the column axis.
    for item in macro.user_arguments:
        assert item.public_shape is not None
        assert "pcols" not in item.public_shape
    assert macro.argument("T0").public_extent(macro.dimensions) == (30,)
    assert macro.argument("tke").native_extent(macro.dimensions) == (16, 31)
    assert macro.argument("nl0").units == "#/kg"
    assert macro.argument("do_cldice").carrier == "logical"
    assert macro.initializers == ("wv_saturation_mp_wv_sat_init_",)


def test_describe_reads_like_a_signature() -> None:
    text = load_function_spec("mmacro_pcond").describe()
    assert "cldwat2m_macro::mmacro_pcond" in text
    assert "t0             [lev]         K" in text
    assert "cldfrc_premib" in text and "default=70000.0" in text


def test_parameters_and_module_state_agree() -> None:
    spec = load_function_spec("mmacro_pcond")
    owned = {symbol for item in spec.parameters.values() for symbol in item.symbols}
    declared = {item.symbol for item in spec.module_state if item.write == "parameter"}
    assert owned == declared
    # The premit/premib ramp is read from the cldfrc2m copy on every call.
    assert "cldfrc2m_mp_premib_" in spec.parameters["cldfrc_premib"].symbols
    assert spec.image.stub_symbols & {"cam_history_mp_outfld_", "shr_sys_mod_mp_shr_sys_abort_"}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["arguments"].append(dict(d["arguments"][2])), "names repeat"),
        (lambda d: d["arguments"][2].__setitem__("role", "tunable"), "unsupported role"),
        (lambda d: d["arguments"][2].__setitem__("native_shape", ["pcols", "nlev"]), "unknown dimension"),
        (lambda d: d["arguments"][2].__setitem__("public_shape", ["pcols", "pver"]), "without the column axis"),
        (lambda d: d["arguments"][0].pop("value"), "needs a value"),
        (lambda d: d["arguments"][5].__setitem__("intent", "in"), "does not admit intent"),
        (lambda d: d["parameters"]["nlvdry"].__setitem__("symbols", ["elsewhere_"]), "write: parameter"),
        (lambda d: d["image"]["stubs"]["inert"].append("mpibcast_"), "more than one class"),
        (lambda d: d["module_state"][0].__setitem__("value", 1.0), "not declared"),
    ],
)
def test_spec_validation_fails_closed(mutate, message: str) -> None:
    document = copy.deepcopy(_document("dadadj"))
    mutate(document)
    with pytest.raises(PhysicsSpecError, match=message):
        parse_function_spec(document)


def test_spec_name_must_match_its_file(tmp_path: Path) -> None:
    document = _document("dadadj")
    path = tmp_path / "renamed.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(PhysicsSpecError, match="expected 'renamed'"):
        load_function_spec(path)


def test_shipped_specs_verify_against_inventory_and_source() -> None:
    for name in ("dadadj", "mmacro_pcond"):
        spec = load_function_spec(name)
        record = _inventory_record(spec.qualified_name)
        report = verify_against_inventory(spec, record)
        source = (PROJECT / "external/iCESM1.3.1_fzhu" / record["source"]).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        verify_against_source(
            spec, source, line_start=record["line_start"], line_end=record["line_end"], report=report
        )
        assert report.passed, report.failures


def test_inventory_drift_is_reported() -> None:
    spec = load_function_spec("dadadj")
    record = copy.deepcopy(_inventory_record("dadadj::dadadj"))
    record["arguments"][5], record["arguments"][6] = record["arguments"][6], record["arguments"][5]
    report = verify_against_inventory(spec, record)
    assert not report.passed
    assert any("inventory has 'q' at this position" in text for text in report.failures)

    record = copy.deepcopy(_inventory_record("dadadj::dadadj"))
    record["arguments"][2]["dimensions"] = ["pcols", "pverp"]
    report = verify_against_inventory(spec, record)
    assert any("native_shape" in text for text in report.failures)


def test_units_are_read_from_declaration_comments_including_continuations() -> None:
    lines = [
        "   real(r8), intent(in)    :: T0(pcols,pver)    ! Temperature [K]",
        "   real(r8), intent(in)    :: C_qlst(pcols,pver) ! Forcing of ql",
        "                                                 ! within liquid stratus [kg/kg/s]",
        "   real(r8), intent(out)   :: qme  (pcols,pver)  ! Net condensation rate [kg/kg/s]",
        "   real(r8), pointer       :: tke(:,:)           ! (pcols,pverp) TKE",
        "   integer, intent(in) :: lchnk               ! chunk identifier",
    ]
    found = declaration_units(lines, ["T0", "C_qlst", "qme", "tke", "lchnk"])
    assert found == {"t0": "K", "c_qlst": "kg/kg/s", "qme": "kg/kg/s", "tke": None, "lchnk": None}


def test_unit_conflicts_with_the_source_fail() -> None:
    spec = load_function_spec("dadadj")
    lines = [
        "   integer, intent(in) :: lchnk",
        "   integer, intent(in) :: ncol",
        "   real(r8), intent(in) :: pmid(pcols,pver)   ! pressure [hPa]",
        "   real(r8), intent(in) :: pint(pcols,pverp)",
        "   real(r8), intent(in) :: pdel(pcols,pver)",
        "   real(r8), intent(inout) :: t(pcols,pver)",
        "   real(r8), intent(inout) :: q(pcols,pver)",
    ]
    report = verify_against_source(spec, lines, line_start=1, line_end=len(lines))
    assert report.failures == ["argument pmid: source declares [hPa], spec says 'Pa'"]


def test_archive_members_may_name_a_second_archive() -> None:
    """uwshcu needs shr_spfn_erfc, which libatm does not hold."""

    spec = load_function_spec("uwshcu")
    assert spec.image.archives == ("atm", "csm_share")
    from_share = [item.member for item in spec.image.archive_members if item.archive == "csm_share"]
    assert from_share == ["shr_spfn_mod.o", "water_isotopes.o", "water_types.o"]
    assert "uwshcu.o" in spec.image.member_names
    assert str(spec.image.archive_members[0]) == "uwshcu.o"
    assert str(next(i for i in spec.image.archive_members if i.archive == "csm_share")) == "csm_share:shr_spfn_mod.o"


def test_archive_members_fail_closed_on_malformed_entries() -> None:
    from freecam.physics.spec import _archive_members

    assert [item.archive for item in _archive_members(["a.o"])] == ["atm"]
    for bad, message in (
        ([], "non-empty"),
        (["a.o", "a.o"], "listed twice"),
        (["a.c"], "not an object file"),
        ([{"archive": "csm_share"}], "must be a name"),
        ([{"member": "a.o", "surprise": 1}], "unsupported keys"),
    ):
        with pytest.raises(PhysicsSpecError, match=message):
            _archive_members(bad)


def test_dimension_aliases_let_a_routine_name_its_own_extents() -> None:
    """uwshcu's dummies say mix/mkx where the spec says pcols/pver."""

    spec = load_function_spec("uwshcu")
    assert spec.dimension_aliases == {"mix": "pcols", "mkx": "pver", "mkx + 1": "pverp", "ncnst": "pcnst"}
    assert spec.argument("p0_inv").native_shape == ("pcols", "pver")
    assert spec.argument("tr0_inv").native_shape == ("pcols", "pver", "pcnst")
    # mmacro_pcond needs none: its dummies are already declared in pcols/pver.
    assert load_function_spec("mmacro_pcond").dimension_aliases == {}


def test_dimension_aliases_must_name_a_declared_dimension() -> None:
    document = yaml.safe_load(Path(load_function_spec("uwshcu").path).read_text())
    document["dimension_aliases"]["mix"] = "pnotadimension"
    with pytest.raises(PhysicsSpecError, match="unknown dimension"):
        parse_function_spec(document)


def test_uwshcu_boundary_is_single_column_and_complete() -> None:
    spec = load_function_spec("uwshcu")
    assert len(spec.arguments) == 54
    assert [item.name for item in spec.structural] == ["mix", "mkx", "iend", "ncnst", "lchnk"]
    assert spec.argument("iend").value == 1 and spec.argument("mix").value == 16
    assert [item.name for item in spec.inouts] == ["cush"]
    # The tracer axis is public in full, as reviewed.
    assert spec.argument("tr0_inv").public_shape == ("pver", "pcnst")
    assert spec.argument("trten_inv").public_shape == ("pver", "pcnst")
    assert set(spec.parameters) == {"uwshcu_rpen"}
    # Water isotope tracing is live in this configuration, so it is pinned: the model's
    # logicals read -1 (ifort's .true. in this build), and the constituent index table the
    # kernel walks is snapshotted with them -- without it the standalone image answers wrongly
    pinned = {entry.symbol: entry for entry in spec.module_state}
    assert pinned["water_tracer_vars_mp_trace_water_"].expected == -1
    assert pinned["water_tracer_vars_mp_wisotope_"].expected == -1
    assert pinned["water_tracer_vars_mp_wtrc_iatype_"].shape == (700, 7) and pinned["water_tracer_vars_mp_wtrc_iatype_"].write == "snapshot"
    assert pinned["constituents_mp_cnst_type_"].dtype == "S3"
    assert pinned["uwshcu_mp_rpen_"].write == "parameter"
    assert pinned["uwshcu_mp_xlv_"].write == "snapshot"


def test_overwritten_on_entry_is_for_an_inout_at_a_source_line() -> None:
    document = copy.deepcopy(_document("micro_mg_tend"))
    reff = next(item for item in document["arguments"] if item["name"] == "reff_rain")
    assert parse_function_spec(document).argument("reff_rain").overwritten_on_entry == 995
    reff["overwritten_on_entry"] = "995"
    with pytest.raises(PhysicsSpecError, match="source line number"):
        parse_function_spec(document)
    reff["overwritten_on_entry"] = 995
    tn = next(item for item in document["arguments"] if item["name"] == "tn")
    tn["overwritten_on_entry"] = 900
    with pytest.raises(PhysicsSpecError, match="only an inout dummy"):
        parse_function_spec(document)


def test_a_dummy_overwritten_on_entry_may_carry_a_non_finite_value_in() -> None:
    spec = load_function_spec("micro_mg_tend")
    inputs = {item.name: np.zeros(item.public_extent(spec.dimensions)) for item in spec.user_arguments}
    inputs["reff_rain"] = np.full_like(inputs["reff_rain"], np.nan)
    assert np.isnan(coerce_inputs(spec, inputs)["reff_rain"]).all()
    inputs["qc"] = np.full_like(inputs["qc"], np.nan)
    with pytest.raises(InvalidInput, match="qc contains non-finite"):
        coerce_inputs(spec, inputs)


def test_micro_mg_tend_zeroes_reff_rain_and_reff_snow_before_reading_them() -> None:
    spec = load_function_spec("micro_mg_tend")
    record = _inventory_record(spec.qualified_name)
    source = (PROJECT / "external/iCESM1.3.1_fzhu" / record["source"]).read_text(
        encoding="utf-8", errors="replace").splitlines()
    report = verify_overwritten_on_entry(spec, source, line_start=record["line_start"], line_end=record["line_end"])
    assert report.passed, report.failures
    assert len(report.checks) == 2


_TOY_ROUTINE = """subroutine toy(ncol, a, &
     b)
  integer, intent(in) :: ncol
  real(r8), intent(inout) :: a(pcols,pver) ! a (m)
  real(r8), intent(in) :: b(pcols)
  {before}
  a(1:ncol,1:pver) = 0._r8   ! the claim
  a(1,1) = b(1)
{contains}end subroutine toy"""


@pytest.mark.parametrize("before, contains, line, message", [
    ("", "", 7, None),
    ("b(1) = a(1,1)", "", 7, "uses it before the assignment"),
    ("", "contains\n  subroutine inner\n  a = 1\n  end subroutine inner\n", 7, "internal procedure"),
    ("", "", 8, "a part of it"),
    ("", "", 5, "not an assignment"),
])
def test_the_overwrite_claim_is_checked_against_the_routine(before, contains, line, message) -> None:
    from freecam.physics.spec import parse_function_spec as parse
    document = {
        "schema_version": 1, "function": "toy", "qualified_name": "toymod::toy", "routine": "toy", "source": "toy.F90",
        "module": "toymod", "dimensions": {"pcols": 4, "pver": 3},
        "arguments": [
            {"name": "ncol", "role": "structural", "fortran_type": "integer", "dtype": "int32", "rank": 0,
             "intent": "in", "native_shape": [], "value": 1},
            {"name": "a", "role": "inout", "fortran_type": "real", "dtype": "float64", "rank": 2, "intent": "inout",
             "native_shape": ["pcols", "pver"], "public_shape": ["pver"], "overwritten_on_entry": line},
            {"name": "b", "role": "input", "fortran_type": "real", "dtype": "float64", "rank": 1, "intent": "in",
             "native_shape": ["pcols"], "public_shape": []},
        ],
        "image": {"archive_members": ["toy.o"], "stubs": {}, "base_address": 0x50000000},
    }
    source = _TOY_ROUTINE.format(before=before or "! nothing", contains=contains).splitlines()
    report = verify_overwritten_on_entry(parse(document), source, line_start=1, line_end=len(source))
    if message is None:
        assert report.passed, report.failures
    else:
        assert not report.passed and message in report.failures[0]


def test_gathered_columns_name_their_count_and_arrays() -> None:
    spec = load_function_spec("zm_convr")
    assert spec.columns == "gathered" and spec.gather_count == "lengath" and "mu" in spec.gathered
    assert load_function_spec("zm_conv_evap").columns == "independent"
    document = copy.deepcopy(_document("zm_convr"))
    document.pop("gathered")
    with pytest.raises(PhysicsSpecError, match="go together"):
        parse_function_spec(document)
    document = copy.deepcopy(_document("zm_convr"))
    document["gathered"]["count"] = "t"
    with pytest.raises(PhysicsSpecError, match="integer scalar"):
        parse_function_spec(document)
    document = copy.deepcopy(_document("zm_convr"))
    document["gathered"]["arguments"].append("delt")
    with pytest.raises(PhysicsSpecError, match="column axis"):
        parse_function_spec(document)


def test_state_set_by_an_initializer_needs_an_initializer() -> None:
    document = copy.deepcopy(_document("instratus_condensate"))
    assert parse_function_spec(document).initializers == ("wv_saturation_mp_wv_sat_init_",)
    document["initializers"] = []
    with pytest.raises(PhysicsSpecError, match="no initializers are named"):
        parse_function_spec(document)
