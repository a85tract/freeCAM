"""The reviewed function specs: loading, validation, and build-time checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from freecam.physics import PhysicsSpecError, load_function_spec
from freecam.physics.spec import default_functions_dir, parse_function_spec
from freecam.physics.verify import (
    VerificationReport,
    declaration_units,
    verify_against_inventory,
    verify_against_source,
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
