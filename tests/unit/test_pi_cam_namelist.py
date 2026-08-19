"""Validated editing of CAM's atm_in namelist."""

from pathlib import Path

import pytest

from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.namelist import (
    apply_overrides,
    coerce_text,
    coerce_value,
    format_fortran_value,
    load_catalog,
    parse_namelist,
    read_values,
    validate_overrides,
)

# Reproduces the real definition file's defects: raw ampersands and angle
# brackets in documentation text, one entry with no closing tag, an id on
# its own line, array types with numeric and symbolic dimensions.
_CATALOG_XML = """<?xml version="1.0"?>
<namelist_definition>
<!-- a comment with <entry id="commented_out" type="real" group="x_nl"> -->
<entry id="cldfrc_rhminl" type="real" category="cldfrc"
       group="cldfrc_nl" valid_values="" >
Minimum rh for low stable clouds & such, wang & sassen.
</entry>
<entry id="zmconv_org" type="logical" category="conv" group="zmconv_nl" valid_values="" >
If < 0 nothing happens.
</entry>
<entry
id="nhtfrq" type="integer(10)" category="history" group="cam_inparm" valid_values="" >
History write frequency.
</entry>
<entry id="rad_climate" type="char*256(n_rad_cnst)" category="rad"
       group="rad_cnst_nl" valid_values="" >
Symbolic array dimension.
</entry>
<entry id="deep_scheme" type="char*16" category="conv" group="phys_ctl_nl"
       valid_values="ZM,off" >
Deep convection scheme.
</entry>
<entry id="broken_tag" type="integer" category="x" group="cam_inparm" valid_values="" >
An entry whose closing tag is missing, like pio_root upstream.
<entry id="after_broken" type="real" category="x" group="cldfrc_nl" valid_values="" >
Still readable.
</entry>
</namelist_definition>
"""

_ATM_IN = """&cam_inparm
 nhtfrq\t\t= -50
 mfilt\t\t= 20
/
&cldfrc_nl
 cldfrc_rhminl\t\t= 0.870D0
/
&rad_cnst_nl
 rad_climate\t\t= 'A:Q:H2O', 'N:O2:O2',
\t\t'mam3_mode2:aitken:=', 'A:num_a2:N:num_c2:num_mr:+',
\t\t'trailing:element'
/
&zmconv_nl
 zmconv_org\t\t= .false.
/
"""


@pytest.fixture()
def catalog(tmp_path: Path):
    xml = (
        tmp_path
        / "source"
        / "components"
        / "cam"
        / "bld"
        / "namelist_files"
        / "namelist_definition.xml"
    )
    xml.parent.mkdir(parents=True)
    xml.write_text(_CATALOG_XML)
    return load_catalog(tmp_path / "source")


@pytest.fixture()
def atm_in(tmp_path: Path) -> Path:
    path = tmp_path / "atm_in"
    path.write_text(_ATM_IN)
    return path


def test_catalog_survives_the_vendor_files_defects(catalog) -> None:
    assert "commented_out" not in catalog
    assert catalog["cldfrc_rhminl"].base_type == "real"
    assert catalog["cldfrc_rhminl"].group == "cldfrc_nl"
    assert catalog["nhtfrq"].array_dims == 10
    assert catalog["rad_climate"].char_length == 256
    assert catalog["rad_climate"].array_dims == "n_rad_cnst"
    assert catalog["deep_scheme"].valid_values == ("ZM", "off")
    # The unclosed entry and the one after it are both recovered.
    assert catalog["broken_tag"].base_type == "integer"
    assert catalog["after_broken"].base_type == "real"


def test_catalog_fails_closed_when_the_definition_is_missing(tmp_path) -> None:
    with pytest.raises(PICAMConfigurationError, match="not found"):
        load_catalog(tmp_path)


def test_real_pinned_catalog_when_present() -> None:
    source = Path(__file__).resolve().parents[2] / "external" / "iCESM1.3.1_fzhu"
    if not (
        source / "components/cam/bld/namelist_files/namelist_definition.xml"
    ).is_file():
        pytest.skip("pinned iCESM submodule not checked out")
    catalog = load_catalog(source)
    assert len(catalog) >= 891
    assert catalog["cldfrc_rhminl"].base_type == "real"
    assert catalog["cldfrc_rhminl"].group == "cldfrc_nl"
    assert catalog["zmconv_c0_lnd"].group == "zmconv_nl"


def test_parser_handles_continuations_containing_equals() -> None:
    assignments = {item.name: item for item in parse_namelist(_ATM_IN)}
    assert assignments["nhtfrq"].raw_value == "-50"
    assert assignments["nhtfrq"].group == "cam_inparm"
    # The ':=' inside a quoted string is a continuation, not an assignment.
    assert "'mam3_mode2:aitken:='" in assignments["rad_climate"].raw_value
    assert "'trailing:element'" in assignments["rad_climate"].raw_value
    assert "mam3_mode2" not in assignments


def test_parser_fails_closed_on_unclassifiable_lines() -> None:
    with pytest.raises(PICAMConfigurationError, match="unrecognised"):
        parse_namelist("&group_nl\n 4 = broken\n/\n")


def test_empty_overrides_leave_bytes_and_mtime_untouched(
    catalog, atm_in
) -> None:
    before = atm_in.read_bytes()
    mtime = atm_in.stat().st_mtime_ns
    assert apply_overrides(atm_in, {}, catalog) == {}
    assert atm_in.read_bytes() == before
    assert atm_in.stat().st_mtime_ns == mtime


def test_override_rewrites_only_the_target_line(catalog, atm_in) -> None:
    before = atm_in.read_text().splitlines()
    report = apply_overrides(atm_in, {"cldfrc_rhminl": 0.9}, catalog)
    after = atm_in.read_text().splitlines()
    assert report == {"cldfrc_rhminl": ("0.870D0", "0.9D0")}
    changed = [
        index
        for index, (old, new) in enumerate(zip(before, after))
        if old != new
    ]
    assert len(changed) == 1
    assert after[changed[0]] == " cldfrc_rhminl\t\t= 0.9D0"
    assert read_values(atm_in)["cldfrc_rhminl"] == "0.9D0"


def test_multi_line_value_is_replaced_by_a_single_line(
    catalog, atm_in
) -> None:
    report = apply_overrides(
        atm_in, {"rad_climate": ["A:Q:H2O", "N:O2:O2"]}, catalog
    )
    assert report["rad_climate"][1] == "'A:Q:H2O', 'N:O2:O2'"
    values = read_values(atm_in)
    assert values["rad_climate"] == "'A:Q:H2O', 'N:O2:O2'"
    # The other groups are untouched.
    assert values["zmconv_org"] == ".false."


def test_new_variable_joins_its_group_before_the_terminator(
    catalog, atm_in
) -> None:
    report = apply_overrides(atm_in, {"zmconv_org": True}, catalog)
    assert report["zmconv_org"] == (".false.", ".true.")
    report = apply_overrides(atm_in, {"after_broken": 1.5}, catalog)
    assert report["after_broken"] == (None, "1.5D0")
    lines = atm_in.read_text().splitlines()
    position = lines.index(" after_broken\t\t= 1.5D0")
    assert lines[position - 1] == " cldfrc_rhminl\t\t= 0.870D0"
    assert lines[position + 1].strip() == "/"


def test_variable_of_an_absent_group_is_refused(catalog, atm_in) -> None:
    with pytest.raises(PICAMConfigurationError, match="ignores unknown"):
        apply_overrides(atm_in, {"deep_scheme": "off"}, catalog)


def test_unknown_variable_is_refused_with_suggestions(
    catalog, atm_in
) -> None:
    with pytest.raises(PICAMConfigurationError, match="cldfrc_rhminl"):
        apply_overrides(atm_in, {"cldfrc_rhminls": 0.9}, catalog)


def test_duplicate_assignment_is_refused(catalog, tmp_path) -> None:
    path = tmp_path / "atm_in"
    path.write_text(
        "&cldfrc_nl\n cldfrc_rhminl = 0.9D0\n/\n"
        "&cldfrc_nl\n cldfrc_rhminl = 0.8D0\n/\n"
    )
    with pytest.raises(PICAMConfigurationError, match="2 times"):
        apply_overrides(path, {"cldfrc_rhminl": 0.85}, catalog)


def test_type_coercion_fails_closed(catalog) -> None:
    real = catalog["cldfrc_rhminl"]
    logical = catalog["zmconv_org"]
    integer_array = catalog["nhtfrq"]
    with pytest.raises(PICAMConfigurationError, match="boolean"):
        coerce_value(True, real)
    with pytest.raises(PICAMConfigurationError, match="logical"):
        coerce_value(1, logical)
    with pytest.raises(PICAMConfigurationError, match="integer"):
        coerce_value(3.5, integer_array)
    with pytest.raises(PICAMConfigurationError, match="at most 10"):
        coerce_value(list(range(11)), integer_array)
    with pytest.raises(PICAMConfigurationError, match="must be one of"):
        coerce_value("UW", catalog["deep_scheme"])
    assert coerce_value("ZM", catalog["deep_scheme"]) == "ZM"
    assert coerce_value(1, real) == 1.0
    assert coerce_value([0, -24], integer_array) == (0, -24)


def test_command_line_text_coercion(catalog) -> None:
    assert coerce_text("0.9", catalog["cldfrc_rhminl"]) == 0.9
    assert coerce_text("1.0D-6", catalog["cldfrc_rhminl"]) == 1.0e-6
    assert coerce_text(".true.", catalog["zmconv_org"]) is True
    assert coerce_text("false", catalog["zmconv_org"]) is False
    assert coerce_text("0,-24", catalog["nhtfrq"]) == (0, -24)
    with pytest.raises(PICAMConfigurationError, match="logical"):
        coerce_text("yes", catalog["zmconv_org"])


def test_fortran_formatting(catalog) -> None:
    real = catalog["cldfrc_rhminl"]
    assert format_fortran_value(0.9, real) == "0.9D0"
    assert format_fortran_value(1.0e-6, real) == "1D-06"
    assert format_fortran_value(True, catalog["zmconv_org"]) == ".true."
    assert (
        format_fortran_value("it's", catalog["deep_scheme"]) == "'it''s'"
    )
    assert format_fortran_value((0, -24), catalog["nhtfrq"]) == "0, -24"


def test_validate_overrides_normalizes_names(catalog) -> None:
    validated = validate_overrides({"CLDFRC_RHMINL": 0.9}, catalog)
    assert validated == {"cldfrc_rhminl": 0.9}
