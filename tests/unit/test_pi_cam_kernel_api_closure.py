"""The kernel-API closure record: configuration values, guard evaluation, rules and checks."""
from pathlib import Path
import shutil

import pytest
import yaml

from freecam.pi_cam.call_tree import GuardPart
from freecam.pi_cam.errors import PICAMConfigurationError
from freecam.pi_cam.kernel_api_closure import (
    CATEGORIES,
    ConditionEvaluator,
    _looks_static,
    config_cache_values,
    fortran_literal,
    guard_state,
    load_closure_rules,
    namelist_values,
)


def test_namelist_values_parse_scalars_of_every_kind() -> None:
    text = " deep_scheme\t\t= 'ZM'\n do_tms = .true.\n micro_mg_version =  1\n cldfrc_dp1 = 0.10D0\n other = 'x'\n"
    values = namelist_values(text, ["deep_scheme", "do_tms", "micro_mg_version", "cldfrc_dp1", "absent"])
    assert values == {"deep_scheme": "ZM", "do_tms": True, "micro_mg_version": 1, "cldfrc_dp1": 0.1}


def test_config_cache_values_select_named_options() -> None:
    text = '<entry id="phys" value="cam5"/>\n<entry id="chem" value="trop_mam3"/>\n<entry id="dyn" value="se"/>'
    assert config_cache_values(text, ["phys", "chem"]) == {"phys": "cam5", "chem": "trop_mam3"}


def test_fortran_literal_keeps_unknown_text() -> None:
    assert fortran_literal("'a''b'") == "a'b"
    assert fortran_literal("1.5_r8") == 1.5
    assert fortran_literal(".FALSE.") is False
    assert fortran_literal("some_name") == "some_name"


@pytest.fixture
def evaluator() -> ConditionEvaluator:
    return ConditionEvaluator(
        {"deep_scheme": "ZM", "do_tms": True, "micro_mg_version": 1, "carma_flag": False},
        {"cam_physpkg_is": lambda args: args[0] == "cam5"},
    )


def test_evaluator_decides_string_logical_and_numeric_conditions(evaluator: ConditionEvaluator) -> None:
    assert evaluator.truth("deep_scheme == 'ZM'") is True
    assert evaluator.truth("deep_scheme /= 'ZM' .or. do_tms") is True
    assert evaluator.truth("trim(deep_scheme) .eq. 'UNICON'") is False
    assert evaluator.truth("micro_mg_version == 1 .and. .not. carma_flag") is True
    assert evaluator.truth("cam_physpkg_is('cam5')") is True
    assert evaluator.truth("cam_physpkg_is('cam4')") is False


def test_evaluator_never_guesses_unknown_names(evaluator: ConditionEvaluator) -> None:
    assert evaluator.truth("unknown_flag") is None
    assert evaluator.truth("deep_scheme == 'ZM' .and. unknown_flag") is None
    assert evaluator.truth("deep_scheme == 'UNICON' .and. unknown_flag") is False
    assert evaluator.truth("do_tms .or. unknown_flag") is True
    assert evaluator.truth("present(x)") is None
    assert evaluator.truth("state%ncol > 0") is None
    assert evaluator.truth("this is not fortran ((") is None


def test_guard_state_combines_if_and_case_parts(evaluator: ConditionEvaluator) -> None:
    assert guard_state((), evaluator) is None
    zm = GuardPart(kind="if", text="deep_scheme == 'ZM'")
    assert guard_state((zm,), evaluator) == "enabled"
    assert guard_state((GuardPart(kind="if", text="deep_scheme == 'ZM'", negate=True),), evaluator) == "disabled"
    case = GuardPart(kind="case", text="trim(deep_scheme)", values=("'UNICON'", "'HK'"))
    assert guard_state((case,), evaluator) == "disabled"
    default = GuardPart(kind="case", text="trim(deep_scheme)", values=("'UNICON'",), negate=True)
    assert guard_state((default,), evaluator) == "enabled"
    assert guard_state((zm, GuardPart(kind="if", text="dosw")), evaluator) == "undecided"
    assert guard_state((GuardPart(kind="case", text="n", values=("1:3",)),), evaluator) == "undecided"


def test_static_looking_guards_are_names_and_literals_only() -> None:
    assert _looks_static("dosw .OR. dolw")
    assert _looks_static("micro_mg_version == 1")
    assert not _looks_static("present(x)")
    assert not _looks_static("state % ncol > 0")


def _rules(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema_version": 1,
        "source_directories": ["components/cam/src/physics/cam"],
        "templates": [],
        "macros": ["CAM"],
        "step_roots": ["physpkg::phys_run1"],
        "init_roots": [],
        "selectors": ["deep_scheme"],
        "build_options": ["phys"],
        "option_functions": {"cam_physpkg_is": "phys"},
        "constants": {},
        "library_names": ["^mpi_"],
        "categories": [
            {"id": "services", "category": "internal_service", "match": {"module_regex": "^cam_history$"}},
        ],
        "default_category": "numeric_kernel",
    }
    payload.update(overrides)
    path = tmp_path / "rules.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_rules_load_and_reject_unknown_keys(tmp_path: Path) -> None:
    rules = load_closure_rules(_rules(tmp_path))
    assert rules.categories[0].category == "internal_service"
    assert rules.default_category == "numeric_kernel" and rules.default_category in CATEGORIES
    with pytest.raises(PICAMConfigurationError):
        load_closure_rules(_rules(tmp_path, surprise=1))
    with pytest.raises(PICAMConfigurationError):
        load_closure_rules(_rules(tmp_path, categories=[{"id": "x", "category": "made_up", "match": {}}]))
    with pytest.raises(PICAMConfigurationError):
        load_closure_rules(_rules(tmp_path, categories=[{"id": "x", "category": "numeric_kernel", "match": {"colour": "red"}}]))


REPO = Path(__file__).resolve().parents[2]


def test_committed_rules_load_and_name_the_step_drivers() -> None:
    rules = load_closure_rules(REPO / "native/pi_cam/kernel_api_closure_rules.yaml")
    assert set(rules.step_roots) == {"physpkg::phys_run1", "physpkg::phys_run2"}
    assert "MODAL_AERO_3MODE" in rules.macros and "SPMD" in rules.macros
    assert "deep_scheme" in rules.selectors and "phys" in rules.build_options
    assert all(rule.category in CATEGORIES for rule in rules.categories)


@pytest.mark.skipif(shutil.which("cpp") is None, reason="the C preprocessor is not on PATH")
def test_build_closure_on_a_toy_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from freecam.pi_cam import kernel_api_closure as closure
    from freecam.pi_cam.call_tree import CallTree, fortran_files

    src = tmp_path / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam"
    src.mkdir(parents=True)
    (src / "physpkg.F90").write_text(
        "module physpkg\ncontains\n"
        "subroutine phys_run1()\n  use deep, only: deep_tend\n  use cam_history, only: outfld\n"
        "  character(len=8) :: deep_scheme\n  deep_scheme = 'ZM'\n"
        "  if (deep_scheme == 'ZM') call deep_tend()\n  if (deep_scheme == 'HK') call hk_tend()\n  call outfld()\n"
        "end subroutine phys_run1\nend module physpkg\n"
    )
    (src / "deep.F90").write_text(
        "module deep\ncontains\nsubroutine deep_tend()\n  real :: x\n  x = helper(1.0)\nend subroutine deep_tend\n"
        "real function helper(a)\n  real :: a\n  helper = a\nend function helper\nend module deep\n"
    )
    (src / "hk.F90").write_text("subroutine hk_tend()\nend subroutine hk_tend\n")
    (src / "cam_history.F90").write_text("module cam_history\ncontains\nsubroutine outfld()\nend subroutine outfld\nend module cam_history\n")
    rules_path = _rules(tmp_path)
    rules = load_closure_rules(rules_path)
    tree = CallTree.scan(fortran_files([src]), source_root=tmp_path / "external/iCESM1.3.1_fzhu", macros=["CAM"], workers=1)
    inputs = closure.ClosureInputs(
        project_root=tmp_path,
        rules_path=Path("rules.yaml"),
        rules=rules,
        previous_record={"configuration": {"selectors": {"deep_scheme": "ZM"}, "build_options": {"phys": "cam5"}}},
    )
    record = closure.build_closure(inputs, tree=tree)
    by_name = {item["qualified"]: item for item in record["procedures"]}
    assert by_name["deep::deep_tend"]["in_configuration"] and by_name["deep::helper"]["in_configuration"]
    assert not by_name["hk::hk_tend"]["in_configuration"]
    assert by_name["cam_history::outfld"]["category"] == "internal_service"
    assert by_name["physpkg::phys_run1"]["category"] == "numeric_kernel"  # no physpkg rule in the toy rules
    helper_site = next(site for site in by_name["deep::deep_tend"]["sites"] if site["name"] == "helper")
    assert helper_site["kind"] == "function" and helper_site["in_expression"]
    assert record["summary"]["procedures_config_disabled"] == 1
    assert record["configuration"]["sources"]["selectors"] == "previous record"
    assert closure.check_closure(record, inputs) == []
    rules_path.write_text(rules_path.read_text() + "\n# changed\n")
    inputs.rules = load_closure_rules(rules_path)
    assert closure.check_closure(record, inputs) == ["the closure rules changed"]


def test_committed_record_still_matches_its_inputs() -> None:
    """The committed closure record was generated from the pinned source, rules and generator in this tree."""
    import json

    from freecam.pi_cam import kernel_api_closure as closure

    record_path = REPO / closure.DEFAULT_RECORD
    if not record_path.is_file():
        pytest.skip("no committed closure record")
    if closure.source_revision(REPO / closure.SOURCE_ROOT) is None:
        pytest.skip("pinned source not checked out")
    record = json.loads(record_path.read_text())
    inputs = closure.ClosureInputs(
        project_root=REPO,
        rules_path=closure.DEFAULT_RULES,
        rules=load_closure_rules(REPO / closure.DEFAULT_RULES),
        previous_record=record,
    )
    assert closure.check_closure(record, inputs) == []
    summary = record["summary"]
    assert summary["unresolved_references"] == 0
    assert summary["parse_failures"] == 0
    assert record["roots"]["missing"] == []
