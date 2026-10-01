"""The PI-atm CESM cases as the repository makes them, and the comparison with hand-made ones.

The oracle, python-state and pyCESM cases were made by hand.  The recipe holds
what each was made with; these tests hold the recipe to its own references and
the comparison to what it claims: a case's configuration, read without where it
sits or whose it is.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import build_pi_cam_cases as cases  # noqa: E402

VALUES = {"MAPPINGS": "/maps", "OUTPUT_ROOT": "/out", "CASE": "c", "ACCOUNT": "A1", "QUEUE": "develop"}


def _env(directory: Path, **entries: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    body = "".join(f'  <entry id="{key}" value="{value}"/>\n' for key, value in entries.items())
    (directory / "env_case.xml").write_text(f"<file>\n{body}</file>\n")


def test_every_case_names_a_variant_of_the_source_recipe() -> None:
    import prepare_cesm_source

    variants = prepare_cesm_source.load_recipe()["variants"]
    recipe = cases.load_recipe()

    for key, case in recipe["cases"].items():
        assert case["variant"] in variants, key
        assert case["name"].startswith("f.e13.F1850C5.ne16_g16.icesm131_ihesp."), key


def test_a_clone_comes_after_the_case_it_clones() -> None:
    recipe = cases.load_recipe()
    order = list(recipe["cases"])

    for key, case in recipe["cases"].items():
        if case.get("clone"):
            assert order.index(case["clone"]) < order.index(key)
    assert recipe["cases"]["state"]["clone"] == "oracle"


def test_the_recipe_is_the_512_rank_layout_and_the_50_step_run() -> None:
    commands = [" ".join(command) for command in cases.xmlchange_commands(cases.load_recipe(), VALUES)]

    assert "./xmlchange NTASKS_ATM=512,NTHRDS_ATM=1,ROOTPE_ATM=0" in commands
    assert "./xmlchange NTASKS_ICE=128,NTHRDS_ICE=1,ROOTPE_ICE=256" in commands
    assert "./xmlchange STOP_OPTION=nsteps,STOP_N=50" in commands
    assert "./xmlchange CLM_BLDNML_OPTS=-ignore_warnings --append" in commands
    assert "./xmlchange ATM2OCN_FMAPNAME=/maps/map_ne16np4_TO_gx1v6_aave.231103.nc" in commands
    assert "./xmlchange DOUT_S_ROOT=/out/c/archive,PROJECT=A1" in commands


def test_the_recipe_names_no_site() -> None:
    text = cases.RECIPE.read_text()

    assert "/glade" not in text and "PROJECT=${ACCOUNT}" in text


def test_a_value_the_recipe_has_no_value_for_is_refused() -> None:
    with pytest.raises(SystemExit, match="WHERE"):
        cases.fill("X=${WHERE}", VALUES)


def test_the_state_case_leaves_physpkg_to_the_image() -> None:
    # its generated physpkg uses support modules only the image's control source has
    assert cases.load_recipe()["cases"]["state"]["python_state"] is True
    assert cases.STATE_BUILD_EXCLUDES == ("physpkg.F90",)


def test_cases_differing_only_in_where_they_are_and_whose_compare_the_same(tmp_path: Path) -> None:
    made, hand_made = tmp_path / "made/c", tmp_path / "hand/c"
    _env(made, CASEROOT=str(made), SRCROOT="/build/source/cases", CIMEROOT="/build/source/cases/cime",
         CIME_OUTPUT_ROOT="/build/output", ATM_DOMAIN_PATH="/maps", PROJECT="A1", USER="me",
         DOUT_S_ROOT="/build/output/c/archive", MACHDIR="/build/source/cases/cime/config/cesm/machines",
         NTASKS_ATM="512")
    _env(hand_made, CASEROOT=str(hand_made), SRCROOT="/work/tree/cime/..", CIMEROOT="/work/tree/cime",
         CIME_OUTPUT_ROOT="/scratch/x", ATM_DOMAIN_PATH="/other/maps", PROJECT="B2", USER="them",
         DOUT_S_ROOT="/scratch/x/c/archive", MACHDIR="/work/tree/cime/config/cesm/machines",
         NTASKS_ATM="512")
    for directory in (made, hand_made):
        (directory / "user_nl_cam").write_text("nhtfrq = -50\n")

    result = cases.compare_case(made, hand_made)

    assert (result["differ"], result["only_made"], result["only_hand_made"]) == ([], [], [])


def test_a_configured_difference_is_reported(tmp_path: Path) -> None:
    made, hand_made = tmp_path / "made/c", tmp_path / "hand/c"
    _env(made, CASEROOT=str(made), NTASKS_ATM="256")
    _env(hand_made, CASEROOT=str(hand_made), NTASKS_ATM="512", JOB_IDS="1")
    (made / "user_nl_cam").write_text("mfilt = 1\n")
    (hand_made / "user_nl_cam").write_text("mfilt = 2\n")

    result = cases.compare_case(made, hand_made)

    assert result["differ"] == ["env_case.xml:NTASKS_ATM", "user_nl_cam"]
    assert result["values"]["env_case.xml:NTASKS_ATM"] == {"made": "256", "hand_made": "512"}
    assert result["only_hand_made"] == ["env_case.xml:JOB_IDS"]


def test_the_model_version_is_locked_with_the_case(tmp_path: Path, monkeypatch) -> None:
    # case.setup locks env_case.xml; a value written after it fails the next case.build
    # unless the locked copy is the same, as CIME's own lock_file leaves it.
    case, source = tmp_path / "case", tmp_path / "source"
    (case / "LockedFiles").mkdir(parents=True)
    source.mkdir()
    (case / "env_case.xml").write_text('<file><entry id="MODEL_VERSION" value="unknown"/></file>\n')
    (case / "LockedFiles/env_case.xml").write_text((case / "env_case.xml").read_text())
    (source / ".cesm-source.json").write_text('{"model_version": "iCESM1.3.1"}')

    def xmlchange(command: list[str], cwd: Path) -> None:
        name, value = command[1].split("=", 1)
        (cwd / "env_case.xml").write_text(f'<file><entry id="{name}" value="{value}"/></file>\n')

    monkeypatch.setattr(cases, "_run", xmlchange)
    cases.set_model_version(case, source)

    assert cases.env_entries(case)["MODEL_VERSION"] == "iCESM1.3.1"
    assert (case / "LockedFiles/env_case.xml").read_text() == (case / "env_case.xml").read_text()
