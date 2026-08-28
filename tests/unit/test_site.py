"""The site description: one file, read the same way by Python and bash."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from freecam import site


def _checkout(tmp_path: Path, contents: str | None = None) -> Path:
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'freecam'\n")
    if contents is not None:
        (repo / "site.env").write_text(contents)
    return repo


def test_repository_root_walks_up_to_the_project_file(tmp_path) -> None:
    repo = _checkout(tmp_path)
    deep = repo / "src" / "freecam" / "pi_cam"
    deep.mkdir(parents=True)

    assert site.repository_root(deep) == repo


def test_repository_root_says_so_when_there_is_no_checkout(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="no freeCAM checkout"):
        site.repository_root(tmp_path)


def test_an_absent_site_file_is_an_empty_site_not_an_error(tmp_path) -> None:
    repo = _checkout(tmp_path)

    assert site.load(repo) == {}
    assert site.setting("FREECAM_ACCOUNT", repo=repo) is None
    assert site.origin("FREECAM_ACCOUNT", repo=repo) == "unset"


def test_assignments_comments_quotes_and_export_all_parse(tmp_path) -> None:
    repo = _checkout(
        tmp_path,
        "# a comment\n"
        "\n"
        "FREECAM_ACCOUNT=PROJECT01\n"
        "export FREECAM_QUEUE=develop\n"
        'FREECAM_CASES="/a path/CESM_cases"\n'
        "FREECAM_SCRATCH=/scratch  # trailing note\n",
    )

    values = site.load(repo)

    assert values == {
        "FREECAM_ACCOUNT": "PROJECT01",
        "FREECAM_QUEUE": "develop",
        "FREECAM_CASES": "/a path/CESM_cases",
        "FREECAM_SCRATCH": "/scratch",
    }


def test_references_expand_from_the_environment_and_from_earlier_lines(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("SCRATCH", "/scratch/someone")
    monkeypatch.delenv("FREECAM_CASES", raising=False)
    repo = _checkout(
        tmp_path,
        "FREECAM_SCRATCH=${SCRATCH}/work\n"
        "FREECAM_CASES=$FREECAM_SCRATCH/CESM_cases\n"
        "FREECAM_QUEUE=${NOT_SET_ANYWHERE:-develop}\n",
    )

    values = site.load(repo)

    assert values["FREECAM_SCRATCH"] == "/scratch/someone/work"
    assert values["FREECAM_CASES"] == "/scratch/someone/work/CESM_cases"
    assert values["FREECAM_QUEUE"] == "develop"


def test_a_single_quoted_value_is_literal(tmp_path) -> None:
    repo = _checkout(tmp_path, "FREECAM_ACCOUNT='$NOT_EXPANDED'\n")

    assert site.load(repo)["FREECAM_ACCOUNT"] == "$NOT_EXPANDED"


def test_an_unresolvable_reference_names_the_variable(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("NOWHERE", raising=False)
    repo = _checkout(tmp_path, "FREECAM_SCRATCH=$NOWHERE/work\n")

    with pytest.raises(ValueError, match=r"\$NOWHERE"):
        site.load(repo)


def test_a_line_that_is_not_an_assignment_is_refused(tmp_path) -> None:
    # bash would act on it; a reader that silently skipped it would give the
    # job and the notebook different answers.
    repo = _checkout(tmp_path, "if [ -d /tmp ]; then echo hi; fi\n")

    with pytest.raises(ValueError, match="is not KEY=value"):
        site.load(repo)


def test_the_environment_wins_over_the_file(tmp_path, monkeypatch) -> None:
    repo = _checkout(tmp_path, "FREECAM_ACCOUNT=FROM_FILE\n")
    monkeypatch.setenv("FREECAM_ACCOUNT", "FROM_ENVIRONMENT")

    assert site.setting("FREECAM_ACCOUNT", repo=repo) == "FROM_ENVIRONMENT"
    assert site.origin("FREECAM_ACCOUNT", repo=repo) == "environment"

    monkeypatch.delenv("FREECAM_ACCOUNT")

    assert site.setting("FREECAM_ACCOUNT", repo=repo) == "FROM_FILE"
    assert site.origin("FREECAM_ACCOUNT", repo=repo) == str(site.site_file(repo))


def test_an_edited_file_is_re_read(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)
    repo = _checkout(tmp_path, "FREECAM_ACCOUNT=FIRST\n")
    assert site.setting("FREECAM_ACCOUNT", repo=repo) == "FIRST"

    file = site.site_file(repo)
    file.write_text("FREECAM_ACCOUNT=SECOND\n")
    import os

    stamp = file.stat().st_mtime + 1
    os.utime(file, (stamp, stamp))

    assert site.setting("FREECAM_ACCOUNT", repo=repo) == "SECOND"


def test_missing_names_the_required_settings_that_have_no_value(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)
    repo = _checkout(tmp_path)

    assert [entry.name for entry in site.missing(repo=repo)] == ["FREECAM_ACCOUNT"]
    assert all(entry.produced_by for entry in site.missing(repo=repo))

    (repo / "site.env").write_text("FREECAM_ACCOUNT=PROJECT01\n")

    assert site.missing(repo=repo) == ()


def test_report_covers_every_declared_setting(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)
    repo = _checkout(tmp_path, "FREECAM_ACCOUNT=PROJECT01\n")

    rows = site.report(repo=repo)

    assert [row[0] for row in rows] == [entry.name for entry in site.SETTINGS]
    account = next(row for row in rows if row[0] == "FREECAM_ACCOUNT")
    assert account[1] == "PROJECT01"
    assert account[2] == str(site.site_file(repo))
    assert account[3]


def test_bash_and_python_read_the_committed_example_the_same_way(tmp_path) -> None:
    # The example is the file a new user copies.  If the two readers disagree
    # about it, the notebook and the PBS job disagree about the site.
    example = site.repository_root() / "site.env.example"
    repo = _checkout(tmp_path, example.read_text())

    from_python = dict(site.load(repo))

    script = (
        f'set -eu\n. "{site.site_file(repo)}"\n'
        "for name in " + " ".join(site.SETTINGS[index].name for index in range(len(site.SETTINGS)))
        + '; do eval "value=\\${$name:-}"; echo "$name=$value"; done\n'
    )
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )
    from_bash = {}
    for line in result.stdout.splitlines():
        name, _, value = line.partition("=")
        if value:
            from_bash[name] = value

    assert from_bash == {
        name: value for name, value in from_python.items() if value
    }


def test_preflight_names_every_prerequisite_a_clone_lacks(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)
    monkeypatch.delenv("FREECAM_REFERENCE_CASE", raising=False)
    monkeypatch.delenv("FREECAM_REFERENCE_RUN", raising=False)
    repo = _checkout(tmp_path)
    config = repo / "configs" / "pi_cam_icesm131.yaml"
    config.parent.mkdir()
    config.write_text(
        "case_name: a-case\nnative_manifest: build/pi_cam_promoted/manifest.json\n"
    )

    checks = {check.name: check for check in site.preflight(repo=repo)}

    assert not checks["allocation"].ok
    assert not checks["environment"].ok
    assert not checks["native image"].ok
    assert not checks["reference case"].ok
    assert not checks["reference run"].ok
    # A failed check is only useful if it says what would satisfy it.
    assert all(check.produced_by for check in checks.values())
    assert "uv sync" in checks["environment"].produced_by
    assert "site.env" in checks["allocation"].produced_by


def test_resolved_derives_every_path_from_the_checkout(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    for name in ("FREECAM_SCRATCH", "FREECAM_CASES", "FREECAM_REFERENCE_CASE"):
        monkeypatch.delenv(name, raising=False)
    repo = _checkout(tmp_path)
    config = repo / "configs" / "pi_cam_icesm131.yaml"
    config.parent.mkdir()
    config.write_text("case_name: a-case\n")

    where = site.resolved(repo=repo)

    assert where["scratch"] == tmp_path / "scratch"
    assert where["cases"] == tmp_path / "CESM_cases"
    assert where["reference case"] == tmp_path / "CESM_cases" / "a-case"
    assert where["reference run"] == (
        tmp_path / "scratch" / "pyCAM" / "PI-cam" / "a-case" / "run"
    )
