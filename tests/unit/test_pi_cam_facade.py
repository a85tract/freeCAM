from __future__ import annotations

from pathlib import Path

from freecam.pi_cam import Driver


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self, config, **kwargs) -> None:
        self.config = Path(config)
        self.kwargs = kwargs
        self.running = False
        self.closed = False
        self.state = object()
        self.workflow = object()
        self.fields = object()
        self.physics = object()
        self.phases = object()
        self.kernels = object()
        self._actions = 2
        self._trace = []
        type(self).instances.append(self)

    @property
    def status(self):
        return {"step": len(self._trace), "actions": self._actions}

    def start(self):
        self.running = True
        return self

    def advance(self, steps=1):
        for step in range(steps):
            self._trace.append(
                {
                    "model_step": step + 1,
                    "phase": "cam_run1",
                    "name": "dadadj",
                }
            )
        self._actions += steps
        return self.status

    def trace(self, *, since=0):
        assert since == 2
        return tuple(self._trace)

    def close(self):
        self.closed = True
        self.running = False


def _driver_tree(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "freeCAM"
    config = repo / "configs" / "pi_cam.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "\n".join(
            (
                "case_name: test-case",
                f"source_root: {repo}",
                "mpi_size: 1",
                "stop_n: 5",
            )
        )
        + "\n"
    )
    reference_case = tmp_path / "case"
    reference_case.mkdir()
    (reference_case / ".env_mach_specific.sh").write_text("true\n")
    (reference_case / "env_batch.xml").write_text(
        '<config><entry id="PROJECT" value="TEST_ACCOUNT"/></config>\n'
    )
    reference_run = tmp_path / "reference-run"
    reference_run.mkdir()
    (reference_run / "atm_in").write_text("&atm_in /\n")
    (reference_run / "keep-me").write_text("input\n")
    (reference_run / "test.cam.h0.0001.nc").write_text("output\n")
    (reference_run / "timing").mkdir()
    boundary = tmp_path / "boundary"
    boundary.mkdir()
    (boundary / "manifest.json").write_text("{}\n")
    return {
        "repo": repo,
        "config": config,
        "reference_case": reference_case,
        "reference_run": reference_run,
        "boundary": boundary,
    }


def test_driver_hides_run_preparation_and_lazily_reuses_one_session(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    FakeSession.instances.clear()
    driver = Driver(
        case="PI-atm",
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        scratch=tmp_path / "scratch",
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )

    assert not driver.running
    assert driver.run_dir is None
    assert driver.cam.state is driver.cam.state
    assert driver.running
    assert len(FakeSession.instances) == 1
    assert (driver.run_dir / "atm_in").is_file()
    assert (driver.run_dir / "keep-me").is_file()
    assert not (driver.run_dir / "test.cam.h0.0001.nc").exists()

    trace = driver.execute()

    assert [row["name"] for row in trace] == ["dadadj", "dadadj"]
    assert len(FakeSession.instances) == 1
    driver.close()
    assert FakeSession.instances[0].closed


def test_driver_rejects_unknown_case_and_boundary_overrun(tmp_path) -> None:
    paths = _driver_tree(tmp_path)

    try:
        Driver(case="unknown", repo=paths["repo"])
    except ValueError as error:
        assert "available cases" in str(error)
    else:
        raise AssertionError("unknown case was accepted")

    try:
        Driver(
            nsteps=6,
            repo=paths["repo"],
            config=paths["config"],
        )
    except ValueError as error:
        assert "replay boundary" in str(error)
    else:
        raise AssertionError("boundary overrun was accepted")


def test_driver_reads_case_account_and_preserves_venv_python_symlink(
    tmp_path, monkeypatch
) -> None:
    paths = _driver_tree(tmp_path)
    python = paths["repo"] / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3.11")
    monkeypatch.setenv("PBS_ACCOUNT", "INVALID_FOR_DERECHO")
    monkeypatch.delenv("PBS_ACCOUNT_DERECHO", raising=False)

    driver = Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )

    assert driver.account == "TEST_ACCOUNT"
    assert driver.python_executable == python.absolute()


def test_driver_explicit_account_overrides_reference_case(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    driver = Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        account="EXPLICIT_ACCOUNT",
        session_factory=FakeSession,
    )

    assert driver.account == "EXPLICIT_ACCOUNT"


def test_driver_does_not_use_login_shell_pbs_account_as_fallback(
    tmp_path, monkeypatch
) -> None:
    paths = _driver_tree(tmp_path)
    (paths["reference_case"] / "env_batch.xml").unlink()
    monkeypatch.setenv("PBS_ACCOUNT", "UNRELATED_LOGIN_ACCOUNT")
    monkeypatch.delenv("PBS_ACCOUNT_DERECHO", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)

    driver = Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )

    assert driver.account is None
