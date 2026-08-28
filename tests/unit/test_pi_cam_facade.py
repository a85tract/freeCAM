from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import (
    CASES,
    CaseConfig,
    CaseRegistry,
    CESMOnlineBoundaryProvider,
    Driver,
    FreeCAM,
    Physics,
    Property,
    OnlineBoundaryProvider,
    ProcessSpec,
    PICAMStepPlan,
    RunResult,
    process,
)
from freecam.pi_cam import facade
from freecam.pi_cam.errors import PICAMConfigurationError


class FakeSession:
    instances: list["FakeSession"] = []

    def __init__(self, config, **kwargs) -> None:
        self.config = Path(config)
        self.kwargs = kwargs
        self.running = False
        self.closed = False
        self.state = object()
        self.workflow_replacements = []
        self.output_config = None

        class Workflow:
            def replace(inner_self, processes):
                self.workflow_replacements.append(tuple(processes))

        self.workflow = Workflow()
        self.fields = object()
        self.physics = object()
        self.phases = object()
        self.kernels = object()
        self.trace_limit = kwargs.get("trace_limit")
        self._steps = 0
        # Two records exist before any run, mirroring driver initialization.
        self._records = [
            {"sequence": 0, "model_step": 0, "phase": "init", "name": "seed"},
            {"sequence": 1, "model_step": 0, "phase": "init", "name": "seed"},
        ]
        type(self).instances.append(self)

    def _retained(self):
        if self.trace_limit is None:
            return list(self._records)
        return self._records[-self.trace_limit :]

    @property
    def status(self):
        retained = self._retained()
        return {
            "step": self._steps,
            "actions": len(self._records),
            "trace_retained": len(retained),
            "trace_first_sequence": len(self._records) - len(retained),
            "trace_limit": self.trace_limit,
        }

    def start(self):
        self.running = True
        return self

    def advance(self, steps=1):
        for _ in range(steps):
            self._steps += 1
            self._records.append(
                {
                    "sequence": len(self._records),
                    "model_step": self._steps,
                    "phase": "cam_run1",
                    "name": "dadadj",
                }
            )
        return self.status

    def configure_output(self, **options):
        self.output_config = dict(options)
        return self.status

    def trace_window(self, *, since=0):
        total = len(self._records)
        assert 0 <= since <= total
        records = tuple(
            record for record in self._retained() if record["sequence"] >= since
        )
        return {
            "first_sequence": records[0]["sequence"] if records else total,
            "total": total,
            "records": records,
        }

    def trace(self, *, since=0):
        return self.trace_window(since=since)["records"]

    def close(self):
        self.closed = True
        self.running = False


class FakeWorkflowAction:
    def __init__(self, session, name, phase="cam") -> None:
        self.session = session
        self.name = name
        self.operation = name
        self.phase = phase
        self.enabled = True
        self.removed = False

    @property
    def qualified_name(self):
        return f"{self.phase}.{self.name}"

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def remove(self):
        self.removed = True


class FakeDeclarativeSession(FakeSession):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        actions = [
            FakeWorkflowAction(self, "boundary_import", "boundary"),
            FakeWorkflowAction(self, "radiation", "physics"),
            FakeWorkflowAction(self, "advance_timestep", "clock"),
            FakeWorkflowAction(self, "boundary_export", "boundary"),
        ]
        self.original_actions = tuple(actions)
        self.workflow_replacements = []

        class Workflow:
            def __init__(inner_self):
                inner_self.current = list(actions)

            def __getitem__(inner_self, index):
                return inner_self.current[index]

            def replace(inner_self, processes):
                inner_self.current = list(processes)
                self.workflow_replacements.append(tuple(processes))

        class PhysicsCollection:
            def __init__(inner_self):
                inner_self.installations = []

            def install_python(
                inner_self,
                function,
                *,
                name,
                before,
                reads,
                writes,
                parameters=None,
                enabled,
                transactional,
                native=False,
            ):
                del function, reads, writes, parameters, enabled, transactional, native
                phase = before.split(".", 1)[0]
                handle = FakeWorkflowAction(self, name, phase)
                inner_self.installations.append(
                    {"name": name, "before": before, "handle": handle}
                )
                return handle

        self.workflow = Workflow()
        self.physics = PhysicsCollection()


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
    (reference_run / "SEMapping.nc").write_bytes(b"mapping")
    (reference_run / "test.cam.h0.0001.nc").write_text("output\n")
    (reference_run / "test.clm2.r.0001.nc").write_text("output\n")
    (reference_run / "test.docn.rs1.bin").write_text("output\n")
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
    assert FakeSession.instances[0].output_config == {
        "history_every": 1,
        "restart_every": "end",
    }
    assert FakeSession.instances[0].kwargs["verify_boundary_exports"] is False
    driver.cam.workflow = ("physics-a", "physics-b")
    assert FakeSession.instances[0].workflow_replacements == [
        ("physics-a", "physics-b")
    ]
    assert (driver.run_dir / "atm_in").is_file()
    assert (driver.run_dir / "keep-me").is_file()
    assert (driver.run_dir / "SEMapping.nc").is_file()
    assert not (driver.run_dir / "test.cam.h0.0001.nc").exists()
    assert not (driver.run_dir / "test.clm2.r.0001.nc").exists()
    assert not (driver.run_dir / "test.docn.rs1.bin").exists()

    result = driver.run()

    assert isinstance(result, RunResult)
    assert [row["name"] for row in result.trace] == ["dadadj", "dadadj"]
    assert driver.trace[-2:] == result.trace
    assert result.summary()["completed_steps"] == 2
    assert result.actions == 2
    assert result.trace_truncated is False
    assert result.history is driver.cam.history
    assert "actions=2" in repr(result)
    assert len(FakeSession.instances) == 1
    driver.close()
    assert FakeSession.instances[0].closed


def test_driver_reports_progress_and_runs_in_background(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    updates = []
    driver = Driver(
        nsteps=3,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )

    run = driver.run_async(steps=3, progress=updates.append)
    result = run.result(timeout=5)

    assert run.done()
    assert not run.cancelled()
    assert run.progress.completed_steps == 3
    assert run.progress.model_step == 3
    assert [update.completed_steps for update in updates] == [1, 2, 3]
    assert result.completed_steps == 3
    assert len(result.trace) == 3
    driver.close()


def test_driver_samples_registered_live_plots_at_each_step(tmp_path) -> None:
    paths = _driver_tree(tmp_path)

    class TrackingSession(FakeSession):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.captures = 0

        @property
        def has_step_plots(self):
            return True

        def capture_step_plots(self):
            self.captures += 1

    driver = Driver(
        nsteps=3,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=TrackingSession,
    )

    result = driver.run(steps=3)

    assert result.completed_steps == 3
    assert TrackingSession.instances[-1].captures == 3
    driver.close()


def test_background_run_cancels_between_complete_steps(tmp_path) -> None:
    paths = _driver_tree(tmp_path)

    class BlockingSession(FakeSession):
        entered = threading.Event()
        release = threading.Event()

        def advance(self, steps=1):
            type(self).entered.set()
            if not type(self).release.wait(timeout=5):
                raise TimeoutError("test did not release complete step")
            return super().advance(steps=steps)

    BlockingSession.entered.clear()
    BlockingSession.release.clear()
    driver = Driver(
        nsteps=3,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=BlockingSession,
    )

    run = driver.run_async(steps=3)
    assert BlockingSession.entered.wait(timeout=5)
    assert run.cancel()
    BlockingSession.release.set()
    result = run.result(timeout=5)

    assert run.cancelled()
    assert run.progress.completed_steps == 1
    assert result.cancelled
    assert result.completed_steps == 1
    assert len(result.trace) == 1
    driver.close()


def _bounded_driver(paths, *, trace_limit, nsteps=5) -> Driver:
    return Driver(
        nsteps=nsteps,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
        trace_limit=trace_limit,
    )


def test_run_result_reports_truncation_exactly(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    driver = _bounded_driver(paths, trace_limit=2)

    result = driver.run(steps=5)

    assert result.actions == 5
    assert len(result.trace) == 2
    assert result.trace_truncated is True
    assert result.summary()["actions"] == 5
    assert result.summary()["trace_records"] == 2
    assert result.summary()["trace_truncated"] is True
    assert result.first_process == "dadadj"
    assert result.last_process == "dadadj"
    assert "trace_truncated=True" in repr(result)
    assert "showing last 2 of 5 actions" in result._repr_html_()
    driver.close()


def test_driver_passes_trace_limit_to_session_factory(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    driver = _bounded_driver(paths, trace_limit=99)
    driver.run(steps=1)
    assert FakeSession.instances[-1].kwargs["trace_limit"] == 99
    driver.close()

    default_driver = Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )
    default_driver.run(steps=1)
    assert FakeSession.instances[-1].kwargs["trace_limit"] == 4096
    default_driver.close()

    from freecam.pi_cam.errors import PICAMConfigurationError

    with pytest.raises(PICAMConfigurationError):
        _bounded_driver(paths, trace_limit=0)


def test_driver_trace_property_warns_when_truncated(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    driver = _bounded_driver(paths, trace_limit=2)
    driver.run(steps=3)

    with pytest.warns(RuntimeWarning, match="showing last 2 of 5 actions"):
        trace = driver.trace

    assert len(trace) == 2
    driver.close()


def test_verbose_run_notes_unretained_actions(tmp_path, capsys) -> None:
    paths = _driver_tree(tmp_path)
    driver = _bounded_driver(paths, trace_limit=2)

    driver.run(steps=5, verbose=True)

    output = capsys.readouterr().out
    assert "... 3 earlier actions not retained (trace_limit)" in output
    driver.close()


def test_driver_diagnose_never_starts_mpi(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    driver = Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        python_executable="/usr/bin/python3",
        session_factory=FakeSession,
    )

    diagnosis = driver.diagnose()

    assert diagnosis["ready"] is True
    assert diagnosis["mpi_ranks"] == 1
    assert diagnosis["checks"]["boundary"] is True
    assert not driver.running


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


def test_online_driver_accepts_steps_beyond_configured_default(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    config = paths["config"]
    config.write_text(config.read_text() + "boundary_mode: online\n")
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "manifest.json").write_text(
        '{"schema_version":1,"storage":"rank_bootstrap_v1",'
        '"rank_count":1,"file_pattern":"rank-{rank:04d}.npz"}\n'
    )
    np.savez(
        bootstrap / "rank-0000.npz",
        x2a_rattr=np.zeros((2, 3)),
        a2x_rattr=np.zeros((4, 3)),
    )
    provider = OnlineBoundaryProvider.held(bootstrap)
    driver = Driver(
        case="PI-atm",
        nsteps=6,
        repo=paths["repo"],
        config=config,
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=provider,
        python_executable="/usr/bin/python3",
        session_factory=FakeSession,
    )

    diagnosis = driver.diagnose()

    assert diagnosis["ready"] is True
    assert diagnosis["checks"]["boundary_bootstrap"] is True
    assert diagnosis["boundary"] == "OnlineBoundaryProvider"


def test_default_online_driver_prepares_exact_provider_lazily(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    config = paths["config"]
    config.write_text(config.read_text() + "boundary_mode: online\n")
    library = tmp_path / "libpycesm_support.so"
    library.write_bytes(b"test library")
    seed = tmp_path / "cesm-seed"
    seed.mkdir()
    (seed / "drv_in").write_text("driver input\n")
    (seed / "SEMapping.nc").write_bytes(b"mapping")
    (seed / "lnd_in").write_text("land input\n")
    (seed / "history.nc").write_bytes(b"do not copy")

    driver = Driver(
        case="PI-atm",
        nsteps=6,
        repo=paths["repo"],
        config=config,
        scratch=tmp_path / "scratch",
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        online_library=library,
        online_seed_run=seed,
        python_executable="/usr/bin/python3",
        session_factory=FakeSession,
    )

    diagnosis = driver.diagnose()
    assert diagnosis["boundary"] == "CESMOnlineBoundaryProvider (automatic)"
    assert diagnosis["ready"] is True
    assert driver.boundary is None
    assert driver.run_dir is None

    _ = driver.cam.state

    assert isinstance(driver.boundary, CESMOnlineBoundaryProvider)
    assert driver.boundary.oracle is None
    assert driver.boundary.run_dir == driver.run_dir.parent / "cesm-provider-run"
    assert (driver.boundary.run_dir / "drv_in").is_file()
    assert (driver.boundary.run_dir / "SEMapping.nc").is_file()
    assert not (driver.boundary.run_dir / "history.nc").exists()
    assert FakeSession.instances[-1].kwargs["boundary"] is driver.boundary
    driver.close()


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


def test_driver_reads_the_account_from_the_site_file(tmp_path, monkeypatch) -> None:
    paths = _driver_tree(tmp_path)
    (paths["repo"] / "pyproject.toml").write_text("[project]\nname = 'freecam'\n")
    (paths["repo"] / "site.env").write_text("FREECAM_ACCOUNT=SITE_ACCOUNT\n")
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)
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

    # The site file beats the reference case, whose CHARGE_ACCOUNT is whoever
    # happened to configure it.
    assert driver.account == "SITE_ACCOUNT"
    assert "site.env" in driver.account_source


def test_driver_environment_account_overrides_the_site_file(
    tmp_path, monkeypatch
) -> None:
    paths = _driver_tree(tmp_path)
    (paths["repo"] / "pyproject.toml").write_text("[project]\nname = 'freecam'\n")
    (paths["repo"] / "site.env").write_text("FREECAM_ACCOUNT=SITE_ACCOUNT\n")
    monkeypatch.setenv("FREECAM_ACCOUNT", "ENVIRONMENT_ACCOUNT")

    driver = Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )

    assert driver.account == "ENVIRONMENT_ACCOUNT"
    assert driver.account_source == "$FREECAM_ACCOUNT"


def test_driver_warns_when_the_account_belongs_to_another_user(
    tmp_path, monkeypatch
) -> None:
    paths = _driver_tree(tmp_path)
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)
    monkeypatch.delenv("PBS_ACCOUNT_DERECHO", raising=False)
    # The reference case is somebody else's, as it is whenever a shared
    # installation is used.
    monkeypatch.setattr(facade.os, "getuid", lambda: os.stat(__file__).st_uid + 1)

    with pytest.warns(RuntimeWarning, match="belongs to another user"):
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


def test_driver_does_not_warn_about_the_users_own_site_file(
    tmp_path, monkeypatch, recwarn
) -> None:
    paths = _driver_tree(tmp_path)
    (paths["repo"] / "pyproject.toml").write_text("[project]\nname = 'freecam'\n")
    (paths["repo"] / "site.env").write_text("FREECAM_ACCOUNT=SITE_ACCOUNT\n")
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)

    Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )

    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_driver_warns_when_the_site_file_belongs_to_another_user(
    tmp_path, monkeypatch
) -> None:
    # Running out of a colleague's installation is the normal way to use one:
    # their site.env is right about every path and wrong about who pays.
    paths = _driver_tree(tmp_path)
    (paths["repo"] / "pyproject.toml").write_text("[project]\nname = 'freecam'\n")
    (paths["repo"] / "site.env").write_text("FREECAM_ACCOUNT=THEIR_ACCOUNT\n")
    monkeypatch.delenv("FREECAM_ACCOUNT", raising=False)
    monkeypatch.setattr(facade.os, "getuid", lambda: os.stat(__file__).st_uid + 1)

    with pytest.warns(RuntimeWarning, match="belongs to another user"):
        driver = Driver(
            nsteps=2,
            repo=paths["repo"],
            config=paths["config"],
            reference_case=paths["reference_case"],
            reference_run=paths["reference_run"],
            boundary=paths["boundary"],
            session_factory=FakeSession,
        )

    assert driver.account == "THEIR_ACCOUNT"


def test_driver_takes_scratch_and_queue_from_the_site_file(
    tmp_path, monkeypatch
) -> None:
    paths = _driver_tree(tmp_path)
    (paths["repo"] / "pyproject.toml").write_text("[project]\nname = 'freecam'\n")
    (paths["repo"] / "site.env").write_text(
        "FREECAM_ACCOUNT=SITE_ACCOUNT\n"
        f"FREECAM_SCRATCH={tmp_path / 'site-scratch'}\n"
        "FREECAM_QUEUE=main\n"
    )
    monkeypatch.delenv("FREECAM_SCRATCH", raising=False)
    monkeypatch.delenv("FREECAM_QUEUE", raising=False)

    driver = Driver(
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
    )

    assert driver.scratch == tmp_path / "site-scratch"
    assert driver.queue == "main"


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


def test_public_case_registry_and_driver_output_options(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    registry = CaseRegistry()
    custom = registry.register(
        CaseConfig(
            name="diagnostic",
            description="registered test case",
            forcing="test",
            base="PI-atm",
        )
    )

    assert registry["diagnostic"] is custom
    assert "PI-atm" in CASES
    assert CASES["PI-atm-1month"].base == "PI-atm-1month"
    CASES.register(custom)
    try:
        driver = Driver(
            case="diagnostic",
            nsteps=2,
            repo=paths["repo"],
            config=paths["config"],
            reference_case=paths["reference_case"],
            reference_run=paths["reference_run"],
            boundary=paths["boundary"],
            history_every=5,
            restart_every=None,
            session_factory=FakeSession,
        ).initialize()

        assert driver.case is custom
        assert FakeSession.instances[-1].output_config == {
            "history_every": 5,
            "restart_every": None,
        }
        driver.close()
    finally:
        CASES.unregister("diagnostic")


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


class VolcanicAerosol(Physics):
    name = "volcanic_aerosol"
    writes = ("air_temperature",)

    def tendency(self, fields, context):
        del fields, context


def test_one_argument_physics_callback_uses_friendly_state_view() -> None:
    observed = []

    class Heating(Physics):
        writes = ("T",)

        def run(self, state):
            state.T += 1.0
            observed.append(float(state.T[0]))

    values = {"T": np.zeros(2)}

    result = Heating()._runtime_callback()(values, object())

    assert result is None
    assert observed == [1.0]
    assert values["T"].tolist() == [1.0, 1.0]


def test_process_spec_declares_complete_workflow_before_launch() -> None:
    declared = [process(action.qualified_name) for action in PICAMStepPlan.default()]

    preview = FreeCAM(workflow=declared).preview()

    assert isinstance(declared[0], ProcessSpec)
    assert len(preview.debug) == len(declared)
    assert preview.debug[0].operation == "boundary_import"
    assert preview.debug[-1].operation == "boundary_export"


def test_case_config_installs_repeated_python_processes_in_declared_order(
    tmp_path,
) -> None:
    paths = _driver_tree(tmp_path)
    FakeSession.instances.clear()

    def volcanic_workflow(default):
        workflow = default.copy()
        workflow.insert_after("radiation", VolcanicAerosol())
        workflow.insert_before("advance_timestep", VolcanicAerosol())
        return workflow

    case = CaseConfig(
        name="PI-atm-volcanic",
        description="two runtime aerosol processes",
        forcing="fixed PI plus volcanic aerosol",
        make_atm=lambda: FreeCAM(workflow=volcanic_workflow),
    )
    driver = Driver(
        case=case,
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        scratch=tmp_path / "scratch",
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeDeclarativeSession,
    )

    assert not driver.running
    preview = driver.preview()
    assert not driver.running
    assert [item.name for item in preview if item.kind == "python_process"] == [
        "volcanic_aerosol_1",
        "volcanic_aerosol_2",
    ]
    assert case.workflow.describe() == preview.describe()
    driver.initialize()

    session = FakeSession.instances[0]
    assert driver.case is case
    assert [item.name for item in driver.cam.configured_processes] == [
        "volcanic_aerosol_1",
        "volcanic_aerosol_2",
    ]
    assert [item.name for item in session.workflow.current] == [
        "boundary_import",
        "radiation",
        "volcanic_aerosol_1",
        "volcanic_aerosol_2",
        "advance_timestep",
        "boundary_export",
    ]
    assert [item["before"] for item in session.physics.installations] == [
        "clock.advance_timestep",
        "clock.advance_timestep",
    ]


def test_case_workflow_can_omit_optional_process_but_not_control_boundary(
    tmp_path,
) -> None:
    paths = _driver_tree(tmp_path)

    def no_radiation(default):
        default.remove(default.process("radiation"))
        return default

    driver = Driver(
        case=CaseConfig(
            name="no-radiation",
            description="test",
            forcing="test",
            make_atm=lambda: FreeCAM(no_radiation),
        ),
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeDeclarativeSession,
    )
    driver.initialize()
    session = FakeSession.instances[-1]
    radiation = next(
        item for item in session.original_actions if item.name == "radiation"
    )
    assert not radiation.enabled
    assert "radiation" not in [item.name for item in session.workflow.current]

    def invalid(default):
        default.remove(default.process("boundary_export"))
        return default

    invalid_driver = Driver(
        case=CaseConfig(
            name="invalid",
            description="test",
            forcing="test",
            make_atm=lambda: FreeCAM(invalid),
        ),
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeDeclarativeSession,
    )
    with pytest.raises(ValueError, match="cannot remove required CAM control"):
        invalid_driver.initialize()
    assert FakeSession.instances[-1].closed


_MINI_CATALOG = """<?xml version="1.0"?>
<namelist_definition>
<entry id="cldfrc_rhminl" type="real" category="cldfrc"
       group="cldfrc_nl" valid_values="" >
Minimum rh & such.
</entry>
<entry id="zmconv_c0_lnd" type="real" category="conv" group="zmconv_nl" valid_values="" >
Autoconversion over land.
</entry>
</namelist_definition>
"""


def _namelist_tree(tmp_path) -> dict[str, Path]:
    paths = _driver_tree(tmp_path)
    catalog = (
        paths["repo"]
        / "components"
        / "cam"
        / "bld"
        / "namelist_files"
        / "namelist_definition.xml"
    )
    catalog.parent.mkdir(parents=True)
    catalog.write_text(_MINI_CATALOG)
    (paths["reference_run"] / "atm_in").write_text(
        "&cldfrc_nl\n cldfrc_rhminl\t\t= 0.870D0\n/\n"
    )
    return paths


def _namelist_driver(tmp_path, paths, **kwargs) -> Driver:
    return Driver(
        case="PI-atm",
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        scratch=tmp_path / "scratch",
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
        **kwargs,
    )


def test_namelist_overrides_are_validated_at_construction(tmp_path) -> None:
    paths = _namelist_tree(tmp_path)
    with pytest.raises(PICAMConfigurationError, match="cldfrc_rhminl"):
        _namelist_driver(tmp_path, paths, namelist={"cldfrc_rhminls": 0.9})
    with pytest.raises(PICAMConfigurationError, match="boolean"):
        _namelist_driver(tmp_path, paths, namelist={"cldfrc_rhminl": True})


def test_namelist_overrides_reach_the_run_directory(tmp_path) -> None:
    paths = _namelist_tree(tmp_path)
    driver = _namelist_driver(
        tmp_path, paths, namelist={"cldfrc_rhminl": 0.9}
    )
    assert driver.cam.namelist["cldfrc_rhminl"] == "0.870D0"
    run_dir = driver._prepare_run_dir()
    assert (
        (run_dir / "atm_in").read_text()
        == "&cldfrc_nl\n cldfrc_rhminl\t\t= 0.9D0\n/\n"
    )
    assert driver.cam.namelist["cldfrc_rhminl"] == "0.9D0"
    assert driver.cam.namelist.overrides == {
        "cldfrc_rhminl": ("0.870D0", "0.9D0")
    }


def test_default_run_directory_namelist_is_byte_identical(tmp_path) -> None:
    paths = _namelist_tree(tmp_path)
    driver = _namelist_driver(tmp_path, paths)
    run_dir = driver._prepare_run_dir()
    assert (
        (run_dir / "atm_in").read_bytes()
        == (paths["reference_run"] / "atm_in").read_bytes()
    )
    assert driver.cam.namelist.overrides == {}


def test_case_namelist_merges_under_the_driver_kwarg(tmp_path) -> None:
    paths = _namelist_tree(tmp_path)
    case = CaseConfig(
        name="PI-atm-tuned",
        description="perturbed cloud fraction",
        forcing="fixed PI",
        namelist={"cldfrc_rhminl": 0.9, "zmconv_c0_lnd": 0.004},
    )
    driver = Driver(
        case=case,
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        scratch=tmp_path / "scratch",
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=FakeSession,
        namelist={"cldfrc_rhminl": 0.95},
    )
    assert driver._namelist_overrides == {
        "cldfrc_rhminl": 0.95,
        "zmconv_c0_lnd": 0.004,
    }


def test_physics_properties_inject_before_run() -> None:
    observed = []

    class Heating(Physics):
        rate = Property(0.01)

        def run(self, state, context):
            del context
            observed.append(self.rate)
            state.T += self.rate

    values = {"T": np.zeros(2)}
    callback = Heating()._runtime_callback()

    callback(values, object())
    assert observed == [0.01]

    callback(values, object(), properties={"rate": 0.5})
    assert observed == [0.01, 0.5]
    assert values["T"].tolist() == [0.51, 0.51]

    with pytest.raises(TypeError, match="declares no property"):
        callback(values, object(), properties={"missing": 1.0})


def test_two_instances_of_one_class_keep_independent_properties() -> None:
    class Cooling(Physics):
        rate = Property(1.0)

        def run(self, state, context):
            del state, context

    first, second = Cooling(), Cooling()
    first.rate = 2.0
    assert first.rate == 2.0
    assert second.rate == 1.0
    assert Cooling.rate.default == 1.0


def test_property_cannot_shadow_the_physics_contract() -> None:
    # Python 3.11 wraps a __set_name__ failure in RuntimeError and carries the
    # original as its cause; 3.12 and later raise the original.  What matters
    # is that the class is refused and says why, on either.
    with pytest.raises((RuntimeError, TypeError)) as excinfo:

        class Broken(Physics):  # noqa: F811 - intentionally discarded
            enabled = Property(True)

    raised = excinfo.value
    reason = raised.__cause__ if isinstance(raised, RuntimeError) else raised
    assert "contract attribute" in str(reason)


def test_pickled_physics_instance_does_not_forward_property_writes() -> None:
    import cloudpickle

    forwarded = []

    class ForwardingSession:
        def set_python_parameters(self, name, parameters):
            forwarded.append((name, parameters))
            return {}

    class Tracer(Physics):
        scale = Property(1.0)

        def run(self, state, context):
            del state, context

    from freecam.pi_cam.facade import _LIVE_PHYSICS

    instance = Tracer()
    _LIVE_PHYSICS[instance] = (ForwardingSession(), "tracer")
    callback = instance._runtime_callback()

    restored = cloudpickle.loads(cloudpickle.dumps(callback))
    restored({"T": np.zeros(1)}, object(), properties={"scale": 3.0})
    assert forwarded == []

    instance.scale = 2.0
    assert forwarded == [("tracer", {"properties": {"scale": 2.0}})]
    assert instance.scale == 2.0


def test_failed_forwarding_leaves_the_local_mirror_unchanged() -> None:
    class RejectingSession:
        def set_python_parameters(self, name, parameters):
            raise RuntimeError("ranks disagreed")

    class Tracer(Physics):
        scale = Property(1.0)

        def run(self, state, context):
            del state, context

    from freecam.pi_cam.facade import _LIVE_PHYSICS

    instance = Tracer()
    _LIVE_PHYSICS[instance] = (RejectingSession(), "tracer")
    with pytest.raises(RuntimeError, match="ranks disagreed"):
        instance.scale = 2.0
    assert instance.scale == 1.0


def test_install_ships_properties_and_registers_forwarding() -> None:
    installations = []

    class PhysicsCollection:
        def install_python(self, function, **kwargs):
            installations.append(kwargs)
            return "handle"

    class SessionStub:
        def __init__(self):
            self.physics = PhysicsCollection()
            self.parameter_updates = []

        def set_python_parameters(self, name, parameters):
            self.parameter_updates.append((name, parameters))
            return {}

    class Heating(Physics):
        name = "notebook_heating"
        after = "dadadj"
        rate = Property(0.01)

        def run(self, state, context):
            del context
            state.T += self.rate

    session = SessionStub()
    instance = Heating()
    assert instance._install(session) == "handle"
    assert installations[0]["parameters"] == {"properties": {"rate": 0.01}}

    instance.rate = 0.02
    assert session.parameter_updates == [
        ("notebook_heating", {"properties": {"rate": 0.02}})
    ]

    plain = VolcanicAerosol()
    plain._install(session)
    assert installations[1]["parameters"] is None


def test_cam_parameters_view_reads_and_writes_collectively(tmp_path) -> None:
    paths = _driver_tree(tmp_path)
    FakeSession.instances.clear()

    described = {
        "parameters": {
            "zmconv_c0_lnd": {
                "value": 0.0059,
                "baseline": 0.0059,
                "workflow_action": "cam_run1.deep_convection",
            }
        },
        "unavailable": {"uwshcu_rpen": "not set in this run's atm_in"},
    }
    writes = []

    class ParameterSession(FakeSession):
        def get_module_parameters(self):
            return described

        def set_module_parameter(self, name, value):
            writes.append((name, value))
            described["parameters"][name]["value"] = value
            return {"name": name, "value": value}

    driver = Driver(
        case="PI-atm",
        nsteps=2,
        repo=paths["repo"],
        config=paths["config"],
        scratch=tmp_path / "scratch",
        reference_case=paths["reference_case"],
        reference_run=paths["reference_run"],
        boundary=paths["boundary"],
        session_factory=ParameterSession,
    )

    assert dict(driver.cam.parameters) == {"zmconv_c0_lnd": 0.0059}
    assert driver.cam.parameters.overrides == {}
    driver.cam.parameters["zmconv_c0_lnd"] = 0.0075
    assert writes == [("zmconv_c0_lnd", 0.0075)]
    assert driver.cam.parameters["zmconv_c0_lnd"] == 0.0075
    assert driver.cam.parameters.overrides == {
        "zmconv_c0_lnd": (0.0059, 0.0075)
    }
    assert (
        driver.cam.parameters.unavailable["uwshcu_rpen"]
        == "not set in this run's atm_in"
    )
