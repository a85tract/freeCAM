from pathlib import Path

import pytest

import pycam_sima
from pycam_sima.model import (
    ModelConfig,
    ModelOptions,
    PHYSICS_BEFORE_COUPLER,
)
from pycam_sima.notebook.session import NotebookSession, NotebookWorkerError


def _session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    env_script: Path | None = None,
) -> NotebookSession:
    del monkeypatch
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "atm_in").write_text("&cam_initfiles_nl /\n")
    library = tmp_path / "libpycam_sima_kernels.so"
    library.touch()
    if env_script is None:
        env_script = tmp_path / "machine-env.sh"
        env_script.write_text("export PYCAM_SIMA_NOTEBOOK_TEST=ready\n")
    return NotebookSession(
        ModelConfig(),
        run_dir=run_dir,
        library=library,
        env_script=env_script,
    )


def test_notebook_session_is_public_and_requires_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    assert pycam_sima.NotebookSession is NotebookSession
    assert session.runtime == "model"
    assert session.ranks == 24
    assert not session.running
    with pytest.raises(RuntimeError, match="not running"):
        session.step()


def test_worker_response_and_environment_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_script = tmp_path / "machine-env.sh"
    environment_script.write_text("export PYCAM_SIMA_NOTEBOOK_TEST=ready\n")
    session = _session(tmp_path, monkeypatch, env_script=environment_script)
    environment = session._worker_environment()
    assert environment["PYCAM_SIMA_NOTEBOOK_TEST"] == "ready"
    assert "lib-abi-mpich" in environment["LD_LIBRARY_PATH"]
    assert session._unwrap({"status": "ok", "result": 7}) == 7
    with pytest.raises(NotebookWorkerError, match="remote failure"):
        session._unwrap({"status": "error", "error": "remote failure"})


def test_jupyter_without_pbs_uses_local_compute_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "pycam_sima.notebook.session.socket.gethostname", lambda: "dec9999"
    )
    assert session._launcher_command({}) == [
        "mpiexec",
        "--hosts",
        "dec9999",
        "--no-vni",
    ]
    assert session._launcher_command({"PBS_NODEFILE": "/tmp/nodes"}) == ["mpiexec"]


def test_auto_mode_uses_pbs_on_login_and_forced_local_mode_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "pycam_sima.notebook.session.socket.gethostname", lambda: "derecho6"
    )
    assert session._resolve_launch_mode({}) == "pbs"
    with pytest.raises(RuntimeError, match="login node"):
        session._launcher_command({})


def test_phase_api_tracks_worker_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    session._phase_names = ("dynamics_to_physics",)
    requests = []
    monkeypatch.setattr(session, "_validate_started_options", lambda: None)

    def request(payload):
        requests.append(payload)
        return {
            "sequence_safe": True,
            "step": 0,
            "native_nstep": 0,
            "native_calls": 3,
            "phase_status": {
                "last_phase": payload["phase"],
                "next_phase": None,
                "step": 0,
            },
            "scheme_status": session._scheme_status,
        }

    monkeypatch.setattr(session, "_request", request)
    status = session.run_phase("dynamics_to_physics")
    assert status["last_phase"] == "dynamics_to_physics"
    assert session.next_phase is None
    assert requests == [{"op": "run_phase", "phase": "dynamics_to_physics"}]

    with pytest.raises(ValueError, match="unknown CAM phase"):
        session.run_phase("not_a_phase")


def test_step_uses_fixed_model_sequence_and_options_lock_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    requests = []
    session._started_options_fingerprint = session.options.fingerprint()
    monkeypatch.setattr(session, "_ensure_running", lambda: None)

    def request(payload):
        requests.append(payload)
        return {
            "step": 1,
            "phase_status": {"step": 1, "next_phase": None},
            "scheme_status": session._scheme_status,
        }

    monkeypatch.setattr(session, "_request", request)
    assert session.step() == 1
    assert requests == [{"op": "step", "count": 1}]

    session.options.timestep_seconds = 900
    with pytest.raises(ValueError, match="1800 second"):
        session.step()


def test_scheme_plan_is_editable_and_scheme_calls_are_collective(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    assert len(session.scheme_plan.describe(PHYSICS_BEFORE_COUPLER)) == 19
    session.scheme_plan.disable("kessler_diagnostics", unsafe=True)
    assert not session._scheme_plan.scheme("kessler_diagnostics").enabled
    assert session.scheme_plan.sequence_safe is False
    session.scheme_plan.reset()

    requests = []
    monkeypatch.setattr(session, "_validate_started_options", lambda: None)

    def request(payload):
        requests.append(payload)
        plan = session._scheme_plan.to_payload()
        return {
            "step": 0,
            "phase_status": {"step": 0, "last_phase": None},
            "scheme_status": {
                "last_scheme": (
                    payload["scheme"] if payload["op"] == "run_scheme" else None
                ),
                "sequence_safe": True,
                "plan": plan,
            },
        }

    monkeypatch.setattr(session, "_request", request)
    status = session.run_scheme("kessler", group=PHYSICS_BEFORE_COUPLER)
    assert status["last_scheme"] == ("physics_before_coupler.kessler")
    assert requests == [
        {
            "op": "run_scheme",
            "scheme": "physics_before_coupler.kessler",
        }
    ]


def test_typed_model_parameter_facade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    session._fields = {
        "air_temperature": {"shape": (4, 4, 30, 3, 3), "dtype": "<f8"},
        "zonal_wind": {"shape": (4, 4, 30, 3, 3), "dtype": "<f8"},
    }
    monkeypatch.setattr(
        session,
        "get_field_stats",
        lambda name, rank=0: {"field": name, "rank": rank, "mean": 240.0},
    )

    assert session.parameters.air_temperature.name == "air_temperature"
    assert session.parameters.air_temperature.stats(rank=3) == {
        "field": "air_temperature",
        "rank": 3,
        "mean": 240.0,
    }
    description = session.parameters.describe()
    assert description["runtime"]["runtime"] == "model"
    assert "air_temperature" in description["key_fields"]


def test_model_controller_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "atm_in").write_text("&cam_initfiles_nl /\n")
    session = NotebookSession(
        Path("configs/fkessler_model.yaml"),
        run_dir=run_dir,
        options=ModelOptions(),
    )
    assert session.runtime == "model"
    assert session.ranks == 24
    assert session.options.describe()["state_owner"] == "python"
    assert session.library.name == "libpycam_sima_kernels.so"


def test_model_config_object_is_materialized_for_worker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "atm_in").write_text("&cam_initfiles_nl /\n")
    config = ModelConfig.from_yaml(Path("configs/fkessler_model.yaml"))
    session = NotebookSession(config, run_dir=run_dir, options=ModelOptions())
    assert session.config_path is None
