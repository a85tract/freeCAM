from pathlib import Path
from types import SimpleNamespace

import pytest

import pycam_sima
from pycam_sima.notebook_session import NotebookSession, NotebookWorkerError


def _session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, env_script=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "atm_in").write_text("&cam_initfiles_nl /\n")
    library = tmp_path / "libpycam_sima_full.so"
    library.touch()
    config_path = tmp_path / "case.yaml"
    config_path.touch()
    config = SimpleNamespace(
        config_path=config_path,
        mpi_ranks=24,
        native=SimpleNamespace(se_library=library),
    )
    monkeypatch.setattr(
        "pycam_sima.notebook_session.CaseConfig.from_yaml",
        lambda path: config,
    )
    return NotebookSession(config_path, run_dir=run_dir, env_script=env_script)


def test_notebook_session_is_public_and_requires_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    assert pycam_sima.NotebookSession is NotebookSession
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
    monkeypatch.setattr("pycam_sima.notebook_session.socket.gethostname", lambda: "dec9999")
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
    monkeypatch.setattr("pycam_sima.notebook_session.socket.gethostname", lambda: "derecho6")
    assert session._resolve_launch_mode({}) == "pbs"
    with pytest.raises(RuntimeError, match="login node"):
        session._launcher_command({})


def test_phase_api_tracks_worker_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    session._phase_names = (
        "cam_run2",
        "cam_run3",
        "cam_run4",
        "cam_timestep_final",
        "advance_timestep",
        "cam_timestep_init",
        "cam_run1",
    )
    requests = []

    def request(payload):
        requests.append(payload)
        return {
            "last_phase": payload["phase"],
            "next_phase": "cam_run3",
            "sequence_safe": True,
            "cycle_kind": "initial_send",
            "cycle_complete": False,
            "step": 0,
            "native_nstep": 0,
        }

    monkeypatch.setattr(session, "_request", request)
    status = session.run_phase("cam_run2")
    assert status["last_phase"] == "cam_run2"
    assert session.next_phase == "cam_run3"
    assert requests == [
        {
            "op": "run_phase",
            "phase": "cam_run2",
            "allow_unsafe_order": False,
        }
    ]

    with pytest.raises(ValueError, match="unknown CAM phase"):
        session.run_phase("not_a_phase")
