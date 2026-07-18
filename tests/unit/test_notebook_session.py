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


def test_jupyter_refuses_derecho_login_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path, monkeypatch)
    monkeypatch.setattr("pycam_sima.notebook_session.socket.gethostname", lambda: "derecho6")
    with pytest.raises(RuntimeError, match="login node"):
        session._launcher_command({})
