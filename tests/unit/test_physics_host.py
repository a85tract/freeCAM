"""Worker isolation: the host classifies how a worker died and carries on."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from freecam.physics.host import StubCalledError, SubprocessHost, WorkerRestartLimit

FAKE = (sys.executable, str(Path(__file__).resolve().parent / "fake_physics_worker.py"))


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "function": "dadadj", "library": str(tmp_path / "lib.so"), "library_sha256": "abc",
        "intel_math_library": str(tmp_path / "libimf.so"),
    }))
    return path


def _host(tmp_path: Path, monkeypatch, mode: str, **kwargs) -> SubprocessHost:
    monkeypatch.setenv("FAKE_WORKER_MODE", mode)
    return SubprocessHost(_manifest(tmp_path), {"entries": {}}, worker_command=FAKE, **kwargs)


def test_calls_round_trip_and_parameters_reach_the_worker(tmp_path: Path, monkeypatch) -> None:
    with _host(tmp_path, monkeypatch, "ok") as host:
        assert host.verification["all_equal"]
        pool = {"f.x": np.ones((2, 1), order="F"), "f.y": np.full((2, 1), 3.0, order="F")}
        outcome = host.call(pool, returned=("f.y",))
        assert outcome.status == "ok" and np.array_equal(outcome.pool["f.y"], [[6.0], [6.0]])
        host.set_parameters({"gain": 1.0})
        assert np.array_equal(host.call(pool, returned=("f.y",)).pool["f.y"], [[9.0], [9.0]])
        host.restore_parameters()
        assert np.array_equal(host.call(pool, returned=("f.y",)).pool["f.y"], [[6.0], [6.0]])
        assert host.restarts == 0


def test_a_fortran_abort_fails_only_that_sample_and_keeps_the_parameters(tmp_path: Path, monkeypatch) -> None:
    with _host(tmp_path, monkeypatch, "abort:2") as host:
        host.set_parameters({"gain": 1.0})
        pool = {"f.y": np.ones((1, 1), order="F")}
        assert host.call(pool).status == "ok"
        aborted = host.call(pool)
        assert aborted.status == "fortran_abort"
        assert aborted.message == "Impossible case1 in instratus_condensate"
        assert host.restarts == 1
        # The fresh worker is back at the same parameter state.
        assert np.array_equal(host.call(pool).pool["f.y"], [[3.0]])


def test_a_crash_is_reported_as_a_crash(tmp_path: Path, monkeypatch) -> None:
    with _host(tmp_path, monkeypatch, "segv:2") as host:
        assert host.call({"f.y": np.ones((1, 1), order="F")}).status == "ok"
        outcome = host.call({"f.y": np.ones((1, 1), order="F")})
        assert outcome.status == "worker_crash" and "signal 11" in outcome.message
        assert host.call({"f.y": np.ones((1, 1), order="F")}).status == "ok"


def test_a_stub_reached_is_an_error_not_a_sample(tmp_path: Path, monkeypatch) -> None:
    with _host(tmp_path, monkeypatch, "stub:1") as host:
        with pytest.raises(StubCalledError, match="cam_history_mp_addfld_"):
            host.call({"f.y": np.ones((1, 1), order="F")})


def test_restarts_are_capped(tmp_path: Path, monkeypatch) -> None:
    with _host(tmp_path, monkeypatch, "abort:1", max_restarts=1) as host:
        assert host.call({"f.y": np.ones((1, 1), order="F")}).status == "fortran_abort"
        with pytest.raises(WorkerRestartLimit):
            host.call({"f.y": np.ones((1, 1), order="F")})


def test_worker_runs_with_the_math_library_preloaded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LD_PRELOAD", "/somewhere/else.so")
    with _host(tmp_path, monkeypatch, "ok") as host:
        hello = host._request({"op": "hello", "manifest": str(host.manifest_path)})
        assert hello["ld_preload"] == str(tmp_path / "libimf.so")
