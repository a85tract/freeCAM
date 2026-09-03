"""The CLI's per-rank cProfile switch: off by default, on for the named ranks."""

from __future__ import annotations

import cProfile

import pytest

from freecam.pi_cam.cli import _cprofile_for, _write_cprofile


def test_unset_or_empty_profiles_no_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FREECAM_CPROFILE_RANKS", raising=False)
    assert _cprofile_for(0) is None
    monkeypatch.setenv("FREECAM_CPROFILE_RANKS", "  ")
    assert _cprofile_for(0) is None


def test_named_ranks_get_a_profiler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREECAM_CPROFILE_RANKS", "0, 128,511")
    assert isinstance(_cprofile_for(0), cProfile.Profile)
    assert isinstance(_cprofile_for(128), cProfile.Profile)
    assert isinstance(_cprofile_for(511), cProfile.Profile)
    assert _cprofile_for(1) is None


def test_write_leaves_binary_stats_and_a_table(tmp_path) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    sum(range(1000))
    profiler.disable()
    _write_cprofile(profiler, tmp_path, 7)
    assert (tmp_path / "cprofile.rank-7.prof").stat().st_size > 0
    table = (tmp_path / "cprofile.rank-7.txt").read_text()
    assert "tottime" in table
