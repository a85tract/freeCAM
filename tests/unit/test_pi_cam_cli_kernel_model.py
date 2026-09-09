"""--kernel-model: a cloudpickled callable in a named kernel slot, recorded by path and hash."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cloudpickle
import pytest

from freecam.pi_cam.cli import _kernel_models_summary, _load_kernel_model, _parse_kernel_models


def test_name_path_pairs_are_parsed_and_malformed_ones_refused() -> None:
    models = _parse_kernel_models(["instratus_condensate=/tmp/a.pkl", "mmacro_pcond = ~/b.pkl "])
    assert list(models) == ["instratus_condensate", "mmacro_pcond"]
    assert models["mmacro_pcond"] == Path("~/b.pkl").expanduser()
    assert _parse_kernel_models(None) == {} and _parse_kernel_models([]) == {}
    for bad in (["nopath"], ["=only_path"], ["name="], ["a=x", "a=y"]):
        with pytest.raises(SystemExit):
            _parse_kernel_models(bad)


def test_a_pickled_callable_loads_and_is_recorded_by_hash(tmp_path: Path) -> None:
    def model(batch):
        return {"t_out": batch["t0_in"]}

    path = tmp_path / "model.pkl"
    path.write_bytes(cloudpickle.dumps(model))
    loaded = _load_kernel_model(path)
    assert loaded({"t0_in": 3.0}) == {"t_out": 3.0}
    summary = _kernel_models_summary({"instratus_condensate": path})
    assert summary["instratus_condensate"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary["instratus_condensate"]["path"] == str(path)
    assert _kernel_models_summary({}) is None
    (tmp_path / "not_callable.pkl").write_bytes(cloudpickle.dumps({"weights": [1, 2]}))
    with pytest.raises(SystemExit, match="not a callable"):
        _load_kernel_model(tmp_path / "not_callable.pkl")
    with pytest.raises(SystemExit, match="not a file"):
        _load_kernel_model(tmp_path / "missing.pkl")
