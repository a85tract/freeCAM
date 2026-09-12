"""The radiation process slot: capture, replay and a model answering the computing branch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from freecam.physics.errors import PhysicsError
from freecam.physics.radiation import PROCESS_INPUT_FIELDS, Radiation
from freecam.physics.radiation_process import (
    OUTPUTS,
    RadiationProcessCapture,
    RadiationProcessModel,
    RadiationReplay,
    load_process_model,
)


def _call(nstep: int, lchnk: int, ncol: int = 16, pver: int = 30):
    rng = np.random.default_rng(nstep * 100 + lchnk)
    inputs = {"nstep": nstep, "lchnk": lchnk, "ncol": ncol, "dt": 1800.0, "calday": 1.5, "dosw": True, "dolw": True,
              "coszrs": rng.random(ncol), "state_t": np.asfortranarray(rng.random((ncol, pver)) * 50 + 220),
              "rstate_o3vmr": np.asfortranarray(rng.random((ncol, pver)))}
    outputs = {name: (np.asfortranarray(rng.random((ncol, pver))) if name in ("qrs", "qrl") else rng.random(ncol))
               for name in OUTPUTS}
    return inputs, outputs


def test_a_capture_saves_every_call_and_a_replay_answers_the_same_step_and_chunk(tmp_path: Path) -> None:
    capture = RadiationProcessCapture()
    assert capture.records and not capture.answers
    calls = [_call(2, 0), _call(2, 1), _call(4, 0)]
    for inputs, outputs in calls:
        capture.record(inputs, outputs)
        outputs["qrs"][:] = -1.0                 # the record is a copy, not the live view
    assert capture.calls == 3 and capture.describe() == {"kind": "capture", "calls": 3}
    path = capture.save(tmp_path / "radiation_tend.rank-0007.npz")
    assert path.is_file()

    replay = RadiationReplay(tmp_path, rank=7)
    assert replay.answers and not replay.records and replay.describe()["records"] == 3
    inputs, _ = _call(4, 0)
    answer = replay(inputs)
    fresh_inputs, fresh_outputs = _call(4, 0)
    assert set(answer) == set(OUTPUTS)
    assert np.array_equal(answer["qrs"], fresh_outputs["qrs"]) and np.array_equal(answer["flwds"], fresh_outputs["flwds"])
    with pytest.raises(PhysicsError, match="no record for step 9"):
        replay({"nstep": 9, "lchnk": 0})
    with pytest.raises(PhysicsError, match="no radiation capture for rank 3"):
        RadiationReplay(tmp_path, rank=3)
    assert replay.calls == 1


def test_a_process_model_wraps_a_function_and_insists_on_every_output() -> None:
    def emulator(inputs):
        ncol, pver = inputs["state_t"].shape
        answer = {name: np.zeros(ncol) for name in OUTPUTS}
        answer["qrs"] = np.zeros((ncol, pver)); answer["qrl"] = np.zeros((ncol, pver))
        return answer

    model = RadiationProcessModel(emulator, label="test:emulator")
    inputs, _ = _call(2, 0)
    assert set(model(inputs)) == set(OUTPUTS) and model.calls == 1
    assert model.describe() == {"kind": "model", "function": "test:emulator", "calls": 1}
    incomplete = RadiationProcessModel(lambda inputs: {"qrs": 0}, label="bad")
    with pytest.raises(PhysicsError, match="returned no"):
        incomplete(inputs)
    with pytest.raises(PhysicsError, match="must be callable"):
        RadiationProcessModel(object(), label="x")


def test_the_loader_tells_a_replay_from_a_function(tmp_path: Path) -> None:
    capture = RadiationProcessCapture()
    capture.record(*_call(2, 0))
    capture.save(tmp_path / "radiation_tend.rank-0000.npz")
    assert isinstance(load_process_model(f"replay:{tmp_path}", rank=0), RadiationReplay)
    source = tmp_path / "emu.py"
    source.write_text("def emulate(inputs):\n    return {}\n")
    model = load_process_model(f"{source}:emulate", rank=0)
    assert isinstance(model, RadiationProcessModel) and model.label == "emu.py:emulate"
    with pytest.raises(PhysicsError, match="takes replay:DIR"):
        load_process_model("nonsense", rank=0)


def test_a_process_slot_forces_the_walk_and_is_described(tmp_path: Path) -> None:
    stage = Radiation()
    assert stage.process is None and stage.describe_process() is None
    stage.process = RadiationProcessCapture()
    assert stage.select_mode(None) == "legacy-python"
    assert stage.describe_process() == {"kind": "capture", "calls": 0}
    # the optics' inputs the slot records: buffer fields of other modules, two of them rank 3
    names = {row[0] for row in PROCESS_INPUT_FIELDS}
    assert {"DEI", "MU", "LAMBDAC", "ICIWP", "ICLWP", "DGNUMWET", "QAERWAT"} <= names
    assert all(row[1].endswith("_") and "_mp_" in row[1] for row in PROCESS_INPUT_FIELDS)
    assert {row[3] for row in PROCESS_INPUT_FIELDS if row[0] in ("DGNUMWET", "QAERWAT")} == {3}
