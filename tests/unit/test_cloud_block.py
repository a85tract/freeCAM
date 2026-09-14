"""The cloud stage's two compute blocks as contracts, and the Python driver around them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from freecam.physics import cloud_block as CB
from freecam.physics.cloud_macro_microphysics import PTEND, VIEW, CloudMacroMicrophysics
from freecam.physics.errors import PhysicsError


def test_the_contracts_name_every_field_once_and_carry_the_tendency() -> None:
    for block in (CB.MACRO_BLOCK, CB.MICRO_BLOCK):
        assert len(set(block.inputs)) == len(block.inputs)
        assert len(set(block.outputs)) == len(block.outputs)
        assert block.inputs[:4] == CB.SCALARS
        assert set(CB.TENDENCY_OUTPUTS) <= set(block.outputs)
        assert set(block.buffer_names) <= set(block.inputs) and set(block.buffer_names) <= set(block.outputs)
        for field in block.buffers:
            assert field.symbol.endswith("_") and "_mp_" in field.symbol, field
    assert {"det_s", "det_ice"} <= set(CB.MACRO_BLOCK.outputs)
    assert "det_s" not in CB.MICRO_BLOCK.outputs
    assert {"state_t", "state_q", "cam_in_landfrac", "dlf", "cmfmc", "CLD", "NAAI"} <= set(CB.MACRO_BLOCK.inputs)
    assert {"NPCCN", "PREC_STR", "QME", "CLDFSNOW", "NACON"} <= set(CB.MICRO_BLOCK.buffer_names)
    assert CB.MACRO_BLOCK.ptend_name == "macrop" and CB.MICRO_BLOCK.ptend_name == "cldwat"


def _answer(block: CB.BlockContract, ncol: int, pver: int, pcnst: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    lq = np.zeros(pcnst, dtype=np.int32)
    lq[[0, 2, 3, 7]] = 1
    q = np.zeros((ncol, pver, pcnst), order="F")
    q[:, :, lq != 0] = rng.random((ncol, pver, 4))
    answer = {"ptend_s": rng.random((ncol, pver)), "ptend_q": q, "ptend_ls": 1, "ptend_lq": lq}
    if block is CB.MACRO_BLOCK:
        answer["det_s"] = rng.random(ncol)
        answer["det_ice"] = rng.random(ncol)
    for name in block.buffer_names[:5]:
        answer[name] = rng.random((ncol, pver))
    return answer


def test_a_capture_saves_both_blocks_and_a_replay_answers_the_same_step_and_chunk(tmp_path: Path) -> None:
    ncol, pver, pcnst = 14, 30, 57
    capture = CB.CloudBlockCapture()
    inputs = {"nstep": 5, "lchnk": 1540, "ncol": ncol, "dt": 1800.0, "state_t": np.full((16, pver), 250.0), "CLD": np.zeros((16, pver))}
    recorded = {}
    for block in (CB.MACRO_BLOCK, CB.MICRO_BLOCK):
        before = capture.of(block).begin(inputs)
        assert before["state_t"].dtype == np.float32 and before["nstep"] == 5
        answer = _answer(block, ncol, pver, pcnst, seed=len(block.name))
        capture.of(block).finish(before, answer)
        recorded[block.name] = answer
    files = capture.save(tmp_path, rank=3)
    assert files == ["cloud_macro.rank-0003.npz", "cloud_micro.rank-0003.npz"]
    assert capture.describe() == {"kind": "capture", "macro_calls": 1, "micro_calls": 1}
    for block in (CB.MACRO_BLOCK, CB.MICRO_BLOCK):
        replay = CB.load_block_model(f"replay:{tmp_path}", block=block, rank=3)
        assert isinstance(replay, CB.BlockReplay) and replay.block is block
        answer = replay({"nstep": 5, "lchnk": 1540})
        for name, value in recorded[block.name].items():
            np.testing.assert_array_equal(np.asarray(answer[name]), np.asarray(value), err_msg=name)
        assert answer["ptend_q"].shape == (ncol, pver, pcnst)
        with pytest.raises(PhysicsError):
            replay({"nstep": 6, "lchnk": 1540})
        assert replay.describe()["calls"] == 1
    with pytest.raises(PhysicsError):
        CB.BlockReplay(tmp_path, 4, CB.MACRO_BLOCK)


def test_a_block_model_insists_on_the_tendency_and_the_macrophysics_detrainment() -> None:
    model = CB.BlockModel(lambda inputs: {"ptend_s": 1, "ptend_q": 1, "ptend_ls": 1, "ptend_lq": 1}, label="f", block=CB.MACRO_BLOCK)
    with pytest.raises(PhysicsError, match="det_s"):
        model({})
    micro = CB.BlockModel(lambda inputs: {"ptend_s": 1, "ptend_q": 1, "ptend_ls": 1, "ptend_lq": 1}, label="f", block=CB.MICRO_BLOCK)
    assert micro({})["ptend_s"] == 1 and micro.calls == 1
    with pytest.raises(PhysicsError):
        CB.BlockModel("not callable", label="x", block=CB.MICRO_BLOCK)   # type: ignore[arg-type]


def test_a_slot_or_the_capture_turns_the_stage_into_the_python_driver() -> None:
    stage = CloudMacroMicrophysics(whole_drivers=True)
    assert not stage.block_armed and stage.describe_process() is None
    stage.micro_process = CB.BlockModel(lambda inputs: {}, label="m", block=CB.MICRO_BLOCK)
    assert stage.block_armed and stage.select_mode() == "python-driver"
    described = stage.describe_process()
    assert described["process"] == {"kind": "original", "block": "process"}
    assert described["micro_process"]["kind"] == "block-model"
    stage.micro_process = None
    stage.block_capture = CB.CloudBlockCapture()
    assert stage.select_mode() == "python-driver" and "capture" in stage.describe_process()


class _Handles:
    """Enough of _MMHandles for the write-back: views by code and the tendency init it must call first."""

    def __init__(self, pcols=16, pver=30, pcnst=57):
        self.arrays = {VIEW["ptend_s"]: np.zeros((pcols, pver), order="F"), VIEW["ptend_q"]: np.zeros((pcols, pver, pcnst), order="F"),
                       VIEW["det_s"]: np.zeros(pcols), VIEW["det_ice"]: np.zeros(pcols)}
        self.inits: list[tuple] = []
        self.pcnst = pcnst

    def ptend_init(self, lchnk, which, name, *, ls, lq):
        self.inits.append((lchnk, which, name, ls, tuple(int(v) for v in lq)))

    def view(self, lchnk, code):
        assert self.inits, "the tendency object must be initialised before its arrays are written"
        return self.arrays[code]


def test_the_write_back_puts_a_blocks_answer_where_the_driver_leaves_it() -> None:
    stage = CloudMacroMicrophysics(whole_drivers=True)
    handles = _Handles()
    st = type("Runtime", (), {"handles": handles})()
    views = {name: np.zeros((16, 30), order="F") for name in CB.MACRO_BLOCK.buffer_names}
    answer = _answer(CB.MACRO_BLOCK, 14, 30, 57, seed=1)
    stage._write_block(st, 1540, 14, CB.MACRO_BLOCK, answer, views)
    assert handles.inits == [(1540, PTEND, "macrop", True, tuple(int(v) for v in answer["ptend_lq"]))]
    np.testing.assert_array_equal(handles.arrays[VIEW["ptend_s"]][:14], answer["ptend_s"])
    np.testing.assert_array_equal(handles.arrays[VIEW["ptend_q"]][:14], answer["ptend_q"])
    assert handles.arrays[VIEW["ptend_s"]][14:].sum() == 0.0
    np.testing.assert_array_equal(handles.arrays[VIEW["det_ice"]][:14], answer["det_ice"])
    written = [name for name in CB.MACRO_BLOCK.buffer_names if views[name].any()]
    assert written == list(CB.MACRO_BLOCK.buffer_names[:5])
