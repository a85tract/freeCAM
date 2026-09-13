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
    # installing a stage cloudpickles it and compares the payload across ranks (7417389): a replay pickles
    # without its rank's table and finds it again in this process
    import cloudpickle

    other = RadiationProcessCapture(); other.record(*_call(2, 5)); other.save(tmp_path / "b" / "radiation_tend.rank-0008.npz")
    replay8 = RadiationReplay(tmp_path / "b", rank=8)          # one rank per process: one table per directory
    payload7 = cloudpickle.dumps(RadiationReplay(tmp_path, rank=7))
    assert payload7 == cloudpickle.dumps(RadiationReplay(tmp_path, rank=7)) and len(payload7) < 2000
    revived = cloudpickle.loads(cloudpickle.dumps(replay8))
    assert np.array_equal(revived({"nstep": 2, "lchnk": 5})["qrs"], _call(2, 5)[1]["qrs"])


def test_a_process_model_wraps_a_function_and_insists_on_every_output() -> None:
    def emulator(inputs):
        ncol, pver = inputs["state_t"].shape
        answer = {name: np.zeros(ncol) for name in OUTPUTS}
        answer["qrs"] = np.zeros((ncol, pver)); answer["qrl"] = np.zeros((ncol, pver))
        return answer

    model = RadiationProcessModel(emulator, label="test:emulator")
    inputs, _ = _call(2, 0)
    assert set(model(inputs)) == set(OUTPUTS) and model.calls == 1
    described = model.describe()
    assert described["kind"] == "model" and described["function"] == "test:emulator" and described["calls"] == 1
    assert described["seconds"] >= described["first_call_seconds"] > 0.0
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


def test_a_capture_can_keep_every_nth_radiative_step() -> None:
    capture = RadiationProcessCapture(every=3)
    for nstep in (1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7):          # three chunks a radiative step
        capture.record(*_call(nstep, nstep % 3))
    assert capture.calls == 6 and capture.skipped == 6                 # steps 1 and 7 kept, 3 and 5 skipped
    assert {int(record["nstep"]) for record in capture.inputs} == {1, 7}
    assert capture.describe() == {"kind": "capture", "calls": 6, "every": 3, "skipped": 6}


def test_the_fortran_slots_table_is_the_python_one() -> None:
    """pycam_rad_process hands a plugin its inputs in TABLE_INPUTS' order, and takes TABLE_OUTPUTS back."""
    import re

    from freecam.physics.radiation_process import TABLE_INPUTS, TABLE_OUTPUTS

    source = (Path(__file__).resolve().parents[2] / "native/pi_cam/support/pycam_rad_process.F90").read_text()
    def names(parameter):
        block = re.search(parameter + r"\(\w+\) = \[character\(len=16\) :: &\n(.*?)\]", source, re.S).group(1)
        return re.findall(r"'([a-z0-9_]+)'", block)
    assert names("table_inputs") == [name for name, _ in TABLE_INPUTS]
    assert names("table_outputs") == [name for name, _ in TABLE_OUTPUTS]
    assert "integer, parameter :: n_in = 46, n_out = 12" in source and len(TABLE_INPUTS) == 46 and len(TABLE_OUTPUTS) == 12
    # the runner asks the slot at the top of the radiative branch, and continues after it when it answered
    runner = (Path(__file__).resolve().parents[2] / "native/pi_cam/support/pycam_radt_runner.F90").read_text()
    assert "if (pycam_rad_process_answer(state, pbuf, cam_in, cam_out, coszrs, dosw, dolw, qrs, qrl, fsns, fsnt, flns, flnt, fsds)) then" in runner
    assert "use pycam_rad_process, only: pycam_rad_process_answer" in runner


def test_a_table_kernel_compiles_to_the_slots_interface_and_writes_the_outputs() -> None:
    pytest.importorskip("numba")
    import numpy as np

    from freecam.physics.native_model import NativePlugin
    from freecam.physics.numba_kernel import call_plugin_from_python
    from freecam.physics.radiation_process import TABLE_INPUTS, TABLE_OUTPUTS, compile_radiation_plugin

    def kernel(nstep, lchnk, ncol, calday, dosw, dolw, coszrs, clat, clon, state_t, state_pmid, state_pint, state_pdel,
               state_lnpint, state_lnpmid, state_q, cld, cldfsnow, dei, mu, lambdac, iciwp, iclwp, des, icswp, dgnumwet, qaerwat,
               cam_in_lwup, cam_in_asdir, cam_in_asdif, cam_in_aldir, cam_in_aldif, rstate_h2ovmr, rstate_o3vmr, rstate_co2vmr,
               rstate_ch4vmr, rstate_o2vmr, rstate_n2ovmr, rstate_cfc11vmr, rstate_cfc12vmr, rstate_cfc22vmr, rstate_ccl4vmr,
               rstate_pmidmb, rstate_pintmb, rstate_tlay, rstate_tlev,
               o_qrs, o_qrl, o_fsns, o_fsnt, o_flns, o_flnt, o_fsds, o_sols, o_soll, o_solsd, o_solld, o_flwds):
        n = int(ncol)
        for i in range(n):
            for k in range(state_t.shape[1]):
                o_qrs[i, k] = state_t[i, k] * 2.0 + state_q[i, k, 0]
                o_qrl[i, k] = -rstate_o3vmr[i, k]
            o_fsnt[i] = coszrs[i] * 100.0 + nstep
            o_flwds[i] = cam_in_lwup[i] + dgnumwet[i, 0, 2]

    plugin = compile_radiation_plugin(kernel)
    assert isinstance(plugin, NativePlugin) and plugin.describe()["kernel"] == "radiation_process" and plugin.address
    rng = np.random.default_rng(1)
    shapes = {0: (), 1: (16,), 2: (16, 30), 3: (16, 30, 3)}
    inputs = []
    for name, rank in TABLE_INPUTS:
        shape = (16, 30, 57) if name == "state_q" else shapes[rank]
        inputs.append(np.asfortranarray(rng.random(shape)) if rank else float(rng.random()))
    inputs[TABLE_INPUTS.index(("ncol", 0))] = 14.0
    outputs = [np.zeros((16, 30), order="F") if rank == 2 else np.zeros(16) for _, rank in TABLE_OUTPUTS]
    assert call_plugin_from_python(plugin, inputs, outputs) == 0
    t, q, o3, cosz, nstep = (inputs[[n for n, _ in TABLE_INPUTS].index(k)] for k in ("state_t", "state_q", "rstate_o3vmr", "coszrs", "nstep"))
    o = dict(zip([n for n, _ in TABLE_OUTPUTS], outputs))
    assert np.array_equal(o["qrs"][:14], t[:14] * 2.0 + q[:14, :, 0]) and np.array_equal(o["qrl"][:14], -o3[:14]) and (o["qrs"][14:] == 0).all()
    assert np.array_equal(o["fsnt"][:14], cosz[:14] * 100.0 + nstep) and o["fsns"].sum() == 0.0


def test_a_compiled_plugin_in_the_slot_runs_the_driver_whole_through_the_runner() -> None:
    pytest.importorskip("numba")
    import ctypes
    from types import SimpleNamespace

    from freecam.physics.radiation_process import compile_radiation_plugin
    from freecam.physics.segments import SegmentEvent

    def zeros(*args):
        pass

    plugin = compile_radiation_plugin(zeros, shadow=True)
    stage = Radiation()
    stage.process = plugin
    assert stage.select_mode(None) == "segmented"                 # the runner, no pause armed
    assert stage.describe_process()["kind"] == "native-plugin"
    calls: list = []

    class _Entry:
        """A library entry: a callable object, so ctypes' restype/argtypes can be set on it."""

        def __init__(self, f): self.f = f
        def __call__(self, *a): return self.f(*a)

    def counts(calls_ref, seconds_ref, first_ref):
        calls_ref._obj.value, seconds_ref._obj.value, first_ref._obj.value = 3, 0.25, 0.1
        return 0

    library = SimpleNamespace(
        pycam_stagehost_bind_v1=_Entry(lambda: calls.append(("stage_hosts",)) or 0),
        pycam_rad_bind_hosts_v1=_Entry(lambda: calls.append(("rad_hosts",)) or 0),
        pycam_rad_set_owner_v1=_Entry(lambda owns: calls.append(("owner", owns.value if isinstance(owns, ctypes.c_int) else owns)) or 0),
        pycam_rad_process_bind_v1=_Entry(lambda address, shadow: calls.append(("plugin", int(getattr(address, "value", address)), int(shadow))) or 0),
        pycam_rad_process_counts_v1=_Entry(counts))
    started: list = []

    class Runner:
        kernels = ("rad_rrtmg_sw", "rad_rrtmg_lw")
        runs_original = True

        def create(self, stage_name): return 7
        def start(self, context, mask): started.append(dict(mask)); return SegmentEvent.DONE

    native = SimpleNamespace(library=library, segment_runner=lambda name: Runner() if name == Radiation.STAGE else None,
                             run_action=lambda *a, **k: pytest.fail("the runner, not the whole action, hosts the slot"))
    stage.tend(None, SimpleNamespace(native=native))
    stage.tend(None, SimpleNamespace(native=native))
    # the hosts and the plugin are bound before the first start; the plugin once; the resume half then owns the result
    assert calls[:3] == [("stage_hosts",), ("rad_hosts",), ("plugin", plugin.address, 1)]
    assert calls.count(("plugin", plugin.address, 1)) == 1 and calls.count(("owner", 1)) == 2
    assert started == [{"rad_rrtmg_sw": False, "rad_rrtmg_lw": False}] * 2   # whole: no pause armed
    assert stage.execution.native_segment_calls == 2 and stage.execution.segment_pauses == 0
    described = stage.describe_process()
    assert (described["kind"], described["calls"], described["seconds"], described["first_call_seconds"]) == \
        ("native-plugin", 3, 0.25, 0.1)
