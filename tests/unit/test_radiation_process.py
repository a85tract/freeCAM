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
    with pytest.raises(PhysicsError, match="takes original, replay:DIR"):
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
    # a TorchScript model is the slot's second answerer, over the same tables as tensors
    assert "pycam_rad_process_bind_model_v1" in source and "use ftorch" in source and "call torch_model_forward(model, in_t, out_t)" in source
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


def _torchscript_file(path: Path) -> Path:
    torch = pytest.importorskip("torch")

    class Zero(torch.nn.Module):
        def forward(self, x):
            return x * 0.0

    torch.jit.script(Zero()).save(str(path))
    return path


def test_a_torchscript_model_in_the_slot_is_bound_through_the_image_and_runs_the_driver_whole(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from freecam.physics.native_model import NativeModel
    from freecam.physics.segments import SegmentEvent

    model = NativeModel(_torchscript_file(tmp_path / "rad.pt"), shadow=True)
    stage = Radiation()
    stage.process = model
    assert stage.select_mode(None) == "segmented"
    assert stage.describe_process()["kind"] == "native-model"
    calls: list = []

    class _Entry:
        def __init__(self, f): self.f = f
        def __call__(self, *a): return self.f(*a)

    def counts(calls_ref, seconds_ref, first_ref):
        calls_ref._obj.value, seconds_ref._obj.value, first_ref._obj.value = 2, 0.5, 0.3
        return 0

    library = SimpleNamespace(
        pycam_stagehost_bind_v1=_Entry(lambda: 0), pycam_rad_bind_hosts_v1=_Entry(lambda: 0),
        pycam_rad_set_owner_v1=_Entry(lambda owns: 0),
        pycam_rad_process_bind_model_v1=_Entry(lambda path, length, shadow: calls.append((bytes(path)[:int(length)].decode(), int(length), int(shadow))) or 0),
        pycam_rad_process_counts_v1=_Entry(counts))

    class Runner:
        kernels = ("rad_rrtmg_sw", "rad_rrtmg_lw")
        runs_original = True

        def create(self, stage_name): return 3
        def start(self, context, mask): return SegmentEvent.DONE

    native = SimpleNamespace(library=library, segment_runner=lambda name: Runner() if name == Radiation.STAGE else None,
                             run_action=lambda *a, **k: pytest.fail("the runner hosts the slot"))
    stage.tend(None, SimpleNamespace(native=native))
    stage.tend(None, SimpleNamespace(native=native))
    assert calls == [(str(model.path), len(str(model.path)), 1)]          # bound once, in shadow
    described = stage.describe_process()
    assert (described["kind"], described["binding"], described["calls"], described["seconds"]) == ("native-model", "torchscript", 2, 0.5)


def test_the_spec_and_the_frame_descriptor_carry_the_slots_table() -> None:
    """The runner's Python pause at the slot hands the same 46 inputs and 12 outputs, in the same order."""
    import yaml

    from freecam.physics.radiation_process import TABLE_INPUTS, TABLE_OUTPUTS
    from freecam.pi_cam.segment_runner import runner_spec

    repo = Path(__file__).resolve().parents[2]
    spec = yaml.safe_load((repo / "native/pi_cam/pausable/radiation.yaml").read_text())
    slot = spec["process_slot"]
    assert slot["name"] == "radiation_process"
    assert [(n, r) for n, r in slot["inputs"]] == list(TABLE_INPUTS) and [(n, r) for n, r in slot["outputs"]] == list(TABLE_OUTPUTS)
    frames = yaml.safe_load((repo / "native/pi_cam/segment_frames.yaml").read_text())["kernels"]["radiation_process"]
    assert [(f["name"], f["rank"], f["intent"]) for f in frames] == \
        [(n, r, "in") for n, r in TABLE_INPUTS] + [(n, r, "out") for n, r in TABLE_OUTPUTS]
    assert all(f["dtype"] == "float64" for f in frames)
    manifest = runner_spec("cam_run1.radiation")
    assert manifest.process_slot == "radiation_process" and manifest.pause_names[-1] == "radiation_process"
    assert manifest.kernel_names == ("rad_rrtmg_sw", "rad_rrtmg_lw")          # the slot is not a kernel of the ledger
    runner = (repo / "native/pi_cam/support/pycam_radt_runner.F90").read_text()
    assert "kernel_radiation_process = 3_c_int" in runner and "call pycam_rad_process_frame(ptrs, ndims, shapes, dtypes, intents, ncol_out)" in runner
    assert "if (pycam_rad_process_prepare(state, pbuf, cam_in, coszrs, dosw, dolw)) then" in runner
    assert "call pycam_rad_process_finish(cam_out, dosw, dolw, qrs, qrl, fsns, fsnt, flns, flnt, fsds)" in runner


def _slot_frame(ncol: int, token: int):
    """A paused slot's frame as the Python runner would decode it: 46 inputs, 12 outputs, live lanes ncol."""
    from freecam.physics.radiation_process import TABLE_INPUTS, TABLE_OUTPUTS
    from freecam.physics.segments import FrameArgument, KernelFrame

    pcols, pver = 16, 30
    arguments = []
    for name, rank in TABLE_INPUTS:
        if rank == 0:
            array = np.array([float(ncol if name == "ncol" else 1.0)])       # the table's scalars: one-element arrays
        elif rank == 1:
            array = np.full(pcols, 0.5, order="F")
        elif rank == 2:
            array = np.full((pcols, pver + (1 if name in ("state_pint", "state_lnpint", "rstate_o3vmr", "rstate_pintmb", "rstate_tlev") else 0)), 2.0, order="F")
        else:
            array = np.full((pcols, pver, 57 if name == "state_q" else 3), 1e-6, order="F")
        arguments.append(FrameArgument(name, array, "in"))
    for name, rank in TABLE_OUTPUTS:
        arguments.append(FrameArgument(name, np.zeros((pcols, pver) if rank == 2 else pcols, order="F"), "out"))
    return KernelFrame(kernel="radiation_process", call_index=0, lchnk=1, ncol=ncol, substep=1, arguments=tuple(arguments), token=token)


class _SlotRunner:
    """A runner that pauses once per start at the process slot (kernel id 3) and runs the original on request."""

    kernels = ("rad_rrtmg_sw", "rad_rrtmg_lw", "radiation_process")
    runs_original = True

    def __init__(self, ncol: int = 12) -> None:
        self.ncol = ncol; self.frames: list = []; self.originals = 0; self.masks: list = []; self.token = 0

    def create(self, stage_name): return 5
    def start(self, context, mask):
        from freecam.physics.segments import SegmentEvent
        self.masks.append(dict(mask)); self.token += 1
        return SegmentEvent.NEEDS_PYTHON_KERNEL
    def frame(self, context):
        frame = _slot_frame(self.ncol, self.token); self.frames.append(frame); return frame
    def run_original(self, context, kernel): assert kernel == "radiation_process"; self.originals += 1
    def resume(self, context, kernel, token):
        from freecam.physics.segments import SegmentEvent
        assert (kernel, token) == ("radiation_process", self.token); return SegmentEvent.DONE
    def error(self, context): return ""
    def destroy(self, context): pass


def _slot_native(runner):
    from types import SimpleNamespace

    class _Entry:
        def __init__(self, f): self.f = f
        def __call__(self, *a): return self.f(*a)

    library = SimpleNamespace(pycam_stagehost_bind_v1=_Entry(lambda: 0), pycam_rad_bind_hosts_v1=_Entry(lambda: 0),
                              pycam_rad_set_owner_v1=_Entry(lambda owns: 0))
    return SimpleNamespace(library=library, segment_runner=lambda name: runner if name == Radiation.STAGE else None,
                           run_action=lambda *a, **k: pytest.fail("the runner hosts the slot"))


def test_a_python_model_answers_the_slot_at_the_runners_pause() -> None:
    from types import SimpleNamespace

    from freecam.physics.radiation_process import OUTPUTS, RadiationProcessModel

    seen: list = []

    def model(inputs):
        seen.append(inputs)
        ncol = int(inputs["ncol"]); pcols = 16
        answer = {}
        for name in OUTPUTS:
            out = np.zeros((pcols, 30), order="F") if name in ("qrs", "qrl") else np.zeros(pcols)
            out[:ncol] = 3.0
            answer[name] = out
        return answer

    stage = Radiation()
    stage.process = RadiationProcessModel(model, label="test:model")
    runner = _SlotRunner(ncol=12); native = _slot_native(runner)
    assert stage.select_mode(native) == "segmented"                    # the pause, not the walk
    assert stage.select_mode(SimpleNamespace(segment_runner=lambda name: None)) == "legacy-python"   # no runner: the walk
    stage.tend(None, SimpleNamespace(native=native))
    assert runner.masks == [{"rad_rrtmg_sw": False, "rad_rrtmg_lw": False, "radiation_process": True}]
    inputs = seen[0]
    assert isinstance(inputs["ncol"], float) and inputs["ncol"] == 12.0          # scalars as Python numbers
    assert isinstance(inputs["calday"], float) and float(inputs["calday"]) == 1.0
    assert inputs["state_t"].shape == (12, 30) and inputs["state_q"].shape == (12, 30, 57)    # live lanes only
    frame = runner.frames[0]
    qrs = frame.argument("qrs").array; fsnt = frame.argument("fsnt").array
    assert np.all(qrs[:12] == 3.0) and np.all(qrs[12:] == 0.0) and np.all(fsnt[:12] == 3.0) and np.all(fsnt[12:] == 0.0)
    assert stage.execution.segment_pauses == 1 and runner.originals == 0
    described = stage.describe_process()
    assert (described["kind"], described["pauses"], described["calls"]) == ("model-at-slot", 1, 1)


def test_the_original_branch_answers_the_slot_at_the_runners_pause() -> None:
    from types import SimpleNamespace

    from freecam.physics.radiation_process import OriginalProcess, load_process_model

    assert isinstance(load_process_model("original", rank=0), OriginalProcess)
    stage = Radiation()
    stage.process = OriginalProcess()
    runner = _SlotRunner(); native = _slot_native(runner)
    assert stage.select_mode(native) == "segmented"
    stage.tend(None, SimpleNamespace(native=native))
    assert runner.originals == 1 and stage.execution.segment_pauses == 1
    assert stage.describe_process()["kind"] == "original-at-slot"
    stage.execution_policy = "legacy-python"                                # the walk stays reachable on request
    assert stage.select_mode(native) == "legacy-python"


def test_the_image_runner_numbers_the_slot_after_the_kernels_and_decodes_its_frame() -> None:
    """ImageSegmentRunner over a fake radt image paused at the process slot: id 3, the table's names, resume and original by id 3."""
    from freecam.physics.radiation_process import TABLE_INPUTS, TABLE_OUTPUTS
    from freecam.physics.segments import SegmentEvent
    from freecam.pi_cam import segment_runner as runners

    names = [n for n, _ in TABLE_INPUTS] + [n for n, _ in TABLE_OUTPUTS]
    ranks = [r for _, r in TABLE_INPUTS] + [r for _, r in TABLE_OUTPUTS]
    arrays = [np.zeros(() if r == 0 else (16,) if r == 1 else (16, 30) if r == 2 else (16, 30, 3), order="F") for r in ranks]

    class Lib:
        def __init__(self):
            self.calls: list = []; self.mask = None; self.resumed = None; self.original = None
            for suffix in runners.ENTRY_SUFFIXES + ("original",):
                setattr(self, f"pycam_radt_{suffix}_v1", self._entry(f"pycam_radt_{suffix}_v1"))

        def _entry(self, name):
            lib = self

            class E:
                argtypes = None; restype = None
                def __call__(self, *args):
                    lib.calls.append(name)
                    if name.endswith("_create_v1"): args[0]._obj.value = 1; return 0
                    if name.endswith("_start_v1"): lib.mask = list(args[2]); args[3]._obj.value = 1; return 0
                    if name.endswith("_frame_v1"):
                        kernel, index, lchnk, ncol, substep, token, count, ptrs, ndims, shapes, dtypes, intents = args[1:]
                        kernel._obj.value = 3; index._obj.value = 0; lchnk._obj.value = 2; ncol._obj.value = 9; substep._obj.value = 1; token._obj.value = 4
                        assert count >= 58
                        for i, a in enumerate(arrays):
                            ptrs[i] = a.ctypes.data; ndims[i] = a.ndim
                            for axis, extent in enumerate(a.shape): shapes[runners.FRAME_MAX_RANK * i + axis] = extent
                            dtypes[i] = 1; intents[i] = 0 if i < 46 else 1
                        return 0
                    if name.endswith("_resume_v1"): lib.resumed = (args[1], args[2]); args[3]._obj.value = 0; return 0
                    if name.endswith("_original_v1"): lib.original = args[1]; return 0
                    return 0
            return E()

    lib = Lib()
    spec = runners.runner_spec("cam_run1.radiation")
    runner = runners.ImageSegmentRunner(lib, spec)
    assert runner.kernels == ("rad_rrtmg_sw", "rad_rrtmg_lw", "radiation_process") and runner.slots == 58
    context = runner.create("cam_run1.radiation")
    assert runner.start(context, {"rad_rrtmg_sw": False, "rad_rrtmg_lw": False, "radiation_process": True}) == SegmentEvent.NEEDS_PYTHON_KERNEL
    assert lib.mask == [0, 0, 1]
    frame = runner.frame(context)
    assert frame.kernel == "radiation_process" and frame.ncol == 9 and [a.name for a in frame.arguments] == names
    assert frame.argument("state_q").array.shape == (16, 30, 3) and frame.argument("qrs").intent == "out" and frame.argument("nstep").array.shape == ()
    runner.run_original(context, "radiation_process")
    assert lib.original == 3
    assert runner.resume(context, "radiation_process", frame.token) == SegmentEvent.DONE and lib.resumed == (3, 4)


def test_the_block_contract_decides_radiative_steps_and_loads_its_answerers(tmp_path: Path) -> None:
    from freecam.physics.radiation_process import (BLOCK_INPUTS, RadiationBlockModel, RadiationBlockReplay, load_block_model,
                                                   radiation_steps)

    # radiation.F90:240-246 with iradsw = iradlw = 2, irad_always = 0: step 0, then every odd step from 3
    assert [radiation_steps(n, 2, 2, 0)[0] for n in range(0, 8)] == [True, False, False, True, False, True, False, True]
    assert radiation_steps(5, 1, 3, 0) == (True, False) and radiation_steps(3, 4, 4, 10) == (True, True)
    assert "coszrs" not in BLOCK_INPUTS and "rstate_o3vmr" not in BLOCK_INPUTS and "ozone" in BLOCK_INPUTS
    module = tmp_path / "blk.py"; module.write_text("def f(inputs):\n    return {}\n")
    model = load_block_model(f"{module}:f", rank=0)
    assert isinstance(model, RadiationBlockModel) and model.block and model.describe()["kind"] == "block-model"
    with pytest.raises(PhysicsError, match="returned no"):
        model({"ncol": 1})
    stage = Radiation()
    stage.process = model
    assert stage.select_mode(None) == "python-driver"
    directory = tmp_path / "cap"; directory.mkdir()
    capture = RadiationProcessCapture()
    capture.record({"nstep": 2, "lchnk": 7, "ncol": 3}, {name: np.zeros(3) for name in OUTPUTS})
    capture.save(directory / "radiation_tend.rank-0000.npz")
    replay = load_block_model(f"replay:{directory}", rank=0)
    assert isinstance(replay, RadiationBlockReplay) and replay.block and replay.describe()["kind"] == "block-replay"


def test_the_block_trainers_features_are_block_inputs() -> None:
    import importlib.util
    import sys

    from freecam.physics.radiation_process import BLOCK_INPUTS

    root = Path(__file__).resolve().parents[2] / "examples/plugins/numba_kernels"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("train_rad_block", root / "train_rad_block.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    sources = {src for _, src, _, _, _, _ in module.LAYOUT}
    assert sources <= set(BLOCK_INPUTS), sources - set(BLOCK_INPUTS)
    assert module.NF == 33 * 30 + 30 + 8


def test_the_transformer_examples_take_the_slots_table_and_the_blocks_inputs() -> None:
    """The Numba transformer plugin's kernel has the slot's 46 inputs then 12 outputs by name, in the table's order;
    the block transformer's callable names only block inputs as scalars.  Read from source: importing either loads weights."""
    import ast

    from freecam.physics.radiation_process import BLOCK_INPUTS, TABLE_INPUTS, TABLE_OUTPUTS

    root = Path(__file__).resolve().parents[2] / "examples" / "plugins" / "numba_kernels"
    tree = ast.parse((root / "rad_tf_plugin.py").read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "radiation_table_kernel")
    assert [arg.arg for arg in kernel.args.args] == [name for name, _ in TABLE_INPUTS] + [f"o_{name}" for name, _ in TABLE_OUTPUTS]
    tree = ast.parse((root / "rad_block_tf.py").read_text())
    assert any(isinstance(node, ast.FunctionDef) and node.name == "radiation_block_tf" for node in tree.body)
    scalars = next(node.value for node in tree.body if isinstance(node, ast.Assign) and node.targets[0].id == "SCALARS")
    names = {ast.literal_eval(element) for element in scalars.args[0].elts}
    assert names <= set(BLOCK_INPUTS)
