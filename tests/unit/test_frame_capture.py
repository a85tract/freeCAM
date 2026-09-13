"""Frames captured at a pause, saved per rank, and turned into standalone replay samples."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

from freecam.physics.segments import FrameArgument, FrameCapture, KernelFrame

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import replay_pi_cam_frame_capture as replay  # noqa: E402


def _frame(ncol: int, token: int) -> tuple[KernelFrame, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(1.0, 1.0 + ncol) * 100.0
    q = np.full((ncol,), 0.01)
    out = np.zeros((ncol,))
    frame = KernelFrame(kernel="virtem", call_index=1, lchnk=1, ncol=ncol, substep=1, token=token, arguments=(
        FrameArgument("t", t, "in"), FrameArgument("q", q, "in"), FrameArgument("virtem", out, "out")))
    return frame, t, q, out


def test_a_capture_records_inputs_before_and_outputs_after_the_original(tmp_path: Path) -> None:
    capture = FrameCapture("virtem")
    capture.current_step = 3

    class Runner:
        def run_original(self, context, kernel):
            frame.argument("virtem").array[:] = 2.0 * frame.argument("t").array     # what the original does

    frame, t, q, out = _frame(4, token=9)
    answer = capture(frame, Runner(), 1)
    assert capture.calls == 1 and set(answer) == {"virtem"}
    assert np.all(answer["virtem"] == 2.0 * t) and np.all(out == 0.0)           # zeroed, like OriginalAtPause
    frame.write_back(answer)
    assert np.all(out == 2.0 * t)
    assert capture.meta == [{"step": 3, "ncol": 4, "token": 9, "kernel": "virtem"}]
    assert set(capture.inputs[0]) == {"t", "q"} and np.all(capture.inputs[0]["t"] == t)
    path = capture.save(tmp_path / "virtem.rank-0000.npz")
    with np.load(path) as bundle:
        assert sorted(bundle.files) == ["in/0/q", "in/0/t", "meta", "out/0/virtem"]
        assert json.loads(str(bundle["meta"]))[0]["step"] == 3


def test_replay_samples_follow_the_spec_layout(tmp_path: Path) -> None:
    from freecam.physics.spec import load_function_spec

    spec = load_function_spec(str(Path(__file__).resolve().parents[2] / "native/pi_cam/functions/virtem.yaml"))
    capture = FrameCapture("virtem")

    class Runner:
        def run_original(self, context, kernel):
            frame.argument("virtem").array[:] = frame.argument("t").array + frame.argument("q").array

    frame, t, q, out = _frame(3, token=1)
    frame.write_back(capture(frame, Runner(), 1))
    path = capture.save(tmp_path / "virtem.rank-0007.npz")
    (call,) = replay._load_capture(path)
    samples = list(replay._samples(spec, call))
    # an elemental function on sections: one scalar sample per live element, the result as `result`
    assert len(samples) == 3
    inputs, expected = samples[1]
    assert inputs == {"t": t[1], "q": q[1]} and set(expected) == {"result"} and expected["result"] == t[1] + q[1]


def test_a_frame_without_a_column_count_is_live_in_full() -> None:
    """A hook frame of a routine with no ncol dummy reports ncol 0: every element counts (capture 7343844)."""
    from freecam.physics.segments import FrameArgument, FrameCapture, KernelFrame, OriginalAtPause

    ps0 = np.linspace(1.0, 31.0, 31)
    xflx = np.zeros(31)
    frame = KernelFrame(kernel="fluxbelowinv", call_index=1, lchnk=1, ncol=0, substep=1, token=3,
                        arguments=(FrameArgument("ps0", ps0, "in"), FrameArgument("xflx", xflx, "out")))
    assert frame.batch() == {"ps0": pytest.approx(ps0)} and frame.batch()["ps0"].shape == (31,)

    class Runner:
        def run_original(self, cid, kernel):
            xflx[...] = 2.0 * ps0                       # the original writes the whole profile

    capture = FrameCapture("fluxbelowinv")
    answer = capture(frame, Runner(), 1)
    assert answer["xflx"].shape == (31,) and np.array_equal(answer["xflx"], 2.0 * ps0)
    assert np.all(xflx == 0.0)                          # zeroed after the snapshot, as OriginalAtPause does
    assert frame.write_back(answer) == ("xflx",) and np.array_equal(xflx, 2.0 * ps0)
    assert capture.inputs[0]["ps0"].shape == (31,) and capture.outputs[0]["xflx"].shape == (31,)
    assert OriginalAtPause()(frame, Runner(), 1)["xflx"].shape == (31,)


def test_the_stage_hands_a_capture_the_frame_itself_never_a_batch_wrapper() -> None:
    """A frame-taking model must reach the pause as-is (gate 7359288 captured nothing:
    the cloud stage wrapped it through Microphysics.frame_kernel as a batch model)."""

    from types import SimpleNamespace

    from freecam.physics.microphysics import Microphysics
    from freecam.physics.segments import OriginalAtPause, OriginalKernel

    stage = Microphysics()
    capture = FrameCapture("micro_mg_tend")
    stage.kernels["micro_mg_tend"] = capture
    resolved = stage._segment_kernels(native=None, runner=SimpleNamespace(runs_original=True))
    assert resolved["micro_mg_tend"] is capture         # not wrapped, not copied

    stage.kernels["micro_mg_tend"] = OriginalKernel()
    resolved = stage._segment_kernels(native=None, runner=SimpleNamespace(runs_original=True))
    assert isinstance(resolved["micro_mg_tend"], OriginalAtPause)

    def model(batch):                                   # a plain batch model still gets the adapter
        return {}

    model.takes_packed_batch = True
    stage.kernels["micro_mg_tend"] = model
    resolved = stage._segment_kernels(native=None, runner=SimpleNamespace(runs_original=True))
    assert resolved["micro_mg_tend"] is not model
