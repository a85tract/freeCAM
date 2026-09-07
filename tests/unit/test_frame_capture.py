"""Frames captured at a pause, saved per rank, and turned into standalone replay samples."""
from pathlib import Path
import json
import sys

import numpy as np

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
