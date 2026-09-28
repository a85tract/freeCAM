"""The segmented protocol driven against a fake runner: pauses, frames, write-back, lifecycle."""

from __future__ import annotations

import numpy as np
import pytest

from freecam.physics.errors import PhysicsError
from freecam.physics.segments import (
    FrameArgument, KernelFrame, SegmentEvent, SegmentedStage,
)

PCOLS, PVER = 8, 4


class FakeRunner:
    """A stage of two chunks and two substeps calling kernels a then b each substep.

    Kernel a takes x (in) and writes y (out) and z (inout); b takes y.  A
    replaced kernel pauses the program; an original one runs a fixed Fortran
    stand-in (y = 2x, z += 1 for a; y *= 3 for b), so the program's effect
    is checkable either way.  Tokens guard resume: the wrong kernel or a
    stale token is refused, as the image will refuse it.
    """

    kernels = ("a", "b")                    # what the image's runner would advertise

    def __init__(self) -> None:
        self.x = {10: np.arange(PCOLS * PVER, dtype=float).reshape(PCOLS, PVER, order="F").copy(order="F"),
                  11: np.full((PCOLS, PVER), 5.0, order="F")}
        self.y = {c: np.zeros((PCOLS, PVER), order="F") for c in (10, 11)}
        self.z = {c: np.zeros((PCOLS, PVER), order="F") for c in (10, 11)}
        self.ncol = {10: 6, 11: 5}
        self.log: list[tuple] = []
        self.contexts: dict[int, dict] = {}
        self.next_context = 1
        self.next_token = 100
        self.fail_at: tuple[int, int] | None = None     # (lchnk, substep) where "Fortran" errors

    # -- runner interface ---------------------------------------------------
    def create(self, stage):
        cid = self.next_context; self.next_context += 1
        self.contexts[cid] = {"stage": stage, "pc": None, "mask": {}, "token": None, "calls": 0}
        self.log.append(("create", stage)); return cid

    def start(self, cid, mask):
        ctx = self.contexts[cid]
        ctx["mask"] = dict(mask); ctx["pc"] = (0, 0, "a"); ctx["calls"] = 0
        self.log.append(("start", dict(mask)))
        return self._advance(cid)

    def frame(self, cid):
        ctx = self.contexts[cid]
        chunk_i, substep, kernel = ctx["pc"]; lchnk = (10, 11)[chunk_i]
        ctx["token"] = self.next_token; self.next_token += 1
        args = {"a": (FrameArgument("x", self.x[lchnk], "in"), FrameArgument("y", self.y[lchnk], "out"),
                      FrameArgument("z", self.z[lchnk], "inout")),
                "b": (FrameArgument("y", self.y[lchnk], "inout"),)}[kernel]
        self.log.append(("frame", kernel, lchnk, substep))
        return KernelFrame(kernel=kernel, call_index=ctx["calls"], lchnk=lchnk, ncol=self.ncol[lchnk],
                           substep=substep, arguments=args, token=ctx["token"])

    def resume(self, cid, kernel, token):
        ctx = self.contexts[cid]
        if ctx["pc"] is None or ctx["pc"][2] != kernel or token != ctx["token"]:
            self.log.append(("refused-resume", kernel, token)); return SegmentEvent.ERROR
        ctx["token"] = None
        self.log.append(("resume", kernel))
        ctx["pc"] = self._next(ctx["pc"]); ctx["calls"] += 1
        return self._advance(cid)

    def error(self, cid):
        return "fake runner error"

    def reset(self, cid):
        self.contexts[cid]["pc"] = None

    def destroy(self, cid):
        self.log.append(("destroy", cid)); self.contexts.pop(cid)

    # -- the "Fortran" program ------------------------------------------------
    def _next(self, pc):
        chunk_i, substep, kernel = pc
        if kernel == "a": return (chunk_i, substep, "b")
        if substep == 0: return (chunk_i, 1, "a")
        if chunk_i == 0: return (1, 0, "a")
        return None

    def _advance(self, cid):
        ctx = self.contexts[cid]
        while ctx["pc"] is not None:
            chunk_i, substep, kernel = ctx["pc"]; lchnk = (10, 11)[chunk_i]; n = self.ncol[lchnk]
            if self.fail_at == (lchnk, substep) and kernel == "b":
                return SegmentEvent.ERROR
            if ctx["mask"].get(kernel):
                return SegmentEvent.NEEDS_PYTHON_KERNEL
            if kernel == "a":
                self.y[lchnk][:n] = 2.0 * self.x[lchnk][:n]; self.z[lchnk][:n] += 1.0
            else:
                self.y[lchnk][:n] *= 3.0
            self.log.append(("fortran", kernel, lchnk, substep))
            ctx["pc"] = self._next(ctx["pc"]); ctx["calls"] += 1
        return SegmentEvent.DONE


def _original_a(batch):
    return {"y": 2.0 * batch["x"], "z": batch["z"] + 1.0}


def test_the_runner_pauses_only_at_the_replaced_kernel_in_program_order() -> None:
    runner = FakeRunner()
    stage = SegmentedStage("cam_run1.widgets", runner)
    stage.run({"a": _original_a, "b": None})
    frames = [e for e in runner.log if e[0] == "frame"]
    assert frames == [("frame", "a", 10, 0), ("frame", "a", 10, 1), ("frame", "a", 11, 0), ("frame", "a", 11, 1)]
    assert [e for e in runner.log if e[0] == "fortran"] == [
        ("fortran", "b", 10, 0), ("fortran", "b", 10, 1), ("fortran", "b", 11, 0), ("fortran", "b", 11, 1)]
    assert stage.idle and stage.counters.pauses == 4 and stage.counters.resumes == 4
    assert stage.counters.model_calls == 4 and stage.counters.starts == 1
    assert stage.counters.crossings == 1 + 1 + 4 * 2       # create, start, (frame + resume) x 4


def test_a_model_that_computes_what_the_original_did_leaves_the_same_arrays() -> None:
    reference = FakeRunner(); SegmentedStage("s", reference).run({"a": lambda b: b, "b": None}) if False else None
    whole = FakeRunner()
    ctx = whole.create("s"); assert whole.start(ctx, {"a": False, "b": False}) == SegmentEvent.DONE
    segmented = FakeRunner()
    SegmentedStage("s", segmented).run({"a": _original_a, "b": None})
    for c in (10, 11):
        assert np.array_equal(whole.y[c], segmented.y[c]) and np.array_equal(whole.z[c], segmented.z[c])
        assert np.all(segmented.y[c][segmented.ncol[c]:] == 0.0)          # padding lanes untouched


def test_the_frame_hands_the_model_live_lanes_only_and_writes_back_exactly() -> None:
    runner = FakeRunner()
    seen = []

    def model(batch):
        seen.append({k: v.shape for k, v in batch.items()})
        return _original_a(batch)

    SegmentedStage("s", runner).run({"a": model, "b": None})
    assert seen[0] == {"x": (6, PVER), "z": (6, PVER)} and seen[2] == {"x": (5, PVER), "z": (5, PVER)}
    assert "y" not in seen[0]                                              # an output is not an input


@pytest.mark.parametrize("answer, message", [
    (lambda b: {"y": 2.0 * b["x"]}, "also writes \\['z'\\]"),
    (lambda b: {"y": 2.0 * b["x"], "z": np.zeros((2, 2))}, "must be \\(6, 4\\)"),
    (lambda b: {"y": (2.0 * b["x"]).astype(np.float32), "z": b["z"]}, "must be float64"),
])
def test_a_bad_answer_taints_the_stage_and_frees_the_context(answer, message) -> None:
    runner = FakeRunner()
    stage = SegmentedStage("s", runner)
    with pytest.raises(PhysicsError, match=message):
        stage.run({"a": answer, "b": None})
    assert stage.tainted is not None and stage.context is None and not runner.contexts
    with pytest.raises(PhysicsError, match="tainted"):
        stage.run({"a": _original_a, "b": None})


def test_a_model_exception_taints_the_stage_too() -> None:
    runner = FakeRunner()
    stage = SegmentedStage("s", runner)

    def boom(batch):
        raise RuntimeError("model blew up")

    with pytest.raises(RuntimeError, match="model blew up"):
        stage.run({"a": boom, "b": None})
    assert "model blew up" in stage.tainted and not runner.contexts


def test_the_runner_s_own_error_is_reported_and_the_context_freed() -> None:
    runner = FakeRunner(); runner.fail_at = (11, 0)
    stage = SegmentedStage("s", runner)
    with pytest.raises(PhysicsError, match="the runner failed after 3 pause\\(s\\) this run: fake runner error"):
        stage.run({"a": _original_a, "b": None})
    assert not runner.contexts and stage.tainted


def test_a_stale_or_misdirected_resume_is_refused_by_the_runner() -> None:
    runner = FakeRunner()
    cid = runner.create("s")
    assert runner.start(cid, {"a": True, "b": False}) == SegmentEvent.NEEDS_PYTHON_KERNEL
    frame = runner.frame(cid)
    assert runner.resume(cid, "b", frame.token) == SegmentEvent.ERROR                # wrong kernel
    assert runner.resume(cid, "a", frame.token + 1) == SegmentEvent.ERROR            # stale token
    assert runner.resume(cid, "a", frame.token) == SegmentEvent.NEEDS_PYTHON_KERNEL  # the right one


def test_nothing_replaced_is_refused_and_the_context_is_kept_between_steps() -> None:
    runner = FakeRunner()
    stage = SegmentedStage("s", runner)
    with pytest.raises(PhysicsError, match="nothing is replaced"):
        stage.run({"a": None, "b": None})
    stage.run({"a": _original_a, "b": None})
    stage.run({"a": _original_a, "b": None})
    assert [e for e in runner.log if e[0] == "create"] == [("create", "s")]      # one context, two steps
    assert stage.generation == 2
    stage.close()
    assert not runner.contexts


def test_an_empty_output_takes_any_empty_answer_and_nothing_else() -> None:
    """A field the configuration never registered is packed as a zero-size
    array; a chunk may pack no cloudy column at all.  Neither leaves the
    model anything to write, and the frame does not hold it to the extents
    of an array with nothing in it."""

    import numpy as np

    from freecam.physics.segments import FrameArgument, KernelFrame

    frame = KernelFrame(
        kernel="k", call_index=1, lchnk=1, ncol=0, substep=1, token=1,
        arguments=(FrameArgument("nothing", np.zeros((0, 0), order="F"), "out"),
                   FrameArgument("empty_rows", np.zeros((0, 30), order="F"), "out")))
    assert frame.write_back({"nothing": np.zeros((0, 30)), "empty_rows": np.zeros((0, 30))}) == ("nothing", "empty_rows")
    with pytest.raises(PhysicsError, match="no storage"):
        frame.write_back({"nothing": np.zeros((2, 30)), "empty_rows": np.zeros((0, 30))})


# -- batched mode: every chunk waits at the kernel together ---------------------

import dataclasses  # noqa: E402

from freecam.physics.segments import ByChunk, ChunkBatch  # noqa: E402


class BatchedFakeRunner(FakeRunner):
    """The fake stage with the runner's batched mode: a slot per chunk, one live at a time.

    A slot runs its own chunk only; its program counter and pause token are kept
    while another slot is live.  Kernel a is also handed the chunk's column count
    ``n`` (a chunk's own) and a time step ``dt`` (every chunk's alike).
    """

    offers_batch = True

    def __init__(self) -> None:
        super().__init__()
        self.batched = False

    def batch_begin(self, cid, mask):
        ctx = self.contexts[cid]
        ctx.update(mask=dict(mask), pc=None, token=None, calls=0, live=None,
                   slot_pc={0: (0, 0, "a"), 1: (1, 0, "a")}, slot_token={0: None, 1: None})
        self.batched = True
        self.log.append(("batch_begin",))
        return 2

    def batch_select(self, cid, slot):
        ctx = self.contexts[cid]
        if ctx["live"] is not None:
            ctx["slot_pc"][ctx["live"]], ctx["slot_token"][ctx["live"]] = ctx["pc"], ctx["token"]
        ctx["live"], ctx["pc"], ctx["token"] = slot, ctx["slot_pc"][slot], ctx["slot_token"][slot]
        self.log.append(("select", slot))

    def batch_advance(self, cid):
        return self._advance(cid)

    def batch_end(self, cid):
        ctx = self.contexts[cid]
        ctx["slot_pc"][ctx["live"]] = ctx["pc"]
        assert all(pc is None for pc in ctx["slot_pc"].values()), ctx["slot_pc"]
        self.batched = False
        self.log.append(("batch_end",))

    def frame(self, cid):
        # a pause has one token, however often its frame is read
        ctx = self.contexts[cid]
        token = ctx["token"]
        frame = super().frame(cid)
        if token is not None:
            ctx["token"] = token
            frame = dataclasses.replace(frame, token=token)
        if frame.kernel == "a":
            extra = (FrameArgument("n", np.array(frame.ncol, dtype=np.int32), "in"),
                     FrameArgument("dt", np.array(1800.0), "in"))
            frame = dataclasses.replace(frame, arguments=frame.arguments + extra)
        return frame

    def _next(self, pc):
        chunk_i, substep, kernel = pc
        if kernel == "a":
            return (chunk_i, substep, "b")
        if substep == 0:
            return (chunk_i, 1, "a")
        if chunk_i == 0 and not self.batched:
            return (1, 0, "a")
        return None


class StackedOriginal:
    """A chunk-batch model computing kernel a's original on the stacked lanes, recording what it saw."""

    takes_chunk_batches = True

    def __init__(self) -> None:
        self.seen: list[ChunkBatch] = []

    def __call__(self, batch: ChunkBatch):
        self.seen.append(batch)
        return _original_a(batch.inputs)


def _whole() -> FakeRunner:
    whole = FakeRunner()
    assert whole.start(whole.create("s"), {"a": False, "b": False}) == SegmentEvent.DONE
    return whole


def test_batched_chunks_are_answered_in_one_call_a_round_and_leave_what_the_original_did() -> None:
    runner, model = BatchedFakeRunner(), StackedOriginal()
    stage = SegmentedStage("s", runner)
    stage.run({"a": model, "b": None}, batched=True)
    whole = _whole()
    for c in (10, 11):
        assert np.array_equal(whole.y[c], runner.y[c]) and np.array_equal(whole.z[c], runner.z[c])
    # two rounds (a's two substeps), each one call for both chunks: 6 + 5 columns stacked
    assert stage.counters.model_calls == 2 and stage.counters.pauses == 4 and stage.counters.resumes == 4
    first = model.seen[0]
    assert first.rows == (6, 5) and first.lchnks == (10, 11) and first.columns == 11
    assert first.inputs["x"].shape == (11, PVER) and first.stacked == {"x", "z"}
    assert np.array_equal(first.inputs["x"][6:], np.full((5, PVER), 5.0))
    assert float(first.inputs["dt"]) == 1800.0 and "n" not in first.inputs           # alike: once
    assert [int(v) for v in first.per_chunk["n"]] == [6, 5]                            # a chunk's own
    assert stage.idle and ("batch_end",) in runner.log


def test_the_chunks_run_interleaved_and_the_last_parked_resumes_first() -> None:
    runner = BatchedFakeRunner()
    SegmentedStage("s", runner).run({"a": StackedOriginal(), "b": None}, batched=True)
    steps = [e for e in runner.log if e[0] in ("select", "frame", "resume")]
    assert steps[:6] == [("select", 0), ("frame", "a", 10, 0), ("select", 1), ("frame", "a", 11, 0),
                         ("select", 1), ("frame", "a", 11, 0)]
    assert steps[6] == ("resume", "a")                                                 # slot 1 first, it is live


def test_a_per_chunk_model_is_called_for_each_chunk_and_by_chunk_matches_it() -> None:
    plain, wrapped = BatchedFakeRunner(), BatchedFakeRunner()
    calls = []

    def per_chunk(batch):
        calls.append(batch["x"].shape[0])
        return _original_a(batch)

    first = SegmentedStage("s", plain)
    first.run({"a": per_chunk, "b": None}, batched=True)
    assert first.counters.model_calls == 4 and calls == [5, 6, 5, 6]                   # at resume, last first
    second = SegmentedStage("s", wrapped)
    second.run({"a": ByChunk(per_chunk), "b": None}, batched=True)
    assert second.counters.model_calls == 2
    assert calls[4:] == [6, 5, 6, 5]                                                   # chunk order inside a call
    for c in (10, 11):
        assert np.array_equal(plain.y[c], wrapped.y[c]) and np.array_equal(plain.z[c], wrapped.z[c])


def test_a_frame_taking_model_answers_each_chunk_on_its_live_frame() -> None:
    runner = BatchedFakeRunner()
    seen = []

    class AtFrame:
        takes_frame = True

        def __call__(self, frame, runner_, context):
            seen.append(frame.lchnk)
            n = frame.ncol
            return {"y": 2.0 * frame.argument("x").array[:n], "z": frame.argument("z").array[:n] + 1.0}

    SegmentedStage("s", runner).run({"a": AtFrame(), "b": None}, batched=True)
    assert seen == [11, 10, 11, 10]
    whole = _whole()
    assert all(np.array_equal(whole.y[c], runner.y[c]) for c in (10, 11))


def test_a_chunk_batch_answer_of_the_wrong_length_taints_the_stage() -> None:
    runner = BatchedFakeRunner()

    class Short:
        takes_chunk_batches = True

        def __call__(self, batch):
            answer = _original_a(batch.inputs)
            return {name: value[:-1] for name, value in answer.items()}

    stage = SegmentedStage("s", runner)
    with pytest.raises(PhysicsError, match="answered for every stacked column \\(11\\)"):
        stage.run({"a": Short(), "b": None}, batched=True)
    assert stage.tainted and not runner.contexts


def test_batched_chunks_need_a_runner_with_a_batched_mode() -> None:
    stage = SegmentedStage("s", FakeRunner())
    with pytest.raises(PhysicsError, match="no batched mode"):
        stage.run({"a": _original_a, "b": None}, batched=True)
