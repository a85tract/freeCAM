"""The stage-7 segment runner: the generated module is fresh, and its Python binding decodes frames."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

MODULE = REPO / "native/pi_cam/support/pycam_stage7_runner.F90"


def test_the_committed_runner_is_what_the_generator_writes_and_the_source_is_where_it_was() -> None:
    import generate_pi_cam_stage7_runner as gen

    for name, (first, last, digest) in gen.ANCHORS.items():
        assert gen.range_digest(name, first, last) == digest, f"{name}:{first}-{last} moved under the runner"
    assert gen.render_module() == MODULE.read_text()


def test_every_argument_of_the_kernel_has_a_home_in_the_frame_in_call_order() -> None:
    import generate_pi_cam_stage7_runner as gen
    from freecam.pi_cam.kernel_codegen import load_direct_kernels
    from freecam.physics.macrophysics import Macrophysics

    arguments = gen.kernel_arguments()
    names = [a["field"].split(".", 1)[1] for a in arguments]
    assert names == [a.field.split(".", 1)[1] for k in load_direct_kernels(Macrophysics.DESCRIPTORS)
                     if k.name == "mmacro_pcond" for a in k.arguments]
    assert set(names) == set(gen.FRAME_SOURCES)                     # no argument without a slot, no stray slot
    text = MODULE.read_text()
    assert f"frame_slots = {len(names)}" in text
    for index, name in enumerate(names, start=1):
        expression, rank = gen.FRAME_SOURCES[name]
        helper = "slot2_or" if "|" in expression else {0: "scalar_slot", 1: "slot1", 2: "slot2"}[rank]
        assert f"call {helper}({index}, " in text
    # the original call carries the same arguments in the same order
    call = text[text.index("call mmacro_pcond("):text.index("end subroutine original_pcond")]
    assert call.count(",") >= len(names) - 1
    # no callback from Fortran into Python anywhere in the module
    assert "c_funptr" not in text.lower() and "c_f_procpointer" not in text.lower()


class _FakeStageSevenLibrary:
    """Answers the seven entries the way the module does, for one pause."""

    def __init__(self) -> None:
        self.calls: list = []
        self.arrays = {"t0": np.zeros((8, 4), order="F"), "s_tendout": np.zeros((8, 4), order="F")}
        self.token = 41
        for name in ("pycam_stage7_create_v1", "pycam_stage7_start_v1", "pycam_stage7_frame_v1",
                     "pycam_stage7_resume_v1", "pycam_stage7_error_v1", "pycam_stage7_reset_v1",
                     "pycam_stage7_destroy_v1"):
            setattr(self, name, _Entry(self, name))


class _Entry:
    def __init__(self, lib, name):
        self.lib, self.name = lib, name
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        lib = self.lib
        lib.calls.append((self.name, args[:3]))
        if self.name == "pycam_stage7_create_v1":
            args[0]._obj.value = 1; return 0
        if self.name == "pycam_stage7_start_v1":
            lib.mask = list(args[2]); args[3]._obj.value = 1 if lib.mask[0] else 0; return 0
        if self.name == "pycam_stage7_frame_v1":
            (context, kernel, index, lchnk, ncol, substep, token, n, ptrs, ndims, shapes, dtypes, intents) = args
            kernel._obj.value, index._obj.value, lchnk._obj.value = 1, 3, 10
            ncol._obj.value, substep._obj.value, token._obj.value = 6, 2, lib.token
            for i in range(n):                                     # every slot: a 2-D double, intent in
                ptrs[i] = lib.arrays["t0"].ctypes.data; ndims[i] = 2
                shapes[3 * i], shapes[3 * i + 1] = 8, 4; dtypes[i] = 1; intents[i] = 0
            names = lib.names
            ptrs[names.index("s_tendout")] = lib.arrays["s_tendout"].ctypes.data
            intents[names.index("s_tendout")] = 1
            ptrs[names.index("t0")] = lib.arrays["t0"].ctypes.data; intents[names.index("t0")] = 2
            ndims[names.index("ncol")] = 0; dtypes[names.index("ncol")] = 2
            lib.ncol_cell = np.array([6], dtype=np.int32); ptrs[names.index("ncol")] = lib.ncol_cell.ctypes.data
            return 0
        if self.name == "pycam_stage7_resume_v1":
            context, kernel, token, event = args
            if kernel != 1 or token != lib.token:
                return 4
            event._obj.value = 0; return 0
        if self.name == "pycam_stage7_error_v1":
            args[1].value = b"fake error"; return 0
        return 0


def test_the_binding_starts_decodes_a_frame_and_resumes_with_the_token() -> None:
    from freecam.pi_cam.segment_runner import STAGE, StageSevenRunner, image_offers_runner
    from freecam.physics.macrophysics import Macrophysics
    from freecam.physics.segments import SegmentEvent

    lib = _FakeStageSevenLibrary()
    assert image_offers_runner(lib)
    runner = StageSevenRunner(lib, Macrophysics.DESCRIPTORS)
    lib.names = runner.names
    assert runner.names[:5] == ("lchnk", "ncol", "dt", "p", "dp") and runner.names[-1] == "do_cldice"
    context = runner.create(STAGE)
    assert runner.start(context, {"mmacro_pcond": True}) == SegmentEvent.NEEDS_PYTHON_KERNEL
    frame = runner.frame(context)
    assert (frame.kernel, frame.call_index, frame.lchnk, frame.ncol, frame.substep, frame.token) == \
        ("mmacro_pcond", 3, 10, 6, 2, 41)
    assert frame.argument("t0").array.ctypes.data == lib.arrays["t0"].ctypes.data       # a view, not a copy
    assert frame.argument("t0").intent == "inout" and frame.argument("s_tendout").intent == "out"
    assert frame.argument("ncol").array.shape == () and frame.argument("ncol").array.dtype == np.int32
    batch = frame.batch()
    assert batch["t0"].shape == (6, 4) and "s_tendout" not in batch and int(batch["ncol"]) == 6
    frame.write_back({"t0": np.full((6, 4), 2.0), **{a.name: np.ones((6, 4)) for a in frame.arguments
                                                       if a.intent == "out"}})
    assert np.all(lib.arrays["s_tendout"][:6] == 1.0) and np.all(lib.arrays["s_tendout"][6:] == 0.0)
    assert runner.resume(context, "mmacro_pcond", frame.token) == SegmentEvent.DONE
    with pytest.raises(Exception, match="refused to resume"):
        runner.resume(context, "mmacro_pcond", frame.token + 1)
    assert runner.error(context) == "fake error"
    runner.destroy(context)
    assert [c[0] for c in lib.calls][:4] == ["pycam_stage7_create_v1", "pycam_stage7_start_v1",
                                              "pycam_stage7_frame_v1", "pycam_stage7_resume_v1"]
