"""The stage machinery is generic: a second stage reuses it without editing it.

``Macrophysics`` was the first stage whose driver layer moved to Python, and
everything about it that is not macrophysics lives in
``freecam.physics.stage``.  The way to prove that is to build a stage that
has nothing to do with macrophysics -- its own Fortran prefix, its own
kernel, its own constants -- and drive it through the same code.  If the
prefix, the scratch allocation, the exact copies or the kernel swap had
macrophysics baked into them, this file would not run.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

from freecam.physics.errors import PhysicsError
from freecam.physics.stage import (
    HOST_ENTRIES, HostEntries, HostServices, Local, NativeStage, StageRuntime, _StageProcess,
)

PCOLS, PVER = 8, 4

DESCRIPTORS = """
schema_version: 1
kernels:
- name: widget_step
  routine: widget_step
  symbol: pycam_widget_step_v1
  action_id: 0
  modules: {}
  arguments:
  - field: widget.ncol
    dtype: int32
    rank: 1
    intent: in
    chunk_axis: 1
    extents: [chunks]
  - field: widget.x
    dtype: float64
    rank: 3
    intent: in
    chunk_axis: 3
    extents: [pcols, pver, chunks]
  - field: widget.y
    dtype: float64
    rank: 3
    intent: out
    chunk_axis: 3
    extents: [pcols, widgets, chunks]
"""


class _Lib:
    """A fake image whose entries all succeed and whose views are real arrays."""

    def __init__(self) -> None:
        self.views: dict[tuple[int, int], np.ndarray] = {}
        self.calls: list[str] = []
        self.owner = 0
        self.history: list[tuple[str, float]] = []

    def __getattr__(self, name):
        if not name.startswith("pycam_"):
            raise AttributeError(name)
        lib = self

        def entry(*args):
            lib.calls.append(name)
            if name == "pycam_widget_view_v1":
                lchnk, code, ptr, ndims, extents = args
                array = lib.views.setdefault((lchnk, code), np.zeros((PCOLS, PVER), order="F"))
                ptr._obj.value = array.ctypes.data
                ndims._obj.value = array.ndim
                for i, e in enumerate(array.shape):
                    extents[i] = e
            elif name == "pycam_widget_set_owner_v1":
                lib.owner = args[0]
            elif name == "pycam_widget_nstep_v1":
                return 7
            elif name == "pycam_widget_dt_v1":
                return 900
            elif name == "pycam_outfld_v1":
                lib.history.append((args[0].decode(), 0.0))
            return 0

        return entry


class _Native:
    def __init__(self, lib, descriptors: Path):
        self.library = lib
        self.pool = _Pool({"grid.chunk_id": np.array([10, 11]),
                           "grid.chunk_ncols": np.array([6, 5]),
                           "cam_in.landfrac": np.zeros((PCOLS, 2), order="F")})
        self.kernels: list[str] = []
        from freecam.pi_cam.kernel_codegen import load_direct_kernels

        self._args = {k.name: [{"field": a.field, "dtype": a.dtype, "rank": a.rank}
                               for a in k.arguments]
                      for k in load_direct_kernels(descriptors)}

    @property
    def chunks(self):
        return self.pool["grid.chunk_id"], self.pool["grid.chunk_ncols"]

    def kernel_arguments(self, name):
        return tuple(self._args[name])

    def run_kernel(self, name, arrays):
        self.kernels.append(name)
        arrays["widget.y"][...] = 2.0


class _Pool(dict):
    @property
    def dimensions(self):
        return {"pcols": PCOLS, "pver": PVER, "pverp": PVER + 1, "pcnst": 3, "chunks": 2}


class _Context:
    def __init__(self, native):
        self.native = native
        self.timestep_seconds = 1800
        self.step = 1
        self.rank = 2


class Widget(NativeStage):
    """A stage that is not macrophysics, driven by the same machinery."""

    STAGE = "cam_run1.widgets"
    FIRST_HALF = "cam_run1.widget_pre"
    SECOND_HALF = "cam_run1.widget_post"
    PREFIX = "widget"
    PROCESS_NAME = "widget_tend"
    TRACE_ENV = "FREECAM_WIDGET_TRACE"
    KERNELS = ("widget_step",)
    EXTRA_SCRATCH = (("spare", ("pcols", "pver", "chunks")),)
    CAM_IN = ("landfrac",)

    def read_constants(self, library):
        return {"widgets": 2}

    def extra_extents(self, constants):
        return {"widgets": constants["widgets"]}

    def tend_chunk(self, st, lchnk, ncol, index, dt, nstep):
        self.calls.append("view")
        x = st.handles.view(lchnk, 1)
        self.calls.append("cam_in")
        st.cam_in(index)
        self.calls.append("outfld")
        st.handles.outfld("WIDGET  ", x, PCOLS, lchnk)
        self.calls.append("kernel")
        st.swappable_kernel("widget_step", {"ncol": ncol, "x": x},
                            outputs={"y": st.local["y"]}, ncol=ncol, lchnk=lchnk, dt=dt,
                            kernel=self.kernel)


@pytest.fixture
def widget(tmp_path):
    descriptors = tmp_path / "widget.yaml"
    descriptors.write_text(DESCRIPTORS)
    Widget.DESCRIPTORS = descriptors
    lib = _Lib()
    return _Native(lib, descriptors)


# -- the prefix is a parameter, not a constant -----------------------------------


def test_the_entries_are_bound_under_the_stage_s_own_prefix(widget) -> None:
    entries = HostEntries(widget.library, "widget")
    for attribute, (template, _, _) in HOST_ENTRIES.items():
        assert getattr(entries, attribute) is not None, attribute
    entries.state_copy(10)
    entries.bind_hosts()
    assert widget.library.calls == ["pycam_widget_state_copy_v1", "pycam_widget_bind_hosts_v1"]


def test_an_absent_required_entry_is_refused_and_an_optional_one_is_not() -> None:
    class Bare:
        def __getattr__(self, name):
            if name.endswith(("_nstep_v1", "_dt_v1")):
                raise AttributeError(name)
            def entry(*_):
                return 0
            return entry

    entries = HostEntries(Bare(), "widget")
    assert entries.nstep is None and entries.dt is None

    class Empty:
        def __getattr__(self, name):
            raise AttributeError(name)

    from freecam.pi_cam.errors import PICAMConfigurationError

    with pytest.raises(PICAMConfigurationError, match="exposes no pycam_widget_set_owner_v1"):
        HostEntries(Empty(), "widget")


# -- the runtime -----------------------------------------------------------------


def test_scratch_is_sized_from_the_descriptors_and_the_stage_s_own_extents(widget) -> None:
    stage = Widget()
    runtime = stage.runtime(widget)
    assert runtime.scratch["x"].shape == (PCOLS, PVER, 1)
    # `widgets` is not a pool dimension; the stage supplied it from its constants
    assert runtime.scratch["y"].shape == (PCOLS, 2, 1)
    assert runtime.scratch["spare"].shape == (PCOLS, PVER, 1)
    assert all(a.flags.f_contiguous for a in runtime.scratch.values())
    assert widget.library.owner == 1


def test_a_field_list_the_image_disagrees_with_is_refused(widget) -> None:
    from freecam.pi_cam.errors import PICAMConfigurationError

    widget._args["widget_step"] = [{"field": "widget.ncol", "dtype": "int32", "rank": 1}]
    with pytest.raises(PICAMConfigurationError, match="the descriptors say"):
        Widget().runtime(widget)


def test_the_runtime_is_built_once_per_pool_and_not_shared_between_stages(widget) -> None:
    one, two = Widget(), Widget()
    assert one.runtime(widget) is one.runtime(widget)
    assert one.runtime(widget) is not two.runtime(widget)


def test_local_drops_the_chunk_axis_and_follows_late_allocations() -> None:
    scratch = {"a": np.zeros((3, 2, 1), order="F")}
    local = Local(scratch)
    assert local["a"].shape == (3, 2)
    scratch["b"] = np.ones((4, 1), order="F")
    assert local["b"].shape == (4,) and len(local) == 2


# -- the exact copies ------------------------------------------------------------


def test_copy_out_writes_live_lanes_only(widget) -> None:
    stage = Widget()
    runtime = stage.runtime(widget)
    target = np.full((PCOLS, 2), -777.0, order="F")
    runtime.scratch["y"][...] = 1.0
    runtime._copy_out(target, runtime.scratch["y"], ncol=5)
    assert np.all(target[:5] == 1.0)
    assert np.all(target[5:] == -777.0), "padding must stay what CAM left there"


def test_copy_in_takes_scalars_and_arrays_without_arithmetic(widget) -> None:
    runtime = Widget().runtime(widget)
    runtime._copy_in(runtime.scratch["ncol"], 6)
    assert int(runtime.scratch["ncol"][0]) == 6
    values = np.arange(PCOLS * PVER, dtype=np.float64).reshape(PCOLS, PVER, order="F")
    runtime._copy_in(runtime.scratch["x"], values)
    assert np.array_equal(runtime.scratch["x"][..., 0], values)


# -- running -------------------------------------------------------------------


def test_tend_walks_every_chunk_and_reads_the_clock_from_the_image(widget) -> None:
    stage = Widget()
    stage.tend(None, _Context(widget))
    assert stage.calls == ["view", "cam_in", "outfld", "kernel"] * 2
    assert widget.kernels == ["widget_step", "widget_step"]
    assert stage.runtime(widget).nstep == 7        # the image's clock, not the context's step


def test_tend_refuses_to_run_as_an_ordinary_process() -> None:
    class Context:
        native = None

    with pytest.raises(PhysicsError, match="native"):
        Widget().tend(None, Context())


def test_a_model_replaces_the_kernel_and_must_answer_every_output(widget) -> None:
    seen = {}

    def model(batch):
        seen.update({k: np.asarray(v).shape for k, v in batch.items()})
        return {"y": np.full((batch["x"].shape[0], 2), 5.0)}

    stage = Widget()
    stage.kernel = model
    stage.tend(None, _Context(widget))
    assert seen["x"] == (5, PVER)                  # the last chunk's live columns only
    assert widget.kernels == []                    # the original never ran

    stage = Widget()
    stage.kernel = lambda batch: {}
    with pytest.raises(PhysicsError, match="missing"):
        stage.tend(None, _Context(widget))


def test_the_trace_records_both_the_original_and_a_model_in_its_place(
        widget, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FREECAM_WIDGET_TRACE", str(tmp_path))
    Widget().tend(None, _Context(widget))
    files = list(tmp_path.glob("widget_trace.rank-*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert [(l["lchnk"], l["ncol"], l["replaced"]) for l in lines] == [(10, 6, False), (11, 5, False)]
    assert set(lines[0]["before"]) == {"ncol", "x"} and set(lines[0]["after"]) == {"y"}

    stage = Widget()
    stage.kernel = lambda batch: {"y": np.zeros((batch["x"].shape[0], 2))}
    stage.tend(None, _Context(widget))
    lines = [json.loads(l) for l in files[0].read_text().splitlines()]
    assert [l["replaced"] for l in lines[2:]] == [True, True]


# -- installing ------------------------------------------------------------------


def test_attach_swaps_the_stage_for_its_halves_and_sits_between_them() -> None:
    class Action:
        def __init__(self): self.enabled = None
        def enable(self, **_): self.enabled = True
        def disable(self, **_): self.enabled = False

    class Workflow:
        def __init__(self):
            self.items = {Widget.STAGE: Action(), Widget.FIRST_HALF: Action(),
                          Widget.SECOND_HALF: Action()}
            self.inserted = []
        def process(self, name): return self.items[name]
        def insert_after(self, anchor, process):
            self.inserted.append((anchor, process)); return process

    class Run:
        workflow = Workflow()

    handle = Widget().attach(Run)
    assert Run.workflow.items[Widget.STAGE].enabled is False
    assert Run.workflow.items[Widget.FIRST_HALF].enabled is True
    assert Run.workflow.items[Widget.SECOND_HALF].enabled is True
    assert Run.workflow.inserted == [(Widget.FIRST_HALF, handle)]
    assert isinstance(handle, _StageProcess)
    assert handle.name == "widget_tend"
    assert handle.native is True and handle.transactional is False
    assert handle.reads == () and handle.writes == ()
