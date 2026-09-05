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
        self.actions: list[tuple] = []
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

    def run_action(self, name, *, phase=None):
        self.actions.append((name, phase))

    def segment_runner(self, stage):
        return getattr(self, "runner", None)


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
    SWAPPABLE = ("widget_step",)
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
                            outputs={"y": st.local["y"]}, ncol=ncol, lchnk=lchnk, dt=dt)


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


# -- several swappable kernels ---------------------------------------------------


def test_a_stage_with_one_swappable_kernel_keeps_the_singular_kernel_attribute() -> None:
    stage = Widget()
    assert stage.kernels == {"widget_step": None}
    assert stage.kernel is None
    model = lambda batch: {}
    stage.kernel = model
    assert stage.kernels["widget_step"] is model


def test_a_stage_with_two_swappable_kernels_refuses_the_singular_attribute() -> None:
    class Pair(Widget):
        SWAPPABLE = ("widget_step", "widget_other")

    stage = Pair()
    assert stage.kernels == {"widget_step": None, "widget_other": None}
    with pytest.raises(PhysicsError, match="assign into .kernels"):
        stage.kernel
    with pytest.raises(PhysicsError, match="assign into .kernels"):
        stage.kernel = lambda batch: {}
    stage.kernels["widget_other"] = lambda batch: {}      # the way to do it
    assert stage.kernels["widget_step"] is None


def test_constructing_with_an_unknown_kernel_name_is_refused() -> None:
    with pytest.raises(PhysicsError, match="no swappable kernel named"):
        Widget(kernels={"nope": lambda batch: {}})


def test_the_stage_s_kernels_mapping_is_what_swappable_kernel_looks_up(widget) -> None:
    """Nothing threads the model through the call; the runtime reads the stage."""

    stage = Widget()
    stage.kernels["widget_step"] = lambda batch: {"y": np.full((batch["x"].shape[0], 2), 9.0)}
    stage.tend(None, _Context(widget))
    assert widget.kernels == []                       # the original never ran
    assert stage.runtime(widget).scratch["y"][0, 0, 0] == 9.0


# -- an original that is not a direct kernel -------------------------------------


def test_original_runs_in_the_direct_kernel_s_place_and_is_traced(
        widget, tmp_path, monkeypatch) -> None:
    """A routine taking a derived type cannot be a direct kernel; the stage
    hands swappable_kernel a closure that calls its handle wrapper instead."""

    monkeypatch.setenv("FREECAM_WIDGET_TRACE", str(tmp_path))
    ran = []

    class ViaWrapper(Widget):
        def tend_chunk(self, st, lchnk, ncol, index, dt, nstep):
            x = st.handles.view(lchnk, 1)
            st.swappable_kernel("widget_step", {"ncol": ncol, "x": x},
                                outputs={"y": st.local["y"]}, ncol=ncol, lchnk=lchnk, dt=dt,
                                original=lambda: ran.append(lchnk))

    stage = ViaWrapper()
    stage.tend(None, _Context(widget))
    assert ran == [10, 11]
    assert widget.kernels == []                       # the direct kernel was not used
    lines = [json.loads(l) for l in
             next(tmp_path.glob("widget_trace.rank-*.jsonl")).read_text().splitlines()]
    assert [l["replaced"] for l in lines] == [False, False]
    assert set(lines[0]["before"]) == {"ncol", "x"}

    # a model still wins over the closure
    ran.clear()
    stage = ViaWrapper()
    stage.kernels["widget_step"] = lambda batch: {"y": np.zeros((batch["x"].shape[0], 2))}
    stage.tend(None, _Context(widget))
    assert ran == []


# -- host services a stage did not declare ---------------------------------------


def test_a_service_the_stage_never_declared_is_refused_by_name() -> None:
    from freecam.pi_cam.errors import PICAMConfigurationError
    from freecam.physics.stage import CORE_ENTRIES, HOST_ENTRIES, PTEND_ENTRIES

    assert HOST_ENTRIES == {**CORE_ENTRIES, **PTEND_ENTRIES}
    assert set(CORE_ENTRIES) & set(PTEND_ENTRIES) == set()

    class CoreOnly(HostEntries):
        TABLE = CORE_ENTRIES

    services = HostServices(CoreOnly(_Lib(), "widget"), pcnst=3)
    services.outfld("X       ", np.zeros((PCOLS, PVER), order="F"), PCOLS, 10)   # core: fine
    with pytest.raises(PICAMConfigurationError, match="declares no 'state_copy' entry"):
        services.state_copy(10)
    with pytest.raises(PICAMConfigurationError, match="declares no 'update' entry"):
        services.update(10, 1800.0)


# -- profiling -------------------------------------------------------------------


def test_the_profile_times_every_entry_kernel_and_tend_by_name(widget, tmp_path, monkeypatch) -> None:
    """Where a stage's time goes is a question the gate cannot answer -- it
    sees one region -- so the runtime can be asked to time each handle entry,
    each kernel and its copies, and each trace hash under its own key."""

    monkeypatch.setenv("FREECAM_WIDGET_TRACE", str(tmp_path / "trace"))
    monkeypatch.setenv("FREECAM_WIDGET_PROFILE", str(tmp_path / "profile"))

    class Profiled(Widget):
        PROFILE_ENV = "FREECAM_WIDGET_PROFILE"

    stage = Profiled()
    stage.tend(None, _Context(widget))
    files = list((tmp_path / "profile").glob("widget_profile.rank-*.json"))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report["rank"] == 2
    seconds, calls = report["seconds"], report["calls"]
    # the whole walk, and one region per chunk
    assert calls["tend"] == 1 and calls["tend_chunk"] == 2
    # every handle entry the walk used, under its own name
    for entry in ("entry:view", "entry:outfld", "entry:set_owner", "entry:bind_hosts"):
        assert entry in calls, entry
    assert calls["entry:view"] == 2 and calls["entry:outfld"] == 2
    # the kernel, split into its copies and its run
    for key in ("kernel-copy-in:widget_step", "kernel-run:widget_step",
                "kernel-copy-out:widget_step"):
        assert calls[key] == 2, key
    # the trace's hashing and its write, separately
    assert calls["trace-hash:widget_step"] == 4      # before and after, per chunk
    assert calls["trace-write:widget_step"] == 2
    assert all(v >= 0.0 for v in seconds.values())
    # the totals are ordered so the biggest cost reads first
    assert list(seconds) == sorted(seconds, key=lambda k: -seconds[k])


def test_with_profiling_off_nothing_is_timed_and_nothing_is_written(widget, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("FREECAM_WIDGET_PROFILE", raising=False)
    stage = Widget()
    stage.tend(None, _Context(widget))
    from freecam.physics.stage import _NoProfile

    assert isinstance(stage.runtime(widget).profile, _NoProfile)
    assert not list(tmp_path.glob("*profile*"))


# -- a stage that is a whole action --------------------------------------------


class _Workflow:
    def __init__(self, *names):
        self.items = {n: _Action() for n in names}
        self.inserted = []

    def process(self, name):
        return self.items[name]

    def insert_after(self, anchor, process):
        self.inserted.append((anchor, process))
        return process


class _Action:
    def __init__(self):
        self.enabled = None

    def enable(self, **_):
        self.enabled = True

    def disable(self, **_):
        self.enabled = False


def test_a_stage_with_no_halves_replaces_its_whole_action() -> None:
    class Whole(Widget):
        STAGE = "cam_run1.widgets"
        FIRST_HALF = ""
        SECOND_HALF = ""

    class Run:
        workflow = _Workflow("cam_run1.widgets")

    stage = Whole()
    assert stage.replaces_whole_action
    handle = stage.attach(Run)
    assert Run.workflow.items["cam_run1.widgets"].enabled is False
    # the process takes the action's slot: inserted right after the disabled action
    assert Run.workflow.inserted == [("cam_run1.widgets", handle)]


def test_a_stage_with_halves_still_sits_between_them() -> None:
    class Run:
        workflow = _Workflow(Widget.STAGE, Widget.FIRST_HALF, Widget.SECOND_HALF)

    stage = Widget()
    assert not stage.replaces_whole_action
    handle = stage.attach(Run)
    assert Run.workflow.items[Widget.FIRST_HALF].enabled is True
    assert Run.workflow.inserted == [(Widget.FIRST_HALF, handle)]


# -- composition -----------------------------------------------------------------


def test_a_composed_stage_shares_one_kernels_mapping_with_its_sub_walks() -> None:
    class Inner(Widget):
        PREFIX = "inner"
        SWAPPABLE = ("inner_core",)
        KERNELS = ()

    class Outer(Widget):
        PREFIX = "outer"
        SWAPPABLE = ("outer_core",)
        KERNELS = ()

    outer = Outer()
    inner = Inner()
    outer.compose(inner=inner)
    assert outer.inner is inner
    assert list(outer.components) == ["inner"]
    assert set(outer.kernels) == {"outer_core", "inner_core"}
    # one mapping: assigning through the outer reaches the inner's lookup
    model = lambda batch: {}
    outer.kernels["inner_core"] = model
    assert inner.kernels["inner_core"] is model
    assert inner.kernels is outer.kernels


def test_composing_refuses_a_kernel_name_two_stages_both_claim() -> None:
    class A(Widget):
        SWAPPABLE = ("core",)
        KERNELS = ()

    class B(Widget):
        SWAPPABLE = ("core",)
        KERNELS = ()

    outer = A()
    outer.kernels["core"] = lambda batch: {}
    with pytest.raises(PhysicsError, match="already a swappable kernel"):
        outer.compose(other=B())


# -- kernels are bound once, run every call -------------------------------------


def test_a_kernel_is_bound_once_per_chunk_and_run_on_every_call_when_the_native_can_bind(widget) -> None:
    """The argument marshalling is paid per distinct set of arrays, not per call.

    An intent(in) array is handed to the kernel in place, so each chunk's view
    of x is its own binding; across two chunks and two steps that is two
    bindings and four runs.  A native that offers bind_kernel is never called
    the plain way.
    """

    binds: list[int] = []
    runs: list[str] = []

    class _Binding(_Native):
        def bind_kernel(self, name, arrays):
            binds.append(arrays["widget.x"].ctypes.data)

            def run():
                runs.append(name)
                arrays["widget.y"][...] = 2.0
            return run

        def run_kernel(self, name, arrays):
            raise AssertionError("a native that binds must not be called one shot at a time")

    native = _Binding(widget.library, Widget.DESCRIPTORS)
    stage = Widget()
    stage.tend(None, _Context(native))
    stage.tend(None, _Context(native))

    assert binds == [widget.library.views[(10, 1)].ctypes.data, widget.library.views[(11, 1)].ctypes.data]
    assert runs == ["widget_step"] * 4
    assert np.all(stage.runtime(native).local["y"] == 2.0)


# -- what is resolved once and kept ----------------------------------------------


def test_a_view_of_unchanged_storage_is_the_same_object_and_moved_storage_a_new_one(widget) -> None:
    runtime = Widget().runtime(widget)
    first = runtime.handles.view(10, 1)
    assert runtime.handles.view(10, 1) is first          # same address and extents
    widget.library.views[(10, 1)] = np.zeros((PCOLS, PVER), order="F")   # the image moved it
    moved = runtime.handles.view(10, 1)
    assert moved is not first
    assert moved.ctypes.data == widget.library.views[(10, 1)].ctypes.data
    assert runtime.handles.view(11, 1) is not first      # another chunk, another view


def test_a_kernel_is_planned_once_and_an_input_view_is_read_in_place(widget) -> None:
    binds: list[dict] = []
    runs: list[str] = []

    def bind_kernel(name, arrays):
        binds.append(dict(arrays))
        target = arrays["widget.y"]

        def run():
            runs.append(name)
            target[...] = 2.0
        return run

    widget.bind_kernel = bind_kernel
    stage = Widget()
    stage.tend(None, _Context(widget))
    stage.tend(None, _Context(widget))
    assert len(binds) == 2 and runs == ["widget_step"] * 4    # one bind per chunk's view of x
    assert widget.kernels == []                                # the plain path was never taken
    runtime = stage.runtime(widget)
    assert len(runtime._plans) == 1
    (plan,) = runtime._plans.values()
    assert plan.fields == tuple(a.field for a in runtime.descriptors["widget_step"].arguments)
    assert [local for local, _, _ in plan.slots] == [f.removeprefix("widget.") for f in plan.fields]
    # x is intent(in): the kernel read CAM's storage itself, and the scratch copy stayed untouched
    x_view = widget.library.views[(10, 1)]
    assert binds[0]["widget.x"].ctypes.data == x_view.ctypes.data
    assert binds[0]["widget.x"].shape == runtime.scratch["x"].shape       # given the chunk axis
    assert np.all(runtime.scratch["x"] == 0.0)
    # y is written: it is the runtime's own scratch, copied out afterwards
    assert binds[0]["widget.y"] is runtime.scratch["y"]
    assert np.all(runtime.local["y"] == 2.0)


def test_the_same_kernel_under_two_field_maps_keeps_two_plans(widget) -> None:
    """The maps are per-call temporaries; a plan must follow the map's content,
    never the dict object -- a freed dict's address comes back for another."""

    seen: list[dict[str, np.ndarray]] = []

    def bind_kernel(name, arrays):
        seen.append(dict(arrays))
        return lambda: None

    widget.bind_kernel = bind_kernel
    runtime = Widget().runtime(widget)
    cube = np.zeros((PCOLS, PVER, 2), order="F")
    ones, twos = cube[:, :, 0], cube[:, :, 1]                # views, as the walks hand
    ones[...] = 1.0
    twos[...] = 2.0
    for _ in range(3):                                       # as a walk does: one map per rate name
        runtime.kernel_on_chunk("widget_step", {"p": ones, "n": np.int32(6)}, outputs={},
                                fields={"p": "widget.x", "q": "widget.y", "n": "widget.ncol"})
        runtime.kernel_on_chunk("widget_step", {"r": twos, "n": np.int32(6)}, outputs={},
                                fields={"r": "widget.x", "s": "widget.y", "n": "widget.ncol"})
    assert len(seen) == 2                                    # one bind per distinct map
    assert seen[0]["widget.x"].ctypes.data == ones.ctypes.data     # each map fed its own array
    assert seen[1]["widget.x"].ctypes.data == twos.ctypes.data
    assert seen[0]["widget.y"] is runtime.scratch["q"]
    assert seen[1]["widget.y"] is runtime.scratch["s"]
    assert len(runtime._plans) == 2


def test_an_input_that_is_also_written_or_of_another_dtype_still_goes_through_scratch(widget) -> None:
    seen: list[dict[str, np.ndarray]] = []
    widget.bind_kernel = lambda name, arrays: (seen.append(dict(arrays)), lambda: None)[1]
    runtime = Widget().runtime(widget)
    x = np.ones((PCOLS, PVER, 1), order="F")[:, :, 0]
    # named among the outputs: the kernel writes it, so it must not be CAM's storage
    runtime.kernel_on_chunk("widget_step", {"x": x, "ncol": np.int32(6)}, outputs={"x": x})
    assert seen[-1]["widget.x"] is runtime.scratch["x"] and np.all(runtime.scratch["x"] == 1.0)
    # an array that owns its memory is a temporary the walk computed: copied, never bound
    runtime.kernel_on_chunk("widget_step", {"x": np.ones((PCOLS, PVER), order="F"), "ncol": np.int32(6)}, outputs={})
    assert seen[-1]["widget.x"] is runtime.scratch["x"]
    # a float32 array or a C-ordered one is copied into the F-ordered double scratch
    runtime.kernel_on_chunk("widget_step", {"x": x.astype(np.float32), "ncol": np.int32(6)}, outputs={})
    assert seen[-1]["widget.x"] is runtime.scratch["x"]
    runtime.kernel_on_chunk("widget_step", {"x": np.ascontiguousarray(x), "ncol": np.int32(6)}, outputs={})
    assert seen[-1]["widget.x"] is runtime.scratch["x"]


def test_a_moved_input_view_is_bound_again_and_the_table_stays_small(widget) -> None:
    binds: list[int] = []
    widget.bind_kernel = lambda name, arrays: (binds.append(arrays["widget.x"].ctypes.data), lambda: None)[1]
    runtime = Widget().runtime(widget)
    block = np.zeros((PCOLS, PVER, 12), order="F")
    views = [block[:, :, i] for i in range(12)]
    for i, view in enumerate(views):                          # storage that moves every call
        view[...] = float(i)
        runtime.kernel_on_chunk("widget_step", {"x": view, "ncol": np.int32(6)}, outputs={})
    assert binds == [v.ctypes.data for v in views]
    (plan,) = runtime._plans.values()
    assert len(plan.bound) <= 64
    runtime.kernel_on_chunk("widget_step", {"x": views[-1], "ncol": np.int32(6)}, outputs={})
    assert len(binds) == 12                                   # the last one was still held


def test_local_hands_back_the_same_view_while_the_scratch_is_the_same(widget) -> None:
    runtime = Widget().runtime(widget)
    first = runtime.local["x"]
    assert runtime.local["x"] is first
    runtime.scratch["x"] = np.zeros_like(runtime.scratch["x"])   # a late re-allocation is followed
    assert runtime.local["x"] is not first
    assert runtime.local["x"].base is runtime.scratch["x"]


def test_a_fresh_slice_of_the_same_storage_reuses_the_binding(widget) -> None:
    """The walks slice their views on every call; the same memory is the same binding."""

    binds: list[int] = []
    widget.bind_kernel = lambda name, arrays: (binds.append(arrays["widget.x"].ctypes.data), lambda: None)[1]
    runtime = Widget().runtime(widget)
    cube = np.zeros((PCOLS, PVER, 3), order="F")
    for _ in range(5):
        runtime.kernel_on_chunk("widget_step", {"x": cube[:, :, 1], "ncol": np.int32(6)}, outputs={})
    assert binds == [cube[:, :, 1].ctypes.data]              # five new view objects, one binding
    runtime.kernel_on_chunk("widget_step", {"x": cube[:, :, 2], "ncol": np.int32(6)}, outputs={})
    assert len(binds) == 2                                    # other memory, another binding


def test_a_kernel_whose_callers_always_hand_new_arrays_goes_back_to_copying(widget) -> None:
    from freecam.physics.stage import REBINDS_BEFORE_COPYING

    binds: list[int] = []
    widget.bind_kernel = lambda name, arrays: (binds.append(arrays["widget.x"].ctypes.data), lambda: None)[1]
    runtime = Widget().runtime(widget)
    block = np.zeros((PCOLS, PVER, REBINDS_BEFORE_COPYING + 20), order="F")
    for i in range(REBINDS_BEFORE_COPYING + 20):
        fresh = block[:, :, i]                                # a view somewhere new every call
        fresh[...] = float(i)
        runtime.kernel_on_chunk("widget_step", {"x": fresh, "ncol": np.int32(6)}, outputs={})
    (plan,) = runtime._plans.values()
    assert plan.in_place is False
    # the switch happens on the bind after the limit, which was still in place;
    # from then on every call reads the scratch copy through one more binding
    assert len(binds) == REBINDS_BEFORE_COPYING + 2
    assert binds[-1] == runtime.scratch["x"].ctypes.data
    assert np.all(runtime.scratch["x"][..., 0] == float(REBINDS_BEFORE_COPYING + 19))


def test_a_surface_column_is_the_same_view_while_the_pool_array_is(widget) -> None:
    runtime = Widget().runtime(widget)
    first = runtime.column("cam_in.landfrac", 1)
    assert runtime.column("cam_in.landfrac", 1) is first
    assert runtime.column("cam_in.landfrac", 0) is not first
    assert runtime.cam_in(1)["landfrac"] is first
    widget.pool["cam_in.landfrac"] = np.ones((PCOLS, 2), order="F")     # the pool re-bound it
    again = runtime.column("cam_in.landfrac", 1)
    assert again is not first and np.all(again == 1.0)


class _Retargetable:
    """A fake bound call that remembers where each argument points, like BoundCall."""

    def __init__(self, log: list, name: str, arrays: dict[str, np.ndarray]) -> None:
        self.log = log
        self.name = name
        self.arrays = list(arrays.values())
        log.append(("bind", name, self.arrays[1].ctypes.data))

    def __call__(self) -> None:
        self.log.append(("run", self.name, self.arrays[1].ctypes.data))

    def retarget(self, index: int, array: np.ndarray) -> None:
        self.arrays[index] = array
        self.log.append(("retarget", self.name, index, array.ctypes.data))


def test_a_full_table_points_its_oldest_call_at_the_new_arrays_instead_of_rebuilding(widget) -> None:
    from freecam.physics.stage import BOUND_PER_PLAN

    log: list = []
    widget.bind_kernel = lambda name, arrays: _Retargetable(log, name, arrays)
    runtime = Widget().runtime(widget)
    block = np.zeros((PCOLS, PVER, BOUND_PER_PLAN + 3), order="F")
    for i in range(BOUND_PER_PLAN + 3):                      # storage that moves on every call
        runtime.kernel_on_chunk("widget_step", {"x": block[:, :, i], "ncol": np.int32(6)}, outputs={})
    binds = [e for e in log if e[0] == "bind"]
    retargets = [e for e in log if e[0] == "retarget"]
    runs = [e for e in log if e[0] == "run"]
    assert len(binds) == BOUND_PER_PLAN                      # the table filled, then no more builds
    assert [e[3] for e in retargets] == [block[:, :, i].ctypes.data for i in range(BOUND_PER_PLAN, BOUND_PER_PLAN + 3)]
    assert [e[2] for e in runs] == [block[:, :, i].ctypes.data for i in range(BOUND_PER_PLAN + 3)]
    (plan,) = runtime._plans.values()
    assert len(plan.bound) == BOUND_PER_PLAN and plan.in_place is True
    # a set already in the table is a plain hit: no bind, no retarget
    before = len(log)
    runtime.kernel_on_chunk("widget_step", {"x": block[:, :, BOUND_PER_PLAN + 2], "ncol": np.int32(6)}, outputs={})
    assert [e[0] for e in log[before:]] == ["run"]


# -- execution modes -------------------------------------------------------------


class WholeWidget(Widget):
    """The widget as a whole workflow action, with an original Fortran stage to run."""

    STAGE = "cam_run1.widgets"
    WHOLE_ACTION = True
    FIRST_HALF = ""
    SECOND_HALF = ""


def test_with_nothing_replaced_the_original_stage_runs_once_and_the_walk_not_at_all(widget) -> None:
    stage = WholeWidget()
    stage.tend(None, _Context(widget))
    stage.tend(None, _Context(widget))
    assert widget.actions == [("cam_run1.widgets", None)] * 2       # one native call a step
    assert widget.kernels == [] and stage.calls == []                # tend_chunk never ran
    assert stage.execution.mode == "native-whole"
    assert stage.execution.describe() == {
        "execution_mode": "native-whole", "active_replacements": [],
        "native_stage_calls": 2, "native_segment_calls": 0, "segment_pauses": 0,
        "python_model_calls": 0, "legacy_steps": 0, "python_fortran_crossings_per_step": 1,
    }


def test_a_replacement_switches_to_the_walk_and_its_removal_back(widget) -> None:
    stage = WholeWidget()
    stage.kernels["widget_step"] = lambda batch: {"y": np.full((batch["ncol"], 2), 3.0)}
    stage.tend(None, _Context(widget))
    assert widget.actions == []                                       # no whole-stage call
    assert stage.calls == ["view", "cam_in", "outfld", "kernel"] * 2  # the walk ran
    assert stage.execution.mode == "legacy-python"
    assert stage.execution.replacements == ("widget_step",)
    assert stage.execution.python_model_calls == 2 and stage.execution.legacy_steps == 1
    stage.kernels["widget_step"] = None
    stage.tend(None, _Context(widget))
    assert widget.actions == [("cam_run1.widgets", None)]
    assert stage.execution.mode == "native-whole"


def test_the_policy_can_force_the_walk_and_refuses_what_it_cannot_do(widget) -> None:
    stage = WholeWidget()
    stage.execution_policy = "legacy-python"
    stage.tend(None, _Context(widget))
    assert widget.actions == [] and stage.execution.mode == "legacy-python"

    stage = WholeWidget()
    stage.execution_policy = "native-whole"
    stage.kernels["widget_step"] = lambda batch: {}
    with pytest.raises(PhysicsError, match="are replaced"):
        stage.tend(None, _Context(widget))
    split = Widget()                                                  # sits inside a split action
    split.execution_policy = "native-whole"
    with pytest.raises(PhysicsError, match="not the whole of"):
        split.tend(None, _Context(widget))
    stage = WholeWidget()
    stage.execution_policy = "segmented"
    with pytest.raises(PhysicsError, match="nothing is replaced"):
        stage.tend(None, _Context(widget))
    stage.kernels["widget_step"] = lambda batch: {}
    with pytest.raises(PhysicsError, match="no segment runner"):     # the fake image offers none
        stage.tend(None, _Context(widget))
    stage.execution_policy = "sideways"
    with pytest.raises(PhysicsError, match="unknown stage execution policy"):
        stage.tend(None, _Context(widget))


def test_a_split_stage_walks_under_auto_because_it_has_no_whole_action(widget) -> None:
    stage = Widget()
    stage.tend(None, _Context(widget))
    assert widget.actions == [] and stage.execution.mode == "legacy-python"


def test_segmented_execution_drives_the_runner_the_image_offers(widget) -> None:
    from tests.unit.test_physics_segments import FakeRunner, _original_a

    widget.runner = FakeRunner()                      # what the image would offer for this stage

    class Segmentable(WholeWidget):
        SWAPPABLE = ("a", "b")
        KERNELS = ("widget_step",)

    stage = Segmentable()
    stage.execution_policy = "segmented"
    stage.kernels["a"] = _original_a
    stage.tend(None, _Context(widget))
    stage.tend(None, _Context(widget))
    assert widget.actions == [] and stage.calls == []          # neither the whole action nor the walk
    described = stage.execution.describe()
    assert described["execution_mode"] == "segmented"
    assert described["active_replacements"] == ["a"]
    assert described["segment_pauses"] == 8 and described["python_model_calls"] == 8
    assert described["native_segment_calls"] == 2 + 8            # two starts, eight resumes
    assert [e for e in widget.runner.log if e[0] == "create"] == [("create", "cam_run1.widgets")]


def test_an_original_kernel_marker_runs_the_direct_kernel_on_the_frame_s_lanes(widget) -> None:
    from freecam.physics.segments import OriginalKernel
    from tests.unit.test_physics_segments import FakeRunner

    widget.runner = FakeRunner()
    ran: list[dict] = []

    def run_kernel(name, arrays):
        ran.append({k: v.copy() for k, v in arrays.items()})
        arrays["widget.y"][..., 0] = 7.0                      # the "original" writes its output

    widget.run_kernel = run_kernel

    class Segmentable(WholeWidget):
        SWAPPABLE = ("a", "b", "widget_step")
        KERNELS = ("widget_step",)

    stage = Segmentable()
    stage.execution_policy = "segmented"
    # the fake runner pauses on kernel "a"; route it to the widget's direct kernel by name
    stage.kernels["a"] = OriginalKernel()
    original = stage._original_through_python(widget, "widget_step")
    answer = original({"ncol": np.int32(6), "x": np.full((6, PVER), 3.0)})
    assert list(answer) == ["y"] and answer["y"].shape == (6, 2) and np.all(answer["y"] == 7.0)
    assert ran[0]["widget.x"].shape == (PCOLS, PVER, 1) and np.all(ran[0]["widget.x"][:6, :, 0] == 3.0)
    assert np.all(ran[0]["widget.x"][6:] == 0.0) and int(ran[0]["widget.ncol"][0]) == 6


def test_the_original_kernel_answers_in_the_frame_s_names_whatever_the_descriptor_s_prefix(widget, tmp_path) -> None:
    """A composed stage's kernels carry a sub-walk's prefix; the frame strips that one."""

    from freecam.physics.segments import OriginalKernel

    other = tmp_path / "other.yaml"
    other.write_text(DESCRIPTORS.replace("widget.", "other."))
    ran: list[dict] = []

    class Other(WholeWidget):
        DESCRIPTORS = other
        SWAPPABLE = ("widget_step",)

    native = _Native(widget.library, other)

    def run_kernel(name, arrays):
        ran.append(dict(arrays))
        arrays["other.y"][..., 0] = 9.0

    native.run_kernel = run_kernel
    stage = Other()
    stage.kernels["widget_step"] = OriginalKernel()
    original = stage._original_through_python(native, "widget_step")
    answer = original({"ncol": np.int32(5), "x": np.full((5, PVER), 4.0)})
    assert list(answer) == ["y"] and np.all(answer["y"] == 9.0) and answer["y"].shape == (5, 2)
    assert np.all(ran[0]["other.x"][:5, :, 0] == 4.0)             # the input was found under its frame name
