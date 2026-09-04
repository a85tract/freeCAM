from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import Physics, Variable
from freecam.pi_cam.ui import PICAMStateView, PICAMWorkflowView, plot_steps

PROJECT = Path(__file__).resolve().parents[2]


class FakeSession:
    def __init__(self) -> None:
        temperature = np.full((4, 3, 2), 999.0, order="F")
        temperature[:2, :, 0] = ((210.0, 220.0, 230.0), (230.0, 240.0, 250.0))
        temperature[:3, :, 1] = (
            (250.0, 260.0, 270.0),
            (270.0, 280.0, 290.0),
            (290.0, 300.0, 310.0),
        )
        humidity = np.full((4, 3, 2, 2), 99.0, order="F")
        humidity[:, :, 0, :] = temperature * 1.0e-3
        humidity[:, :, 1, :] = 42.0
        self._arrays = {
            "phys_state.ncol": np.asarray((2, 3), dtype=np.int32),
            "phys_state.t": temperature,
            "phys_state.u": temperature - 250.0,
            "phys_state.v": np.zeros_like(temperature),
            "phys_state.q": humidity,
            "experiment_tracer": np.arange(15.0).reshape(5, 3),
        }
        self._status = {
            "step": 7,
            "date": 10103,
            "fields": {
                name: {"units": "1", "shape": list(values.shape)}
                for name, values in self._arrays.items()
            },
            "step_plan": (
                {
                    "index": 0,
                    "phase": "coupling",
                    "name": "boundary_import",
                    "operation": "boundary_import",
                    "kind": "boundary",
                    "native_id": 202,
                    "enabled": True,
                },
                {
                    "index": 1,
                    "phase": "cam_run1",
                    "name": "dry_adjustment",
                    "operation": "dadadj",
                    "kind": "scheme",
                    "native_id": 423,
                    "enabled": False,
                },
            ),
        }
        self._step_plots = []

        class Field:
            def __init__(inner_self, name, selection=None):
                inner_self.name = name
                inner_self.selection = selection

            @property
            def metadata(inner_self):
                values = self._arrays[inner_self.name]
                dimensions = (
                    ("pcols", "pver", "chunks")
                    if values.ndim == 3
                    else ("nphys_local", "pver")
                )
                return {"dimensions": dimensions, "units": "1"}

            def __getitem__(inner_self, selection):
                return Field(inner_self.name, selection)

            def stats(inner_self, *, rank=0):
                assert rank in {0, "global"}
                selection = (
                    Ellipsis
                    if inner_self.selection is None
                    else inner_self.selection
                )
                values = np.asarray(self._arrays[inner_self.name][selection])
                return {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                }

        class Fields:
            def __getitem__(inner_self, name):
                return Field(name)

            @staticmethod
            def _resolve(name):
                return name

        self.fields = Fields()

    @property
    def status(self):
        return self._status

    def field(self, name: str, *, rank: int = 0):
        assert rank == 0
        return self._arrays[name].copy(order="F")

    def _register_step_plot(self, plot):
        self._step_plots.append(plot)

    def capture_step_plots(self):
        for plot in self._step_plots:
            plot.capture()


def test_state_view_profiles_ignore_padded_columns_and_select_water_vapor() -> None:
    state = PICAMStateView(FakeSession())

    assert np.array_equal(state.profile("T"), (250.0, 260.0, 270.0))
    assert np.array_equal(state.profile("q"), (250.0, 260.0, 270.0))
    assert np.array_equal(
        state.profile("experiment_tracer"),
        (6.0, 7.0, 8.0),
    )
    summary = state.summary()
    assert "rank=0" in summary
    assert "step=7" in summary
    assert "T_mean=260.00 K" in summary
    assert "q_mean=260.000 g/kg" in summary


def test_workflow_view_is_iterable_and_has_notebook_html() -> None:
    workflow = PICAMWorkflowView(FakeSession())

    assert list(workflow) == []
    assert [action.qualified_name for action in workflow.debug] == [
        "coupling.boundary_import"
    ]
    assert len(workflow.actions(include_disabled=True)) == 1
    assert len(workflow.debug.actions(include_disabled=True)) == 2
    assert workflow.debug.describe()[0]["native_id"] == 202
    html = workflow._repr_html_()
    assert "dry_adjustment" in html
    assert "freecam-disabled" in html
    assert "boundary_import" not in html
    assert "boundary_import" in workflow.debug._repr_html_()


def test_notebook_documents_scientific_workflow_list_operations() -> None:
    notebook = json.loads((PROJECT / "examples/try_pi_cam.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "7067eaef")
    source = "".join(cell["source"])

    # This example may be commented while a long-lived Notebook session is
    # active.  Validate the documented Python list API without requiring the
    # cell to mutate a live scientific workflow when the test suite reads it.
    assert "source_order.copy()" in source
    assert "custom_order.remove(radiation)" in source
    assert "custom_order.insert(" in source
    assert "workflow[:] = custom_order" in source
    assert "workflow[:] = source_order" in source
    assert "boundary_import" not in source
    assert "leaf_" not in source


def test_state_view_plot_supports_before_after_overlay() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    state = PICAMStateView(FakeSession())

    figure, axes = state.plot(label="initial")
    state.plot_profile(
        "T",
        ax=axes[0, 0],
        color="tab:orange",
        label="later",
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].yaxis_inverted()
    assert len(axes[0, 0].lines) == 2
    assert "step 7" in figure._suptitle.get_text()


def test_latest_plot_refreshes_in_place_when_displayed() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    session = FakeSession()
    state = PICAMStateView(session)
    plot = state.plot(variables=("T",), mode="latest")

    original = plot.axes[0, 0].lines[0].get_xdata().copy()
    session._arrays["phys_state.t"] += 3.0
    rendered = plot._repr_png_()

    assert rendered.startswith(b"\x89PNG")
    assert len(plot.axes[0, 0].lines) == 1
    assert np.array_equal(plot.axes[0, 0].lines[0].get_xdata(), original + 3.0)
    assert len(plot.snapshots) == 1


def test_history_plot_keeps_changed_profiles_without_duplicate_displays() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    session = FakeSession()
    state = PICAMStateView(session)
    plot = state.plot(variables=("T",), mode="history", label="initial")

    plot._repr_png_()
    assert len(plot.axes[0, 0].lines) == 1
    session._arrays["phys_state.t"] += 2.0
    plot._repr_png_()

    assert len(plot.axes[0, 0].lines) == 2
    assert len(plot.snapshots) == 2
    assert np.array_equal(
        plot.axes[0, 0].lines[1].get_xdata(),
        plot.axes[0, 0].lines[0].get_xdata() + 2.0,
    )


def test_live_step_plot_records_one_scalar_per_complete_step() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    session = FakeSession()
    state = PICAMStateView(session)
    plot = state.plot_steps(
        "experiment_tracer",
        rank=0,
        statistic="mean",
        level=-1,
        figsize=(8.0, 4.0),
    )

    session._status["step"] = 8
    session._arrays["experiment_tracer"] += 3.0
    session.capture_step_plots()
    axis = plot.plot(label="tracer")

    assert plot.steps == [7, 8]
    assert plot.values == [8.0, 11.0]
    assert np.array_equal(axis.lines[0].get_xdata(), (7, 8))
    assert np.array_equal(axis.lines[0].get_ydata(), (8.0, 11.0))
    assert axis.get_xlabel() == "model step"
    assert np.array_equal(axis.figure.get_size_inches(), (8.0, 4.0))


def test_live_step_plot_rejects_invalid_figsize() -> None:
    state = PICAMStateView(FakeSession())

    with pytest.raises(ValueError, match="two positive numbers"):
        state.plot_steps("experiment_tracer", figsize=(5.0, 0.0))


def test_live_step_plot_accepts_multiple_variables() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    session = FakeSession()
    state = PICAMStateView(session)
    plot = state.plot_steps(
        ("phys_state.t", "phys_state.u"),
        rank="global",
        statistic="mean",
        level=-1,
        figsize=(8.0, 4.0),
    )

    session._status["step"] = 8
    session._arrays["phys_state.t"] += 1.0
    session._arrays["phys_state.u"] -= 1.0
    session.capture_step_plots()
    figure, axes = plot.plot()

    assert plot.steps == (7, 8)
    assert tuple(plot.values) == ("phys_state.t", "phys_state.u")
    assert axes.shape == (1, 2)
    assert all(len(axis.lines[0].get_xdata()) == 2 for axis in axes.flat)
    assert np.array_equal(figure.get_size_inches(), (8.0, 4.0))


def test_case_step_plot_overlays_same_variable() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    control_session = FakeSession()
    volcanic_session = FakeSession()
    control = PICAMStateView(control_session).plot_steps("phys_state.t")
    volcanic = PICAMStateView(volcanic_session).plot_steps("phys_state.t")

    control_session._status["step"] = 8
    volcanic_session._status["step"] = 8
    volcanic_session._arrays["phys_state.t"] -= 2.0
    control_session.capture_step_plots()
    volcanic_session.capture_step_plots()
    comparison = plot_steps(
        {"PI-atm": control, "PI-atm-volcanic": volcanic},
        figsize=(7.0, 3.0),
    )
    axis = comparison.plot()

    assert len(axis.lines) == 2
    assert [line.get_label() for line in axis.lines] == [
        "PI-atm",
        "PI-atm-volcanic",
    ]
    assert np.array_equal(axis.figure.get_size_inches(), (7.0, 3.0))


def test_state_view_plot_builds_grid_from_requested_variables() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    state = PICAMStateView(FakeSession())

    figure, axes = state.plot(
        variables=("T", "u", "v", "q", "experiment_tracer"),
    )

    assert axes.shape == (2, 3)
    assert sum(axis.get_visible() for axis in axes.flat) == 5
    assert all(len(axis.lines) == 1 for axis in axes.flat[:5])
    assert axes.flat[5].get_visible() is False
    assert figure is axes.flat[0].figure


def test_state_view_plot_resolves_unique_statepool_short_names() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    session = FakeSession()
    session._arrays["phys_state.omega"] = session._arrays["phys_state.t"] * 0.01
    session._arrays["phys_state.pmid"] = session._arrays["phys_state.t"] * 100.0
    for name in ("phys_state.omega", "phys_state.pmid"):
        session._status["fields"][name] = {
            "units": "1",
            "shape": list(session._arrays[name].shape),
        }

    class Fields:
        def _resolve(self, name):
            matches = [
                canonical
                for canonical in session._status["fields"]
                if canonical.rsplit(".", 1)[-1] == name
            ]
            if len(matches) == 1:
                return matches[0]
            raise KeyError(name)

    session.fields = Fields()
    state = PICAMStateView(session)

    figure, axes = state.plot(variables=("omega", "pmid"))

    assert axes.shape == (1, 2)
    assert all(len(axis.lines) == 1 for axis in axes.flat)
    assert axes[0, 0].get_xlabel() == "vertical pressure velocity omega (Pa/s)"
    assert axes[0, 1].get_xlabel() == "midpoint pressure (Pa)"
    assert figure is axes.flat[0].figure


def test_state_view_plot_rejects_empty_variables() -> None:
    state = PICAMStateView(FakeSession())

    with pytest.raises(ValueError, match="at least one"):
        state.plot(variables=())


def test_state_attribute_assignment_creates_and_deletes_distributed_variable() -> None:
    calls = []

    class Field:
        def delete(self):
            calls.append(("delete", "experiment_tracer"))

    class Fields:
        def create(self, name, **kwargs):
            calls.append(("create", name, kwargs))

        def __getitem__(self, name):
            assert name == "experiment_tracer"
            return Field()

    session = FakeSession()
    session.fields = Fields()
    state = PICAMStateView(session)

    state.experiment_tracer = Variable(
        ("pcols", "pver", "chunks"),
        units="kg kg-1",
        initial=0.0,
        aliases=("tracer",),
    )
    state.scratch_probe = Variable(("pcols", "chunks"), output=False)
    del state.experiment_tracer

    assert calls[0][0:2] == ("create", "experiment_tracer")
    assert calls[0][2]["dims"] == ("pcols", "pver", "chunks")
    assert calls[0][2]["aliases"] == ("tracer",)
    # A field joins CAM's history by default and stays out of it on request.
    assert calls[0][2]["output"] is True
    assert calls[1][2]["output"] is False
    assert calls[2] == ("delete", "experiment_tracer")


def test_state_attribute_assignment_accepts_rank_independent_numpy_array() -> None:
    calls = []

    class Fields:
        def create_array(self, name, values):
            calls.append((name, values.copy()))

    session = FakeSession()
    session.fields = Fields()
    state = PICAMStateView(session)
    relative_humidity = np.zeros(30)

    state.rh = relative_humidity

    assert calls[0][0] == "rh"
    assert np.array_equal(calls[0][1], relative_humidity)


def test_state_augmented_assignment_edits_existing_distributed_field() -> None:
    calls = []

    class Field:
        name = "phys_state.t"

        def __init__(self, session):
            self.session = session

        def __iadd__(self, value):
            calls.append(("add", self.name, value))
            return self

    class Fields:
        def __init__(self, session):
            self.session = session

        def __getitem__(self, name):
            assert name == "phys_state.t"
            return Field(self.session)

        def _resolve(self, name):
            if name != "phys_state.t":
                raise KeyError(name)
            return name

    session = FakeSession()
    session.fields = Fields(session)
    state = PICAMStateView(session)

    state.T += 1.0

    assert calls == [("add", "phys_state.t", 1.0)]


def test_state_attribute_assignment_rejects_ambiguous_python_sequence() -> None:
    state = PICAMStateView(FakeSession())

    with pytest.raises(TypeError, match="NumPy array.*distributed field"):
        state.rh = [0.0] * 30


def test_state_view_exposes_mapping_and_metadata_interfaces() -> None:
    session = FakeSession()

    class Fields:
        def _resolve(self, name):
            if name not in session._arrays:
                raise KeyError(name)
            return name

        def __getitem__(self, name):
            return session._arrays[name]

    session.fields = Fields()
    state = PICAMStateView(session)

    assert "phys_state.t" in state
    assert "missing" not in state
    assert tuple(state) == tuple(session.status["fields"])
    assert len(state) == len(session.status["fields"])
    metadata = {row["name"]: row for row in state.describe()}
    assert metadata["phys_state.t"]["shape"] == (4, 3, 2)
    assert metadata["phys_state.t"]["units"] == "1"


def test_workflow_insert_installs_physics_object_with_declared_placement() -> None:
    calls = []

    class PhysicsCollection:
        def install_python(self, function, **kwargs):
            calls.append((function, kwargs))
            return "installed"

    class Heating(Physics):
        name = "notebook_heating"
        after = "dadadj"
        reads = ("phys_state.q",)
        writes = ("phys_state.t",)

        def tendency(self, fields, context):
            fields["phys_state.t"][...] += context.timestep_seconds

    session = FakeSession()
    session.physics = PhysicsCollection()

    workflow = PICAMWorkflowView(session)
    result = workflow.insert(Heating())

    assert result is None
    assert calls[0][1] == {
        "name": "notebook_heating",
        "before": None,
        "after": "dadadj",
        "reads": ("phys_state.q",),
        "writes": ("phys_state.t",),
        "parameters": None,
        "enabled": True,
        "transactional": True,
        # an ordinary notebook process reads and writes StatePool fields
        # inside the snapshot, so it is neither native nor unsafe
        "native": False,
        "trusted_native": False,
        "unsafe": False,
    }

    session._status["step_plan"][1]["enabled"] = True
    result = workflow.insert(1, Heating())
    assert result is None
    assert calls[1][1]["before"] is None
    assert calls[1][1]["after"] == "dry_adjustment"


@pytest.mark.parametrize("source_available", (True, False))
def test_workflow_infers_simple_python_process_field_contract(
    monkeypatch, source_available
) -> None:
    calls = []

    class PhysicsCollection:
        def install_python(self, function, **kwargs):
            calls.append((function, kwargs))
            return "installed"

    class Heating(Physics):
        name = "automatic_heating"
        after = "dadadj"

        def run(self, state, context):
            state.T += 0.01 * context.timestep_seconds
            state.q[...] = np.maximum(state.q, 0.0)

    session = FakeSession()
    session.physics = PhysicsCollection()
    if not source_available:
        monkeypatch.setattr(
            "freecam.pi_cam.facade.inspect.getsource",
            lambda callback: (_ for _ in ()).throw(OSError("not in linecache")),
        )

    assert PICAMWorkflowView(session).insert(Heating()) is None
    assert calls[0][1]["reads"] == ()
    assert calls[0][1]["writes"] == ("T", "q")


def test_state_create_like_reuses_distributed_field_dimensions() -> None:
    calls = []

    class Field:
        metadata = {
            "dimensions": ("pcols", "pver", "chunks"),
            "dtype": "<f8",
            "units": "K",
        }

    class Fields:
        def __getattr__(self, name):
            if name == "T":
                return Field()
            raise AttributeError(name)

        def __getitem__(self, name):
            return Field()

        def create(self, name, **kwargs):
            calls.append((name, kwargs))

    session = FakeSession()
    session.fields = Fields()
    state = PICAMStateView(session)

    created = state.zeros_like("heating_rate", "T", units="K s-1")

    assert isinstance(created, Field)
    assert calls == [
        (
            "heating_rate",
            {
                "dims": ("pcols", "pver", "chunks"),
                "dtype": "<f8",
                "units": "K s-1",
                "initial": 0.0,
                "writable": True,
                "restart": True,
                "output": True,
                "aliases": (),
                "standard_name": None,
            },
        )
    ]


def test_workflow_moves_and_toggles_process_objects() -> None:
    calls = []

    class Process:
        def __init__(self, name, phase):
            self.name = name
            self.phase = phase

        @property
        def qualified_name(self):
            return f"{self.phase}.{self.name}"

        def enable(self):
            calls.append(("enable", self.qualified_name))
            return {"enabled": True}

        def disable(self):
            calls.append(("disable", self.qualified_name))
            return {"enabled": False}

    processes = {
        "dry_adjustment": Process("dry_adjustment", "cam_run1"),
        "vertical_diffusion": Process("vertical_diffusion", "cam_run2"),
    }

    class PhysicsCollection:
        def process(self, name, phase=None):
            result = processes[name]
            assert phase in {None, result.phase}
            return result

    session = FakeSession()
    session.physics = PhysicsCollection()
    session.move_action = lambda name, **kwargs: calls.append(
        ("move", name, kwargs)
    ) or {"name": name}
    workflow = PICAMWorkflowView(session)

    workflow.move("dry_adjustment", before="vertical_diffusion")
    workflow.disable(processes["dry_adjustment"])
    workflow.enable("dry_adjustment")

    assert calls == [
        (
            "move",
            "dry_adjustment",
            {
                "phase": "cam_run1",
                "before": "cam_run2.vertical_diffusion",
                "after": None,
            },
        ),
        ("disable", "cam_run1.dry_adjustment"),
        ("enable", "cam_run1.dry_adjustment"),
    ]


def test_workflow_assignment_runs_the_listed_processes_only() -> None:
    """``workflow[:] = [process]`` leaves exactly that process in the step."""

    calls = []

    class Process:
        def __init__(self, name, phase, kind="scheme", operation=None):
            self.name = name
            self.phase = phase
            self.kind = kind
            self.operation = operation or name

        @property
        def qualified_name(self):
            return f"{self.phase}.{self.name}"

        def enable(self):
            calls.append(("enable", self.qualified_name))

        def disable(self):
            calls.append(("disable", self.qualified_name))

        def remove(self):
            calls.append(("remove", self.qualified_name))

    rows = (
        ("coupling", "boundary_import", "boundary", "boundary_import", True),
        ("cam_run1", "dry_adjustment", "scheme", "dadadj", False),
        ("cam_run1", "cloud_macro_microphysics", "scheme", "macro_microphysics", True),
        ("cam_run1", "notebook_heating", "python_process", "notebook_heating", True),
        ("cam_run1", "cloud_diagnostics_leaf", "scheme", "leaf_cloud_diagnostics_calc", True),
        ("cam_run1", "radiation", "scheme", "radiation_tend", True),
        ("cam_run4", "history", "io", "wshist", True),
        ("coupling", "boundary_export", "boundary", "boundary_export", True),
    )
    processes = {
        name: Process(name, phase, kind, operation)
        for phase, name, kind, operation, _ in rows
    }
    session = FakeSession()
    session._status["step_plan"] = tuple(
        {"phase": phase, "name": name, "kind": kind, "operation": operation,
         "enabled": enabled}
        for phase, name, kind, operation, enabled in rows
    )
    session.workflow_action = lambda name, phase, kind: processes[name]
    session.replace_workflow = lambda order: calls.append(tuple(order)) or {}
    workflow = PICAMWorkflowView(session)

    # Names work like handles; a leaf goes by the name the workflow displays.
    workflow[:] = ["macro_microphysics", "dry_adjustment", "cloud_diagnostics"]

    assert calls == [
        ("enable", "cam_run1.dry_adjustment"),        # listed, was disabled
        ("remove", "cam_run1.notebook_heating"),      # omitted runtime process
        ("disable", "cam_run1.radiation"),            # omitted CAM process
        (                                             # hidden actions keep slots
            "coupling.boundary_import",
            "cam_run1.cloud_macro_microphysics",
            "cam_run1.dry_adjustment",
            "cam_run1.cloud_diagnostics_leaf",
            "cam_run4.history",
            "coupling.boundary_export",
        ),
    ]
    with pytest.raises(ValueError, match="only once"):
        workflow[:] = ["radiation", "radiation"]


def test_workflow_slice_assignment_replaces_one_complete_remote_order() -> None:
    calls = []

    class Process:
        def __init__(self, name, phase):
            self.name = name
            self.phase = phase

        @property
        def qualified_name(self):
            return f"{self.phase}.{self.name}"

    session = FakeSession()
    session._status["step_plan"] = (
        {
            "phase": "coupling",
            "name": "boundary_import",
            "operation": "boundary_import",
            "kind": "boundary",
            "enabled": True,
        },
        {
            "phase": "cam_run1",
            "name": "radiation",
            "operation": "radiation_tend",
            "kind": "scheme",
            "enabled": True,
        },
        {
            "phase": "cam_run2",
            "name": "vertical_diffusion",
            "operation": "vertical_diffusion_tend",
            "kind": "scheme",
            "enabled": True,
        },
        {
            "phase": "coupling",
            "name": "boundary_export",
            "operation": "boundary_export",
            "kind": "boundary",
            "enabled": True,
        },
    )
    session.workflow_action = lambda name, phase, kind: Process(name, phase)
    session.replace_workflow = lambda order: calls.append(tuple(order)) or {
        "plan": session._status["step_plan"]
    }
    workflow = PICAMWorkflowView(session)
    order = workflow[:]
    order[0], order[1] = order[1], order[0]

    workflow[:] = order

    assert calls == [
        (
            "coupling.boundary_import",
            "cam_run2.vertical_diffusion",
            "cam_run1.radiation",
            "coupling.boundary_export",
        )
    ]


def test_workflow_list_operations_preserve_required_control_actions() -> None:
    calls = []

    class Process:
        def __init__(self, name, phase, operation, kind="scheme", enabled=True):
            self.name = name
            self.phase = phase
            self.operation = operation
            self.kind = kind
            self.enabled = enabled

        @property
        def qualified_name(self):
            return f"{self.phase}.{self.name}"

        def enable(self):
            self.enabled = True
            calls.append(("enable", self.qualified_name))

        def disable(self):
            self.enabled = False
            calls.append(("disable", self.qualified_name))

    processes = {
        "boundary_import": Process(
            "boundary_import", "coupling", "boundary_import", "boundary"
        ),
        "radiation": Process("radiation", "cam_run1", "radiation_tend"),
        "optional": Process(
            "optional", "cam_run1", "optional", enabled=False
        ),
        "advance_timestep": Process(
            "advance_timestep", "clock", "advance_timestep", "clock"
        ),
        "boundary_export": Process(
            "boundary_export", "coupling", "boundary_export", "boundary"
        ),
    }
    session = FakeSession()
    session._status["step_plan"] = tuple(
        {
            "phase": item.phase,
            "name": item.name,
            "operation": item.operation,
            "kind": item.kind,
            "enabled": item.enabled,
        }
        for item in processes.values()
    )
    session.workflow_action = (
        lambda name, phase, kind: processes[name]
    )
    session.move_action = lambda name, **kwargs: calls.append(
        ("move", name, kwargs)
    ) or {"name": name}
    workflow = PICAMWorkflowView(session)

    removed = workflow.pop(0)
    appended = workflow.append(processes["optional"])

    assert removed is processes["radiation"]
    assert appended is None
    assert calls == [
        ("disable", "cam_run1.radiation"),
        ("enable", "cam_run1.optional"),
        (
            "move",
            "optional",
            {
                "phase": "cam_run1",
                "before": None,
                "after": "cam_run1.radiation",
            },
        ),
    ]
    with pytest.raises(ValueError, match="cannot be popped"):
        workflow.debug.pop(0)


class _HandleSession(FakeSession):
    """A session that serves live process handles, as the real one does."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, str]] = []

    def workflow_action(self, name: str, *, phase: str, kind: str):
        calls = self.calls

        class Handle:
            def __init__(self) -> None:
                self.name, self.phase = name, phase

            def enable(self):
                calls.append(("enable", name, phase))
                return {"enabled": True}

            def disable(self):
                calls.append(("disable", name, phase))
                return {"enabled": False}

        return Handle()


def test_naming_a_process_and_telling_it_to_stop_are_one_path() -> None:
    """``workflow.process(name).disable()`` is ``workflow.disable(name)``.

    The container's verb resolves the name and calls the handle's, so there
    is one implementation of stopping a process and two ways to spell it --
    the list-like form for code that is reordering a step, the handle form
    for code that already holds one process.
    """

    session = _HandleSession()
    workflow = PICAMWorkflowView(session)

    workflow.process("dry_adjustment").disable()
    workflow.disable("dry_adjustment")

    assert session.calls == [("disable", "dry_adjustment", "cam_run1")] * 2


def test_the_lookup_is_the_subscript_and_refuses_a_name_it_cannot_resolve() -> None:
    workflow = PICAMWorkflowView(FakeSession())

    assert workflow.process("dry_adjustment") == workflow["dry_adjustment"]
    with pytest.raises(KeyError, match="no_such_process"):
        workflow.process("no_such_process")


def test_every_way_of_reaching_a_process_spells_the_lookup_the_same() -> None:
    """One word for "give me this process", whichever object is asked.

    The live workflow, the flat physics catalogue and the order declared
    before MPI starts are three different objects at three different times;
    a notebook should not have to remember which of them accepts the word.
    """

    from freecam.pi_cam.facade import WorkflowTemplate
    from freecam.pi_cam.session import _SessionPhysicsCollection

    for owner in (PICAMWorkflowView, WorkflowTemplate, _SessionPhysicsCollection):
        assert callable(getattr(owner, "process", None)), owner.__name__
