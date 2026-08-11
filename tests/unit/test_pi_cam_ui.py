from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import Physics, Variable
from freecam.pi_cam.plan import PICAMStepPlan
from freecam.pi_cam.ui import PICAMStateView, PICAMWorkflowView


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

    @property
    def status(self):
        return self._status

    def field(self, name: str, *, rank: int = 0):
        assert rank == 0
        return self._arrays[name].copy(order="F")


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

    assert [action.qualified_name for action in workflow] == [
        "coupling.boundary_import"
    ]
    assert len(workflow.actions(include_disabled=True)) == 2
    assert workflow.describe()[0]["native_id"] == 202
    html = workflow._repr_html_()
    assert "boundary_import" in html
    assert "dry_adjustment" in html
    assert "freecam-disabled" in html


def test_notebook_defines_the_complete_workflow_as_one_explicit_list() -> None:
    notebook = json.loads((PROJECT / "examples/try_pi_cam.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "7067eaef")
    tree = ast.parse("".join(cell["source"]))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pi_cam_workflow"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.List)
    operations = tuple(
        element.slice.value
        for element in assignment.value.elts
        if isinstance(element, ast.Subscript)
        and isinstance(element.slice, ast.Constant)
    )
    expected = tuple(
        action.operation
        for action in PICAMStepPlan.default()
        if action.enabled
    )

    assert operations == expected


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
    del state.experiment_tracer

    assert calls[0][0:2] == ("create", "experiment_tracer")
    assert calls[0][2]["dims"] == ("pcols", "pver", "chunks")
    assert calls[0][2]["aliases"] == ("tracer",)
    assert calls[1] == ("delete", "experiment_tracer")


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

    result = PICAMWorkflowView(session).insert(Heating())

    assert result == "installed"
    assert calls[0][1] == {
        "name": "notebook_heating",
        "before": None,
        "after": "dadadj",
        "reads": ("phys_state.q",),
        "writes": ("phys_state.t",),
        "enabled": True,
        "transactional": True,
    }

    session._status["step_plan"][1]["enabled"] = True
    result = PICAMWorkflowView(session).insert(1, Heating())
    assert result == "installed"
    assert calls[1][1]["before"] == "dry_adjustment"
    assert calls[1][1]["after"] is None


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
    order[1], order[2] = order[2], order[1]

    workflow[:] = order

    assert calls == [
        (
            "coupling.boundary_import",
            "cam_run2.vertical_diffusion",
            "cam_run1.radiation",
            "coupling.boundary_export",
        )
    ]
