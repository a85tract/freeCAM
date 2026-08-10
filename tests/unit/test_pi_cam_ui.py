from __future__ import annotations

import numpy as np
import pytest

from freecam.pi_cam import Physics, Variable
from freecam.pi_cam.ui import PICAMStateView, PICAMWorkflowView


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
        phase = "cam_run1"
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
        "phase": "cam_run1",
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
    assert calls[1][1]["phase"] == "cam_run1"
    assert calls[1][1]["before"] == "dry_adjustment"
    assert calls[1][1]["after"] is None
