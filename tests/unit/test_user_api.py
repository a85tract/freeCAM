from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from freecam import (
    BlockingModel,
    FieldCollection,
    ModelGroup,
    PhaseCollection,
    PlanBuilder,
    PhysicsCollection,
    PhysicsPluginSpec,
    SegmentPlan,
)


class _Future:
    def __init__(self, value: Any) -> None:
        self.value = value

    def result(self) -> Any:
        return self.value


class _Owner:
    def __init__(self, *, futures: bool = False) -> None:
        self.futures = futures
        self.calls: list[tuple[Any, ...]] = []
        self.values = {
            "air_temperature": np.asfortranarray(
                np.arange(6, dtype=np.float64).reshape(2, 3)
            )
        }
        self._closed = False

    def _return(self, value: Any) -> Any:
        return _Future(value) if self.futures else value

    def define_variable(self, spec: Any, *, initial: Any) -> Any:
        self.calls.append(("define_variable", spec, initial))
        return self._return({"name": spec.name})

    def delete_variable(self, name: str) -> Any:
        self.calls.append(("delete_variable", name))
        return self._return({"name": name})

    def get_field(self, name: str, *, rank: int = 0) -> Any:
        self.calls.append(("get_field", name, rank))
        return self._return(self.values[name].copy())

    field = get_field

    def set_field(
        self,
        name: str,
        value: Any,
        *,
        rank: int = 0,
        unsafe: bool = False,
    ) -> Any:
        self.calls.append(("set_field", name, rank, unsafe))
        self.values[name] = np.asarray(value).copy()
        return self._return({"name": name})

    def install_physics(
        self,
        spec: Any,
        *,
        initial_values: Any,
        effective: str,
        unsafe: bool,
    ) -> Any:
        self.calls.append(
            (
                "install_physics",
                spec,
                initial_values,
                effective,
                unsafe,
            )
        )
        return self._return({"name": spec.name or "inferred"})

    def run_scheme(self, name: str, *, group: str | None = None) -> Any:
        self.calls.append(("run_scheme", name, group))
        return self._return({"last_scheme": name})

    def run_phase(self, name: str) -> Any:
        self.calls.append(("run_phase", name))
        return self._return({"last_phase": name})

    def prepare_initial_step(self) -> Any:
        self.calls.append(("prepare_initial_step",))
        return self._return({"prepared": True})

    def set_scheme_enabled(
        self,
        name: str,
        enabled: bool,
        *,
        group: str | None,
        unsafe: bool,
    ) -> Any:
        self.calls.append(("set_scheme_enabled", name, enabled, group, unsafe))
        return self._return({"enabled": enabled})

    def move_scheme(self, name: str, **kwargs: Any) -> Any:
        self.calls.append(("move_scheme", name, kwargs))
        return self._return({"name": name})

    def activate_physics(self, name: str, *, unsafe: bool) -> Any:
        self.calls.append(("activate_physics", name, unsafe))
        return self._return({"active": True})

    def deactivate_physics(self, name: str, *, unsafe: bool) -> Any:
        self.calls.append(("deactivate_physics", name, unsafe))
        return self._return({"active": False})

    def step(self, count: int = 1) -> Any:
        self.calls.append(("step", count))
        return self._return({"step": count})

    def describe(self) -> Any:
        return self._return(
            {
                "name": "test-model",
                "running": True,
                "ranks": 24,
                "step": 5,
                "native_calls": 99,
                "mpi_launch_count": 1,
                "worker_host": "worker",
                "worker_pid": 123,
                "launch_mode": "pbs",
                "pbs_job_id": "123.server",
                "outer_pbs_job_id": None,
                "field_count": len(self.values),
                "snapshot_transport": "initialization",
                "run_dir": "/tmp/run",
                "history_dir": "/tmp/history",
                "log_path": "/tmp/model.log",
            }
        )

    def checkpoint(self, path: Any = None) -> Any:
        self.calls.append(("checkpoint", path))
        return self._return(
            {
                "checkpoint_dir": "/tmp/checkpoint",
                "step": 5,
                "native_calls": 99,
                "mpi_launch_count": 1,
            }
        )

    def memory_checkpoint(self) -> Any:
        self.calls.append(("memory_checkpoint",))
        return self._return(b"snapshot")

    def run_plan(self, plan: Any) -> Any:
        self.calls.append(("run_plan", plan))
        return self._return(
            {
                "name": plan.name,
                "step": plan.step_count,
                "action_trace": (),
            }
        )

    def close(self) -> Any:
        self._closed = True
        return self._return({"closed": True})


def test_fields_create_translates_friendly_dimensions() -> None:
    owner = _Owner()
    fields = FieldCollection(owner)

    result = fields.create(
        "experiment_tracer",
        dims=("column", "level"),
        units="kg kg-1",
        initial=2.0,
    )

    assert result == {"name": "experiment_tracer"}
    _, spec, initial = owner.calls[-1]
    assert spec.standard_name == "experiment_tracer"
    assert spec.dimensions == ("nphys_local", "pver")
    assert spec.dtype == "float64"
    assert initial == 2.0


def test_fields_delete_and_remove_use_collective_owner_operation() -> None:
    owner = _Owner()
    fields = FieldCollection(owner)

    assert fields.delete("experiment_tracer") == {"name": "experiment_tracer"}
    assert fields.remove("second_tracer") == {"name": "second_tracer"}
    assert owner.calls[-2:] == [
        ("delete_variable", "experiment_tracer"),
        ("delete_variable", "second_tracer"),
    ]


def test_field_reference_reads_writes_and_computes_local_stats() -> None:
    owner = _Owner()
    fields = FieldCollection(owner)

    values = fields["air_temperature"].get(rank=0)
    fields["air_temperature"].set(values + 1.0, rank=0)
    stats = fields["air_temperature"].stats(rank=0)

    assert np.array_equal(owner.values["air_temperature"], values + 1.0)
    assert stats["shape"] == (2, 3)
    assert stats["mean"] == 3.5
    assert np.array_equal(
        fields.air_temperature.get(rank=0),
        owner.values["air_temperature"],
    )


def test_physics_install_infers_process_and_hides_protocol_objects(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(
        "schema_version: 1\n"
        "name: my_microphysics\n"
        "processes:\n"
        "  my_microphysics: run\n"
    )
    owner = _Owner()
    physics = PhysicsCollection(owner)

    result = physics.install(
        descriptor,
        after="kessler",
        inputs={"required_input": 1.0},
    )

    assert result == {"name": "inferred"}
    _, spec, inputs, effective, unsafe = owner.calls[-1]
    assert isinstance(spec, PhysicsPluginSpec)
    assert spec.source == str(descriptor)
    assert len(spec.placements) == 1
    placement = spec.placements[0]
    assert placement.process == "my_microphysics"
    assert placement.group == "physics_before_coupler"
    assert placement.after == "kessler"
    assert inputs == {"required_input": 1.0}
    assert effective == "now"
    assert unsafe is True


def test_scheme_reference_uses_short_group_names() -> None:
    owner = _Owner()
    scheme = PhysicsCollection(owner).scheme("my_microphysics", group="before")

    scheme.run()
    scheme.disable()
    scheme.enable()
    scheme.move(to_group="after", before="thermo_water_update")

    assert owner.calls == [
        (
            "run_scheme",
            "my_microphysics",
            "physics_before_coupler",
        ),
        (
            "set_scheme_enabled",
            "my_microphysics",
            False,
            "physics_before_coupler",
            True,
        ),
        (
            "set_scheme_enabled",
            "my_microphysics",
            True,
            "physics_before_coupler",
            True,
        ),
        (
            "move_scheme",
            "my_microphysics",
            {
                "before": "thermo_water_update",
                "after": None,
                "group": "physics_before_coupler",
                "to_group": "physics_after_coupler",
                "unsafe": True,
            },
        ),
    ]


def test_phase_reference_runs_named_phase_and_prepare_boundary() -> None:
    owner = _Owner()
    phases = PhaseCollection(owner)

    assert phases["dynamics_to_physics"].run() == {"last_phase": "dynamics_to_physics"}
    assert phases.prepare() == {"prepared": True}
    assert owner.calls == [
        ("run_phase", "dynamics_to_physics"),
        ("prepare_initial_step",),
    ]


def test_plan_builder_compiles_pythonic_calls_to_serializable_actions(
    tmp_path: Path,
) -> None:
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(
        "schema_version: 1\n"
        "name: my_microphysics\n"
        "processes:\n"
        "  my_microphysics: run\n"
    )
    plan = PlanBuilder("custom-path", experimental=True)
    plan.fields.create(
        "experiment_tracer",
        dims=("column", "level"),
        units="kg kg-1",
    )
    plan.physics.install(
        descriptor,
        after="kessler",
        inputs={"required_input": 1.0},
    )
    plan.physics.scheme("my_microphysics", group="before").run()
    plan.observe("experiment_tracer")
    plan.phases["dynamics_to_physics"].run()

    compiled = plan.build()
    payload = compiled.as_dict()

    assert compiled.unsafe is True
    assert [row["type"] for row in payload["actions"]] == [
        "define_variable",
        "install_physics",
        "run_scheme",
        "observe_fields",
        "run_phase",
    ]
    assert payload["actions"][0]["spec"]["dimensions"] == [
        "nphys_local",
        "pver",
    ]
    assert payload["actions"][2]["group"] == "physics_before_coupler"


def test_plan_builder_serializes_python_process_parameters() -> None:
    def heating(fields, context, *, increment):
        del fields, context, increment

    plan = PlanBuilder("parameterized", experimental=True)
    plan.physics.install_python(
        heating,
        name="heating",
        writes=("air_temperature",),
        parameters={"increment": 1.0},
    )
    plan.physics.scheme("heating", group="before").run(increment=2.0)

    payload = plan.as_dict()
    assert payload["actions"][0]["process"]["parameters"] == {"increment": 1.0}
    assert payload["actions"][1]["parameters"] == {"increment": 2.0}
    assert SegmentPlan.from_mapping(payload).as_dict() == payload


def test_blocking_model_waits_while_submit_keeps_futures() -> None:
    asynchronous = _Owner(futures=True)
    model = BlockingModel(asynchronous)

    assert model.step(2) == {"step": 2}
    assert model.fields["air_temperature"].get(rank=0).shape == (2, 3)
    assert model.phases["physics_timestep_initial"].run() == {
        "last_phase": "physics_timestep_initial"
    }
    future = model.submit.step(3)
    assert isinstance(future, _Future)
    assert future.result() == {"step": 3}

    assert model.advance(steps=2) is model
    assert model.status.step == 5
    assert model.step_count == 5
    assert model.mpi_launch_count == 1
    checkpoint = model.save()
    assert checkpoint.path == Path("/tmp/checkpoint")
    assert checkpoint.step == 5
    assert model.snapshot() == b"snapshot"
    plan = PlanBuilder("live-plan").step(2)
    assert model.execute(plan)["step"] == 2

    assert model.close() == {"closed": True}
    assert asynchronous._closed


def test_model_group_advances_and_closes_every_model() -> None:
    first_owner = _Owner(futures=True)
    second_owner = _Owner(futures=True)
    group = ModelGroup(
        {
            "first": BlockingModel(first_owner),
            "second": BlockingModel(second_owner),
        }
    )

    with group as models:
        assert models.advance(steps=2) is models
        assert set(models.statuses) == {"first", "second"}

    assert first_owner._closed
    assert second_owner._closed
