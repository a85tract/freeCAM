import json
from pathlib import Path
from typing import Any

from distributed import Client
import numpy as np
import pytest

from pycam_sima import (
    ActivatePhysics,
    BlockingModel,
    BranchSpec,
    CCPPSuitePlan,
    CheckpointBundle,
    DaskExperimentClient,
    DeactivatePhysics,
    DefineVariable,
    FieldEdit,
    InstallPhysics,
    ObserveFields,
    PersistentDaskSession,
    PhysicsPluginSpec,
    RunSteps,
    SegmentPlan,
    SchemePlacement,
    VariableSpec,
)
from pycam_sima.notebook.persistent_dask import (
    PersistentCAMActor,
    PersistentDaskRequest,
)


ROOT = Path(__file__).resolve().parents[2]
KESSLER_SUITE = (
    ROOT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


def _scheme_plan() -> CCPPSuitePlan:
    return CCPPSuitePlan.from_xml(KESSLER_SUITE)


class _FakeSession:
    def __init__(self, *_args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.running = False
        self.ranks = 24
        self.current_step = 0
        self.native_calls = 0
        self.launch_mode_used = kwargs["launch_mode"]
        self.job_id = None
        self.field_names = ("air_temperature",)
        self.physics_plugins: list[dict[str, Any]] = []
        self.phase_names = ("dynamics_to_physics",)
        self.phase_status = {"last_phase": None, "next_phase": None}
        plan = kwargs["scheme_plan"]
        self.scheme_names = plan.keys
        self.scheme_status = {
            "last_scheme": None,
            "last_scheme_group": None,
            "plan": plan.to_payload(),
            "sequence_safe": plan.sequence_safe,
        }
        self.scheme_plan = _FakeSchemeEditor(self, plan)
        self.values = [
            np.full((2, 2), 240.0 + rank, dtype=np.float64) for rank in range(24)
        ]

    def start(self) -> "_FakeSession":
        self.running = True
        return self

    def close(self) -> None:
        self.running = False

    def step(self, count: int = 1) -> int:
        self.current_step += count
        self.native_calls += count * 3
        return self.current_step

    def prepare_initial_step(self) -> dict[str, Any]:
        return dict(self.phase_status)

    def run_phase(self, name: str) -> dict[str, Any]:
        self.phase_status = {"last_phase": name, "next_phase": None}
        return dict(self.phase_status)

    def run_scheme(self, name: str, *, group: str | None = None) -> dict[str, Any]:
        selected = self.scheme_plan.plan.scheme(name, group=group)
        self.native_calls += 1
        self.scheme_status["last_scheme"] = selected.key
        self.scheme_status["last_scheme_group"] = selected.group
        return dict(self.scheme_status)

    def run_scheme_group(self, group: str) -> dict[str, Any]:
        self.native_calls += len(self.scheme_plan.plan.active(group))
        self.scheme_status["last_scheme_group"] = group
        return dict(self.scheme_status)

    def get_field(self, name: str, *, rank: int | str = 0) -> Any:
        assert name == "air_temperature"
        if rank == "all":
            return [value.copy() for value in self.values]
        return self.values[int(rank)].copy()

    def get_field_stats(self, name: str, *, rank: int | str = 0) -> Any:
        def stats(index: int) -> dict[str, Any]:
            values = self.values[index]
            return {
                "rank": index,
                "shape": values.shape,
                "dtype": values.dtype.str,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }

        if rank == "all":
            return [stats(index) for index in range(24)]
        return stats(int(rank))

    def set_field(
        self,
        name: str,
        value: Any,
        *,
        rank: int | str = 0,
        unsafe: bool = False,
    ) -> None:
        del unsafe
        assert name == "air_temperature"
        if rank == "all":
            for index in range(24):
                self.values[index][...] = value
        else:
            self.values[int(rank)][...] = value

    def define_variable(
        self, spec: VariableSpec, *, initial: Any = 0.0
    ) -> dict[str, Any]:
        self.field_names = (*self.field_names, spec.name)
        return {
            "standard_name": spec.name,
            "shape": (2, 2),
            "dtype": np.dtype(spec.dtype).str,
            "writable": spec.writable,
        }

    def delete_variable(self, name: str) -> dict[str, Any]:
        self.field_names = tuple(
            item for item in self.field_names if item != name
        )
        return {"standard_name": name, "owner": "python"}

    def install_physics(
        self,
        spec: PhysicsPluginSpec,
        *,
        initial_values: Any = None,
        effective: str = "now",
        unsafe: bool = False,
    ) -> dict[str, Any]:
        del initial_values, unsafe
        record = {
            "name": spec.name,
            "active": effective == "now",
            "pending": effective == "next_step",
        }
        self.physics_plugins.append(record)
        return record

    def activate_physics(
        self, name: str, *, unsafe: bool = False
    ) -> dict[str, Any]:
        del unsafe
        record = next(
            item for item in self.physics_plugins if item["name"] == name
        )
        record.update(active=True, pending=False)
        return record

    def deactivate_physics(
        self, name: str, *, unsafe: bool = False
    ) -> dict[str, Any]:
        del unsafe
        record = next(
            item for item in self.physics_plugins if item["name"] == name
        )
        record.update(active=False, pending=False)
        return record

    def edit_field(
        self,
        name: str,
        operation: str,
        value: float,
        *,
        unsafe: bool = False,
    ) -> dict[str, Any]:
        del unsafe
        assert name == "air_temperature"
        for current in self.values:
            if operation == "set":
                current[...] = value
            elif operation == "add":
                current[...] += value
            else:
                current[...] *= value
        return {"step": self.current_step}

    def field_info(self, name: str) -> dict[str, Any]:
        if name != "air_temperature":
            raise KeyError(name)
        return {"writable": True, "shape": (2, 2), "dtype": "<f8"}

    def write_checkpoint(self, path: str | Path) -> Path:
        target = Path(path)
        target.mkdir(parents=True)
        (target / "manifest.json").write_text("{}")
        return target

    def memory_checkpoint(self) -> CheckpointBundle:
        payload = {
            "step": self.current_step,
            "native_calls": self.native_calls,
            "values": [value.tolist() for value in self.values],
        }
        return CheckpointBundle(
            (("fake-state.json", json.dumps(payload).encode("utf-8")),)
        )

    def restore_memory_checkpoint(
        self,
        snapshot: CheckpointBundle,
    ) -> dict[str, Any]:
        payload = json.loads(dict(snapshot.files)["fake-state.json"])
        self.current_step = int(payload["step"])
        self.native_calls = int(payload["native_calls"])
        self.values = [
            np.array(value, dtype=np.float64, order="F")
            for value in payload["values"]
        ]
        return {
            "step": self.current_step,
            "native_calls": self.native_calls,
            "phase_status": self.phase_status,
            "scheme_status": self.scheme_status,
            "restored_from_memory": True,
        }


class _FakeSchemeEditor:
    def __init__(self, session: _FakeSession, plan: CCPPSuitePlan) -> None:
        self.session = session
        self.plan = plan

    def _sync(self) -> None:
        self.session.scheme_status["plan"] = self.plan.to_payload()
        self.session.scheme_status["sequence_safe"] = self.plan.sequence_safe

    def enable(self, name: str, *, group: str | None = None) -> None:
        self.plan.enable(name, group=group)
        self._sync()

    def disable(
        self,
        name: str,
        *,
        group: str | None = None,
        unsafe: bool = False,
    ) -> None:
        self.plan.disable(name, group=group, unsafe=unsafe)
        self._sync()

    def move(self, name: str, **kwargs: Any) -> None:
        self.plan.move(name, **kwargs)
        self._sync()

    def reset(self) -> None:
        self.plan = _scheme_plan()
        self._sync()

    def describe(self, group: str | None = None) -> list[dict[str, object]]:
        return self.plan.describe(group)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "config.yaml"
    config.write_text("{}\n")
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "atm_in").write_text("&cam_initfiles_nl /\n")
    library = tmp_path / "libkernels.so"
    library.touch()
    environment = tmp_path / "environment.sh"
    environment.touch()
    python_target = tmp_path / "python-target"
    python_target.touch()
    python = tmp_path / "python"
    python.symlink_to(python_target)
    return {
        "config": config,
        "initial_run_dir": initial,
        "run_root": tmp_path / "runs",
        "library": library,
        "environment_script": environment,
        "python_executable": python,
        "log_dir": tmp_path / "logs",
    }


def _request(tmp_path: Path) -> PersistentDaskRequest:
    inputs = _inputs(tmp_path)
    return PersistentDaskRequest(
        name="live",
        config=str(inputs["config"]),
        initial_run_dir=str(inputs["initial_run_dir"]),
        run_root=str(inputs["run_root"]),
        library=str(inputs["library"]),
        environment_script=str(inputs["environment_script"]),
        python_executable=str(inputs["python_executable"]),
        log_dir=str(inputs["log_dir"]),
        ranks=24,
        launch_mode="pbs",
        pbs_account="UCUB0188",
        pbs_queue="develop",
        pbs_walltime="00:10:00",
        startup_timeout=60.0,
        request_timeout=30.0,
        options={
            "timestep_seconds": 1800,
            "physics_profile": "kessler",
            "mediator_present": False,
        },
        scheme_plan=_scheme_plan().to_payload(),
        execution_mode="pbs",
    )


def test_persistent_actor_reuses_one_live_session_for_all_actions(
    tmp_path: Path,
) -> None:
    actor = PersistentCAMActor(_request(tmp_path), session_factory=_FakeSession)
    started = actor.describe()
    assert started["mpi_launch_count"] == 1
    assert started["step"] == 0
    assert actor.delete_variable("temporary_probe") == {
        "standard_name": "temporary_probe",
        "owner": "python",
    }

    result = actor.run_plan(
        SegmentPlan(
            "live-actions",
            (
                FieldEdit("air_temperature", "add", 1.0),
                RunSteps(2),
                ObserveFields(("air_temperature",)),
            ),
        ).as_dict()
    )
    stats = actor.get_field_stats("air_temperature", rank=0)
    checkpoint = actor.checkpoint()
    memory_checkpoint = actor.memory_checkpoint()
    closed = actor.close()

    assert result["step"] == 2
    assert result["native_calls"] == 6
    assert result["mpi_launch_count"] == 1
    assert [row["type"] for row in result["action_trace"]] == [
        "field_edit",
        "run_steps",
        "observe_fields",
    ]
    assert stats["mean"] == 241.0
    assert Path(checkpoint["checkpoint_dir"]).is_dir()
    assert memory_checkpoint.nbytes > 0
    assert closed["mpi_launch_count"] == 1


def test_persistent_actor_validates_full_plan_before_mutation(
    tmp_path: Path,
) -> None:
    actor = PersistentCAMActor(_request(tmp_path), session_factory=_FakeSession)
    with pytest.raises(ValueError, match="unknown model phase"):
        actor.run_plan(
            {
                "schema_version": 1,
                "name": "invalid",
                "unsafe": True,
                "actions": [
                    {
                        "type": "field_edit",
                        "name": "air_temperature",
                        "operation": "add",
                        "value": 1.0,
                    },
                    {"type": "run_phase", "name": "not-a-phase"},
                ],
            }
        )
    assert actor.get_field_stats("air_temperature", rank=0)["mean"] == 240.0
    actor.close()


def test_persistent_actor_runs_dynamic_extension_actions(
    tmp_path: Path,
) -> None:
    actor = PersistentCAMActor(_request(tmp_path), session_factory=_FakeSession)
    variable = VariableSpec(
        "runtime_control",
        "float64",
        ("nphys_local",),
        standard_name="runtime_control",
    )
    plugin = PhysicsPluginSpec(
        "/shared/runtime_probe/device.json",
        placements=(SchemePlacement("runtime_probe"),),
        name="runtime_probe",
    )

    result = actor.run_plan(
        SegmentPlan(
            "dynamic",
            (
                DefineVariable(variable, 2.0),
                InstallPhysics(plugin),
                DeactivatePhysics("runtime_probe"),
                ActivatePhysics("runtime_probe"),
            ),
            unsafe=True,
        ).as_dict()
    )

    assert [item["type"] for item in result["action_trace"]] == [
        "define_variable",
        "install_physics",
        "deactivate_physics",
        "activate_physics",
    ]
    assert actor.describe()["plugins"][0]["active"] is True
    actor.close()


class _EchoPersistentActor:
    def __init__(
        self,
        request: PersistentDaskRequest,
        snapshot: CheckpointBundle | None = None,
    ) -> None:
        self.name = request.name
        self.parent_name = request.parent_name
        self.snapshot_transport = "initialization"
        if snapshot is None:
            self.step_count = 0
        else:
            payload = json.loads(dict(snapshot.files)["echo.json"])
            self.step_count = int(payload["step"])
            self.snapshot_transport = "memory"
        if request.startup_plan is not None:
            plan = SegmentPlan.from_mapping(request.startup_plan)
            self.step_count += plan.step_count

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent_name": self.parent_name,
            "running": True,
            "ranks": 24,
            "step": self.step_count,
            "native_calls": self.step_count,
            "mpi_launch_count": 1,
            "worker_host": "test-worker",
            "worker_pid": 123,
            "launch_mode": "pbs",
            "pbs_job_id": "123.server",
            "outer_pbs_job_id": None,
            "field_count": 1,
            "snapshot_transport": self.snapshot_transport,
            "run_dir": "/tmp/run",
            "history_dir": "/tmp/history",
            "log_path": "/tmp/model.log",
        }

    def step(self, count: int = 1) -> dict[str, Any]:
        self.step_count += count
        return {
            "step": self.step_count,
            "mpi_launch_count": 1,
        }

    def close(self) -> dict[str, Any]:
        return {"closed": True, "mpi_launch_count": 1}

    def memory_checkpoint(self) -> CheckpointBundle:
        return CheckpointBundle(
            (
                (
                    "echo.json",
                    json.dumps({"step": self.step_count}).encode("utf-8"),
                ),
            )
        )


def test_dask_client_pins_persistent_actor_and_returns_actor_futures(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            persistent_actor_factory=_EchoPersistentActor,
            **_inputs(tmp_path),
        )
        model = experiments.start_persistent("interactive")
        assert isinstance(model, PersistentDaskSession)
        assert model.worker in client.scheduler_info()["workers"]
        assert model.describe().result()["mpi_launch_count"] == 1
        assert model.step(3).result() == {
            "step": 3,
            "mpi_launch_count": 1,
        }
        assert model.describe().result()["step"] == 3
        assert model.close().result()["closed"] is True


def test_dask_client_keeps_legacy_blocking_persistent_model(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            persistent_actor_factory=_EchoPersistentActor,
            **_inputs(tmp_path),
        )
        with experiments.open_persistent("blocking") as model:
            assert isinstance(model, BlockingModel)
            assert model.status.mpi_launch_count == 1
            assert model.advance(steps=2) is model
            asynchronous = model.submit.step(3)
            assert asynchronous.result()["step"] == 5
            assert model.step_count == 5


def test_dask_client_forks_independent_persistent_actors_from_memory(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=3,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            persistent_actor_factory=_EchoPersistentActor,
            **_inputs(tmp_path),
        )
        base = experiments.open_persistent("base")
        assert base.step(10)["step"] == 10
        assert base.describe()["step"] == 10
        worker_names = tuple(sorted(client.scheduler_info()["workers"]))
        control = experiments.plan("control").step(2)
        experiment = experiments.plan("experiment").step(3)
        children = experiments.fork_models(
            base,
            (control, experiment),
            workers={
                "control": worker_names[1],
                "experiment": worker_names[2],
            },
            close_parent=True,
        )
        statuses = {
            name: child.describe()
            for name, child in children.items()
        }

        assert statuses["control"]["step"] == 12
        assert statuses["experiment"]["step"] == 13
        assert statuses["control"]["parent_name"] == "base"
        assert statuses["experiment"]["parent_name"] == "base"
        assert statuses["control"]["snapshot_transport"] == "memory"
        with pytest.raises(RuntimeError, match="closed"):
            base.describe()
        assert not tuple((tmp_path / "runs").glob("**/checkpoints"))

        for child in children.values():
            child.close()
