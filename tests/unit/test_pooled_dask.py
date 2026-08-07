from __future__ import annotations

from pathlib import Path
import socket
from typing import Any

from distributed import Client
import numpy as np
import pytest

from freecam import (
    DaskExperimentClient,
    InstalledPythonProcess,
    PersistentPoolActor,
    PhysicsPluginSpec,
    PooledDaskRequest,
    PooledModelGroup,
    PythonProcessSpec,
)


ROOT = Path(__file__).resolve().parents[2]


class _FakePooledSession:
    launches = 0
    batches: list[tuple[str, ...]] = []

    def __init__(self, _config: str, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.ranks = int(kwargs["ranks_per_model"])
        self.slot_count = int(kwargs["model_slots"])
        self.running = False
        self.models: dict[str, dict[str, Any]] = {}
        self.slot_names: list[str | None] = [None] * self.slot_count
        self.retained: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        type(self).launches += 1
        self.running = True

    def describe(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "world_size": self.ranks * self.slot_count,
            "models": tuple(self.models),
        }

    @property
    def slots(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for index, name in enumerate(self.slot_names):
            row = {
                "slot_id": index,
                "state": "idle" if name is None else "ready",
                "model_name": name,
                "ranks": tuple(range(index * self.ranks, (index + 1) * self.ranks)),
                "state_bytes": 0 if name is None else self.ranks * 32,
            }
            if name is not None:
                row.update(
                    step=self.models[name]["step"],
                    native_calls=self.models[name]["native_calls"],
                )
            rows.append(row)
        return tuple(rows)

    def create_model(
        self,
        name: str,
        *,
        run_dir: Path,
        history_dir: Path,
        slot: int | None,
    ) -> dict[str, Any]:
        if name in self.models:
            raise ValueError(name)
        selected = self.slot_names.index(None) if slot is None else int(slot)
        if self.slot_names[selected] is not None:
            raise RuntimeError("occupied")
        self.slot_names[selected] = name
        self.models[name] = {
            "slot": selected,
            "step": 0,
            "native_calls": 0,
            "temperature": [
                np.full((2, 2), 240.0 + rank) for rank in range(self.ranks)
            ],
            "kessler": True,
            "python_processes": {},
            "run_dir": str(run_dir),
            "history_dir": str(history_dir),
        }
        return self._status(name)

    def call(self, model_name: str, op: str, **kwargs: Any) -> Any:
        name = model_name
        model = self.models[name]
        if op == "describe":
            return self._status(name)
        if op == "step":
            count = int(kwargs["count"])
            model["step"] += count
            model["native_calls"] += count
            return self._status(name)
        if op == "write_checkpoint":
            return {
                **self._status(name),
                "checkpoint_dir": str(kwargs["path"]),
                "mpi_launch_count": 1,
            }
        if op in {"run_phase", "run_scheme", "run_scheme_group"}:
            if op == "run_scheme" and kwargs.get("scheme") == "transactional_failure":
                raise RuntimeError("transactional callback failed")
            model["native_calls"] += 1
            return {
                **self._status(name),
                "last_operation": op,
            }
        if op == "get_field":
            rank = kwargs["rank"]
            values = model["temperature"]
            if rank == "all":
                return [value.copy() for value in values]
            return values[int(rank)].copy()
        if op == "get_field_stats":
            rank = int(kwargs["rank"])
            values = model["temperature"][rank]
            return {
                "rank": rank,
                "shape": values.shape,
                "dtype": values.dtype.str,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        if op == "edit_field":
            edit, value = kwargs["operation"], kwargs["value"]
            for values in model["temperature"]:
                if edit == "add":
                    values += value
                elif edit == "multiply":
                    values *= value
                else:
                    values[...] = value
            return self._status(name)
        if op == "configure_scheme_plan":
            rows = kwargs["plan"]["groups"]["physics_before_coupler"]["children"]
            model["kessler"] = next(
                item["scheme"]["enabled"] for item in rows if item["name"] == "kessler"
            )
            return self._status(name)
        if op == "install_physics":
            return {
                **self._status(name),
                "installed_plugin": {
                    "name": "runtime_temperature_offset",
                    "source": kwargs["plugin"]["source"],
                },
            }
        if op == "install_python_process":
            spec = PythonProcessSpec.from_mapping(kwargs["process"])
            model["python_processes"][spec.name] = {
                "spec": spec.as_dict(),
                "scheme_key": spec.name,
                "read_bindings": {},
                "write_bindings": {field: field for field in spec.writes},
            }
            return {
                **self._status(name),
                "installed_python_process": {
                    "name": spec.name,
                    "group": spec.group,
                    "payload_hash": spec.payload_hash,
                    "reads": spec.reads,
                    "writes": spec.writes,
                    "transactional": spec.transactional,
                },
            }
        if op == "remove_python_process":
            removed = model["python_processes"].pop(str(kwargs["name"]))
            return {
                **self._status(name),
                "removed_python_process": {
                    "name": removed["spec"]["name"],
                    "payload_hash": removed["spec"]["payload_hash"],
                },
            }
        if op == "set_python_process_parameters":
            record = model["python_processes"][str(kwargs["name"])]
            spec = PythonProcessSpec.from_mapping(record["spec"])
            spec = spec.with_parameters(kwargs["parameters"])
            record["spec"] = spec.as_dict()
            return {
                **self._status(name),
                "updated_python_process": {
                    "name": spec.name,
                    "parameters": dict(spec.parameters or {}),
                    "payload_hash": spec.payload_hash,
                },
            }
        if op == "delete_variable":
            return {
                **self._status(name),
                "deleted_variable": {
                    "standard_name": kwargs["name"],
                    "owner": "python",
                },
            }
        raise NotImplementedError(op)

    def fork_model(
        self,
        parent: str,
        children: tuple[dict[str, str], ...],
        *,
        require_concurrent: bool,
    ) -> dict[str, Any]:
        available = self.slot_names.count(None)
        if require_concurrent and len(children) > available:
            raise RuntimeError("insufficient slots")
        source = self.models[parent]
        results = {}
        for child in children:
            name = child["name"]
            self.create_model(
                name,
                run_dir=Path(child["run_dir"]),
                history_dir=Path(child["history_dir"]),
                slot=None,
            )
            target = self.models[name]
            target["step"] = source["step"]
            target["native_calls"] = source["native_calls"]
            target["temperature"] = [value.copy() for value in source["temperature"]]
            target["kessler"] = source["kessler"]
            target["python_processes"] = dict(source["python_processes"])
            results[name] = self._status(name)
        return results

    def restore_model(
        self,
        name: str,
        _checkpoint: Path,
        *,
        run_dir: Path,
        history_dir: Path,
        slot: int | None,
    ) -> dict[str, Any]:
        result = self.create_model(
            name,
            run_dir=run_dir,
            history_dir=history_dir,
            slot=slot,
        )
        return {
            **result,
            "snapshot_transport": "checkpoint",
            "scheme_status": {
                "sequence_safe": True,
                "plan": self.kwargs["scheme_plan"].to_payload(),
            },
        }

    def retain_model(
        self,
        name: str,
        *,
        snapshot_id: str,
        label: str,
    ) -> dict[str, Any]:
        source = self.models[name]
        temperatures = [value.copy() for value in source["temperature"]]
        self.retained[snapshot_id] = {
            **source,
            "temperature": temperatures,
            "python_processes": dict(source["python_processes"]),
        }
        nbytes = sum(value.nbytes for value in temperatures)
        return {
            "snapshot_id": snapshot_id,
            "label": label,
            "source_model": name,
            "source_slot": int(source["slot"]),
            "step": int(source["step"]),
            "rank_count": self.ranks,
            "nbytes": nbytes,
            "config_hash": "fake-config",
        }

    def restore_retained(
        self,
        name: str,
        snapshot_id: str,
        *,
        run_dir: Path,
        history_dir: Path,
    ) -> dict[str, Any]:
        source = self.retained[snapshot_id]
        slot = int(source["slot"])
        if self.slot_names[slot] is not None:
            raise RuntimeError("retained source slot is occupied")
        self.slot_names[slot] = name
        self.models[name] = {
            **source,
            "temperature": [value.copy() for value in source["temperature"]],
            "python_processes": dict(source["python_processes"]),
            "run_dir": str(run_dir),
            "history_dir": str(history_dir),
        }
        return {
            **self._status(name),
            "snapshot_transport": "rank-local-memory",
            "retained_snapshot_id": snapshot_id,
            "scheme_status": {
                "sequence_safe": True,
                "plan": self.kwargs["scheme_plan"].to_payload(),
            },
        }

    def drop_retained(self, snapshot_id: str) -> dict[str, Any]:
        source = self.retained.pop(snapshot_id)
        return {
            "snapshot_id": snapshot_id,
            "source_slot": int(source["slot"]),
        }

    @property
    def retained_states(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "snapshot_id": snapshot_id,
                "source_slot": int(source["slot"]),
            }
            for snapshot_id, source in self.retained.items()
        )

    def advance_models(self, names: tuple[str, ...], count: int) -> dict[str, Any]:
        return {name: self.call(name, "step", count=count) for name in names}

    def call_models(
        self,
        calls: tuple[tuple[str, str, tuple[Any, ...], dict[str, Any]], ...]
        | list[tuple[str, str, tuple[Any, ...], dict[str, Any]]],
    ) -> dict[str, Any]:
        type(self).batches.append(tuple(item[0] for item in calls))
        results = {}
        for name, operation, args, kwargs in calls:
            if operation == "step":
                results[name] = self.call(
                    name,
                    operation,
                    count=int(args[0]),
                )
            else:
                results[name] = self.call(name, operation, **kwargs)
        return results

    def close_model(self, name: str) -> dict[str, Any]:
        slot = self.models.pop(name)["slot"]
        self.slot_names[slot] = None
        return {"closed": True, "name": name, "slot_id": slot}

    def close(self) -> None:
        self.models.clear()
        self.retained.clear()
        self.slot_names[:] = [None] * self.slot_count
        self.running = False

    def _status(self, name: str) -> dict[str, Any]:
        model = self.models[name]
        return {
            "name": name,
            "running": True,
            "ranks": self.ranks,
            "step": model["step"],
            "native_calls": model["native_calls"],
            "mpi_launch_count": 1,
            "worker_host": socket.gethostname(),
            "worker_pid": 1,
            "launch_mode": self.kwargs["launch_mode"],
            "pbs_job_id": None,
            "outer_pbs_job_id": None,
            "field_count": 1,
            "snapshot_transport": "pool",
            "run_dir": model["run_dir"],
            "history_dir": model["history_dir"],
            "log_path": str(self.kwargs["log_path"]),
            "phase_names": ("physics_to_dynamics",),
            "scheme_names": ("kessler",),
            "scheme_status": {"sequence_safe": True},
            "python_processes": tuple(model["python_processes"].values()),
            "slot_id": model["slot"],
        }


def _fake_pool_actor(request: PooledDaskRequest) -> PersistentPoolActor:
    return PersistentPoolActor(request, session_factory=_FakePooledSession)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "config.yaml"
    config.write_text("mpi_size: 2\n" f"source_root: {ROOT / 'external/CAM-SIMA'}\n")
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


def _add_rank_local_heating(fields: Any, context: Any) -> None:
    fields["air_temperature"][...] += 0.01 * context.timestep_seconds


def _parameter_heating(
    fields: Any,
    context: Any,
    *,
    increment: float,
) -> None:
    del context
    fields["air_temperature"][...] += increment


def test_model_per_worker_installs_python_process_with_blocking_and_future_api(
    tmp_path: Path,
) -> None:
    _FakePooledSession.launches = 0
    with Client(
        processes=False,
        n_workers=3,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=2,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )
        with experiments.pool("python-processes", resource_plan=plan) as pool:
            with pool.model("base") as base:
                installed = base.physics.install_python(
                    _parameter_heating,
                    name="custom_heating",
                    group="physics_before_coupler",
                    after="kessler",
                    writes=("air_temperature",),
                    parameters={"increment": 1.0},
                )
                assert isinstance(installed, InstalledPythonProcess)
                assert installed.writes == ("air_temperature",)
                installed.run(increment=2.0)
                assert installed.parameters["increment"] == 1.0
                installed.parameters["increment"] = 0.5
                assert installed.parameters["increment"] == 0.5
                installed.disable()
                installed.enable()

                child = base.fork("child").child
                inherited = child.status.details["python_processes"]
                assert inherited[0]["spec"]["name"] == "custom_heating"
                assert inherited[0]["spec"]["parameters"] == {"increment": 0.5}
                child.close()

                assert installed.remove()["name"] == "custom_heating"

                installed_future = base.submit.install_python_process(
                    _add_rank_local_heating,
                    name="future_heating",
                    writes=("air_temperature",),
                    after="kessler",
                )
                installed_metadata = installed_future.result()
                assert installed_metadata["name"] == "future_heating"
                removed_future = base.submit.remove_python_process(
                    "future_heating",
                    depends_on=installed_future,
                )
                assert removed_future.result()["name"] == "future_heating"

        assert _FakePooledSession.launches == 1


def test_pool_reuses_one_launch_and_forks_private_slot_memory(
    tmp_path: Path,
) -> None:
    _FakePooledSession.launches = 0
    _FakePooledSession.batches = []
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=3,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )
        with experiments.pool(
            "science",
            resource_plan=plan,
            actor_layout="legacy-single-worker",
        ) as pool:
            with pool.model("base") as base:
                base.advance(2)
                installed = base.install_physics(
                    PhysicsPluginSpec("runtime-device.yaml"),
                    unsafe=True,
                )
                assert installed == {
                    "name": "runtime_temperature_offset",
                    "source": "runtime-device.yaml",
                }
                assert base.fields.delete("temporary_probe") == {
                    "standard_name": "temporary_probe",
                    "owner": "python",
                }
                branches = base.fork("control", "warm")
                assert isinstance(branches, PooledModelGroup)
                branches.warm.fields.air_temperature += 1.0
                branches.control.physics.kessler.enabled = False
                branches.advance(3)

                assert base.fields.air_temperature.stats(rank=0)["mean"] == 240.0
                assert (
                    branches.warm.fields.air_temperature.stats(rank=0)["mean"] == 241.0
                )
                assert branches.control.physics.kessler.enabled is False
                assert branches.warm.physics.kessler.enabled is True
                assert base.step_count == 2
                assert branches.control.step_count == 5
                assert pool.status["mpi_launch_count"] == 1
                assert len(pool.slots) == 3
                branches.close()

        assert _FakePooledSession.launches == 1


def test_pool_rejects_concurrent_fork_larger_than_free_slots(
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
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=2,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=4,
            memory_per_node="128GB",
        )
        with experiments.pool(
            "limited",
            resource_plan=plan,
            actor_layout="legacy-single-worker",
        ) as pool:
            with pool.model("base") as base:
                with pytest.raises(RuntimeError, match="insufficient slots"):
                    base.fork(
                        "one",
                        "two",
                        require_concurrent=True,
                    )


def test_pool_restores_checkpoint_into_idle_slot_without_new_launch(
    tmp_path: Path,
) -> None:
    _FakePooledSession.launches = 0
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text("{}")
    with Client(
        processes=False,
        n_workers=3,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=2,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )
        with experiments.pool("restore", resource_plan=plan) as pool:
            with pool.model("base") as base:
                base.advance(2)
            with pool.restore("restarted", checkpoint) as restarted:
                assert restarted.slot_id == 0
                assert restarted.status.snapshot_transport == "checkpoint"
                assert pool.status["mpi_launch_count"] == 1

        assert _FakePooledSession.launches == 1


def test_pooled_model_save_returns_typed_checkpoint_metadata(
    tmp_path: Path,
) -> None:
    _FakePooledSession.launches = 0
    with Client(
        processes=False,
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=1,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )
        with experiments.pool("save", resource_plan=plan) as pool:
            with pool.model("base") as base:
                base.advance(steps=3)
                checkpoint = base.save(tmp_path / "saved")

                assert checkpoint.path == (tmp_path / "saved")
                assert checkpoint.step == 3
                assert checkpoint.native_calls == 3
                assert checkpoint.mpi_launch_count == 1


def test_blocking_model_can_continue_after_observed_command_failure(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=1,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )
        with experiments.pool("recover", resource_plan=plan) as pool:
            with pool.model("base") as base:
                with pytest.raises(RuntimeError, match="transactional callback failed"):
                    base.physics.scheme("transactional_failure").run()

                assert base.status.step == 0
                base.advance(steps=1)
                assert base.status.step == 1


def test_model_per_worker_pool_exposes_dask_future_dependencies(
    tmp_path: Path,
) -> None:
    _FakePooledSession.launches = 0
    with Client(
        processes=False,
        n_workers=4,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=3,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )
        with experiments.pool("actors", resource_plan=plan) as pool:
            base = pool.model("base")
            first = base.submit.advance(steps=2)
            scheme = base.submit.run_scheme(
                "kessler",
                group="physics_before_coupler",
                unsafe=True,
                depends_on=first,
            )
            phase = base.submit.run_phase(
                "physics_to_dynamics",
                unsafe=True,
                depends_on=scheme,
            )
            second = base.submit.fields.air_temperature.stats(
                rank=0,
                depends_on=phase,
            )
            assert second.result()["mean"] == 240.0

            branches = base.fork("control", "warm", depends_on=second)
            branches.warm.fields.air_temperature += 1.0
            control = branches.control.submit.advance(steps=3)
            warm = branches.warm.submit.advance(steps=3)
            client.gather((control, warm))
            combined = pool.advance(
                (base, branches.control, branches.warm),
                steps=1,
            )
            assert set(combined) == {"base", "control", "warm"}

            assert base.worker != branches.control.worker
            assert base.worker != branches.warm.worker
            assert branches.control.worker != branches.warm.worker
            assert base.slot_id == 0
            assert branches.control.slot_id == 1
            assert branches.warm.slot_id == 2
            assert base.status.details["dask_worker"] == base.worker
            assert base.step_count == 3
            assert branches.control.step_count == 6
            assert branches.warm.fields.air_temperature.stats(rank=0)["mean"] == 241.0
            assert pool.status["mpi_launch_count"] == 1
            assert pool.scheduler_status["actor_layout"] == "model-per-worker"
            branches.close()
            replacement = pool.model("replacement")
            assert replacement.slot_id == 1
            assert replacement.worker != base.worker
            replacement.close()
            base.close()

        assert _FakePooledSession.launches == 1


def test_model_per_worker_pool_reports_required_worker_capacity(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=2,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=4,
            memory_per_node="128GB",
        )
        with pytest.raises(RuntimeError, match="need 3, found 2"):
            experiments.pool("too-small", resource_plan=plan)


def test_shared_worker_policy_keeps_new_actor_layout_available(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=2,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=1,
            ranks_per_model=2,
            available_nodes=1,
            cpus_per_node=2,
            memory_per_node="128GB",
        )
        with experiments.pool(
            "shared",
            resource_plan=plan,
            worker_policy="shared",
        ) as pool:
            with pool.model("base") as base:
                assert base.worker == pool.worker
                assert base.submit.advance(steps=1).result()["step"] == 1


def test_model_hides_default_single_slot_runtime_planning(
    tmp_path: Path,
) -> None:
    _FakePooledSession.launches = 0
    with Client(
        processes=False,
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )

        with experiments.model("base") as base:
            assert base.runtime.resource_plan.model_slots == 1
            assert base.runtime.resource_plan.retained_snapshots == 1
            assert base.runtime.resource_plan.world_size == 2
            assert base.advance(steps=2) is base
            assert base.status.step == 2
            assert base.runtime.status["mpi_launch_count"] == 1

        assert base.runtime._closed
        assert _FakePooledSession.launches == 1


def test_single_model_runtime_rejects_multi_slot_resource_plan(
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
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=2,
            ranks_per_model=2,
            retained_snapshots=1,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )

        with pytest.raises(
            ValueError,
            match="resource_plan.model_slots == 1",
        ):
            experiments.runtime(resource_plan=plan)


def test_single_slot_retained_state_restores_reusable_private_branches(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            pool_actor_factory=_fake_pool_actor,
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            ranks_per_model=2,
            retained_snapshots=1,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="128GB",
        )
        with experiments.pool("retained", resource_plan=plan) as pool:
            base = pool.model("base")
            base.advance(steps=5)
            installed = base.physics.install_python(
                _add_rank_local_heating,
                name="retained_heating",
                group="physics_before_coupler",
                after="kessler",
                writes=("air_temperature",),
            )
            assert installed.name == "retained_heating"
            expected = base.fields.air_temperature.get(rank=0)
            state = base.retain("after-step-5")
            assert state.step == 5
            assert state.rank_count == 2
            assert state.nbytes > 0
            assert pool.retained_states == (state,)
            base.close()

            with pool.restore_retained("control", state) as control:
                assert control.status.snapshot_transport == "rank-local-memory"
                assert (
                    control.status.details["python_processes"][0]["spec"]["name"]
                    == "retained_heating"
                )
                assert np.array_equal(
                    control.fields.air_temperature.get(rank=0), expected
                )
                control.advance(steps=1)

            with pool.restore_retained("warm", state) as warm:
                assert np.array_equal(warm.fields.air_temperature.get(rank=0), expected)
                warm.fields.air_temperature += 1.0
                assert not np.array_equal(
                    warm.fields.air_temperature.get(rank=0), expected
                )

            state.close()
            assert pool.retained_states == ()
            with pytest.raises(RuntimeError, match="closed"):
                pool.restore_retained("late", state)
