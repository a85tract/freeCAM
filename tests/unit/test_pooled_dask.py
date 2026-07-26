from __future__ import annotations

from pathlib import Path
import socket
from typing import Any

from distributed import Client
import numpy as np
import pytest

from pycam_sima import (
    DaskExperimentClient,
    PersistentPoolActor,
    PhysicsPluginSpec,
    PooledDaskRequest,
    PooledModelGroup,
)


ROOT = Path(__file__).resolve().parents[2]


class _FakePooledSession:
    launches = 0

    def __init__(self, _config: str, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.ranks = int(kwargs["ranks_per_model"])
        self.slot_count = int(kwargs["model_slots"])
        self.running = False
        self.models: dict[str, dict[str, Any]] = {}
        self.slot_names: list[str | None] = [None] * self.slot_count

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
                "ranks": tuple(
                    range(index * self.ranks, (index + 1) * self.ranks)
                ),
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
        selected = (
            self.slot_names.index(None) if slot is None else int(slot)
        )
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
            "run_dir": str(run_dir),
            "history_dir": str(history_dir),
        }
        return self._status(name)

    def call(self, name: str, op: str, **kwargs: Any) -> Any:
        model = self.models[name]
        if op == "step":
            count = int(kwargs["count"])
            model["step"] += count
            model["native_calls"] += count
            return self._status(name)
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
            rows = kwargs["plan"]["groups"]["physics_before_coupler"][
                "children"
            ]
            model["kessler"] = next(
                item["scheme"]["enabled"]
                for item in rows
                if item["name"] == "kessler"
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
            target["temperature"] = [
                value.copy() for value in source["temperature"]
            ]
            target["kessler"] = source["kessler"]
            results[name] = self._status(name)
        return results

    def advance_models(
        self, names: tuple[str, ...], count: int
    ) -> dict[str, Any]:
        return {
            name: self.call(name, "step", count=count)
            for name in names
        }

    def close_model(self, name: str) -> dict[str, Any]:
        slot = self.models.pop(name)["slot"]
        self.slot_names[slot] = None
        return {"closed": True, "name": name, "slot_id": slot}

    def close(self) -> None:
        self.models.clear()
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
            "slot_id": model["slot"],
        }


def _fake_pool_actor(request: PooledDaskRequest) -> PersistentPoolActor:
    return PersistentPoolActor(request, session_factory=_FakePooledSession)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "config.yaml"
    config.write_text(
        "mpi_size: 2\n"
        f"source_root: {ROOT / 'external/CAM-SIMA'}\n"
    )
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


def test_pool_reuses_one_launch_and_forks_private_slot_memory(
    tmp_path: Path,
) -> None:
    _FakePooledSession.launches = 0
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
        with experiments.pool("science", resource_plan=plan) as pool:
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
                branches = base.fork("control", "warm")
                assert isinstance(branches, PooledModelGroup)
                branches.warm.fields.air_temperature += 1.0
                branches.control.physics.kessler.enabled = False
                branches.advance(3)

                assert base.fields.air_temperature.stats(rank=0)["mean"] == 240.0
                assert branches.warm.fields.air_temperature.stats(rank=0)[
                    "mean"
                ] == 241.0
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
        with experiments.pool("limited", resource_plan=plan) as pool:
            with pool.model("base") as base:
                with pytest.raises(RuntimeError, match="insufficient slots"):
                    base.fork(
                        "one",
                        "two",
                        require_concurrent=True,
                    )
