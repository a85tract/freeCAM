"""Controller for a single MPI launch containing multiple live model slots."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shlex
import shutil
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import yaml

from ..model import (
    CCPPSuitePlan,
    ModelConfig,
    ModelOptions,
    PhysicsPluginSpec,
    SegmentPlan,
    VariableSpec,
)
from .session import NotebookSession, NotebookWorkerError


class PooledWorkerSession(NotebookSession):
    """Launch and control one partitioned persistent MPI world.

    This is intentionally a low-level surface for ``PersistentPoolActor``.
    Individual model handles live above it; this object owns the one socket,
    one MPI launch, and the model-name-to-slot table.
    """

    def __init__(
        self,
        config: str | Path | ModelConfig,
        *,
        run_root: str | Path,
        library: str | Path | None = None,
        ranks_per_model: int,
        model_slots: int,
        initial_run_dir: str | Path | None = None,
        env_script: str | Path | None = None,
        launcher: str | Sequence[str] = "mpiexec",
        hosts: str | Sequence[str] | None = None,
        launch_mode: str = "auto",
        pbs_account: str = "UCUB0188",
        pbs_queue: str = "develop",
        pbs_walltime: str = "00:30:00",
        python_executable: str | Path | None = None,
        startup_timeout: float = 900.0,
        request_timeout: float = 600.0,
        log_path: str | Path | None = None,
        options: ModelOptions | None = None,
        scheme_plan: CCPPSuitePlan | None = None,
        pool_name: str = "cam-pool",
        resource_plan: Mapping[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        self.pool_name = str(pool_name)
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.initial_run_dir = (
            None if initial_run_dir is None else Path(initial_run_dir).resolve()
        )
        self.ranks_per_model = int(ranks_per_model)
        self.model_slots = int(model_slots)
        if self.ranks_per_model <= 0 or self.model_slots <= 0:
            raise ValueError("ranks_per_model and model_slots must be positive")
        self.world_size = self.ranks_per_model * self.model_slots
        self.resource_plan = dict(resource_plan or {})
        if self.resource_plan:
            if int(self.resource_plan["world_size"]) != self.world_size:
                raise ValueError(
                    "resource plan world_size differs from pooled session"
                )
            if int(self.resource_plan["ranks_per_model"]) != self.ranks_per_model:
                raise ValueError(
                    "resource plan ranks_per_model differs from pooled session"
                )
            if int(self.resource_plan["model_slots"]) != self.model_slots:
                raise ValueError(
                    "resource plan model_slots differs from pooled session"
                )
        source_config = (
            config
            if isinstance(config, ModelConfig)
            else ModelConfig.from_yaml(config)
        )
        pooled_config = source_config.with_overrides(
            mpi_size=self.ranks_per_model
        )
        control_dir = self.run_root / f".pool-control-{self.pool_name}"
        control_dir.mkdir(parents=True, exist_ok=True)
        placeholder = control_dir / "atm_in"
        if not placeholder.exists():
            placeholder.write_text("&cam_initfiles_nl /\n", encoding="utf-8")

        super().__init__(
            pooled_config,
            run_dir=control_dir,
            library=library,
            ranks=self.ranks_per_model,
            env_script=env_script,
            launcher=launcher,
            hosts=hosts,
            launch_mode=launch_mode,
            pbs_account=pbs_account,
            pbs_queue=pbs_queue,
            pbs_walltime=pbs_walltime,
            python_executable=python_executable,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            log_path=log_path,
            options=options,
            scheme_plan=scheme_plan,
        )
        self.runtime = "model-pool"
        # NotebookSession validates one model's rank count.  The external
        # launch contains every fixed slot.
        self.ranks = self.world_size
        self._models: dict[str, int] = {}
        self._model_paths: dict[str, tuple[Path, Path]] = {}
        self._slots: tuple[dict[str, Any], ...] = ()

    @property
    def slots(self) -> tuple[dict[str, Any], ...]:
        if self.running:
            self.describe()
        return tuple(dict(item) for item in self._slots)

    def start(self) -> "PooledWorkerSession":
        if self.running or self._connection is not None:
            raise RuntimeError("PooledWorkerSession is already running")
        self.options.validate(self.config)
        if self.config_path is None:
            self._ephemeral_config_path = (
                self.run_dir / f".pycam-pool-{secrets.token_hex(6)}.yaml"
            )
            self._ephemeral_config_path.write_text(
                yaml.safe_dump(self.config.as_dict(), sort_keys=False),
                encoding="utf-8",
            )
            self.config_path = self._ephemeral_config_path

        environment = self._worker_environment()
        environment["PYTHONUNBUFFERED"] = "1"
        launch_mode = self._resolve_launch_mode(environment)
        self._launch_mode_used = launch_mode
        authkey = secrets.token_bytes(32)
        listener = Listener(
            ("0.0.0.0" if launch_mode == "pbs" else "127.0.0.1", 0),
            authkey=authkey,
        )
        self._listener = listener
        _, port = listener.address
        host = socket.getfqdn() if launch_mode == "pbs" else "127.0.0.1"
        launcher = (
            list(self.launcher)
            if launch_mode == "pbs"
            else self._launcher_command(environment)
        )
        command = [
            *launcher,
            "-n",
            str(self.world_size),
            self.python_executable,
            "-m",
            "pycam_sima.cli",
            "pool-worker",
            "--host",
            host,
            "--port",
            str(port),
            "--authkey",
            base64.urlsafe_b64encode(authkey).decode("ascii"),
            "--ranks-per-model",
            str(self.ranks_per_model),
            "--model-slots",
            str(self.model_slots),
            "--config",
            str(self.config_path),
            "--library",
            str(self.library),
            "--timestep-seconds",
            str(self.options.timestep_seconds),
            "--physics-profile",
            self.options.physics_profile,
            "--scheme-plan-json",
            json.dumps(self._scheme_plan.to_payload(), separators=(",", ":")),
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if launch_mode == "pbs":
                self._submit_pbs_worker(command, environment)
            else:
                self._log_handle = self.log_path.open("ab", buffering=0)
                self._process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            self._connection = self._accept_worker(listener, self.startup_timeout)
            ready = self._unwrap(self._receive(self.startup_timeout))
            expected = {
                "event": "ready",
                "runtime": self.runtime,
                "world_size": self.world_size,
                "ranks_per_model": self.ranks_per_model,
                "model_slots": self.model_slots,
                "mpi_launch_count": 1,
            }
            mismatches = {
                key: (ready.get(key), value)
                for key, value in expected.items()
                if ready.get(key) != value
            }
            if mismatches:
                raise NotebookWorkerError(
                    f"pooled worker startup differs from request: {mismatches}"
                )
            self.describe()
            return self
        except BaseException:
            self._abort()
            raise
        finally:
            listener.close()
            self._listener = None

    def describe(self) -> dict[str, Any]:
        result = self._request({"op": "status"})
        self._slots = tuple(dict(item) for item in result["slots"])
        self._models = {
            str(item["model_name"]): int(item["slot_id"])
            for item in self._slots
            if item["model_name"] is not None
        }
        return dict(result)

    def create_model(
        self,
        name: str,
        *,
        run_dir: str | Path | None = None,
        history_dir: str | Path | None = None,
        slot: int | None = None,
    ) -> dict[str, Any]:
        if name in self._models:
            raise ValueError(f"pooled model already exists: {name!r}")
        status = self.describe()
        idle = [
            int(item["slot_id"])
            for item in status["slots"]
            if item["state"] == "idle"
        ]
        if slot is None:
            if not idle:
                raise RuntimeError("persistent model pool has no idle slot")
            slot = idle[0]
        if int(slot) not in idle:
            raise RuntimeError(f"persistent model slot {slot} is not idle")
        selected_run, selected_history = self._prepare_model_paths(
            name, run_dir=run_dir, history_dir=history_dir
        )
        result = self._request(
            {
                "op": "create_model",
                "name": name,
                "slot": int(slot),
                "run_dir": str(selected_run),
                "history_dir": str(selected_history),
            }
        )
        self._models[name] = int(slot)
        self._model_paths[name] = (selected_run, selected_history)
        self.describe()
        return dict(result)

    def call(
        self, model_name: str, op: str, *args: Any, **payload: Any
    ) -> Any:
        slot = self._model_slot(model_name)
        if op in {"close", "close_pool", "restore_memory_checkpoint"}:
            raise ValueError(f"{op!r} is not a model command in pooled mode")
        command = self._model_command(model_name, op, args, payload)
        return self._request(
            {
                "op": "model_command",
                "slot": slot,
                "name": model_name,
                "command": command,
            }
        )

    def step(self, name: str, count: int = 1) -> dict[str, Any]:
        if int(count) <= 0:
            raise ValueError("step count must be positive")
        return dict(self.call(name, "step", count=int(count)))

    def advance_models(
        self, names: Sequence[str], count: int = 1
    ) -> dict[str, Any]:
        """Advance distinct slots concurrently with one controller command."""

        if int(count) < 0:
            raise ValueError("step count cannot be negative")
        selected = tuple(str(name) for name in names)
        if len(set(selected)) != len(selected):
            raise ValueError("advance_models names must be unique")
        if int(count) == 0:
            statuses = {
                str(item["model_name"]): dict(item)
                for item in self.describe()["slots"]
                if item["model_name"] is not None
            }
            return {name: statuses[name] for name in selected}
        commands = [
            {
                "slot": self._model_slot(name),
                "name": name,
                "command": {"op": "step", "count": int(count)},
            }
            for name in selected
        ]
        by_slot = self._request({"op": "model_commands", "commands": commands})
        return {
            name: by_slot[str(self._model_slot(name))]
            for name in selected
        }

    def call_models(
        self,
        calls: Sequence[
            tuple[str, str, Sequence[Any], Mapping[str, Any]]
        ],
    ) -> dict[str, Any]:
        """Execute one command per distinct model in one MPI-world broadcast."""

        selected = tuple(calls)
        names = tuple(str(item[0]) for item in selected)
        if len(names) != len(set(names)):
            raise ValueError("call_models accepts at most one command per model")
        commands = [
            {
                "slot": self._model_slot(name),
                "name": name,
                "command": self._model_command(
                    name,
                    str(operation),
                    tuple(args),
                    dict(kwargs),
                ),
            }
            for name, operation, args, kwargs in selected
        ]
        by_slot = self._request({"op": "model_commands", "commands": commands})
        return {
            name: by_slot[str(self._model_slot(name))]
            for name in names
        }

    def fork_model(
        self,
        parent: str,
        children: Sequence[str | Mapping[str, Any]],
        *,
        require_concurrent: bool = False,
    ) -> dict[str, Any]:
        del require_concurrent  # All children in this request occupy live slots.
        parent_slot = self._model_slot(parent)
        status = self.describe()
        idle = [
            int(item["slot_id"])
            for item in status["slots"]
            if item["state"] == "idle"
        ]
        if len(children) > len(idle):
            raise RuntimeError(
                f"fork needs {len(children)} idle slots, found {len(idle)}"
            )
        records = []
        paths: dict[str, tuple[Path, Path]] = {}
        parent_run, _parent_history = self._model_paths[parent]
        for value, slot in zip(children, idle):
            item = {"name": value} if isinstance(value, str) else dict(value)
            name = str(item["name"])
            if name in self._models or any(record["name"] == name for record in records):
                raise ValueError(f"duplicate pooled model name: {name!r}")
            run_dir, history_dir = self._prepare_model_paths(
                name,
                run_dir=item.get("run_dir"),
                history_dir=item.get("history_dir"),
                seed_run_dir=parent_run,
            )
            records.append(
                {
                    "name": name,
                    "slot": slot,
                    "run_dir": str(run_dir),
                    "history_dir": str(history_dir),
                }
            )
            paths[name] = (run_dir, history_dir)
        result = self._request(
            {
                "op": "fork_model",
                "parent_name": parent,
                "parent_slot": parent_slot,
                "children": records,
            }
        )
        for record in records:
            self._models[record["name"]] = int(record["slot"])
            self._model_paths[record["name"]] = paths[record["name"]]
        self.describe()
        return dict(result)

    def close_model(self, name: str) -> dict[str, Any]:
        slot = self._model_slot(name)
        result = self._request(
            {"op": "close_model", "slot": slot, "name": name}
        )
        self._models.pop(name, None)
        self._model_paths.pop(name, None)
        self.describe()
        return dict(result)

    def close(self) -> None:
        if self._connection is None and self._process is None and self._job_id is None:
            return
        error: BaseException | None = None
        if self._connection is not None and self.running:
            try:
                self._request({"op": "close_pool"})
            except BaseException as exc:
                error = exc
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._terminate_process(process)
        self._models.clear()
        self._model_paths.clear()
        self._slots = ()
        self._cleanup_handles()
        if error is not None:
            raise error

    def __enter__(self) -> "PooledWorkerSession":
        return self.start()

    def _model_slot(self, name: str) -> int:
        try:
            return self._models[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown pooled model: {name!r}") from exc

    def _model_command(
        self,
        model_name: str,
        operation: str,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Translate the Pythonic model facade into the compact worker wire API."""

        values = dict(kwargs)
        wire_operations = {
            "describe",
            "step",
            "prepare_initial_step",
            "run_phase",
            "run_scheme",
            "run_scheme_group",
            "configure_scheme_plan",
            "define_variable",
            "delete_variable",
            "install_physics",
            "activate_physics",
            "deactivate_physics",
            "write_checkpoint",
            "capture_memory_checkpoint",
            "edit_field",
            "get_field",
            "get_field_stats",
            "set_field",
        }
        if not args and operation in wire_operations:
            return {
                "op": operation,
                **values,
                "model_name": model_name,
            }
        if operation == "describe":
            command = {"op": "describe"}
        elif operation == "step":
            command = {
                "op": "step",
                "count": int(args[0] if args else values.pop("count", 1)),
            }
        elif operation == "prepare_initial_step":
            command = {"op": operation}
        elif operation == "run_phase":
            command = {"op": operation, "phase": str(args[0])}
        elif operation == "run_scheme":
            command = {
                "op": operation,
                "scheme": str(args[0]),
                "group": values.pop("group", None),
            }
        elif operation == "run_scheme_group":
            command = {"op": operation, "group": str(args[0])}
        elif operation in {"get_field", "get_field_stats"}:
            command = {
                "op": operation,
                "field": str(args[0]),
                "rank": values.pop("rank", 0),
            }
        elif operation == "field_info":
            command = {"op": operation, "field": str(args[0])}
        elif operation == "set_field":
            command = {
                "op": operation,
                "field": str(args[0]),
                "value": args[1],
                "rank": values.pop("rank", 0),
                "unsafe": bool(values.pop("unsafe", False)),
            }
        elif operation == "edit_field":
            command = {
                "op": operation,
                "field": str(args[0]),
                "operation": str(args[1]),
                "value": float(args[2]),
                "unsafe": bool(values.pop("unsafe", False)),
            }
        elif operation == "define_variable":
            spec = args[0]
            if not isinstance(spec, VariableSpec):
                raise TypeError("define_variable requires VariableSpec")
            command = {
                "op": operation,
                "spec": spec.as_dict(),
                "initial_value": values.pop("initial", 0.0),
            }
        elif operation == "delete_variable":
            command = {
                "op": operation,
                "name": str(args[0]),
            }
        elif operation == "install_physics":
            spec = args[0]
            if not isinstance(spec, PhysicsPluginSpec):
                spec = PhysicsPluginSpec(str(spec))
            command = {
                "op": operation,
                "plugin": spec.as_dict(),
                "initial_values": dict(values.pop("initial_values", None) or {}),
                "effective": str(values.pop("effective", "now")),
                "unsafe": bool(values.pop("unsafe", False)),
            }
        elif operation in {"activate_physics", "deactivate_physics"}:
            command = {
                "op": operation,
                "name": str(args[0]),
                "unsafe": bool(values.pop("unsafe", False)),
            }
        elif operation == "set_scheme_enabled":
            command = {
                "op": operation,
                "scheme": str(args[0]),
                "enabled": bool(args[1]),
                "group": values.pop("group", None),
                "unsafe": bool(values.pop("unsafe", False)),
            }
        elif operation == "move_scheme":
            command = {
                "op": operation,
                "scheme": str(args[0]),
                "before": values.pop("before", None),
                "after": values.pop("after", None),
                "group": values.pop("group", None),
                "to_group": values.pop("to_group", None),
                "unsafe": bool(values.pop("unsafe", False)),
            }
        elif operation in {"reset_scheme_plan", "describe_scheme_plan"}:
            command = {"op": operation}
            if operation == "describe_scheme_plan":
                command["group"] = args[0] if args else None
        elif operation == "run_plan":
            plan = args[0]
            if not isinstance(plan, SegmentPlan):
                plan = SegmentPlan.from_mapping(plan)
            command = {"op": operation, "plan": plan.as_dict()}
        elif operation in {"checkpoint", "write_checkpoint"}:
            requested = args[0] if args else None
            path = (
                self.run_root
                / model_name
                / "checkpoints"
                / f"checkpoint-{time.time_ns()}"
                if requested is None
                else Path(requested).resolve()
            )
            command = {"op": "write_checkpoint", "path": str(path)}
        elif operation in {"memory_checkpoint", "snapshot"}:
            command = {"op": "capture_memory_checkpoint"}
        else:
            raise ValueError(f"unknown pooled model operation: {operation!r}")
        if values:
            raise TypeError(
                f"unexpected arguments for {operation}: {sorted(values)}"
            )
        command["model_name"] = model_name
        return command

    def _prepare_model_paths(
        self,
        name: str,
        *,
        run_dir: str | Path | None,
        history_dir: str | Path | None,
        seed_run_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        branch = self.run_root / name
        selected_run = Path(run_dir or branch / "run").resolve()
        selected_history = Path(history_dir or branch / "history").resolve()
        selected_run.mkdir(parents=True, exist_ok=True)
        atm_in = selected_run / "atm_in"
        if not atm_in.is_file():
            candidates = [
                None if seed_run_dir is None else seed_run_dir / "atm_in",
                (
                    None
                    if self.initial_run_dir is None
                    else self.initial_run_dir / "atm_in"
                ),
                self.run_root / "atm_in",
            ]
            source = next(
                (candidate for candidate in candidates if candidate and candidate.is_file()),
                None,
            )
            if source is None:
                raise FileNotFoundError(
                    f"cannot prepare {atm_in}; provide initial_run_dir or an "
                    "existing run_dir containing atm_in"
                )
            shutil.copy2(source, atm_in)
        if selected_history.exists():
            raise FileExistsError(
                f"refusing to replace existing history directory: {selected_history}"
            )
        return selected_run, selected_history

    def _submit_pbs_worker(
        self,
        worker_command: Sequence[str],
        environment: Mapping[str, str],
    ) -> None:
        """Submit one PBS job containing all slots, never one job per model."""

        script = self.run_dir / f".pycam-pool-{secrets.token_hex(6)}.pbs"
        source = (
            f"source {shlex.quote(str(self.env_script))} >/dev/null 2>&1\n"
            if self.env_script is not None
            else ""
        )
        command = " ".join(shlex.quote(value) for value in worker_command)
        pbs_select = self.resource_plan.get("pbs_select")
        if not pbs_select:
            raise ValueError(
                "PBS pooled launch requires ResourcePlan.pbs_select; "
                "use allocation mode or pass the plan returned by plan_pool()"
            )
        script.write_text(
            "#!/bin/bash\n"
            f"#PBS -N {self.pool_name[:15]}\n"
            f"#PBS -A {self.pbs_account}\n"
            f"#PBS -q {self.pbs_queue}\n"
            f"#PBS -l {pbs_select}\n"
            f"#PBS -l walltime={self.pbs_walltime}\n"
            "#PBS -j oe\n"
            f"#PBS -o {self.log_path}\n"
            "set -euo pipefail\n"
            f"{source}"
            f"export LD_LIBRARY_PATH={shlex.quote(environment['LD_LIBRARY_PATH'])}\n"
            f"exec {command}\n",
            encoding="utf-8",
        )
        self._pbs_script = script
        try:
            result = subprocess.run(
                ["qsub", str(script)],
                check=True,
                capture_output=True,
                text=True,
                env=dict(environment),
            )
        except subprocess.CalledProcessError as exc:
            raise NotebookWorkerError(
                f"cannot submit pooled CAM worker: {exc.stderr.strip()}"
            ) from exc
        self._job_id = result.stdout.strip().splitlines()[-1]
        print(
            f"PyCAM-SIMA pool submitted as {self._job_id}; waiting for "
            f"{self.model_slots} x {self.ranks_per_model} MPI ranks ...",
            flush=True,
        )
