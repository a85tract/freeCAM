from __future__ import annotations

import base64
import json
import os
import queue
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..core.runtime_env import mpi_loader_environment
from ..model import (
    KesslerSchemePlan,
    ModelConfig,
    ModelOptions,
    ModelParameters,
)


class NotebookWorkerError(RuntimeError):
    """An error reported by the external MPI CAM-SIMA worker."""


class NotebookSchemePlan:
    """Notebook-facing editor for the plan shared by every MPI worker."""

    def __init__(self, session: "NotebookSession") -> None:
        self._session = session

    @property
    def sequence_safe(self) -> bool:
        return self._session._scheme_plan.sequence_safe

    @property
    def keys(self) -> tuple[str, ...]:
        return self._session._scheme_plan.keys

    def describe(self, group: str | None = None) -> list[dict[str, object]]:
        return self._session._scheme_plan.describe(group)

    def enable(self, name: str, *, group: str | None = None) -> None:
        self._mutate("enable", name, group=group)

    def disable(
        self,
        name: str,
        *,
        group: str | None = None,
        unsafe: bool = False,
    ) -> None:
        self._mutate("disable", name, group=group, unsafe=unsafe)

    def move(
        self,
        name: str,
        *,
        before: str | None = None,
        after: str | None = None,
        group: str | None = None,
        unsafe: bool = False,
    ) -> None:
        self._mutate(
            "move",
            name,
            before=before,
            after=after,
            group=group,
            unsafe=unsafe,
        )

    def reset(self) -> None:
        candidate = KesslerSchemePlan.default()
        self._session._install_scheme_plan(candidate)

    def _mutate(self, method: str, *args: Any, **kwargs: Any) -> None:
        candidate = self._session._scheme_plan.copy()
        getattr(candidate, method)(*args, **kwargs)
        self._session._install_scheme_plan(candidate)


class NotebookSession:
    """Interactive controller for a CAM-SIMA MPI worker from one Python process.

    A normal Jupyter kernel is not part of an MPI world.  This class starts a
    separate ``mpiexec`` worker, sends collective model commands to rank zero,
    and returns copied NumPy fields to the notebook over an authenticated
    socket. On a Derecho login node it submits the worker through PBS; inside
    an allocation it launches directly. The worker always runs the Python CAM
    driver whose persistent arrays are owned by NumPy.
    """

    def __init__(
        self,
        config: str | Path | ModelConfig,
        *,
        run_dir: str | Path,
        library: str | Path | None = None,
        history_dir: str | Path | None = None,
        ranks: int | None = None,
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
        scheme_plan: KesslerSchemePlan | None = None,
    ) -> None:
        self.runtime = "model"
        self.run_dir = Path(run_dir).resolve()
        project_root = Path(__file__).resolve().parents[3]
        if isinstance(config, ModelConfig):
            self.config = config
            self.config_path: Path | None = None
        else:
            self.config_path = Path(config).resolve()
            self.config = ModelConfig.from_yaml(self.config_path)
        if options is not None and not isinstance(options, ModelOptions):
            raise TypeError("options must be ModelOptions")
        self.options = options or ModelOptions.from_config(self.config)
        self.options.validate(self.config)
        self.parameters = ModelParameters(self)
        if scheme_plan is not None and not isinstance(
            scheme_plan, KesslerSchemePlan
        ):
            raise TypeError("scheme_plan must be KesslerSchemePlan")
        self._scheme_plan = (
            KesslerSchemePlan.default()
            if scheme_plan is None
            else scheme_plan.copy()
        )
        self.scheme_plan = NotebookSchemePlan(self)
        default_library = project_root / "build" / "libpycam_sima_kernels.so"
        required_ranks = self.config.mpi_size
        if env_script is None:
            env_script = (
                project_root
                / "reference"
                / "cases"
                / "FKESSLER_ne3pg3_gnu_24x50"
                / ".env_mach_specific.sh"
            )

        self.library = Path(library or default_library).resolve()
        self.history_dir = Path(history_dir or self.run_dir / "history").resolve()
        self.ranks = required_ranks if ranks is None else int(ranks)
        if self.ranks != required_ranks:
            raise ValueError(
                f"the validated configuration requires {required_ranks} ranks, "
                f"got {self.ranks}"
            )
        if not (self.run_dir / "atm_in").is_file():
            raise FileNotFoundError(f"NotebookSession run directory lacks atm_in: {self.run_dir}")
        if not self.library.is_file():
            raise FileNotFoundError(f"CAM-SIMA runtime library not found: {self.library}")

        self.env_script = Path(env_script).resolve() if env_script else None
        if self.env_script is not None and not self.env_script.is_file():
            raise FileNotFoundError(f"environment script not found: {self.env_script}")
        self.launcher = tuple(shlex.split(launcher) if isinstance(launcher, str) else launcher)
        if not self.launcher:
            raise ValueError("launcher cannot be empty")
        if isinstance(hosts, str):
            self.hosts = tuple(host for host in hosts.split(",") if host)
        else:
            self.hosts = tuple(hosts or ())
        if launch_mode not in {"auto", "local", "pbs"}:
            raise ValueError("launch_mode must be auto, local, or pbs")
        self.launch_mode = launch_mode
        self.pbs_account = pbs_account
        self.pbs_queue = pbs_queue
        self.pbs_walltime = pbs_walltime
        self.python_executable = str(python_executable or sys.executable)
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        if self.startup_timeout <= 0 or self.request_timeout <= 0:
            raise ValueError("session timeouts must be positive")

        if log_path is None:
            log_root = Path(tempfile.gettempdir()) / "pycam-sima"
            log_root.mkdir(parents=True, exist_ok=True)
            stamp = f"{os.getpid()}-{time.time_ns()}"
            self.log_path = log_root / f"notebook-session-{stamp}.log"
        else:
            self.log_path = Path(log_path).resolve()

        self._process: subprocess.Popen[bytes] | None = None
        self._job_id: str | None = None
        self._pbs_script: Path | None = None
        self._launch_mode_used: str | None = None
        self._connection: Connection | None = None
        self._listener: Listener | None = None
        self._log_handle: Any = None
        self._fields: dict[str, dict[str, Any]] = {}
        self._current_step = 0
        self._phase_names: tuple[str, ...] = ()
        self._phase_status: dict[str, Any] = {}
        self._scheme_names: tuple[str, ...] = self._scheme_plan.keys
        self._scheme_status: dict[str, Any] = {
            "last_scheme": None,
            "sequence_safe": self._scheme_plan.sequence_safe,
            "plan": self._scheme_plan.to_payload(),
        }
        self._started_options_fingerprint: tuple[int, str, bool] | None = None
        self.initialized_native_calls: int | None = None
        self.initialized_abi_checked: bool | None = None
        self._ephemeral_config_path: Path | None = None

    @property
    def running(self) -> bool:
        local_running = self._process is not None and self._process.poll() is None
        batch_running = self._job_id is not None
        return self._connection is not None and (local_running or batch_running)

    @property
    def launch_mode_used(self) -> str | None:
        return self._launch_mode_used

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def phase_names(self) -> tuple[str, ...]:
        return self._phase_names

    @property
    def phase_status(self) -> Mapping[str, Any]:
        return dict(self._phase_status)

    @property
    def scheme_names(self) -> tuple[str, ...]:
        return self._scheme_names

    @property
    def scheme_status(self) -> Mapping[str, Any]:
        return dict(self._scheme_status)

    @property
    def next_phase(self) -> str | None:
        value = self._phase_status.get("next_phase")
        return None if value is None else str(value)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self._fields)

    def field_info(self, name: str) -> Mapping[str, Any]:
        try:
            return dict(self._fields[name])
        except KeyError as exc:
            raise KeyError(f"unknown CAM-SIMA field: {name}") from exc

    def start(self) -> "NotebookSession":
        if self.running or self._connection is not None:
            raise RuntimeError("NotebookSession is already running")
        if self.history_dir.exists():
            raise FileExistsError(
                f"refusing to replace existing history directory: {self.history_dir}"
            )

        self.options.validate(self.config)
        if self.config_path is None:
            self._ephemeral_config_path = (
                self.run_dir / f".pycam-model-{secrets.token_hex(6)}.yaml"
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
        bind_host = "0.0.0.0" if launch_mode == "pbs" else "127.0.0.1"
        listener = Listener((bind_host, 0), authkey=authkey)
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
            str(self.ranks),
            self.python_executable,
            "-m",
            "pycam_sima.notebook.worker",
            "--host",
            str(host),
            "--port",
            str(port),
            "--authkey",
            base64.urlsafe_b64encode(authkey).decode("ascii"),
            "--expected-ranks",
            str(self.ranks),
            "--config",
            str(self.config_path),
            "--run-dir",
            str(self.run_dir),
            "--library",
            str(self.library),
            "--history-dir",
            str(self.history_dir),
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
            ready = self._receive(self.startup_timeout)
            result = self._unwrap(ready)
            if result.get("event") != "ready":
                raise NotebookWorkerError(f"unexpected worker startup response: {result!r}")
            if result.get("runtime") != self.runtime:
                raise NotebookWorkerError(
                    f"worker runtime differs from controller: {result.get('runtime')!r}"
                )
            self._fields = dict(result["fields"])
            self._current_step = int(result["step"])
            self._phase_names = tuple(result["phase_names"])
            self._scheme_names = tuple(result["scheme_names"])
            self._update_phase_status(result["phase_status"])
            self._update_scheme_status(result["scheme_status"])
            if self._scheme_plan.to_payload() != result["scheme_status"]["plan"]:
                raise NotebookWorkerError(
                    "worker scheme plan differs from the controller"
                )
            worker_options = dict(result["runtime_options"])
            if worker_options != self.options.describe():
                raise NotebookWorkerError(
                    f"worker runtime options differ from the controller: {worker_options!r}"
                )
            self.initialized_native_calls = result.get("initialized_native_calls")
            self.initialized_abi_checked = result.get("initialized_abi_checked")
            if self.initialized_native_calls != 0:
                raise NotebookWorkerError(
                    "model initialization executed a native kernel"
                )
            if self.initialized_abi_checked is not False:
                raise NotebookWorkerError(
                    "model initialization touched the kernel ABI"
                )
            self._started_options_fingerprint = self.options.fingerprint()
            return self
        except BaseException:
            self._abort()
            raise
        finally:
            listener.close()
            self._listener = None

    def step(self, count: int = 1) -> int:
        count = int(count)
        if count <= 0:
            raise ValueError("step count must be positive")
        self._validate_started_options()
        result = self._request({"op": "step", "count": count})
        self._update_runtime_status(result)
        return self._current_step

    def prepare_initial_step(self) -> Mapping[str, Any]:
        """Prime nstep=0 history without advancing the model clock."""

        self._validate_started_options()
        result = self._request({"op": "prepare_initial_step"})
        self._update_runtime_status(result)
        return self.phase_status

    def run_phase(
        self,
        phase: str,
    ) -> Mapping[str, Any]:
        self._validate_started_options()
        if phase not in self._phase_names:
            raise ValueError(
                f"unknown CAM phase {phase!r}; choose one of {self._phase_names}"
            )
        result = self._request(
            {
                "op": "run_phase",
                "phase": phase,
            }
        )
        self._current_step = int(result["step"])
        self._update_phase_status(result)
        return self.phase_status

    def run_sequence(
        self,
        phases: Sequence[str],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.run_phase(phase) for phase in phases)

    def run_scheme(
        self,
        scheme: str,
        *,
        group: str | None = None,
    ) -> Mapping[str, Any]:
        """Collectively run one scheme on all MPI ranks."""

        self._validate_started_options()
        selected = self._scheme_plan.scheme(scheme, group=group)
        result = self._request(
            {
                "op": "run_scheme",
                "scheme": selected.name,
                "group": selected.group,
            }
        )
        self._update_runtime_status(result)
        return self.scheme_status

    def run_scheme_group(self, group: str) -> Mapping[str, Any]:
        """Collectively run all enabled schemes in one group."""

        self._validate_started_options()
        # Validate before a socket round trip.
        self._scheme_plan.active(group)
        result = self._request({"op": "run_scheme_group", "group": group})
        self._update_runtime_status(result)
        return self.scheme_status

    def run_scheme_sequence(
        self, schemes: Sequence[str]
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.run_scheme(scheme) for scheme in schemes)

    def get_field(self, name: str, *, rank: int | str = 0) -> np.ndarray | list[np.ndarray]:
        self._validate_field(name)
        selector = self._validate_rank(rank)
        value = self._request({"op": "get_field", "field": name, "rank": selector})
        if selector == "all":
            return [np.asarray(piece) for piece in value]
        return np.asarray(value)

    def get_field_stats(self, name: str, *, rank: int | str = 0) -> Any:
        self._validate_field(name)
        selector = self._validate_rank(rank)
        return self._request({"op": "get_field_stats", "field": name, "rank": selector})

    def set_field(
        self,
        name: str,
        value: Any,
        *,
        rank: int | str = 0,
        unsafe: bool = False,
    ) -> None:
        self._validate_field(name)
        selector = self._validate_rank(rank)
        self._request(
            {
                "op": "set_field",
                "field": name,
                "rank": selector,
                "value": np.asarray(value),
                "unsafe": bool(unsafe),
            }
        )

    def close(self) -> None:
        if self._connection is None and self._process is None and self._job_id is None:
            return
        error: BaseException | None = None
        if self._connection is not None and self.running:
            try:
                self._request({"op": "close"})
            except BaseException as exc:
                error = exc
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._terminate_process(process)
        self._cleanup_handles()
        if error is not None:
            raise error

    def __enter__(self) -> "NotebookSession":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise

    def _validate_field(self, name: str) -> None:
        self._ensure_running()
        if name not in self._fields:
            raise KeyError(f"unknown CAM-SIMA field: {name}")

    def _validate_started_options(self) -> None:
        self._ensure_running()
        self.options.validate(self.config)
        if self._started_options_fingerprint != self.options.fingerprint():
            raise RuntimeError(
                "model options changed after initialization; close this "
                "session, create a fresh run directory, and start a new session"
            )

    def _update_phase_status(self, status: Mapping[str, Any]) -> None:
        self._phase_status = dict(status)
        if "step" in status:
            self._current_step = int(status["step"])

    def _update_scheme_status(self, status: Mapping[str, Any]) -> None:
        self._scheme_status = dict(status)
        payload = status.get("plan")
        if payload is not None:
            self._scheme_plan = KesslerSchemePlan.from_payload(payload)
            self._scheme_names = self._scheme_plan.keys

    def _update_runtime_status(self, result: Mapping[str, Any]) -> None:
        self._current_step = int(result["step"])
        self._update_phase_status(result["phase_status"])
        self._update_scheme_status(result["scheme_status"])

    def _install_scheme_plan(self, candidate: KesslerSchemePlan) -> None:
        if self.running:
            self._validate_started_options()
            result = self._request(
                {
                    "op": "configure_scheme_plan",
                    "plan": candidate.to_payload(),
                }
            )
            self._update_runtime_status(result)
            return
        self._scheme_plan = candidate
        self._scheme_names = candidate.keys
        self._scheme_status = {
            "last_scheme": None,
            "sequence_safe": candidate.sequence_safe,
            "plan": candidate.to_payload(),
        }

    def _validate_rank(self, rank: int | str) -> int | str:
        if rank == "all":
            return rank
        if not isinstance(rank, int) or not 0 <= rank < self.ranks:
            raise ValueError(f"rank must be 0..{self.ranks - 1} or 'all'")
        return rank

    def _request(self, request: dict[str, Any]) -> Any:
        self._ensure_running()
        assert self._connection is not None
        try:
            self._connection.send(request)
            response = self._receive(self.request_timeout)
        except (EOFError, BrokenPipeError, OSError) as exc:
            details = self._log_tail()
            self._abort()
            raise NotebookWorkerError(
                f"CAM-SIMA MPI worker disconnected: {exc}\n{details}"
            ) from exc
        return self._unwrap(response)

    def _receive(self, timeout: float) -> Any:
        assert self._connection is not None
        if not self._connection.poll(timeout):
            details = self._log_tail()
            self._abort()
            raise TimeoutError(f"CAM-SIMA worker timed out after {timeout}s\n{details}")
        return self._connection.recv()

    @staticmethod
    def _unwrap(response: Any) -> Any:
        if not isinstance(response, dict) or response.get("status") not in {"ok", "error"}:
            raise NotebookWorkerError(f"invalid worker response: {response!r}")
        if response["status"] == "error":
            raise NotebookWorkerError(str(response.get("error", "unknown worker error")))
        return response.get("result")

    def _ensure_running(self) -> None:
        if self._connection is None or not self.running:
            details = self._log_tail() if self._process is not None else ""
            raise RuntimeError(f"NotebookSession is not running{details}")

    def _worker_environment(self) -> dict[str, str]:
        if self.env_script is None:
            return mpi_loader_environment()
        command = f"source {shlex.quote(str(self.env_script))} >/dev/null 2>&1 && env -0"
        raw = subprocess.check_output(["bash", "-c", command], env=os.environ)
        environment: dict[str, str] = {}
        for entry in raw.split(b"\0"):
            if not entry or b"=" not in entry:
                continue
            key, value = entry.split(b"=", 1)
            environment[os.fsdecode(key)] = os.fsdecode(value)
        return mpi_loader_environment(environment)

    def _launcher_command(self, environment: Mapping[str, str]) -> list[str]:
        command = list(self.launcher)
        has_host_option = any(
            option in {"--hosts", "--hostfile"} or option.startswith("--hosts=")
            for option in command
        )
        if has_host_option:
            return command
        if self.hosts:
            return [*command, "--hosts", ",".join(self.hosts)]
        if environment.get("PBS_NODEFILE"):
            return command

        local_host = socket.gethostname()
        short_host = local_host.split(".", 1)[0]
        if short_host.startswith("derecho") and short_host[7:].isdigit():
            raise RuntimeError(
                f"cannot start CAM-SIMA on Derecho login node {local_host}; "
                "open the Notebook in a compute-node allocation"
            )
        # Cray PALS normally receives a host list from PBS.  Jupyter compute
        # sessions do not always retain PBS_NODEFILE, so explicitly launch all
        # validated ranks on the Notebook's one allocated node.
        return [*command, "--hosts", local_host, "--no-vni"]

    def _resolve_launch_mode(self, environment: Mapping[str, str]) -> str:
        if self.launch_mode != "auto":
            return self.launch_mode
        if environment.get("PBS_NODEFILE"):
            return "local"
        short_host = socket.gethostname().split(".", 1)[0]
        if short_host.startswith("derecho") and short_host[7:].isdigit():
            return "pbs"
        return "local"

    def _submit_pbs_worker(
        self,
        worker_command: Sequence[str],
        environment: Mapping[str, str],
    ) -> None:
        script = self.run_dir / f".pycam-notebook-{secrets.token_hex(6)}.pbs"
        source = (
            f"source {shlex.quote(str(self.env_script))} >/dev/null 2>&1\n"
            if self.env_script is not None
            else ""
        )
        loader_path = shlex.quote(environment["LD_LIBRARY_PATH"])
        command = " ".join(shlex.quote(value) for value in worker_command)
        script.write_text(
            "#!/bin/bash\n"
            "#PBS -N pycam_nb\n"
            f"#PBS -A {self.pbs_account}\n"
            f"#PBS -q {self.pbs_queue}\n"
            f"#PBS -l select=1:ncpus={self.ranks}:mpiprocs={self.ranks}:mem=45GB\n"
            f"#PBS -l walltime={self.pbs_walltime}\n"
            "#PBS -j oe\n"
            f"#PBS -o {self.log_path}\n"
            "set -euo pipefail\n"
            f"{source}"
            f"export LD_LIBRARY_PATH={loader_path}\n"
            f"exec {command}\n"
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
                f"cannot submit CAM-SIMA PBS worker: {exc.stderr.strip()}"
            ) from exc
        self._job_id = result.stdout.strip().splitlines()[-1]
        print(
            f"PyCAM-SIMA PBS worker submitted as {self._job_id}; "
            f"waiting for {self.ranks} MPI ranks ...",
            flush=True,
        )

    def _accept_worker(self, listener: Listener, timeout: float) -> Connection:
        results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def accept() -> None:
            try:
                results.put((True, listener.accept()))
            except BaseException as exc:
                results.put((False, exc))

        threading.Thread(target=accept, daemon=True).start()
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"MPI worker did not connect within {timeout}s")
            try:
                ok, result = results.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise NotebookWorkerError(
                        f"MPI worker exited with code {self._process.returncode}\n{self._log_tail()}"
                    )
                continue
            if not ok:
                raise NotebookWorkerError(f"cannot accept MPI worker connection: {result}")
            return result

    def _abort(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            self._terminate_process(process)
        if self._job_id is not None:
            subprocess.run(
                ["qdel", self._job_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        self._cleanup_handles()

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _cleanup_handles(self) -> None:
        if self._connection is not None:
            self._connection.close()
        if self._listener is not None:
            self._listener.close()
        if self._log_handle is not None:
            self._log_handle.close()
        self._connection = None
        self._listener = None
        self._process = None
        self._job_id = None
        self._log_handle = None
        self._fields = {}
        self._phase_names = ()
        self._phase_status = {}
        self._scheme_names = self._scheme_plan.keys
        self._scheme_status = {
            "last_scheme": None,
            "sequence_safe": self._scheme_plan.sequence_safe,
            "plan": self._scheme_plan.to_payload(),
        }
        self._started_options_fingerprint = None
        self.initialized_native_calls = None
        self.initialized_abi_checked = None
        if self._ephemeral_config_path is not None:
            self._ephemeral_config_path.unlink(missing_ok=True)
            self.config_path = None
            self._ephemeral_config_path = None

    def _log_tail(self, lines: int = 40) -> str:
        try:
            content = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        tail = "\n".join(content[-lines:])
        return f"\nWorker log ({self.log_path}):\n{tail}" if tail else ""
