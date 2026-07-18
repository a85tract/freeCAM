from __future__ import annotations

import base64
import os
import queue
import secrets
import shlex
import signal
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

from .config import CaseConfig
from .runtime_env import mpi_loader_environment


class NotebookWorkerError(RuntimeError):
    """An error reported by the external MPI CAM-SIMA worker."""


class NotebookSession:
    """Interactive controller for a CAM-SIMA MPI worker from one Python process.

    A normal Jupyter kernel is not part of an MPI world.  This class starts a
    separate ``mpiexec`` worker, sends collective model commands to rank zero,
    and returns copied NumPy fields to the notebook over an authenticated local
    socket.  CAM's live state remains zero-copy inside each worker rank.
    """

    def __init__(
        self,
        config: str | Path | CaseConfig,
        *,
        run_dir: str | Path,
        library: str | Path | None = None,
        ranks: int | None = None,
        env_script: str | Path | None = None,
        launcher: str | Sequence[str] = "mpiexec",
        python_executable: str | Path | None = None,
        startup_timeout: float = 300.0,
        request_timeout: float = 600.0,
        log_path: str | Path | None = None,
    ) -> None:
        if isinstance(config, CaseConfig):
            if config.config_path is None:
                raise ValueError("a CaseConfig used by NotebookSession needs config_path")
            self.config = config
            self.config_path = config.config_path.resolve()
        else:
            self.config_path = Path(config).resolve()
            self.config = CaseConfig.from_yaml(self.config_path)

        self.run_dir = Path(run_dir).resolve()
        self.library = Path(library or self.config.native.se_library).resolve()
        self.ranks = self.config.mpi_ranks if ranks is None else int(ranks)
        if self.ranks != self.config.mpi_ranks:
            raise ValueError(
                f"the validated configuration requires {self.config.mpi_ranks} ranks, "
                f"got {self.ranks}"
            )
        if not (self.run_dir / "atm_in").is_file():
            raise FileNotFoundError(f"NotebookSession run directory lacks atm_in: {self.run_dir}")
        if not self.library.is_file():
            raise FileNotFoundError(f"full CAM-SIMA library not found: {self.library}")

        self.env_script = Path(env_script).resolve() if env_script else None
        if self.env_script is not None and not self.env_script.is_file():
            raise FileNotFoundError(f"environment script not found: {self.env_script}")
        self.launcher = tuple(shlex.split(launcher) if isinstance(launcher, str) else launcher)
        if not self.launcher:
            raise ValueError("launcher cannot be empty")
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
        self._connection: Connection | None = None
        self._listener: Listener | None = None
        self._log_handle: Any = None
        self._fields: dict[str, dict[str, Any]] = {}
        self._current_step = 0

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def current_step(self) -> int:
        return self._current_step

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

        authkey = secrets.token_bytes(32)
        listener = Listener(("127.0.0.1", 0), authkey=authkey)
        self._listener = listener
        host, port = listener.address
        environment = self._worker_environment()
        environment["PYTHONUNBUFFERED"] = "1"
        command = [
            *self.launcher,
            "-n",
            str(self.ranks),
            self.python_executable,
            "-m",
            "pycam_sima.notebook_worker",
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
        ]

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_path.open("ab", buffering=0)
        try:
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
            self._fields = dict(result["fields"])
            self._current_step = int(result["step"])
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
        result = self._request({"op": "step", "count": count})
        self._current_step = int(result["step"])
        return self._current_step

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
    ) -> None:
        self._validate_field(name)
        selector = self._validate_rank(rank)
        self._request(
            {
                "op": "set_field",
                "field": name,
                "rank": selector,
                "value": np.asarray(value),
            }
        )

    def close(self) -> None:
        if self._connection is None and self._process is None:
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
        self._log_handle = None
        self._fields = {}

    def _log_tail(self, lines: int = 40) -> str:
        try:
            content = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        tail = "\n".join(content[-lines:])
        return f"\nWorker log ({self.log_path}):\n{tail}" if tail else ""
