"""Interactive Jupyter controller for one persistent PI-CAM MPI model."""

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
import threading
import time
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any

from freecam.core.runtime_env import mpi_loader_environment
from freecam.model.python_processes import PythonProcessSpec

from .config import PICAMConfig
from .state import PICAMVariableSpec


class PICAMNotebookError(RuntimeError):
    """The persistent PI-CAM MPI worker could not complete a request."""


def _authkey_argument(authkey: bytes) -> str:
    """Encode a secret without letting a leading dash confuse argparse."""

    return "--authkey=" + base64.urlsafe_b64encode(authkey).decode("ascii")


class _SessionFieldReference:
    """One rank-local StatePool field exposed through the live MPI session."""

    def __init__(self, session: "PICAMNotebookSession", name: str) -> None:
        self.session = session
        self.name = name

    @property
    def metadata(self) -> Mapping[str, Any]:
        fields = self.session.status.get("fields", {})
        if self.name not in fields:
            raise KeyError(self.name)
        return dict(fields[self.name])

    def get(self, *, rank: int = 0) -> Any:
        return self.session.field(self.name, rank=rank)

    def stats(self, *, rank: int | str = 0) -> Mapping[str, Any]:
        return self.session.stats(self.name, rank=rank)

    def delete(self) -> Mapping[str, Any]:
        return self.session.delete_field(self.name)


class _SessionFieldCollection:
    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session

    @property
    def names(self) -> tuple[str, ...]:
        return self.session.field_names

    def __dir__(self) -> list[str]:
        fields = self.session.status.get("fields", {})
        candidates = set(fields)
        for metadata in fields.values():
            candidates.update(metadata.get("aliases", ()))
            if metadata.get("standard_name"):
                candidates.add(str(metadata["standard_name"]))
        return sorted(
            set(super().__dir__())
            | {item for item in candidates if item.isidentifier()}
        )

    def __getattr__(self, name: str) -> _SessionFieldReference:
        try:
            canonical = self._resolve(name)
        except KeyError as exc:
            raise AttributeError(name) from exc
        return _SessionFieldReference(self.session, canonical)

    def __getitem__(self, name: str) -> _SessionFieldReference:
        return _SessionFieldReference(self.session, self._resolve(name))

    def _resolve(self, name: str) -> str:
        fields = self.session.status.get("fields", {})
        if name in fields:
            return name
        matches = [
            canonical
            for canonical, metadata in fields.items()
            if name in metadata.get("aliases", ())
            or name == metadata.get("standard_name")
        ]
        if len(matches) != 1:
            raise KeyError(name)
        return str(matches[0])

    def create(
        self,
        name: str,
        *,
        dims: Sequence[str],
        dtype: str = "float64",
        units: str = "1",
        initial: float | int = 0.0,
        writable: bool = True,
        restart: bool = True,
        aliases: Sequence[str] = (),
        standard_name: str | None = None,
    ) -> _SessionFieldReference:
        self.session.create_field(
            name,
            dimensions=dims,
            dtype=dtype,
            units=units,
            initial=initial,
            writable=writable,
            restart=restart,
            aliases=aliases,
            standard_name=standard_name,
        )
        return _SessionFieldReference(self.session, name)

    def delete(self, name: str) -> Mapping[str, Any]:
        return self.session.delete_field(name)


class _SessionActionReference:
    """A physics action that can be run or edited without string plumbing."""

    def __init__(
        self,
        session: "PICAMNotebookSession",
        name: str,
        phase: str,
        *,
        kind: str = "scheme",
    ) -> None:
        self.session = session
        self.name = name
        self.phase = phase
        self.kind = kind

    def run(self) -> Mapping[str, Any]:
        return self.session.run_action(self.name, phase=self.phase)

    @property
    def enabled(self) -> bool:
        return bool(self._record()["enabled"])

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.session.set_action_enabled(self.name, bool(value), phase=self.phase)

    def enable(self) -> Mapping[str, Any]:
        return self.session.set_action_enabled(self.name, True, phase=self.phase)

    def disable(self) -> Mapping[str, Any]:
        return self.session.set_action_enabled(self.name, False, phase=self.phase)

    def move(
        self,
        *,
        before: str | None = None,
        after: str | None = None,
    ) -> Mapping[str, Any]:
        return self.session.move_action(
            self.name,
            phase=self.phase,
            before=before,
            after=after,
        )

    def remove(self) -> Mapping[str, Any]:
        if self.kind == "python_process":
            return self.session.remove_python(self.name)
        if self.kind == "runtime_fortran_process":
            return self.session.remove_fortran(self.name)
        raise TypeError(f"source physics action {self.phase}.{self.name} cannot be removed")

    def _record(self) -> Mapping[str, Any]:
        matches = [
            row
            for row in self.session.status.get("step_plan", ())
            if row["phase"] == self.phase and row["name"] == self.name
        ]
        if len(matches) != 1:
            raise KeyError(f"{self.phase}.{self.name}")
        return dict(matches[0])


class _SessionPhysicsCollection:
    _KINDS = {"scheme", "python_process", "runtime_fortran_process"}

    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            str(row["name"])
            for row in self.session.status.get("step_plan", ())
            if row["kind"] in self._KINDS
        )

    def __dir__(self) -> list[str]:
        return sorted(
            set(super().__dir__())
            | {name for name in self.names if name.isidentifier()}
        )

    def __getattr__(self, name: str) -> _SessionActionReference:
        try:
            return self.scheme(name)
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> _SessionActionReference:
        return self.scheme(name)

    def scheme(
        self, name: str, *, phase: str | None = None
    ) -> _SessionActionReference:
        matches = [
            row
            for row in self.session.status.get("step_plan", ())
            if row["kind"] in self._KINDS
            and (row["name"] == name or row["operation"] == name)
            and (phase is None or row["phase"] == phase)
        ]
        if len(matches) != 1:
            raise KeyError(f"physics action {name!r} is unknown or ambiguous")
        row = matches[0]
        return _SessionActionReference(
            self.session,
            str(row["name"]),
            str(row["phase"]),
            kind=str(row["kind"]),
        )

    def install_python(
        self,
        function: Any,
        *,
        name: str,
        phase: str,
        before: str | None = None,
        after: str | None = None,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        enabled: bool = True,
        transactional: bool = True,
        unsafe: bool = False,
    ) -> _SessionActionReference:
        result = self.session.install_python(
            function,
            name=name,
            phase=phase,
            before=before,
            after=after,
            reads=reads,
            writes=writes,
            parameters=parameters,
            enabled=enabled,
            transactional=transactional,
            unsafe=unsafe,
        )
        return _SessionActionReference(
            self.session,
            str(result["name"]),
            str(result["phase"]),
            kind="python_process",
        )

    def install_fortran(
        self,
        source: str | Path,
        *,
        process: str,
        phase: str,
        before: str | None = None,
        after: str | None = None,
        project_root: str | Path | None = None,
        enabled: bool = True,
        unsafe: bool = False,
    ) -> _SessionActionReference:
        result = self.session.install_fortran(
            source,
            process=process,
            phase=phase,
            before=before,
            after=after,
            project_root=project_root,
            enabled=enabled,
            unsafe=unsafe,
        )
        return _SessionActionReference(
            self.session,
            str(result["name"]),
            str(result["phase"]),
            kind="runtime_fortran_process",
        )


class _SessionPhaseReference:
    def __init__(self, session: "PICAMNotebookSession", name: str) -> None:
        self.session = session
        self.name = name

    @property
    def actions(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            dict(row)
            for row in self.session.status.get("step_plan", ())
            if row["phase"] == self.name
        )

    def run(self) -> tuple[Mapping[str, Any], ...]:
        return self.session.run_phase(self.name)

    def expand(self) -> tuple[Mapping[str, Any], ...]:
        expanders = {
            "cam_run1": self.session.expand_cam_run1_leaves,
            "cam_run2": self.session.expand_cam_run2_leaves,
            "cam_run4": self.session.expand_cam_run4_leaves,
        }
        if self.name not in expanders:
            raise TypeError(f"phase {self.name!r} has no finer validated expansion")
        return expanders[self.name]()


class _SessionPhaseCollection:
    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                str(row["phase"])
                for row in self.session.status.get("step_plan", ())
            )
        )

    def __getattr__(self, name: str) -> _SessionPhaseReference:
        if name not in self.names:
            raise AttributeError(name)
        return _SessionPhaseReference(self.session, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.names))

    def __getitem__(self, name: str) -> _SessionPhaseReference:
        if name not in self.names:
            raise KeyError(name)
        return _SessionPhaseReference(self.session, name)


class _SessionKernelReference:
    def __init__(self, session: "PICAMNotebookSession", name: str) -> None:
        self.session = session
        self.name = name

    def run(self) -> Mapping[str, Any]:
        return self.session.run_kernel(self.name)


class _SessionKernelCollection:
    def __init__(self, session: "PICAMNotebookSession") -> None:
        self.session = session

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.session.status.get("kernels", ()))

    def __getattr__(self, name: str) -> _SessionKernelReference:
        if name not in self.names:
            raise AttributeError(name)
        return _SessionKernelReference(self.session, name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.names))

    def __getitem__(self, name: str) -> _SessionKernelReference:
        if name not in self.names:
            raise KeyError(name)
        return _SessionKernelReference(self.session, name)


class PICAMNotebookSession:
    """Keep 512 CAM ranks alive and execute one Python-controlled action at a time."""

    def __init__(
        self,
        config: str | Path | PICAMConfig,
        *,
        boundary: str | Path,
        run_dir: str | Path,
        env_script: str | Path,
        python_executable: str | Path | None = None,
        launcher: str | Sequence[str] = "mpiexec",
        launch_mode: str = "auto",
        pbs_account: str = "UCUB0188",
        pbs_queue: str = "develop",
        pbs_walltime: str = "02:00:00",
        startup_timeout: float = 1200.0,
        request_timeout: float = 1200.0,
        log_path: str | Path | None = None,
    ) -> None:
        if isinstance(config, PICAMConfig):
            raise TypeError("PICAMNotebookSession currently requires a YAML config path")
        self.config_path = Path(config).resolve()
        self.config = PICAMConfig.from_yaml(self.config_path)
        self.boundary = Path(boundary).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.env_script = Path(env_script).resolve()
        self.python_executable = str(python_executable or sys.executable)
        self.launcher = tuple(
            shlex.split(launcher) if isinstance(launcher, str) else launcher
        )
        if launch_mode not in {"auto", "local", "pbs"}:
            raise ValueError("launch_mode must be auto, local, or pbs")
        self.launch_mode = launch_mode
        self.pbs_account = pbs_account
        self.pbs_queue = pbs_queue
        self.pbs_walltime = pbs_walltime
        self.startup_timeout = float(startup_timeout)
        self.request_timeout = float(request_timeout)
        self.log_path = Path(log_path or self.run_dir / "pi_cam_notebook_worker.log").resolve()
        if not self.env_script.is_file():
            raise FileNotFoundError(self.env_script)
        if not (self.run_dir / "atm_in").is_file():
            raise FileNotFoundError(f"PI-CAM run directory lacks atm_in: {self.run_dir}")
        if not (self.boundary / "manifest.json").is_file():
            raise FileNotFoundError(f"PI-CAM boundary replay is incomplete: {self.boundary}")
        self._connection: Connection | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._job_id: str | None = None
        self._log_handle: Any = None
        self._pbs_script: Path | None = None
        self._status: dict[str, Any] = {}
        self.fields = _SessionFieldCollection(self)
        self.physics = _SessionPhysicsCollection(self)
        self.phases = _SessionPhaseCollection(self)
        self.kernels = _SessionKernelCollection(self)

    @property
    def running(self) -> bool:
        return self._connection is not None

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def status(self) -> Mapping[str, Any]:
        if self.running:
            self._status = dict(self._request({"op": "status"}))
        return dict(self._status)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.status.get("fields", ()))

    def start(self) -> "PICAMNotebookSession":
        if self.running:
            raise RuntimeError("PI-CAM Notebook session is already running")
        environment = self._environment()
        mode = self._launch_mode(environment)
        authkey = secrets.token_bytes(32)
        listener = Listener(("0.0.0.0" if mode == "pbs" else "127.0.0.1", 0), authkey=authkey)
        host = socket.getfqdn() if mode == "pbs" else "127.0.0.1"
        _, port = listener.address
        command = [
            *self.launcher,
            "-n",
            str(self.config.mpi_size),
            self.python_executable,
            "-m",
            "freecam.pi_cam.session_worker",
            "--host",
            host,
            "--port",
            str(port),
            # URL-safe base64 may start with "-".  Use argparse's
            # --option=value form so such a secret is never mistaken for a
            # new command-line option on all 512 ranks.
            _authkey_argument(authkey),
            "--config",
            str(self.config_path),
            "--boundary",
            str(self.boundary),
            "--run-dir",
            str(self.run_dir),
            "--expected-ranks",
            str(self.config.mpi_size),
        ]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mode == "pbs":
                self._submit_pbs(command, environment)
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
            self._connection = self._accept(listener)
            self._status = dict(self._unwrap(self._receive(self.startup_timeout)))
        except BaseException:
            self._abort()
            raise
        finally:
            listener.close()
        return self

    def step(self, count: int = 1) -> Mapping[str, Any]:
        if count < 1:
            raise ValueError("count must be positive")
        self._status = dict(self._request({"op": "step", "count": int(count)}))
        return dict(self._status)

    def advance(self, steps: int = 1) -> Mapping[str, Any]:
        """Advance complete CAM steps while keeping ``step`` compatible."""

        return self.step(steps)

    def run_action(self, name: str, *, phase: str | None = None) -> Mapping[str, Any]:
        """Run one scheme or installed runtime process without advancing time."""

        return dict(
            self._request({"op": "run_action", "name": name, "phase": phase})
        )

    def run_scheme(self, name: str, *, phase: str | None = None) -> Mapping[str, Any]:
        """Compatibility alias for :meth:`run_action`."""

        return self.run_action(name, phase=phase)

    def run_phase(self, name: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._request({"op": "run_phase", "phase": name}))

    def run_kernel(self, name: str) -> Mapping[str, Any]:
        """Run one experimental raw-array kernel on all live MPI ranks."""

        return dict(self._request({"op": "run_kernel", "name": name}))

    def expand_cam_run1_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Replace three composite stages with ordered native leaf actions."""

        return tuple(self._request({"op": "expand_cam_run1_leaves"}))

    def expand_cam_run2_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Replace admitted ``cam_run2`` composites with native leaves."""

        return tuple(self._request({"op": "expand_cam_run2_leaves"}))

    def expand_cam_run4_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Replace the ``cam_run4`` finish composite with native leaves."""

        return tuple(self._request({"op": "expand_cam_run4_leaves"}))

    def expand_cam_run2_run4_leaves(self) -> tuple[Mapping[str, Any], ...]:
        """Expand admitted leaf actions from ``cam_run2`` through run4."""

        return tuple(self._request({"op": "expand_cam_run2_run4_leaves"}))

    def create_field(
        self,
        name: str,
        *,
        dimensions: Sequence[str],
        dtype: str = "float64",
        units: str = "1",
        initial: float | int = 0.0,
        writable: bool = True,
        restart: bool = True,
        aliases: Sequence[str] = (),
        standard_name: str | None = None,
    ) -> Mapping[str, Any]:
        spec = PICAMVariableSpec(
            name=name,
            dimensions=tuple(dimensions),
            dtype=dtype,
            units=units,
            initial=initial,
            writable=writable,
            restart=restart,
            aliases=tuple(aliases),
            standard_name=standard_name,
        )
        return dict(self._request({"op": "create_field", "spec": spec.to_payload()}))

    def delete_field(self, name: str) -> Mapping[str, Any]:
        return dict(self._request({"op": "delete_field", "name": name}))

    def install_python(
        self,
        function: Any,
        *,
        name: str,
        phase: str,
        before: str | None = None,
        after: str | None = None,
        reads: Sequence[str] = (),
        writes: Sequence[str] = (),
        parameters: Mapping[str, Any] | None = None,
        enabled: bool = True,
        transactional: bool = True,
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        spec = PythonProcessSpec.from_callable(
            function,
            name=name,
            group=phase,
            before=before,
            after=after,
            reads=reads,
            writes=writes,
            parameters=parameters,
            enabled=enabled,
            transactional=transactional,
        )
        return dict(
            self._request(
                {
                    "op": "install_python",
                    "spec": spec.as_dict(),
                    "unsafe": bool(unsafe),
                }
            )
        )

    def remove_python(self, name: str) -> Mapping[str, Any]:
        return dict(self._request({"op": "remove_python", "name": name}))

    def install_fortran(
        self,
        source: str | Path,
        *,
        process: str,
        phase: str,
        before: str | None = None,
        after: str | None = None,
        project_root: str | Path | None = None,
        enabled: bool = True,
        unsafe: bool = False,
    ) -> Mapping[str, Any]:
        return dict(
            self._request(
                {
                    "op": "install_fortran",
                    "spec": {
                        "schema_version": 1,
                        "source": str(Path(source).expanduser().resolve()),
                        "process": process,
                        "phase": phase,
                        "before": before,
                        "after": after,
                        "project_root": (
                            None
                            if project_root is None
                            else str(Path(project_root).expanduser().resolve())
                        ),
                        "enabled": bool(enabled),
                    },
                    "unsafe": bool(unsafe),
                }
            )
        )

    def remove_fortran(self, name: str) -> Mapping[str, Any]:
        return dict(self._request({"op": "remove_fortran", "name": name}))

    def set_action_enabled(
        self,
        name: str,
        enabled: bool,
        *,
        phase: str | None = None,
    ) -> Mapping[str, Any]:
        return dict(
            self._request(
                {
                    "op": "set_action_enabled",
                    "name": name,
                    "phase": phase,
                    "enabled": bool(enabled),
                }
            )
        )

    def move_action(
        self,
        name: str,
        *,
        phase: str | None = None,
        before: str | None = None,
        after: str | None = None,
    ) -> Mapping[str, Any]:
        return dict(
            self._request(
                {
                    "op": "move_action",
                    "name": name,
                    "phase": phase,
                    "before": before,
                    "after": after,
                }
            )
        )

    def field(self, name: str, *, rank: int = 0) -> Any:
        return self._request({"op": "field", "name": name, "rank": int(rank)})

    def stats(self, name: str, *, rank: int | str = 0) -> Mapping[str, Any]:
        return dict(self._request({"op": "stats", "name": name, "rank": rank}))

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._request({"op": "close"})
            finally:
                self._cleanup()

    def __enter__(self) -> "PICAMNotebookSession":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.close()
        except BaseException:
            if exc_type is None:
                raise

    def _request(self, command: dict[str, Any]) -> Any:
        if self._connection is None:
            raise RuntimeError("PI-CAM Notebook session is not running")
        self._connection.send(command)
        return self._unwrap(self._receive(self.request_timeout))

    def _receive(self, timeout: float) -> Any:
        assert self._connection is not None
        if not self._connection.poll(timeout):
            raise TimeoutError(f"PI-CAM worker timed out after {timeout}s\n{self._log_tail()}")
        return self._connection.recv()

    @staticmethod
    def _unwrap(response: Any) -> Any:
        if not isinstance(response, dict) or response.get("status") not in {"ok", "error"}:
            raise PICAMNotebookError(f"invalid PI-CAM worker response: {response!r}")
        if response["status"] == "error":
            raise PICAMNotebookError(str(response.get("error", "unknown worker error")))
        return response.get("result")

    def _environment(self) -> dict[str, str]:
        command = f"source {shlex.quote(str(self.env_script))} >/dev/null 2>&1 && env -0"
        raw = subprocess.check_output(["bash", "-c", command], env=os.environ)
        environment: dict[str, str] = {}
        for entry in raw.split(b"\0"):
            if entry and b"=" in entry:
                key, value = entry.split(b"=", 1)
                environment[os.fsdecode(key)] = os.fsdecode(value)
        environment = mpi_loader_environment(environment)
        manifest = self.config.native_manifest
        if manifest is not None:
            payload = json.loads(manifest.read_text())
            math_library = payload.get("intel_math_library")
            if math_library:
                math_path = Path(str(math_library)).resolve()
                if not math_path.is_file():
                    raise FileNotFoundError(
                        f"PI-CAM Intel math runtime does not exist: {math_path}"
                    )
                existing = environment.get("LD_PRELOAD", "")
                entries = [item for item in existing.split(":") if item]
                if str(math_path) not in entries:
                    entries.insert(0, str(math_path))
                environment["LD_PRELOAD"] = ":".join(entries)
        return environment

    def _launch_mode(self, environment: Mapping[str, str]) -> str:
        if self.launch_mode != "auto":
            return self.launch_mode
        if environment.get("PBS_NODEFILE"):
            return "local"
        short_host = socket.gethostname().split(".", 1)[0]
        return "pbs" if short_host.startswith("derecho") else "local"

    def _submit_pbs(self, command: Sequence[str], environment: Mapping[str, str]) -> None:
        script = self.run_dir / f".pi-cam-notebook-{secrets.token_hex(6)}.pbs"
        nodes = (self.config.mpi_size + 127) // 128
        rendered = " ".join(shlex.quote(item) for item in command)
        preload = environment.get("LD_PRELOAD")
        preload_export = (
            f"export LD_PRELOAD={shlex.quote(preload)}\n" if preload else ""
        )
        script.write_text(
            "#!/bin/bash\n"
            "#PBS -N pi-cam-nb\n"
            f"#PBS -A {self.pbs_account}\n"
            f"#PBS -q {self.pbs_queue}\n"
            f"#PBS -l select={nodes}:ncpus=64:mpiprocs=128:ompthreads=1:mem=64GB\n"
            f"#PBS -l walltime={self.pbs_walltime}\n"
            "#PBS -j oe\n"
            f"#PBS -o {self.log_path}\n"
            "set -euo pipefail\n"
            f"source {shlex.quote(str(self.env_script))} >/dev/null 2>&1\n"
            f"export LD_LIBRARY_PATH={shlex.quote(environment['LD_LIBRARY_PATH'])}\n"
            f"{preload_export}"
            f"export PYTHONPATH={shlex.quote(str(Path(__file__).resolve().parents[2]))}\n"
            f"exec {rendered}\n"
        )
        result = subprocess.run(
            ["qsub", str(script)], check=True, capture_output=True, text=True,
            env=dict(environment),
        )
        self._pbs_script = script
        self._job_id = result.stdout.strip().splitlines()[-1]
        print(
            f"PI-CAM Notebook worker submitted as {self._job_id}; "
            f"waiting for {self.config.mpi_size} ranks ...",
            flush=True,
        )

    def _accept(self, listener: Listener) -> Connection:
        results: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def accept() -> None:
            try:
                results.put((True, listener.accept()))
            except BaseException as error:
                results.put((False, error))

        threading.Thread(target=accept, daemon=True).start()
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            try:
                ok, result = results.get(timeout=0.1)
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    raise PICAMNotebookError(
                        f"PI-CAM worker exited with {self._process.returncode}\n{self._log_tail()}"
                    )
                continue
            if not ok:
                raise PICAMNotebookError(f"cannot accept PI-CAM worker: {result}")
            return result
        raise TimeoutError(f"PI-CAM worker did not connect\n{self._log_tail()}")

    def _log_tail(self) -> str:
        if not self.log_path.is_file():
            return f"worker log not created: {self.log_path}"
        return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-80:])

    def _abort(self) -> None:
        if self._process is not None and self._process.poll() is None:
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self._job_id is not None:
            subprocess.run(["qdel", self._job_id], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._connection is not None:
            self._connection.close()
        if self._log_handle is not None:
            self._log_handle.close()
        self._connection = None
        self._process = None
        self._job_id = None
        self._log_handle = None
