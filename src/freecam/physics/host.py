"""Where a physics function's calls execute: in a worker, or in this process.

The subprocess host is the default.  It starts one worker that owns one
initialized image, hands it one sample per request, and classifies how the
worker died if it dies: exit 86 is a Fortran abort (the sample was refused;
the message is kept), 87 means a link-time stub was reached (an
implementation error, raised), a signal is a crash.  After any death the
sample is marked, a fresh worker is started and brought back to the same
parameter state, and the batch continues -- up to a restart cap.

The in-process host exists for validation tools and tests that already run
with the image's math library preloaded.
"""

from __future__ import annotations

import base64
from collections import deque
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import re
import resource
import secrets
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping

import numpy as np

from freecam.core.fortran_adapter import FortranAdapterError

from .errors import PhysicsError
from .image import StandaloneImage

EXIT_ABORT = 86
EXIT_STUB_CALLED = 87


_NUMBER = re.compile(r"^[-+]?(?:\d+\.?\d*(?:[eEdD][-+]?\d+)?|\.\d+(?:[eEdD][-+]?\d+)?|nan|inf(?:inity)?)$", re.IGNORECASE)


def _is_prose(line: str) -> bool:
    """A diagnostic sentence, as opposed to a dump of values (incl. NaN)."""

    tokens = line.split()
    return bool(tokens) and not all(_NUMBER.match(token) for token in tokens)


class StubCalledError(PhysicsError):
    """A fail-closed link-time stub was reached: an implementation error."""


class WorkerRestartLimit(PhysicsError):
    """The worker died more often than the host is willing to restart it."""


class CallOutcome:
    __slots__ = ("status", "pool", "message")

    def __init__(self, status: str, pool: Mapping[str, np.ndarray] | None = None, message: str | None = None) -> None:
        self.status = status
        self.pool = pool
        self.message = message


class InProcessHost:
    """The image in this process; requires the math library to be preloaded."""

    def __init__(self, manifest: str | Path, snapshot: Mapping[str, Any]) -> None:
        self.image = StandaloneImage(manifest)
        self.verification = self.image.initialize(snapshot)
        self.restarts = 0

    @property
    def spec(self):
        return self.image.spec

    def set_parameters(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return self.image.set_parameters(values)

    def restore_parameters(self) -> dict[str, Any]:
        return self.image.restore_parameters()

    def call(self, pool: Mapping[str, np.ndarray], returned: tuple[str, ...] | None = None) -> CallOutcome:
        try:
            self.image.call(pool)
        except FortranAdapterError as error:
            return CallOutcome("internal_error", None, str(error))
        return CallOutcome("ok", pool)

    def close(self) -> None:
        return None


class SubprocessHost:
    """One worker process per host, restarted after it dies."""

    def __init__(
        self,
        manifest: str | Path,
        snapshot: Mapping[str, Any],
        *,
        max_restarts: int = 100,
        socket_dir: str | Path | None = None,
        worker_command: tuple[str, ...] | None = None,
    ) -> None:
        # Tests substitute a fake worker that speaks the same protocol.
        self._worker_command = tuple(worker_command) if worker_command else (sys.executable, "-m", "freecam.physics.worker")
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.snapshot = snapshot
        self.max_restarts = max_restarts
        self.restarts = 0
        self._parameters: dict[str, Any] = {}
        # AF_UNIX socket paths are limited to 108 bytes, so the socket lives
        # under /tmp rather than in a deep scratch tree; it is unlinked on close.
        self._socket_dir = Path(socket_dir) if socket_dir else Path(tempfile.mkdtemp(prefix="fcp-", dir="/tmp"))
        self._process: subprocess.Popen | None = None
        self._connection = None
        self._stderr: deque[str] = deque(maxlen=200)
        # Fortran unit 6 (iulog) goes to stdout: endrun is often called with
        # no message right after the diagnostic was written there.
        self._stdout: deque[str] = deque(maxlen=200)
        self._reader: threading.Thread | None = None
        self._stdout_reader: threading.Thread | None = None
        self.verification: dict[str, Any] = {}
        self._start()

    # -- lifecycle -----------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
        env["LD_PRELOAD"] = str(self.manifest["intel_math_library"])
        env["PYTHONUNBUFFERED"] = "1"
        source = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source), env.get("PYTHONPATH", ""))))
        return env

    @staticmethod
    def _raise_stack_limit() -> None:
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
        except (ValueError, OSError):
            pass

    @staticmethod
    def _drain(stream, sink: deque[str]) -> None:
        for line in iter(stream.readline, ""):
            sink.append(line.rstrip("\n"))

    def _start(self) -> None:
        authkey = secrets.token_bytes(32)
        address = str(self._socket_dir / f"worker-{os.getpid()}-{self.restarts}.sock")
        listener = Listener(address, family="AF_UNIX", authkey=authkey)
        self._process = subprocess.Popen(
            [
                *self._worker_command,
                "--manifest", str(self.manifest_path), "--address", address,
                "--authkey", base64.urlsafe_b64encode(authkey).decode("ascii"),
            ],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=self._environment(), start_new_session=True, preexec_fn=self._raise_stack_limit,
        )
        self._stderr.clear()
        self._stdout.clear()
        self._reader = threading.Thread(target=self._drain, args=(self._process.stderr, self._stderr), daemon=True)
        self._stdout_reader = threading.Thread(target=self._drain, args=(self._process.stdout, self._stdout), daemon=True)
        self._reader.start()
        self._stdout_reader.start()
        try:
            self._connection = listener.accept()
        finally:
            listener.close()
        hello = self._request({"op": "hello", "manifest": str(self.manifest_path)})
        if hello["image_sha256"] != self.manifest["library_sha256"]:
            raise PhysicsError("worker loaded a different image than the manifest names")
        self.verification = self._request({"op": "initialize", "snapshot": self.snapshot})["verification"]
        if self._parameters:
            self._request({"op": "set_parameters", "values": self._parameters})

    def _request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        assert self._connection is not None
        self._connection.send(dict(message))
        reply = self._connection.recv()
        if not reply.get("ok", False):
            raise PhysicsError(reply.get("error", "worker error") + ("\n" + reply["traceback"] if "traceback" in reply else ""))
        return reply

    def _death(self) -> CallOutcome:
        assert self._process is not None
        self._process.wait()
        for reader in (self._reader, self._stdout_reader):
            if reader is not None:
                reader.join(timeout=5)
        code = self._process.returncode
        lines = list(self._stderr)
        abort = next((line for line in reversed(lines) if line.startswith("FREECAM_FORTRAN_ABORT:")), None)
        stub = next((line for line in reversed(lines) if line.startswith("FREECAM_STUB_CALLED:")), None)
        if code == EXIT_STUB_CALLED or stub:
            raise StubCalledError(stub or f"worker exited {code}")
        if code == EXIT_ABORT or abort:
            message = (abort or "").removeprefix("FREECAM_FORTRAN_ABORT:").strip()
            if not message:
                # endrun without an argument: the diagnostic is the last thing
                # the routine wrote to iulog.
                # The diagnostic is prose; the value dumps that may follow it
                # are not.  Take the last line with letters in it.
                diagnostics = [line.strip() for line in self._stdout if _is_prose(line)]
                message = diagnostics[-1] if diagnostics else "Fortran abort"
            return CallOutcome("fortran_abort", None, message)
        if code is not None and code < 0:
            return CallOutcome("worker_crash", None, f"worker died from signal {-code}")
        return CallOutcome("worker_crash", None, f"worker exited {code}: " + " | ".join(lines[-3:]))

    def _restart(self) -> None:
        self.restarts += 1
        if self.restarts > self.max_restarts:
            raise WorkerRestartLimit(f"worker restarted {self.restarts - 1} times; giving up")
        self._start()

    # -- the host interface ----------------------------------------------

    @property
    def spec(self):
        from .spec import load_function_spec

        return load_function_spec(str(self.manifest["function"]))

    def set_parameters(self, values: Mapping[str, Any]) -> dict[str, Any]:
        written = self._request({"op": "set_parameters", "values": dict(values)})["written"]
        self._parameters.update(values)
        return written

    def restore_parameters(self) -> dict[str, Any]:
        written = self._request({"op": "restore_parameters"})["written"]
        self._parameters.clear()
        return written

    def call(self, pool: Mapping[str, np.ndarray], returned: tuple[str, ...] | None = None) -> CallOutcome:
        message = {"op": "call", "id": secrets.token_hex(4), "pool": dict(pool)}
        if returned is not None:
            message["returned"] = list(returned)
        try:
            assert self._connection is not None
            self._connection.send(message)
            reply = self._connection.recv()
        except (EOFError, ConnectionError, OSError):
            outcome = self._death()
            self._restart()
            return outcome
        if not reply.get("ok", False):
            return CallOutcome(reply.get("status", "internal_error"), None, reply.get("error"))
        merged = dict(pool)
        merged.update(reply["pool"])
        return CallOutcome("ok", merged)

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.send({"op": "close"})
                self._connection.recv()
            except (EOFError, ConnectionError, OSError):
                pass
            self._connection.close()
            self._connection = None
        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        for path in self._socket_dir.glob("worker-*.sock"):
            path.unlink(missing_ok=True)

    def __enter__(self) -> "SubprocessHost":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


__all__ = ["CallOutcome", "InProcessHost", "StubCalledError", "SubprocessHost", "WorkerRestartLimit"]
