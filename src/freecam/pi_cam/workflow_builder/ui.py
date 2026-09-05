"""``Driver.ui()`` and ``freecam ui``: serve the page for one Driver.

Starting the page starts nothing else: no PBS, no MPI, no model.  The
service listens on loopback with a session token in the address; the model
is initialized by the first Run the user asks for.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import webbrowser
from typing import Any

from .service import WorkflowService, create_app


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])


class WorkflowUI:
    """A running page: its address, and ``close()``."""

    def __init__(self, service: WorkflowService, host: str, port: int) -> None:
        self.service = service
        self.host = host
        self.port = port
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/?token={self.service.token}"

    def start(self, *, block: bool = False) -> "WorkflowUI":
        import uvicorn

        app = create_app(self.service)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)
        if block:
            self._server.run()
            return self
        self._thread = threading.Thread(target=self._server.run, name="freecam-ui", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 10
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        return self

    def close(self, *, close_model: bool = False) -> None:
        """Stop serving; the model stays alive unless ``close_model`` is set."""

        if close_model:
            self.service.shutdown()
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _repr_html_(self) -> str:
        return (
            f'<div>freeCAM Workflow Builder: <a href="{self.url}" target="_blank" rel="noopener">{self.url}</a>'
            f"<br><small>The page edits the workflow; the model starts on the first Run.</small></div>"
        )

    def __repr__(self) -> str:
        return f"WorkflowUI(url={self.url!r})"


def launch_ui(driver: Any, *, host: str = "127.0.0.1", port: int | None = None,
              open_browser: bool = False, block: bool = False) -> WorkflowUI:
    """Serve the Workflow Builder for ``driver``; returns at once unless ``block``."""

    chosen = port if port else _free_port(host)
    ui = WorkflowUI(WorkflowService(driver), host, chosen)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(ui.url)).start()
    return ui.start(block=block)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="freecam ui", description="Serve the Workflow Builder for one case.")
    parser.add_argument("--case", default="PI-atm")
    parser.add_argument("--nsteps", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1", help="loopback by default; reach a remote one through SSH port forwarding")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the page in a browser")
    arguments = parser.parse_args(argv)
    from freecam import Driver

    driver = Driver(case=arguments.case, nsteps=arguments.nsteps)
    ui = WorkflowUI(WorkflowService(driver), arguments.host, arguments.port)
    print(f"freeCAM Workflow Builder for {arguments.case}: {ui.url}", file=sys.stderr)
    print("The model starts on the first Run; Ctrl+C stops the page (and closes a started model).", file=sys.stderr)
    if arguments.open:
        threading.Timer(0.5, lambda: webbrowser.open(ui.url)).start()
    try:
        ui.start(block=True)
    except KeyboardInterrupt:
        pass
    finally:
        ui.service.shutdown()
    return 0


__all__ = ["WorkflowUI", "launch_ui", "main"]
