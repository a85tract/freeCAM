"""A stand-in for freecam.physics.worker that speaks the same protocol.

FAKE_WORKER_MODE selects how it misbehaves: ``ok`` answers every call by
doubling the returned arrays; ``abort:N`` aborts like Fortran on the Nth call
(message on stderr, exit 86); ``segv:N`` dies from a signal on the Nth call;
``stub:N`` reaches a fail-closed stub on the Nth call (exit 87).
"""

from __future__ import annotations

import argparse
import base64
import json
from multiprocessing.connection import Client
import os
import signal
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--authkey", required=True)
    args = parser.parse_args()
    mode, _, arg = os.environ.get("FAKE_WORKER_MODE", "ok").partition(":")
    trigger = int(arg) if arg else 0
    manifest = json.loads(open(args.manifest).read())
    connection = Client(args.address, family="AF_UNIX", authkey=base64.urlsafe_b64decode(args.authkey.encode()))
    calls = 0
    parameters: dict = {}
    while True:
        message = connection.recv()
        op = message["op"]
        if op == "hello":
            connection.send({"ok": True, "pid": os.getpid(), "image_sha256": manifest["library_sha256"],
                             "mxcsr": "0x9fc0", "ld_preload": os.environ.get("LD_PRELOAD")})
        elif op == "initialize":
            connection.send({"ok": True, "verification": {"all_equal": True, "entries": 0}})
        elif op == "set_parameters":
            parameters.update(message["values"])
            connection.send({"ok": True, "written": dict(message["values"])})
        elif op == "restore_parameters":
            parameters.clear()
            connection.send({"ok": True, "written": {}})
        elif op == "call":
            calls += 1
            if mode == "abort" and calls == trigger:
                sys.stderr.write("FREECAM_FORTRAN_ABORT: Impossible case1 in instratus_condensate\n"); sys.stderr.flush()
                os._exit(86)
            if mode == "segv" and calls == trigger:
                os.kill(os.getpid(), signal.SIGSEGV)
            if mode == "stub" and calls == trigger:
                sys.stderr.write("FREECAM_STUB_CALLED: cam_history_mp_addfld_\n"); sys.stderr.flush()
                os._exit(87)
            pool = message["pool"]
            returned = {key: pool[key] * (2.0 + parameters.get("gain", 0.0)) for key in message.get("returned", pool)}
            connection.send({"ok": True, "id": message.get("id"), "pool": returned})
        elif op == "close":
            connection.send({"ok": True})
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
