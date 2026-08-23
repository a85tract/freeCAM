"""The process that owns a standalone image and runs one call at a time.

Run as ``python -m freecam.physics.worker --manifest ... --address ...``.
It loads the image, answers the host over a local authenticated connection,
and executes exactly one wrapper call per request so that if the Fortran
aborts -- exit 86 with the message on stderr -- the host knows which sample
was running.  Nothing else is shared with the host: the image is mapped
only here, so no Fortran failure can reach the user's process.
"""

from __future__ import annotations

import argparse
import base64
import os
from multiprocessing.connection import Client
import sys
import traceback

from freecam.core.fortran_adapter import FortranAdapterError

from .image import StandaloneImage


def serve(manifest: str, address: str, authkey: bytes) -> int:
    connection = Client(address, family="AF_UNIX", authkey=authkey)
    image: StandaloneImage | None = None
    try:
        while True:
            message = connection.recv()
            op = message.get("op")
            try:
                if op == "hello":
                    image = StandaloneImage(message["manifest"])
                    reply = {
                        "ok": True,
                        "pid": os.getpid(),
                        "image_sha256": image.manifest["library_sha256"],
                        "mxcsr": hex(int(image.library.pycam_pi_cam_get_mxcsr_v1())),
                        "ld_preload": os.environ.get("LD_PRELOAD"),
                    }
                elif op == "initialize":
                    assert image is not None
                    reply = {"ok": True, "verification": image.initialize(message["snapshot"])}
                elif op == "set_parameters":
                    assert image is not None
                    reply = {"ok": True, "written": image.set_parameters(message["values"])}
                elif op == "restore_parameters":
                    assert image is not None
                    reply = {"ok": True, "written": image.restore_parameters()}
                elif op == "call":
                    assert image is not None
                    pool = message["pool"]
                    try:
                        image.call(pool)
                    except FortranAdapterError as error:
                        reply = {"ok": False, "id": message.get("id"), "status": "internal_error", "error": str(error)}
                    else:
                        returned = {key: pool[key] for key in message.get("returned", pool)}
                        reply = {"ok": True, "id": message.get("id"), "pool": returned}
                elif op == "close":
                    connection.send({"ok": True})
                    return 0
                else:
                    reply = {"ok": False, "status": "internal_error", "error": f"unknown op {op!r}"}
            except Exception as error:  # noqa: BLE001 - reported to the host, never hidden
                reply = {
                    "ok": False,
                    "id": message.get("id"),
                    "status": "internal_error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
            connection.send(reply)
    except EOFError:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--authkey", required=True)
    args = parser.parse_args(argv)
    return serve(args.manifest, args.address, base64.urlsafe_b64decode(args.authkey.encode("ascii")))


if __name__ == "__main__":
    sys.exit(main())
