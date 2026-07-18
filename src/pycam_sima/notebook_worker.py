from __future__ import annotations

import argparse
import base64
import traceback
from multiprocessing.connection import Client, Connection
from typing import Any

import numpy as np

from .config import CaseConfig
from .full_driver import FullCAMDriver


def _error() -> str:
    return traceback.format_exc()


def _field_metadata(driver: FullCAMDriver) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in driver.pool:
        array = driver.pool[name]
        spec = driver.pool.spec(name)
        result[name] = {
            "shape": tuple(array.shape),
            "dtype": array.dtype.str,
            "dimensions": spec.dimensions,
            "owner": spec.owner,
        }
    return result


def _local_command(request: dict[str, Any], driver: FullCAMDriver, comm: Any) -> Any:
    operation = request.get("op")
    if operation == "step":
        driver.run(int(request["count"]))
        return {"step": driver.clock.step, "native_nstep": driver.backend.nstep}

    if operation in {"get_field", "get_field_stats", "set_field"}:
        name = str(request["field"])
        array = driver.pool.require(name)
        selector = request["rank"]
        selected = selector == "all" or selector == comm.rank
        if operation == "get_field":
            return np.array(array, copy=True, order="F") if selected else None
        if operation == "get_field_stats":
            if not selected:
                return None
            return {
                "rank": int(comm.rank),
                "shape": tuple(array.shape),
                "dtype": array.dtype.str,
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
            }
        if selected:
            value = np.asarray(request["value"])
            if value.ndim == 0:
                array.fill(value.item())
            else:
                if value.shape != array.shape:
                    raise ValueError(
                        f"{name} on rank {comm.rank}: expected shape {array.shape}, "
                        f"got {value.shape}"
                    )
                array[...] = value
        return None

    if operation == "close":
        driver.finalize()
        return {"closed": True}

    raise ValueError(f"unknown NotebookSession operation: {operation!r}")


def _collect_response(
    request: dict[str, Any],
    driver: FullCAMDriver,
    comm: Any,
) -> dict[str, Any] | None:
    try:
        payload = _local_command(request, driver, comm)
        failure = None
    except BaseException:
        payload = None
        failure = _error()

    gathered = comm.gather((failure, payload), root=0)
    if comm.rank != 0:
        return None
    failures = [f"rank {rank}:\n{item[0]}" for rank, item in enumerate(gathered) if item[0]]
    if failures:
        return {"status": "error", "error": "\n".join(failures)}

    operation = request["op"]
    selector = request.get("rank")
    if operation in {"get_field", "get_field_stats"}:
        if selector == "all":
            result = [item[1] for item in gathered]
        else:
            result = gathered[int(selector)][1]
    else:
        result = gathered[0][1]
    return {"status": "ok", "result": result}


def _send(connection: Connection, payload: dict[str, Any]) -> bool:
    try:
        connection.send(payload)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--expected-ranks", required=True, type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    connection: Connection | None = None
    connected = True
    if comm.rank == 0:
        try:
            connection = Client(
                (args.host, args.port),
                authkey=base64.urlsafe_b64decode(args.authkey.encode("ascii")),
            )
        except BaseException:
            connected = False
    connected = comm.bcast(connected, root=0)
    if not connected:
        return 2
    assert connection is not None if comm.rank == 0 else True

    if comm.size != args.expected_ranks:
        if comm.rank == 0:
            assert connection is not None
            _send(
                connection,
                {
                    "status": "error",
                    "error": f"expected {args.expected_ranks} MPI ranks, got {comm.size}",
                },
            )
        return 2

    driver: FullCAMDriver | None = None
    initialized = False
    try:
        config = CaseConfig.from_yaml(args.config)
        driver = FullCAMDriver(
            config,
            comm,
            library=args.library,
            run_dir=args.run_dir,
        )
        try:
            driver.initialize()
            initialized = True
            failure = None
        except BaseException:
            failure = _error()
        failures = comm.gather(failure, root=0)
        startup_ok = True
        if comm.rank == 0:
            assert connection is not None
            messages = [f"rank {rank}:\n{value}" for rank, value in enumerate(failures) if value]
            if messages:
                startup_ok = _send(
                    connection,
                    {"status": "error", "error": "\n".join(messages)},
                )
                startup_ok = False
            else:
                startup_ok = _send(
                    connection,
                    {
                        "status": "ok",
                        "result": {
                            "event": "ready",
                            "rank_count": comm.size,
                            "step": driver.clock.step,
                            "native_nstep": driver.backend.nstep,
                            "fields": _field_metadata(driver),
                        },
                    },
                )
        startup_ok = comm.bcast(startup_ok, root=0)
        if not startup_ok:
            initialized = False
            return 3

        while True:
            abandoned = False
            if comm.rank == 0:
                assert connection is not None
                try:
                    request = connection.recv()
                except (EOFError, OSError):
                    request = {"op": "close"}
                    abandoned = True
            else:
                request = None
            request = comm.bcast(request, root=0)
            abandoned = comm.bcast(abandoned, root=0)
            assert driver is not None
            response = _collect_response(request, driver, comm)
            if request.get("op") == "close":
                initialized = False
            delivered = True
            if comm.rank == 0 and not abandoned:
                assert connection is not None and response is not None
                delivered = _send(connection, response)
            delivered = comm.bcast(delivered, root=0)
            if request.get("op") == "close" or not delivered:
                break
    finally:
        if initialized and driver is not None:
            try:
                driver.finalize()
            except BaseException:
                traceback.print_exc()
        if connection is not None:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
