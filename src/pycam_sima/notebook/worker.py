"""MPI worker controlled by :class:`pycam_sima.NotebookSession`."""

from __future__ import annotations

import argparse
import base64
import json
import traceback
from multiprocessing.connection import Client, Connection
from typing import Any, Mapping

import numpy as np

from ..model import CAMDriver, KesslerSchemePlan, ModelConfig, ModelOptions
from ..model.checkpoint import (
    CheckpointBundle,
    deserialize_snapshot,
    restore_driver,
    serialize_snapshot,
)
from ..model.control import model_field_metadata


def _error() -> str:
    return traceback.format_exc()


def _runtime_status(driver: CAMDriver) -> dict[str, Any]:
    return {
        "step": driver.clock.nstep,
        "native_nstep": driver.clock.nstep,
        "native_calls": driver.backend.call_count,
        "phase_status": driver.phase_status,
        "scheme_status": driver.scheme_status,
    }


def _local_command(request: dict[str, Any], driver: CAMDriver, comm: Any) -> Any:
    operation = request.get("op")
    if operation == "step":
        for _ in range(int(request["count"])):
            driver.step()
        return _runtime_status(driver)
    if operation == "prepare_initial_step":
        driver.prepare_initial_step()
        return _runtime_status(driver)
    if operation == "run_phase":
        driver.run_phase(str(request["phase"]))
        return _runtime_status(driver)
    if operation == "run_scheme":
        driver.run_scheme(
            str(request["scheme"]),
            group=(None if request.get("group") is None else str(request["group"])),
        )
        return _runtime_status(driver)
    if operation == "run_scheme_group":
        driver.run_scheme_group(str(request["group"]))
        return _runtime_status(driver)
    if operation == "configure_scheme_plan":
        driver.scheme_plan = KesslerSchemePlan.from_payload(request["plan"])
        return _runtime_status(driver)
    if operation == "write_checkpoint":
        return str(driver.write_checkpoint(str(request["path"])))
    if operation == "capture_memory_checkpoint":
        return serialize_snapshot(driver.snapshot())
    if operation == "edit_field":
        name = str(request["field"])
        current = driver.pool.get(name, unsafe=bool(request.get("unsafe", False)))
        value = float(request["value"])
        edit = str(request["operation"])
        if edit == "set":
            updated = np.full_like(current, value)
        elif edit == "add":
            updated = np.add(current, value)
        elif edit == "multiply":
            updated = np.multiply(current, value)
        else:
            raise ValueError(f"unknown field edit operation: {edit!r}")
        driver.pool.set(
            name,
            updated,
            unsafe=bool(request.get("unsafe", False)),
        )
        return _runtime_status(driver)
    if operation in {"get_field", "get_field_stats", "set_field"}:
        name = str(request["field"])
        array = driver.pool.get(name)
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
            driver.pool.set(
                name,
                np.asarray(request["value"]),
                unsafe=bool(request.get("unsafe", False)),
            )
        return None
    if operation == "close":
        driver.finalize()
        return {"closed": True}
    raise ValueError(f"unknown NotebookSession operation: {operation!r}")


def _collect_response(
    request: dict[str, Any],
    driver: CAMDriver,
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
    failures = [
        f"rank {rank}:\n{item[0]}" for rank, item in enumerate(gathered) if item[0]
    ]
    if failures:
        return {"status": "error", "error": "\n".join(failures)}

    operation = request["op"]
    selector = request.get("rank")
    if operation == "capture_memory_checkpoint":
        result = CheckpointBundle.from_rank_payloads(
            [item[1] for item in gathered]
        )
    elif operation in {"get_field", "get_field_stats"}:
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


def _restore_memory_checkpoint(
    payload: tuple[Mapping[str, Any], bytes] | None,
    driver: CAMDriver,
    comm: Any,
    *,
    config: ModelConfig,
    run_dir: str,
    library: str,
    history_dir: str,
) -> tuple[CAMDriver, dict[str, Any] | None]:
    """Collectively replace a live driver from rank-local in-memory bytes."""

    candidate: CAMDriver | None = None
    failure: str | None = None
    try:
        if payload is None:
            raise RuntimeError("rank-local in-memory checkpoint payload is absent")
        metadata, content = payload
        snapshot = deserialize_snapshot(metadata, content)
        candidate = restore_driver(
            snapshot,
            run_dir=run_dir,
            comm=comm,
            kernel_library=library,
            history_dir=history_dir,
            expected_config=config,
        )
    except BaseException:
        failure = _error()

    failures = comm.allgather(failure)
    messages = [
        f"rank {rank}:\n{message}"
        for rank, message in enumerate(failures)
        if message
    ]
    if messages:
        response = (
            {"status": "error", "error": "\n".join(messages)}
            if comm.rank == 0
            else None
        )
        return driver, response

    assert candidate is not None
    driver.finalize()
    status = _runtime_status(candidate)
    statuses = comm.gather(status, root=0)
    response = None
    if comm.rank == 0:
        response = {
            "status": "ok",
            "result": {
                **statuses[0],
                "restored_from_memory": True,
            },
        }
    return candidate, response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--expected-ranks", required=True, type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--history-dir", required=True)
    parser.add_argument("--timestep-seconds", required=True, type=int)
    parser.add_argument("--physics-profile", required=True)
    parser.add_argument("--scheme-plan-json", required=True)
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

    driver: CAMDriver | None = None
    initialized = False
    try:
        config = ModelConfig.from_yaml(args.config)
        options = ModelOptions(
            timestep_seconds=args.timestep_seconds,
            physics_profile=args.physics_profile,
            mediator_present=False,
        )
        options.validate(config)
        scheme_plan = KesslerSchemePlan.from_payload(json.loads(args.scheme_plan_json))
        driver = CAMDriver(
            config,
            run_dir=args.run_dir,
            comm=comm,
            kernel_library=args.library,
            history_dir=args.history_dir,
            scheme_plan=scheme_plan,
        )
        try:
            driver.start()
            initialized = True
            failure = None
        except BaseException:
            failure = _error()
        failures = comm.gather(failure, root=0)
        startup_ok = True
        if comm.rank == 0:
            assert connection is not None
            messages = [
                f"rank {rank}:\n{value}" for rank, value in enumerate(failures) if value
            ]
            if messages:
                _send(connection, {"status": "error", "error": "\n".join(messages)})
                startup_ok = False
            else:
                startup_ok = _send(
                    connection,
                    {
                        "status": "ok",
                        "result": {
                            "event": "ready",
                            "rank_count": comm.size,
                            "runtime": "model",
                            "step": driver.clock.nstep,
                            "native_nstep": driver.clock.nstep,
                            "initialized_native_calls": driver.backend.call_count,
                            "initialized_abi_checked": driver.backend._abi_checked,
                            "fields": model_field_metadata(driver.pool),
                            "phase_names": driver.phase_names,
                            "phase_status": driver.phase_status,
                            "scheme_names": driver.scheme_names,
                            "scheme_status": driver.scheme_status,
                            "runtime_options": options.describe(),
                        },
                    },
                )
        startup_ok = comm.bcast(startup_ok, root=0)
        if not startup_ok:
            initialized = False
            return 3

        while True:
            abandoned = False
            rank_payloads: tuple[
                tuple[Mapping[str, Any], bytes], ...
            ] | None = None
            snapshot_error: str | None = None
            if comm.rank == 0:
                assert connection is not None
                try:
                    request = connection.recv()
                except (EOFError, OSError):
                    request = {"op": "close"}
                    abandoned = True
                if request.get("op") == "restore_memory_checkpoint":
                    try:
                        snapshot = request.pop("snapshot")
                        if not isinstance(snapshot, CheckpointBundle):
                            raise TypeError(
                                "restore_memory_checkpoint requires "
                                "CheckpointBundle"
                            )
                        rank_payloads = snapshot.rank_payloads()
                        if len(rank_payloads) != comm.size:
                            raise ValueError(
                                "in-memory checkpoint requires "
                                f"{len(rank_payloads)} ranks, got {comm.size}"
                            )
                    except BaseException:
                        snapshot_error = _error()
            else:
                request = None
            request = comm.bcast(request, root=0)
            abandoned = comm.bcast(abandoned, root=0)
            snapshot_error = comm.bcast(snapshot_error, root=0)
            assert driver is not None
            if request.get("op") == "restore_memory_checkpoint":
                if snapshot_error is not None:
                    response = (
                        {"status": "error", "error": snapshot_error}
                        if comm.rank == 0
                        else None
                    )
                else:
                    local_payload = comm.scatter(rank_payloads, root=0)
                    driver, response = _restore_memory_checkpoint(
                        local_payload,
                        driver,
                        comm,
                        config=config,
                        run_dir=args.run_dir,
                        library=args.library,
                        history_dir=args.history_dir,
                    )
            else:
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
