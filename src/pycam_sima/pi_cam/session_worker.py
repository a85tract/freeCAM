"""Long-lived MPI worker used by :class:`PICAMNotebookSession`."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict
from multiprocessing.connection import Client
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from mpi4py import MPI

from .boundary import ReplayBoundaryProvider
from .case import PICAMCase


def _field_catalog(driver: Any) -> dict[str, dict[str, object]]:
    return {
        name: {
            **driver.pool.contract(name).to_payload(),
            "shape": list(driver.pool[name].shape),
            "nbytes": int(driver.pool[name].nbytes),
        }
        for name in driver.pool
    }


def _status(driver: Any) -> dict[str, object]:
    return {
        "lifecycle": driver.lifecycle.value,
        "step": driver.clock.nstep,
        "coupling_step": driver.coupling_step,
        "date": driver.clock.yyyymmdd,
        "seconds": driver.clock.seconds,
        "actions": len(driver.trace),
        "fields": _field_catalog(driver),
        "state_bytes": driver.pool.nbytes,
    }


def _command(command: dict[str, Any], driver: Any, comm: Any) -> object:
    operation = command["op"]
    if operation == "status":
        return _status(driver) if comm.rank == 0 else None
    if operation == "step":
        for _ in range(int(command.get("count", 1))):
            driver.step()
        return _status(driver) if comm.rank == 0 else None
    if operation == "run_action":
        trace = driver.run_action(
            str(command["name"]),
            phase=command.get("phase"),
            experimental=True,
        )
        return asdict(trace) if comm.rank == 0 else None
    if operation == "run_phase":
        traces = driver.run_phase(str(command["phase"]), experimental=True)
        return [asdict(trace) for trace in traces] if comm.rank == 0 else None
    if operation == "field":
        selected = int(command["rank"])
        if not 0 <= selected < comm.size:
            raise ValueError(f"rank must be in 0..{comm.size - 1}")
        return driver.pool[str(command["name"])].copy(order="F") if comm.rank == selected else None
    if operation == "stats":
        values = driver.pool[str(command["name"])]
        selected = command.get("rank", 0)
        if selected == "global":
            local_sum = float(np.asarray(values, dtype=np.float64).sum())
            local_count = int(values.size)
            total_sum = comm.allreduce(local_sum, op=MPI.SUM)
            total_count = comm.allreduce(local_count, op=MPI.SUM)
            minimum = comm.allreduce(float(values.min()), op=MPI.MIN)
            maximum = comm.allreduce(float(values.max()), op=MPI.MAX)
            if comm.rank == 0:
                return {
                    "rank": "global",
                    "count": total_count,
                    "min": minimum,
                    "max": maximum,
                    "mean": total_sum / total_count,
                }
            return None
        selected = int(selected)
        if not 0 <= selected < comm.size:
            raise ValueError(f"rank must be in 0..{comm.size - 1} or 'global'")
        if comm.rank == selected:
            return {
                "rank": selected,
                "shape": tuple(values.shape),
                "dtype": values.dtype.str,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
        return None
    if operation == "close":
        driver.finalize()
        return {"closed": True} if comm.rank == 0 else None
    raise ValueError(f"unknown PI-CAM notebook operation {operation!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--boundary", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-ranks", required=True, type=int)
    args = parser.parse_args(argv)
    comm = MPI.COMM_WORLD
    connection = None
    if comm.rank == 0:
        connection = Client(
            (args.host, args.port),
            authkey=base64.urlsafe_b64decode(args.authkey.encode("ascii")),
        )
    driver = None
    try:
        try:
            if comm.size != args.expected_ranks:
                raise RuntimeError(
                    f"PI-CAM worker expected {args.expected_ranks} ranks, got {comm.size}"
                )
            case = PICAMCase.from_yaml(args.config)
            driver = case.runtime(
                boundary=ReplayBoundaryProvider(args.boundary),
                communicator=comm,
                run_dir=args.run_dir,
            )
            driver.initialize()
            startup_error = None
        except BaseException:
            startup_error = traceback.format_exc()
        startup_errors = comm.gather(startup_error, root=0)
        if comm.rank == 0:
            failures = [
                f"rank {rank}:\n{error}"
                for rank, error in enumerate(startup_errors)
                if error is not None
            ]
            if failures:
                connection.send({"status": "error", "error": "\n".join(failures[:8])})
            else:
                connection.send({"status": "ok", "result": _status(driver)})
        if startup_error is not None or (comm.rank == 0 and failures):
            return 2

        while True:
            command = connection.recv() if comm.rank == 0 else None
            command = comm.bcast(command, root=0)
            try:
                local_result = _command(command, driver, comm)
                local_error = None
            except BaseException:
                local_result = None
                local_error = traceback.format_exc()
            errors = comm.gather(local_error, root=0)
            results = comm.gather(local_result, root=0)
            if comm.rank == 0:
                failures = [
                    f"rank {rank}:\n{error}"
                    for rank, error in enumerate(errors)
                    if error is not None
                ]
                if failures:
                    connection.send(
                        {"status": "error", "error": "\n".join(failures[:8])}
                    )
                else:
                    result = next((item for item in results if item is not None), None)
                    connection.send({"status": "ok", "result": result})
            if command["op"] == "close":
                break
    finally:
        if driver is not None and driver.lifecycle.value != "finalized":
            try:
                driver.finalize()
            except BaseException:
                pass
        if connection is not None:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
