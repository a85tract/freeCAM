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
        "dynamic_fields": tuple(sorted(driver.pool.dynamic_fields)),
        "python_processes": tuple(sorted(driver.python_processes.process_names)),
        "fortran_processes": tuple(sorted(driver.fortran_processes.installed)),
        "kernels": tuple(getattr(driver.backend, "direct_kernels", ())),
        "step_plan": driver.step_plan.describe(),
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
    if operation == "trace":
        since = int(command.get("since", 0))
        if not 0 <= since <= len(driver.trace):
            raise ValueError(f"trace cursor must be in 0..{len(driver.trace)}")
        return (
            [asdict(record) for record in driver.trace[since:]]
            if comm.rank == 0
            else None
        )
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
    if operation == "run_kernel":
        trace = driver.run_kernel(str(command["name"]), experimental=True)
        return asdict(trace) if comm.rank == 0 else None
    leaf_expansions = {
        "expand_cam_run1_leaves": driver.expand_cam_run1_leaves,
        "expand_cam_run2_leaves": driver.expand_cam_run2_leaves,
        "expand_cam_run4_leaves": driver.expand_cam_run4_leaves,
        "expand_cam_run2_run4_leaves": (
            driver.expand_cam_run2_run4_leaves
        ),
    }
    if operation in leaf_expansions:
        actions = leaf_expansions[operation](experimental=True)
        return (
            [
                {
                    "phase": action.phase,
                    "name": action.name,
                    "operation": action.operation,
                    "native_id": action.native_id,
                }
                for action in actions
            ]
            if comm.rank == 0
            else None
        )
    if operation == "create_field":
        values = driver.define_variable(command["spec"])
        if comm.rank == 0:
            return {
                "name": driver.pool.canonical_name(str(command["spec"]["name"])),
                "shape": tuple(values.shape),
                "dtype": values.dtype.str,
                "nbytes": int(values.nbytes),
            }
        return None
    if operation == "delete_field":
        name = driver.pool.canonical_name(str(command["name"]))
        driver.delete_variable(name)
        return {"name": name, "deleted": True} if comm.rank == 0 else None
    if operation == "install_python":
        record = driver.python_processes.install(
            command["spec"], unsafe=bool(command.get("unsafe", False))
        )
        if comm.rank == 0:
            return {
                "name": record.spec.name,
                "phase": record.spec.group,
                "payload_hash": record.spec.payload_hash,
                "reads": tuple(record.read_bindings),
                "writes": tuple(record.write_bindings),
            }
        return None
    if operation == "remove_python":
        result = driver.python_processes.remove(str(command["name"]))
        return result if comm.rank == 0 else None
    if operation == "install_fortran":
        record = driver.fortran_processes.install(
            command["spec"], unsafe=bool(command.get("unsafe", False))
        )
        if comm.rank == 0:
            return {
                "name": record.spec.process,
                "phase": record.spec.phase,
                "device": record.device.name,
                "manifest": str(record.device.manifest_path),
                "library": str(record.device.library_path),
            }
        return None
    if operation == "remove_fortran":
        result = driver.fortran_processes.remove(str(command["name"]))
        return result if comm.rank == 0 else None
    if operation == "set_action_enabled":
        driver.step_plan.set_enabled(
            str(command["name"]),
            bool(command["enabled"]),
            phase=command.get("phase"),
            experimental=True,
        )
        action = driver.step_plan.select(
            str(command["name"]), phase=command.get("phase")
        )
        return (
            {"name": action.name, "phase": action.phase, "enabled": action.enabled}
            if comm.rank == 0
            else None
        )
    if operation == "move_action":
        driver.step_plan.move(
            str(command["name"]),
            phase=command.get("phase"),
            before=command.get("before"),
            after=command.get("after"),
            experimental=True,
        )
        action = driver.step_plan.select(
            str(command["name"]), phase=command.get("phase")
        )
        return (
            {"name": action.name, "phase": action.phase, "plan": driver.step_plan.describe()}
            if comm.rank == 0
            else None
        )
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
