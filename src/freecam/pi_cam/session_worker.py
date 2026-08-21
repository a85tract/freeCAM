"""Long-lived MPI worker used by :class:`PICAMNotebookSession`."""

from __future__ import annotations

import argparse
import base64
import traceback
from dataclasses import asdict
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
from mpi4py import MPI

from freecam.model.collective import collective_error_message

from .boundary import CAMBoundaryProvider, ReplayBoundaryProvider
from .case import PICAMCase
from .config import DEFAULT_TRACE_LIMIT
from .expressions import assign_expression, evaluate_expression
from .state import edit_active_field, selected_active_values


def _parse_trace_limit(text: str) -> int | None:
    return None if text.strip().lower() == "none" else int(text)


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
    timing_dir = None if driver.run_dir is None else driver.run_dir / "timing"
    return {
        "lifecycle": driver.lifecycle.value,
        "step": driver.clock.nstep,
        "native_step": driver.native_step,
        "coupling_step": driver.coupling_step,
        "date": driver.clock.yyyymmdd,
        "seconds": driver.clock.seconds,
        "actions": driver.trace_count,
        "trace_retained": driver.trace_retained,
        "trace_first_sequence": driver.trace_first_sequence,
        "trace_limit": driver.trace_limit,
        "fields": _field_catalog(driver),
        "dynamic_fields": tuple(sorted(driver.pool.dynamic_fields)),
        "python_processes": tuple(sorted(driver.python_processes.process_names)),
        "fortran_processes": tuple(sorted(driver.fortran_processes.installed)),
        "promoted_processes": tuple(
            driver.process_contexts.record(name).to_payload()
            for name in driver.process_contexts.names
        ),
        "kernels": tuple(getattr(driver.backend, "direct_kernels", ())),
        "process_adapters": tuple(
            getattr(driver.backend, "process_adapters", ())
        ),
        "step_plan": driver.step_plan.describe(),
        "state_bytes": driver.pool.nbytes,
        "output": {
            "history_every": driver.history_every,
            "restart_every": driver.restart_every,
            "restart_at_end": driver.restart_at_end,
        },
        "boundary_export_verification": bool(
            getattr(driver.boundary, "verify_exports", False)
        ),
        "boundary": dict(getattr(driver.boundary, "diagnostics", {})),
        "timing": {
            "enabled": True,
            "clock": "MPI_Wtime",
            "recorded_regions": driver.profiler.region_count,
            "detail": (
                None
                if timing_dir is None
                else str(timing_dir / "freecam_timing.0000")
            ),
            "global_stats": (
                None
                if timing_dir is None
                else str(timing_dir / "freecam_timing_stats")
            ),
            "written_on": "model.close()",
        },
    }


def _command(command: dict[str, Any], driver: Any, comm: Any) -> object:
    operation = command["op"]
    if operation == "status":
        return _status(driver) if comm.rank == 0 else None
    if operation == "step":
        for _ in range(int(command.get("count", 1))):
            driver.step()
        return _status(driver) if comm.rank == 0 else None
    if operation == "configure_output":
        driver.configure_output(
            history_every=command.get("history_every"),
            restart_every=command.get("restart_every"),
            restart_at_end=bool(command.get("restart_at_end", False)),
        )
        return _status(driver) if comm.rank == 0 else None
    if operation == "trace":
        if comm.rank != 0:
            return None
        records = driver.trace_since(int(command.get("since", 0)))
        return {
            "first_sequence": (
                records[0].sequence if records else driver.trace_count
            ),
            "total": driver.trace_count,
            "records": [asdict(record) for record in records],
        }
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
    if operation == "promote_process":
        record = driver.promote_process(
            str(command["name"]),
            bindings=command.get("bindings"),
            initials=command.get("initials"),
            dimensions=command.get("dimensions"),
        )
        return record.to_payload() if comm.rank == 0 else None
    if operation == "run_promoted_process":
        trace = driver.run_promoted_process(
            str(command["name"]), experimental=True
        )
        return asdict(trace) if comm.rank == 0 else None
    if operation == "install_promoted_process":
        action = driver.install_promoted_process(
            str(command["name"]),
            before=command.get("before"),
            after=command.get("after"),
            enabled=bool(command.get("enabled", True)),
        )
        return (
            {
                "name": action.name,
                "phase": action.phase,
                "enabled": action.enabled,
                "plan": driver.step_plan.describe(),
            }
            if comm.rank == 0
            else None
        )
    if operation == "remove_promoted_process":
        record = driver.remove_promoted_process(str(command["name"]))
        return record.to_payload() if comm.rank == 0 else None
    if operation == "install_history_stream":
        action = driver.install_history_stream(
            str(command["name"]),
            fields=(
                None if command.get("fields") is None else tuple(command["fields"])
            ),
            stream=str(command.get("stream", "h0")),
            nhtfrq=int(command.get("nhtfrq", 0)),
            before=command.get("before"),
            after=command.get("after", "wshist"),
            enabled=bool(command.get("enabled", True)),
            precision=str(command.get("precision", "float32")),
            time_period=str(command.get("time_period", "mean")),
        )
        # ``describe`` resolves each stream's fields, and resolving them
        # builds the collective column map.  Every rank has to reach it.
        streams = driver.history_streams.describe()
        return (
            {
                "action": {
                    "name": action.name,
                    "phase": action.phase,
                    "operation": action.operation,
                    "kind": action.kind,
                    "enabled": action.enabled,
                },
                "streams": streams,
                "plan": driver.step_plan.describe(),
            }
            if comm.rank == 0
            else None
        )
    if operation == "remove_history_stream":
        driver.remove_history_stream(str(command["name"]))
        streams = driver.history_streams.describe()
        return (
            {"removed": str(command["name"]), "streams": streams}
            if comm.rank == 0
            else None
        )
    if operation == "history_streams":
        streams = driver.history_streams.describe()
        return streams if comm.rank == 0 else None
    if operation == "drain_history_streams":
        written = driver.history_streams.drain()
        return {"written": int(written)} if comm.rank == 0 else None
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
    if operation == "create_array":
        values = driver.define_array(str(command["name"]), command["values"])
        if comm.rank == 0:
            return {
                "name": driver.pool.canonical_name(str(command["name"])),
                "shape": tuple(values.shape),
                "dtype": values.dtype.str,
                "nbytes": int(values.nbytes),
            }
        return None
    if operation == "delete_field":
        name = driver.pool.canonical_name(str(command["name"]))
        driver.delete_variable(name)
        return {"name": name, "deleted": True} if comm.rank == 0 else None
    if operation == "edit_field":
        name = driver.pool.canonical_name(str(command["name"]))
        contract = driver.pool.contract(name)
        if not contract.writable:
            raise ValueError(f"StatePool field {name!r} is read-only")
        value = command["value"]
        edit = str(command["operation"])
        selection = command.get("selection", Ellipsis)
        edit_active_field(
            driver.pool,
            name,
            selection=selection,
            operation=edit,
            value=value,
        )
        return (
            {
                "name": name,
                "operation": edit,
                "value": value,
                "selection": selection,
            }
            if comm.rank == 0
            else None
        )
    if operation == "assign_expression":
        name = driver.pool.canonical_name(str(command["name"]))
        if not driver.pool.contract(name).writable:
            raise ValueError(f"field {name!r} is read-only")
        selection = command.get("selection", Ellipsis)
        count = assign_expression(
            driver.pool,
            name,
            command["expression"],
            selection=selection,
        )
        return (
            {"name": name, "selection": selection, "active_values": count}
            if comm.rank == 0
            else None
        )
    if operation == "evaluate_expression":
        selected = int(command["rank"])
        if not 0 <= selected < comm.size:
            raise ValueError(f"rank must be in 0..{comm.size - 1}")
        if comm.rank != selected:
            return None
        return np.array(
            evaluate_expression(driver.pool, command["expression"]),
            copy=True,
            order="F",
            subok=False,
        )
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
    if operation == "reload_python":
        result = driver.python_processes.reload(
            str(command["name"]),
            command["spec"],
            preserve_parameters=bool(command.get("preserve_parameters", False)),
            preserve_transactional=bool(
                command.get("preserve_transactional", False)
            ),
            unsafe=bool(command.get("unsafe", False)),
        )
        return result if comm.rank == 0 else None
    if operation == "remove_python":
        result = driver.python_processes.remove(str(command["name"]))
        return result if comm.rank == 0 else None
    if operation == "set_python_parameters":
        result = driver.python_processes.set_parameters(
            str(command["name"]), dict(command["parameters"])
        )
        return result if comm.rank == 0 else None
    if operation == "get_python_parameters":
        result = driver.python_processes.parameters(str(command["name"]))
        return result if comm.rank == 0 else None
    if operation == "set_module_parameter":
        result = driver.module_parameters.set(
            str(command["name"]), command["value"]
        )
        return result if comm.rank == 0 else None
    if operation == "get_module_parameters":
        result = driver.module_parameters.describe()
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
            {
                "name": action.name,
                "phase": action.phase,
                "enabled": action.enabled,
                "plan": driver.step_plan.describe(),
            }
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
    if operation == "replace_workflow":
        driver.step_plan.replace_enabled_order(
            tuple(str(name) for name in command["order"]),
            experimental=True,
        )
        return (
            {"plan": driver.step_plan.describe()}
            if comm.rank == 0
            else None
        )
    if operation == "field":
        selected = int(command["rank"])
        if not 0 <= selected < comm.size:
            raise ValueError(f"rank must be in 0..{comm.size - 1}")
        if comm.rank != selected:
            return None
        selection = command.get("selection", Ellipsis)
        return np.array(
            driver.pool[str(command["name"])][selection],
            copy=True,
            order="F",
            subok=False,
        )
    if operation == "stats":
        name = driver.pool.canonical_name(str(command["name"]))
        values = driver.pool[name]
        selection = command.get("selection", Ellipsis)
        active = selected_active_values(driver.pool, name, selection)
        if active.size == 0:
            raise ValueError("field selection contains no active CAM values")
        selected = command.get("rank", 0)
        if selected == "global":
            local_sum = float(np.asarray(active, dtype=np.float64).sum())
            local_count = int(active.size)
            total_sum = comm.allreduce(local_sum, op=MPI.SUM)
            total_count = comm.allreduce(local_count, op=MPI.SUM)
            local_minimum = float(active.min())
            local_maximum = float(active.max())
            minimum = comm.allreduce(local_minimum, op=MPI.MIN)
            maximum = comm.allreduce(local_maximum, op=MPI.MAX)
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
                "shape": tuple(np.asarray(values[selection]).shape),
                "dtype": values.dtype.str,
                "count": int(active.size),
                "min": float(active.min()),
                "max": float(active.max()),
                "mean": float(np.asarray(active, dtype=np.float64).sum())
                / int(active.size),
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
    boundary_group = parser.add_mutually_exclusive_group(required=True)
    boundary_group.add_argument("--boundary", type=Path)
    boundary_group.add_argument("--boundary-provider", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-ranks", required=True, type=int)
    parser.add_argument(
        "--verify-exports",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare every CAM export with the replay oracle",
    )
    parser.add_argument(
        "--trace-limit",
        type=_parse_trace_limit,
        default=DEFAULT_TRACE_LIMIT,
        help="retained action-trace records per rank; 'none' for unbounded",
    )
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
            if args.boundary_provider is None:
                boundary = ReplayBoundaryProvider(
                    args.boundary,
                    verify_exports=args.verify_exports,
                )
            else:
                boundary = cloudpickle.loads(args.boundary_provider.read_bytes())
                if not isinstance(boundary, CAMBoundaryProvider):
                    raise TypeError(
                        "boundary provider payload is not a CAMBoundaryProvider"
                    )
            driver = case.runtime(
                boundary=boundary,
                communicator=comm,
                run_dir=args.run_dir,
                trace_limit=args.trace_limit,
            )
            driver.initialize()
            startup_error = None
        except BaseException:
            startup_error = traceback.format_exc()
        startup_errors = comm.gather(startup_error, root=0)
        if comm.rank == 0:
            failure = collective_error_message(
                "PI-CAM worker startup", startup_errors
            )
            if failure is not None:
                connection.send({"status": "error", "error": failure})
            else:
                connection.send({"status": "ok", "result": _status(driver)})
        if startup_error is not None or (comm.rank == 0 and failure is not None):
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
                failure = collective_error_message(
                    f"PI-CAM command {command.get('op', '<unknown>')!r}",
                    errors,
                )
                if failure is not None:
                    connection.send({"status": "error", "error": failure})
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
