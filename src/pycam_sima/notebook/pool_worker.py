"""One persistent MPI world partitioned into independent CAM model slots.

Only rank zero owns the authenticated controller socket.  Commands are
broadcast over ``MPI.COMM_WORLD`` and executed by the selected slot
communicator.  A fork never sends model arrays through the controller socket:
each parent rank sends its local serialized snapshot directly to the
corresponding rank in every child slot.
"""

from __future__ import annotations

import argparse
import base64
import json
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..model import (
    CAMDriver,
    CCPPSuitePlan,
    ModelConfig,
    ModelOptions,
    SegmentPlan,
    execute_segment_plan,
)
from ..model.checkpoint import deserialize_snapshot, restore_driver, serialize_snapshot
from ..model.control import model_field_metadata
from .worker import _local_command, _runtime_status, _send


def validate_pool_layout(
    world_size: int,
    ranks_per_model: int,
    model_slots: int,
) -> None:
    """Validate the fixed communicator layout before any model is allocated."""

    if ranks_per_model <= 0:
        raise ValueError("ranks_per_model must be positive")
    if model_slots <= 0:
        raise ValueError("model_slots must be positive")
    expected = ranks_per_model * model_slots
    if world_size != expected:
        raise ValueError(
            f"pool requires {model_slots} slots x {ranks_per_model} ranks = "
            f"{expected} MPI ranks, got {world_size}"
        )


def slot_for_world_rank(world_rank: int, ranks_per_model: int) -> tuple[int, int]:
    """Return ``(slot_id, slot_rank)`` for one world rank."""

    return divmod(int(world_rank), int(ranks_per_model))


@dataclass(slots=True)
class SlotRuntime:
    """Rank-local part of one fixed model slot."""

    slot_id: int
    slot_rank: int
    comm: Any
    config: ModelConfig
    library: str
    scheme_plan: CCPPSuitePlan
    driver: CAMDriver | None = None
    model_name: str | None = None
    run_dir: str | None = None
    history_dir: str | None = None
    state: str = "idle"

    def require_model(self, expected_name: str | None = None) -> CAMDriver:
        if self.driver is None or self.model_name is None:
            raise RuntimeError(f"model slot {self.slot_id} is idle")
        if expected_name is not None and expected_name != self.model_name:
            raise RuntimeError(
                f"slot {self.slot_id} contains {self.model_name!r}, "
                f"not {expected_name!r}"
            )
        return self.driver

    def release(self) -> None:
        driver, self.driver = self.driver, None
        self.state = "idle"
        self.model_name = None
        self.run_dir = None
        self.history_dir = None
        if driver is not None:
            driver.finalize()

    def local_state_bytes(self) -> int:
        if self.driver is None or self.driver.pool is None:
            return 0
        return self.driver.pool.array_nbytes


def _traceback() -> str:
    return traceback.format_exc()


def _leader_result(world: Any, slot: SlotRuntime, value: Any) -> Any:
    """Move one slot leader's small result to world rank zero."""

    candidate = value if slot.slot_rank == 0 else None
    gathered = world.gather(candidate, root=0)
    if world.rank != 0:
        return None
    return next((item for item in gathered if item is not None), None)


def _world_failures(world: Any, failure: str | None) -> list[str]:
    failures = world.allgather(failure)
    return [
        f"world rank {rank}:\n{message}"
        for rank, message in enumerate(failures)
        if message
    ]


def _slot_status(slot: SlotRuntime) -> dict[str, Any] | None:
    state_bytes = slot.comm.reduce(slot.local_state_bytes(), root=0)
    if slot.slot_rank != 0:
        return None
    result: dict[str, Any] = {
        "slot_id": slot.slot_id,
        "state": slot.state,
        "model_name": slot.model_name,
        "ranks": tuple(
            range(
                slot.slot_id * int(slot.comm.size),
                (slot.slot_id + 1) * int(slot.comm.size),
            )
        ),
        "rank_count": int(slot.comm.size),
        "state_bytes": int(state_bytes),
    }
    if slot.driver is not None:
        result.update(
            step=int(slot.driver.clock.nstep),
            native_calls=int(slot.driver.backend.call_count),
        )
    return result


def _all_slot_statuses(world: Any, slot: SlotRuntime) -> list[dict[str, Any]] | None:
    local = _slot_status(slot)
    gathered = world.gather(local, root=0)
    if world.rank != 0:
        return None
    return sorted(
        (item for item in gathered if item is not None),
        key=lambda item: int(item["slot_id"]),
    )


def _collect_slot_command(
    slot: SlotRuntime,
    command: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Execute one existing Notebook worker command inside one slot."""

    failure: str | None = None
    payload: Any = None
    try:
        if command.get("op") == "close":
            raise ValueError("use close_model to release a pooled model slot")
        slot.state = "running"
        driver = slot.require_model(command.get("model_name"))
        payload = _pooled_local_command(dict(command), driver, slot.comm)
    except BaseException:
        failure = _traceback()
    finally:
        if slot.driver is not None:
            slot.state = "ready"
    gathered = slot.comm.gather((failure, payload), root=0)
    if slot.slot_rank != 0:
        return None
    failures = [
        f"slot rank {rank}:\n{item[0]}"
        for rank, item in enumerate(gathered)
        if item[0]
    ]
    if failures:
        return {"error": "\n".join(failures)}

    operation = str(command["op"])
    selector = command.get("rank")
    if operation == "capture_memory_checkpoint":
        # Explicit persistence remains supported; pooled fork does not use it.
        from ..model.checkpoint import CheckpointBundle

        result = CheckpointBundle.from_rank_payloads(
            [item[1] for item in gathered]
        )
    elif operation in {"get_field", "get_field_stats"}:
        result = (
            [item[1] for item in gathered]
            if selector == "all"
            else gathered[int(selector)][1]
        )
    else:
        result = gathered[0][1]
    return {"result": result}


def _pooled_local_command(
    command: dict[str, Any],
    driver: CAMDriver,
    comm: Any,
) -> Any:
    """Add pooled-model introspection/edit operations to the base protocol."""

    operation = command.get("op")
    if operation == "describe":
        return {
            **_runtime_status(driver),
            "fields": model_field_metadata(driver.pool),
            "phase_names": driver.phase_names,
            "scheme_names": driver.scheme_names,
        }
    if operation == "field_info":
        name = str(command["field"])
        try:
            return model_field_metadata(driver.pool)[name]
        except KeyError as exc:
            raise KeyError(f"unknown CAM-SIMA field: {name}") from exc
    if operation == "set_scheme_enabled":
        plan = driver.scheme_plan.copy()
        method = plan.enable if bool(command["enabled"]) else plan.disable
        method(
            str(command["scheme"]),
            group=command.get("group"),
            **(
                {}
                if bool(command["enabled"])
                else {"unsafe": bool(command.get("unsafe", False))}
            ),
        )
        driver.scheme_plan = plan
        return _runtime_status(driver)
    if operation == "move_scheme":
        plan = driver.scheme_plan.copy()
        plan.move(
            str(command["scheme"]),
            before=command.get("before"),
            after=command.get("after"),
            group=command.get("group"),
            to_group=command.get("to_group"),
            unsafe=bool(command.get("unsafe", False)),
        )
        driver.scheme_plan = plan
        return _runtime_status(driver)
    if operation == "reset_scheme_plan":
        driver.scheme_plan = CCPPSuitePlan.from_xml(driver.config.verify_suite())
        return _runtime_status(driver)
    if operation == "describe_scheme_plan":
        return driver.scheme_plan.describe(command.get("group"))
    if operation == "run_plan":
        plan = SegmentPlan.from_mapping(command["plan"])
        trace = execute_segment_plan(driver, plan)
        return {**_runtime_status(driver), "action_trace": trace}
    return _local_command(command, driver, comm)


def _create_model(
    world: Any,
    slot: SlotRuntime,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    target = int(request["slot"])
    failure: str | None = None
    candidate: CAMDriver | None = None
    if slot.slot_id == target:
        try:
            if slot.driver is not None:
                raise RuntimeError(f"model slot {target} is already occupied")
            name = str(request["name"])
            run_dir = str(Path(request["run_dir"]).resolve())
            history_dir = str(Path(request["history_dir"]).resolve())
            slot.state = "initializing"
            candidate = CAMDriver(
                slot.config,
                run_dir=run_dir,
                comm=slot.comm,
                kernel_library=slot.library,
                history_dir=history_dir,
                scheme_plan=slot.scheme_plan.copy(),
            ).start()
            slot.driver = candidate
            slot.model_name = name
            slot.run_dir = run_dir
            slot.history_dir = history_dir
            slot.state = "ready"
        except BaseException:
            if candidate is not None:
                try:
                    candidate.finalize()
                except BaseException:
                    pass
            slot.state = "failed"
            failure = _traceback()
    if not 0 <= target < int(world.size) // int(slot.comm.size):
        failure = f"model slot {target} is outside this pool"

    failures = _world_failures(world, failure)
    if failures:
        return {"status": "error", "error": "\n".join(failures)} if world.rank == 0 else None

    local: dict[str, Any] | None = None
    if slot.slot_id == target and slot.slot_rank == 0:
        driver = slot.require_model()
        local = {
            **_runtime_status(driver),
            "slot_id": target,
            "model_name": slot.model_name,
            "fields": model_field_metadata(driver.pool),
            "phase_names": driver.phase_names,
            "scheme_names": driver.scheme_names,
        }
    result = _leader_result(world, slot, local)
    return {"status": "ok", "result": result} if world.rank == 0 else None


def _run_model_commands(
    world: Any,
    slot: SlotRuntime,
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    by_slot: dict[int, Mapping[str, Any]] = {}
    validation_error: str | None = None
    try:
        for item in commands:
            target = int(item["slot"])
            if not 0 <= target < int(world.size) // int(slot.comm.size):
                raise ValueError(f"model slot {target} is outside this pool")
            if target in by_slot:
                raise ValueError(
                    f"one pooled command may contain at most one action per slot: {target}"
                )
            by_slot[target] = item
    except BaseException:
        validation_error = _traceback()
    validation_errors = _world_failures(world, validation_error)
    if validation_errors:
        return (
            {"status": "error", "error": "\n".join(validation_errors)}
            if world.rank == 0
            else None
        )

    local = None
    if slot.slot_id in by_slot:
        item = by_slot[slot.slot_id]
        command = dict(item["command"])
        command["model_name"] = item.get("name")
        local = _collect_slot_command(slot, command)

    gathered = world.gather(
        (
            int(slot.slot_id),
            None if local is None else local.get("error"),
            None if local is None else local.get("result"),
        )
        if slot.slot_rank == 0
        else None,
        root=0,
    )
    if world.rank != 0:
        return None
    records = [item for item in gathered if item is not None]
    failures = [
        f"slot {slot_id}:\n{error}"
        for slot_id, error, _result in records
        if error
    ]
    if failures:
        return {"status": "error", "error": "\n".join(failures)}
    results = {
        str(slot_id): result
        for slot_id, _error, result in records
        if slot_id in by_slot
    }
    return {"status": "ok", "result": results}


def _fork_models(
    world: Any,
    slot: SlotRuntime,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Copy one parent into child slots using only in-world MPI transfers."""

    parent_slot = int(request["parent_slot"])
    parent_name = str(request["parent_name"])
    children = tuple(dict(item) for item in request["children"])
    child_slots = tuple(int(item["slot"]) for item in children)
    failure: str | None = None
    payload: tuple[Mapping[str, Any], bytes] | None = None

    try:
        if not children:
            raise ValueError("fork_model requires at least one child")
        if len(set(child_slots)) != len(child_slots):
            raise ValueError("fork_model child slots must be unique")
        if parent_slot in child_slots:
            raise ValueError("fork_model cannot overwrite its parent slot")
        if not all(0 <= value for value in (parent_slot, *child_slots)):
            raise ValueError("fork_model slot ids must be non-negative")
        model_slots = int(world.size) // int(slot.comm.size)
        if any(value >= model_slots for value in (parent_slot, *child_slots)):
            raise ValueError(
                f"fork_model slot id is outside 0..{model_slots - 1}"
            )
        if slot.slot_id == parent_slot:
            driver = slot.require_model(parent_name)
            slot.comm.Barrier()
            payload = serialize_snapshot(
                driver.snapshot(allow_recreatable_process_state=True)
            )
        elif slot.slot_id in child_slots and slot.driver is not None:
            raise RuntimeError(f"fork target slot {slot.slot_id} is already occupied")
    except BaseException:
        failure = _traceback()

    failures = _world_failures(world, failure)
    if failures:
        return {"status": "error", "error": "\n".join(failures)} if world.rank == 0 else None

    # Same local rank in parent and child slot.  The bytes never visit rank
    # zero unless rank zero is itself the corresponding parent rank.
    tag_base = 21000
    if slot.slot_id == parent_slot:
        assert payload is not None
        for child_slot in child_slots:
            destination = child_slot * int(slot.comm.size) + slot.slot_rank
            world.send(payload, dest=destination, tag=tag_base + child_slot)
    elif slot.slot_id in child_slots:
        source = parent_slot * int(slot.comm.size) + slot.slot_rank
        payload = world.recv(source=source, tag=tag_base + slot.slot_id)

    candidate: CAMDriver | None = None
    failure = None
    if slot.slot_id in child_slots:
        child = children[child_slots.index(slot.slot_id)]
        try:
            assert payload is not None
            metadata, content = payload
            snapshot = deserialize_snapshot(metadata, content)
            candidate = restore_driver(
                snapshot,
                run_dir=str(Path(child["run_dir"]).resolve()),
                comm=slot.comm,
                kernel_library=slot.library,
                history_dir=str(Path(child["history_dir"]).resolve()),
                expected_config=slot.config,
            )
        except BaseException:
            failure = _traceback()

    failures = _world_failures(world, failure)
    if failures:
        if candidate is not None:
            candidate.finalize()
        return {"status": "error", "error": "\n".join(failures)} if world.rank == 0 else None

    if slot.slot_id in child_slots:
        child = children[child_slots.index(slot.slot_id)]
        assert candidate is not None
        slot.driver = candidate
        slot.model_name = str(child["name"])
        slot.run_dir = str(Path(child["run_dir"]).resolve())
        slot.history_dir = str(Path(child["history_dir"]).resolve())
        slot.state = "ready"

    statuses = _all_slot_statuses(world, slot)
    if world.rank != 0:
        return None
    selected = {
        str(item["model_name"]): item
        for item in statuses
        if int(item["slot_id"]) in child_slots
    }
    return {"status": "ok", "result": selected}


def _close_model(
    world: Any,
    slot: SlotRuntime,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    target = int(request["slot"])
    failure: str | None = None
    if slot.slot_id == target:
        try:
            if slot.driver is not None:
                slot.require_model(request.get("name"))
            elif slot.state != "failed":
                raise RuntimeError(f"model slot {target} is already idle")
            slot.release()
        except BaseException:
            failure = _traceback()
    failures = _world_failures(world, failure)
    if failures:
        return {"status": "error", "error": "\n".join(failures)} if world.rank == 0 else None
    statuses = _all_slot_statuses(world, slot)
    return (
        {"status": "ok", "result": statuses[target]}
        if world.rank == 0
        else None
    )


def serve_pool(args: argparse.Namespace, *, world: Any | None = None) -> int:
    """Run the persistent pooled worker protocol until ``close_pool``."""

    if world is None:
        from mpi4py import MPI

        world = MPI.COMM_WORLD
    try:
        validate_pool_layout(
            int(world.size), int(args.ranks_per_model), int(args.model_slots)
        )
    except ValueError as exc:
        if world.rank == 0:
            print(str(exc), flush=True)
        return 2

    slot_id, slot_rank = slot_for_world_rank(world.rank, args.ranks_per_model)
    slot_comm = world.Split(color=slot_id, key=slot_rank)
    config = ModelConfig.from_yaml(args.config).with_overrides(
        mpi_size=int(args.ranks_per_model)
    )
    options = ModelOptions(
        timestep_seconds=int(args.timestep_seconds),
        physics_profile=str(args.physics_profile),
        mediator_present=False,
    )
    options.validate(config)
    scheme_plan = CCPPSuitePlan.from_payload(json.loads(args.scheme_plan_json))
    slot = SlotRuntime(
        slot_id=slot_id,
        slot_rank=slot_rank,
        comm=slot_comm,
        config=config,
        library=str(Path(args.library).resolve()),
        scheme_plan=scheme_plan,
    )

    connection: Connection | None = None
    connected = True
    if world.rank == 0:
        try:
            connection = Client(
                (args.host, int(args.port)),
                authkey=base64.urlsafe_b64decode(args.authkey.encode("ascii")),
            )
        except BaseException:
            connected = False
    connected = world.bcast(connected, root=0)
    if not connected:
        return 2

    try:
        if world.rank == 0:
            assert connection is not None
            if not _send(
                connection,
                {
                    "status": "ok",
                    "result": {
                        "event": "ready",
                        "runtime": "model-pool",
                        "world_size": int(world.size),
                        "ranks_per_model": int(args.ranks_per_model),
                        "model_slots": int(args.model_slots),
                        "mpi_launch_count": 1,
                        "runtime_options": options.describe(),
                    },
                },
            ):
                return 3

        while True:
            abandoned = False
            if world.rank == 0:
                assert connection is not None
                try:
                    request = connection.recv()
                except (EOFError, OSError):
                    request = {"op": "close_pool"}
                    abandoned = True
            else:
                request = None
            request = world.bcast(request, root=0)
            abandoned = world.bcast(abandoned, root=0)
            operation = str(request.get("op"))

            if operation == "create_model":
                response = _create_model(world, slot, request)
            elif operation == "model_command":
                response = _run_model_commands(world, slot, (request,))
                if world.rank == 0 and response and response["status"] == "ok":
                    response["result"] = response["result"][str(request["slot"])]
            elif operation == "model_commands":
                response = _run_model_commands(world, slot, request["commands"])
            elif operation == "fork_model":
                response = _fork_models(world, slot, request)
            elif operation == "close_model":
                response = _close_model(world, slot, request)
            elif operation == "status":
                statuses = _all_slot_statuses(world, slot)
                response = (
                    {
                        "status": "ok",
                        "result": {
                            "world_size": int(world.size),
                            "ranks_per_model": int(args.ranks_per_model),
                            "model_slots": int(args.model_slots),
                            "mpi_launch_count": 1,
                            "slots": statuses,
                        },
                    }
                    if world.rank == 0
                    else None
                )
            elif operation == "close_pool":
                failure = None
                try:
                    slot.release()
                except BaseException:
                    failure = _traceback()
                failures = _world_failures(world, failure)
                response = (
                    (
                        {"status": "error", "error": "\n".join(failures)}
                        if failures
                        else {"status": "ok", "result": {"closed": True}}
                    )
                    if world.rank == 0
                    else None
                )
            else:
                response = (
                    {
                        "status": "error",
                        "error": f"unknown pooled worker operation: {operation!r}",
                    }
                    if world.rank == 0
                    else None
                )

            delivered = True
            if world.rank == 0 and not abandoned:
                assert connection is not None and response is not None
                delivered = _send(connection, response)
            delivered = world.bcast(delivered, root=0)
            if operation == "close_pool" or not delivered:
                break
    finally:
        if slot.driver is not None:
            try:
                slot.release()
            except BaseException:
                traceback.print_exc()
        slot_comm.Free()
        if connection is not None:
            connection.close()
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--ranks-per-model", required=True, type=int)
    parser.add_argument("--model-slots", required=True, type=int)
    parser.add_argument("--config", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--timestep-seconds", required=True, type=int)
    parser.add_argument("--physics-profile", required=True)
    parser.add_argument("--scheme-plan-json", required=True)


def command_pool_worker(args: argparse.Namespace) -> int:
    return serve_pool(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pycam-sima-pool-worker")
    add_arguments(parser)
    return serve_pool(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
