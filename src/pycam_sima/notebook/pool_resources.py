"""Resource planning for a dynamically sized persistent model pool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping

import numpy as np

from ..model.config import ModelConfig
from ..model.contracts import default_contracts
from ..model.grid import dimensions_for_rank


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MEMORY = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*([kmgtpe]?i?b?)?\s*$",
    re.IGNORECASE,
)
_MEMORY_FACTORS = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
    "p": 1024**5,
    "pb": 1024**5,
    "pib": 1024**5,
    "e": 1024**6,
    "eb": 1024**6,
    "eib": 1024**6,
}


def parse_memory(value: int | str) -> int:
    """Return a byte count from an integer or PBS-style memory string."""

    if isinstance(value, bool):
        raise TypeError("memory must be a byte count or memory string")
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("memory must be positive")
        return value
    match = _MEMORY.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid memory value {value!r}")
    amount, suffix = match.groups()
    result = int(float(amount) * _MEMORY_FACTORS[(suffix or "").lower()])
    if result <= 0:
        raise ValueError("memory must be positive")
    return result


def format_memory(value: int) -> str:
    """Format a byte count without losing information."""

    for suffix, factor in (
        ("PiB", 1024**5),
        ("TiB", 1024**4),
        ("GiB", 1024**3),
        ("MiB", 1024**2),
        ("KiB", 1024),
    ):
        if value >= factor and value % factor == 0:
            return f"{value // factor}{suffix}"
    return f"{value}B"


def format_pbs_memory(value: int) -> str:
    """Format bytes using memory suffixes accepted by Derecho PBS."""

    gib = 1024**3
    if value % gib == 0:
        return f"{value // gib}GB"
    mib = 1024**2
    return f"{math.ceil(value / mib)}MB"


def _parse_dynamic_budget(value: int | str) -> int:
    if value == 0:
        return 0
    return parse_memory(value)


@dataclass(frozen=True, slots=True)
class PoolRequest:
    """User-requested shape of one persistent MPI model pool."""

    name: str
    max_concurrent_models: int | None = None
    ranks_per_model: int | str | None = None
    memory_per_model: int | str = "auto"
    placement: str = "auto"
    dynamic_field_budget: int | str = "auto"

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "pool name may contain only letters, digits, dot, dash, "
                "and underscore"
            )
        if (
            self.max_concurrent_models is not None
            and (
                isinstance(self.max_concurrent_models, bool)
                or self.max_concurrent_models <= 0
            )
        ):
            raise ValueError("max_concurrent_models must be positive")
        if self.ranks_per_model not in {None, "auto"}:
            if (
                isinstance(self.ranks_per_model, bool)
                or not isinstance(self.ranks_per_model, int)
                or self.ranks_per_model <= 0
            ):
                raise ValueError(
                    "ranks_per_model must be None, 'auto', or a positive integer"
                )
        if self.memory_per_model != "auto":
            parse_memory(self.memory_per_model)
        if self.dynamic_field_budget != "auto":
            _parse_dynamic_budget(self.dynamic_field_budget)
        if self.placement not in {"auto", "compact", "scatter"}:
            raise ValueError(
                "placement must be 'auto', 'compact', or 'scatter'"
            )


@dataclass(frozen=True, slots=True)
class ModelSlotStatus:
    """Small serializable description of one model slot."""

    slot_id: int
    state: str
    model_name: str | None
    ranks: tuple[int, ...]
    state_bytes: int = 0

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("slot_id cannot be negative")
        if self.state not in {
            "idle",
            "initializing",
            "ready",
            "running",
            "failed",
        }:
            raise ValueError(f"invalid model slot state {self.state!r}")
        if not self.ranks:
            raise ValueError("a model slot must contain at least one rank")
        if self.state_bytes < 0:
            raise ValueError("state_bytes cannot be negative")

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    """Resolved, executable resource layout for one MPI pool."""

    available_nodes: int
    available_cpus: int
    available_memory_bytes: int
    ranks_per_model: int
    model_slots: int
    world_size: int
    slot_placements: tuple[tuple[int, ...], ...]
    estimated_model_bytes: int
    memory_per_model_bytes: int
    reserve_bytes: int
    cpus_per_node: int
    memory_per_node_bytes: int
    threads_per_rank: int
    static_state_bytes: int
    dynamic_field_budget_bytes: int
    placement: str
    resource_source: str

    def describe(self) -> dict[str, Any]:
        result = asdict(self)
        result["available_memory"] = format_memory(
            self.available_memory_bytes
        )
        result["estimated_model_memory"] = format_memory(
            self.estimated_model_bytes
        )
        result["memory_per_model"] = format_memory(
            self.memory_per_model_bytes
        )
        result["reserve_memory"] = format_memory(self.reserve_bytes)
        result["pbs_select"] = self.pbs_select
        return result

    @property
    def pbs_select(self) -> str:
        ranks_per_node = max(
            1, self.cpus_per_node // self.threads_per_rank
        )
        cpu_nodes = math.ceil(self.world_size / ranks_per_node)
        required_memory = (
            self.model_slots * self.memory_per_model_bytes
            + self.reserve_bytes
        )
        memory_nodes = math.ceil(
            required_memory / self.memory_per_node_bytes
        )
        nodes = max(1, cpu_nodes, memory_nodes)
        return (
            f"select={nodes}:ncpus={self.cpus_per_node}:"
            f"mpiprocs={ranks_per_node}:"
            f"ompthreads={self.threads_per_rank}:"
            f"mem={format_pbs_memory(self.memory_per_node_bytes)}"
        )


@dataclass(frozen=True, slots=True)
class _AvailableResources:
    nodes: int
    cpus: int
    memory_bytes: int
    cpus_per_node: int
    memory_per_node_bytes: int
    source: str


def _local_resources() -> _AvailableResources:
    cpus = os.cpu_count() or 1
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    pages = int(os.sysconf("SC_PHYS_PAGES"))
    memory = page_size * pages
    return _AvailableResources(1, cpus, memory, cpus, memory, "local")


def _logical_qstat_lines(output: str) -> dict[str, str]:
    records: dict[str, str] = {}
    current: str | None = None
    for raw in output.splitlines():
        line = raw.rstrip()
        assignment = re.match(
            r"^\s*([A-Za-z][A-Za-z0-9_.]*)\s+=\s*(.*)$",
            line,
        )
        if assignment is not None:
            current, value = assignment.groups()
            records[current] = value.strip()
        elif current is not None and line[:1].isspace():
            records[current] += line.strip()
    return records


def _parse_select(value: str) -> tuple[int, int, int] | None:
    nodes = cpus = memory = 0
    for chunk in value.split("+"):
        parts = chunk.strip().split(":")
        if not parts:
            continue
        count = int(parts.pop(0)) if parts[0].isdigit() else 1
        attributes = dict(
            item.split("=", 1)
            for item in parts
            if "=" in item
        )
        if "ncpus" not in attributes or "mem" not in attributes:
            return None
        nodes += count
        cpus += count * int(attributes["ncpus"])
        memory += count * parse_memory(attributes["mem"])
    if nodes <= 0:
        return None
    return nodes, cpus, memory


def discover_pbs_resources(
    *,
    environ: Mapping[str, str] | None = None,
    qstat_runner: Callable[[str], Any] | None = None,
) -> _AvailableResources | None:
    """Discover the active PBS allocation, returning ``None`` outside PBS."""

    environment = os.environ if environ is None else environ
    job_id = environment.get("PBS_JOBID")
    nodefile = environment.get("PBS_NODEFILE")
    host_slots: dict[str, int] = {}
    if nodefile and Path(nodefile).is_file():
        for line in Path(nodefile).read_text(encoding="utf-8").splitlines():
            host = line.strip()
            if host:
                host_slots[host] = host_slots.get(host, 0) + 1
    if not job_id and not host_slots:
        return None

    output = ""
    if job_id:
        if qstat_runner is None:
            result = subprocess.run(
                ("qstat", "-f", job_id),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        else:
            result = qstat_runner(job_id)
        output = result if isinstance(result, str) else str(result.stdout)
    fields = _logical_qstat_lines(output)
    selected = _parse_select(fields.get("Resource_List.select", ""))

    if selected is None:
        if not host_slots:
            raise RuntimeError(
                "PBS allocation does not expose Resource_List.select or "
                "a readable PBS_NODEFILE"
            )
        local = _local_resources()
        nodes = len(host_slots)
        cpus_per_node = max(host_slots.values())
        cpus = sum(host_slots.values())
        memory_per_node = local.memory_bytes
        memory = nodes * memory_per_node
    else:
        nodes, cpus, memory = selected
        cpus_per_node = max(
            host_slots.values(), default=math.ceil(cpus / nodes)
        )
        memory_per_node = memory // nodes
    return _AvailableResources(
        nodes,
        cpus,
        memory,
        cpus_per_node,
        memory_per_node,
        "pbs",
    )


def estimate_state_pool_bytes(
    config: ModelConfig,
    ranks_per_model: int,
    *,
    dynamic_field_budget: int | str = "auto",
) -> tuple[int, int, int]:
    """Estimate total canonical NumPy storage across all model ranks."""

    element_count = 6 * config.ne * config.ne
    if ranks_per_model > element_count:
        raise ValueError(
            f"ranks_per_model={ranks_per_model} exceeds the "
            f"{element_count} available SFC elements"
        )
    static_bytes = 0
    contracts = default_contracts()
    for rank in range(ranks_per_model):
        dimensions = dimensions_for_rank(
            rank,
            ranks_per_model,
            pver=config.pver,
            np_value=config.np,
            fv_nphys=config.fv_nphys,
            constituent_count=config.constituent_count,
            extra_dimensions=config.dimension_overrides,
        )
        for contract in contracts:
            item_count = math.prod(contract.shape(dimensions))
            static_bytes += item_count * np.dtype(contract.dtype).itemsize
    if dynamic_field_budget == "auto":
        dynamic_bytes = math.ceil(static_bytes * 0.10)
    else:
        dynamic_bytes = _parse_dynamic_budget(dynamic_field_budget)
    return static_bytes, dynamic_bytes, static_bytes + dynamic_bytes


def _resolve_available_resources(
    *,
    available_nodes: int | None,
    cpus_per_node: int | None,
    memory_per_node: int | str | None,
    environ: Mapping[str, str] | None,
    qstat_runner: Callable[[str], Any] | None,
) -> _AvailableResources:
    discovered = discover_pbs_resources(
        environ=environ,
        qstat_runner=qstat_runner,
    )
    baseline = discovered or _local_resources()
    nodes = baseline.nodes if available_nodes is None else int(available_nodes)
    node_cpus = (
        baseline.cpus_per_node
        if cpus_per_node is None
        else int(cpus_per_node)
    )
    node_memory = (
        baseline.memory_per_node_bytes
        if memory_per_node is None
        else parse_memory(memory_per_node)
    )
    if nodes <= 0 or node_cpus <= 0:
        raise ValueError("available_nodes and cpus_per_node must be positive")
    source = baseline.source
    if any(
        item is not None
        for item in (available_nodes, cpus_per_node, memory_per_node)
    ):
        source = "override"
    return _AvailableResources(
        nodes,
        nodes * node_cpus,
        nodes * node_memory,
        node_cpus,
        node_memory,
        source,
    )


def plan_pool_resources(
    config: ModelConfig,
    *,
    max_concurrent_models: int | None = None,
    ranks_per_model: int | str | None = None,
    memory_per_model: int | str = "auto",
    placement: str = "auto",
    dynamic_field_budget: int | str = "auto",
    available_nodes: int | None = None,
    cpus_per_node: int | None = None,
    memory_per_node: int | str | None = None,
    reserve_fraction: float = 0.15,
    environ: Mapping[str, str] | None = None,
    qstat_runner: Callable[[str], Any] | None = None,
) -> ResourcePlan:
    """Resolve a pool request against PBS, explicit, or local resources."""

    request = PoolRequest(
        "resource-plan",
        max_concurrent_models=max_concurrent_models,
        ranks_per_model=ranks_per_model,
        memory_per_model=memory_per_model,
        placement=placement,
        dynamic_field_budget=dynamic_field_budget,
    )
    if not 0.0 <= reserve_fraction < 1.0:
        raise ValueError("reserve_fraction must be in [0, 1)")
    available = _resolve_available_resources(
        available_nodes=available_nodes,
        cpus_per_node=cpus_per_node,
        memory_per_node=memory_per_node,
        environ=environ,
        qstat_runner=qstat_runner,
    )
    threads = config.threads_per_rank
    usable_rank_count = available.cpus // threads
    element_count = 6 * config.ne * config.ne
    if request.ranks_per_model == "auto":
        concurrent = request.max_concurrent_models or 1
        ranks = min(element_count, usable_rank_count // concurrent)
    elif request.ranks_per_model is None:
        ranks = config.mpi_size
    else:
        ranks = int(request.ranks_per_model)
    if ranks <= 0:
        raise ValueError("available CPU resources cannot host one model rank")
    if ranks > element_count:
        raise ValueError(
            f"ranks_per_model={ranks} exceeds the {element_count} "
            "available SFC elements"
        )

    static_bytes, dynamic_bytes, estimated_bytes = estimate_state_pool_bytes(
        config,
        ranks,
        dynamic_field_budget=request.dynamic_field_budget,
    )
    if request.memory_per_model == "auto":
        effective_model_bytes = estimated_bytes
    else:
        effective_model_bytes = parse_memory(request.memory_per_model)
        if effective_model_bytes < estimated_bytes:
            raise ValueError(
                f"memory_per_model={format_memory(effective_model_bytes)} "
                f"is below the estimated StatePool requirement "
                f"{format_memory(estimated_bytes)}"
            )

    reserve_bytes = math.ceil(
        available.memory_bytes * reserve_fraction
    )
    cpu_capacity = usable_rank_count // ranks
    memory_capacity = (
        available.memory_bytes - reserve_bytes
    ) // effective_model_bytes
    capacity = min(cpu_capacity, memory_capacity)
    if capacity < 1:
        raise ValueError(
            "available resources cannot host one model: "
            f"need {ranks * threads} CPUs and "
            f"{format_memory(effective_model_bytes)} model memory; "
            f"have {available.cpus} CPUs and "
            f"{format_memory(available.memory_bytes - reserve_bytes)} "
            "usable memory"
        )
    if (
        request.max_concurrent_models is not None
        and request.max_concurrent_models > capacity
    ):
        requested_cpus = request.max_concurrent_models * ranks * threads
        requested_memory = (
            request.max_concurrent_models * effective_model_bytes
            + reserve_bytes
        )
        raise ValueError(
            f"requested {request.max_concurrent_models} concurrent models, "
            f"but resources fit {capacity}; request needs "
            f"{requested_cpus} CPUs and "
            f"{format_memory(requested_memory)}, allocation has "
            f"{available.cpus} CPUs and "
            f"{format_memory(available.memory_bytes)}"
        )
    slots = request.max_concurrent_models or capacity
    world_size = slots * ranks
    slot_placements = tuple(
        tuple(range(slot * ranks, (slot + 1) * ranks))
        for slot in range(slots)
    )
    return ResourcePlan(
        available_nodes=available.nodes,
        available_cpus=available.cpus,
        available_memory_bytes=available.memory_bytes,
        ranks_per_model=ranks,
        model_slots=slots,
        world_size=world_size,
        slot_placements=slot_placements,
        estimated_model_bytes=estimated_bytes,
        memory_per_model_bytes=effective_model_bytes,
        reserve_bytes=reserve_bytes,
        cpus_per_node=available.cpus_per_node,
        memory_per_node_bytes=available.memory_per_node_bytes,
        threads_per_rank=threads,
        static_state_bytes=static_bytes,
        dynamic_field_budget_bytes=dynamic_bytes,
        placement=request.placement,
        resource_source=available.source,
    )


class PoolResourcePlanner:
    """Reusable planner bound to one model configuration."""

    def __init__(self, config: ModelConfig) -> None:
        config.validate()
        self.config = config

    def plan(self, **options: Any) -> ResourcePlan:
        return plan_pool_resources(self.config, **options)


__all__ = [
    "ModelSlotStatus",
    "PoolRequest",
    "PoolResourcePlanner",
    "ResourcePlan",
    "discover_pbs_resources",
    "estimate_state_pool_bytes",
    "format_memory",
    "format_pbs_memory",
    "parse_memory",
    "plan_pool_resources",
]
