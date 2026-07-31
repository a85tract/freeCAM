from __future__ import annotations

from pathlib import Path

import pytest

from pycam_sima import (
    ModelConfig,
    ModelSlotStatus,
    PoolRequest,
    PoolResourcePlanner,
    ResourcePlan,
    plan_pool_resources,
)
from pycam_sima.notebook.pool_resources import (
    discover_pbs_resources,
    estimate_state_pool_bytes,
    format_memory,
    format_pbs_memory,
    parse_memory,
)


def test_memory_values_use_binary_pbs_units() -> None:
    assert parse_memory("80GB") == 80 * 1024**3
    assert parse_memory("1.5 GiB") == int(1.5 * 1024**3)
    assert format_memory(80 * 1024**3) == "80GiB"
    assert format_pbs_memory(80 * 1024**3) == "80GB"
    assert format_pbs_memory(1024**3 + 1) == "1025MB"
    with pytest.raises(ValueError, match="invalid memory"):
        parse_memory("lots")


def test_pool_request_and_slot_status_validate_public_values() -> None:
    request = PoolRequest(
        "science-pool",
        max_concurrent_models=3,
        ranks_per_model="auto",
        placement="scatter",
        dynamic_field_budget=0,
    )
    assert request.max_concurrent_models == 3
    status = ModelSlotStatus(2, "ready", "warm", (8, 9, 10, 11), 4096)
    assert status.describe()["model_name"] == "warm"
    with pytest.raises(ValueError, match="pool name"):
        PoolRequest("not a name")
    with pytest.raises(ValueError, match="invalid model slot state"):
        ModelSlotStatus(0, "closed", None, (0,))


def test_pbs_resources_come_from_nodefile_and_qstat_select(
    tmp_path: Path,
) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text(
        "dec001\ndec001\ndec002\ndec002\n",
        encoding="utf-8",
    )
    output = """
Job Id: 123.server
    Resource_List.select = 2:ncpus=32:mpiprocs=24:
        ompthreads=1:mem=80GB
"""
    resources = discover_pbs_resources(
        environ={
            "PBS_JOBID": "123.server",
            "PBS_NODEFILE": str(nodefile),
        },
        qstat_runner=lambda job_id: output,
    )
    assert resources is not None
    assert resources.nodes == 2
    assert resources.cpus == 64
    assert resources.memory_bytes == 160 * 1024**3
    assert resources.cpus_per_node == 2
    assert resources.source == "pbs"


def test_estimate_uses_runtime_rank_count_and_dynamic_budget() -> None:
    config = ModelConfig(mpi_size=7)
    static, dynamic, total = estimate_state_pool_bytes(
        config,
        7,
        dynamic_field_budget=0,
    )
    assert static > 0
    assert dynamic == 0
    assert total == static
    _, automatic, automatic_total = estimate_state_pool_bytes(config, 7)
    assert automatic == pytest.approx(static * 0.10, abs=1)
    assert automatic_total == static + automatic


def test_resource_plan_uses_config_rank_count_without_fixed_values() -> None:
    config = ModelConfig(mpi_size=7, threads_per_rank=2)
    plan = plan_pool_resources(
        config,
        max_concurrent_models=3,
        available_nodes=2,
        cpus_per_node=32,
        memory_per_node="80GB",
        environ={},
    )
    assert isinstance(plan, ResourcePlan)
    assert plan.ranks_per_model == 7
    assert plan.model_slots == 3
    assert plan.world_size == 21
    assert plan.slot_placements == (
        tuple(range(0, 7)),
        tuple(range(7, 14)),
        tuple(range(14, 21)),
    )
    assert plan.reserve_bytes == int(160 * 1024**3 * 0.15)
    assert plan.describe()["resource_source"] == "override"
    assert plan.pbs_select.startswith("select=")
    assert plan.pbs_select.endswith("mem=80GB")
    assert "GiB" not in plan.pbs_select


def test_resource_plan_defaults_to_one_slot_and_budgets_retained_state() -> None:
    config = ModelConfig(mpi_size=7)
    plan = plan_pool_resources(
        config,
        available_nodes=4,
        cpus_per_node=32,
        memory_per_node="80GB",
        retained_snapshots=1,
        environ={},
    )

    assert plan.model_slots == 1
    assert plan.world_size == 7
    assert plan.retained_snapshots == 1
    assert plan.retained_snapshot_budget_bytes == plan.estimated_model_bytes
    assert plan.describe()["retained_snapshot_memory"] == format_memory(
        plan.estimated_model_bytes
    )

    without_snapshot = plan_pool_resources(
        config,
        available_nodes=4,
        cpus_per_node=32,
        memory_per_node="80GB",
        retained_snapshots=0,
        environ={},
    )
    assert without_snapshot.retained_snapshot_budget_bytes == 0


def test_complete_resource_override_does_not_query_pbs() -> None:
    def fail_qstat(_job_id: str) -> str:
        raise AssertionError("qstat must not run for a complete override")

    plan = plan_pool_resources(
        ModelConfig(mpi_size=24),
        max_concurrent_models=4,
        available_nodes=1,
        cpus_per_node=128,
        memory_per_node="80GB",
        environ={"PBS_JOBID": "transient.server"},
        qstat_runner=fail_qstat,
    )

    assert plan.world_size == 96
    assert plan.describe()["resource_source"] == "override"


def test_auto_ranks_respect_requested_concurrency_and_sfc_elements() -> None:
    config = ModelConfig(mpi_size=7)
    plan = PoolResourcePlanner(config).plan(
        max_concurrent_models=3,
        ranks_per_model="auto",
        available_nodes=2,
        cpus_per_node=12,
        memory_per_node="80GB",
        dynamic_field_budget=0,
        environ={},
    )
    assert plan.ranks_per_model == 8
    assert plan.world_size == 24


def test_requested_concurrency_fails_with_resource_requirements() -> None:
    config = ModelConfig(mpi_size=8)
    with pytest.raises(ValueError, match="requested 5 concurrent models"):
        plan_pool_resources(
            config,
            max_concurrent_models=5,
            available_nodes=1,
            cpus_per_node=32,
            memory_per_node="80GB",
            environ={},
        )


def test_explicit_model_memory_cannot_understate_state_pool() -> None:
    with pytest.raises(ValueError, match="below the estimated StatePool"):
        plan_pool_resources(
            ModelConfig(mpi_size=4),
            memory_per_model=1,
            available_nodes=1,
            cpus_per_node=8,
            memory_per_node="80GB",
            environ={},
        )


def test_retained_snapshot_count_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="retained_snapshots"):
        PoolRequest("invalid-retain", retained_snapshots=-1)
