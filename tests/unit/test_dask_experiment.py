from io import BytesIO
from pathlib import Path

from distributed import Client
import numpy as np
import pytest

from pycam_sima import (
    BranchSpec,
    CheckpointBundle,
    FieldEdit,
    ObserveFields,
    RunPhase,
    RunScheme,
    SegmentPlan,
)
from pycam_sima.notebook.dask import (
    _allocation_launcher,
    DaskExperimentClient,
    DaskPBSOptions,
    DaskRunResult,
    run_allocation_segment,
)


def _fake_segment(request, parent):
    value = 10 if parent is None else int(parent.stats["value"])
    for action in request.plan.actions:
        if isinstance(action, FieldEdit) and action.operation == "add":
            value += int(action.value)
    stream = BytesIO()
    np.savez(
        stream,
        air_temperature=np.full((2, 2), value, dtype=np.float64, order="F"),
    )
    bundle = CheckpointBundle(
        (
            ("manifest.json", f'{{"value": {value}}}'.encode()),
            ("rank-000.npz", stream.getvalue()),
        )
    )
    action_trace = tuple(
        {"index": index, "type": type(action).__name__}
        for index, action in enumerate(request.plan.actions)
    )
    return DaskRunResult(
        branch=request.plan.name,
        parent_branch=None if parent is None else parent.branch,
        run_dir=f"/run/{request.plan.name}",
        history_dir=f"/history/{request.plan.name}",
        checkpoint_dir=f"/checkpoint/{request.plan.name}",
        log_path=f"/logs/{request.plan.name}.log",
        execution_mode=request.execution_mode,
        pbs_job_id=None,
        stats={"value": value, "step": request.plan.step_count},
        snapshot=bundle,
        action_trace=action_trace,
        segment_plan=request.plan.as_dict(),
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "config.yaml"
    config.touch()
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "atm_in").touch()
    library = tmp_path / "libkernels.so"
    library.touch()
    environment = tmp_path / "environment.sh"
    environment.touch()
    python_target = tmp_path / "python-target"
    python_target.touch()
    python = tmp_path / "python"
    python.symlink_to(python_target)
    return {
        "config": config,
        "initial_run_dir": initial,
        "run_root": tmp_path / "runs",
        "library": library,
        "environment_script": environment,
        "python_executable": python,
        "log_dir": tmp_path / "logs",
    }


def test_dask_future_fans_out_independent_branches(tmp_path: Path) -> None:
    with Client(
        processes=False,
        n_workers=2,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            task_runner=_fake_segment,
            **_inputs(tmp_path),
        )
        assert experiments.python_executable == tmp_path / "python"
        assert experiments.python_executable.is_symlink()
        base = experiments.submit_base(BranchSpec("base", steps=3))
        branches = experiments.fork(
            base,
            (
                BranchSpec(
                    "plus-one",
                    field_edits=(FieldEdit("air_temperature", "add", 1.0),),
                ),
                BranchSpec(
                    "plus-two",
                    field_edits=(FieldEdit("air_temperature", "add", 2.0),),
                ),
            ),
        )
        summaries = experiments.summaries(branches)
        temperature = experiments.field(
            branches["plus-one"], "T", rank=0
        ).result()
        results = experiments.gather(branches)
        base_result = base.result()

    assert base_result.stats["value"] == 10
    assert results["plus-one"].stats["value"] == 11
    assert results["plus-two"].stats["value"] == 12
    assert summaries["plus-one"]["step"] == 1
    assert summaries["plus-one"]["run_dir"] == "/run/plus-one"
    assert summaries["plus-one"]["history_dir"] == "/history/plus-one"
    assert summaries["plus-one"]["checkpoint_dir"] == "/checkpoint/plus-one"
    assert summaries["plus-one"]["execution_mode"] == "pbs"
    assert summaries["plus-one"]["action_count"] == 2
    assert np.array_equal(temperature, np.full((2, 2), 11.0))
    assert results["plus-one"].parent_branch == "base"
    assert results["plus-two"].parent_branch == "base"
    assert results["plus-one"].snapshot is not results["plus-two"].snapshot


def test_dask_pbs_rank_count_defaults_to_model_configuration(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["config"].write_text("mpi_size: 18\n")
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            task_runner=_fake_segment,
            **inputs,
        )
        assert experiments.pbs.ranks == 18

        with pytest.raises(ValueError, match="does not match"):
            DaskExperimentClient(
                client,
                task_runner=_fake_segment,
                pbs=DaskPBSOptions(ranks=24),
                **inputs,
            )


def test_pool_planning_outside_allocation_uses_pbs_model_resources(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            task_runner=_fake_segment,
            pbs=DaskPBSOptions(ranks=24, memory="80GB"),
            execution_mode="pbs",
            **_inputs(tmp_path),
        )
        plan = experiments.plan_pool(
            max_concurrent_models=4,
            ranks_per_model=None,
            environ={},
        )

    assert plan.available_nodes == 4
    assert plan.cpus_per_node == 24
    assert plan.model_slots == 4
    assert plan.world_size == 96
    assert plan.pbs_select == (
        "select=4:ncpus=24:mpiprocs=24:ompthreads=1:mem=80GB"
    )


def test_submit_plan_and_single_action_keep_parent_future(
    tmp_path: Path,
) -> None:
    with Client(
        processes=False,
        n_workers=1,
        threads_per_worker=1,
        dashboard_address=None,
    ) as client:
        experiments = DaskExperimentClient(
            client,
            task_runner=_fake_segment,
            **_inputs(tmp_path),
        )
        base = experiments.submit_base(BranchSpec("base", steps=0))
        batch = experiments.submit_plan(
            base,
            SegmentPlan(
                "batch",
                (
                    RunScheme(
                        "kessler", group="physics_before_coupler"
                    ),
                    ObserveFields(("air_temperature",)),
                ),
                unsafe=True,
            ),
        )
        single = experiments.submit_action(
            batch,
            name="single-phase",
            action=RunPhase("dynamics_to_physics"),
        )
        pythonic_plan = experiments.plan(
            "pythonic", experimental=True
        )
        pythonic_plan.physics.scheme(
            "kessler", group="before"
        ).run()
        pythonic_plan.observe("air_temperature")
        pythonic = experiments.submit_plan(base, pythonic_plan)
        batch_result, single_result, pythonic_result = client.gather(
            (batch, single, pythonic)
        )

    assert batch_result.parent_branch == "base"
    assert single_result.parent_branch == "batch"
    assert batch_result.segment_plan["unsafe"] is True
    assert single_result.segment_plan["actions"] == [
        {"type": "run_phase", "name": "dynamics_to_physics"}
    ]
    assert pythonic_result.segment_plan["actions"] == [
        {
            "type": "run_scheme",
            "name": "kessler",
            "group": "physics_before_coupler",
        },
        {
            "type": "observe_fields",
            "fields": ["air_temperature"],
            "statistics": ["min", "max", "mean"],
        },
    ]


class _SubmitClient:
    def submit(self, *args, **kwargs):
        return args, kwargs


def test_allocation_mode_selects_direct_mpi_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiments = DaskExperimentClient(
        _SubmitClient(),
        execution_mode="allocation",
        **_inputs(tmp_path),
    )
    request = experiments._request(BranchSpec("base", steps=0))

    assert experiments.task_runner is run_allocation_segment
    assert request.execution_mode == "allocation"
    monkeypatch.delenv("PBS_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="active PBS allocation"):
        run_allocation_segment(request, None)
    assert not (tmp_path / "runs/base").exists()


def test_allocation_launcher_uses_the_active_pbs_nodefile() -> None:
    assert _allocation_launcher({"PBS_NODEFILE": "/tmp/nodes"}, 24) == [
        "mpiexec",
        "-n",
        "24",
    ]


def test_execution_mode_rejects_unknown_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        DaskExperimentClient(
            _SubmitClient(),
            execution_mode="nested",  # type: ignore[arg-type]
            **_inputs(tmp_path),
        )
