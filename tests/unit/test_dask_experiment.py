from pathlib import Path

from distributed import Client

from pycam_sima import BranchSpec, CheckpointBundle, FieldEdit
from pycam_sima.notebook.dask import (
    DaskExperimentClient,
    DaskRunResult,
)


def _fake_segment(request, parent):
    value = 10 if parent is None else int(parent.stats["value"])
    for edit in request.branch.field_edits:
        if edit.operation == "add":
            value += int(edit.value)
    bundle = CheckpointBundle(
        (("manifest.json", f'{{"value": {value}}}'.encode()),)
    )
    return DaskRunResult(
        branch=request.branch.name,
        parent_branch=None if parent is None else parent.branch,
        run_dir=f"/run/{request.branch.name}",
        history_dir=f"/history/{request.branch.name}",
        checkpoint_dir=f"/checkpoint/{request.branch.name}",
        log_path=f"/logs/{request.branch.name}.log",
        pbs_job_id=None,
        stats={"value": value, "step": request.branch.steps},
        snapshot=bundle,
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
        results = experiments.gather(branches)
        base_result = base.result()

    assert base_result.stats["value"] == 10
    assert results["plus-one"].stats["value"] == 11
    assert results["plus-two"].stats["value"] == 12
    assert summaries["plus-one"]["step"] == 1
    assert summaries["plus-one"]["run_dir"] == "/run/plus-one"
    assert summaries["plus-one"]["history_dir"] == "/history/plus-one"
    assert summaries["plus-one"]["checkpoint_dir"] == "/checkpoint/plus-one"
    assert results["plus-one"].parent_branch == "base"
    assert results["plus-two"].parent_branch == "base"
    assert results["plus-one"].snapshot is not results["plus-two"].snapshot
