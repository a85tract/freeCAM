import tomllib
from pathlib import Path

import freecam
from freecam import (
    CAMDriver,
    DaskExperimentClient,
    DeviceBuildError,
    DeviceContractError,
    DeviceRegistry,
    FortranDevice,
    NotebookSession,
    RunPhase,
    RunScheme,
    SegmentPlan,
)
from freecam.model import CAMDriver as ModelDriver
from freecam.notebook import (
    DaskExperimentClient as DaskController,
    NotebookSession as NotebookController,
)


def test_top_level_package_contains_only_public_entrypoints() -> None:
    package = Path(freecam.__file__).resolve().parent
    modules = {path.stem for path in package.glob("*.py")}
    assert modules == {"__init__", "cli"}
    packages = {
        path.name
        for path in package.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert packages == {"core", "model", "notebook", "pi_cam"}


def test_distribution_import_and_cli_share_the_freecam_name() -> None:
    repository = Path(__file__).resolve().parents[2]
    project = tomllib.loads((repository / "pyproject.toml").read_text())

    assert project["project"]["name"] == "freecam"
    assert project["project"]["scripts"] == {"freecam": "freecam.cli:main"}
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/freecam"
    ]
    assert not (repository / "src" / "pycam_sima").exists()


def test_root_api_delegates_to_responsibility_packages() -> None:
    assert CAMDriver is ModelDriver
    assert NotebookSession is NotebookController
    assert DaskExperimentClient is DaskController
    assert freecam.DeviceRegistry is DeviceRegistry
    assert freecam.DeviceBuildError is DeviceBuildError
    assert freecam.DeviceContractError is DeviceContractError
    assert freecam.FortranDevice is FortranDevice
    assert freecam.SegmentPlan is SegmentPlan
    assert freecam.RunPhase is RunPhase
    assert freecam.RunScheme is RunScheme


def test_maintained_pbs_jobs_write_to_the_shared_log_directory() -> None:
    repository = Path(__file__).resolve().parents[2]
    output_directive = "#PBS -o /glade/work/ruitong/freeCAM/logs/"

    jobs = tuple((repository / "jobs").glob("*.pbs"))
    assert jobs
    for job in jobs:
        assert output_directive in job.read_text().splitlines()
    assert (repository / "logs" / ".gitkeep").is_file()
