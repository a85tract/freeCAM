from pathlib import Path

import pycam_sima
from pycam_sima import CAMDriver, NotebookSession
from pycam_sima.model import CAMDriver as ModelDriver
from pycam_sima.notebook import NotebookSession as NotebookController


def test_top_level_package_contains_only_public_entrypoints() -> None:
    package = Path(pycam_sima.__file__).resolve().parent
    modules = {path.stem for path in package.glob("*.py")}
    assert modules == {"__init__", "cli"}
    packages = {
        path.name
        for path in package.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }
    assert packages == {"core", "model", "notebook"}


def test_root_api_delegates_to_responsibility_packages() -> None:
    assert CAMDriver is ModelDriver
    assert NotebookSession is NotebookController


def test_maintained_pbs_jobs_write_to_the_shared_log_directory() -> None:
    repository = Path(__file__).resolve().parents[2]
    output_directive = "#PBS -o /glade/work/ruitong/pycam-sima/logs/"

    jobs = tuple((repository / "jobs").glob("*.pbs"))
    assert jobs
    for job in jobs:
        assert output_directive in job.read_text().splitlines()
    assert (repository / "logs" / ".gitkeep").is_file()
