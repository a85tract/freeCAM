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
