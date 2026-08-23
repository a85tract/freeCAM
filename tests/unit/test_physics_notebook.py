"""The physics-function notebook executes, end to end, against the real image."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

PROJECT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT / "examples/physics_function.ipynb"
MANIFEST = PROJECT / "build/pi_cam_standalone/mmacro_pcond/manifest.json"
SNAPSHOT = PROJECT / "validation/pi_cam_mmacro_pcond_module_state.json"


def test_notebook_never_touches_the_driver_and_has_outputs() -> None:
    notebook = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    for forbidden in ("fc.Driver", "driver.initialize", "driver.cam.state", "driver.cam.workflow", "freecam.pi_cam", ".REPO"):
        assert forbidden not in code, forbidden
    assert "np.full(30" not in code
    executed = [cell for cell in notebook["cells"] if cell["cell_type"] == "code" and cell.get("outputs")]
    assert len(executed) >= 8, "the committed notebook carries its outputs"
    leaks = re.compile(r"/glade/|/tmp/|desched|\.hpc\.ucar\.edu")
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                continue
            assert output.get("output_type") != "error"
            assert not leaks.search(json.dumps(output)), "outputs must not carry machine paths"


@pytest.mark.skipif(
    not (MANIFEST.is_file() and SNAPSHOT.is_file()), reason="mmacro_pcond standalone image not built here"
)
def test_notebook_executes(tmp_path: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    notebook = nbformat.read(str(NOTEBOOK), as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    )
    client.execute()
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            assert output.get("output_type") != "error", f"{output.get('ename')}: {output.get('evalue')}"
    assert (tmp_path / "mmacro_pcond_training.nc").is_file()
