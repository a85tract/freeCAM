"""The kernel-slot notebook names only slots, kernels and functions that exist."""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT / "examples/replace_kernel.ipynb"


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_every_import_resolves_and_no_machine_path_is_written() -> None:
    code = _code()
    for module_name, names in re.findall(r"^from (freecam[\w.]*) import ([\w, ]+)$", code, re.MULTILINE):
        module = importlib.import_module(module_name)
        for name in names.split(","):
            if hasattr(module, name.strip()):
                continue
            importlib.import_module(f"{module_name}.{name.strip()}")   # a submodule, such as freecam.site
    assert not re.search(r"/glade/|/tmp/|desched|\.hpc\.ucar\.edu", NOTEBOOK.read_text())
    notebook = json.loads(NOTEBOOK.read_text())
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_the_kernels_named_belong_to_the_stages_used() -> None:
    from freecam.physics.pausable import DeepConvection, VerticalDiffusion
    from freecam.pi_cam.hooks import load_hooks

    code = _code()
    used = {stage: set(re.findall(rf"{stage}\.kernels\['(\w+)'\]", code)) for stage in ("vdiff", "deep")}
    assert used["vdiff"] <= set(VerticalDiffusion.SWAPPABLE) and used["vdiff"]
    assert used["deep"] <= set(DeepConvection.SWAPPABLE) and used["deep"]
    # the compiled plugin stands at a hook that takes a model
    hooked = re.findall(r"compile_kernel\('(\w+)'", code)
    table = load_hooks()
    assert hooked and all(table.hook(name).takes_model for name in hooked)
    # the example function it compiles exists
    assert (PROJECT / "examples/plugins/numba_kernels/cldfrc_fice.py").is_file()
    source = (PROJECT / "examples/plugins/numba_kernels/cldfrc_fice.py").read_text()
    assert "def cldfrc_fice(" in source and "def cldfrc_fice_reference(" in source
