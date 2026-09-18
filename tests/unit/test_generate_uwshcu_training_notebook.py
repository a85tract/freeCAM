"""The compute_uwshcu_inv training-data notebook and script name what exists and write no machine path."""
from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT / "examples/generate_compute_uwshcu_inv_training_data.ipynb"
SCRIPT = PROJECT / "examples/generate_compute_uwshcu_inv_dataset.py"


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_the_notebook_resolves_its_imports_and_writes_no_machine_path() -> None:
    code = _code()
    for module_name, names in re.findall(r"^from (freecam[\w.]*) import ([\w, ]+)$", code, re.MULTILINE):
        module = importlib.import_module(module_name)
        for name in names.split(","):
            if not hasattr(module, name.strip()):
                importlib.import_module(f"{module_name}.{name.strip()}")
    text = NOTEBOOK.read_text()
    assert not re.search(r"/glade/|/tmp/|desched|\.hpc\.ucar\.edu", text)
    notebook = json.loads(text)
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    # the notebook drives the script that exists, on the function and kernel that exist
    assert "examples/generate_compute_uwshcu_inv_dataset.py" in code and SCRIPT.is_file()
    assert "FUNCTION, KERNEL = 'uwshcu', 'compute_uwshcu_inv'" in code
    assert (PROJECT / "native/pi_cam/functions/uwshcu.yaml").is_file()


def test_the_script_draws_every_input_and_keeps_the_column_consistent() -> None:
    import importlib.util

    import numpy as np

    spec_ = importlib.util.spec_from_file_location("generate_uwshcu", SCRIPT)
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    from freecam.physics.spec import load_function_spec

    spec = load_function_spec("uwshcu")
    names = [item.name for item in spec.user_arguments]
    assert len(names) == 20 and set(module.WATER) <= set(names) and set(module.HYDROSTATIC) <= set(names)
    # every input is either perturbed, rebuilt by a rule, taken whole from the anchor, or drawn (dt)
    covered = set(module.RELATIVE) | set(module.ABSOLUTE) | set(module.HYDROSTATIC) | {"s0_inv", "tr0_inv", "concldfrct_inv", "cush", "dt"}
    assert covered == set(names)
    # the rules on a toy column: static energy follows the temperature, water constituents equal the water,
    # isotopes keep their ratio, a seeded cloud gets droplets, cush stays -1 without cumulus
    rng = np.random.default_rng(0)
    lev = 30
    anchor = {"t0_inv": np.full(lev, 260.0), "s0_inv": np.full(lev, 1004.64 * 260.0 + 9.80616 * 1000.0 + 500.0),
              "qv0_inv": np.full(lev, 2e-3), "ql0_inv": np.r_[np.zeros(lev - 1), 1e-4], "qi0_inv": np.zeros(lev),
              "tr0_inv": np.zeros((lev, 57)), "concldfrct_inv": np.full(lev, 0.3), "cush": np.asarray(-1.0)}
    anchor["tr0_inv"][:, 0] = anchor["qv0_inv"]; anchor["tr0_inv"][:, 1] = anchor["ql0_inv"]
    for vapour, liquid, ice in module.ISOTOPE_SETS:
        anchor["tr0_inv"][:, vapour] = 0.9 * anchor["qv0_inv"]; anchor["tr0_inv"][:, liquid] = 1.1 * anchor["ql0_inv"]
    anchor["tr0_inv"][:, 3] = 1e8 * (anchor["ql0_inv"] > 0)
    drawn = {"t0_inv": anchor["t0_inv"] + 1.0, "qv0_inv": anchor["qv0_inv"] * 1.5,
             "ql0_inv": np.r_[np.zeros(lev - 2), 2e-6, 2e-4], "qi0_inv": np.r_[np.zeros(lev - 1), 1e-6], "cldfrct_inv": np.full(lev, 0.2)}
    assert np.allclose(module.static_energy(rng, anchor, drawn), anchor["s0_inv"] + 1004.64)
    tracers = module.tracers(rng, anchor, drawn)
    assert np.array_equal(tracers[:, 0], drawn["qv0_inv"]) and np.array_equal(tracers[:, 1], drawn["ql0_inv"]) and np.array_equal(tracers[:, 2], drawn["qi0_inv"])
    assert np.allclose(tracers[:, 9], 0.9 * drawn["qv0_inv"])                        # vapour isotope at the anchor's ratio
    assert np.isclose(tracers[-1, 10], 1.1 * drawn["ql0_inv"][-1])                    # liquid isotope where the anchor had liquid
    assert np.isclose(tracers[-2, 10], 0.9 * drawn["ql0_inv"][-2])                    # a seeded level takes the vapour ratio
    assert np.isclose(tracers[-1, 3], 1e8 * 2.0) and tracers[-2, 3] > 0 and tracers[-1, 11] > 0   # numbers scale, seeds get droplets
    assert np.all(module.concld(rng, anchor, drawn) <= drawn["cldfrct_inv"])
    assert float(module.cush(rng, anchor, drawn)) == -1.0
    assert float(module.cush(rng, {"cush": np.asarray(1500.0)}, drawn)) > 1.0
