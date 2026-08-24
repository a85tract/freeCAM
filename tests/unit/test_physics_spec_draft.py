"""The spec drafter recovers what the inventory records and flags the rest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

PROJECT = Path(__file__).resolve().parents[2]
TOOL = PROJECT / "tools/draft_pi_cam_function_spec.py"
INVENTORY = PROJECT / "validation/pi_cam_kernel_inventory.json"


def _module():
    spec = importlib.util.spec_from_file_location("draft_spec", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_declared_names_ignores_commas_inside_dimensions() -> None:
    declared = _module().declared_names
    assert declared("tr0_inv(mix,mkx,ncnst)") == ["tr0_inv"]
    assert declared("a, b(1,2), c") == ["a", "b", "c"]
    assert declared(" cush(mix) ") == ["cush"]


def test_public_shape_drops_the_column_axis_and_flags_unknown_extents() -> None:
    module = _module()
    canonical = {"mix": "pcols", "mkx": "pver", "mkx+1": "pverp", "ncnst": "pcnst"}
    # Native names, not the public-axis aliases: this is what spec.py checks.
    assert module.public_shape(["mix", "mkx"], canonical) == ["pver"]
    assert module.public_shape(["mix", "mkx+1"], canonical) == ["pverp"]
    assert module.public_shape(["mix", "mkx", "ncnst"], canonical) == ["pver", "pcnst"]
    # An assumed-shape or explicitly bounded dimension carries no checkable extent.
    assert module.public_shape(["pcols", "0:pver"], {"pcols": "pcols"}) == ["REVIEW extent 0:pver"]
    assert module.public_shape([":", ":"], {}) == ["REVIEW extent :", "REVIEW extent :"]


def test_sequence_and_quoted_survive_yaml() -> None:
    module = _module()
    document = yaml.safe_load(
        f"shape: [{module.sequence(['pcols', '0:pver', ':'])}]\n"
        f"units: {module.quoted('#, kg/kg')}\n"
        f"description: {module.quoted('Time step : 2*delta_t')}\n"
        f"missing: {module.quoted('')}\n"
    )
    assert document["shape"] == ["pcols", "0:pver", ":"]
    assert document["units"] == "#, kg/kg"
    assert document["description"] == "Time step : 2*delta_t"
    assert document["missing"] == "REVIEW"


@pytest.mark.skipif(not INVENTORY.is_file(), reason="kernel inventory not present")
def test_uwshcu_draft_parses_and_carries_the_recorded_facts(tmp_path: Path) -> None:
    import subprocess

    out = tmp_path / "uwshcu.yaml"
    subprocess.run(
        ["python3", str(TOOL), "uwshcu::compute_uwshcu_inv",
         "--dimensions", "mix=pcols", "mkx=pver", "ncnst=pcnst", "--out", str(out)],
        check=True, capture_output=True, cwd=PROJECT,
    )
    draft = yaml.safe_load(out.read_text())
    assert draft["qualified_name"] == "uwshcu::compute_uwshcu_inv"
    assert len(draft["arguments"]) == 54
    by_name = {argument["name"]: argument for argument in draft["arguments"]}
    # Fortran order is preserved.
    assert [a["name"] for a in draft["arguments"]][:5] == ["mix", "mkx", "iend", "ncnst", "dt"]
    # Facts the inventory and the source already carry are filled in.
    assert by_name["p0_inv"]["native_shape"] == ["pcols", "pver"]
    assert by_name["p0_inv"]["public_shape"] == ["pver"]
    assert by_name["p0_inv"]["units"] == "Pa"
    assert by_name["tr0_inv"]["public_shape"] == ["pver", "pcnst"]
    assert by_name["umf_inv"]["role"] == "output" and by_name["umf_inv"]["public_shape"] == ["pverp"]
    assert by_name["cush"]["role"] == "inout"
    assert by_name["mix"]["role"] == "structural"
    # Judgement is left to the reviewer, never guessed.
    assert by_name["p0_inv"]["range"] == "REVIEW"
    assert draft["parameters"] == "REVIEW" and draft["module_state"] == "REVIEW"
