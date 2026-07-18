import importlib.util
from pathlib import Path


def load_generator():
    path = Path("tools/generate_kessler_contract.py")
    spec = importlib.util.spec_from_file_location("generate_kessler_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contract_has_exact_suite_order(tmp_path):
    generator = load_generator()
    contract = generator.generate(Path("external/CAM-SIMA").resolve(), tmp_path / "contract.json")
    groups = contract["groups"]
    before = [entry["name"] for entry in groups["physics_before_coupler"]]
    after = [entry["name"] for entry in groups["physics_after_coupler"]]
    assert len(before) == 19
    assert len(after) == 5
    assert before[0] == "calc_exner"
    assert before[6] == "kessler"
    assert before[-1] == "kessler_diagnostics"
    assert after == [
        "thermo_water_update",
        "check_energy_scaling",
        "dycore_energy_consistency_adjust",
        "apply_tendency_of_air_temperature",
        "sima_tend_diagnostics",
    ]
