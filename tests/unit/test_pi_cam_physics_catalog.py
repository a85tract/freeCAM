import json
from pathlib import Path

from freecam.pi_cam.physics_catalog import (
    PICAMPhysicsCatalog,
    PICAMPhysicsRules,
    build_physics_catalog,
)


PROJECT = Path(__file__).resolve().parents[2]


def test_default_catalog_is_flat_unique_and_case_reachable() -> None:
    catalog = PICAMPhysicsCatalog.load_default()

    assert catalog.reachable_procedures == 372
    assert len(catalog.rules_sha256) == 64
    assert len(catalog.processes) == 371
    assert len(catalog.physics_processes) == 276
    assert len(catalog.helpers) == 95
    assert catalog.excluded_lifecycle == 1
    assert len({process.name for process in catalog.processes}) == 371
    assert catalog.process("cloud_fraction_fice").qualified_name == (
        "cloud_fraction::cldfrc_fice"
    )
    assert catalog.process("cldfrc_fice").name == "cloud_fraction_fice"
    assert catalog.process("zm_conv_evap").parent_processes == (
        "deep_convection",
        "shallow_convection",
    )
    assert catalog.process("math_lib::gamma").level == "helper"
    assert sum(process.generated_adapter for process in catalog.processes) == 22


def test_committed_catalog_is_reproducible_from_validation_evidence() -> None:
    inventory = json.loads(
        (PROJECT / "validation/pi_cam_kernel_inventory.json").read_text()
    )
    adapters = json.loads(
        (PROJECT / "validation/pi_cam_generated_adapter_validation.json").read_text()
    )

    rules = PICAMPhysicsRules.load(
        PROJECT / "native/pi_cam/physics_process_rules.yaml"
    )
    rebuilt = build_physics_catalog(
        inventory,
        generated_adapters=adapters,
        rules=rules,
    )
    committed = PICAMPhysicsCatalog.load_default()

    assert rebuilt.machine_record() == committed.machine_record()


def test_colliding_fortran_names_receive_stable_flat_names() -> None:
    catalog = PICAMPhysicsCatalog.load_default()
    radiation = tuple(
        process
        for process in catalog.processes
        if process.routine == "radiation_tend"
    )

    assert {process.name for process in radiation} == {
        "cam_radiation",
        "rrtmg_radiation",
    }
    assert all("." not in process.name for process in catalog.processes)
    assert all("::" not in process.name for process in catalog.processes)
