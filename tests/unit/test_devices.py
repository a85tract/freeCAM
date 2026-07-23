from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pycam_sima import DeviceRegistry
from pycam_sima.model.contracts import FieldContract
from pycam_sima.model.device_codegen import (
    DeviceDescription,
    _load_ccpp_entrypoints,
    _validate_dependencies,
)
from pycam_sima.model.errors import DeviceBuildError, DeviceContractError
from pycam_sima.model.state import StatePool


ROOT = Path(__file__).resolve().parents[2]
DEVICE_ROOT = ROOT / "build/devices"


def _field(
    name: str,
    ccpp_name: str,
    dimensions: tuple[str, ...],
    units: str,
    *,
    intent: str = "inout",
) -> FieldContract:
    return FieldContract(
        standard_name=name,
        ccpp_standard_name=ccpp_name,
        dtype="float64",
        dimensions=dimensions,
        intent=intent,
        category="test",
        units=units,
    )


def _kessler_pool(*, timestep: float = 1800.0) -> StatePool:
    dimensions = {"nphys_local": 5, "pver": 30}
    contracts = (
        _field(
            "dt", "timestep_for_physics", (), "s", intent="in"
        ),
        _field(
            "lv",
            "latent_heat_of_vaporization_of_water_at_0c",
            (),
            "J kg-1",
            intent="in",
        ),
        _field(
            "pref", "surface_reference_pressure", (), "Pa", intent="in"
        ),
        _field(
            "rhoqr",
            "fresh_liquid_water_density_at_0c",
            (),
            "kg m-3",
            intent="in",
        ),
        _field(
            "cpair",
            (
                "composition_dependent_specific_heat_of_dry_air_at_"
                "constant_pressure"
            ),
            ("nphys_local", "pver"),
            "J kg-1 K-1",
            intent="in",
        ),
        _field(
            "rair",
            "composition_dependent_gas_constant_of_dry_air",
            ("nphys_local", "pver"),
            "J kg-1 K-1",
            intent="in",
        ),
        _field(
            "rho",
            "dry_air_density",
            ("nphys_local", "pver"),
            "kg m-3",
            intent="in",
        ),
        _field(
            "z",
            "geopotential_height_wrt_surface",
            ("nphys_local", "pver"),
            "m",
            intent="in",
        ),
        _field(
            "pk",
            "dimensionless_exner_function",
            ("nphys_local", "pver"),
            "1",
            intent="in",
        ),
        _field(
            "theta",
            "air_potential_temperature",
            ("nphys_local", "pver"),
            "K",
        ),
        _field(
            "qv",
            "water_vapor_mixing_ratio_wrt_dry_air",
            ("nphys_local", "pver"),
            "kg kg-1",
        ),
        _field(
            "qc",
            "cloud_liquid_water_mixing_ratio_wrt_dry_air",
            ("nphys_local", "pver"),
            "kg kg-1",
        ),
        _field(
            "qr",
            "rain_mixing_ratio_wrt_dry_air",
            ("nphys_local", "pver"),
            "kg kg-1",
        ),
        _field(
            "precl",
            "total_precipitation_rate_at_surface",
            ("nphys_local",),
            "m s-1",
        ),
        _field(
            "relhum",
            "relative_humidity",
            ("nphys_local", "pver"),
            "%",
        ),
    )
    pool = StatePool(dimensions, contracts=contracts, alias_rules=())
    level = np.linspace(0.0, 1.0, dimensions["pver"])

    def tile(value: np.ndarray) -> np.ndarray:
        return np.asfortranarray(
            np.tile(value, (dimensions["nphys_local"], 1))
        )

    pool.set("dt", timestep)
    pool.set("lv", 2.501e6)
    pool.set("pref", 100000.0)
    pool.set("rhoqr", 1000.0)
    pool.set("cpair", tile(np.full(30, 1004.64)))
    pool.set("rair", tile(np.full(30, 287.0423113650487)))
    pool.set("rho", tile(0.02 + 1.18 * level))
    pool.set("z", tile(25000.0 - 24900.0 * level))
    pool.set("pk", tile(0.2 + 0.8 * level))
    pool.set("theta", tile(310.0 - 15.0 * level))
    pool.set("qv", tile(1.0e-6 + 0.012 * level))
    pool.set(
        "qc", tile(2.0e-4 * np.exp(-((level - 0.75) / 0.15) ** 2))
    )
    pool.set(
        "qr", tile(2.0e-5 * np.exp(-((level - 0.85) / 0.10) ** 2))
    )
    return pool


def test_ccpp_standard_names_resolve_zero_copy_state() -> None:
    pool = _kessler_pool()
    assert pool.ccpp_field_name("air_potential_temperature") == "theta"
    assert np.shares_memory(
        pool.get_ccpp("air_potential_temperature"), pool.get("theta")
    )


def test_generated_registry_discovers_both_source_devices() -> None:
    registry = DeviceRegistry(DEVICE_ROOT)
    assert registry.process_names == frozenset(
        {"kessler", "kessler_update"}
    )
    descriptions = {item["name"]: item for item in registry.describe()}
    assert descriptions["kessler"]["state_policy"] == (
        "reinitialize_each_run"
    )
    assert descriptions["kessler_update"]["entrypoints"] == (
        "initialize",
        "run",
        "timestep_final",
        "timestep_initial",
    )


def test_registry_runs_original_kessler_without_replacing_arrays() -> None:
    registry = DeviceRegistry(DEVICE_ROOT)
    pool = _kessler_pool()
    before = pool.pointer_records()
    original_theta = pool.get("theta").copy()
    registry.invoke("kessler", pool)
    pool.assert_pointer_stability(before)
    assert not np.array_equal(pool.get("theta"), original_theta)
    assert np.isfinite(pool.get("relhum")).all()


def test_device_state_policies_control_initialize_calls(monkeypatch) -> None:
    device = DeviceRegistry(DEVICE_ROOT).devices["kessler"]
    calls: list[str] = []
    monkeypatch.setattr(
        device, "call", lambda entrypoint, pool: calls.append(entrypoint)
    )

    device.invoke_process("kessler", object())
    device.invoke_process("kessler", object())
    assert calls == ["initialize", "run", "initialize", "run"]

    calls.clear()
    device.state_policy = "initialize_once"
    device._initialized = False
    device.invoke_process("kessler", object())
    device.invoke_process("kessler", object())
    assert calls == ["initialize", "run", "run"]

    calls.clear()
    device.state_policy = "stateless"
    device.invoke_process("kessler", object())
    assert calls == ["run"]


def test_original_error_message_reaches_python() -> None:
    registry = DeviceRegistry(DEVICE_ROOT)
    pool = _kessler_pool(timestep=0.0)
    with pytest.raises(RuntimeError, match="nonpositive dt"):
        registry.invoke("kessler", pool)


def test_device_contract_rejects_wrong_host_units() -> None:
    registry = DeviceRegistry(DEVICE_ROOT)
    pool = _kessler_pool()
    contract = pool.contracts["theta"]
    pool.contracts["theta"] = FieldContract(
        standard_name=contract.standard_name,
        ccpp_standard_name=contract.ccpp_standard_name,
        dtype=contract.dtype,
        dimensions=contract.dimensions,
        intent=contract.intent,
        category=contract.category,
        units="m",
    )
    with pytest.raises(DeviceContractError, match="with units"):
        registry.invoke("kessler", pool)


def test_ccpp_parser_verifies_original_source_and_metadata() -> None:
    description = DeviceDescription.from_yaml(
        ROOT / "devices/kessler/device.yaml", project_root=ROOT
    )
    entrypoints = _load_ccpp_entrypoints(description)
    assert tuple(entrypoints) == ("kessler_init", "kessler_run")
    run = entrypoints["kessler_run"]
    assert run.module == "kessler"
    assert [item.local_name for item in run.arguments[:5]] == [
        "ncol",
        "nz",
        "dt",
        "lyr_surf",
        "lyr_toa",
    ]


def test_host_framework_dependency_is_rejected_before_build(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bad_scheme.F90"
    source.write_text(
        "module bad_scheme\n"
        "  use mpi\n"
        "contains\n"
        "end module bad_scheme\n"
    )
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "bad_scheme",
                "fortran_module": "bad_scheme",
                "sources": [str(source)],
                "metadata": [
                    str(
                        ROOT
                        / "external/CAM-SIMA/src/physics/ncar_ccpp/"
                        "schemes/kessler/kessler.meta"
                    )
                ],
                "source_modules": ["bad_scheme"],
                "providers": {},
                "state_policy": "stateless",
                "dimension_bindings": {},
                "entrypoints": {"run": {"table": "bad_scheme_run"}},
                "processes": {"bad_scheme": "run"},
            },
            sort_keys=False,
        )
    )
    description = DeviceDescription.from_yaml(
        descriptor, project_root=ROOT
    )
    with pytest.raises(DeviceBuildError, match="host/framework dependency"):
        _validate_dependencies(description)


def test_generated_manifest_points_to_pinned_original_source() -> None:
    manifest = json.loads(
        (DEVICE_ROOT / "kessler/device.json").read_text()
    )
    assert manifest["source"]["files"] == [
        "external/CAM-SIMA/src/physics/ncar_ccpp/schemes/kessler/kessler.F90"
    ]
    assert manifest["persistent_native_state"] is False
    generated = (
        DEVICE_ROOT / "kessler/generated/kessler_adapter.F90"
    ).read_text()
    assert "use kessler, only: kessler_init,kessler_run" in generated
    assert "call kessler_run(" in generated
    assert "36.34" not in generated
    assert not (ROOT / "native/kernels/kessler_kernel.F90").exists()
