from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pycam_sima import DeviceRegistry, FortranDevice
from pycam_sima.model.contracts import FieldContract
from pycam_sima.model.device_codegen import (
    DeviceDescription,
    _load_ccpp_entrypoints,
    _validate_dependencies,
    build_device,
    resolve_source_closure,
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
    assert {"kessler", "kessler_update"} <= registry.process_names
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


def test_composite_registry_discovers_catalog_and_prefers_validated_kessler():
    catalog_registry = DeviceRegistry(ROOT / "build/catalog_devices")
    assert len(catalog_registry.devices) == 100
    for device in catalog_registry.devices.values():
        device._ensure_abi()

    registry = DeviceRegistry(
        (ROOT / "build/devices", ROOT / "build/catalog_devices")
    )
    assert "calculate_net_heating" in registry.process_names
    assert len(registry.devices) == 100
    assert registry.devices["kessler"].manifest_path.parent == (
        ROOT / "build/devices/kessler"
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


def test_default_fortran_logical_is_bridged_from_numpy_bool() -> None:
    dimensions = {"nphys_local": 3, "pver": 2}

    def contract(name, ccpp_name, dtype, dims, units, intent="in"):
        return FieldContract(
            standard_name=name,
            ccpp_standard_name=ccpp_name,
            dtype=dtype,
            dimensions=dims,
            intent=intent,
            category="test",
            units=units,
        )

    contracts = (
        contract(
            "qrl",
            (
                "tendency_of_dry_air_enthalpy_at_constant_pressure_due_to_"
                "longwave_radiation"
            ),
            "float64",
            ("nphys_local", "pver"),
            "J kg-1 s-1",
        ),
        contract(
            "qrs",
            (
                "tendency_of_dry_air_enthalpy_at_constant_pressure_due_to_"
                "shortwave_radiation"
            ),
            "float64",
            ("nphys_local", "pver"),
            "J kg-1 s-1",
        ),
        contract(
            "offline",
            "is_offline_dynamical_core",
            "bool",
            (),
            "flag",
        ),
        contract(
            "fsns",
            "shortwave_net_upward_flux_at_surface",
            "float64",
            ("nphys_local",),
            "W m-2",
        ),
        contract(
            "fsnt",
            "shortwave_net_outgoing_flux_at_model_top",
            "float64",
            ("nphys_local",),
            "W m-2",
        ),
        contract(
            "flns",
            "longwave_net_upward_flux_at_surface",
            "float64",
            ("nphys_local",),
            "W m-2",
        ),
        contract(
            "flnt",
            "longwave_net_outgoing_flux_at_model_top",
            "float64",
            ("nphys_local",),
            "W m-2",
        ),
        contract(
            "heating",
            "tendency_of_dry_air_enthalpy_at_constant_pressure",
            "float64",
            ("nphys_local", "pver"),
            "J kg-1 s-1",
            "inout",
        ),
        contract(
            "net_flux",
            "total_column_radiative_flux",
            "float64",
            ("nphys_local",),
            "W m-2",
            "out",
        ),
    )
    pool = StatePool(dimensions, contracts=contracts, alias_rules=())
    pool.set("qrl", 2.0)
    pool.set("qrs", 3.0)
    pool.set("offline", False)
    pool.set("fsns", [1.0, 2.0, 3.0])
    pool.set("fsnt", [11.0, 12.0, 13.0])
    pool.set("flns", [4.0, 5.0, 6.0])
    pool.set("flnt", [7.0, 8.0, 9.0])
    device = FortranDevice(
        ROOT / "build/catalog_devices/calculate_net_heating/device.json"
    )
    before = pool.pointer_records()
    device.invoke_process("calculate_net_heating", pool)
    pool.assert_pointer_stability(before)
    np.testing.assert_array_equal(pool.get("heating"), 5.0)
    np.testing.assert_array_equal(pool.get("net_flux"), 7.0)


def test_generated_descriptor_recursively_resolves_source_dependencies():
    description = DeviceDescription.from_yaml(
        ROOT
        / "devices/generated/hb_diff_exchange_coefficients/device.yaml",
        project_root=ROOT,
    )
    resolved = resolve_source_closure(description)
    modules = set(resolved.source_modules)
    assert "holtslag_boville_diff" in modules
    assert "atmos_phys_pbl_utils" in modules


def test_character_argument_round_trips_without_modifying_scheme_source(
    tmp_path: Path,
):
    source = tmp_path / "character_echo.F90"
    source.write_text(
        "module character_echo\n"
        "  implicit none\n"
        "  private\n"
        "  public :: character_echo_run\n"
        "contains\n"
        "  !> \\section arg_table_character_echo_run  Argument Table\n"
        "  !! \\htmlinclude character_echo_run.html\n"
        "  subroutine character_echo_run(input,output,errmsg,errflg)\n"
        "    character(len=*), intent(in) :: input\n"
        "    character(len=*), intent(out) :: output\n"
        "    character(len=*), intent(out) :: errmsg\n"
        "    integer, intent(out) :: errflg\n"
        "    output=input\n"
        "    errmsg=''\n"
        "    errflg=0\n"
        "  end subroutine character_echo_run\n"
        "end module character_echo\n"
    )
    metadata = tmp_path / "character_echo.meta"
    metadata.write_text(
        "[ccpp-table-properties]\n"
        "  name = character_echo\n"
        "  type = scheme\n"
        "[ccpp-arg-table]\n"
        "  name = character_echo_run\n"
        "  type = scheme\n"
        "[ input ]\n"
        "  standard_name = character_echo_input\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = in\n"
        "[ output ]\n"
        "  standard_name = character_echo_output\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errmsg ]\n"
        "  standard_name = ccpp_error_message\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errflg ]\n"
        "  standard_name = ccpp_error_code\n"
        "  units = 1\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
    )
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "character_echo",
                "fortran_module": "character_echo",
                "sources": [str(source)],
                "metadata": [str(metadata)],
                "source_modules": ["character_echo"],
                "providers": {},
                "state_policy": "stateless",
                "dimension_bindings": {},
                "entrypoints": {
                    "run": {"table": "character_echo_run"}
                },
                "processes": {"character_echo": "run"},
            },
            sort_keys=False,
        )
    )
    manifest = build_device(
        descriptor,
        project_root=ROOT,
        output_root=tmp_path / "build",
        compiler="/opt/cray/pe/gcc/12.2.0/bin/gfortran",
        fflags=(
            "-O0",
            "-fPIC",
            "-ffree-line-length-none",
            "-cpp",
        ),
        ldflags=("-Wl,--no-undefined",),
    )
    contracts = (
        FieldContract(
            "input",
            "S16",
            (),
            "in",
            "test",
            "none",
            ccpp_standard_name="character_echo_input",
        ),
        FieldContract(
            "output",
            "S16",
            (),
            "out",
            "test",
            "none",
            ccpp_standard_name="character_echo_output",
        ),
    )
    pool = StatePool({}, contracts=contracts, alias_rules=())
    pool.set("input", np.asarray(b"hello", dtype="S16"))
    FortranDevice(manifest).invoke_process("character_echo", pool)
    assert pool.get("output").item().rstrip() == b"hello"


def test_opaque_derived_state_is_created_reused_and_released(
    tmp_path: Path,
):
    source = tmp_path / "opaque_counter.F90"
    source.write_text(
        "module opaque_counter\n"
        "  implicit none\n"
        "  type :: counter_t\n"
        "    integer :: value=0\n"
        "  end type counter_t\n"
        "contains\n"
        "  !> \\section arg_table_opaque_counter_init Argument Table\n"
        "  !! \\htmlinclude opaque_counter_init.html\n"
        "  subroutine opaque_counter_init(state,errmsg,errflg)\n"
        "    type(counter_t), intent(out) :: state\n"
        "    character(len=*), intent(out) :: errmsg\n"
        "    integer, intent(out) :: errflg\n"
        "    state%value=7; errmsg=''; errflg=0\n"
        "  end subroutine opaque_counter_init\n"
        "  !> \\section arg_table_opaque_counter_run Argument Table\n"
        "  !! \\htmlinclude opaque_counter_run.html\n"
        "  subroutine opaque_counter_run(state,value,errmsg,errflg)\n"
        "    type(counter_t), intent(inout) :: state\n"
        "    integer, intent(out) :: value\n"
        "    character(len=*), intent(out) :: errmsg\n"
        "    integer, intent(out) :: errflg\n"
        "    state%value=state%value+1; value=state%value\n"
        "    errmsg=''; errflg=0\n"
        "  end subroutine opaque_counter_run\n"
        "end module opaque_counter\n"
    )
    metadata = tmp_path / "opaque_counter.meta"
    metadata.write_text(
        "[ccpp-table-properties]\n"
        "  name = opaque_counter\n"
        "  type = scheme\n"
        "[ccpp-arg-table]\n"
        "  name = opaque_counter_init\n"
        "  type = scheme\n"
        "[ state ]\n"
        "  standard_name = opaque_counter_state\n"
        "  units = none\n"
        "  type = counter_t\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errmsg ]\n"
        "  standard_name = ccpp_error_message\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errflg ]\n"
        "  standard_name = ccpp_error_code\n"
        "  units = 1\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ccpp-arg-table]\n"
        "  name = opaque_counter_run\n"
        "  type = scheme\n"
        "[ state ]\n"
        "  standard_name = opaque_counter_state\n"
        "  units = none\n"
        "  type = counter_t\n"
        "  dimensions = ()\n"
        "  intent = inout\n"
        "[ value ]\n"
        "  standard_name = opaque_counter_value\n"
        "  units = count\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errmsg ]\n"
        "  standard_name = ccpp_error_message\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errflg ]\n"
        "  standard_name = ccpp_error_code\n"
        "  units = 1\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
    )
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "opaque_counter",
                "fortran_module": "opaque_counter",
                "sources": [str(source)],
                "metadata": [str(metadata)],
                "source_modules": ["opaque_counter"],
                "providers": {},
                "state_policy": "stateless",
                "dimension_bindings": {},
                "entrypoints": {
                    "initialize": {"table": "opaque_counter_init"},
                    "run": {
                        "table": "opaque_counter_run"
                    },
                },
                "processes": {
                    "opaque_counter:create": "initialize",
                    "opaque_counter:increment": "run",
                },
            },
            sort_keys=False,
        )
    )
    manifest = build_device(
        descriptor,
        project_root=ROOT,
        output_root=tmp_path / "build",
        compiler="/opt/cray/pe/gcc/12.2.0/bin/gfortran",
        fflags=(
            "-O0",
            "-fPIC",
            "-ffree-line-length-none",
            "-cpp",
        ),
        ldflags=("-Wl,--no-undefined",),
    )
    pool = StatePool(
        {},
        contracts=(
            FieldContract(
                "counter_value",
                "int32",
                (),
                "out",
                "test",
                "count",
                ccpp_standard_name="opaque_counter_value",
            ),
        ),
        alias_rules=(),
    )
    device = FortranDevice(manifest)
    device.invoke_process("opaque_counter:create", pool)
    first_address = pool.get_process_state(
        "opaque_counter_state"
    ).address
    device.invoke_process("opaque_counter:increment", pool)
    assert pool.get("counter_value").item() == 8
    assert (
        pool.get_process_state("opaque_counter_state").address
        == first_address
    )
    with pytest.raises(Exception, match="cannot checkpoint opaque"):
        pool.snapshot_arrays()
    pool.release_process_state()
    assert not pool.process_state_names


def test_python_owned_physical_constant_is_injected_into_fortran(
    tmp_path: Path,
):
    source = tmp_path / "constant_probe.F90"
    source.write_text(
        "module constant_probe\n"
        "  use ccpp_kinds, only: kind_phys\n"
        "  use physconst, only: gravit\n"
        "  implicit none\n"
        "contains\n"
        "  !> \\section arg_table_constant_probe_run Argument Table\n"
        "  !! \\htmlinclude constant_probe_run.html\n"
        "  subroutine constant_probe_run(value,errmsg,errflg)\n"
        "    real(kind_phys), intent(out) :: value\n"
        "    character(len=*), intent(out) :: errmsg\n"
        "    integer, intent(out) :: errflg\n"
        "    value=gravit; errmsg=''; errflg=0\n"
        "  end subroutine constant_probe_run\n"
        "end module constant_probe\n"
    )
    metadata = tmp_path / "constant_probe.meta"
    metadata.write_text(
        "[ccpp-table-properties]\n"
        "  name = constant_probe\n"
        "  type = scheme\n"
        "[ccpp-arg-table]\n"
        "  name = constant_probe_run\n"
        "  type = scheme\n"
        "[ value ]\n"
        "  standard_name = constant_probe_value\n"
        "  units = m s-2\n"
        "  type = real | kind = kind_phys\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errmsg ]\n"
        "  standard_name = ccpp_error_message\n"
        "  units = none\n"
        "  type = character | kind = len=*\n"
        "  dimensions = ()\n"
        "  intent = out\n"
        "[ errflg ]\n"
        "  standard_name = ccpp_error_code\n"
        "  units = 1\n"
        "  type = integer\n"
        "  dimensions = ()\n"
        "  intent = out\n"
    )
    descriptor = tmp_path / "device.yaml"
    descriptor.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "constant_probe",
                "fortran_module": "constant_probe",
                "sources": [str(source)],
                "metadata": [str(metadata)],
                "source_modules": ["constant_probe"],
                "providers": {
                    "ccpp_kinds": str(
                        ROOT / "native/devices/support/ccpp_kinds.F90"
                    ),
                    "physconst": str(
                        ROOT / "native/devices/support/physconst.F90"
                    )
                },
                "state_policy": "stateless",
                "dimension_bindings": {},
                "entrypoints": {
                    "run": {"table": "constant_probe_run"}
                },
                "processes": {"constant_probe": "run"},
            },
            sort_keys=False,
        )
    )
    manifest = build_device(
        descriptor,
        project_root=ROOT,
        output_root=tmp_path / "build",
        compiler="/opt/cray/pe/gcc/12.2.0/bin/gfortran",
        fflags=(
            "-O0",
            "-fPIC",
            "-ffree-line-length-none",
            "-cpp",
        ),
        ldflags=("-Wl,--no-undefined",),
    )
    pool = StatePool(
        {},
        contracts=(
            FieldContract(
                "gravitational_acceleration",
                "float64",
                (),
                "in",
                "constants",
                "m s-2",
            ),
            FieldContract(
                "value",
                "float64",
                (),
                "out",
                "test",
                "m s-2",
                ccpp_standard_name="constant_probe_value",
            ),
        ),
        alias_rules=(),
    )
    pool.set("gravitational_acceleration", 3.711)
    FortranDevice(manifest).invoke_process("constant_probe", pool)
    assert pool.get("value").item() == 3.711
