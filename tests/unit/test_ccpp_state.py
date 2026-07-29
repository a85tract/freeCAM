from pathlib import Path

import pytest

from pycam_sima import CCPPStateSchema, CCPPSuitePlan, DeviceCatalog
from pycam_sima.model.contracts import (
    default_contracts,
    model_ccpp_field_aliases,
)
from pycam_sima.model.devices import DeviceRegistry
from pycam_sima.model.errors import DeviceContractError
from pycam_sima.model.grid import dimensions_for_rank
from pycam_sima.model.state import StatePool


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog():
    return DeviceCatalog.discover(ROOT)


def test_every_suite_has_a_complete_machine_readable_state_schema(catalog):
    for suite in catalog.summary()["suite_names"]:
        schema = CCPPStateSchema.from_catalog(catalog, suite)
        report = schema.report()
        assert report["field_count"] > 0
        assert (
            report["primitive_field_count"]
            + report["opaque_field_count"]
            == report["field_count"]
        )
        assert {"nphys_local", "pver", "pverp"} <= set(
            report["required_dimensions"]
        )


def test_original_namelist_xml_generates_state_bindings(catalog):
    schema = CCPPStateSchema.from_catalog(catalog, "musica")

    assert {
        "micm_solver_type",
        "filename_of_micm_configuration",
        "filename_of_tuvx_configuration",
        "filename_of_tuvx_micm_mapping_configuration",
    } <= set(schema.namelist_bindings)
    binding = schema.namelist_bindings["micm_solver_type"][0]
    assert binding.group == "musica_ccpp"
    assert binding.local_name == "micm_solver_type"
    assert binding.default_value == "Rosenbrock"


def test_kessler_schema_extends_existing_python_state_without_conflicts(
    catalog,
):
    schema = CCPPStateSchema.from_catalog(catalog, "kessler")
    additions = schema.additional_contracts(default_contracts())
    names = {
        contract.ccpp_standard_name for contract in additions
    }
    assert "air_pressure" in names
    assert "air_temperature" not in names
    assert not schema.conversion_fields


def test_ccpp_constituent_minima_are_not_component_registry_aliases(catalog):
    aliases = model_ccpp_field_aliases(
        ("cloud_liquid_water", "rain", "water_vapor")
    )
    assert "ccpp_constituent_minimum_values" not in aliases

    schema = CCPPStateSchema.from_catalog(catalog, "cam4")
    generated = schema.additional_contracts(
        default_contracts(),
        provided_standard_names=aliases,
    )
    minima = next(
        contract
        for contract in generated
        if contract.ccpp_standard_name
        == "ccpp_constituent_minimum_values"
    )
    assert minima.standard_name == "ccpp_ccpp_constituent_minimum_values"
    assert minima.dimensions == ("number_of_ccpp_constituents",)


def test_pool_schema_selects_process_fields_from_the_active_suite(catalog):
    kessler = CCPPStateSchema.from_catalog(catalog, "kessler")
    held_suarez = CCPPStateSchema.from_catalog(
        catalog, "held_suarez_1994"
    )
    kessler_names = {
        contract.standard_name for contract in kessler.pool_contracts()
    }
    held_suarez_names = {
        contract.standard_name
        for contract in held_suarez.pool_contracts()
    }

    assert "air_temperature_previous_timestep" in kessler_names
    assert "large_scale_precipitation_rate" in kessler_names
    assert "air_temperature_previous_timestep" not in held_suarez_names
    assert "large_scale_precipitation_rate" not in held_suarez_names


def test_pool_schema_always_contains_python_host_phase_work_fields(catalog):
    schema = CCPPStateSchema.from_catalog(catalog, "musica")
    names = {
        contract.standard_name for contract in schema.pool_contracts()
    }

    assert {
        "column_dry_air_specific_heat",
        "column_dry_air_gas_constant",
        "static_energy",
        "thermodynamic_level_height",
    } <= names


def test_custom_suite_schema_uses_process_names_not_a_pinned_suite_name(
    catalog,
):
    plan = CCPPSuitePlan.from_xml(
        ROOT
        / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/"
        "suite_kessler.xml"
    )
    custom = CCPPStateSchema.from_scheme_names(
        catalog,
        "my_experiment",
        (scheme.name for scheme in plan.schemes),
    )
    pinned = CCPPStateSchema.from_catalog(catalog, "kessler")
    assert custom.requirements == pinned.requirements
    assert not custom.unresolved_schemes

    unresolved = CCPPStateSchema.from_scheme_names(
        catalog,
        "runtime_plugin_suite",
        ("kessler", "not_installed_yet"),
    )
    assert unresolved.unresolved_schemes == ("not_installed_yet",)


def test_metadata_generated_inputs_fail_closed_until_initialized(catalog):
    schema = CCPPStateSchema.from_catalog(catalog, "kessler")
    initialized, generated = schema.pool_contract_groups()
    pool = StatePool(dimensions_for_rank(0), contracts=initialized)
    for contract in generated:
        pool.register_field(
            contract,
            initialized=False,
            dynamic=False,
        )

    assert not pool.is_initialized("ccpp_air_pressure")
    registry = DeviceRegistry(ROOT / "build/catalog_devices")
    with pytest.raises(DeviceContractError, match="uninitialized"):
        registry.invoke("calc_exner", pool)
    pool.set("ccpp_air_pressure", 90000.0)
    assert pool.is_initialized("ccpp_air_pressure")


def test_cam4_schema_marks_real_conversion_and_opaque_boundaries(catalog):
    schema = CCPPStateSchema.from_catalog(catalog, "cam4")
    assert (
        "inverse_cosine_of_radiation_transport_angle_per_column_and_g_point"
        in schema.primitive_fields
    )
    assert "air_pressure_thickness_of_dry_air" in schema.conversion_fields
    assert schema.opaque_fields
    with pytest.raises(DeviceContractError, match="conversion policy"):
        schema.additional_contracts(default_contracts())
