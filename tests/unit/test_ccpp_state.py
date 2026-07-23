from pathlib import Path

import pytest

from pycam_sima import CCPPStateSchema, DeviceCatalog
from pycam_sima.model.contracts import default_contracts
from pycam_sima.model.errors import DeviceContractError


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
