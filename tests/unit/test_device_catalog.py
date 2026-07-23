from pathlib import Path

import pytest

from pycam_sima import DeviceCatalog
from pycam_sima.model.device_codegen import DeviceDescription


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog() -> DeviceCatalog:
    return DeviceCatalog.discover(ROOT)


def test_catalog_covers_every_pinned_suite_scheme(catalog: DeviceCatalog):
    summary = catalog.summary()
    assert summary["suite_count"] == 7
    assert set(summary["suite_names"]) == {
        "adiabatic",
        "cam4",
        "cam7",
        "held_suarez_1994",
        "kessler",
        "musica",
        "tj2016",
    }
    assert summary["active_scheme_count"] > 100
    assert all(entry.metadata and entry.source for entry in catalog.entries.values())


def test_catalog_records_lifecycle_and_suite_occurrences(
    catalog: DeviceCatalog,
):
    kessler = catalog.entries["kessler"]
    assert kessler.module == "kessler"
    assert kessler.lifecycle == ("initialize", "run")
    assert {item.suite for item in kessler.occurrences} == {"kessler"}
    assert kessler.device_abi_v1_compatible


def test_catalog_fail_closed_blockers_are_machine_readable(
    catalog: DeviceCatalog,
):
    rrtmgp = catalog.entries["rrtmgp_lw_gas_optics"]
    assert not any(
        item.startswith(("derived_type:", "character_argument:"))
        for item in rrtmgp.blockers
    )
    assert (
        "air_temperature_for_rrtmgp"
        in rrtmgp.missing_statepool_fields
    )
    payload = catalog.machine_record()
    assert payload["schema_version"] == 1
    assert len(payload["schemes"]) == len(catalog.entries)


def test_catalog_accepts_python_owned_shaped_allocatable_fields(
    catalog: DeviceCatalog,
):
    rte = catalog.entries["rrtmgp_lw_rte"]
    lw_ds = next(
        argument
        for entrypoint in rte.entrypoints
        for argument in entrypoint.arguments
        if argument.local_name == "lw_ds"
    )
    assert lw_ds.allocatable
    assert lw_ds.caller_owned_allocatable
    assert not any(
        blocker.endswith(".lw_ds")
        for blocker in rte.blockers
        if blocker.startswith("allocatable_argument:")
    )


def test_catalog_generates_one_valid_descriptor_per_active_scheme(
    catalog: DeviceCatalog, tmp_path: Path
):
    descriptors = catalog.write_descriptors(tmp_path)
    assert len(descriptors) == len(catalog.entries) == 155
    assert len({path.parent.name for path in descriptors}) == 155
    for path in descriptors:
        description = DeviceDescription.from_yaml(path, project_root=ROOT)
        assert description.name == path.parent.name
        assert description.processes

    kessler = DeviceDescription.from_yaml(
        tmp_path / "kessler/device.yaml", project_root=ROOT
    )
    assert kessler.processes == {
        "kessler:initialize": "initialize",
        "kessler": "run",
        "kessler:run": "run",
    }
    assert kessler.providers["ccpp_kinds"] == (
        ROOT / "native/devices/support/ccpp_kinds.F90"
    )
    assert "shr_kind_mod" in kessler.providers
