from pathlib import Path
from types import SimpleNamespace

import pytest

from freecam import DeviceBuildError, DeviceCatalog
from freecam import cli
from freecam.model.ccpp_suite import CCPPSuitePlan
from freecam.model.device_codegen import (
    DeviceDescription,
    _project_module_index,
)
from freecam.model.device_catalog import (
    _load_descriptor_overrides,
    _module_source_index,
)


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
    assert summary["descriptor_override_count"] == 3
    assert summary["descriptor_override_source"] == "devices/overrides.yaml"
    assert all(
        entry.metadata and entry.source
        for entry in catalog.entries.values()
    )


def test_descriptor_source_tree_has_no_parallel_scheme_directories() -> None:
    directories = {
        path.name
        for path in (ROOT / "devices").iterdir()
        if path.is_dir()
    }
    assert directories == {"generated"}
    assert (ROOT / "devices/overrides.yaml").is_file()


def test_module_indexes_prefer_serial_cpu_source_over_api_and_accel(
    tmp_path: Path,
) -> None:
    cam_root = tmp_path / "external/CAM-SIMA"
    for variant in ("api", "accel", "serial"):
        source = cam_root / "kernels" / variant / "duplicate.F90"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(
            "module duplicate_provider\n"
            "  implicit none\n"
            "end module duplicate_provider\n"
        )

    expected = (cam_root / "kernels/serial/duplicate.F90").resolve()
    assert _module_source_index(cam_root)["duplicate_provider"] == expected

    _project_module_index.cache_clear()
    try:
        assert (
            _project_module_index(tmp_path)["duplicate_provider"] == expected
        )
    finally:
        _project_module_index.cache_clear()


def test_catalog_records_lifecycle_and_suite_occurrences(
    catalog: DeviceCatalog,
):
    kessler = catalog.entries["kessler"]
    assert kessler.module == "kessler"
    assert kessler.lifecycle == ("initialize", "run")
    assert {item.suite for item in kessler.occurrences} == {"kessler"}
    assert kessler.device_abi_v1_compatible


@pytest.mark.parametrize(
    ("suite", "expected"),
    (
        (
            "cam7",
            (
                "water_vapor_mixing_ratio_wrt_moist_air_and_"
                "condensed_water",
            ),
        ),
        (
            "cam4",
            (
                "water_vapor_mixing_ratio_wrt_moist_air_and_"
                "condensed_water",
                "cloud_ice_mixing_ratio_wrt_moist_air_and_"
                "condensed_water",
                "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_"
                "condensed_water",
            ),
        ),
    ),
)
def test_catalog_reconstructs_generated_suite_constituent_registry(
    catalog: DeviceCatalog,
    suite: str,
    expected: tuple[str, ...],
) -> None:
    plan = CCPPSuitePlan.from_xml(
        ROOT
        / "external/CAM-SIMA/src/physics/ncar_ccpp/suites"
        / f"suite_{suite}.xml"
    )
    assert catalog.suite_constituent_standard_names(
        scheme.name for scheme in plan.schemes
    ) == expected


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
    assert kessler.state_policy == "reinitialize_each_run"
    assert kessler.initialize_entrypoint == "initialize"
    assert kessler.global_bindings == {
        "vertical_index_at_surface_adjacent_layer": {
            "source": "dimension",
            "name": "pver",
        },
        "vertical_index_at_top_adjacent_layer": {
            "source": "literal",
            "value": 1,
        },
    }
    assert kessler.providers["ccpp_kinds"] == (
        ROOT / "native/devices/support/ccpp_kinds.F90"
    )
    assert "shr_kind_mod" in kessler.providers

    kessler_update = DeviceDescription.from_yaml(
        tmp_path / "kessler_update/device.yaml", project_root=ROOT
    )
    assert kessler_update.state_policy == "reinitialize_each_run"
    assert kessler_update.initialize_entrypoint == "initialize"


def test_descriptor_overrides_fail_closed(
    tmp_path: Path,
) -> None:
    override = tmp_path / "overrides.yaml"
    override.write_text(
        "schema_version: 1\n"
        "schemes:\n"
        "  not_in_suite:\n"
        "    state_policy: stateless\n"
    )
    with pytest.raises(DeviceBuildError, match="inactive schemes"):
        _load_descriptor_overrides(override, frozenset({"kessler"}))

    override.write_text(
        "schema_version: 1\n"
        "schemes:\n"
        "  kessler:\n"
        "    sources: [replacement.F90]\n"
    )
    with pytest.raises(DeviceBuildError, match="unsupported keys"):
        _load_descriptor_overrides(override, frozenset({"kessler"}))


def test_build_kernels_regenerates_descriptors_before_make(
    monkeypatch,
) -> None:
    calls: list[tuple] = []

    class _Catalog:
        def write_descriptors(self, output, *, clean):
            calls.append(("generate", Path(output), clean))

    monkeypatch.setattr(
        cli.DeviceCatalog, "discover", lambda _root: _Catalog()
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, *, check: calls.append(
            ("make", tuple(command), check)
        ),
    )

    assert cli.command_build_kernels(SimpleNamespace()) == 0
    assert calls[0] == (
        "generate",
        ROOT / "devices/generated",
        True,
    )
    assert calls[1][0] == "make"
