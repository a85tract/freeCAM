from pathlib import Path

from freecam.model.device_codegen import (
    DeviceDescription,
    _load_ccpp_entrypoints,
    _validate_dependencies,
)


PROJECT = Path(__file__).resolve().parents[2]
DESCRIPTOR = (
    PROJECT
    / "examples"
    / "plugins"
    / "runtime_temperature_offset"
    / "device.yaml"
)


def test_runtime_device_uses_builtin_metadata_parser() -> None:
    description = DeviceDescription.from_yaml(DESCRIPTOR, project_root=PROJECT)

    entrypoints = _load_ccpp_entrypoints(description)
    dependencies = _validate_dependencies(description)

    run = entrypoints["runtime_temperature_offset_run"]
    assert run.module == "runtime_temperature_offset"
    assert [argument.local_name for argument in run.arguments] == [
        "field",
        "increment",
        "errmsg",
        "errflg",
    ]
    assert run.arguments[0].dimensions == (
        "horizontal_loop_extent",
        "vertical_layer_dimension",
    )
    assert dependencies == ("ccpp_kinds",)
    assert description.providers["ccpp_kinds"] == (
        PROJECT / "native/pi_cam/support/ccpp_kinds.F90"
    ).resolve()
