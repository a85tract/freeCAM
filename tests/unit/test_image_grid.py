"""The grid a native image was compiled for, and a configuration that must match it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freecam.pi_cam.config import PICAMConfig
from freecam.pi_cam.driver import PICAMDriver
from freecam.pi_cam.errors import NativeCAMError
from freecam.pi_cam.image_grid import check_config_grid, grid_from_commands, image_grid

FLAGS = ["ftn", "-c", "-DPCOLS=16", "-DPLEV=30", "-DPCNST=57", "-DPSUBCOLS=1", "-DSPMD"]


def test_the_grid_is_read_from_every_compile_command_and_must_agree() -> None:
    assert grid_from_commands({"a.F90": FLAGS, "b.F90": " ".join(FLAGS)}) == {
        "pcols": 16, "pver": 30, "pcnst": 57, "psubcols": 1}
    with pytest.raises(NativeCAMError, match="disagree on PCOLS: 16 and 32"):
        grid_from_commands({"a.F90": FLAGS, "b.F90": [*FLAGS[:2], "-DPCOLS=32"]})


def test_a_manifest_s_recorded_dimensions_come_first() -> None:
    assert image_grid({"dimensions": {"pcols": 32, "pver": 30}, "compile_commands": {"a": FLAGS}}) == {"pcols": 32, "pver": 30}
    assert image_grid({"compile_commands": {"a": FLAGS}})["pcols"] == 16      # a manifest from before they were recorded


def test_a_configuration_runs_only_on_an_image_of_its_grid() -> None:
    config = PICAMConfig(case_name="t", source_root=".")
    check_config_grid(config, {"pcols": 16, "pver": 30, "pcnst": 57})
    with pytest.raises(NativeCAMError, match="compiled with pcols=32 .PCOLS.; the configuration says pcols=16"):
        check_config_grid(config, {"pcols": 32, "pver": 30})
    with pytest.raises(NativeCAMError, match="compiled with pcols=32"):
        PICAMDriver(config, SimpleNamespace(), SimpleNamespace(grid={"pcols": 32, "pver": 30}), rank=0, size=1)
