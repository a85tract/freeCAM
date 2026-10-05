"""The components' layout over a run's MPI ranks, which CESM reads from drv_in."""

from __future__ import annotations

from pathlib import Path

import pytest

from freecam.pi_cam import CESMOnlineBoundaryProvider
from freecam.pi_cam.layout import (
    LayoutError,
    component_layout,
    derive_config,
    lay_out,
    read_layout,
    rewrite_drv_in,
)

#: the drv_in every rank-sweep run used (validation/pi_cam_online_model_*rank_*-tl.json):
#: (lnd_ntasks, ice_rootpe, ocn_rootpe, rof/glc/wav ntasks)
SWEEP = {128: (128, 0, 0, 128), 192: (32, 32, 160, 32), 256: (96, 96, 224, 96), 300: (140, 140, 268, 128),
         384: (192, 192, 320, 128), 400: (200, 200, 328, 128), 512: (256, 256, 384, 128)}


def drv_in(ranks: int = 512) -> str:
    entries = component_layout(ranks) | {"stop_n": 50, "restart_n": 50}
    return "&ccsm_pes\n" + "".join(f"  {name} = {value}\n" for name, value in entries.items()) + "/\n"


@pytest.mark.parametrize("ranks", sorted(SWEEP))
def test_the_layout_is_the_one_each_sweep_run_used(ranks: int) -> None:
    layout = component_layout(ranks)
    land, ice_root, ocn_root, other = SWEEP[ranks]

    assert (layout["atm_ntasks"], layout["cpl_ntasks"]) == (ranks, ranks)
    assert (layout["lnd_ntasks"], layout["ice_rootpe"], layout["ocn_rootpe"]) == (land, ice_root, ocn_root)
    assert (layout["ice_ntasks"], layout["ocn_ntasks"]) == (128, 32)
    assert layout["rof_ntasks"] == layout["glc_ntasks"] == layout["wav_ntasks"] == other


def test_512_is_the_admitted_layout() -> None:
    assert component_layout(512) == {"atm_ntasks": 512, "cpl_ntasks": 512, "lnd_ntasks": 256, "ice_ntasks": 128,
                                     "ice_rootpe": 256, "ocn_ntasks": 32, "ocn_rootpe": 384, "rof_ntasks": 128,
                                     "glc_ntasks": 128, "wav_ntasks": 128}


def test_fewer_ranks_than_cice_is_compiled_for_is_refused() -> None:
    with pytest.raises(LayoutError, match="128 tasks"):
        component_layout(96)


@pytest.mark.parametrize("ranks", sorted(SWEEP))
def test_every_component_fits_inside_the_run(ranks: int, tmp_path: Path) -> None:
    # the provider's own check, run as the online provider runs it
    (tmp_path / "drv_in").write_text(drv_in(ranks))
    provider = CESMOnlineBoundaryProvider(library=tmp_path / "lib.so", run_dir=tmp_path, ranks=ranks)

    provider._check_layout(ranks)


def test_a_drv_in_is_rewritten_entry_by_entry(tmp_path: Path) -> None:
    path = tmp_path / "drv_in"
    path.write_text(drv_in(512))

    values = lay_out(path, 256, steps=240)

    assert read_layout(path.read_text()) == component_layout(256)
    assert values["stop_n"] == 240 and "restart_n = 240" in path.read_text()
    with pytest.raises(LayoutError, match="atm_ntasks"):
        rewrite_drv_in("&ccsm_pes\n/\n", {"atm_ntasks": 256})


def test_a_derived_configuration_changes_only_what_it_is_asked(tmp_path: Path) -> None:
    source = tmp_path / "case.yaml"
    source.write_text("case_name: x\nmpi_size: 512\nstop_n: 50\npcols: 16\n")

    derived = derive_config(source, tmp_path / "out/config.yaml", mpi_size=256)

    assert derived.read_text() == "case_name: x\nmpi_size: 256\nstop_n: 50\npcols: 16\n"
    (tmp_path / "bare.yaml").write_text("case_name: x\n")
    with pytest.raises(LayoutError, match="mpi_size"):
        derive_config(tmp_path / "bare.yaml", tmp_path / "again.yaml", mpi_size=128)


def test_the_steps_the_surface_components_run_are_read_from_drv_in(tmp_path: Path) -> None:
    from freecam.pi_cam.layout import read_horizon

    month = 'stop_option = "nsteps"\n  stop_n = 1488\n  restart_n = 1488\n'
    assert read_horizon(month) == 1488
    assert read_horizon("stop_option = 'ndays'\n stop_n = 31\n") is None        # counted another way
    assert read_horizon("stop_n = 1488\n") is None
    run = tmp_path / "provider-run"
    run.mkdir()
    (run / "drv_in").write_text("&seq_timemgr_inparm\n  " + month + "/\n")
    provider = CESMOnlineBoundaryProvider.__new__(CESMOnlineBoundaryProvider)
    provider.run_dir = run
    assert provider.steps_horizon == 1488
