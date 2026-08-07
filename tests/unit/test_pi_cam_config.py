from dataclasses import replace
from pathlib import Path

import pytest

from freecam.pi_cam import PICAMConfig, PICAMConfigurationError


def test_repository_pi_cam_config_is_cam_only() -> None:
    root = Path(__file__).parents[2]
    config = PICAMConfig.from_yaml(root / "configs/pi_cam_icesm131.yaml")

    assert config.mpi_size == 512
    assert config.orbital_year == 1850
    assert config.resolution == "ne16"
    assert config.substeps_per_coupling == 1
    assert config.initialization_lookahead_steps == 1
    assert config.native_manifest is not None
    assert config.native_manifest.is_absolute()
    assert config.native_manifest.name == "native_cam_manifest.json"
    assert config.source_root == root / "external/iCESM1.3.1_fzhu"
    payload = config.to_payload()
    assert "components" not in payload
    assert "coupler" not in payload


def test_pi_cam_config_rejects_unimplemented_case_keys() -> None:
    with pytest.raises(PICAMConfigurationError, match="unsupported"):
        PICAMConfig.from_mapping(
            {
                "case_name": "test",
                "source_root": "/tmp/source",
                "components": {"lnd": {}},
            }
        )


def test_pi_cam_config_requires_integral_substeps() -> None:
    with pytest.raises(PICAMConfigurationError, match="integer multiple"):
        PICAMConfig(
            case_name="test",
            source_root=Path("/tmp/source"),
            mpi_size=1,
            timestep_seconds=1000,
            coupling_seconds=1800,
        )


def test_pi_cam_fingerprint_ignores_checkout_and_build_paths() -> None:
    original = PICAMConfig(
        case_name="test",
        source_root=Path("/first/source"),
        native_manifest=Path("/first/build/manifest.json"),
        mpi_size=1,
    )
    relocated = replace(
        original,
        source_root=Path("/second/source"),
        native_manifest=Path("/second/build/manifest.json"),
    )

    assert original.fingerprint == relocated.fingerprint
