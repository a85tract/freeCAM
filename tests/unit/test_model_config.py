from pathlib import Path

import pytest

from pycam_sima import CAM_SE_FVM_V1, ModelConfig
from pycam_sima.model.errors import ConfigurationError


ROOT = Path(__file__).resolve().parents[2]


def test_model_config_is_not_locked_to_fkessler_or_one_timestep() -> None:
    config = ModelConfig(
        source_root=str(ROOT / "external/CAM-SIMA"),
        physics_suite="adiabatic",
        dt_seconds=900,
        stop_n=12,
        case_name="adiabatic-control",
    )
    config.validate()
    assert config.resolve_suite_xml().name == "suite_adiabatic.xml"
    assert config.verify_suite().is_file()


def test_runtime_capabilities_are_separate_from_generic_config() -> None:
    config = ModelConfig(
        source_root=str(ROOT / "external/CAM-SIMA"),
        physics_suite="held_suarez_1994",
        dt_seconds=600,
        stop_n=3,
    )
    CAM_SE_FVM_V1.validate(config)
    CAM_SE_FVM_V1.validate(config.with_overrides(mpi_size=18))
    with pytest.raises(
        ConfigurationError, match="mpi_size=55.*maximum available 54"
    ):
        CAM_SE_FVM_V1.validate(config.with_overrides(mpi_size=55))

    config.with_overrides(calendar="GREGORIAN").validate()
    with pytest.raises(ConfigurationError, match="calendar='GREGORIAN'"):
        CAM_SE_FVM_V1.validate(
            config.with_overrides(calendar="GREGORIAN")
        )
    seven_constituents = config.with_overrides(constituent_count=7)
    seven_constituents.validate()
    with pytest.raises(ConfigurationError, match="constituent_count=7"):
        CAM_SE_FVM_V1.validate(seven_constituents)

    unsupported_grid = config.with_overrides(
        grid="ne4np4.pg3",
        ne=4,
    )
    with pytest.raises(ConfigurationError, match="runtime 'cam-se-fvm-v1'"):
        CAM_SE_FVM_V1.validate(unsupported_grid)


def test_generic_configuration_still_rejects_invalid_values() -> None:
    with pytest.raises(ConfigurationError, match="dt_seconds must be positive"):
        ModelConfig(dt_seconds=0).validate()
