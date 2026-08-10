from pathlib import Path

import pytest

from freecam import CAM_SE_FVM_V1, ModelConfig
from freecam.model.errors import ConfigurationError
from freecam.model.namelist import read_atm_in


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


def test_runtime_capabilities_accept_configurable_model_dimensions() -> None:
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

    gregorian = config.with_overrides(calendar="GREGORIAN")
    gregorian.validate()
    CAM_SE_FVM_V1.validate(gregorian)
    seven_constituents = config.with_overrides(constituent_count=7)
    seven_constituents.validate()
    CAM_SE_FVM_V1.validate(seven_constituents)
    split_constituents = config.with_overrides(
        constituent_count=10,
        advected_constituent_count=7,
    )
    split_constituents.validate()
    assert split_constituents.kernel_specialization[
        "constituent_count"
    ] == 7
    assert ModelConfig.from_mapping(
        split_constituents.as_dict()
    ) == split_constituents

    permuted_constituents = split_constituents.with_overrides(
        advected_constituent_indices=(0, 2, 3, 1, 4, 5, 6),
    )
    assert permuted_constituents.advected_constituent_indices == (
        0,
        2,
        3,
        1,
        4,
        5,
        6,
    )
    assert ModelConfig.from_mapping(
        permuted_constituents.as_dict()
    ) == permuted_constituents

    configurable_grid = config.with_overrides(
        grid="ne4np5.pg4",
        ne=4,
        np=5,
        fv_nphys=4,
        pver=72,
        mpi_size=96,
    )
    configurable_grid.validate()
    CAM_SE_FVM_V1.validate(configurable_grid)


def test_generic_configuration_still_rejects_invalid_values() -> None:
    with pytest.raises(ConfigurationError, match="dt_seconds must be positive"):
        ModelConfig(dt_seconds=0).validate()
    with pytest.raises(
        ConfigurationError,
        match="advected_constituent_count cannot exceed constituent_count",
    ):
        ModelConfig(
            constituent_count=3,
            advected_constituent_count=4,
        ).validate()
    with pytest.raises(
        ConfigurationError,
        match="advected_constituent_indices must be unique",
    ):
        ModelConfig(
            advected_constituent_indices=(0, 0, 2),
        ).validate()
    with pytest.raises(
        ConfigurationError,
        match="history_core_scheme is required",
    ):
        ModelConfig(history_core_boundary="after_scheme").validate()


def test_cam4_samples_history_at_its_native_diagnostic_scheme() -> None:
    config = ModelConfig.from_yaml(ROOT / "configs/cam4_model.yaml")

    assert config.history_core_boundary == "after_scheme"
    assert config.history_core_scheme == "sima_state_diagnostics"
    config.validate()


def test_suite_can_preserve_signed_zero_from_vector_mapping() -> None:
    config = ModelConfig.from_yaml(ROOT / "configs/adiabatic_model.yaml")

    assert config.canonicalize_resting_wind_zero is False
    assert ModelConfig.from_mapping(config.as_dict()) == config


def test_model_config_accepts_suite_namelist_overrides() -> None:
    config = ModelConfig.from_mapping(
        {
            **ModelConfig().as_dict(),
            "namelist_overrides": {
                "MUSICA_CCPP": {
                    "MICM_SOLVER_TYPE": "Rosenbrock",
                }
            },
        }
    )

    assert config.namelist_overrides == {
        "musica_ccpp": {"micm_solver_type": "Rosenbrock"}
    }
    assert ModelConfig.from_mapping(config.as_dict()) == config


def test_namelist_overrides_create_a_missing_group() -> None:
    config = ModelConfig.from_yaml(ROOT / "configs/adiabatic_model.yaml")
    config = config.with_overrides(
        namelist_overrides={
            "new_runtime_group": {"configuration_file": "/tmp/config.json"}
        }
    )

    result = read_atm_in(
        ROOT
        / "reference/cases/FADIAB_ne3pg3_gnu_24x50/CaseDocs/atm_in",
        config,
    )

    assert result["namelist"]["new_runtime_group"][
        "configuration_file"
    ] == "/tmp/config.json"


def test_atm_in_path_expands_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FREECAM_TEST_ROOT", str(ROOT))
    config = ModelConfig.from_yaml(ROOT / "configs/adiabatic_model.yaml")

    result = read_atm_in(
        "$FREECAM_TEST_ROOT/reference/cases/"
        "FADIAB_ne3pg3_gnu_24x50/CaseDocs/atm_in",
        config,
    )

    assert Path(result["ncdata"]).is_file()


def test_restart_modes_require_a_checkpoint_but_not_an_analytic_ic() -> None:
    continued = ModelConfig(
        run_type="continue",
        restart_path="restart",
        analytic_ic_type="not-used-for-restart",
    )
    continued.validate()
    with pytest.raises(ConfigurationError, match="restart_path is required"):
        ModelConfig(run_type="branch").validate()
