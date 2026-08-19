"""Runtime-writable CAM module tunables: table, binding, collective writes."""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from freecam.pi_cam import (
    InMemoryBoundaryProvider,
    PICAMConfig,
    PICAMDriver,
    RecordingCAMBackend,
)
from freecam.pi_cam.errors import PICAMConfigurationError, PICAMStateError
from freecam.pi_cam.parameters import (
    PICAMModuleParameterRegistry,
    PICAMRuntimeParameter,
    load_runtime_parameters,
)

_ATM_IN = """&cldfrc_nl
 cldfrc_rhminl\t\t= 0.870D0
/
&zmconv_nl
 zmconv_c0_lnd\t\t=  0.0059D0
/
"""


def _driver(tmp_path: Path) -> PICAMDriver:
    config = PICAMConfig(
        case_name="unit-parameters",
        source_root=Path("/tmp/source"),
        mpi_size=1,
        stop_n=2,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "atm_in").write_text(_ATM_IN)
    boundary = InMemoryBoundaryProvider(
        {(step, 0): {"sst": np.full((2,), 280.0 + step)} for step in range(4)}
    )
    return PICAMDriver(
        config,
        boundary,
        RecordingCAMBackend(),
        rank=0,
        size=1,
        run_dir=run_dir,
    )


def _fake_storage() -> dict[str, ctypes.c_double]:
    return {
        "fake_c0_lnd": ctypes.c_double(0.0059),
        "fake_rhminl": ctypes.c_double(0.870),
        "fake_rhminl_const": ctypes.c_double(0.870),
    }


def _table() -> dict[str, PICAMRuntimeParameter]:
    return {
        "zmconv_c0_lnd": PICAMRuntimeParameter(
            name="zmconv_c0_lnd",
            dtype="float64",
            symbols=("fake_c0_lnd",),
            workflow_action="cam_run1.deep_convection",
            readers="",
            notes="",
        ),
        "cldfrc_rhminl": PICAMRuntimeParameter(
            name="cldfrc_rhminl",
            dtype="float64",
            symbols=("fake_rhminl", "fake_rhminl_const"),
            workflow_action="cam_run1.cloud_macro_microphysics",
            readers="",
            notes="",
        ),
    }


def _bound_registry(tmp_path: Path):
    driver = _driver(tmp_path)
    driver.initialize()
    storage = _fake_storage()
    driver.backend.module_symbol = (
        lambda symbol, dtype: storage.get(symbol)
    )
    registry = PICAMModuleParameterRegistry(driver)
    registry.bind(table=_table())
    return driver, registry, storage


def test_the_shipped_table_is_admissible() -> None:
    table = load_runtime_parameters()
    assert len(table) == 14
    assert all(entry.dtype == "float64" for entry in table.values())
    assert table["cldfrc_rhminl"].symbols == (
        "cloud_fraction_mp_rhminl_",
        "cldwat2m_macro_mp_rhminl_const_",
    )
    assert table["zmconv_c0_lnd"].symbols == ("zm_conv_mp_c0_lnd_",)


def test_table_loading_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "runtime_parameters.yaml"
    bad.write_text("schema_version: 2\nparameters: {x: {dtype: float64}}\n")
    with pytest.raises(PICAMConfigurationError, match="schema_version"):
        load_runtime_parameters(bad)
    bad.write_text(
        "schema_version: 1\n"
        "parameters:\n  x:\n    dtype: float32\n    symbols: [s_]\n"
    )
    with pytest.raises(PICAMConfigurationError, match="dtype"):
        load_runtime_parameters(bad)
    bad.write_text(
        "schema_version: 1\n"
        "parameters:\n  x:\n    dtype: float64\n    symbols: []\n"
    )
    with pytest.raises(PICAMConfigurationError, match="no symbols"):
        load_runtime_parameters(bad)
    bad.write_text(
        "schema_version: 1\n"
        "parameters:\n  x:\n    dtype: float64\n    symbols: [s_]\n"
        "    surprising: 1\n"
    )
    with pytest.raises(PICAMConfigurationError, match="unsupported keys"):
        load_runtime_parameters(bad)


def test_binding_verifies_against_the_namelist(tmp_path: Path) -> None:
    driver, registry, _ = _bound_registry(tmp_path)
    assert registry.names() == ("cldfrc_rhminl", "zmconv_c0_lnd")
    assert registry.value("zmconv_c0_lnd") == 0.0059
    field = driver.pool["parameters.zmconv_c0_lnd"]
    assert field.shape == ()
    contract = driver.pool.contract("parameters.zmconv_c0_lnd")
    assert contract.writable is False
    assert contract.restart is False
    assert contract.output is False
    assert contract.owner == "native"
    # Views into Fortran-owned storage do not count as Python state bytes.
    assert driver.pool.nbytes == sum(
        driver.pool[name].nbytes
        for name in driver.pool
        if not name.startswith("parameters.")
    )


def test_binding_refuses_a_mismatched_address(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    storage = _fake_storage()
    storage["fake_c0_lnd"].value = 123.0  # wrong storage for this namelist
    driver.backend.module_symbol = (
        lambda symbol, dtype: storage.get(symbol)
    )
    registry = PICAMModuleParameterRegistry(driver)
    registry.bind(table=_table())
    assert "zmconv_c0_lnd" not in registry
    assert "unverified address" in registry.unavailable["zmconv_c0_lnd"]
    with pytest.raises(PICAMConfigurationError, match="unverified"):
        registry.set("zmconv_c0_lnd", 0.0075)


def test_binding_requires_symbol_and_namelist_presence(tmp_path: Path) -> None:
    driver = _driver(tmp_path)
    driver.initialize()
    storage = _fake_storage()
    del storage["fake_rhminl_const"]
    driver.backend.module_symbol = (
        lambda symbol, dtype: storage.get(symbol)
    )
    table = _table()
    table["uwshcu_rpen"] = PICAMRuntimeParameter(
        name="uwshcu_rpen",
        dtype="float64",
        symbols=("fake_rpen",),
        workflow_action="cam_run1.shallow_convection",
        readers="",
        notes="",
    )
    registry = PICAMModuleParameterRegistry(driver)
    registry.bind(table=table)
    assert "not present" in registry.unavailable["cldfrc_rhminl"]
    assert "atm_in" in registry.unavailable["uwshcu_rpen"]
    assert registry.names() == ("zmconv_c0_lnd",)


def test_set_writes_every_storage_location_atomically(tmp_path: Path) -> None:
    driver, registry, storage = _bound_registry(tmp_path)
    report = registry.set("cldfrc_rhminl", 0.9)
    assert report == {
        "name": "cldfrc_rhminl",
        "previous": 0.870,
        "value": 0.9,
        "symbols_written": ["fake_rhminl", "fake_rhminl_const"],
    }
    assert storage["fake_rhminl"].value == 0.9
    assert storage["fake_rhminl_const"].value == 0.9
    assert driver.pool["parameters.cldfrc_rhminl"][()] == 0.9
    assert registry.overrides() == {"cldfrc_rhminl": (0.870, 0.9)}
    assert registry.values()["zmconv_c0_lnd"] == 0.0059


def test_set_rejects_invalid_values(tmp_path: Path) -> None:
    _, registry, storage = _bound_registry(tmp_path)
    for bad in (True, "0.9", float("nan"), float("inf"), None):
        with pytest.raises(PICAMConfigurationError):
            registry.set("zmconv_c0_lnd", bad)
    assert storage["fake_c0_lnd"].value == 0.0059


def test_set_leaves_storage_on_rank_divergence(tmp_path: Path) -> None:
    driver, registry, storage = _bound_registry(tmp_path)

    class DivergentComm:
        rank = 0
        size = 2

        @staticmethod
        def allgather(value):
            if isinstance(value, str):
                return [value, "a-different-hex"]
            return [value, value]

        @staticmethod
        def barrier():
            return None

    driver.comm = DivergentComm()
    with pytest.raises(PICAMStateError, match="differ across"):
        registry.set("zmconv_c0_lnd", 0.0075)
    assert storage["fake_c0_lnd"].value == 0.0059


def test_a_backend_without_module_storage_fails_closed(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    driver.initialize()  # RecordingCAMBackend exposes no module storage
    assert len(driver.module_parameters) == 0
    assert "*" in driver.module_parameters.unavailable
    with pytest.raises(PICAMConfigurationError, match="module storage"):
        driver.set_module_parameter("zmconv_c0_lnd", 0.0075)


def test_for_action_filters_by_workflow_action(tmp_path: Path) -> None:
    _, registry, _ = _bound_registry(tmp_path)
    assert registry.for_action("cam_run1.deep_convection") == (
        "zmconv_c0_lnd",
    )
    assert registry.for_action("cam_run1.radiation") == ()
