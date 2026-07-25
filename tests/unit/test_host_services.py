from pathlib import Path

import numpy as np

from pycam_sima.model.ccpp_suite import CCPPDeviceHost, CCPPSuitePlan
from pycam_sima.model.contracts import FieldContract
from pycam_sima.model.device_catalog import DeviceCatalog
from pycam_sima.model.devices import DeviceRegistry
from pycam_sima.model.host_services import HostServiceRegistry
from pycam_sima.model.state import StatePool


ROOT = Path(__file__).resolve().parents[2]


def test_python_history_service_replaces_cam_history_sink(
    tmp_path: Path,
):
    catalog = DeviceCatalog.discover(ROOT)
    services = HostServiceRegistry.from_catalog(
        catalog, suite="kessler"
    )
    assert "kessler_diagnostics" in services.process_names
    pool = StatePool(
        {"nphys_local": 3},
        contracts=(
            FieldContract(
                "rain",
                "float64",
                ("nphys_local",),
                "in",
                "history",
                "m s-1",
                ccpp_standard_name=(
                    "total_precipitation_rate_at_surface"
                ),
            ),
        ),
        alias_rules=(),
    )
    pool.set("rain", np.array([1.0, 2.0, 4.0]))
    plan = CCPPSuitePlan.from_xml(
        ROOT
        / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/"
        "suite_kessler.xml"
    )
    host = CCPPDeviceHost(
        pool,
        DeviceRegistry(tmp_path),
        plan,
        host_services=services,
    )
    assert host.run_scheme("kessler_diagnostics")
    event = services.events()[-1]
    observation = event["observations"][0]
    assert event["scheme"] == "kessler_diagnostics"
    assert observation["minimum"] == 1.0
    assert observation["maximum"] == 4.0
    assert observation["mean"] == 7.0 / 3.0


def test_history_services_can_be_selected_for_a_custom_suite_by_process():
    catalog = DeviceCatalog.discover(ROOT)
    services = HostServiceRegistry.from_catalog(
        catalog,
        processes=("kessler_diagnostics",),
    )
    assert "kessler_diagnostics" in services.process_names
    assert all(
        process.startswith("kessler_diagnostics")
        for process in services.process_names
    )
