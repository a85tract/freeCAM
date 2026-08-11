import json
from pathlib import Path

from freecam.pi_cam.physics_catalog import PICAMPhysicsCatalog
from freecam.pi_cam.plan import PICAMStepPlan
from freecam.pi_cam.process_support import build_process_support_report


PROJECT = Path(__file__).resolve().parents[2]


def _read(name: str):
    return json.loads((PROJECT / name).read_text())


def test_all_former_catalog_only_processes_have_compiled_statepool_adapters() -> None:
    report = build_process_support_report(
        catalog=PICAMPhysicsCatalog.load_default(),
        runtime_records=PICAMStepPlan.default().describe(),
        generation=_read("validation/pi_cam_in_module_adapter_generation.json"),
        compilation=_read("validation/pi_cam_in_module_adapter_validation.json"),
        loading=_read("validation/pi_cam_process_device_loading.json"),
        runtime_validation=_read("validation/pi_cam_catalog_process_50step.json"),
        bfb_validation=_read(
            "validation/pi_cam_catalog_process_vs_oracle_50step_bfb.json"
        ),
    )

    assert report["formerly_catalog_only_interfaces"] == 262
    assert report["workflow_source_overlap"] == 14
    assert report["adapters_generated"] == 262
    assert report["adapters_compiled"] == 262
    assert report["statepool_pointer_contracts"] == 262
    assert report["current_case_loadable"] == 226
    assert report["configuration_specific"] == 36
    assert report["all_catalog_only_interfaces_supported"] is True
    assert report["representative_runtime_validation"]["passed"] is True
    assert report["representative_bfb_validation"]["bfb"] is True
    assert all(record["supported"] for record in report["processes"])
