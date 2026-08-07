from pathlib import Path

from freecam import DeviceSupportMatrix


ROOT = Path(__file__).resolve().parents[2]


def test_support_matrix_accounts_for_every_active_scheme():
    matrix = DeviceSupportMatrix.discover(ROOT)
    summary = matrix.summary()
    assert summary["scheme_count"] == 155
    assert summary["connectors_generated"] == 155
    assert sum(summary["status_counts"].values()) == 155
    assert summary["native_device_ready"] == 121
    assert summary["python_host_service_ready"] == 30
    assert summary["executable_connector_count"] == 151
    assert summary["external_service_required"] == 4
    # Standalone connector buildability and complete-model provider coverage
    # are deliberately separate metrics.  CAM component services already
    # provide several connectors that should not be linked into a device.
    assert summary["runtime_executable_scheme_count"] == 155
    assert summary["runtime_unresolved_scheme_count"] == 0
    assert summary["runtime_occurrence_count"] == 340
    assert summary["runtime_ready_occurrence_count"] == 340
    assert summary["runtime_unresolved_occurrence_count"] == 0
    assert summary["suite_runtime"]["adiabatic"]["ready"]
    assert summary["suite_runtime"]["held_suarez_1994"]["ready"]
    assert summary["suite_runtime"]["kessler"]["ready"]
    assert summary["suite_runtime"]["tj2016"]["ready"]
    assert all(
        suite["ready"] for suite in summary["suite_runtime"].values()
    )
    assert all(item.status != "connector_missing" for item in matrix.records)
