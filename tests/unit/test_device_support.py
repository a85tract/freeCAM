from pathlib import Path

from pycam_sima import DeviceSupportMatrix


ROOT = Path(__file__).resolve().parents[2]


def test_support_matrix_accounts_for_every_active_scheme():
    matrix = DeviceSupportMatrix.discover(ROOT)
    summary = matrix.summary()
    assert summary["scheme_count"] == 155
    assert summary["connectors_generated"] == 155
    assert sum(summary["status_counts"].values()) == 155
    assert summary["native_device_ready"] == 100
    assert summary["python_host_service_ready"] == 29
    assert summary["executable_connector_count"] == 129
    assert summary["external_service_required"] == 26
    assert all(item.status != "connector_missing" for item in matrix.records)
