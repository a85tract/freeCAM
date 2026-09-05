"""The local service: what it accepts, what it refuses, what it does with a Driver."""

import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from freecam.pi_cam.workflow_builder import load_catalog  # noqa: E402
from freecam.pi_cam.workflow_builder.service import WorkflowService, create_app  # noqa: E402

from _workflow_builder_fakes import FakeDriver  # noqa: E402


@pytest.fixture(scope="module")
def snapshot():
    return load_catalog()[2]


@pytest.fixture
def service(tmp_path):
    driver = FakeDriver(nsteps=3, run_dir=tmp_path / "run")
    return WorkflowService(driver, generated_dir=tmp_path / "generated")


@pytest.fixture
def client(service):
    return TestClient(create_app(service, static_dir=service._generated_dir))   # no built page: API only


def _headers(service, origin=None):
    headers = {"X-FreeCAM-Token": service.token}
    if origin:
        headers["Origin"] = origin
    return headers


def _wait(service, states, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = service.run_payload()
        if payload["state"] in states:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"run did not reach {states}: {service.run_payload()}")


def test_requests_without_the_token_or_from_another_origin_are_refused(client, service) -> None:
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"X-FreeCAM-Token": "wrong"}).status_code == 401
    assert client.get("/api/state", headers=_headers(service, origin="http://evil.example")).status_code == 403
    assert client.get("/api/state", headers=_headers(service, origin="http://testserver")).status_code == 200


def test_the_state_carries_the_snapshot_and_the_driver_s_case(client, service) -> None:
    payload = client.get("/api/state", headers=_headers(service)).json()
    assert payload["mode"] == "local" and payload["draft"] is None
    assert payload["snapshot"]["catalog_hash"] == service.snapshot["catalog_hash"]
    assert payload["case"] == "PI-atm" and payload["nsteps"] == 3
    assert payload["resources"]["ranks"] == 512 and payload["resources"]["account_set"]
    assert payload["driver_initialized"] is False
    assert payload["run"]["state"] == "idle"


def test_the_draft_is_kept_for_a_refresh(client, service) -> None:
    document = dict(service.default.to_payload())
    document["nsteps"] = 7
    saved = client.put("/api/draft", headers=_headers(service), json={"document": document}).json()
    assert saved["workflow_hash"] != service.default.workflow_hash
    assert client.get("/api/state", headers=_headers(service)).json()["draft"]["nsteps"] == 7
    bad = client.put("/api/draft", headers=_headers(service), json={"document": {"nodes": "x"}})
    assert bad.status_code == 422


def test_the_local_check_parses_python(client, service) -> None:
    session = service.session
    session.apply({"operation": "add_python", "name": "heating", "after": "cam_run1.dry_adjustment"})
    session.apply({"operation": "configure", "node_id": "python:heating",
                   "configuration": {"python_source": "class Heating(fc.Physics):\n    name = 'heating'\n    def run(self, state, context)\n        pass\n"}})
    report = client.post("/api/validate", headers=_headers(service), json={"document": session.document.to_payload()}).json()
    assert report["level"] == "local" and report["status"] == "error"
    assert any(issue["code"] == "python-syntax" for issue in report["issues"])


def test_generate_stores_what_the_browser_produced(client, service, tmp_path) -> None:
    document = service.default.to_payload()
    saved = client.post("/api/generate", headers=_headers(service),
                        json={"document": document, "artifacts": {"script": "print('hi')\n", "workflow": json.dumps(document)}}).json()
    assert saved["workflow_hash"] == document["workflow_hash"]
    files = {k: __import__("pathlib").Path(v) for k, v in saved["files"].items()}
    assert files["script"].read_text() == "print('hi')\n"
    assert files["script"].parent.name == document["workflow_hash"][:12]


def test_the_first_run_needs_the_resources_confirmed_then_initializes_applies_and_runs(client, service) -> None:
    document = service.default.to_payload()
    refused = client.post("/api/run", headers=_headers(service), json={"document": document, "steps": 3})
    assert refused.status_code == 409 and "confirm" in refused.json()["detail"]

    started = client.post("/api/run", headers=_headers(service), json={"document": document, "steps": 3, "confirm_resources": True})
    assert started.status_code == 200
    final = _wait(service, {"completed", "error"})
    assert final["state"] == "completed", final
    assert final["step"] == 3 and final["target_step"] == 3
    assert final["job_id"] == "12345.fake"
    assert final["applied_hash"] == document["workflow_hash"]
    assert service.driver.initialized == 1
    events = client.get("/api/events?since=0", headers=_headers(service)).json()["events"]
    messages = " ".join(e["message"] for e in events)
    assert "initializing the model" in messages and "running 3 step(s) from step 0" in messages

    # a second Run continues from the current step and does not re-initialize
    again = client.post("/api/run", headers=_headers(service), json={"document": document, "steps": 2, "confirm_resources": False})
    assert again.status_code == 200
    final = _wait(service, {"completed", "error"})
    assert final["step"] == 5 and service.driver.initialized == 1


def test_a_run_with_structural_errors_is_refused_before_anything_starts(client, service) -> None:
    session = service.session
    session.apply({"operation": "enable", "node_id": "cam_run2.tracers_and_chemistry"})
    response = client.post("/api/run", headers=_headers(service),
                           json={"document": session.document.to_payload(), "steps": 1, "confirm_resources": True})
    assert response.status_code == 409 and "run twice" in response.json()["detail"]
    assert service.driver.initialized == 0


def test_a_change_that_needs_a_restart_ends_the_run_with_that_message(client, service) -> None:
    document = service.default.to_payload()
    client.post("/api/run", headers=_headers(service), json={"document": document, "steps": 1, "confirm_resources": True})
    _wait(service, {"completed"})
    session = service.session
    session.apply({"operation": "set_namelist", "namelist": {"cldfrc_rhminl": 0.9}})
    client.post("/api/run", headers=_headers(service), json={"document": session.document.to_payload(), "steps": 1})
    final = _wait(service, {"error", "completed"})
    assert final["state"] == "error" and "restart required" in final["message"]


def test_stop_and_close_follow_the_model_s_state(client, service) -> None:
    assert client.post("/api/stop", headers=_headers(service), json={}).status_code == 409
    document = service.default.to_payload()
    client.post("/api/run", headers=_headers(service), json={"document": document, "steps": 1, "confirm_resources": True})
    _wait(service, {"completed"})
    closed = client.post("/api/close", headers=_headers(service), json={}).json()
    assert closed["state"] == "closed" and service.driver.closed
    assert client.get("/api/state", headers=_headers(service)).json()["driver_initialized"] is False


def test_without_the_built_page_the_root_says_how_to_build_it(client, service) -> None:
    response = client.get("/")
    assert response.status_code == 503 and "npm run build" in response.json()["detail"]
