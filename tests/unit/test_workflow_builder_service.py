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
    assert service.driver.lengthened == [3]                            # the model set up for the steps asked
    events = client.get("/api/events?since=0", headers=_headers(service)).json()["events"]
    messages = " ".join(e["message"] for e in events)
    assert "initializing the model" in messages and "running 3 step(s) from step 0" in messages

    # a second Run continues from the current step and does not re-initialize
    again = client.post("/api/run", headers=_headers(service), json={"document": document, "steps": 2, "confirm_resources": False})
    assert again.status_code == 200
    final = _wait(service, {"completed", "error"})
    assert final["step"] == 5 and service.driver.initialized == 1
    assert service.driver.lengthened == [3]                            # a started model keeps its length


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


class _RecordingDriver(FakeDriver):
    """A driver that can record its state, as ``freecam.Driver`` can."""

    def __init__(self, *args, state_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_state = None
        self.state_dir = state_dir


def test_the_page_keeps_the_state_in_memory_before_the_model_starts(tmp_path):
    driver = _RecordingDriver(nsteps=3, run_dir=tmp_path / "run")
    service = WorkflowService(driver, generated_dir=tmp_path / "generated")
    from freecam.pi_cam.state_record import BUILDER_FIELDS

    assert driver.record_state == {"store": "memory", "fields": list(BUILDER_FIELDS), "action_steps": []}
    assert service.run_payload()["globe"] == {"enabled": True, "dir": None, "memory": True, "keep_steps": 1000,
                                              "ready": False}
    driver.record_state = {"dir": None, "fields": ["T"], "every": 2}
    WorkflowService(driver, generated_dir=tmp_path / "generated")
    assert driver.record_state["store"] == "memory" and driver.record_state["fields"] == ["T"]   # the caller's stay
    driver.record_state = {"store": "files"}
    WorkflowService(driver, generated_dir=tmp_path / "generated")
    assert driver.record_state == {"store": "files", "flush_every": 1, "action_steps": []}
    driver.record_state = {"dir": tmp_path / "state"}                  # a directory of the caller's: files
    WorkflowService(driver, generated_dir=tmp_path / "generated")
    assert "store" not in driver.record_state and driver.record_state["flush_every"] == 1
    off = _RecordingDriver(nsteps=3, run_dir=tmp_path / "run")
    WorkflowService(off, generated_dir=tmp_path / "generated", globe=False)
    assert off.record_state is None


def test_the_globe_routes_need_the_token_and_a_recording(client, service):
    assert client.get("/globe/api/meta").status_code == 401
    refused = client.get("/globe/api/meta", headers=_headers(service))
    assert refused.status_code == 409 and "records no state" in refused.json()["detail"]
    assert service.run_payload()["globe"]["enabled"] is False
    page = client.get("/globe/")
    assert page.status_code == 200 and "api/meta" in page.text and "/*COASTLINE*/null" not in page.text


def test_the_globe_serves_the_run_it_records(tmp_path, monkeypatch):
    from test_pi_cam_state_record import record

    state = tmp_path / "state"
    driver = _RecordingDriver(nsteps=3, run_dir=tmp_path / "run", state_dir=state)
    driver.record_state = {"store": "files"}
    service = WorkflowService(driver, generated_dir=tmp_path / "generated")
    client = TestClient(create_app(service, static_dir=service._generated_dir))
    waiting = client.get("/globe/api/meta", headers=_headers(service))
    assert waiting.status_code == 409 and "no step recorded yet" in waiting.json()["detail"]
    record(state, monkeypatch)
    assert service.run_payload()["globe"]["ready"] is True
    meta = client.get("/globe/api/meta", headers=_headers(service))
    assert meta.status_code == 200 and meta.json()["step_frames"] == [0, 1, 2]
    # the globe opened in a tab of its own carries the token in its address
    level = client.get("/globe/api/level", params={"token": service.token, "kind": "steps", "i": 2, "field": "T", "lev": 1})
    assert level.status_code == 200 and level.headers["content-type"] == "application/octet-stream"
    assert len(level.content) == 4 * meta.json()["columns"]
    assert client.get("/globe/api/level", params={"token": "wrong", "field": "T", "lev": 1}).status_code == 401
    bad = client.get("/globe/api/level", headers=_headers(service), params={"field": "T"})
    assert bad.status_code == 400


class _LiveDriver(_RecordingDriver):
    """A driver whose ranks keep their state in memory, answered by two fake recorders."""

    def __init__(self, *args, recorders=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorders = recorders
        if recorders is not None:                                      # a model started keeping it
            self._session = object()
            self.record_state = {"store": "memory", "action_steps": []}
        self.asked: list[str] = []

    @property
    def state_live(self):
        return self._session is not None

    def query_state(self, request):
        from test_pi_cam_state_record import ask

        self.asked.append(request["q"])
        return ask(self.recorders, request)


def test_the_globe_asks_the_ranks_of_a_model_keeping_its_state_in_memory(tmp_path, monkeypatch):
    from test_pi_cam_state_record import record_in_memory

    starting = _LiveDriver(nsteps=3, run_dir=tmp_path / "run")
    service = WorkflowService(starting, generated_dir=tmp_path / "generated")
    service._run.state = "initializing"
    client = TestClient(create_app(service, static_dir=service._generated_dir))
    waiting = client.get("/globe/api/meta", headers=_headers(service))
    assert waiting.status_code == 409 and "the model is starting" in waiting.json()["detail"]

    driver = _LiveDriver(nsteps=3, run_dir=tmp_path / "run", recorders=record_in_memory(monkeypatch))
    service = WorkflowService(driver, generated_dir=tmp_path / "generated")
    client = TestClient(create_app(service, static_dir=service._generated_dir))
    assert service.run_payload()["globe"]["ready"] is True and service.run_payload()["globe"]["memory"] is True
    meta = client.get("/globe/api/meta", headers=_headers(service))
    assert meta.status_code == 200 and meta.json()["live"] is True and meta.json()["step_frames"] == [0, 1, 2]
    level = client.get("/globe/api/level", headers=_headers(service),
                       params={"kind": "steps", "i": 2, "field": "T", "lev": 1})
    assert level.status_code == 200 and len(level.content) == 4 * 6
    assert driver.asked == ["layout", "frames", "frame"]               # nothing but what the page shows
    # a model closed and started again is another: its state is asked for afresh
    driver._session = object()
    client.get("/globe/api/meta", headers=_headers(service))
    assert driver.asked[-2:] == ["layout", "frames"]


def test_the_globe_s_fields_are_chosen_before_the_model_starts(tmp_path):
    driver = _RecordingDriver(nsteps=3, run_dir=tmp_path / "run")
    service = WorkflowService(driver, generated_dir=tmp_path / "generated")
    client = TestClient(create_app(service, static_dir=service._generated_dir))
    assert client.get("/api/globe/options").status_code == 401
    options = client.get("/api/globe/options", headers=_headers(service)).json()
    assert options["editable"] is True and options["memory"] is True and "PRECC" in options["fields"]
    offered = {field["name"]: field for field in options["available"]}
    assert offered["PRECL"]["source"] == "cam_out.precl" and offered["PRECL"]["group"] == "surface"
    assert offered["Q"]["source"] == "phys_state.q:Q" and offered["DTDT"]["units"] == "K/day"
    chosen = client.put("/api/globe/options", headers=_headers(service),
                        json={"fields": ["T", "TS", "cam_in.lwup"], "every": 2, "action_steps": "24, 30-31"})
    assert chosen.status_code == 200
    assert driver.record_state["fields"] == ["T", "TS", "cam_in.lwup"] and driver.record_state["every"] == 2
    assert driver.record_state["action_steps"] == [24, 30, 31] and driver.record_state["store"] == "memory"
    for wrong in ({"fields": []}, {"fields": ["temperature"]}, {"every": 0}, {"action_steps": "3-"},
                  {"feilds": ["T"]}):
        refused = client.put("/api/globe/options", headers=_headers(service), json=wrong)
        assert refused.status_code == 409, wrong
    assert driver.record_state["fields"] == ["T", "TS", "cam_in.lwup"]          # a refusal changes nothing
    driver._session = object()                                                  # the model starts
    assert client.get("/api/globe/options", headers=_headers(service)).json()["editable"] is False
    late = client.put("/api/globe/options", headers=_headers(service), json={"fields": ["T"]})
    assert late.status_code == 409 and "Close model" in late.json()["detail"]
