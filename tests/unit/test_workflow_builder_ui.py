"""Driver.ui() and `freecam ui`: the page starts; the model does not."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

import urllib.request  # noqa: E402

from freecam.pi_cam.workflow_builder.service import WorkflowService  # noqa: E402
from freecam.pi_cam.workflow_builder.ui import WorkflowUI, launch_ui  # noqa: E402

from _workflow_builder_fakes import FakeDriver  # noqa: E402


def test_the_page_serves_without_starting_the_model() -> None:
    driver = FakeDriver()
    ui = launch_ui(driver, port=None)
    try:
        assert ui.url.startswith("http://127.0.0.1:") and "token=" in ui.url
        assert driver.initialized == 0 and driver._session is None
        request = urllib.request.Request(ui.url.split("?")[0] + "api/state", headers={"X-FreeCAM-Token": ui.service.token})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
        html = ui._repr_html_()
        assert ui.url in html and "first Run" in html
    finally:
        ui.close()
    assert driver.initialized == 0


def test_closing_the_page_leaves_the_model_unless_asked() -> None:
    driver = FakeDriver()
    ui = WorkflowUI(WorkflowService(driver), "127.0.0.1", 0)
    ui.close()                                   # never started: nothing to stop
    assert not driver.closed
    ui.close(close_model=True)
    assert not driver.closed                     # it was never initialized either


def test_the_console_entry_dispatches_ui(capsys) -> None:
    from freecam.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["ui", "--help"])
    assert exit_info.value.code == 0
    assert "Serve the Workflow Builder" in capsys.readouterr().out
