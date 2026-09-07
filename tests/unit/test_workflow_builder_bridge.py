"""Applying a document to a live model: the bridge, and the code that spells it out."""

import json
import shutil
import types

import pytest

import freecam
from freecam.pi_cam.workflow_builder import WorkflowEditSession, load_catalog, python_process_template
from freecam.pi_cam.workflow_builder.bridge import ApplyError, RestartRequired, apply_document, instantiate_python
from freecam.pi_cam.workflow_builder import codegen

from _workflow_builder_fakes import FakeDriver


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def _session(catalog):
    document, entries, _ = catalog
    return WorkflowEditSession(document, entries, python_template=python_process_template)


def _edited(session, model_path="models/mmacro_pcond.pt"):
    session.apply({"operation": "move", "node_id": "cam_run1.radiation", "before": "cam_run1.dry_adjustment"})
    session.apply({"operation": "disable", "node_id": "cam_run1.shallow_convection"})
    session.apply({"operation": "configure", "node_id": "cam_run1.deep_convection",
                   "configuration": {"parameters": {"zmconv_c0_lnd": 0.0075}}})
    session.apply({"operation": "configure", "node_id": "cam_run1.cloud_macro_microphysics",
                   "configuration": {"kernels": {"mmacro_pcond": {"kind": "surrogate", "path": model_path}}}})
    session.apply({"operation": "add_python", "name": "heating", "after": "cam_run1.dry_adjustment"})
    session.apply({"operation": "configure", "node_id": "python:heating",
                   "configuration": {"parameters": {"rate": 0.5},
                                     "variables": [{"name": "heating_rate", "like": "T", "units": "K s-1"}]}})
    session.apply({"operation": "set_experimental", "experimental": True})
    session.apply({"operation": "set_nsteps", "nsteps": 4})
    return session.document


def test_the_default_document_asks_nothing_of_the_model(catalog) -> None:
    document, _, _ = catalog
    driver = FakeDriver()
    driver.initialize()
    state = apply_document(driver, document, default=document)
    assert state.log == []
    assert driver.workflow.calls == []


def test_an_edited_document_is_applied_in_the_order_the_code_is_generated_in(catalog) -> None:
    session = _session(catalog)
    document = _edited(session)
    driver = FakeDriver()
    driver.initialize()

    state = apply_document(driver, document, default=catalog[0])

    kinds = [call[0] for call in driver.workflow.calls]
    # fields, then the Python process, then the stage class, then the order, then the tunables
    assert kinds.index("create") < kinds.index("insert")
    assert [c for c in driver.workflow.calls if c[0] == "insert"][0] == ("insert", "heating", None, "dry_adjustment")
    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics

    process_name = CloudMacroMicrophysics.PROCESS_NAME
    stage_insert = [c for c in driver.workflow.calls if c[0] == "insert" and c[1] == process_name]
    assert stage_insert == [("insert", process_name, None, "cloud_macro_microphysics")]
    assert ("disable", "cloud_macro_microphysics") in driver.workflow.calls
    replace = next(c for c in driver.workflow.calls if c[0] == "replace")
    assert "shallow_convection" not in replace[1] and "heating" in replace[1]
    assert replace[1].index("radiation") < replace[1].index("dry_adjustment")
    assert kinds.index("replace") < kinds.index("parameter")
    assert ("parameter", "zmconv_c0_lnd", 0.0075) in driver.workflow.calls
    assert driver.cam.state.created["heating_rate"] == {"like": "T", "units": "K s-1"}
    assert state.python_processes["heating"].rate == 0.5
    assert "cam_run1.cloud_macro_microphysics" in state.stages
    stage = state.stages["cam_run1.cloud_macro_microphysics"]
    assert stage.replacements() == ("mmacro_pcond",)             # bound, not loaded


def test_a_second_application_applies_only_the_difference(catalog) -> None:
    session = _session(catalog)
    document = _edited(session)
    driver = FakeDriver()
    driver.initialize()
    state = apply_document(driver, document, default=catalog[0])
    driver.workflow.calls.clear()

    again = apply_document(driver, document, state)
    assert again.log == [] and driver.workflow.calls == []

    session.apply({"operation": "configure", "node_id": "python:heating", "configuration": {"parameters": {"rate": 0.9}}})
    changed = apply_document(driver, session.document, again)
    assert driver.workflow.calls == [("property", "heating", "rate", 0.9)]
    driver.workflow.calls.clear()

    session.apply({"operation": "configure", "node_id": "python:heating",
                   "configuration": {"python_source": python_process_template("heating", "dry_adjustment").replace("0.0", "1.0")}})
    reloaded = apply_document(driver, session.document, changed)
    assert driver.workflow.calls == [("reload", "heating", "Heating")]
    driver.workflow.calls.clear()

    session.apply({"operation": "remove", "node_id": "python:heating"})
    removed = apply_document(driver, session.document, reloaded)
    assert ("remove", "heating") in driver.workflow.calls
    assert "heating" not in removed.python_processes


def test_what_cannot_change_on_a_live_model_is_refused_not_applied(catalog) -> None:
    session = _session(catalog)
    document = _edited(session)
    driver = FakeDriver()
    driver.initialize()
    state = apply_document(driver, document, default=catalog[0])
    session.apply({"operation": "set_namelist", "namelist": {"cldfrc_rhminl": 0.9}})
    with pytest.raises(RestartRequired, match="namelist"):
        apply_document(driver, session.document, state)
    session.apply({"operation": "set_namelist", "namelist": {}})
    session.apply({"operation": "configure", "node_id": "cam_run1.cloud_macro_microphysics",
                   "configuration": {"kernels": {"mmacro_pcond": {"kind": "surrogate", "path": "other.pt"}}}})
    with pytest.raises(RestartRequired, match="kernel binding"):
        apply_document(driver, session.document, state)


def test_a_python_process_is_built_from_its_source_with_its_properties(catalog) -> None:
    session = _session(catalog)
    session.apply({"operation": "add_python", "name": "heating", "after": "cam_run1.dry_adjustment"})
    session.apply({"operation": "configure", "node_id": "python:heating", "configuration": {"parameters": {"rate": 2.0}}})
    process = instantiate_python(session.document.node("python:heating"))
    assert isinstance(process, freecam.Physics) and process.name == "heating" and process.rate == 2.0
    session.apply({"operation": "configure", "node_id": "python:heating", "configuration": {"parameters": {"nonesuch": 1}}})
    with pytest.raises(ApplyError, match="no property"):
        instantiate_python(session.document.node("python:heating"))
    session.apply({"operation": "configure", "node_id": "python:heating", "configuration": {"parameters": {}, "python_source": "x = 1\n"}})
    with pytest.raises(ApplyError, match="defines no class"):
        instantiate_python(session.document.node("python:heating"))


@pytest.mark.skipif(not codegen.available() or shutil.which("node") is None, reason="the generator bundle or node is not here")
def test_the_generated_script_does_what_the_bridge_does(catalog, tmp_path) -> None:
    """The service and the exported script are two spellings of one application."""

    _, _, snapshot = catalog
    session = _session(catalog)
    document = _edited(session, model_path=str(tmp_path / "m.pt"))
    artifacts = codegen.generate(document, snapshot)

    bridged = FakeDriver()
    bridged.initialize()
    apply_document(bridged, document, default=catalog[0])

    scripted = FakeDriver()

    def driver_factory(**kwargs):
        assert kwargs["case"] == "PI-atm" and kwargs["nsteps"] == 4
        scripted.nsteps = kwargs["nsteps"]
        return scripted

    fake_fc = types.SimpleNamespace(Driver=driver_factory, Physics=freecam.Physics, Property=freecam.Property)
    namespace = {"__name__": "__main__", "fc": fake_fc}
    code = artifacts.script.replace("import freecam as fc\n", "")
    exec(compile(code, "<generated>", "exec"), namespace)      # noqa: S102 -- the generated script, on a fake

    def relevant(calls):
        return [c for c in calls if c[0] != "property"]

    assert relevant(scripted.workflow.calls) == relevant(bridged.workflow.calls)
    assert scripted.initialized == 1 and scripted.closed and scripted.step == 4
    assert scripted.cam.state.created == bridged.cam.state.created
    assert dict(scripted.cam.parameters) == dict(bridged.cam.parameters)


@pytest.mark.skipif(not codegen.available(), reason="the generator bundle or node is not here")
def test_the_bundle_and_python_agree_on_the_hash(catalog) -> None:
    session = _session(catalog)
    document = _edited(session)
    assert codegen.browser_hash(document) == document.workflow_hash
