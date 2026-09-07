"""What the checks catch, and what they refuse to claim."""

import pytest

from freecam.pi_cam.workflow_builder import (
    WorkflowEditSession,
    load_catalog,
    python_process_template,
    validate_document,
)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture
def session(catalog):
    document, entries, _ = catalog
    return WorkflowEditSession(document, entries, python_template=python_process_template)


def _codes(report):
    return sorted(issue.code for issue in report.issues)


def _check(session, catalog, level="browser"):
    document, entries, snapshot = catalog
    return validate_document(session.document, default=document, catalog=entries,
                             level=level, catalog_version=snapshot["catalog_hash"])


def test_a_parent_stage_and_its_leaves_may_not_both_run(session, catalog) -> None:
    session.apply({"operation": "enable", "node_id": "cam_run2.tracers_and_chemistry"})
    report = _check(session, catalog)
    assert report.status == "error"
    assert "parent-and-leaf" in _codes(report)
    culprits = {issue.node_id for issue in report.issues if issue.code == "parent-and-leaf"}
    assert "cam_run2.chemistry_tendencies_leaf" in culprits


def test_changing_the_scientific_order_needs_experimental(session, catalog) -> None:
    session.apply({"operation": "move", "node_id": "cam_run1.radiation", "before": "cam_run1.dry_adjustment"})
    report = _check(session, catalog)
    assert "experimental-required" in _codes(report) and report.status == "error"
    session.apply({"operation": "set_experimental", "experimental": True})
    report = _check(session, catalog)
    assert report.status == "warning" and "experimental" in _codes(report)
    assert report.checks["order_changed"] is True


def test_removing_or_replacing_a_physical_process_needs_experimental(session, catalog) -> None:
    session.apply({"operation": "remove", "node_id": "cam_run1.radiation"})
    assert "experimental-required" in _codes(_check(session, catalog))
    session.apply({"operation": "undo"})
    session.apply({"operation": "replace", "node_id": "cam_run1.radiation", "name": "my_radiation"})
    report = _check(session, catalog)
    assert "experimental-required" in _codes(report)
    assert report.checks["processes_replaced_or_removed"] is True


def test_adding_a_python_process_in_place_does_not_need_experimental(session, catalog) -> None:
    session.apply({"operation": "add_python", "name": "heating", "after": "cam_run1.dry_adjustment"})
    browser = _check(session, catalog)
    assert browser.status == "valid", browser.to_payload()
    assert any(issue.code == "python-syntax" and issue.severity == "info" for issue in browser.issues)
    assert browser.checks["not_verified"]                     # reads/writes known only at run time
    local = _check(session, catalog, level="local")
    assert local.status == "valid"


def test_the_local_level_parses_python_and_checks_the_declared_name(session, catalog) -> None:
    session.apply({"operation": "add_python", "name": "heating", "after": "cam_run1.dry_adjustment"})
    session.apply({"operation": "configure", "node_id": "python:heating",
                   "configuration": {"python_source": "class Heating(fc.Physics):\n    name = 'heating'\n    def run(self, state, context)\n        pass\n"}})
    assert "python-syntax" in _codes(_check(session, catalog, level="local"))
    assert _check(session, catalog).status == "valid"          # the browser cannot parse it
    session.apply({"operation": "configure", "node_id": "python:heating",
                   "configuration": {"python_source": "class Heating(fc.Physics):\n    name = 'other'\n    def run(self, state, context):\n        pass\n"}})
    assert "python-name" in _codes(_check(session, catalog, level="local"))


def test_a_kernel_binding_is_offered_only_where_the_runner_covers_it(session, catalog) -> None:
    from dataclasses import replace

    from freecam.pi_cam.workflow_builder.capabilities import kernel_capabilities

    # every exposed kernel has a runner now; a capability table without one shows the refusal
    session.apply({"operation": "configure", "node_id": "cam_run1.radiation",
                   "configuration": {"kernels": {"rad_rrtmg_sw": {"kind": "surrogate", "path": "m.pt"}}}})
    document, entries, snapshot = catalog
    uncovered = [replace(c, bindable=False, validated=False, reason="no runner pauses here")
                 if c.kernel == "rad_rrtmg_sw" else c for c in kernel_capabilities()]
    report = validate_document(session.document, default=document, catalog=entries, level="browser",
                               catalog_version=snapshot["catalog_hash"], capabilities=uncovered)
    assert "kernel-not-bindable" in _codes(report) and report.status == "error"
    # a pause that has not passed its gate is a warning, not an error
    pending = [replace(c, validated=False, reason="bindable, no gate yet") if c.kernel == "rad_rrtmg_sw" else c
               for c in kernel_capabilities()]
    report = validate_document(session.document, default=document, catalog=entries, level="browser",
                               catalog_version=snapshot["catalog_hash"], capabilities=pending)
    assert "kernel-not-validated" in _codes(report) and "kernel-not-bindable" not in _codes(report)
    # every exposed kernel's pause has passed its gate: the binding itself is the only finding
    report = _check(session, catalog)
    assert "kernel-not-validated" not in _codes(report) and "kernel-not-bindable" not in _codes(report)
    session.apply({"operation": "configure", "node_id": "cam_run1.radiation", "configuration": {"kernels": {}}})
    # the runner pauses at micro_mg_tend and that pause has passed its gate: no finding beyond the binding itself
    session.apply({"operation": "configure", "node_id": "cam_run1.cloud_macro_microphysics",
                   "configuration": {"kernels": {"micro_mg_tend": {"kind": "surrogate", "path": "m.pt"}}}})
    report = _check(session, catalog)
    assert "kernel-not-validated" not in _codes(report) and "kernel-not-bindable" not in _codes(report)
    assert report.status == "valid"
    session.apply({"operation": "configure", "node_id": "cam_run1.cloud_macro_microphysics",
                   "configuration": {"kernels": {"mmacro_pcond": {"kind": "surrogate", "path": "m.pt"}}}})
    browser = _check(session, catalog)
    assert browser.status == "valid"
    assert {"kernel-replaced", "model-file"} <= set(_codes(browser))   # both informational in the browser
    local = _check(session, catalog, level="local")
    assert any(issue.code == "model-file" and issue.severity == "error" for issue in local.issues)


def test_the_model_file_is_accepted_when_it_exists_locally(session, catalog, tmp_path) -> None:
    model = tmp_path / "m.pt"
    model.write_bytes(b"weights")
    session.apply({"operation": "configure", "node_id": "cam_run1.cloud_macro_microphysics",
                   "configuration": {"kernels": {"mmacro_pcond": {"kind": "surrogate", "path": str(model)}}}})
    local = _check(session, catalog, level="local")
    assert local.status == "valid"
    assert local.checks["replaced_kernels"] == ["cam_run1.cloud_macro_microphysics:mmacro_pcond"]


def test_parameters_are_checked_against_the_audited_table(session, catalog) -> None:
    session.apply({"operation": "configure", "node_id": "cam_run1.deep_convection",
                   "configuration": {"parameters": {"zmconv_c0_lnd": "fast", "nonesuch": 1.0}}})
    codes = _codes(_check(session, catalog))
    assert "parameter-type" in codes and "parameter-unknown" in codes
    session.apply({"operation": "configure", "node_id": "cam_run1.deep_convection",
                   "configuration": {"parameters": {"zmconv_c0_lnd": 0.0075}}})
    assert _check(session, catalog).status == "valid"


def test_a_document_from_another_catalog_is_flagged_not_refused(session, catalog) -> None:
    document, entries, snapshot = catalog
    report = validate_document(session.document, default=document, catalog=entries,
                               catalog_version="somethingelse")
    assert report.status == "warning" and "catalog-version" in _codes(report)


def test_the_report_never_claims_more_than_it_checked(session, catalog) -> None:
    report = _check(session, catalog)
    assert "bit-for-bit" in report.disclaimer
    payload = report.to_payload()
    assert payload["status"] == "valid" and payload["level"] == "browser"
