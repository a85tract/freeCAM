"""The Workflow Builder's document: edits, history, hashing, import."""

import json

import pytest

from freecam.pi_cam.workflow_builder import (
    KernelBinding,
    NodeConfiguration,
    RevisionConflict,
    WorkflowDocument,
    WorkflowEditError,
    WorkflowEditSession,
    load_catalog,
    python_process_template,
)


@pytest.fixture(scope="module")
def catalog():
    document, entries, snapshot = load_catalog()
    return document, entries, snapshot


@pytest.fixture
def session(catalog):
    document, entries, _ = catalog
    return WorkflowEditSession(document, entries, python_template=python_process_template)


def test_the_default_document_is_the_step_plan_with_control_rows_locked(catalog) -> None:
    document, _, _ = catalog
    assert document.nodes[0].operation == "boundary_import"
    assert document.nodes[-1].operation == "boundary_export"
    assert sum(node.operation == "advance_timestep" for node in document.nodes) == 1
    locked = [node for node in document.nodes if node.locked]
    assert all(not node.movable and not node.removable for node in locked)
    assert len(document.scientific_nodes) < len(document.nodes)      # the canvas hides control rows


def test_every_edit_is_one_undoable_revision(session) -> None:
    before = session.document
    moved = session.apply({"operation": "move", "node_id": "cam_run1.radiation",
                           "before": "cam_run1.dry_adjustment", "revision": 0})
    assert moved.revision == 1
    assert moved.ids.index("cam_run1.radiation") < moved.ids.index("cam_run1.dry_adjustment")
    assert session.apply({"operation": "undo"}).ids == before.ids
    assert session.apply({"operation": "redo"}).ids == moved.ids
    assert session.apply({"operation": "reset"}).workflow_hash == before.workflow_hash


def test_a_stale_revision_is_refused(session) -> None:
    session.apply({"operation": "disable", "node_id": "cam_run1.radiation"})
    with pytest.raises(RevisionConflict):
        session.apply({"operation": "enable", "node_id": "cam_run1.radiation", "revision": 0})


def test_control_actions_cannot_move_be_removed_or_disabled(session) -> None:
    for edit in (
        {"operation": "move", "node_id": "coupling.boundary_import", "index": 3},
        {"operation": "remove", "node_id": "clock.advance_timestep"},
        {"operation": "disable", "node_id": "coupling.boundary_export"},
    ):
        with pytest.raises(WorkflowEditError):
            session.apply(edit)


def test_removal_and_disabling_are_different_things(session) -> None:
    disabled = session.apply({"operation": "disable", "node_id": "cam_run1.radiation"})
    assert "cam_run1.radiation" in disabled and not disabled.node("cam_run1.radiation").enabled
    removed = session.apply({"operation": "remove", "node_id": "cam_run1.radiation"})
    assert "cam_run1.radiation" not in removed
    restored = session.apply({"operation": "restore", "node_id": "cam_run1.radiation"})
    # back in its default slot: right after the process that precedes it by default
    assert restored.ids.index("cam_run1.radiation") == session.default_document.ids.index("cam_run1.radiation")


def test_a_python_process_gets_a_runnable_template_and_may_exist_several_times(session) -> None:
    first = session.apply({"operation": "add_python", "name": "heating_a", "after": "cam_run1.dry_adjustment"})
    node = first.node("python:heating_a")
    assert node.kind == "python_process" and node.origin == "python"
    assert "class HeatingA(fc.Physics)" in node.configuration.python_source
    assert 'after = "dry_adjustment"' in node.configuration.python_source
    second = session.apply({"operation": "add_python", "name": "heating_b", "after": "python:heating_a"})
    assert [n.name for n in second.python_nodes] == ["heating_a", "heating_b"]
    with pytest.raises(WorkflowEditError, match="already"):
        session.apply({"operation": "add_python", "name": "heating_a"})
    with pytest.raises(WorkflowEditError, match="identifier"):
        session.apply({"operation": "add_python", "name": "not a name"})


def test_replacing_a_process_keeps_the_slot_and_drops_the_old_configuration(session) -> None:
    session.apply({"operation": "configure", "node_id": "cam_run1.deep_convection",
                   "configuration": {"parameters": {"zmconv_c0_lnd": 0.0075}}})
    slot = session.document.index("cam_run1.deep_convection")
    replaced = session.apply({"operation": "replace", "node_id": "cam_run1.deep_convection",
                              "name": "my_convection"})
    assert replaced.nodes[slot].id == "python:my_convection"
    assert "cam_run1.deep_convection" not in replaced
    assert replaced.nodes[slot].configuration.parameters == {}


def test_the_hash_covers_configuration_not_just_order(session) -> None:
    base = session.document.workflow_hash
    configured = session.apply({"operation": "configure", "node_id": "cam_run1.deep_convection",
                                "configuration": {"parameters": {"zmconv_ke": 1.0e-6}}})
    assert configured.workflow_hash != base
    assert configured.order_hash == session.default_document.order_hash
    bound = session.apply({"operation": "configure", "node_id": "cam_run1.cloud_macro_microphysics",
                           "configuration": {"kernels": {"mmacro_pcond": {"kind": "surrogate", "path": "m.pt"}}}})
    assert bound.workflow_hash != configured.workflow_hash
    steps = session.apply({"operation": "set_nsteps", "nsteps": 10})
    assert steps.workflow_hash != bound.workflow_hash
    # the experimental flag is a permission, not something that runs
    flagged = session.apply({"operation": "set_experimental", "experimental": True})
    assert flagged.workflow_hash == steps.workflow_hash


def test_kernel_bindings_are_either_the_original_or_a_model_by_path() -> None:
    assert not KernelBinding().replaces
    assert KernelBinding("surrogate", "model.pt").replaces
    with pytest.raises(WorkflowEditError):
        KernelBinding("surrogate")
    with pytest.raises(WorkflowEditError):
        KernelBinding("original", "model.pt")
    with pytest.raises(WorkflowEditError):
        KernelBinding("callback")


def test_the_document_round_trips_through_json_and_import(session) -> None:
    session.apply({"operation": "add_python", "name": "heating", "after": "cam_run1.dry_adjustment"})
    session.apply({"operation": "configure", "node_id": "python:heating",
                   "configuration": {"parameters": {"rate": 0.5},
                                     "variables": [{"name": "heating_rate", "like": "T", "units": "K s-1"}]}})
    session.apply({"operation": "set_namelist", "namelist": {"cldfrc_rhminl": 0.9}})
    payload = json.loads(json.dumps(session.document.to_payload()))
    again = WorkflowDocument.from_payload(payload)
    assert again.workflow_hash == session.document.workflow_hash
    assert again.node("python:heating").configuration.variables[0].units == "K s-1"

    fresh = WorkflowEditSession(session.default_document, session.catalog,
                                python_template=python_process_template)
    imported = fresh.import_document(payload)
    assert imported.workflow_hash == again.workflow_hash
    assert fresh.can_undo                                     # the import is one undoable step


def test_a_version_one_document_still_imports_with_defaults(catalog) -> None:
    document, _, _ = catalog
    payload = document.to_payload()
    payload["schema_version"] = 1
    for key in ("case", "nsteps", "namelist", "catalog_version", "source_version"):
        payload.pop(key)
    for node in payload["nodes"]:
        node.pop("configuration")
    old = WorkflowDocument.from_payload(payload)
    assert old.case == "PI-atm" and old.nsteps == 2
    assert all(node.configuration.is_empty or node.configuration.kernels for node in old.nodes)


def test_an_imported_document_naming_an_unknown_process_is_refused(session) -> None:
    payload = session.document.to_payload()
    payload["nodes"][5]["id"] = "cam_run1.nonesuch"
    payload["nodes"][5]["qualified_name"] = "cam_run1.nonesuch"
    with pytest.raises(WorkflowEditError, match="does not have"):
        session.import_document(payload)


def test_configuration_updates_replace_only_the_named_fields() -> None:
    configuration = NodeConfiguration(parameters={"a": 1}, python_source="x = 1")
    updated = configuration.updated({"parameters": {"b": 2}})
    assert updated.parameters == {"b": 2} and updated.python_source == "x = 1"
    with pytest.raises(WorkflowEditError):
        configuration.updated({"colour": "red"})


@pytest.mark.parametrize("value, expected", [
    (1.0, "1"), (-3.0, "-3"), (0.0075, "0.0075"), (1e-6, "0.000001"), (1e-7, "1e-7"),
    (1.5e-7, "1.5e-7"), (1e21, "1e+21"), (1e20, "100000000000000000000"), (123456.789, "123456.789"),
    (0.1, "0.1"), (2.5e-5, "0.000025"), (1.7976931348623157e308, "1.7976931348623157e+308"),
    (5e-324, "5e-324"), (0.0, "0"), (12345678901234567890.0, "12345678901234567000"),
])
def test_numbers_are_written_as_javascript_writes_them(value, expected) -> None:
    from freecam.pi_cam.workflow_builder.document import js_number

    assert js_number(value) == expected


def test_the_canonical_form_agrees_with_node_when_node_is_available(catalog) -> None:
    import shutil
    import subprocess

    from freecam.pi_cam.workflow_builder.document import _canonical

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed here")
    document, _, _ = catalog
    record = document.execution_record()
    record["numbers"] = [1.0, 0.0075, 1e-7, 1e21, 2.5e-5, -0.5, 1234567.125]
    record["text"] = "héating °C   \"quoted\" \\ /"
    script = (
        "const sortKeys = (v) => Array.isArray(v) ? v.map(sortKeys) : (v && typeof v === 'object')"
        " ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, sortKeys(v[k])])) : v;"
        "let input = ''; process.stdin.on('data', (d) => input += d);"
        "process.stdin.on('end', () => process.stdout.write(JSON.stringify(sortKeys(JSON.parse(input)))));"
    )
    result = subprocess.run([node, "-e", script], input=json.dumps(record), capture_output=True,
                            text=True, check=True)
    assert result.stdout == _canonical(record)
