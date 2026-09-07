"""Checks over a workflow document, at two levels.

The browser level is what a page can decide from the document and the
catalog alone: names, duplicates, the control skeleton, parent/leaf
exclusivity, required bindings, parameter types, and whether the order is
still the validated default.  The local level adds what only a machine with
the checkout can tell: Python syntax, that a named model file exists, and
that the catalog the document was built against is this one.  Neither level
runs the user's code, loads a network, or touches a model.

A passing check says the declared constraints hold.  It does not say the
workflow is physically right, stable, or bit-for-bit -- only a gate does.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capabilities import KernelCapability, kernel_capabilities
from .document import (
    CONTROL_SKELETON_KINDS,
    WorkflowCatalogEntry,
    WorkflowDocument,
    WorkflowIssue,
    WorkflowNode,
    WorkflowValidationReport,
)

LEVELS = ("browser", "local")

_NUMBER_TYPES = {"float64": (int, float), "float32": (int, float), "int32": (int,), "int64": (int,),
                 "bool": (bool,)}


def _issue(severity: str, code: str, message: str, node: WorkflowNode | None = None,
           field: str | None = None) -> WorkflowIssue:
    return WorkflowIssue(severity, code, message, None if node is None else node.id, field)


def validate_document(
    document: WorkflowDocument,
    *,
    default: WorkflowDocument,
    catalog: Mapping[str, WorkflowCatalogEntry],
    level: str = "browser",
    catalog_version: str | None = None,
    capabilities: Iterable[KernelCapability] | None = None,
    root: Path | None = None,
) -> WorkflowValidationReport:
    if level not in LEVELS:
        raise ValueError(f"unknown validation level {level!r}; one of {LEVELS}")
    caps = {(c.stage_action, c.kernel): c for c in (capabilities or kernel_capabilities())}
    issues: list[WorkflowIssue] = []
    checks: dict[str, Any] = {"level": level}

    issues += _check_skeleton(document, default)
    issues += _check_names(document)
    issues += _check_membership(document, catalog)
    issues += _check_parent_leaf(document)
    issues += _check_bindings(document, caps, level=level, root=root)
    issues += _check_parameters(document, catalog)
    issues += _check_python(document, level=level)
    changed = _check_experimental(document, default, issues)
    checks["order_changed"] = changed["order"]
    checks["processes_replaced_or_removed"] = changed["membership"]
    checks["experimental"] = document.experimental
    checks["python_processes"] = len(document.python_nodes)
    checks["replaced_kernels"] = sorted(
        f"{node.id}:{k}" for node in document.nodes for k in node.configuration.replaced_kernels
    )
    if catalog_version is not None and document.catalog_version and document.catalog_version != catalog_version:
        issues.append(WorkflowIssue(
            "warning", "catalog-version",
            "the document was made against a different catalog; the library and the "
            "default order may have changed since",
        ))
    checks["not_verified"] = _not_verified(document)
    return WorkflowValidationReport(
        revision=document.revision, workflow_hash=document.workflow_hash,
        issues=tuple(issues), level=level, checks=checks,
    )


def _check_skeleton(document: WorkflowDocument, default: WorkflowDocument) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    nodes = document.nodes
    if not nodes or nodes[0].operation != "boundary_import":
        issues.append(_issue("error", "skeleton", "the step must start with boundary_import"))
    if not nodes or nodes[-1].operation != "boundary_export":
        issues.append(_issue("error", "skeleton", "the step must end with boundary_export"))
    clocks = [node for node in nodes if node.operation == "advance_timestep"]
    if len(clocks) != 1:
        issues.append(_issue("error", "skeleton", "the step must advance the clock exactly once"))
    for node in nodes:
        if node.locked and not node.enabled:
            issues.append(_issue("error", "skeleton",
                                 f"{node.display_name} is a required control action and must stay enabled",
                                 node))
    control_now = [n.id for n in nodes if n.kind in CONTROL_SKELETON_KINDS]
    control_default = [n.id for n in default.nodes if n.kind in CONTROL_SKELETON_KINDS and n.id in set(control_now)]
    if [c for c in control_now if c in set(control_default)] != control_default:
        issues.append(_issue("error", "skeleton", "control actions must keep their relative order"))
    return issues


def _check_names(document: WorkflowDocument) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    seen: dict[str, WorkflowNode] = {}
    for node in document.nodes:
        key = node.qualified_name
        if key in seen:
            issues.append(_issue("error", "duplicate", f"{node.display_name} appears twice", node))
        seen[key] = node
    names: dict[str, WorkflowNode] = {}
    for node in document.nodes:
        if node.name in names and node.scientific:
            issues.append(_issue("error", "duplicate-name",
                                 f"two processes are both named {node.name!r}", node, "name"))
        names.setdefault(node.name, node)
    return issues


def _check_membership(document: WorkflowDocument, catalog: Mapping[str, WorkflowCatalogEntry]) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    for node in document.nodes:
        if node.is_python:
            continue
        entry = catalog.get(node.id)
        if entry is None:
            issues.append(_issue("error", "unknown-process",
                                 f"{node.display_name} is not in this catalog", node))
        elif not entry.addable and node.enabled and node.scientific:
            issues.append(_issue("error", "not-addable",
                                 f"{node.display_name} cannot be added: {entry.reason}", node))
    return issues


def _check_parent_leaf(document: WorkflowDocument) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    enabled = {node.id: node for node in document.nodes if node.enabled}
    for node in document.nodes:
        if node.enabled and node.parent_stage and node.parent_stage in enabled:
            parent = enabled[node.parent_stage]
            issues.append(_issue(
                "error", "parent-and-leaf",
                f"{node.display_name} is a leaf of {parent.display_name}; both are enabled, "
                f"so the routine would run twice",
                node,
            ))
    return issues


def _check_bindings(document: WorkflowDocument, caps: Mapping[tuple[str, str], KernelCapability],
                    *, level: str, root: Path | None) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    for node in document.nodes:
        for kernel, binding in node.configuration.kernels.items():
            capability = caps.get((node.id, kernel))
            if capability is None:
                issues.append(_issue("error", "unknown-kernel",
                                     f"{node.display_name} has no swappable kernel {kernel!r}",
                                     node, "kernels"))
                continue
            if not binding.replaces:
                continue
            if not capability.bindable:
                issues.append(_issue("error", "kernel-not-bindable",
                                     f"{kernel} cannot be replaced: {capability.reason}",
                                     node, "kernels"))
                continue
            if not capability.validated:
                issues.append(_issue("warning", "kernel-not-validated",
                                     f"{kernel} can be bound, but its pause path has not passed a gate",
                                     node, "kernels"))
            if binding.kind == "surrogate":
                issues.append(_issue("info", "kernel-replaced",
                                     f"{kernel} is answered by the model at {binding.path}; the run "
                                     f"is that model's answer, not bit-for-bit with the original",
                                     node, "kernels"))
                if level == "local":
                    path = Path(str(binding.path)).expanduser()
                    if root is not None and not path.is_absolute():
                        path = root / path
                    if not path.is_file():
                        issues.append(_issue("error", "model-file",
                                             f"model file not found: {binding.path}", node, "kernels"))
                else:
                    issues.append(_issue("info", "model-file",
                                         "the model file is checked where the model runs", node, "kernels"))
    return issues


def _check_parameters(document: WorkflowDocument, catalog: Mapping[str, WorkflowCatalogEntry]) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    for node in document.nodes:
        values = node.configuration.parameters
        if not values:
            continue
        if node.is_python:
            for name, value in values.items():
                if not isinstance(value, (int, float, str, bool, list, dict, type(None))):
                    issues.append(_issue("error", "parameter-type",
                                         f"property {name!r} must be a JSON value", node, "parameters"))
            continue
        declared = {str(item["name"]): item for item in node.metadata.get("parameters", ())}
        for name, value in values.items():
            spec = declared.get(name)
            if spec is None:
                issues.append(_issue("error", "parameter-unknown",
                                     f"{node.display_name} has no runtime parameter {name!r}",
                                     node, "parameters"))
                continue
            allowed = _NUMBER_TYPES.get(str(spec.get("dtype", "float64")), (int, float))
            if isinstance(value, bool) and bool not in allowed:
                issues.append(_issue("error", "parameter-type",
                                     f"{name} must be a number", node, "parameters"))
            elif not isinstance(value, allowed):
                issues.append(_issue("error", "parameter-type",
                                     f"{name} must be {spec.get('dtype')}", node, "parameters"))
    return issues


def _check_python(document: WorkflowDocument, *, level: str) -> list[WorkflowIssue]:
    issues: list[WorkflowIssue] = []
    for node in document.python_nodes:
        source = node.configuration.python_source
        if not source or not source.strip():
            issues.append(_issue("error", "python-source", f"{node.name} has no source", node, "python_source"))
            continue
        if level == "local":
            try:
                tree = ast.parse(source)
            except SyntaxError as error:
                issues.append(_issue("error", "python-syntax",
                                     f"{node.name}: {error.msg} (line {error.lineno})", node, "python_source"))
                continue
            classes = [item for item in tree.body if isinstance(item, ast.ClassDef)]
            if not classes:
                issues.append(_issue("error", "python-class",
                                     f"{node.name} must define one fc.Physics class", node, "python_source"))
            else:
                declared = _class_name_attribute(classes[-1])
                if declared is not None and declared != node.name:
                    issues.append(_issue("error", "python-name",
                                         f"the class declares name = {declared!r}, but the process is "
                                         f"{node.name!r}", node, "python_source"))
        else:
            issues.append(_issue("info", "python-syntax",
                                 f"{node.name}: the source is parsed where the model runs", node, "python_source"))
    return issues


def _class_name_attribute(klass: ast.ClassDef) -> str | None:
    for statement in klass.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id == "name" and isinstance(statement.value, ast.Constant):
                return str(statement.value.value)
    return None


def _check_experimental(document: WorkflowDocument, default: WorkflowDocument,
                        issues: list[WorkflowIssue]) -> dict[str, bool]:
    default_sci = [n.id for n in default.nodes if n.scientific and n.enabled]
    current_sci = [n.id for n in document.nodes if n.scientific and n.enabled]
    shared_default = [n for n in default_sci if n in set(current_sci)]
    shared_current = [n for n in current_sci if n in set(default_sci)]
    order_changed = shared_default != shared_current
    default_all = {n.id for n in default.nodes if n.scientific}
    membership_changed = any(
        (n.id in default_all and (n.id not in document or not document.node(n.id).enabled) and n.enabled)
        for n in default.nodes if n.scientific and not n.parent_stage
    ) or any(n.enabled and n.id in default_all and not default.node(n.id).enabled and n.scientific
             for n in document.nodes if n.id in default_all)
    if (order_changed or membership_changed) and not document.experimental:
        what = "the scientific order" if order_changed else "the set of physical processes"
        issues.append(WorkflowIssue(
            "error", "experimental-required",
            f"{what} differs from the validated default; enable Experimental to run it",
        ))
    elif order_changed or membership_changed:
        issues.append(WorkflowIssue(
            "warning", "experimental",
            "this order or process set is not the validated default; results are exploratory",
        ))
    return {"order": order_changed, "membership": membership_changed}


def _not_verified(document: WorkflowDocument) -> list[str]:
    """What no static check can prove; said so rather than passed silently."""

    items: list[str] = []
    for node in document.python_nodes:
        items.append(f"{node.name}: which fields it reads and writes is known only when it runs")
    for node in document.nodes:
        if node.origin == "catalog" and node.enabled:
            items.append(f"{node.display_name}: its field bindings are made on the live model")
    return items


__all__ = ["LEVELS", "validate_document"]
