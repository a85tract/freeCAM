"""The default workflow and the process library, from the model's own records.

Nothing here is a hand-written list.  The default order is the current
``PICAMStepPlan``; the processes that can be added come from the physics
catalog the package ships and from the process-support record under
``validation/``; the kernel bindings from the stage classes and the segment
runner; the tunables from ``native/pi_cam/runtime_parameters.yaml``.  The
snapshot the published page reads is the same data, serialised once, with
the commit and a content hash so the page can say what it describes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ..plan import PICAMStepPlan
from .capabilities import KernelCapability, kernel_capabilities
from .document import (
    LOCKED_OPERATIONS,
    SCIENTIFIC_KINDS,
    KernelBinding,
    NodeConfiguration,
    WorkflowCatalogEntry,
    WorkflowDocument,
    WorkflowNode,
)

SNAPSHOT_SCHEMA_VERSION = 1

#: The cases a Driver accepts by name; ``configs/`` holds their files.
CASES: Mapping[str, str] = {
    "PI-atm": "configs/pi_cam_icesm131.yaml",
    "PI-atm-online": "configs/pi_cam_icesm131_online.yaml",
    "PI-atm-replay": "configs/pi_cam_icesm131_replay.yaml",
    "PI-atm-1month": "configs/pi_cam_icesm131_1month.yaml",
}

CATEGORY_BY_KIND: Mapping[str, str] = {
    "scheme": "Physics",
    "coupling": "Coupling",
    "dynamics": "Dynamics",
    "boundary": "Control",
    "clock": "Control",
    "kernel": "Control",
    "service": "Control",
    "io": "Output",
    "python_process": "Python",
    "runtime_catalog_process": "Catalog process",
}

CONTROL_SKELETON = ("boundary_import", "advance_timestep", "boundary_export")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# the default workflow
# ---------------------------------------------------------------------------


def _node_from_row(row: Mapping[str, Any], *, parameters: Mapping[str, Any],
                   capabilities: Mapping[str, tuple[KernelCapability, ...]]) -> WorkflowNode:
    qualified = f"{row['phase']}.{row['name']}"
    kind = str(row["kind"])
    scientific = kind in SCIENTIFIC_KINDS
    metadata: dict[str, Any] = {
        "control_owner": row.get("control_owner", "python"),
        "parameters": list(parameters.get(qualified, ())),
        "kernels": [capability.to_payload() for capability in capabilities.get(qualified, ())],
    }
    return WorkflowNode(
        id=qualified,
        name=str(row["name"]),
        qualified_name=qualified,
        operation=str(row["operation"]),
        phase=str(row["phase"]),
        kind=kind,
        implementation=str(row.get("implementation", "python")),
        enabled=bool(row.get("enabled", True)),
        movable=scientific and str(row["operation"]) not in LOCKED_OPERATIONS,
        removable=scientific and str(row["operation"]) not in LOCKED_OPERATIONS,
        parent_stage=row.get("parent_stage"),
        granularity=str(row.get("granularity", "stage")),
        origin="default",
        native_id=row.get("native_id"),
        default_index=int(row["index"]),
        configuration=NodeConfiguration(
            kernels={c.kernel: KernelBinding() for c in capabilities.get(qualified, ())}
        ),
        metadata=metadata,
    )


def runtime_parameters(root: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """The audited runtime tunables, grouped by the workflow action that reads them."""

    path = (root or _repo_root()) / "native" / "pi_cam" / "runtime_parameters.yaml"
    table = _read_yaml(path).get("parameters", {}) if path.is_file() else {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for name, spec in sorted(table.items()):
        grouped.setdefault(str(spec["workflow_action"]), []).append(
            {"name": str(name), "dtype": str(spec.get("dtype", "float64")),
             "notes": str(spec.get("notes", "")).strip()}
        )
    return grouped


def default_nodes(*, root: Path | None = None) -> tuple[WorkflowNode, ...]:
    parameters = runtime_parameters(root)
    capabilities = _capabilities_by_action()
    return tuple(
        _node_from_row(row, parameters=parameters, capabilities=capabilities)
        for row in PICAMStepPlan.default().describe()
    )


def _capabilities_by_action() -> dict[str, tuple[KernelCapability, ...]]:
    grouped: dict[str, list[KernelCapability]] = {}
    for capability in kernel_capabilities():
        grouped.setdefault(capability.stage_action, []).append(capability)
    return {action: tuple(items) for action, items in grouped.items()}


def default_document(case: str = "PI-atm", nsteps: int = 2, *,
                     root: Path | None = None, catalog_version: str = "",
                     source_version: str = "") -> WorkflowDocument:
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; one of {sorted(CASES)}")
    return WorkflowDocument(
        nodes=default_nodes(root=root), case=case, nsteps=int(nsteps),
        catalog_version=catalog_version, source_version=source_version,
    )


# ---------------------------------------------------------------------------
# the library
# ---------------------------------------------------------------------------


def _load_physics_catalog() -> Mapping[str, Any]:
    resource = files("freecam.pi_cam.data").joinpath("pi_cam_physics_catalog.json")
    return json.loads(resource.read_text())


def _load_process_support(root: Path) -> Mapping[str, Mapping[str, Any]]:
    path = root / "validation" / "pi_cam_process_support.json"
    if not path.is_file():
        return {}
    record = json.loads(path.read_text())
    return {str(item["name"]): item for item in record.get("processes", ())}


def catalog_entries(nodes: Sequence[WorkflowNode], *, root: Path | None = None
                    ) -> dict[str, WorkflowCatalogEntry]:
    """Every process the library lists, addable or not, with the reason when not."""

    root = root or _repo_root()
    entries: dict[str, WorkflowCatalogEntry] = {}
    default_ids = {node.id for node in nodes}
    for node in nodes:
        entries[node.id] = WorkflowCatalogEntry(
            node=replace(node, enabled=True),
            category=CATEGORY_BY_KIND.get(node.kind, "Other"),
            addable=node.scientific,
            reason=None if node.scientific else "a control action; always part of the step",
            in_default=node.enabled,
            description=_describe_default(node),
        )

    catalog = _load_physics_catalog()
    support = _load_process_support(root)
    for process in catalog.get("processes", ()):
        if process.get("level") != "process":
            continue
        name = str(process["name"])
        qualified = str(process["qualified_name"])
        actions = tuple(process.get("workflow_actions", ()))
        if any(action in default_ids for action in actions):
            continue                       # it is a workflow action already
        parents = tuple(process.get("parent_actions", ()))
        phase = parents[0].split(".", 1)[0] if parents and "." in parents[0] else "cam_run1"
        status = str(process.get("adapter_status", ""))
        loadable = bool(support.get(name, {}).get("current_case_loadable", False))
        addable = status == "validated" and loadable
        if addable:
            reason = None
        elif status == "validated":
            reason = "its adapter is not loadable in the current case"
        else:
            blockers = ", ".join(str(item) for item in process.get("blockers", ())) or status
            reason = f"not independently runnable ({status}): {blockers}"
        node = WorkflowNode(
            id=f"catalog:{name}",
            name=name,
            qualified_name=qualified,
            operation=name,
            phase=phase,
            kind="runtime_catalog_process",
            implementation="fortran-numerical-kernel",
            enabled=True,
            movable=True,
            removable=True,
            source=process.get("source"),
            parent_stage=parents[0] if parents else None,
            granularity="process",
            origin="catalog",
            metadata={
                "adapter_status": status,
                "role": process.get("role"),
                "parent_actions": list(parents),
                # the argument list only where the process can actually be added
                "arguments": [
                    {"name": a.get("name"), "intent": a.get("intent"), "dtype": a.get("dtype"),
                     "dimensions": list(a.get("dimensions", ()))}
                    for a in process.get("arguments", ())
                ] if addable else [],
                "current_case_loadable": loadable,
            },
        )
        entries[node.id] = WorkflowCatalogEntry(
            node=node, category="Catalog process", addable=addable, reason=reason,
            in_default=False,
            description=f"{qualified} from {process.get('source', 'the pinned source')}",
        )
    return entries


def _describe_default(node: WorkflowNode) -> str:
    if node.locked:
        return "A required control action of every CAM step."
    if node.kind in {"io", "service", "clock", "kernel"}:
        return "Part of the step's control skeleton; runs every step, hidden from the canvas."
    if node.parent_stage:
        return f"A leaf of {node.parent_stage}; the parent and its leaves never run together."
    return f"Original CAM process {node.operation} in {node.phase}."


# ---------------------------------------------------------------------------
# the snapshot
# ---------------------------------------------------------------------------


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
            check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def parent_leaf_groups(nodes: Sequence[WorkflowNode]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for node in nodes:
        if node.parent_stage:
            groups.setdefault(node.parent_stage, []).append(node.id)
    return groups


def build_snapshot(*, case: str = "PI-atm", nsteps: int = 2, root: Path | None = None,
                   stamp: bool = True) -> dict[str, Any]:
    """Everything the browser needs, as one JSON-able record.

    ``catalog_hash`` covers the content; the timestamp and the commit are
    outside it, so two snapshots of the same data compare equal.
    """

    root = root or _repo_root()
    nodes = default_nodes(root=root)
    entries = catalog_entries(nodes, root=root)
    capabilities = [c.to_payload() for c in kernel_capabilities()]
    parameters = runtime_parameters(root)
    physics = _load_physics_catalog()
    content = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "cases": dict(CASES),
        "default_nodes": [node.to_payload() for node in nodes],
        "entries": [entry.to_payload() for entry in entries.values()],
        "capabilities": capabilities,
        "parameters": parameters,
        "rules": {
            "locked_operations": sorted(LOCKED_OPERATIONS),
            "control_skeleton": list(CONTROL_SKELETON),
            "parent_leaf_groups": parent_leaf_groups(nodes),
        },
        "source_revision": str(physics.get("source_revision", "unknown")),
    }
    catalog_hash = sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    document = default_document(
        case, nsteps, root=root, catalog_version=catalog_hash,
        source_version=content["source_revision"],
    )
    snapshot = dict(content)
    snapshot["catalog_hash"] = catalog_hash
    snapshot["default_document"] = document.to_payload()
    if stamp:
        snapshot["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        snapshot["commit"] = _git_commit(root)
    return snapshot


def load_catalog(*, root: Path | None = None) -> tuple[WorkflowDocument, dict[str, WorkflowCatalogEntry], dict[str, Any]]:
    """The default document, the library, and the snapshot, built together."""

    snapshot = build_snapshot(root=root)
    document = WorkflowDocument.from_payload(snapshot["default_document"])
    entries = catalog_entries(document.nodes, root=root)
    return document, entries, snapshot


__all__ = [
    "CASES",
    "CATEGORY_BY_KIND",
    "CONTROL_SKELETON",
    "SNAPSHOT_SCHEMA_VERSION",
    "build_snapshot",
    "catalog_entries",
    "default_document",
    "default_nodes",
    "load_catalog",
    "parent_leaf_groups",
    "runtime_parameters",
]
