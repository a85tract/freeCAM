"""Framework-independent data model for the freeCAM Workflow Builder.

The document is the single ordered list of workflow actions, including the
control, clock and I/O rows the browser hides by default, together with the
configuration of each: parameter values, a Python process's source, the
kernel bindings of a stage, and the variables a process declares.  Every edit
produces a new immutable document; :class:`WorkflowEditSession` owns the
current revision, the undo/redo stacks, and the optimistic-concurrency check
that rejects edits made against a stale revision.

The document's hash covers everything that decides what a run executes --
the case, the step count, the namelist overrides, the order and membership,
each node's enabled flag and its configuration.  What the browser shows
(selection, theme, panel sizes) is not part of the document at all.
"""

from __future__ import annotations

import json
import keyword
import threading
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Callable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 2

LOCKED_OPERATIONS = frozenset(
    {"boundary_import", "advance_timestep", "boundary_export"}
)
"""Control actions that a case workflow can never remove, move, or disable."""

SCIENTIFIC_KINDS = frozenset(
    {
        "scheme",
        "coupling",
        "dynamics",
        "python_process",
        "runtime_fortran_process",
        "runtime_catalog_process",
        "catalog_process",
    }
)
"""Kinds shown in the normal (non-advanced) canvas view."""

CONTROL_SKELETON_KINDS = frozenset({"boundary", "clock", "io", "service", "kernel"})
"""Kinds whose relative order defines the CAM step lifecycle."""

KERNEL_BINDING_KINDS = frozenset({"original", "surrogate"})
"""What may stand in a kernel slot: the original routine, or a trained network by path."""

PYTHON_NODE_PREFIX = "python:"


class WorkflowEditError(ValueError):
    """An edit is structurally impossible for this document."""


class RevisionConflict(WorkflowEditError):
    """The edit was made against a revision that is no longer current."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"workflow revision {expected} is stale; the current revision is {actual}"
        )
        self.expected = int(expected)
        self.actual = int(actual)


def js_number(value: float) -> str:
    """A float as JavaScript's ``Number.prototype.toString`` writes it.

    The browser and the service hash the same document, so both must write
    numbers the same way.  JavaScript has one number type: an integral double
    below 1e21 prints with no decimal point, and the switch to exponent form
    happens at 1e21 and below 1e-6 rather than where Python's ``repr`` puts
    it.  The digits themselves are the shortest round-trip digits in both
    languages, which is what ``repr`` gives.
    """

    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("a workflow document holds finite numbers only")
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    text = repr(abs(value))
    mantissa, _, exponent = text.partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = (whole + fraction).lstrip("0")
    # n: the position of the decimal point relative to the first significant digit
    n = len(whole) - (len(whole + fraction) - len(digits)) + int(exponent or 0)
    if whole == "0":
        n = -(len(fraction) - len(fraction.lstrip("0"))) + int(exponent or 0)
    digits = digits.rstrip("0") or "0"
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    exponent_text = f"e{'+' if n - 1 >= 0 else '-'}{abs(n - 1)}"
    if k == 1:
        return sign + digits + exponent_text
    return sign + digits[0] + "." + digits[1:] + exponent_text


def _canonical(value: Any) -> str:
    """Compact JSON with sorted keys, written the way the browser writes it.

    Strings keep their non-ASCII characters, numbers follow JavaScript's
    formatting, keys sort by code point: the same bytes on both sides, so the
    same document has one hash wherever it is computed.
    """

    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return js_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, Mapping):
        items = sorted((str(key), inner) for key, inner in value.items())
        return "{" + ",".join(f"{json.dumps(key, ensure_ascii=False)}:{_canonical(inner)}"
                              for key, inner in items) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical(item) for item in value) + "]"
    return json.dumps(str(value), ensure_ascii=False)


# ---------------------------------------------------------------------------
# node configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KernelBinding:
    """What stands in one swappable kernel's slot.

    ``original`` is the routine itself and means the slot is left alone;
    ``surrogate`` names a trained network by path.  The path is a reference:
    weights never enter the document, the generated code, or a published page.
    """

    kind: str = "original"
    path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KERNEL_BINDING_KINDS:
            raise WorkflowEditError(
                f"unknown kernel binding {self.kind!r}; one of {sorted(KERNEL_BINDING_KINDS)}"
            )
        if self.kind == "surrogate" and not (self.path or "").strip():
            raise WorkflowEditError("a surrogate binding names the model file by path")
        if self.kind == "original" and self.path:
            raise WorkflowEditError("the original kernel takes no model path")

    @property
    def replaces(self) -> bool:
        return self.kind != "original"

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path}

    @classmethod
    def from_payload(cls, values: Mapping[str, Any] | str) -> "KernelBinding":
        if isinstance(values, str):
            return cls(kind=values)
        return cls(kind=str(values.get("kind", "original")), path=values.get("path"))


@dataclass(frozen=True, slots=True)
class VariableDeclaration:
    """A Python-owned StatePool field a process asks to exist."""

    name: str
    like: str = "T"
    units: str = "1"
    output: bool = True

    def __post_init__(self) -> None:
        if not _is_identifier(self.name):
            raise WorkflowEditError(f"variable name {self.name!r} is not a Python identifier")

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "like": self.like, "units": self.units, "output": self.output}

    @classmethod
    def from_payload(cls, values: Mapping[str, Any]) -> "VariableDeclaration":
        return cls(
            name=str(values["name"]),
            like=str(values.get("like", "T")),
            units=str(values.get("units", "1")),
            output=bool(values.get("output", True)),
        )


@dataclass(frozen=True, slots=True)
class NodeConfiguration:
    """Everything about one node that decides what it executes, besides its slot.

    ``parameters`` are runtime tunables of a native process, or the
    ``fc.Property`` values of a Python process.  ``python_source`` is the
    complete source of a Python process's class.  ``kernels`` maps a stage's
    swappable kernels to what stands in them.  ``variables`` are the fields
    the process declares.
    """

    parameters: Mapping[str, Any] = field(default_factory=dict)
    python_source: str | None = None
    kernels: Mapping[str, KernelBinding] = field(default_factory=dict)
    variables: tuple[VariableDeclaration, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(
            self,
            "kernels",
            {
                str(name): (binding if isinstance(binding, KernelBinding)
                            else KernelBinding.from_payload(binding))
                for name, binding in dict(self.kernels).items()
            },
        )
        object.__setattr__(
            self,
            "variables",
            tuple(item if isinstance(item, VariableDeclaration)
                  else VariableDeclaration.from_payload(item) for item in self.variables),
        )
        names = [item.name for item in self.variables]
        if len(names) != len(set(names)):
            raise WorkflowEditError("a process declares each variable once")

    @property
    def is_empty(self) -> bool:
        return (not self.parameters and self.python_source is None
                and not self.replaced_kernels and not self.variables)

    @property
    def replaced_kernels(self) -> tuple[str, ...]:
        return tuple(sorted(name for name, binding in self.kernels.items() if binding.replaces))

    def canonical(self) -> dict[str, Any]:
        """The configuration as the hash sees it: only what changes execution."""

        return {
            "parameters": dict(self.parameters),
            "python_source": self.python_source,
            "kernels": {name: binding.to_payload() for name, binding in sorted(self.kernels.items())
                        if binding.replaces},
            "variables": [item.to_payload() for item in self.variables],
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "parameters": dict(self.parameters),
            "python_source": self.python_source,
            "kernels": {name: binding.to_payload() for name, binding in self.kernels.items()},
            "variables": [item.to_payload() for item in self.variables],
        }

    @classmethod
    def from_payload(cls, values: Mapping[str, Any] | None) -> "NodeConfiguration":
        values = dict(values or {})
        return cls(
            parameters=dict(values.get("parameters", {})),
            python_source=values.get("python_source"),
            kernels=dict(values.get("kernels", {})),
            variables=tuple(values.get("variables", ())),
        )

    def updated(self, changes: Mapping[str, Any]) -> "NodeConfiguration":
        """A copy with the given top-level fields replaced."""

        unknown = sorted(set(changes) - {"parameters", "python_source", "kernels", "variables"})
        if unknown:
            raise WorkflowEditError(f"configuration has no field {unknown}")
        payload = self.to_payload()
        payload.update(changes)
        return NodeConfiguration.from_payload(payload)


def _is_identifier(name: str) -> bool:
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    """One action in the complete ordered workflow."""

    id: str
    name: str
    qualified_name: str
    operation: str
    phase: str
    kind: str
    implementation: str = "python"
    enabled: bool = True
    movable: bool = True
    removable: bool = True
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    source: str | None = None
    parent_stage: str | None = None
    granularity: str = "stage"
    origin: str = "default"
    native_id: int | None = None
    default_index: int | None = None
    configuration: NodeConfiguration = field(default_factory=NodeConfiguration)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise WorkflowEditError("workflow node id cannot be empty")
        object.__setattr__(self, "reads", tuple(str(item) for item in self.reads))
        object.__setattr__(self, "writes", tuple(str(item) for item in self.writes))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not isinstance(self.configuration, NodeConfiguration):
            object.__setattr__(
                self, "configuration", NodeConfiguration.from_payload(self.configuration)
            )
        if self.origin not in {"default", "python", "catalog"}:
            raise WorkflowEditError(f"unknown workflow node origin {self.origin!r}")
        if self.origin == "python" and not _is_identifier(self.name):
            raise WorkflowEditError(
                f"Python process name {self.name!r} must be a Python identifier"
            )

    @property
    def locked(self) -> bool:
        return self.operation in LOCKED_OPERATIONS

    @property
    def scientific(self) -> bool:
        return self.kind in SCIENTIFIC_KINDS

    @property
    def control(self) -> bool:
        return self.kind in CONTROL_SKELETON_KINDS

    @property
    def display_name(self) -> str:
        return self.name.removesuffix("_leaf")

    @property
    def runtime_inserted(self) -> bool:
        """Whether this node is installed after ``driver.initialize()``."""

        return self.origin in {"catalog", "python"}

    @property
    def is_python(self) -> bool:
        return self.origin == "python"

    def configured(self, changes: Mapping[str, Any]) -> "WorkflowNode":
        return replace(self, configuration=self.configuration.updated(changes))

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "qualified_name": self.qualified_name,
            "operation": self.operation,
            "phase": self.phase,
            "kind": self.kind,
            "implementation": self.implementation,
            "enabled": self.enabled,
            "movable": self.movable,
            "removable": self.removable,
            "locked": self.locked,
            "scientific": self.scientific,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "source": self.source,
            "parent_stage": self.parent_stage,
            "granularity": self.granularity,
            "origin": self.origin,
            "native_id": self.native_id,
            "default_index": self.default_index,
            "configuration": self.configuration.to_payload(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, values: Mapping[str, Any]) -> "WorkflowNode":
        return cls(
            id=str(values["id"]),
            name=str(values["name"]),
            qualified_name=str(values["qualified_name"]),
            operation=str(values["operation"]),
            phase=str(values["phase"]),
            kind=str(values["kind"]),
            implementation=str(values.get("implementation", "python")),
            enabled=bool(values.get("enabled", True)),
            movable=bool(values.get("movable", True)),
            removable=bool(values.get("removable", True)),
            reads=tuple(values.get("reads", ())),
            writes=tuple(values.get("writes", ())),
            source=values.get("source"),
            parent_stage=values.get("parent_stage"),
            granularity=str(values.get("granularity", "stage")),
            origin=str(values.get("origin", "default")),
            native_id=(
                None if values.get("native_id") is None else int(values["native_id"])
            ),
            default_index=(
                None
                if values.get("default_index") is None
                else int(values["default_index"])
            ),
            configuration=NodeConfiguration.from_payload(values.get("configuration")),
            metadata=dict(values.get("metadata", {})),
        )


def python_node(
    name: str,
    *,
    source: str,
    phase: str = "cam_run1",
    parameters: Mapping[str, Any] | None = None,
    variables: Sequence[Mapping[str, Any] | VariableDeclaration] = (),
    reads: Sequence[str] = (),
    writes: Sequence[str] = (),
) -> WorkflowNode:
    """A Python process node; several may exist under different names."""

    if not _is_identifier(name):
        raise WorkflowEditError(f"Python process name {name!r} must be a Python identifier")
    return WorkflowNode(
        id=f"{PYTHON_NODE_PREFIX}{name}",
        name=name,
        qualified_name=f"{phase}.{name}",
        operation=name,
        phase=phase,
        kind="python_process",
        implementation="python",
        origin="python",
        reads=tuple(reads),
        writes=tuple(writes),
        configuration=NodeConfiguration(
            parameters=dict(parameters or {}), python_source=source, variables=tuple(variables)
        ),
    )


@dataclass(frozen=True, slots=True)
class WorkflowCatalogEntry:
    """A process that can (or cannot) be added to the workflow."""

    node: WorkflowNode
    category: str
    addable: bool = True
    reason: str | None = None
    in_default: bool = False
    description: str | None = None

    @property
    def id(self) -> str:
        return self.node.id

    def to_payload(self, *, present: bool = False) -> dict[str, Any]:
        payload = self.node.to_payload()
        payload.update(
            {
                "category": self.category,
                "addable": bool(self.addable and not present),
                "present": bool(present),
                "reason": (
                    "already in the workflow"
                    if present and self.addable
                    else self.reason
                ),
                "in_default": self.in_default,
                "description": self.description,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class WorkflowIssue:
    """One validation finding."""

    severity: str
    code: str
    message: str
    node_id: str | None = None
    field: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning", "info"}:
            raise ValueError(f"unknown issue severity {self.severity!r}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "node_id": self.node_id,
            "field": self.field,
        }


@dataclass(frozen=True, slots=True)
class WorkflowValidationReport:
    """Result of one check over a complete document."""

    revision: int
    workflow_hash: str
    issues: tuple[WorkflowIssue, ...]
    level: str = "browser"
    checks: Mapping[str, Any] = field(default_factory=dict)
    disclaimer: str = (
        "Passing the check means the structure and the declared constraints are "
        "satisfied.  It does not prove that a modified workflow is physically "
        "correct, stable over a long run, or bit-for-bit with the validated default."
    )

    @property
    def status(self) -> str:
        severities = {issue.severity for issue in self.issues}
        if "error" in severities:
            return "error"
        if "warning" in severities:
            return "warning"
        return "valid"

    @property
    def errors(self) -> tuple[WorkflowIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[WorkflowIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "level": self.level,
            "revision": self.revision,
            "workflow_hash": self.workflow_hash,
            "issues": [issue.to_payload() for issue in self.issues],
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks": dict(self.checks),
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True, slots=True)
class GeneratedWorkflowArtifact:
    """Files written by one Generate action."""

    name: str
    directory: Any
    files: Mapping[str, Any]
    workflow_hash: str
    manifest: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "directory": str(self.directory),
            "files": {key: str(value) for key, value in self.files.items()},
            "workflow_hash": self.workflow_hash,
            "manifest": dict(self.manifest),
        }


# ---------------------------------------------------------------------------
# the document
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkflowDocument:
    """Immutable ordered workflow; every edit returns a new document."""

    nodes: tuple[WorkflowNode, ...]
    revision: int = 0
    experimental: bool = False
    case: str = "PI-atm"
    nsteps: int = 2
    namelist: Mapping[str, Any] = field(default_factory=dict)
    catalog_version: str = ""
    source_version: str = ""

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "namelist", dict(self.namelist))
        if int(self.nsteps) < 1:
            raise WorkflowEditError("nsteps must be positive")
        object.__setattr__(self, "nsteps", int(self.nsteps))
        ids = [node.id for node in nodes]
        if len(ids) != len(set(ids)):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise WorkflowEditError(
                "workflow node ids must be unique: " + ", ".join(duplicates)
            )

    def __iter__(self) -> Iterator[WorkflowNode]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    @property
    def enabled_nodes(self) -> tuple[WorkflowNode, ...]:
        return tuple(node for node in self.nodes if node.enabled)

    @property
    def scientific_nodes(self) -> tuple[WorkflowNode, ...]:
        return tuple(node for node in self.nodes if node.scientific)

    @property
    def python_nodes(self) -> tuple[WorkflowNode, ...]:
        return tuple(node for node in self.nodes if node.is_python)

    def node(self, node_id: str) -> WorkflowNode:
        for node in self.nodes:
            if node.id == str(node_id):
                return node
        raise WorkflowEditError(f"workflow has no node {node_id!r}")

    def index(self, node_id: str) -> int:
        for index, node in enumerate(self.nodes):
            if node.id == str(node_id):
                return index
        raise WorkflowEditError(f"workflow has no node {node_id!r}")

    def __contains__(self, node_id: object) -> bool:
        return any(node.id == node_id for node in self.nodes)

    def execution_record(self) -> dict[str, Any]:
        """Everything the hash covers: what a run of this document executes."""

        return {
            "case": self.case,
            "nsteps": self.nsteps,
            "namelist": dict(self.namelist),
            "nodes": [
                [node.id, node.qualified_name, node.origin, bool(node.enabled),
                 node.configuration.canonical()]
                for node in self.nodes
            ],
        }

    @property
    def workflow_hash(self) -> str:
        """Deterministic digest of everything that decides what runs."""

        return sha256(_canonical(self.execution_record()).encode()).hexdigest()

    @property
    def order_hash(self) -> str:
        """Digest of order and membership only, for comparing with the default."""

        return sha256(_canonical([[n.id, bool(n.enabled)] for n in self.nodes]).encode()).hexdigest()

    def _with_nodes(self, nodes: Sequence[WorkflowNode]) -> "WorkflowDocument":
        return replace(self, nodes=tuple(nodes))

    def _target_index(
        self,
        *,
        before: str | None,
        after: str | None,
        index: int | None,
        moving: str | None = None,
    ) -> int:
        """Resolve an insertion slot in the list without ``moving``."""

        provided = sum(item is not None for item in (before, after, index))
        if provided != 1:
            raise WorkflowEditError("provide exactly one of before, after, or index")
        remaining = [node for node in self.nodes if node.id != moving]
        if index is not None:
            slot = int(index)
            if slot < 0:
                slot += len(remaining) + 1
            if not 0 <= slot <= len(remaining):
                raise WorkflowEditError(f"index {index} is outside the workflow")
            return slot
        anchor = str(before if before is not None else after)
        if anchor == moving:
            raise WorkflowEditError("a node cannot be placed relative to itself")
        for position, node in enumerate(remaining):
            if node.id == anchor:
                return position if before is not None else position + 1
        raise WorkflowEditError(f"workflow has no node {anchor!r}")

    def _check_slot(self, nodes: Sequence[WorkflowNode]) -> None:
        if not nodes:
            raise WorkflowEditError("workflow cannot be empty")
        if nodes[0].operation != "boundary_import":
            raise WorkflowEditError("workflow must start with boundary_import")
        if nodes[-1].operation != "boundary_export":
            raise WorkflowEditError("workflow must end with boundary_export")
        control = [node.id for node in nodes if node.control]
        current = [node.id for node in self.nodes if node.control and node.id in set(control)]
        if [item for item in control if item in set(current)] != current:
            raise WorkflowEditError("control actions keep their relative order")

    def move(
        self,
        node_id: str,
        *,
        before: str | None = None,
        after: str | None = None,
        index: int | None = None,
    ) -> "WorkflowDocument":
        node = self.node(node_id)
        if node.locked or not node.movable:
            raise WorkflowEditError(
                f"{node.qualified_name!r} is a required control action and cannot move"
            )
        slot = self._target_index(before=before, after=after, index=index, moving=node.id)
        remaining = [item for item in self.nodes if item.id != node.id]
        remaining.insert(slot, node)
        self._check_slot(remaining)
        return self._with_nodes(remaining)

    def remove(self, node_id: str) -> "WorkflowDocument":
        node = self.node(node_id)
        if node.locked or not node.removable:
            raise WorkflowEditError(
                f"{node.qualified_name!r} is a required control action and cannot be removed"
            )
        remaining = [item for item in self.nodes if item.id != node.id]
        self._check_slot(remaining)
        return self._with_nodes(remaining)

    def set_enabled(self, node_id: str, enabled: bool) -> "WorkflowDocument":
        node = self.node(node_id)
        if node.locked and not enabled:
            raise WorkflowEditError(
                f"{node.qualified_name!r} is a required control action and cannot be disabled"
            )
        return self._with_nodes(
            tuple(
                replace(item, enabled=bool(enabled)) if item.id == node.id else item
                for item in self.nodes
            )
        )

    def insert(
        self,
        node: WorkflowNode,
        *,
        before: str | None = None,
        after: str | None = None,
        index: int | None = None,
    ) -> "WorkflowDocument":
        if node.id in self:
            raise WorkflowEditError(f"{node.qualified_name!r} is already in the workflow")
        slot = self._target_index(before=before, after=after, index=index)
        nodes = list(self.nodes)
        nodes.insert(slot, node)
        self._check_slot(nodes)
        return self._with_nodes(nodes)

    def replace_node(self, node_id: str, node: WorkflowNode) -> "WorkflowDocument":
        """Remove ``node_id`` and put ``node`` in the same slot.

        Nothing of the old node's configuration carries over: a replacement
        starts from the catalog's own defaults.
        """

        old = self.node(node_id)
        if old.locked or not old.removable:
            raise WorkflowEditError(
                f"{old.qualified_name!r} is a required control action and cannot be replaced"
            )
        if node.id in self and node.id != old.id:
            raise WorkflowEditError(f"{node.qualified_name!r} is already in the workflow")
        nodes = list(self.nodes)
        nodes[self.index(node_id)] = node
        self._check_slot(nodes)
        return self._with_nodes(nodes)

    def configure(self, node_id: str, changes: Mapping[str, Any]) -> "WorkflowDocument":
        node = self.node(node_id)
        if node.locked:
            raise WorkflowEditError(
                f"{node.qualified_name!r} is a required control action and has no configuration"
            )
        return self._with_nodes(
            tuple(node.configured(changes) if item.id == node.id else item for item in self.nodes)
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "revision": self.revision,
            "experimental": self.experimental,
            "case": self.case,
            "nsteps": self.nsteps,
            "namelist": dict(self.namelist),
            "catalog_version": self.catalog_version,
            "source_version": self.source_version,
            "workflow_hash": self.workflow_hash,
            "nodes": [node.to_payload() for node in self.nodes],
        }

    @classmethod
    def from_payload(cls, values: Mapping[str, Any]) -> "WorkflowDocument":
        version = int(values.get("schema_version", 1))
        if version not in (1, SCHEMA_VERSION):
            raise WorkflowEditError(
                f"unsupported workflow document schema {version}; this build reads "
                f"versions 1 and {SCHEMA_VERSION}"
            )
        return cls(
            nodes=tuple(WorkflowNode.from_payload(item) for item in values["nodes"]),
            revision=int(values.get("revision", 0)),
            experimental=bool(values.get("experimental", False)),
            case=str(values.get("case", "PI-atm")),
            nsteps=int(values.get("nsteps", 2)),
            namelist=dict(values.get("namelist", {})),
            catalog_version=str(values.get("catalog_version", "")),
            source_version=str(values.get("source_version", "")),
        )


CatalogLookup = Callable[[str], WorkflowCatalogEntry]


class WorkflowEditSession:
    """Mutable owner of the current document, its history, and its revision."""

    def __init__(
        self,
        default_document: WorkflowDocument,
        catalog: Mapping[str, WorkflowCatalogEntry],
        *,
        history_limit: int = 200,
        python_template: Callable[[str, str | None], str] | None = None,
    ) -> None:
        self._default = replace(default_document, revision=0)
        self._document = self._default
        self._catalog = dict(catalog)
        self._undo: list[WorkflowDocument] = []
        self._redo: list[WorkflowDocument] = []
        self._history_limit = int(history_limit)
        self._lock = threading.RLock()
        self._listeners: list[Callable[[WorkflowDocument], None]] = []
        self._python_template = python_template

    @property
    def default_document(self) -> WorkflowDocument:
        return self._default

    @property
    def document(self) -> WorkflowDocument:
        with self._lock:
            return self._document

    @property
    def catalog(self) -> Mapping[str, WorkflowCatalogEntry]:
        return dict(self._catalog)

    @property
    def can_undo(self) -> bool:
        with self._lock:
            return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        with self._lock:
            return bool(self._redo)

    @property
    def is_default(self) -> bool:
        with self._lock:
            return self._document.workflow_hash == self._default.workflow_hash

    def catalog_payload(self) -> list[dict[str, Any]]:
        """Catalog entries with their availability for the current document."""

        with self._lock:
            present = set(self._document.ids)
        return [
            entry.to_payload(present=entry.id in present)
            for entry in self._catalog.values()
        ]

    def entry(self, entry_id: str) -> WorkflowCatalogEntry:
        try:
            return self._catalog[str(entry_id)]
        except KeyError as exc:
            raise WorkflowEditError(f"catalog has no process {entry_id!r}") from exc

    def subscribe(self, listener: Callable[[WorkflowDocument], None]) -> None:
        self._listeners.append(listener)

    def _commit(self, new: WorkflowDocument, *, record: bool = True) -> WorkflowDocument:
        current = self._document
        new = replace(new, revision=current.revision + 1)
        if record:
            self._undo.append(current)
            del self._undo[: -self._history_limit]
            self._redo.clear()
        self._document = new
        for listener in tuple(self._listeners):
            listener(new)
        return new

    def _check_revision(self, revision: Any) -> None:
        if revision is None:
            return
        if int(revision) != self._document.revision:
            raise RevisionConflict(int(revision), self._document.revision)

    def _node_for_entry(self, entry_id: str, *, document: WorkflowDocument) -> WorkflowNode:
        entry = self.entry(entry_id)
        if entry.id in document:
            raise WorkflowEditError(f"{entry.node.qualified_name!r} is already in the workflow")
        if not entry.addable:
            raise WorkflowEditError(
                f"{entry.node.qualified_name!r} cannot be added: {entry.reason}"
            )
        return replace(entry.node, enabled=True)

    def _new_python_node(self, edit: Mapping[str, Any], document: WorkflowDocument) -> WorkflowNode:
        name = str(edit.get("name", "")).strip()
        if not _is_identifier(name):
            raise WorkflowEditError(f"Python process name {name!r} must be a Python identifier")
        node_id = f"{PYTHON_NODE_PREFIX}{name}"
        if node_id in document or any(node.name == name for node in document.nodes):
            raise WorkflowEditError(f"a process named {name!r} is already in the workflow")
        source = edit.get("source")
        if source is None:
            anchor = edit.get("after") or edit.get("before")
            anchor_name = None if anchor is None else document.node(str(anchor)).display_name
            if self._python_template is None:
                raise WorkflowEditError("no Python process template is configured")
            source = self._python_template(name, anchor_name)
        return python_node(
            name,
            source=str(source),
            phase=str(edit.get("phase", "cam_run1")),
            parameters=edit.get("parameters"),
            variables=tuple(edit.get("variables", ())),
        )

    def apply(self, edit: Mapping[str, Any]) -> WorkflowDocument:
        """Apply one structured edit and return the new document."""

        operation = str(edit.get("operation", "")).strip()
        with self._lock:
            self._check_revision(edit.get("revision"))
            document = self._document
            placement = {
                "before": edit.get("before"),
                "after": edit.get("after"),
                "index": edit.get("index"),
            }
            if operation == "move":
                new = document.move(str(edit["node_id"]), **placement)
            elif operation == "remove":
                new = document.remove(str(edit["node_id"]))
            elif operation in {"enable", "disable"}:
                new = document.set_enabled(str(edit["node_id"]), operation == "enable")
            elif operation == "set_enabled":
                new = document.set_enabled(str(edit["node_id"]), bool(edit["enabled"]))
            elif operation == "add":
                node = self._node_for_entry(str(edit["entry_id"]), document=document)
                if all(value is None for value in placement.values()):
                    placement["before"] = self._default_anchor(node, document)
                new = document.insert(node, **placement)
            elif operation == "add_python":
                node = self._new_python_node(edit, document)
                if all(value is None for value in placement.values()):
                    placement["before"] = document.nodes[-1].id
                new = document.insert(node, **placement)
            elif operation == "restore":
                node = self._node_for_entry(str(edit["node_id"]), document=document)
                if all(value is None for value in placement.values()):
                    placement["before"] = self._default_anchor(node, document)
                new = document.insert(node, **placement)
            elif operation == "replace":
                if "entry_id" in edit:
                    node = self._node_for_entry(str(edit["entry_id"]), document=document)
                else:
                    node = self._new_python_node(edit, document)
                new = document.replace_node(str(edit["node_id"]), node)
            elif operation == "configure":
                new = document.configure(str(edit["node_id"]), dict(edit["configuration"]))
            elif operation == "set_experimental":
                new = replace(document, experimental=bool(edit["experimental"]))
            elif operation == "set_case":
                new = replace(document, case=str(edit["case"]))
            elif operation == "set_nsteps":
                new = replace(document, nsteps=int(edit["nsteps"]))
            elif operation == "set_namelist":
                new = replace(document, namelist=dict(edit["namelist"]))
            elif operation == "reorder":
                new = self._reorder(document, tuple(str(item) for item in edit["order"]))
            elif operation == "undo":
                return self.undo()
            elif operation == "redo":
                return self.redo()
            elif operation == "reset":
                return self.reset()
            else:
                raise WorkflowEditError(f"unknown workflow edit {operation!r}")
            return self._commit(new)

    def import_document(self, payload: Mapping[str, Any]) -> WorkflowDocument:
        """Replace the draft with an imported document, as one undoable step.

        Nodes the catalog knows are refreshed from it -- their metadata,
        movability and default slot are the catalog's -- while keeping the
        imported enabled flag and configuration.  Python nodes travel whole.
        """

        imported = WorkflowDocument.from_payload(payload)
        nodes: list[WorkflowNode] = []
        for node in imported.nodes:
            entry = self._catalog.get(node.id)
            if entry is not None:
                nodes.append(replace(entry.node, enabled=node.enabled,
                                     configuration=node.configuration))
            elif node.is_python:
                nodes.append(node)
            else:
                raise WorkflowEditError(
                    f"imported workflow names {node.qualified_name!r}, which this "
                    f"catalog does not have"
                )
        imported._check_slot(nodes)
        with self._lock:
            return self._commit(replace(imported, nodes=tuple(nodes)))

    def _reorder(
        self, document: WorkflowDocument, order: tuple[str, ...]
    ) -> WorkflowDocument:
        """Replace the complete order with ``order``; membership must match."""

        if sorted(order) != sorted(document.ids):
            raise WorkflowEditError("reorder must list every current node exactly once")
        by_id = {node.id: node for node in document.nodes}
        nodes = [by_id[node_id] for node_id in order]
        for node in nodes:
            if (node.locked or not node.movable) and document.index(node.id) != order.index(node.id):
                raise WorkflowEditError(
                    f"{node.qualified_name!r} is a required control action and cannot move"
                )
        document._check_slot(nodes)
        return document._with_nodes(nodes)

    def _default_anchor(self, node: WorkflowNode, document: WorkflowDocument) -> str:
        """Choose the node that follows ``node``'s slot in the default order."""

        if node.default_index is not None:
            later = sorted(
                (
                    item
                    for item in document.nodes
                    if item.default_index is not None
                    and item.default_index > node.default_index
                ),
                key=lambda item: item.default_index or 0,
            )
            if later:
                return later[0].id
        return document.nodes[-1].id

    def undo(self) -> WorkflowDocument:
        with self._lock:
            if not self._undo:
                raise WorkflowEditError("nothing to undo")
            previous = self._undo.pop()
            self._redo.append(self._document)
            return self._commit(previous, record=False)

    def redo(self) -> WorkflowDocument:
        with self._lock:
            if not self._redo:
                raise WorkflowEditError("nothing to redo")
            following = self._redo.pop()
            self._undo.append(self._document)
            return self._commit(following, record=False)

    def reset(self) -> WorkflowDocument:
        with self._lock:
            return self._commit(
                replace(self._default, experimental=self._document.experimental)
            )

    def state_payload(self) -> dict[str, Any]:
        with self._lock:
            payload = self._document.to_payload()
            payload.update(
                {
                    "can_undo": bool(self._undo),
                    "can_redo": bool(self._redo),
                    "is_default": self.is_default,
                    "default_hash": self._default.workflow_hash,
                }
            )
            return payload


__all__ = [
    "CONTROL_SKELETON_KINDS",
    "GeneratedWorkflowArtifact",
    "KERNEL_BINDING_KINDS",
    "KernelBinding",
    "LOCKED_OPERATIONS",
    "NodeConfiguration",
    "PYTHON_NODE_PREFIX",
    "RevisionConflict",
    "SCHEMA_VERSION",
    "SCIENTIFIC_KINDS",
    "VariableDeclaration",
    "WorkflowCatalogEntry",
    "WorkflowDocument",
    "WorkflowEditError",
    "WorkflowEditSession",
    "WorkflowIssue",
    "WorkflowNode",
    "WorkflowValidationReport",
    "js_number",
    "python_node",
]
