"""Apply a workflow document to a live model, through the public interface.

This is the one place the builder turns a document into calls on a
``Driver``, and it makes the same calls, in the same order, as the code the
page generates: fields first, then Python processes in their places, then
catalog processes, then a stage class with a model in a kernel's slot, then
the order, then the tunables.  The service runs this; the exported script
spells the same thing out; a test runs both against one fake driver and
compares what they did.

A second application applies only the difference against what was applied
before: a new Python process is inserted, a changed one reloaded, a removed
one uninstalled; flags and order are touched only where they changed.  What
cannot change on a live model -- the case, the namelist, a kernel binding
already attached -- is refused with :class:`RestartRequired`, never applied
quietly.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .document import WorkflowDocument, WorkflowEditError, WorkflowNode

#: Stage classes the builder knows how to attach, by workflow action, with the
#: constructor keyword that takes each kernel's model path.
STAGE_CLASSES: Mapping[str, tuple[str, str, Mapping[str, str]]] = {
    "cam_run1.cloud_macro_microphysics": (
        "freecam.physics.cloud_macro_microphysics",
        "CloudMacroMicrophysics",
        {"mmacro_pcond": "macro_surrogate"},
    ),
}


class RestartRequired(RuntimeError):
    """The change cannot be made on the live model; close it and start again."""


class ApplyError(RuntimeError):
    """The document could not be applied; the model is where the report says."""


@dataclass(slots=True)
class AppliedState:
    """What a live model currently has from the builder."""

    document: WorkflowDocument
    python_processes: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, Any] = field(default_factory=dict)
    variables: set[str] = field(default_factory=set)
    log: list[str] = field(default_factory=list)


def _scientific_order(document: WorkflowDocument) -> list[str]:
    return [node.name for node in document.nodes if node.scientific and node.enabled]


def _original_order(document: WorkflowDocument) -> list[str]:
    """The original processes only: an inserted process is already in its place."""

    return [node.name for node in document.nodes if node.scientific and node.enabled and node.origin == "default"]


def _neighbours(document: WorkflowDocument, node: WorkflowNode) -> tuple[WorkflowNode | None, WorkflowNode | None]:
    index = document.index(node.id)
    before = next((n for n in reversed(document.nodes[:index]) if n.scientific and n.enabled), None)
    after = next((n for n in document.nodes[index + 1:] if n.scientific and n.enabled), None)
    return before, after


def instantiate_python(node: WorkflowNode, namespace: Mapping[str, Any] | None = None) -> Any:
    """Build the process object a Python node describes.

    The source is executed in a namespace holding ``fc`` (the freecam
    package); the class it defines is instantiated and its declared
    properties set from the node's parameters.  This is the only place the
    builder runs user code, and it runs only from Run.
    """

    import freecam as fc

    source = node.configuration.python_source or ""
    tree = ast.parse(source)
    classes = [item.name for item in tree.body if isinstance(item, ast.ClassDef)]
    if not classes:
        raise ApplyError(f"{node.name}: the source defines no class")
    scope: dict[str, Any] = {"fc": fc, "__name__": f"freecam_workflow_{node.name}"}
    scope.update(namespace or {})
    exec(compile(source, f"<workflow:{node.name}>", "exec"), scope)   # noqa: S102 -- the user's own process, on Run
    klass = scope[classes[-1]]
    if not isinstance(klass, type) or not issubclass(klass, fc.Physics):
        raise ApplyError(f"{node.name}: {classes[-1]} is not an fc.Physics subclass")
    process = klass()
    declared = getattr(klass, "name", None)
    if declared is not None and declared != node.name:
        raise ApplyError(f"{node.name}: the class declares name = {declared!r}")
    process.name = node.name
    for key, value in node.configuration.parameters.items():
        if key not in type(process)._declared_properties():
            raise ApplyError(f"{node.name}: the class declares no property {key!r}")
        setattr(process, key, value)
    return process


def _stage_for(node: WorkflowNode) -> Any:
    spec = STAGE_CLASSES.get(node.id)
    if spec is None:
        raise ApplyError(f"no stage class is wired for {node.display_name}")
    module_name, class_name, kwarg_by_kernel = spec
    module = __import__(module_name, fromlist=[class_name])
    klass = getattr(module, class_name)
    kwargs: dict[str, Any] = {}
    for kernel, binding in node.configuration.kernels.items():
        if not binding.replaces:
            continue
        kwarg = kwarg_by_kernel.get(kernel)
        if kwarg is None:
            raise ApplyError(f"{kernel} has no binding through {class_name}")
        kwargs[kwarg] = binding.path
    return klass(**kwargs)


def _bindings_of(document: WorkflowDocument) -> dict[str, dict[str, Any]]:
    return {
        node.id: {k: b.to_payload() for k, b in node.configuration.kernels.items() if b.replaces}
        for node in document.nodes
        if node.configuration.replaced_kernels
    }


def apply_document(driver: Any, document: WorkflowDocument, applied: AppliedState | None = None, *,
                   default: WorkflowDocument | None = None) -> AppliedState:
    """Make the live model run ``document``; return what it now has.

    ``applied`` is the previous application, or None for the first.  On the
    first application the order is compared with ``default`` -- the catalog's
    default workflow, which is what the generated code compares against too
    -- and replaced only where it differs, so the validated default asks
    nothing of the model.  The model must be initialized already; nothing
    here starts it.
    """

    cam = driver.cam
    workflow = cam.workflow
    state = AppliedState(
        document=document,
        python_processes=dict(applied.python_processes) if applied else {},
        stages=dict(applied.stages) if applied else {},
        variables=set(applied.variables) if applied else set(),
    )
    log = state.log
    previous = applied.document if applied else None

    if previous is not None:
        if document.case != previous.case:
            raise RestartRequired("the case changed; close the model and start again")
        if dict(document.namelist) != dict(previous.namelist):
            raise RestartRequired("the namelist changed; it is read at initialization, so close the model and start again")
        if _bindings_of(document) != _bindings_of(previous):
            raise RestartRequired("a kernel binding changed on a stage that is already attached; close the model and start again")

    # 1. fields the Python processes declare
    for node in document.nodes:
        if node.origin != "python" or not node.enabled:
            continue
        for variable in node.configuration.variables:
            if variable.name in state.variables:
                continue
            kwargs: dict[str, Any] = {"like": variable.like, "units": variable.units}
            if not variable.output:
                kwargs["output"] = False
            cam.state.create(variable.name, **kwargs)
            state.variables.add(variable.name)
            log.append(f"state.create({variable.name!r})")

    # 2. Python processes: new ones inserted, changed ones reloaded, gone ones removed
    current_python = {node.name: node for node in document.python_nodes}
    previous_python = {node.name: node for node in previous.python_nodes} if previous else {}
    for name in previous_python:
        if name not in current_python:
            workflow[name].remove()
            state.python_processes.pop(name, None)
            log.append(f"workflow[{name!r}].remove()")
    for node in document.python_nodes:
        old = previous_python.get(node.name)
        if old is None:
            process = instantiate_python(node)
            before, after = _neighbours(document, node)
            placement: dict[str, Any] = {}
            if before is not None:
                placement["after"] = before.name
            elif after is not None:
                placement["before"] = after.name
            workflow.insert(process, **placement)
            state.python_processes[node.name] = process
            log.append(f"workflow.insert({node.name}, {placement})")
            if not node.enabled:
                workflow[node.name].disable()
                log.append(f"workflow[{node.name!r}].disable()")
            continue
        if old.configuration.python_source != node.configuration.python_source:
            process = instantiate_python(node)
            workflow[node.name].reload(process)
            state.python_processes[node.name] = process
            log.append(f"workflow[{node.name!r}].reload(...)")
        elif old.configuration.parameters != node.configuration.parameters:
            handle = workflow[node.name]
            for key, value in node.configuration.parameters.items():
                handle.properties[key] = value
                log.append(f"workflow[{node.name!r}].properties[{key!r}] = {value!r}")
        if old.enabled != node.enabled:
            (workflow[node.name].enable if node.enabled else workflow[node.name].disable)()
            log.append(f"workflow[{node.name!r}].{'enable' if node.enabled else 'disable'}()")

    # 3. catalog processes, bound on the live model
    previous_catalog = {node.name for node in previous.nodes if node.origin == "catalog"} if previous else set()
    for node in document.nodes:
        if node.origin != "catalog" or not node.enabled or node.name in previous_catalog:
            continue
        before, after = _neighbours(document, node)
        placement = {}
        if before is not None:
            placement["after"] = before.name
        elif after is not None:
            placement["before"] = after.name
        cam.physics.process(node.name).insert(**placement)
        log.append(f"physics.process({node.name!r}).insert({placement})")

    # 4. a stage class where a kernel is replaced (first application only; changes need a restart)
    if previous is None:
        for node in document.nodes:
            if not node.configuration.replaced_kernels:
                continue
            stage = _stage_for(node)
            stage.attach(cam)
            state.stages[node.id] = stage
            log.append(f"{type(stage).__name__}(...).attach(driver.cam)")

    # 5. the order and membership of the original processes
    order = _scientific_order(document)
    if previous is not None:
        reference = _original_order(previous)
    elif default is not None:
        reference = _original_order(default)
    else:
        reference = _default_order(driver, document)
    if _original_order(document) != reference:
        workflow.replace(list(order))
        log.append(f"workflow.replace({order})")

    # 6. runtime tunables
    for node in document.nodes:
        if node.origin != "default":
            continue
        old_values = previous.node(node.id).configuration.parameters if previous is not None and node.id in previous else {}
        for key in sorted(node.configuration.parameters):
            value = node.configuration.parameters[key]
            if old_values.get(key) == value:
                continue
            cam.parameters[key] = value
            log.append(f"parameters[{key!r}] = {value!r}")

    return state


def _default_order(driver: Any, document: WorkflowDocument) -> list[str]:
    """The model's current scientific order, when no default document was given."""

    try:
        rows = driver.cam.workflow.describe()
    except Exception:
        rows = ()
    names = [str(row["name"]) for row in rows if row.get("enabled", True)]
    if names:
        return names
    # fall back to the document's own default indices
    return [node.name for node in sorted(
        (n for n in document.nodes if n.origin == "default" and n.scientific and n.default_index is not None),
        key=lambda n: n.default_index or 0,
    ) if node.enabled]


def model_calls(state: AppliedState | None) -> dict[str, int]:
    """How often the replaced kernels were answered by their models, from the stages."""

    counts: dict[str, int] = {}
    if state is None:
        return counts
    for node_id, stage in state.stages.items():
        execution = getattr(stage, "execution", None)
        if execution is None:
            continue
        described = execution.describe() if hasattr(execution, "describe") else {}
        counts[node_id] = int(described.get("python_model_calls", 0))
    return counts


__all__ = ["ApplyError", "AppliedState", "RestartRequired", "STAGE_CLASSES", "apply_document", "instantiate_python", "model_calls"]
