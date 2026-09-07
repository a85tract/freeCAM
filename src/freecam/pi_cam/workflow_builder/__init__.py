"""The Workflow Builder: an editable document of the step, and what runs it.

``document`` is the data model, ``catalog`` builds the default workflow and
the process library from the model's own records, ``validate`` checks a
document, and ``templates`` holds the source a new Python process starts
from.  The local service and the ``Driver.ui()`` entry live in ``server``
and ``ui`` and import their web dependencies only when used.
"""

from .capabilities import KernelCapability, kernel_capabilities
from .catalog import CASES, build_snapshot, catalog_entries, default_document, load_catalog
from .document import (
    KernelBinding,
    NodeConfiguration,
    RevisionConflict,
    VariableDeclaration,
    WorkflowCatalogEntry,
    WorkflowDocument,
    WorkflowEditError,
    WorkflowEditSession,
    WorkflowIssue,
    WorkflowNode,
    WorkflowValidationReport,
    python_node,
)
from .templates import python_process_template
from .validate import validate_document

__all__ = [
    "CASES",
    "KernelBinding",
    "KernelCapability",
    "NodeConfiguration",
    "RevisionConflict",
    "VariableDeclaration",
    "WorkflowCatalogEntry",
    "WorkflowDocument",
    "WorkflowEditError",
    "WorkflowEditSession",
    "WorkflowIssue",
    "WorkflowNode",
    "WorkflowValidationReport",
    "build_snapshot",
    "catalog_entries",
    "default_document",
    "kernel_capabilities",
    "load_catalog",
    "python_node",
    "python_process_template",
    "validate_document",
]
