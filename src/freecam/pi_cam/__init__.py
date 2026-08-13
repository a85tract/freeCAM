"""Python-owned control layer for the admitted iCESM PI-CAM case."""

from freecam.model.python_processes import (
    PythonProcessContext,
    PythonProcessSpec,
    PythonStateView,
)

from .adapter_validation import (
    ABISmokeResult,
    AdapterBuildContext,
    AdapterCompileAttempt,
    AdapterCompileResult,
    PICAMAdapterValidator,
    load_adapter_build_contexts,
)
from .boundary import (
    BoundaryManifest,
    CAMBoundaryProvider,
    CESMOnlineBoundaryProvider,
    HeldSurfaceModel,
    InMemoryBoundaryProvider,
    OnlineBoundaryContext,
    OnlineBoundaryFields,
    OnlineBoundaryProvider,
    ReplayBoundaryProvider,
    prepare_cesm_online_run,
    write_boundary_payload,
)
from .case import PICAMCase
from .config import PICAMConfig
from .driver import PICAMActionTrace, PICAMDriver, PICAMLifecycle
from .errors import (
    BoundaryReplayError,
    NativeCAMError,
    PICAMConfigurationError,
    PICAMError,
    PICAMStateError,
)
from .facade import (
    CASES,
    CaseConfig,
    CaseRegistry,
    Driver,
    FreeCAM,
    Physics,
    ProcessSpec,
    PICAMCaseInfo,
    RunHandle,
    RunProgress,
    RunResult,
    Variable,
    WorkflowFactory,
    WorkflowPreview,
    WorkflowPreviewAction,
    WorkflowTemplate,
    process,
)
from .history import PICAMOutputView
from .native import CAMNumericalBackend, NativeCAMDevice, RecordingCAMBackend
from .physics_catalog import (
    PICAMPhysicsCatalog,
    PICAMPhysicsProcess,
    PICAMPhysicsRules,
    build_physics_catalog,
)
from .plan import PICAMAction, PICAMStepPlan
from .process_codegen import generated_promoted_kernels, statepool_promotable
from .process_context import (
    PICAMProcessArgumentBinding,
    PICAMProcessContextRegistry,
    PICAMPromotedProcess,
)
from .runtime_fortran import PICAMFortranProcessSpec
from .session import PICAMNotebookError, PICAMNotebookSession
from .source_catalog import (
    FortranArgument,
    FortranParseFailure,
    FortranProcedure,
    PICAMKernelRules,
    PICAMSourceCatalog,
)
from .state import (
    PICAMFieldContract,
    PICAMStatePool,
    PICAMStateSchema,
    PICAMVariableSpec,
)
from .ui import (
    PICAMProfilePlot,
    PICAMStateView,
    PICAMWorkflowAction,
    PICAMWorkflowView,
)
from .validation import PICAMBFBResult, compare_pi_cam_directories

__all__ = [
    "BoundaryManifest",
    "BoundaryReplayError",
    "CAMBoundaryProvider",
    "CESMOnlineBoundaryProvider",
    "CAMNumericalBackend",
    "CASES",
    "CaseConfig",
    "CaseRegistry",
    "Driver",
    "FreeCAM",
    "HeldSurfaceModel",
    "InMemoryBoundaryProvider",
    "OnlineBoundaryContext",
    "OnlineBoundaryFields",
    "OnlineBoundaryProvider",
    "prepare_cesm_online_run",
    "FortranArgument",
    "FortranParseFailure",
    "FortranProcedure",
    "ABISmokeResult",
    "AdapterBuildContext",
    "AdapterCompileAttempt",
    "AdapterCompileResult",
    "NativeCAMDevice",
    "NativeCAMError",
    "PICAMAction",
    "PICAMActionTrace",
    "PICAMAdapterValidator",
    "PICAMBFBResult",
    "PICAMCase",
    "PICAMCaseInfo",
    "PICAMConfig",
    "PICAMConfigurationError",
    "PICAMDriver",
    "PICAMError",
    "PICAMFieldContract",
    "PICAMLifecycle",
    "PICAMKernelRules",
    "PICAMNotebookError",
    "PICAMNotebookSession",
    "PICAMOutputView",
    "PICAMPhysicsCatalog",
    "PICAMPhysicsProcess",
    "PICAMPhysicsRules",
    "PICAMProfilePlot",
    "PICAMProcessArgumentBinding",
    "PICAMProcessContextRegistry",
    "PICAMPromotedProcess",
    "PICAMStateError",
    "PICAMStatePool",
    "PICAMStateSchema",
    "PICAMSourceCatalog",
    "PICAMStepPlan",
    "PICAMStateView",
    "PICAMWorkflowAction",
    "PICAMWorkflowView",
    "Physics",
    "ProcessSpec",
    "RunHandle",
    "RunProgress",
    "RunResult",
    "PICAMVariableSpec",
    "PICAMFortranProcessSpec",
    "PythonProcessContext",
    "PythonProcessSpec",
    "PythonStateView",
    "RecordingCAMBackend",
    "ReplayBoundaryProvider",
    "write_boundary_payload",
    "Variable",
    "WorkflowFactory",
    "WorkflowPreview",
    "WorkflowPreviewAction",
    "WorkflowTemplate",
    "process",
    "compare_pi_cam_directories",
    "build_physics_catalog",
    "generated_promoted_kernels",
    "load_adapter_build_contexts",
    "statepool_promotable",
]
