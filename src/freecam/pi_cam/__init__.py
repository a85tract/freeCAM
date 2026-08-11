"""Python-owned control layer for the admitted iCESM PI-CAM case."""

from .boundary import (
    BoundaryManifest,
    CAMBoundaryProvider,
    InMemoryBoundaryProvider,
    ReplayBoundaryProvider,
    write_boundary_payload,
)
from .case import PICAMCase
from .config import PICAMConfig
from .facade import Driver, Physics, PICAMCaseInfo, Variable
from .driver import PICAMActionTrace, PICAMDriver, PICAMLifecycle
from .errors import (
    BoundaryReplayError,
    NativeCAMError,
    PICAMConfigurationError,
    PICAMError,
    PICAMStateError,
)
from .native import CAMNumericalBackend, NativeCAMDevice, RecordingCAMBackend
from .plan import PICAMAction, PICAMStepPlan
from .session import PICAMNotebookError, PICAMNotebookSession
from .ui import PICAMStateView, PICAMWorkflowAction, PICAMWorkflowView
from .state import PICAMFieldContract, PICAMStatePool, PICAMStateSchema
from .state import PICAMVariableSpec
from .runtime_fortran import PICAMFortranProcessSpec
from freecam.model.python_processes import PythonProcessContext, PythonProcessSpec
from .validation import PICAMBFBResult, compare_pi_cam_directories
from .source_catalog import (
    FortranArgument,
    FortranParseFailure,
    FortranProcedure,
    PICAMKernelRules,
    PICAMSourceCatalog,
)
from .physics_catalog import (
    PICAMPhysicsCatalog,
    PICAMPhysicsProcess,
    PICAMPhysicsRules,
    build_physics_catalog,
)
from .process_context import (
    PICAMProcessArgumentBinding,
    PICAMProcessContextRegistry,
    PICAMPromotedProcess,
)
from .process_codegen import generated_promoted_kernels, statepool_promotable
from .adapter_validation import (
    ABISmokeResult,
    AdapterBuildContext,
    AdapterCompileAttempt,
    AdapterCompileResult,
    PICAMAdapterValidator,
    load_adapter_build_contexts,
)

__all__ = [
    "BoundaryManifest",
    "BoundaryReplayError",
    "CAMBoundaryProvider",
    "CAMNumericalBackend",
    "Driver",
    "InMemoryBoundaryProvider",
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
    "PICAMPhysicsCatalog",
    "PICAMPhysicsProcess",
    "PICAMPhysicsRules",
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
    "PICAMVariableSpec",
    "PICAMFortranProcessSpec",
    "PythonProcessContext",
    "PythonProcessSpec",
    "RecordingCAMBackend",
    "ReplayBoundaryProvider",
    "write_boundary_payload",
    "Variable",
    "compare_pi_cam_directories",
    "build_physics_catalog",
    "generated_promoted_kernels",
    "load_adapter_build_contexts",
    "statepool_promotable",
]
