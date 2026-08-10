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
    "PICAMConfig",
    "PICAMConfigurationError",
    "PICAMDriver",
    "PICAMError",
    "PICAMFieldContract",
    "PICAMLifecycle",
    "PICAMKernelRules",
    "PICAMNotebookError",
    "PICAMNotebookSession",
    "PICAMStateError",
    "PICAMStatePool",
    "PICAMStateSchema",
    "PICAMSourceCatalog",
    "PICAMStepPlan",
    "PICAMVariableSpec",
    "PICAMFortranProcessSpec",
    "PythonProcessContext",
    "PythonProcessSpec",
    "RecordingCAMBackend",
    "ReplayBoundaryProvider",
    "write_boundary_payload",
    "compare_pi_cam_directories",
    "load_adapter_build_contexts",
]
