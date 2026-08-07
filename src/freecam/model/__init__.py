"""Python-owned CAM model and generic CCPP device runtime.

The public Notebook controller lives in :mod:`freecam.notebook.session`.
This package contains the rank-local implementation used by MPI workers.
"""

from .capabilities import CAM_SE_FVM_V1, RuntimeCapabilities
from .checkpoint import (
    CheckpointBundle,
    ModelSnapshot,
    read_checkpoint,
    restore_driver,
)
from .ccpp_suite import (
    CCPPDeviceHost,
    CCPPSuitePlan,
    DEFAULT_PHYSICS_GROUPS,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
    SuiteNode,
    SuiteScheme,
)
from .ccpp_state import (
    CCPPFieldRequirement,
    CCPPStateSchema,
    FieldVariant,
)
from .config import ModelConfig
from .control import ModelOptions, ModelParameters
from .contracts import FieldContract, default_contracts, export_contract
from .devices import DeviceRegistry, FortranDevice
from .device_catalog import DeviceCatalog, SchemeCatalogEntry
from .device_support import DeviceSupportMatrix, SchemeSupport
from .errors import DeviceBuildError, DeviceContractError
from .driver import CAMDriver, DriverState
from .experiment import (
    Action,
    ActivatePhysics,
    BranchSpec,
    DeactivatePhysics,
    DefineVariable,
    FieldEdit,
    InstallPhysics,
    InstallPythonProcess,
    MoveScheme,
    ObserveFields,
    PrepareInitialStep,
    RunPhase,
    RunScheme,
    RunSchemeGroup,
    RunSteps,
    RemovePythonProcess,
    SchemeMove,
    SegmentPlan,
    SetSchemeEnabled,
    execute_segment_plan,
)
from .grid import (
    homme_space_curve,
    sfc_partition_counts,
    sfc_partition_owner,
)
from .state import NativeObjectHandle, StatePool
from .host_services import (
    HostServiceEvent,
    HostServiceRegistry,
    HistoryObservation,
    PythonHistoryService,
)
from .plugins import (
    InstalledPhysicsPlugin,
    PhysicsPluginManager,
    PhysicsPluginSpec,
    SchemePlacement,
    VariableSpec,
)
from .python_processes import (
    PythonFieldView,
    PythonProcessContext,
    PythonProcessRegistry,
    PythonProcessSpec,
)
from .processes import ProcessRouter
from .user_api import (
    BlockingModel,
    FieldCollection,
    FieldReference,
    InstalledPythonProcess,
    ModelGroup,
    ModelStatus,
    PhaseCollection,
    PhaseReference,
    PlanBuilder,
    PhysicsCollection,
    SavedCheckpoint,
    SchemeReference,
)

__all__ = [
    "Action",
    "ActivatePhysics",
    "BranchSpec",
    "BlockingModel",
    "CheckpointBundle",
    "CCPPDeviceHost",
    "CCPPFieldRequirement",
    "CCPPStateSchema",
    "CCPPSuitePlan",
    "CAM_SE_FVM_V1",
    "DEFAULT_PHYSICS_GROUPS",
    "DeviceRegistry",
    "DeviceSupportMatrix",
    "DeviceBuildError",
    "DeviceCatalog",
    "DeviceContractError",
    "FieldContract",
    "FieldCollection",
    "FieldReference",
    "ModelGroup",
    "ModelStatus",
    "PhaseCollection",
    "PhaseReference",
    "PlanBuilder",
    "FieldVariant",
    "FortranDevice",
    "CAMDriver",
    "DriverState",
    "DeactivatePhysics",
    "DefineVariable",
    "FieldEdit",
    "InstallPhysics",
    "InstallPythonProcess",
    "InstalledPythonProcess",
    "PHYSICS_AFTER_COUPLER",
    "PHYSICS_BEFORE_COUPLER",
    "RuntimeCapabilities",
    "SchemeMove",
    "SchemeCatalogEntry",
    "SchemeSupport",
    "default_contracts",
    "export_contract",
    "ModelConfig",
    "ModelSnapshot",
    "ModelOptions",
    "ModelParameters",
    "MoveScheme",
    "ObserveFields",
    "PrepareInitialStep",
    "RunPhase",
    "RunScheme",
    "RunSchemeGroup",
    "RunSteps",
    "RemovePythonProcess",
    "SegmentPlan",
    "SetSchemeEnabled",
    "StatePool",
    "NativeObjectHandle",
    "HostServiceEvent",
    "HostServiceRegistry",
    "HistoryObservation",
    "homme_space_curve",
    "InstalledPhysicsPlugin",
    "PhysicsPluginManager",
    "PhysicsPluginSpec",
    "PhysicsCollection",
    "SavedCheckpoint",
    "ProcessRouter",
    "SchemePlacement",
    "SchemeReference",
    "sfc_partition_counts",
    "sfc_partition_owner",
    "VariableSpec",
    "PythonHistoryService",
    "PythonFieldView",
    "PythonProcessContext",
    "PythonProcessRegistry",
    "PythonProcessSpec",
    "SuiteNode",
    "SuiteScheme",
    "execute_segment_plan",
    "read_checkpoint",
    "restore_driver",
]
