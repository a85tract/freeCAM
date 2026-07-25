"""Python-owned CAM model and generic CCPP device runtime.

The public Notebook controller lives in :mod:`pycam_sima.notebook.session`.
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
    MoveScheme,
    ObserveFields,
    PrepareInitialStep,
    RunPhase,
    RunScheme,
    RunSchemeGroup,
    RunSteps,
    SchemeMove,
    SegmentPlan,
    SetSchemeEnabled,
    execute_segment_plan,
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
from .processes import ProcessRouter
from .user_api import (
    BlockingModel,
    FieldCollection,
    FieldReference,
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
    "SegmentPlan",
    "SetSchemeEnabled",
    "StatePool",
    "NativeObjectHandle",
    "HostServiceEvent",
    "HostServiceRegistry",
    "HistoryObservation",
    "InstalledPhysicsPlugin",
    "PhysicsPluginManager",
    "PhysicsPluginSpec",
    "PhysicsCollection",
    "SavedCheckpoint",
    "ProcessRouter",
    "SchemePlacement",
    "SchemeReference",
    "VariableSpec",
    "PythonHistoryService",
    "SuiteNode",
    "SuiteScheme",
    "execute_segment_plan",
    "read_checkpoint",
    "restore_driver",
]
