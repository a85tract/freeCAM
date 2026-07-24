"""Fixed-case CAM model controlled and owned by Python.

The public Notebook controller lives in :mod:`pycam_sima.notebook.session`.
This package contains the rank-local implementation used by its 24 MPI
workers.
"""

from .checkpoint import (
    CheckpointBundle,
    ModelSnapshot,
    read_checkpoint,
    restore_driver,
)
from .ccpp_suite import CCPPDeviceHost, CCPPSuitePlan, SuiteNode, SuiteScheme
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
from .scheme_plan import (
    KesslerSchemePlan,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
    PhysicsScheme,
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

__all__ = [
    "Action",
    "ActivatePhysics",
    "BranchSpec",
    "CheckpointBundle",
    "CCPPDeviceHost",
    "CCPPFieldRequirement",
    "CCPPStateSchema",
    "CCPPSuitePlan",
    "DeviceRegistry",
    "DeviceSupportMatrix",
    "DeviceBuildError",
    "DeviceCatalog",
    "DeviceContractError",
    "FieldContract",
    "FieldVariant",
    "FortranDevice",
    "CAMDriver",
    "DriverState",
    "DeactivatePhysics",
    "DefineVariable",
    "FieldEdit",
    "InstallPhysics",
    "KesslerSchemePlan",
    "PHYSICS_AFTER_COUPLER",
    "PHYSICS_BEFORE_COUPLER",
    "PhysicsScheme",
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
    "SchemePlacement",
    "VariableSpec",
    "PythonHistoryService",
    "SuiteNode",
    "SuiteScheme",
    "execute_segment_plan",
    "read_checkpoint",
    "restore_driver",
]
