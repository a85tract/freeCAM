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
from .config import ModelConfig
from .control import ModelOptions, ModelParameters
from .contracts import FieldContract, default_contracts, export_contract
from .driver import CAMDriver, DriverState
from .experiment import (
    Action,
    BranchSpec,
    FieldEdit,
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
from .state import StatePool

__all__ = [
    "Action",
    "BranchSpec",
    "CheckpointBundle",
    "FieldContract",
    "CAMDriver",
    "DriverState",
    "FieldEdit",
    "KesslerSchemePlan",
    "PHYSICS_AFTER_COUPLER",
    "PHYSICS_BEFORE_COUPLER",
    "PhysicsScheme",
    "SchemeMove",
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
    "execute_segment_plan",
    "read_checkpoint",
    "restore_driver",
]
