"""Fixed-case CAM model controlled and owned by Python.

The public Notebook controller lives in :mod:`pycam_sima.notebook.session`.
This package contains the rank-local implementation used by its 24 MPI
workers.
"""

from .config import ModelConfig
from .control import ModelOptions, ModelParameters
from .contracts import FieldContract, default_contracts, export_contract
from .driver import CAMDriver, DriverState
from .scheme_plan import (
    KesslerSchemePlan,
    PHYSICS_AFTER_COUPLER,
    PHYSICS_BEFORE_COUPLER,
    PhysicsScheme,
)
from .state import StatePool

__all__ = [
    "FieldContract",
    "CAMDriver",
    "DriverState",
    "KesslerSchemePlan",
    "PHYSICS_AFTER_COUPLER",
    "PHYSICS_BEFORE_COUPLER",
    "PhysicsScheme",
    "default_contracts",
    "export_contract",
    "ModelConfig",
    "ModelOptions",
    "ModelParameters",
    "StatePool",
]
