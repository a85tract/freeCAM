"""Original CAM physics routines as ordinary single-column functions.

This package hosts a routine outside the model: no ``Driver``, no MPI, no
StatePool, no timestep.  The reviewed specification under
``native/pi_cam/functions`` defines the function boundary; later layers load a
minimal standalone image of the original Fortran and call it one column at a
time.
"""

from .errors import PhysicsError, PhysicsSpecError
from .function import PhysicsFunction, load_function
from .result import FunctionResult
from .spec import (
    ArgumentSpec,
    FunctionSpec,
    ImageSpec,
    ModuleStateEntry,
    ParameterSpec,
    load_function_spec,
)

__all__ = [
    "ArgumentSpec",
    "FunctionResult",
    "FunctionSpec",
    "ImageSpec",
    "ModuleStateEntry",
    "ParameterSpec",
    "PhysicsError",
    "PhysicsFunction",
    "PhysicsSpecError",
    "load_function",
    "load_function_spec",
]
