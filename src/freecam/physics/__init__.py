"""Original CAM physics routines as ordinary single-column functions.

This package hosts a routine outside the model: no ``Driver``, no MPI, no
StatePool, no timestep.  The reviewed specification under
``native/pi_cam/functions`` defines the function boundary; later layers load a
minimal standalone image of the original Fortran and call it one column at a
time.
"""

from .errors import PhysicsError, PhysicsSpecError
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
    "FunctionSpec",
    "ImageSpec",
    "ModuleStateEntry",
    "ParameterSpec",
    "PhysicsError",
    "PhysicsSpecError",
    "load_function_spec",
]
