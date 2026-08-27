"""Original CAM physics routines as ordinary single-column functions.

This package hosts a routine outside the model: no ``Driver``, no MPI, no
StatePool, no timestep.  The reviewed specification under
``native/pi_cam/functions`` defines the function boundary; later layers load a
minimal standalone image of the original Fortran and call it one column at a
time.
"""

from .errors import CallError, FortranAbortError, InternalError, PhysicsError, PhysicsSpecError, WorkerCrashError
from .examples import ExampleColumn, available_examples, load_example_column
from .dataset import Dataset, SampleVerification, open_dataset
from .distributions import (
    Anchored,
    Choice,
    Constant,
    Derived,
    HybridCoordinate,
    HybridPressure,
    LogUniform,
    Normal,
    SamplingSpace,
    Uniform,
)
from .column import InvalidInput
from .function import PhysicsFunction, load_function
from .macrophysics import Macrophysics
from .radiation import Radiation
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
    "Anchored",
    "ArgumentSpec",
    "Choice",
    "CallError",
    "ExampleColumn",
    "FortranAbortError",
    "HybridCoordinate",
    "InternalError",
    "InvalidInput",
    "SampleVerification",
    "WorkerCrashError",
    "available_examples",
    "load_example_column",
    "open_dataset",
    "Constant",
    "Dataset",
    "Derived",
    "HybridPressure",
    "LogUniform",
    "Normal",
    "SamplingSpace",
    "Uniform",
    "FunctionResult",
    "FunctionSpec",
    "ImageSpec",
    "ModuleStateEntry",
    "ParameterSpec",
    "PhysicsError",
    "Macrophysics",
    "Radiation",
    "PhysicsFunction",
    "PhysicsSpecError",
    "load_function",
    "load_function_spec",
]
