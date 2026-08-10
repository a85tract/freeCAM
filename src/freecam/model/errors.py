"""Typed errors for the fail-closed Python model."""


class FreeCAMRuntimeError(RuntimeError):
    """Base runtime error."""


class ConfigurationError(FreeCAMRuntimeError):
    """The requested case is outside the fixed first-release contract."""


class StateTransitionError(FreeCAMRuntimeError):
    """A public operation was requested from an invalid lifecycle state."""


class StateOwnershipError(FreeCAMRuntimeError):
    """An array violates the Python ownership or pointer stability contract."""


class MissingKernelError(FreeCAMRuntimeError):
    """A required main kernel or generated device is not present."""


class DeviceContractError(FreeCAMRuntimeError):
    """A device manifest cannot be connected safely to the Python StatePool."""


class DeviceBuildError(FreeCAMRuntimeError):
    """A source scheme cannot be converted into a standalone device."""


class PythonProcessContractError(FreeCAMRuntimeError):
    """A Notebook-defined Python process violates its runtime contract."""


class PythonProcessExecutionError(FreeCAMRuntimeError):
    """A transactional Python process failed and its writes were restored."""


class PythonProcessTaintedError(FreeCAMRuntimeError):
    """A non-transactional Python process left model state unsafe to reuse."""


class NativeInitializationError(FreeCAMRuntimeError):
    """A native call occurred while running the pure-Python initializer."""


class RemoteRankAccessError(FreeCAMRuntimeError):
    """A worker was asked for a field owned by another rank."""


class ValidationError(FreeCAMRuntimeError):
    """A bitwise or structural validation gate failed."""
