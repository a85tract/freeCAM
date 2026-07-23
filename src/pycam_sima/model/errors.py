"""Typed errors for the fail-closed Python model."""


class PyCamSimaError(RuntimeError):
    """Base runtime error."""


class ConfigurationError(PyCamSimaError):
    """The requested case is outside the fixed first-release contract."""


class StateTransitionError(PyCamSimaError):
    """A public operation was requested from an invalid lifecycle state."""


class StateOwnershipError(PyCamSimaError):
    """An array violates the Python ownership or pointer stability contract."""


class MissingKernelError(PyCamSimaError):
    """A required main kernel or generated device is not present."""


class DeviceContractError(PyCamSimaError):
    """A device manifest cannot be connected safely to the Python StatePool."""


class DeviceBuildError(PyCamSimaError):
    """A source scheme cannot be converted into a standalone device."""


class NativeInitializationError(PyCamSimaError):
    """A native call occurred while running the pure-Python initializer."""


class RemoteRankAccessError(PyCamSimaError):
    """A worker was asked for a field owned by another rank."""


class ValidationError(PyCamSimaError):
    """A bitwise or structural validation gate failed."""
