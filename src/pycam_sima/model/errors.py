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
    """A required stateless kernel is not present in the selected backend."""


class NativeInitializationError(PyCamSimaError):
    """A native call occurred while running the pure-Python initializer."""


class RemoteRankAccessError(PyCamSimaError):
    """A worker was asked for a field owned by another rank."""


class ValidationError(PyCamSimaError):
    """A bitwise or structural validation gate failed."""
