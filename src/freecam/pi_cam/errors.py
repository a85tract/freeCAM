"""PI-CAM-only runtime errors."""


class PICAMError(RuntimeError):
    """Base error for the PI-CAM-only runtime."""


class PICAMConfigurationError(PICAMError):
    """The admitted PI-CAM case or execution plan is invalid."""


class PICAMStateError(PICAMError):
    """A state transition or rank-local field operation is invalid."""


class BoundaryReplayError(PICAMError):
    """Captured CAM boundary data cannot be replayed exactly."""


class NativeCAMError(PICAMError):
    """A native CAM numerical device failed its ABI contract."""
