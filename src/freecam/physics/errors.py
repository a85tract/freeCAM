"""Exceptions raised by the standalone physics-function layer."""


class PhysicsError(Exception):
    """Base class for physics-function errors."""


class PhysicsSpecError(PhysicsError):
    """A function specification is missing, inconsistent, or unverifiable."""


__all__ = ["PhysicsError", "PhysicsSpecError"]
