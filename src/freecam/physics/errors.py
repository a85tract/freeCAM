"""Exceptions raised by the standalone physics-function layer."""


class PhysicsError(Exception):
    """Base class for physics-function errors."""


class PhysicsSpecError(PhysicsError):
    """A function specification is missing, inconsistent, or unverifiable."""


class CallError(PhysicsError):
    """A call did not return a result; ``status`` names why."""

    status = "internal_error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.status)
        self.message = message


class FortranAbortError(CallError):
    """The routine refused the inputs and aborted (CAM's endrun)."""

    status = "fortran_abort"


class WorkerCrashError(CallError):
    """The worker process died without a Fortran abort message."""

    status = "worker_crash"


class InternalError(CallError):
    """The harness itself failed; not a property of the sample."""

    status = "internal_error"


ERRORS_BY_STATUS = {
    FortranAbortError.status: FortranAbortError,
    WorkerCrashError.status: WorkerCrashError,
    InternalError.status: InternalError,
}

__all__ = [
    "ERRORS_BY_STATUS",
    "CallError",
    "FortranAbortError",
    "InternalError",
    "PhysicsError",
    "PhysicsSpecError",
    "WorkerCrashError",
]
