from __future__ import annotations

from .state_pool import StatePool


class IdentityDynamics:
    """Explicit no-op dynamics boundary for the Kessler kernel milestone."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def initialize(self, pool: StatePool) -> None:
        self.calls.append("initialize")

    def dynamics_to_physics(self, pool: StatePool) -> None:
        self.calls.append("dynamics_to_physics")

    def physics_to_dynamics(self, pool: StatePool) -> None:
        self.calls.append("physics_to_dynamics")

    def run(self, pool: StatePool) -> None:
        self.calls.append("run")

    def finalize(self, pool: StatePool) -> None:
        self.calls.append("finalize")


# Kept for control-order tests written before the boundary was named.
RecordingDynamics = IdentityDynamics
