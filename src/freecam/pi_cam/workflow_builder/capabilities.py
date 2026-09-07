"""What the builder may offer as a kernel replacement, and what it has proven.

A stage class declares which of its kernels are swappable; the image's
segment runner says which of those it can pause at; the validation records
say which have run bit-for-bit through that pause with the original kernel
answering.  The builder offers a binding only where the runner covers the
kernel, and labels separately whether the path is validated.  A name in a
catalog is not a claim that it can be replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


#: Kernels whose pause path has passed the 512-rank 50-step gate with the
#: original kernel answering through Python, and the record that says so.
VALIDATED_THROUGH_RUNNER: Mapping[str, tuple[str, ...]] = {
    "mmacro_pcond": (
        "validation/pi_cam_stage7_segmented_original_50step.json",
        "validation/pi_cam_stage7_segmented_original_vs_oracle_50step_bfb.json",
    ),
}


@dataclass(frozen=True, slots=True)
class KernelCapability:
    """One swappable kernel: where it lives and what the builder may do with it."""

    kernel: str
    stage_action: str
    stage_class: str
    owner_class: str
    bindable: bool
    validated: bool
    reason: str | None = None
    evidence: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "stage_action": self.stage_action,
            "stage_class": self.stage_class,
            "owner_class": self.owner_class,
            "bindable": self.bindable,
            "validated": self.validated,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


def kernel_capabilities() -> tuple[KernelCapability, ...]:
    """The swappable kernels of every stage class, with what the image offers for them.

    The composed cloud stage owns the macro, micro and aerosol sub-walks'
    kernels; radiation owns its own two.  ``bindable`` follows the segment
    runner's kernel list, so a kernel gains a binding when the image gains a
    runner for it and not before.
    """

    from freecam.physics.cloud_macro_microphysics import CloudMacroMicrophysics
    from freecam.physics.macrophysics import Macrophysics
    from freecam.physics.microp_aero import MicropAero
    from freecam.physics.microphysics import Microphysics
    from freecam.physics.radiation import Radiation
    from freecam.pi_cam.segment_runner import KERNELS as RUNNER_KERNELS

    owners = (
        (CloudMacroMicrophysics, Macrophysics),
        (CloudMacroMicrophysics, Microphysics),
        (CloudMacroMicrophysics, MicropAero),
        (Radiation, Radiation),
    )
    capabilities: list[KernelCapability] = []
    for stage_class, owner in owners:
        for kernel in getattr(owner, "SWAPPABLE", ()):
            bindable = kernel in RUNNER_KERNELS
            validated = kernel in VALIDATED_THROUGH_RUNNER
            reason = None
            if not bindable:
                reason = (
                    "the image has no segment runner for this kernel yet; a replacement "
                    "would run the statement-by-statement Python walk, which the builder "
                    "does not offer"
                )
            elif not validated:
                reason = "bindable, but the pause path has not passed a bit-for-bit gate"
            capabilities.append(
                KernelCapability(
                    kernel=str(kernel),
                    stage_action=str(stage_class.STAGE),
                    stage_class=f"{stage_class.__module__}.{stage_class.__name__}",
                    owner_class=f"{owner.__module__}.{owner.__name__}",
                    bindable=bindable,
                    validated=validated,
                    reason=reason,
                    evidence=tuple(VALIDATED_THROUGH_RUNNER.get(kernel, ())),
                )
            )
    return tuple(capabilities)


def capabilities_by_action() -> dict[str, tuple[KernelCapability, ...]]:
    grouped: dict[str, list[KernelCapability]] = {}
    for capability in kernel_capabilities():
        grouped.setdefault(capability.stage_action, []).append(capability)
    return {action: tuple(items) for action, items in grouped.items()}


__all__ = ["KernelCapability", "VALIDATED_THROUGH_RUNNER", "capabilities_by_action", "kernel_capabilities"]
