"""Processes whose driver is pausable Fortran: a class per action, kernels behind a runner.

A :class:`PausableStage` owns one workflow action the way
:class:`~freecam.physics.cloud_macro_microphysics.CloudMacroMicrophysics`
owns stage 7, without a Python walk of its own: with nothing replaced the
original Fortran action runs whole, once a step; with a kernel replaced the
image's pausable runner (``tools/generate_pi_cam_pausable_runners.py``, from
the spec under ``native/pi_cam/pausable/``) runs the same Fortran and pauses
at that kernel's call, where the model in ``kernels[name]`` answers the
frame.  The statement-by-statement Python walk does not exist for these
stages and is refused rather than fallen back to.

Every process the ledger lists gets one class here; the ones whose body does
no numerical work in this configuration are :class:`InertStage` s with no
kernel at all, and a gate that disables them proves it.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..pi_cam.errors import PICAMConfigurationError
from .errors import PhysicsError
from .stage import NativeStage


def bind_stage_hosts(native: Any) -> None:
    """Bind the stage hosts (state, tendencies, pbuf, cam_in, cam_out) every pausable runner's glue reads."""

    binder = getattr(native.library, "pycam_stagehost_bind_v1", None)
    if binder is None:
        raise PICAMConfigurationError(
            "this image has no pycam_stagehost_bind_v1; it was built before the pausable runners")
    status = int(binder())
    if status:
        raise PICAMConfigurationError(f"pycam_stagehost_bind_v1 refused ({status}): the state registry is not ready")


class PausableStage(NativeStage):
    """A whole workflow action with its kernels behind a pausable runner."""

    WHOLE_ACTION = True
    #: the runner's entry prefix without the leading ``pycam_``; see the manifest
    RUNNER_PREFIX = ""

    def __init__(self, *, kernels: Mapping[str, Any] | None = None) -> None:
        super().__init__(kernels=kernels)

    def select_mode(self, native: Any = None) -> str:
        policy = self.execution_policy
        if policy == "legacy-python":
            raise PhysicsError(
                f"{type(self).__name__} has no statement-by-statement Python walk; "
                f"its kernels run through the image's pausable runner (auto or segmented)")
        mode = super().select_mode(native)
        if mode == "legacy-python":
            raise PhysicsError(
                f"{type(self).__name__}: {list(self.replacements())} are replaced but the image's "
                f"runner for {self.STAGE!r} does not pause at them; there is no walk to fall back to")
        return mode

    def prepare_segmented(self, native: Any) -> None:
        """Bind the stage hosts the runner reads: one binder for every pausable runner."""

        bind_stage_hosts(native)

    def tend_chunk(self, runtime, lchnk, ncol, index, dt, nstep) -> None:   # pragma: no cover - refused above
        raise PhysicsError(f"{type(self).__name__} has no Python walk")


class InertStage(PausableStage):
    """An action that does no numerical work in this configuration: owned, run whole, no kernel."""

    SWAPPABLE: tuple[str, ...] = ()
    #: why the body is inert here, as the ledger records it
    INERT_BECAUSE = ""


class DryAdjustment(PausableStage):
    """tphysbc stage 3: the dry adiabatic adjustment, ``dadadj`` as its kernel."""

    STAGE = "cam_run1.dry_adjustment"
    PREFIX = "dadadj"
    RUNNER_PREFIX = "dadadj"
    PROCESS_NAME = "dry_adjustment"
    SWAPPABLE = ("dadadj",)


class ShallowConvection(PausableStage):
    """tphysbc stage 5: the UW shallow convection driver, ``compute_uwshcu_inv`` as its kernel."""

    STAGE = "cam_run1.shallow_convection"
    PREFIX = "shcu"
    RUNNER_PREFIX = "shcu"
    PROCESS_NAME = "shallow_convection"
    SWAPPABLE = ("compute_uwshcu_inv",)


def _inert(name: str, stage: str, process_name: str, because: str) -> type:
    return type(name, (InertStage,), {
        "STAGE": stage, "PREFIX": process_name, "RUNNER_PREFIX": "", "PROCESS_NAME": process_name,
        "INERT_BECAUSE": because, "__doc__": f"{stage}: {because}",
    })


RayleighFriction = _inert("RayleighFriction", "cam_run2.rayleigh_friction", "rayleigh_friction",
                          "rayk0 is not set; the routine's tendency is zero")
ChargeNeutrality = _inert("ChargeNeutrality", "cam_run2.charge_neutrality", "charge_neutrality",
                          "no ionosphere: charge_fix reduces to mbar = mwdry")
QBORelaxation = _inert("QBORelaxation", "cam_run2.qbo_relaxation", "qbo_relaxation", "qbo_use_forcing is off")
IonDrag = _inert("IonDrag", "cam_run2.ion_drag", "ion_drag", "no WACCM: iondrag_calc returns without a tendency")
SeaSaltRebin = _inert("SeaSaltRebin", "cam_run1.sea_salt_rebin", "sea_salt_rebin",
                      "sslt_rebin_adv acts on bulk sea salt, which the modal aerosols do not carry")
ModalAerosolPreparation = _inert("ModalAerosolPreparation", "cam_run1.modal_aerosol_preparation_leaf",
                                 "modal_aerosol_preparation_leaf", "no work outside the sub-column path")
CARMAWetDeposition = _inert("CARMAWetDeposition", "cam_run1.carma_wet_deposition_leaf", "carma_wet_deposition_leaf",
                            "carma_model is none")
CARMAAerosolTendencies = _inert("CARMAAerosolTendencies", "cam_run2.carma_aerosol_tendencies_leaf",
                                "carma_aerosol_tendencies_leaf", "carma_model is none")
CARMAStatistics = _inert("CARMAStatistics", "cam_run2.carma_statistics_leaf", "carma_statistics_leaf",
                         "carma_model is none")
TracerTendencies = _inert("TracerTendencies", "cam_run2.tracer_tendencies_leaf", "tracer_tendencies_leaf",
                          "the test tracers are not enabled in this compset")
AgeOfAirTendencies = _inert("AgeOfAirTendencies", "cam_run2.age_of_air_tendencies_leaf", "age_of_air_tendencies_leaf",
                            "aoa_tracers_flag is off")

#: Every pausable and inert class by the action's name, for the command line and the ledger.
STAGES: dict[str, type[PausableStage]] = {
    cls.PROCESS_NAME: cls for cls in (
        DryAdjustment, ShallowConvection,
        RayleighFriction, ChargeNeutrality, QBORelaxation, IonDrag, SeaSaltRebin, ModalAerosolPreparation,
        CARMAWetDeposition, CARMAAerosolTendencies, CARMAStatistics, TracerTendencies, AgeOfAirTendencies,
    )
}

__all__ = ["AgeOfAirTendencies", "CARMAAerosolTendencies", "CARMAStatistics", "CARMAWetDeposition",
           "ChargeNeutrality", "DryAdjustment", "InertStage", "IonDrag", "ModalAerosolPreparation",
           "PausableStage", "QBORelaxation", "RayleighFriction", "STAGES", "SeaSaltRebin", "ShallowConvection",
           "TracerTendencies"]
