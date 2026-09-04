"""tphysbc's cloud macro/microphysics stage as a Python class.

Workflow action 427, ``cloud_macro_microphysics``, is stage 7 of
``tphysbc`` (``physpkg.F90:2188-2393`` in the pinned source): the
macrophysics/microphysics substepping loop and everything around it --
aerosol activation, the two drivers, tendency scaling and application
against ``phys_tend``, the energy checks, the precipitation bookkeeping,
the water-tracer mass fixer.  It is the most expensive physics stage.

:class:`CloudMacroMicrophysics` replaces that action whole: Python owns the
stage's control flow statement for statement, and every floating-point
number is still Fortran's -- the twelve arithmetic statements of the glue
through the four lifted ``mm_*`` kernels, everything that takes a derived
type through ``pycam_mm_handles``.  The stage's own tendency objects
(``ptend``, ``ptend_aero``) and the ``physics_tend`` it accumulates into
live in Fortran and are reached by handle; the six precipitation fields
are physics-buffer storage reached by index.

``macrop_driver_tend`` is the :class:`Macrophysics` sub-walk and
``microp_driver_tend`` the :class:`Microphysics` sub-walk, both composed
into this stage: each walk's tendency object (and the macrophysics
detrainment) is taken over exactly as the split stage's post-leaf took
them.  ``whole_micro=True`` calls the microphysics driver whole (Gate
M-2's form) and ``whole_drivers=True`` both drivers (Gate M-1's); each
composed form is diagnosed against the one before it.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..pi_cam.errors import PICAMConfigurationError
from ..pi_cam.pbuf import PBuf, load_pbuf_table
from .image import module_view
from .macrophysics import FORCING, Macrophysics
from .microp_aero import MicropAero
from .microphysics import Microphysics
from freecam.pi_cam.tables import load_table
from .stage import (
    CORE_ENTRIES,
    HostEntries,
    HostServices,
    NativeStage,
    StageProfile,
    StageRuntime,
    check as _check,
    pointer_of as _ptr,
)

REPO = Path(__file__).resolve().parents[3]

#: The six precipitation fields tphysbc reads for this stage, generated
#: from the pinned source by tools/generate_pi_cam_pbuf_table.py.
PBUF_TABLE = REPO / "native/pi_cam/pbuf_fields_mm.yaml"
PBUF_FIELDS = ("PREC_STR", "SNOW_STR", "PREC_SED", "SNOW_SED", "PREC_PCW", "SNOW_PCW")

#: pycam_mm_handles.F90 view codes.  A test keeps this table equal to the
#: Fortran one.
VIEW = {
    "ptend_s": 21, "ptend_q": 22, "ptend_aero_s": 23, "ptend_aero_q": 24,
    "det_s": 31, "det_ice": 32,
}
#: Which of the stage's two tendency objects a handle acts on.
PTEND, PTEND_AERO = 1, 2

#: The direct kernels tend() runs: the glue's twelve arithmetic statements
#: and the substep length, lifted verbatim (tools/generate_pi_cam_mm_kernels.py).
KERNELS = ("mm_substep_dt", "mm_flux_terms", "mm_precip_accumulate", "mm_precip_average")

#: The order in which tphysbc does things inside stage 7 under the admitted
#: configuration, with both drivers called whole; ``tend`` follows it per
#: chunk and a test compares the two.
SEQUENCE_WHOLE = (
    "pbuf_get_field*", "mm_substep_dt", "macmic_zero",
    "macrop_driver_tend", "mm_flux_terms",
    "physics_ptend_scale", "physics_update", "check_energy_chng:macrop_tend",
    "microp_aero_run", "microp_driver_tend", "physics_ptend_sum:ptend_aero",
    "physics_ptend_scale", "physics_update", "check_energy_chng:microp_tend",
    "mm_precip_accumulate",
    "mm_precip_average", "wtrc_mass_fixer",
)


def _composed(sequence: tuple[str, ...], **walks: tuple[str, str]) -> tuple[str, ...]:
    """``sequence`` with each named driver call replaced by its sub-walk pair."""

    return tuple(name for step in sequence for name in walks.get(step, (step,)))


#: The same with the macrophysics sub-walk in its driver's place (Gate M-2).
SEQUENCE_WHOLE_MICRO = _composed(
    SEQUENCE_WHOLE, macrop_driver_tend=("Macrophysics.tend_chunk", "take_macro"))
#: Both cloud sub-walks in their drivers' places (Gate M-3).
SEQUENCE_WHOLE_AERO = _composed(
    SEQUENCE_WHOLE_MICRO, microp_driver_tend=("Microphysics.tend_chunk", "take_micro"))
#: And the aerosol activation as its own walk (Gate M-4).
SEQUENCE = _composed(
    SEQUENCE_WHOLE_AERO, microp_aero_run=("MicropAero.tend_chunk", "take_aero"))

#: The arguments tphysbc hands macrop_driver_tend beside state, ptend, the
#: substep length, pbuf and the two detrainment outputs -- in its order.
MACROP_ARGUMENTS = ("landfrac", "ocnfrac", "snowhland", "dlf", "dlf2", "wtdlf",
                    "cmfmc", "cmfmc2", "ts", "sst", "zdu")


# -- the image, seen through ctypes ---------------------------------------------

_INT, _DBL, _STR = ctypes.c_int, ctypes.c_double, ctypes.c_char_p
_P_DBL = ctypes.POINTER(ctypes.c_double)


class _MMEntries(HostEntries):
    """The core entries, plus the stage's own calls into CAM.

    The stage copies no state and builds no ptend of its own -- the drivers
    do -- so it declares none of the ptend entries; what it does declare is
    the set of calls tphysbc makes around the drivers, each in the driver's
    own form.  ``forcing`` is tphysbc's, shared with the macrophysics stage:
    the convection outputs the stage reads live in physpkg's buffers.
    """

    TABLE = {
        **CORE_ENTRIES,
        "forcing": ("pycam_macro_forcing_v1",
                    [_INT, _INT, ctypes.POINTER(ctypes.c_void_p),
                     ctypes.POINTER(_INT), ctypes.POINTER(ctypes.c_int64)], False),
        "microp_aero_run": ("pycam_{prefix}_microp_aero_run_v1", [_INT, _DBL], False),
        "macrop_driver_tend": ("pycam_{prefix}_macrop_driver_tend_v1",
                               [_INT, _DBL] + [_P_DBL] * len(MACROP_ARGUMENTS), False),
        "microp_driver_tend": ("pycam_{prefix}_microp_driver_tend_v1", [_INT, _DBL], False),
        "ptend_scale": ("pycam_{prefix}_ptend_scale_v1", [_INT, _INT, _INT, _INT], False),
        "update_tend": ("pycam_{prefix}_update_v1", [_INT, _INT, _DBL], False),
        "check_energy": ("pycam_{prefix}_check_energy_v1",
                         [_INT, _STR, _INT, _INT, _DBL, _INT,
                          _P_DBL, _P_DBL, _P_DBL, _P_DBL, _INT], False),
        "ptend_sum_aero": ("pycam_{prefix}_ptend_sum_aero_v1", [_INT, _INT], False),
        "wtrc_mass_fixer": ("pycam_{prefix}_wtrc_mass_fixer_v1", [_INT], False),
        # optional: an image built for Gate M-1 predates it, and that image
        # still serves the whole-drivers form; the composed form refuses
        "take_macro": ("pycam_{prefix}_take_macro_v1", [_INT], True),
        "take_micro": ("pycam_{prefix}_take_micro_v1", [_INT], True),
        "take_aero": ("pycam_{prefix}_take_aero_v1", [_INT], True),
    }


class _MMHandles(HostServices):
    """CAM's host services, plus the calls only this stage makes."""

    def forcing(self, lchnk: int, name: str) -> np.ndarray:
        """A zero-copy view of one of tphysbc's convection forcing buffers."""

        return self._deref(self.e.forcing, f"pycam_macro_forcing_v1({name}, chunk {lchnk})",
                           lchnk, FORCING[name], ndims_max=4)

    def microp_aero_run(self, lchnk: int, dt: float) -> None:
        _check(self.e.microp_aero_run(lchnk, float(dt)), "microp_aero_run")

    def macrop_driver_tend(self, lchnk: int, dt: float, arrays: Sequence[np.ndarray]) -> None:
        """The whole driver, with tphysbc's eleven array arguments in its order."""

        assert len(arrays) == len(MACROP_ARGUMENTS)
        _check(self.e.macrop_driver_tend(lchnk, float(dt), *[_ptr(a) for a in arrays]),
               "macrop_driver_tend")

    def microp_driver_tend(self, lchnk: int, dt: float) -> None:
        _check(self.e.microp_driver_tend(lchnk, float(dt)), "microp_driver_tend")

    def ptend_scale(self, lchnk: int, which: int, num_steps: int, ncol: int) -> None:
        """``physics_ptend_scale(ptend, 1._r8/cld_macmic_num_steps, ncol)``; the
        factor is formed in Fortran from the count."""

        _check(self.e.ptend_scale(lchnk, which, num_steps, ncol), "physics_ptend_scale")

    def update_tend(self, lchnk: int, which: int, ztodt: float) -> None:
        """``physics_update(state, ptend, ztodt, tend)`` against ``phys_tend``."""

        _check(self.e.update_tend(lchnk, which, float(ztodt)), "physics_update")

    def check_energy(self, lchnk: int, name: str, nstep: int, ztodt: float, num_steps: int,
                     flx_vap: np.ndarray, flx_cnd: np.ndarray, flx_ice: np.ndarray,
                     flx_sen: np.ndarray, *, scaled: bool) -> None:
        """``check_energy_chng``; ``scaled`` is the form whose last three fluxes
        are divided by the substep count inside the call."""

        _check(self.e.check_energy(
            lchnk, name.encode("ascii"), len(name), nstep, float(ztodt), num_steps,
            _ptr(flx_vap), _ptr(flx_cnd), _ptr(flx_ice), _ptr(flx_sen), int(scaled),
        ), f"check_energy_chng({name!r})")

    def ptend_sum_aero(self, lchnk: int, ncol: int) -> None:
        """``physics_ptend_sum(ptend_aero, ptend, ncol)`` then ``dealloc(ptend_aero)``."""

        _check(self.e.ptend_sum_aero(lchnk, ncol), "physics_ptend_sum(ptend_aero)")

    def wtrc_mass_fixer(self, lchnk: int) -> None:
        _check(self.e.wtrc_mass_fixer(lchnk), "wtrc_mass_fixer")

    def take_macro(self, lchnk: int) -> None:
        """The macrophysics sub-walk's ptend and detrainment become the stage's."""

        _check(self.e.take_macro(lchnk), "take_macro (ptend = macro_ptend(lchnk))")

    def take_micro(self, lchnk: int) -> None:
        """The microphysics sub-walk's ptend becomes the stage's."""

        _check(self.e.take_micro(lchnk), "take_micro (ptend = micro_ptend(lchnk))")

    def take_aero(self, lchnk: int) -> None:
        """The aerosol sub-walk's ptend becomes the stage's ptend_aero."""

        _check(self.e.take_aero(lchnk), "take_aero (ptend_aero = aero_ptend(lchnk))")


# -- module constants --------------------------------------------------------------


@dataclass(frozen=True)
class _Constants:
    """The module state stage 7 branches on, read once from the image."""

    cld_macmic_num_steps: int
    macrop_scheme: str
    microp_scheme: str
    micro_do_icesupersat: bool
    use_subcol_microp: bool
    carma_do_cldice: bool
    carma_do_cldliq: bool
    trace_water: bool

    @classmethod
    def read(cls, library: Any) -> "_Constants":
        def i(symbol):
            return int(module_view(library, symbol, "int32", ()))

        def b(symbol):
            return bool(int(module_view(library, symbol, "int32", ())))

        def s(symbol):
            return module_view(library, symbol, "S16", ()).item().decode("ascii").strip()

        return cls(
            cld_macmic_num_steps=i("physpkg_mp_cld_macmic_num_steps_"),
            macrop_scheme=s("phys_control_mp_macrop_scheme_"),
            microp_scheme=s("phys_control_mp_microp_scheme_"),
            micro_do_icesupersat=b("macrop_driver_mp_micro_do_icesupersat_"),
            use_subcol_microp=b("phys_control_mp_use_subcol_microp_"),
            carma_do_cldice=b("carma_flags_mod_mp_carma_do_cldice_"),
            carma_do_cldliq=b("carma_flags_mod_mp_carma_do_cldliq_"),
            trace_water=b("water_tracer_vars_mp_trace_water_"),
        )

    def refuse_unsupported(self) -> None:
        """The paths the admitted configuration never takes are not ported."""

        def refuse(what: str) -> None:
            raise PICAMConfigurationError(
                f"{what}; the Python cloud macro/microphysics stage does not carry that path")

        if self.microp_scheme != "MG":
            refuse(f"microp_scheme is {self.microp_scheme!r}, not 'MG' (stratiform_tend)")
        if self.macrop_scheme == "CLUBB_SGS":
            refuse("macrop_scheme is 'CLUBB_SGS' (clubb_tend_cam)")
        if self.micro_do_icesupersat:
            refuse("micro_do_icesupersat is on (activation before macrophysics)")
        if self.use_subcol_microp:
            refuse("use_subcol_microp is on (subcolumn microphysics)")
        if self.carma_do_cldice or self.carma_do_cldliq:
            refuse("CARMA owns cloud ice or liquid")
        if self.cld_macmic_num_steps < 1:
            refuse(f"cld_macmic_num_steps is {self.cld_macmic_num_steps}")


# -- the class ---------------------------------------------------------------------


class CloudMacroMicrophysics(NativeStage):
    """tphysbc stage 7, the cloud macro/microphysics action, as Python.

    A whole workflow action: :meth:`attach` disables action 427 and puts
    :meth:`tend` in its place.  Per chunk, :meth:`tend_chunk` is the stage's
    Fortran statement for statement, with the two drivers as the composed
    :class:`Macrophysics` and :class:`Microphysics` walks.
    ``kernels["mmacro_pcond"]`` and ``kernels["micro_mg_tend"]`` reach the
    sub-walks' cores.
    """

    STAGE = "cam_run1.cloud_macro_microphysics"
    PREFIX = "mm"
    PROCESS_NAME = "cloud_macro_microphysics"
    TRACE_ENV = "FREECAM_MM_TRACE"
    PROFILE_ENV = "FREECAM_MM_PROFILE"

    KERNELS = KERNELS
    CAM_IN = ("landfrac", "ocnfrac", "snowhland", "ts", "sst")
    #: tphysbc's ``zero(pcols)``: an all-zero flux argument to the energy checks.
    EXTRA_SCRATCH = (("zero", ("pcols", "chunks")),)

    entries_class = _MMEntries
    services_class = _MMHandles

    def __init__(self, *, whole_drivers: bool = False, whole_micro: bool = False,
                 whole_aero: bool = False, micro_core_standalone: bool = False,
                 macro_surrogate: "str | Path | None" = None, kernels=None) -> None:
        super().__init__(kernels=None)
        if macro_surrogate is not None and whole_drivers:
            raise PICAMConfigurationError(
                "a surrogate stands in mmacro_pcond's place inside the macrophysics walk; "
                "with the driver called whole there is no such place")
        #: The sub-walks, or None to call the driver whole.
        self.macro: Macrophysics | None = None
        self.micro: Microphysics | None = None
        self.aero: MicropAero | None = None
        walks: dict[str, NativeStage] = {}
        if micro_core_standalone and (whole_drivers or whole_micro):
            raise PICAMConfigurationError(
                "the standalone core is the microphysics walk's; it has no meaning "
                "when the driver is called whole")
        if not whole_drivers:
            # the trained network, if any, named by path: each rank loads its
            # own copy the first time the kernel is called (see Macrophysics)
            walks["macro"] = Macrophysics(surrogate=macro_surrogate)
            if not whole_micro:
                walks["micro"] = Microphysics(standalone_core=micro_core_standalone)
                if not whole_aero:
                    walks["aero"] = MicropAero()
        self.compose(**walks)
        if kernels:
            unknown = [name for name in kernels if name not in self.kernels]
            if unknown:
                raise PICAMConfigurationError(
                    f"{type(self).__name__} has no swappable kernel named {unknown}; "
                    f"it has {list(self.kernels)}")
            self.kernels.update(kernels)

    # -- what the runtime asks of this stage -------------------------------------

    def read_constants(self, library: Any) -> "_Constants":
        return _Constants.read(library)

    def refuse_unsupported(self, constants: "_Constants") -> None:
        constants.refuse_unsupported()

    def build_pbuf(self, library: Any, runtime: StageRuntime) -> PBuf:

        symbols = [row["symbol"] for row in load_table(PBUF_TABLE)["fields"]]
        indices = {symbol: int(module_view(library, symbol, "int32", ())) for symbol in symbols}
        buffer = PBuf(library, load_pbuf_table(PBUF_TABLE, indices))
        lchnk, _ = runtime.native.chunks
        buffer.verify(int(lchnk[0]), pcols=runtime.pcols, pver=runtime.pver)
        return buffer

    # -- the transliteration -----------------------------------------------------

    def tend_chunk(self, st: StageRuntime, lchnk: int, ncol: int, index: int,
                   dt: float, nstep: int) -> None:
        """``physpkg.F90:2188-2393`` under the admitted configuration, one chunk.

        Line numbers are the pinned source's.  ``dt`` is tphysbc's ``ztodt``
        and ``nstep`` its ``nstep``, both read from the model's clock.
        """

        H, C, pb = st.handles, st.constants, st.pbuf
        L = st.local
        log = self.calls.append

        def K(name, inputs, *, outputs):
            st.kernel_on_chunk(name, inputs, outputs=outputs, ncol=ncol)

        n = C.cld_macmic_num_steps
        zero = L["zero"]                       # 2085: zero = 0; never written

        # 2104-2108: the precipitation fields, physics-buffer storage
        pbv = {name: pb.view(name, lchnk) for name in PBUF_FIELDS}
        log("pbuf_get_field*")

        # 2188-2208: microp_scheme == 'RK' is refused at attach
        # 2210: cld_macmic_ztodt = ztodt/cld_macmic_num_steps
        K("mm_substep_dt", {"ztodt": dt, "cld_macmic_num_steps": n},
          outputs={"cld_macmic_ztodt": None})
        log("mm_substep_dt")
        sub_dt = float(L["cld_macmic_ztodt"][()])
        # 2213-2216: the substep accumulators start at zero
        for name in ("prec_sed_macmic", "snow_sed_macmic", "prec_pcw_macmic", "snow_pcw_macmic"):
            st.scratch[name][...] = 0.0
        log("macmic_zero")

        forcing = {name: H.forcing(lchnk, name) for name in FORCING}
        if self.macro is None:
            cam_in = st.cam_in(index)
            arrays = [cam_in[name] if name in cam_in else forcing[name] for name in MACROP_ARGUMENTS]

        # 2218: do macmic_it = 1, cld_macmic_num_steps
        for _macmic_it in range(1, n + 1):
            # 2220-2234: micro_do_icesupersat is off (refused at attach)
            # 2242-2250: macrop_scheme is not CLUBB_SGS (refused at attach); the driver
            if self.macro is None:
                H.macrop_driver_tend(lchnk, sub_dt, arrays)
                log("macrop_driver_tend")
            else:
                # the sub-walk, with the driver's dtime; then its ptend, det_s
                # and det_ice become the stage's, as the split stage's post-leaf did
                self._sub_walk(self.macro, st, lchnk, ncol, index, sub_dt, nstep)
                log("Macrophysics.tend_chunk")
                H.take_macro(lchnk); log("take_macro")
            det_s, det_ice = H.view(lchnk, VIEW["det_s"]), H.view(lchnk, VIEW["det_ice"])
            # 2254-2255: flx_cnd = -1*rliq ; flx_heat = det_s
            K("mm_flux_terms", {"ncol": ncol, "rliq": forcing["rliq"], "det_s": det_s},
              outputs={"flx_cnd": None, "flx_heat": None})
            log("mm_flux_terms")
            # 2262-2266
            H.ptend_scale(lchnk, PTEND, n, ncol); log("physics_ptend_scale")
            H.update_tend(lchnk, PTEND, dt); log("physics_update")
            H.check_energy(lchnk, "macrop_tend", nstep, dt, n,
                           zero, L["flx_cnd"], det_ice, L["flx_heat"], scaled=True)
            log("check_energy_chng:macrop_tend")
            # 2304-2314: subcolumns are off (refused at attach)
            # 2317-2322: aerosol activation
            if self.aero is None:
                H.microp_aero_run(lchnk, sub_dt); log("microp_aero_run")
            else:
                self._sub_walk(self.aero, st, lchnk, ncol, index, sub_dt, nstep)
                log("MicropAero.tend_chunk")
                H.take_aero(lchnk); log("take_aero")
            # 2325-2352: use_subcol_microp is off; the driver
            if self.micro is None:
                H.microp_driver_tend(lchnk, sub_dt); log("microp_driver_tend")
            else:
                self._sub_walk(self.micro, st, lchnk, ncol, index, sub_dt, nstep)
                log("Microphysics.tend_chunk")
                H.take_micro(lchnk); log("take_micro")
            # 2354-2357: the activation tendencies join the driver's
            H.ptend_sum_aero(lchnk, ncol); log("physics_ptend_sum:ptend_aero")
            # 2361-2366
            H.ptend_scale(lchnk, PTEND, n, ncol); log("physics_ptend_scale")
            H.update_tend(lchnk, PTEND, dt); log("physics_update")
            H.check_energy(lchnk, "microp_tend", nstep, dt, n,
                           zero, pbv["PREC_STR"], pbv["SNOW_STR"], zero, scaled=True)
            log("check_energy_chng:microp_tend")
            # 2369-2372: accumulate the substep's precipitation
            K("mm_precip_accumulate",
              {"ncol": ncol, "prec_sed": pbv["PREC_SED"], "snow_sed": pbv["SNOW_SED"],
               "prec_pcw": pbv["PREC_PCW"], "snow_pcw": pbv["SNOW_PCW"]},
              outputs={})
            log("mm_precip_accumulate")

        # 2376-2381: the substep means, and their sums
        K("mm_precip_average", {"ncol": ncol, "cld_macmic_num_steps": n},
          outputs={"prec_sed": pbv["PREC_SED"], "snow_sed": pbv["SNOW_SED"],
                   "prec_pcw": pbv["PREC_PCW"], "snow_pcw": pbv["SNOW_PCW"],
                   "prec_str": pbv["PREC_STR"], "snow_str": pbv["SNOW_STR"]})
        log("mm_precip_average")
        # 2386-2389: CARMA is off (refused at attach)
        # 2391-2393
        if C.trace_water:
            H.wtrc_mass_fixer(lchnk); log("wtrc_mass_fixer")

    @staticmethod
    def _sub_walk(stage: NativeStage, st: StageRuntime, lchnk: int, ncol: int, index: int,
                  dt: float, nstep: int) -> None:
        """One sub-stage's driver on one chunk, on that stage's own runtime."""

        runtime = stage.runtime(st.native)
        runtime.rank, runtime.nstep = st.rank, nstep
        del stage.calls[:]
        stage.tend_chunk(runtime, lchnk, ncol, index, dt, nstep)

    def tend(self, fields: Any, context: Any) -> None:
        super().tend(fields, context)
        # the sub-walks' profiles are written with this stage's
        for stage in self.components.values():
            for runtime in stage._runtimes.values():
                if isinstance(runtime.profile, StageProfile):
                    runtime.profile.write(runtime.rank)


__all__ = ["CloudMacroMicrophysics", "KERNELS", "MACROP_ARGUMENTS", "PBUF_FIELDS", "PTEND",
           "PTEND_AERO", "SEQUENCE", "SEQUENCE_WHOLE", "SEQUENCE_WHOLE_AERO", "SEQUENCE_WHOLE_MICRO", "VIEW"]
