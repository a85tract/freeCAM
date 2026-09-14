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
from ..pi_cam.pbuf import PBuf, PBufField, PBufFieldAbsent, load_pbuf_table
from .cloud_block import (
    BlockContract,
    CAM_IN_FIELDS,
    FORCING_FIELDS,
    MACRO_BLOCK,
    MICRO_BLOCK,
    STATE_FIELDS,
    OriginalBlock,
    cloud_borne_fields,
)
from .errors import PhysicsError
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
        # the Python driver's block write-back: the stage's tendency object initialised with the
        # flags a block's driver leaves on it, and those flags read for a capture
        "ptend_init": ("pycam_{prefix}_ptend_init_v1", [_INT, _INT, _STR, _INT, _INT, ctypes.POINTER(ctypes.c_int32)], True),
        "ptend_flags": ("pycam_{prefix}_ptend_flags_v1", [_INT, _INT, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)], True),
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

    def ptend_init(self, lchnk: int, which: int, name: str, *, ls: bool, lq: np.ndarray) -> None:
        """``physics_ptend_init(ptend, psetcols, name, ls=, lq=)`` on the stage's own object (``which`` 1) or
        its aerosol one (2): the Python driver allocates and flags it before writing a block's tendency."""

        flags = np.ascontiguousarray(np.asarray(lq, dtype=np.int32))
        assert flags.shape == (self.pcnst,), flags.shape
        _check(self.e.ptend_init(lchnk, int(which), name.encode("ascii"), len(name), int(bool(ls)),
                                 flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))),
               f"physics_ptend_init({name!r})")

    def ptend_flags(self, lchnk: int, which: int) -> tuple[bool, np.ndarray]:
        """The ``ls`` and ``lq`` flags on the stage's tendency object as the driver left them."""

        ls = ctypes.c_int32(0)
        lq = np.zeros(self.pcnst, dtype=np.int32)
        _check(self.e.ptend_flags(lchnk, int(which), ctypes.byref(ls), lq.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))),
               "ptend_flags")
        return bool(ls.value), lq


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
    WHOLE_ACTION = True          # tend is the whole action: stage 7 end to end
    PREFIX = "mm"
    PROCESS_NAME = "cloud_macro_microphysics"
    TRACE_ENV = "FREECAM_MM_TRACE"
    PROFILE_ENV = "FREECAM_MM_PROFILE"

    KERNELS = KERNELS
    CAM_IN = ("landfrac", "ocnfrac", "snowhland", "ts", "sst")
    #: the process slots the Python driver reads: the macrophysics block (``process``, the block
    #: mmacro_pcond lives in) and the microphysics block with its aerosol activation (``micro_process``)
    SLOT_NAMES = ("process", "micro_process")
    #: tphysbc's ``zero(pcols)``: an all-zero flux argument to the energy checks.
    EXTRA_SCRATCH = (("zero", ("pcols", "chunks")),)

    entries_class = _MMEntries
    services_class = _MMHandles

    def __init__(self, *, whole_drivers: bool = False, whole_micro: bool = False,
                 whole_aero: bool = False, micro_core_standalone: bool = False,
                 macro_surrogate: "str | Path | None" = None, kernels=None) -> None:
        super().__init__(kernels=None)
        #: the block slots (freecam.physics.cloud_block): None runs the original driver whole, in place;
        #: a BlockReplay or BlockModel answers the block and the Python driver writes its outputs back
        self.process: Any = None
        self.micro_process: Any = None
        #: a CloudBlockCapture records both blocks' inputs and outputs around the original drivers
        self.block_capture: Any = None
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

    # -- the Python driver: two compute blocks, the memory around them read and written from Python ------

    @property
    def block_armed(self) -> bool:
        """Whether a block slot or the capture is set: the stage then runs as the Python driver."""

        return self.process is not None or self.micro_process is not None or self.block_capture is not None

    def select_mode(self, native: Any = None) -> str:
        if self.block_armed:
            return "python-driver"
        return super().select_mode(native)

    def describe_process(self) -> dict[str, Any] | None:
        if not self.block_armed:
            return None
        described: dict[str, Any] = {}
        for slot in self.SLOT_NAMES:
            value = getattr(self, slot)
            described[slot] = dict(value.describe()) if value is not None else {"kind": "original", "block": slot}
        if self.block_capture is not None:
            described["capture"] = dict(self.block_capture.describe())
        return described

    def _block_pbuf(self, st: StageRuntime) -> PBuf:
        """A physics-buffer accessor over every field the two blocks read or write, built once a runtime."""

        buffer = getattr(st, "block_pbuf", None)
        if buffer is None:
            fields: dict[str, PBufField] = {}
            dynamic: dict[str, tuple[str, ...]] = {}
            for block in (MACRO_BLOCK, MICRO_BLOCK):
                extra = cloud_borne_fields(st.native.library, st.pcnst) if block.cloud_borne else ()
                dynamic[block.name] = tuple(field.name for field in extra)
                for field in block.buffers + block.input_buffers + extra:
                    if field.name in fields:
                        continue
                    index = field.index if field.index is not None else int(module_view(st.native.library, field.symbol, "int32", ()))
                    fields[field.name] = PBufField(field.name, index, field.time_sliced, field.rank, field.dtype)
            buffer = PBuf(st.native.library, {name: f for name, f in fields.items() if f.registered})
            st.block_pbuf = buffer
            #: the buffer fields registered per constituent at run time, by block: written by the block too
            st.block_dynamic = dynamic
        return buffer

    def _written_buffers(self, st: StageRuntime, block: BlockContract) -> tuple[str, ...]:
        """The buffer fields a block writes: its contract's, plus the ones this image registers per constituent."""

        self._block_pbuf(st)
        return block.buffer_names + tuple(getattr(st, "block_dynamic", {}).get(block.name, ()))

    def _block_views(self, st: StageRuntime, lchnk: int, ncol: int, index: int) -> dict[str, np.ndarray]:
        """The chunk's storage the blocks read and write, viewed once and kept: the state, the surface, the
        convection carries, every buffer field of both drivers (a field this configuration never registered is
        left out, as the driver's own pointer would be unassociated)."""

        cache = getattr(self, "_block_view_cache", None)
        if cache is None:
            cache = self._block_view_cache = {}
        if lchnk in cache:
            return cache[lchnk]
        H = st.handles
        pool = st.native.pool
        views: dict[str, np.ndarray] = {f"state_{name}": np.asarray(pool[f"phys_state.{name}"])[..., index] for name in STATE_FIELDS}
        cam_in = st.cam_in(index)
        views.update({f"cam_in_{name}": cam_in[name] for name in CAM_IN_FIELDS})
        views.update({name: H.forcing(lchnk, name) for name in FORCING})
        buffer = self._block_pbuf(st)
        names = set(buffer.fields) if hasattr(buffer, "fields") else set()
        for block in (MACRO_BLOCK, MICRO_BLOCK):
            names.update(self._written_buffers(st, block))
            names.update(field.name for field in block.input_buffers)
        views.update(self._buffer_views(buffer, sorted(names), lchnk))
        cache[lchnk] = views
        return views

    @staticmethod
    def _buffer_views(buffer: Any, names: Sequence[str], lchnk: int) -> dict[str, np.ndarray]:
        """Zero-copy views of the named buffer fields this configuration registered; a field it never registered
        (UNICON's detrainment, say, whose index the driver reads as -1) is left out, as the driver's pointer is."""

        views: dict[str, np.ndarray] = {}
        for name in names:
            if name not in buffer:
                continue
            try:
                views[name] = buffer.view(name, lchnk)
            except PBufFieldAbsent:
                continue
        return views

    def _block_inputs(self, st: StageRuntime, block: BlockContract, views: dict[str, np.ndarray], nstep: int, lchnk: int, ncol: int,
                      dt: float) -> dict[str, Any]:
        """What the block has in memory before its arithmetic, by the contract's names (live views, not copies)."""

        inputs: dict[str, Any] = {"nstep": int(nstep), "lchnk": int(lchnk), "ncol": int(ncol), "dt": float(dt)}
        inputs.update({name: views[name] for name in block.inputs + self._written_buffers(st, block) if name in views})
        return inputs

    def _block_outputs(self, st: StageRuntime, lchnk: int, ncol: int, block: BlockContract, views: dict[str, np.ndarray]) -> dict[str, Any]:
        """What the original block left behind, read from memory: the tendency object, the detrainment, the buffer fields."""

        H = st.handles
        ls, lq = H.ptend_flags(lchnk, PTEND)
        outputs: dict[str, Any] = {"ptend_s": H.view(lchnk, VIEW["ptend_s"])[:ncol], "ptend_q": H.view(lchnk, VIEW["ptend_q"])[:ncol],
                                   "ptend_ls": int(ls), "ptend_lq": lq}
        if block is MACRO_BLOCK:
            outputs["det_s"] = H.view(lchnk, VIEW["det_s"])[:ncol]
            outputs["det_ice"] = H.view(lchnk, VIEW["det_ice"])[:ncol]
        outputs.update({name: views[name][:ncol] for name in self._written_buffers(st, block) if name in views})
        return outputs

    def _write_block(self, st: StageRuntime, lchnk: int, ncol: int, block: BlockContract, answer: dict[str, Any],
                     views: dict[str, np.ndarray]) -> None:
        """Write a block's answer where the driver leaves it: the tendency object (allocated and flagged as the
        driver would have left it), the detrainment, and every buffer field the answer carries."""

        H = st.handles
        lq = np.asarray(answer["ptend_lq"], dtype=np.int32).reshape(-1)
        H.ptend_init(lchnk, PTEND, block.ptend_name, ls=bool(int(np.asarray(answer["ptend_ls"]).reshape(-1)[0])), lq=lq)
        H.view(lchnk, VIEW["ptend_s"])[:ncol] = np.asarray(answer["ptend_s"])[:ncol]
        H.view(lchnk, VIEW["ptend_q"])[:ncol] = np.asarray(answer["ptend_q"])[:ncol]
        if block is MACRO_BLOCK:
            H.view(lchnk, VIEW["det_s"])[:ncol] = np.asarray(answer["det_s"])[:ncol]
            H.view(lchnk, VIEW["det_ice"])[:ncol] = np.asarray(answer["det_ice"])[:ncol]
        for name in self._written_buffers(st, block):
            if name in answer and name in views:
                views[name][:ncol] = np.asarray(answer[name])[:ncol]

    def _tend_python_driver(self, native: Any, context: Any) -> None:
        """Stage 7 with its two compute blocks answered from their slots and everything around them done from memory.

        Per chunk: the macrophysics block (the original driver in place, or the slot's answer written back), the
        glue's flux terms, scaling, update and energy check; the microphysics block (activation, driver and the
        tendency sum, or the slot's answer); scaling, update, energy check, the precipitation means, the tracer
        mass fixer -- each of those the same Fortran call the glue makes.  Substepping is refused: the block
        contract is drawn for ``cld_macmic_num_steps = 1``, the admitted configuration's value.
        """

        st = self.runtime(native)
        C = st.constants
        if C.cld_macmic_num_steps != 1:
            raise PhysicsError(f"the Python driver of stage 7 takes cld_macmic_num_steps = 1, not {C.cld_macmic_num_steps}")
        entries = st.entries
        dt = float(entries.dt()) if entries.dt is not None else float(context.timestep_seconds)
        nstep = int(entries.nstep()) if entries.nstep is not None else int(context.step)
        st.nstep = nstep
        for index, (lchnk, ncol) in enumerate(zip(*native.chunks)):
            lchnk, n = int(lchnk), int(ncol)
            try:
                self._block_chunk(st, lchnk, n, index, dt, nstep)
            except Exception as error:
                print(f"[cloud python-driver] rank {getattr(st, 'rank', '?')} step {nstep} chunk {lchnk}: "
                      f"{type(error).__name__}: {error}", flush=True)
                raise
        self.execution.legacy_steps += 1

    def _block_chunk(self, st: StageRuntime, lchnk: int, n: int, index: int, dt: float, nstep: int) -> None:
        """One chunk of the Python driver: see :meth:`_tend_python_driver`.  Line numbers are physpkg.F90's."""

        H, C, pb = st.handles, st.constants, st.pbuf
        L = st.local
        log = self.calls.append
        capture = self.block_capture

        def K(name, inputs, *, outputs):
            st.kernel_on_chunk(name, inputs, outputs=outputs, ncol=n)

        V = self._block_views(st, lchnk, n, index)
        zero = L["zero"]
        pbv = {name: pb.view(name, lchnk) for name in PBUF_FIELDS}
        # 2210: cld_macmic_ztodt = ztodt/cld_macmic_num_steps, with the count 1: the step itself
        sub_dt = dt
        for name in ("prec_sed_macmic", "snow_sed_macmic", "prec_pcw_macmic", "snow_pcw_macmic"):
            st.scratch[name][...] = 0.0
        # -- 2242-2250: the macrophysics block
        inputs = self._block_inputs(st, MACRO_BLOCK, V, nstep, lchnk, n, sub_dt)
        before = capture.of(MACRO_BLOCK).begin(inputs) if capture is not None else None
        if self.process is None or isinstance(self.process, OriginalBlock):
            arrays = [V[f"cam_in_{name}"] if name in CAM_IN_FIELDS else V[name] for name in MACROP_ARGUMENTS]
            H.macrop_driver_tend(lchnk, sub_dt, arrays); log("macrop_driver_tend")
        else:
            self._write_block(st, lchnk, n, MACRO_BLOCK, self.process(inputs), V); log("macro_block_model")
        if before is not None:
            capture.of(MACRO_BLOCK).finish(before, self._block_outputs(st, lchnk, n, MACRO_BLOCK, V))
        det_s, det_ice = H.view(lchnk, VIEW["det_s"]), H.view(lchnk, VIEW["det_ice"])
        # 2254-2255, 2262-2266
        K("mm_flux_terms", {"ncol": n, "rliq": V["rliq"], "det_s": det_s}, outputs={"flx_cnd": None, "flx_heat": None})
        H.ptend_scale(lchnk, PTEND, 1, n); log("physics_ptend_scale")
        H.update_tend(lchnk, PTEND, dt); log("physics_update")
        H.check_energy(lchnk, "macrop_tend", nstep, dt, 1, zero, L["flx_cnd"], det_ice, L["flx_heat"], scaled=True)
        log("check_energy_chng:macrop_tend")
        # -- 2317-2357: the microphysics block: activation, driver, the tendency sum
        inputs = self._block_inputs(st, MICRO_BLOCK, V, nstep, lchnk, n, sub_dt)
        before = capture.of(MICRO_BLOCK).begin(inputs) if capture is not None else None
        if self.micro_process is None or isinstance(self.micro_process, OriginalBlock):
            H.microp_aero_run(lchnk, sub_dt); log("microp_aero_run")
            H.microp_driver_tend(lchnk, sub_dt); log("microp_driver_tend")
            H.ptend_sum_aero(lchnk, n); log("physics_ptend_sum:ptend_aero")
        else:
            self._write_block(st, lchnk, n, MICRO_BLOCK, self.micro_process(inputs), V); log("micro_block_model")
        if before is not None:
            capture.of(MICRO_BLOCK).finish(before, self._block_outputs(st, lchnk, n, MICRO_BLOCK, V))
        # 2361-2366
        H.ptend_scale(lchnk, PTEND, 1, n); log("physics_ptend_scale")
        H.update_tend(lchnk, PTEND, dt); log("physics_update")
        H.check_energy(lchnk, "microp_tend", nstep, dt, 1, zero, pbv["PREC_STR"], pbv["SNOW_STR"], zero, scaled=True)
        log("check_energy_chng:microp_tend")
        # 2369-2381
        K("mm_precip_accumulate", {"ncol": n, "prec_sed": pbv["PREC_SED"], "snow_sed": pbv["SNOW_SED"],
                                   "prec_pcw": pbv["PREC_PCW"], "snow_pcw": pbv["SNOW_PCW"]}, outputs={})
        K("mm_precip_average", {"ncol": n, "cld_macmic_num_steps": 1},
          outputs={"prec_sed": pbv["PREC_SED"], "snow_sed": pbv["SNOW_SED"], "prec_pcw": pbv["PREC_PCW"],
                   "snow_pcw": pbv["SNOW_PCW"], "prec_str": pbv["PREC_STR"], "snow_str": pbv["SNOW_STR"]})
        # 2391-2393
        if C.trace_water:
            H.wtrc_mass_fixer(lchnk); log("wtrc_mass_fixer")

    def tend(self, fields: Any, context: Any) -> None:
        if self.block_armed:
            native = context.native
            if native is None:
                raise PhysicsError(f"{type(self).__name__}.tend must run as a native process")
            self._current_step = getattr(context, "step", getattr(context, "nstep", None))
            self.execution.mode = "python-driver"
            self.execution.replacements = self.replacements()
            self._tend_python_driver(native, context)
            return
        super().tend(fields, context)
        # the sub-walks' profiles are written with this stage's
        for stage in self.components.values():
            for runtime in stage._runtimes.values():
                if isinstance(runtime.profile, StageProfile):
                    runtime.profile.write(runtime.rank)


__all__ = ["CloudMacroMicrophysics", "KERNELS", "MACROP_ARGUMENTS", "PBUF_FIELDS", "PTEND",
           "PTEND_AERO", "SEQUENCE", "SEQUENCE_WHOLE", "SEQUENCE_WHOLE_AERO", "SEQUENCE_WHOLE_MICRO", "VIEW"]
