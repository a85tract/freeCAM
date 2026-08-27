"""CAM5's RRTMG radiation as one Python class, in two hosts.

``Radiation`` is the radiation stage of the PI-atm workflow in the shape
``Macrophysics`` established: a Python object whose numerical routines are
methods that call the original Fortran, and whose control flow -- fetching
inputs, sequencing, bookkeeping, history output -- is Python.  It works
inside the driver (:meth:`attach` puts :meth:`tend` between the two halves
of the radiation stage, where it reproduces ``radiation_tend`` statement for
statement) and, for the two RRTMG cores, standalone.

Ten numerical routines are methods: the two cores :meth:`rad_rrtmg_sw` and
:meth:`rad_rrtmg_lw`, the six cloud-optics routines, and the two aerosol
ones.  The cores are the ones a model may replace -- assign into
:attr:`kernels` -- because they are where the cost is and where a surrogate
has something to learn.

The rule that makes the in-model path testable is the one macrophysics
proved: **Python computes no floating-point number.**  Every arithmetic
statement of the Fortran driver is one of the seventeen routines lifted into
``pycam_rad_kernels``; every call taking a derived type goes through
``pycam_rad_handles``; every buffer field is CAM's own storage.  Python
orders the calls and passes addresses.  A run driven this way is expected to
be bit-for-bit with the oracle, and Gate R-B2 asserts it.

Two statements of the driver are not reachable that way and are named where
they occur: the pair at 1170-1171 that hands ``outfld`` an expression, which
``pycam_rad_outfld_scaled_v1`` keeps whole, and ``radiation_do``, which is
called rather than re-derived so the stage cannot drift out of cadence.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ..pi_cam.errors import PICAMConfigurationError
from ..pi_cam.pbuf import PBuf, PBufField
from .errors import PhysicsError
from .image import module_view
from .stage import (
    CORE_ENTRIES,
    HostEntries,
    HostServices,
    NativeStage,
    StageRuntime,
    check as _check,
    fortran as _f,
    pointer_of as _ptr,
)

REPO = Path(__file__).resolve().parents[3]

SW, LW = "rad_rrtmg_sw", "rad_rrtmg_lw"

# radconstants.F90 and the RRTMG parameter modules declare these as named
# constants, so the image has no symbol for them.  A test pins each against
# the pinned source.
NBNDSW, NBNDLW = 14, 16
IDX_SW_DIAG = 10
RRTMG_SW_CLOUDSIM_BAND, RRTMG_LW_CLOUDSIM_BAND = 9, 6
N_DIAG = 10
FILLVALUE = 1.0e36

# physconst.F90 declares cpair and stebol as parameters of shr_const_mod, so
# the image has no symbol for those either.  cappa is a module variable there
# and is read from the image.
CPAIR, STEBOL = 1.00464e3, 5.67e-8

#: ``pycam_rad_handles`` view codes.  A test keeps this table equal to the
#: Fortran one.
VIEW = {
    "state_t": 1, "state_pmid": 2, "state_pint": 3, "state_pdel": 4,
    "state_lnpint": 5, "state_lnpmid": 6,
    "ptend_s": 21,
    "cam_in_lwup": 41, "cam_in_asdir": 42, "cam_in_asdif": 43,
    "cam_in_aldir": 44, "cam_in_aldif": 45,
    "cam_out_sols": 51, "cam_out_soll": 52, "cam_out_solsd": 53,
    "cam_out_solld": 54, "cam_out_flwds": 55, "cam_out_netsw": 56,
    "fsns": 61, "fsnt": 62, "flns": 63, "flnt": 64, "fsds": 65,
    "net_flx": 71,
    "rstate_h2ovmr": 81, "rstate_o3vmr": 82, "rstate_co2vmr": 83,
    "rstate_ch4vmr": 84, "rstate_o2vmr": 85, "rstate_n2ovmr": 86,
    "rstate_cfc11vmr": 87, "rstate_cfc12vmr": 88, "rstate_cfc22vmr": 89,
    "rstate_ccl4vmr": 90, "rstate_pmidmb": 91, "rstate_pintmb": 92,
    "rstate_tlay": 93, "rstate_tlev": 94,
}

#: The physics-buffer fields the driver reads, with the two it takes at the
#: older time sample.  ``QRS`` and ``QRL`` it reads and writes in place.
RADIATION_FIELDS = (
    ("CLD", "cld_idx", True),
    ("CLDFSNOW", "cldfsnow_idx", True),
    ("QRS", "qrs_idx", False),
    ("QRL", "qrl_idx", False),
)

#: The direct kernels tend() runs; every one must be in the image and in the
#: reviewed descriptors with the same field list.
KERNELS = (
    "rad_gather_day_night", "rad_combine_cld_optics_sw", "rad_snow_blend_sw",
    "rad_combine_cld_optics_lw", "rad_snow_blend_lw", "rad_interface_temperature",
    "rad_sw_cloud_forcing", "rad_scale_by_cpair", "rad_visible_tau",
    "rad_lwup_cgs", "rad_lw_cloud_forcing", "rad_emissivity", "rad_snow_gridbox",
    "rad_heating_unscale", "rad_theta_heating", "rad_heating_scale", "rad_inp",
    "get_variability", "vertinterp",
)

#: The order in which the Fortran driver calls things; ``tend`` follows it and
#: a test compares the two.  Names are the routine, the handle entry, or the
#: predicate Python branches on.
SEQUENCE_RADIATION_STEP = (
    "calday", "pbuf_get_field*", "outfld*", "latlon", "zenith",
    "rad_gather_day_night", "radiation_do:sw", "radiation_do:lw",
    "rrtmg_state_create",
    "get_ice_optics_sw", "get_liquid_optics_sw", "rad_combine_cld_optics_sw",
    "get_snow_optics_sw", "rad_snow_blend_sw",
    "ice_cloud_get_rad_props_lw", "liquid_cloud_get_rad_props_lw",
    "rad_combine_cld_optics_lw", "snow_cloud_get_rad_props_lw", "rad_snow_blend_lw",
    "radinp", "rad_interface_temperature",
    "get_variability", "rrtmg_state_update", "aer_rad_props_sw", "rad_rrtmg_sw",
    "vertinterp*", "rad_sw_cloud_forcing", "rad_scale_by_cpair", "outfld*",
    "rad_scale_by_cpair", "outfld*", "rad_visible_tau", "outfld*",
    "rad_cnst_out",
    "rad_lwup_cgs", "rrtmg_state_update", "aer_rad_props_lw", "rad_rrtmg_lw",
    "rad_lw_cloud_forcing", "vertinterp*", "outfld_scaled", "outfld_scaled",
    "outfld*",
    "rrtmg_state_destroy", "rad_emissivity", "outfld*", "rad_snow_gridbox",
    "rad_data_write", "radheat_tend", "rad_theta_heating", "outfld*",
    "rad_heating_scale",
)

#: What the driver does on a step where neither shortwave nor longwave runs.
SEQUENCE_QUIET_STEP = (
    "calday", "pbuf_get_field*", "outfld*", "latlon", "zenith",
    "rad_gather_day_night", "radiation_do:sw", "radiation_do:lw",
    "rad_heating_unscale",
    "rad_data_write", "radheat_tend", "rad_theta_heating", "outfld*",
    "rad_heating_scale",
)


# -- the image, seen through ctypes ---------------------------------------------


class _RadEntries(HostEntries):
    """The shared core plus radiation's own wrappers and predicates.

    Radiation declares none of ``PTEND_ENTRIES``: its driver never copies the
    physics state, and ``radheat_tend`` builds the ptend itself.
    """

    _c = ctypes
    _INT, _DBL, _STR = _c.c_int, _c.c_double, _c.c_char_p
    _PD, _PI = _c.POINTER(_c.c_double), _c.POINTER(_c.c_int)

    TABLE = {
        **CORE_ENTRIES,
        # the driver's own predicates and host queries
        "calday": ("pycam_{prefix}_calday_v1", [_PD], False),
        "do": ("pycam_{prefix}_do_v1", [_INT, _PD, _PD], False),
        "latlon": ("pycam_{prefix}_latlon_v1", [_INT, _INT, _PD, _PD], False),
        # zenith is a bare external subroutine, so it is a handle, not a kernel
        "zenith": ("pycam_{prefix}_zenith_v1", [_INT, _INT, _DBL, _PD, _PD, _PD], False),
        "options": ("pycam_{prefix}_options_v1", [_PI], False),
        "hist_active": ("pycam_{prefix}_hist_active_v1", [_STR, _INT], False),
        # the RRTMG state's chunk-local lifetime
        "rstate_create": ("pycam_{prefix}_rstate_create_v1", [_INT], False),
        "rstate_update": ("pycam_{prefix}_rstate_update_v1", [_INT, _INT], False),
        "rstate_destroy": ("pycam_{prefix}_rstate_destroy_v1", [_INT], False),
        # the calls that take a derived type: pointers passed positionally, so
        # ctypes is not told the signature and the wrapper's dummy list is the
        # contract a test audits
        "ice_optics_sw": ("pycam_{prefix}_ice_optics_sw_v1", None, False),
        "liquid_optics_sw": ("pycam_{prefix}_liquid_optics_sw_v1", None, False),
        "snow_optics_sw": ("pycam_{prefix}_snow_optics_sw_v1", None, False),
        "ice_props_lw": ("pycam_{prefix}_ice_props_lw_v1", None, False),
        "liquid_props_lw": ("pycam_{prefix}_liquid_props_lw_v1", None, False),
        "snow_props_lw": ("pycam_{prefix}_snow_props_lw_v1", None, False),
        "aer_props_sw": ("pycam_{prefix}_aer_props_sw_v1", None, False),
        "aer_props_lw": ("pycam_{prefix}_aer_props_lw_v1", None, False),
        "rrtmg_sw": ("pycam_{prefix}_rrtmg_sw_v1", None, False),
        "rrtmg_lw": ("pycam_{prefix}_rrtmg_lw_v1", None, False),
        "tropopause_find": ("pycam_{prefix}_tropopause_find_v1", None, False),
        "cnst_out": ("pycam_{prefix}_cnst_out_v1", [_INT, _INT], False),
        "data_write": ("pycam_{prefix}_data_write_v1", None, False),
        "radheat": ("pycam_{prefix}_radheat_v1", None, False),
        "outfld_scaled": ("pycam_{prefix}_outfld_scaled_v1", None, False),
    }


class _RadHandles(HostServices):
    """CAM's host services, plus the calls only radiation makes."""

    _I, _D = ctypes.c_int, ctypes.c_double

    # -- the driver's predicates, called rather than re-derived ---------------

    def calday(self) -> float:
        value = ctypes.c_double()
        _check(self.e.calday(ctypes.byref(value)), "get_curr_calday")
        return float(value.value)

    def radiation_do(self, op: str) -> bool:
        """``radiation_do('sw'|'lw')``, the Fortran's own answer."""

        unused = ctypes.c_double()
        status = self.e.do({"sw": 1, "lw": 2}[op], ctypes.byref(unused),
                           ctypes.byref(unused))
        if status < 0:
            raise PICAMConfigurationError(f"radiation_do({op!r}) refused")
        return bool(status)

    def latlon(self, lchnk: int, ncol: int, clat: np.ndarray, clon: np.ndarray) -> None:
        _check(self.e.latlon(lchnk, ncol, _ptr(clat), _ptr(clon)), "get_rlat/rlon_all_p")

    def zenith(self, lchnk: int, ncol: int, calday: float, clat: np.ndarray,
               clon: np.ndarray, coszrs: np.ndarray) -> None:
        _check(self.e.zenith(lchnk, ncol, calday, _ptr(clat), _ptr(clon), _ptr(coszrs)),
               "zenith")

    def options(self) -> dict[str, int]:
        codes = (ctypes.c_int * 4)()
        _check(self.e.options(codes), "pycam_rad_options_v1")
        return {"oldcldoptics": codes[0], "icecldoptics_mitchell": codes[1],
                "liqcldoptics_gammadist": codes[2], "active_calls": codes[3]}

    def hist_active(self, name: str) -> bool:
        return bool(self.e.hist_active(name.encode("ascii"), len(name)))

    # -- the RRTMG state ------------------------------------------------------

    def rstate_create(self, lchnk: int) -> None:
        _check(self.e.rstate_create(lchnk), "rrtmg_state_create")

    def rstate_update(self, lchnk: int, icall: int) -> None:
        _check(self.e.rstate_update(lchnk, icall), "rrtmg_state_update")

    def rstate_destroy(self, lchnk: int) -> None:
        _check(self.e.rstate_destroy(lchnk), "rrtmg_state_destroy")

    # -- the calls that take a derived type -----------------------------------

    def _optics(self, entry, what: str, lchnk: int, arrays) -> None:
        _check(entry(self._I(lchnk), *[_ptr(a) for a in arrays]), what)

    def ice_optics_sw(self, lchnk, arrays):
        self._optics(self.e.ice_optics_sw, "get_ice_optics_sw", lchnk, arrays)

    def liquid_optics_sw(self, lchnk, arrays):
        self._optics(self.e.liquid_optics_sw, "get_liquid_optics_sw", lchnk, arrays)

    def snow_optics_sw(self, lchnk, arrays):
        self._optics(self.e.snow_optics_sw, "get_snow_optics_sw", lchnk, arrays)

    def ice_props_lw(self, lchnk, abs_od):
        self._optics(self.e.ice_props_lw, "ice_cloud_get_rad_props_lw", lchnk, [abs_od])

    def liquid_props_lw(self, lchnk, abs_od):
        self._optics(self.e.liquid_props_lw, "liquid_cloud_get_rad_props_lw", lchnk, [abs_od])

    def snow_props_lw(self, lchnk, abs_od):
        self._optics(self.e.snow_props_lw, "snow_cloud_get_rad_props_lw", lchnk, [abs_od])

    def aer_props_sw(self, lchnk, list_idx, nnite, idxnite, arrays):
        _check(self.e.aer_props_sw(
            self._I(lchnk), self._I(list_idx), self._I(nnite),
            idxnite.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            *[_ptr(a) for a in arrays]), "aer_rad_props_sw")

    def aer_props_lw(self, lchnk, list_idx, odap_aer):
        _check(self.e.aer_props_lw(self._I(lchnk), self._I(list_idx), _ptr(odap_aer)),
               "aer_rad_props_lw")

    def tropopause_find(self, lchnk, troplev, trop_p):
        _check(self.e.tropopause_find(
            self._I(lchnk), troplev.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            _ptr(trop_p)), "tropopause_find")

    def cnst_out(self, lchnk, list_idx):
        _check(self.e.cnst_out(lchnk, list_idx), "rad_cnst_out")

    def data_write(self, lchnk, coszrs):
        _check(self.e.data_write(self._I(lchnk), _ptr(coszrs)), "rad_data_write")

    def radheat(self, lchnk, qrl, qrs):
        _check(self.e.radheat(self._I(lchnk), _ptr(qrl), _ptr(qrs)), "radheat_tend")

    def outfld_scaled(self, lchnk, ncol, name, field, cpair):
        """radiation.F90:1170-1171: the division and the shape are one
        expression that outfld is given with ``idim = ncol``."""

        field = _f(field)
        _check(self.e.outfld_scaled(
            self._I(lchnk), self._I(ncol), name.encode("ascii"), self._I(len(name)),
            _ptr(field), self._D(cpair)), f"outfld({name!r}) scaled")

    def rrtmg_sw(self, lchnk, scalars, arrays):
        _check(self.e.rrtmg_sw(*self._core_arguments(lchnk, scalars, arrays)),
               "rad_rrtmg_sw")

    def rrtmg_lw(self, lchnk, scalars, arrays):
        _check(self.e.rrtmg_lw(*self._core_arguments(lchnk, scalars, arrays)),
               "rad_rrtmg_lw")

    def _core_arguments(self, lchnk, scalars, arrays):
        out = [self._I(lchnk)]
        for value in scalars:
            out.append(self._D(value) if isinstance(value, float) else self._I(int(value)))
        for array in arrays:
            if array.dtype == np.int32:
                out.append(array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            else:
                out.append(_ptr(array))
        return out


# -- module constants --------------------------------------------------------------


@dataclass(frozen=True)
class _Constants:
    qrs_idx: int
    qrl_idx: int
    cld_idx: int
    cldfsnow_idx: int
    su_idx: int
    sd_idx: int
    lu_idx: int
    ld_idx: int
    iradsw: int
    iradlw: int
    irad_always: int
    spectralflux: bool
    dohirs: bool
    docosp: bool
    num_rrtmg_levs: int
    cappa: float
    #: from pycam_rad_options_v1: module state module_view cannot read
    oldcldoptics: bool = False
    icecldoptics_mitchell: bool = True
    liqcldoptics_gammadist: bool = True
    active_calls: int = 1
    #: hist_fld_active for the two fields that guard a vertinterp loop
    fsnr_active: bool = False
    flnr_active: bool = False

    @classmethod
    def read(cls, library: Any) -> "_Constants":
        def i(symbol):
            return int(module_view(library, symbol, "int32", ()))

        def b(symbol):
            return bool(int(module_view(library, symbol, "int32", ())))

        def r(symbol):
            return float(module_view(library, symbol, "float64", ()))

        return cls(
            qrs_idx=i("radiation_mp_qrs_idx_"), qrl_idx=i("radiation_mp_qrl_idx_"),
            cld_idx=i("radiation_mp_cld_idx_"),
            cldfsnow_idx=i("radiation_mp_cldfsnow_idx_"),
            su_idx=i("radiation_mp_su_idx_"), sd_idx=i("radiation_mp_sd_idx_"),
            lu_idx=i("radiation_mp_lu_idx_"), ld_idx=i("radiation_mp_ld_idx_"),
            iradsw=i("radiation_mp_iradsw_"), iradlw=i("radiation_mp_iradlw_"),
            irad_always=i("radiation_mp_irad_always_"),
            spectralflux=b("radiation_mp_spectralflux_"),
            dohirs=b("radiation_mp_dohirs_"),
            docosp=b("cospsimulator_intr_mp_docosp_"),
            num_rrtmg_levs=i("rrtmg_state_mp_num_rrtmg_levs_"),
            cappa=r("physconst_mp_cappa_"),
        )

    def refuse_unsupported(self) -> None:
        """The paths the admitted configuration never takes are not ported."""

        refusals = (
            (self.spectralflux, "spectralflux is on; the Python radiation passes the "
                                "driver's null spectral-flux pointers"),
            (self.docosp, "docosp is on; the Python radiation does not carry COSP"),
            (self.dohirs, "dohirs is on; the Python radiation does not carry hirsrtm"),
            (self.oldcldoptics, "oldcldoptics is on; the Python radiation carries the "
                                "select-case optics branches only"),
            (not self.icecldoptics_mitchell,
             "icecldoptics is not 'mitchell'; the Python radiation carries that branch"),
            (not self.liqcldoptics_gammadist,
             "liqcldoptics is not 'gammadist'; the Python radiation carries that branch"),
            (self.active_calls != 1,
             f"{self.active_calls} radiation calls are active; the Python radiation "
             f"carries the climate call alone"),
            (self.fsnr_active or self.flnr_active,
             "FSNR or FLNR is on a history tape; the Python radiation does not carry "
             "their per-column vertinterp loops"),
        )
        for refused, why in refusals:
            if refused:
                raise PICAMConfigurationError(why)


# -- the class ---------------------------------------------------------------------


class Radiation(NativeStage):
    """CAM5 RRTMG radiation: cloud optics, the two cores, radheat, as Python.

    :attr:`kernels` holds what computes each core inside the model:
    ``None`` runs the original through its handle wrapper; a callable taking
    the batch mapping ``{name: (ncol, ...) array}`` and returning the outputs
    by name -- a ``torch.nn.Module`` wrapped that way, say -- takes its
    place.  Nothing else in :meth:`tend_chunk` changes when it does.
    """

    STAGE = "cam_run1.radiation"
    FIRST_HALF = "cam_run1.rad_tend_pre_leaf"
    SECOND_HALF = "cam_run1.rad_tend_post_leaf"
    PREFIX = "rad"
    PROCESS_NAME = "rad_tend"
    TRACE_ENV = "FREECAM_RAD_TRACE"

    KERNELS = KERNELS
    SWAPPABLE = (SW, LW)
    #: What the driver keeps that no kernel declares.
    EXTRA_SCRATCH = (
        ("clat", ("pcols", "chunks")), ("clon", ("pcols", "chunks")),
        # the driver reads CLDFSNOW only when the field is registered; when it
        # is not, the arrays that would have held it stay zero, and the lifted
        # routines take has_snow = .false. and never look
        ("cldfsnow_zero", ("pcols", "pver", "chunks")),
        ("aer_tau", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_tau_w", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_tau_w_g", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_tau_w_f", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_lw_abs", ("pcols", "pver", "nbndlw", "chunks")),
        ("qrsc", ("pcols", "pver", "chunks")), ("qrlc", ("pcols", "pver", "chunks")),
        ("fns", ("pcols", "pverp", "chunks")), ("fcns", ("pcols", "pverp", "chunks")),
        ("fnl", ("pcols", "pverp", "chunks")), ("fcnl", ("pcols", "pverp", "chunks")),
    ) + tuple(
        (name, ("pcols", "chunks")) for name in (
            "solin", "fsutoa", "fsnirt", "fsnrtc", "fsnirtsq",
            "fsntc", "fsnsc", "fsdsc", "flntc", "flnsc", "fldsc",
            "fsn200", "fsn200c", "fln200", "fln200c", "fsnr", "flnr",
        )
    )

    entries_class = _RadEntries
    services_class = _RadHandles

    def __init__(self, *, kernels: Mapping[str, Callable | None] | None = None) -> None:
        super().__init__(kernels=kernels)

    # -- what the runtime asks of this stage -------------------------------------

    def read_constants(self, library: Any) -> _Constants:
        return _Constants.read(library)

    def refuse_unsupported(self, constants: _Constants) -> None:
        constants.refuse_unsupported()

    def extra_extents(self, constants: _Constants) -> Mapping[str, int]:
        # nswbands and nlwbands are radconstants' names for the same band
        # counts parrrsw and parrrtm call nbndsw and nbndlw; get_variability's
        # sfac is declared with the former, the lifted routines with the
        # latter, and a test pins the two definitions equal.
        return {"nbndsw": NBNDSW, "nbndlw": NBNDLW,
                "nswbands": NBNDSW, "nlwbands": NBNDLW,
                "rrtmg_levs": constants.num_rrtmg_levs,
                "rrtmg_levsp": constants.num_rrtmg_levs + 1}

    def build_pbuf(self, library: Any, runtime: StageRuntime) -> PBuf:
        indices = {symbol: int(module_view(library, f"radiation_mp_{symbol}_", "int32", ()))
                   for _, symbol, _ in RADIATION_FIELDS}
        fields = {name: PBufField(name, int(indices[symbol]), sliced)
                  for name, symbol, sliced in RADIATION_FIELDS}
        buffer = PBuf(library, fields)
        lchnk, _ = runtime.native.chunks
        buffer.verify(int(lchnk[0]), pcols=runtime.pcols, pver=runtime.pver)
        return buffer

    def after_runtime(self, runtime: StageRuntime) -> None:
        """Finish the constants with what only a handle can answer."""

        handles, constants = runtime.handles, runtime.constants
        options = handles.options()
        runtime.constants = _Constants(
            **{k: v for k, v in vars(constants).items()
               if k not in ("oldcldoptics", "icecldoptics_mitchell",
                            "liqcldoptics_gammadist", "active_calls",
                            "fsnr_active", "flnr_active")},
            oldcldoptics=bool(options["oldcldoptics"]),
            icecldoptics_mitchell=bool(options["icecldoptics_mitchell"]),
            liqcldoptics_gammadist=bool(options["liqcldoptics_gammadist"]),
            active_calls=int(options["active_calls"]),
            fsnr_active=handles.hist_active("FSNR    "),
            flnr_active=handles.hist_active("FLNR    "),
        )
        runtime.constants.refuse_unsupported()
        runtime.has_snow = runtime.constants.cldfsnow_idx > 0

    # -- the numerical methods ---------------------------------------------------
    #
    # Ten routines the driver calls to compute something.  Each takes the
    # runtime and a chunk and returns the arrays the driver would have had;
    # the two cores additionally accept a model in their place.

    def ice_optics_sw(self, st: StageRuntime, lchnk: int) -> tuple[np.ndarray, ...]:
        """``get_ice_optics_sw``: ice cloud optics for the shortwave bands."""

        out = [st.local[n] for n in ("ice_tau", "ice_tau_w", "ice_tau_w_g", "ice_tau_w_f")]
        st.handles.ice_optics_sw(lchnk, out)
        return tuple(out)

    def liquid_optics_sw(self, st: StageRuntime, lchnk: int) -> tuple[np.ndarray, ...]:
        """``get_liquid_optics_sw``: liquid cloud optics, shortwave."""

        out = [st.local[n] for n in ("liq_tau", "liq_tau_w", "liq_tau_w_g", "liq_tau_w_f")]
        st.handles.liquid_optics_sw(lchnk, out)
        return tuple(out)

    def snow_optics_sw(self, st: StageRuntime, lchnk: int) -> tuple[np.ndarray, ...]:
        """``get_snow_optics_sw``: snow optics, shortwave."""

        out = [st.local[n] for n in ("snow_tau", "snow_tau_w", "snow_tau_w_g", "snow_tau_w_f")]
        st.handles.snow_optics_sw(lchnk, out)
        return tuple(out)

    def ice_props_lw(self, st: StageRuntime, lchnk: int) -> np.ndarray:
        """``ice_cloud_get_rad_props_lw``: ice absorption optical depth."""

        out = st.local["ice_lw_abs"]
        st.handles.ice_props_lw(lchnk, out)
        return out

    def liquid_props_lw(self, st: StageRuntime, lchnk: int) -> np.ndarray:
        """``liquid_cloud_get_rad_props_lw``: liquid absorption optical depth."""

        out = st.local["liq_lw_abs"]
        st.handles.liquid_props_lw(lchnk, out)
        return out

    def snow_props_lw(self, st: StageRuntime, lchnk: int) -> np.ndarray:
        """``snow_cloud_get_rad_props_lw``: snow absorption optical depth."""

        out = st.local["snow_lw_abs"]
        st.handles.snow_props_lw(lchnk, out)
        return out

    def aer_props_sw(self, st: StageRuntime, lchnk: int, nnite: int,
                     idxnite: np.ndarray) -> tuple[np.ndarray, ...]:
        """``aer_rad_props_sw``: aerosol optics for the climate call."""

        out = [st.local[n] for n in ("aer_tau", "aer_tau_w", "aer_tau_w_g", "aer_tau_w_f")]
        st.handles.aer_props_sw(lchnk, 0, nnite, idxnite, out)
        return tuple(out)

    def aer_props_lw(self, st: StageRuntime, lchnk: int) -> np.ndarray:
        """``aer_rad_props_lw``: aerosol absorption optical depth."""

        out = st.local["aer_lw_abs"]
        st.handles.aer_props_lw(lchnk, 0, out)
        return out

    def rad_rrtmg_sw(self, st: StageRuntime, lchnk: int, ncol: int, dt: float,
                     inputs: Mapping[str, Any], outputs: Mapping[str, np.ndarray],
                     scalars, arrays) -> None:
        """The shortwave core.  A model in :attr:`kernels` takes its place."""

        st.swappable_kernel(
            SW, inputs, outputs=outputs, ncol=ncol, lchnk=lchnk, dt=dt,
            original=lambda: st.handles.rrtmg_sw(lchnk, scalars, arrays))

    def rad_rrtmg_lw(self, st: StageRuntime, lchnk: int, ncol: int, dt: float,
                     inputs: Mapping[str, Any], outputs: Mapping[str, np.ndarray],
                     scalars, arrays) -> None:
        """The longwave core.  A model in :attr:`kernels` takes its place."""

        st.swappable_kernel(
            LW, inputs, outputs=outputs, ncol=ncol, lchnk=lchnk, dt=dt,
            original=lambda: st.handles.rrtmg_lw(lchnk, scalars, arrays))

    # -- the transliteration -----------------------------------------------------

    def tend_chunk(self, st: StageRuntime, lchnk: int, ncol: int, index: int,
                   dt: float, nstep: int) -> None:
        """``radiation_tend``, statement for statement, for one chunk.

        Line numbers in the trailing comments are ``physics/rrtmg/radiation.F90``
        in the pinned submodule.  Branches the admitted configuration never
        enters are refused at attach and are not written here; the sequence
        test compares what this walks against the pinned call order with those
        branches removed.
        """

        H, C, pb = st.handles, st.constants, st.pbuf
        L = st.local
        S = {name: H.view(lchnk, code) for name, code in VIEW.items()
             if name.startswith("state_")}
        cam_in = {name.removeprefix("cam_in_"): H.view(lchnk, code)
                  for name, code in VIEW.items() if name.startswith("cam_in_")}
        cam_out = {name.removeprefix("cam_out_"): H.view(lchnk, code)
                   for name, code in VIEW.items() if name.startswith("cam_out_")}
        flux = {name: H.view(lchnk, VIEW[name])
                for name in ("fsns", "fsnt", "flns", "flnt", "fsds")}
        pcols, pver, pverp = st.pcols, st.pver, st.pverp
        has_snow = st.has_snow
        log = self.calls.append

        def K(name, inputs, *, outputs):
            st.kernel_on_chunk(name, inputs, outputs=outputs, ncol=ncol)

        # 807-812: lchnk, ncol; the calendar day the zenith angle needs
        calday = H.calday(); log("calday")

        # 816-830: the physics-buffer fields, older time sample where the
        # source says so.  spectralflux is refused, so su/sd/lu/ld are not
        # fetched here any more than the driver fetches them when it is off.
        cld = pb.view("CLD", lchnk)
        cldfsnow = pb.view("CLDFSNOW", lchnk) if has_snow else L["cldfsnow_zero"]
        qrs = pb.view("QRS", lchnk)
        qrl = pb.view("QRL", lchnk)
        log("pbuf_get_field*")

        # 836-838
        if has_snow:
            H.outfld("CLDFSNOW", cldfsnow, pcols, lchnk)
        log("outfld")

        # 843-845: the cosine of the solar zenith angle for this step
        clat, clon, coszrs = L["clat"], L["clon"], L["coszrs"]
        H.latlon(lchnk, ncol, clat, clon); log("latlon")
        H.zenith(lchnk, ncol, calday, clat, clon, coszrs); log("zenith")

        # 856-866: which columns are lit.  The counts and the index arrays go
        # straight into the shortwave core, which does the gather itself.
        K("rad_gather_day_night", {"ncol": ncol, "coszrs": coszrs},
          outputs={"Nday": None, "Nnite": None, "IdxDay": None, "IdxNite": None})
        log("rad_gather_day_night")
        nday = int(L["Nday"][0])
        nnite = int(L["Nnite"][0])
        idxday, idxnite = L["IdxDay"], L["IdxNite"]

        # 868-869: the driver's own predicates, asked rather than re-derived
        dosw = H.radiation_do("sw"); log("radiation_do:sw")
        dolw = H.radiation_do("lw"); log("radiation_do:lw")

        if dosw or dolw:                                             # 875
            self._radiative_step(st, lchnk, ncol, dt, dosw, dolw, nday, nnite,
                                 idxday, idxnite, cld, cldfsnow, qrs, qrl,
                                 coszrs, S, cam_in, cam_out, flux)
        else:                                                        # 1275
            # 1277-1287: the heating rates are kept as Q*dp between radiation
            # steps, so a quiet step only converts them back.
            K("rad_heating_unscale",
              {"ncol": ncol, "conserve_energy": 1, "pdel": S["state_pdel"],
               "qrs": qrs, "qrl": qrl},
              outputs={"qrs": qrs, "qrl": qrl})
            log("rad_heating_unscale")

        # 1292: rad_data_write returns at once unless radiation output is on
        H.data_write(lchnk, coszrs); log("rad_data_write")

        # 1295-1296: radheat_tend fills the ptend and the net flux the resume
        # half of tphysbc takes back
        H.radheat(lchnk, qrl, qrs); log("radheat_tend")

        # 1298-1304: the heating rate for dtheta/dt
        K("rad_theta_heating",
          {"ncol": ncol, "qrs": qrs, "qrl": qrl, "cpair": CPAIR,
           "pmid": S["state_pmid"], "cappa": C.cappa},
          outputs={"ftem": None})
        log("rad_theta_heating")
        H.outfld("HR      ", L["ftem"], pcols, lchnk); log("outfld")

        # 1306-1316
        K("rad_heating_scale",
          {"ncol": ncol, "conserve_energy": 1, "pdel": S["state_pdel"],
           "qrs": qrs, "qrl": qrl},
          outputs={"qrs": qrs, "qrl": qrl})
        log("rad_heating_scale")

        # 1318
        cam_out["netsw"][:ncol] = flux["fsns"][:ncol]                # [exact] a copy

    def _radiative_step(self, st, lchnk, ncol, dt, dosw, dolw, nday, nnite,
                        idxday, idxnite, cld, cldfsnow, qrs, qrl, coszrs,
                        S, cam_in, cam_out, flux) -> None:
        """``radiation.F90:875-1273``: the branch that computes."""

        H, C = st.handles, st.constants
        L = st.local
        pcols, pver, pverp = st.pcols, st.pver, st.pverp
        has_snow = st.has_snow
        log = self.calls.append

        def K(name, inputs, *, outputs):
            st.kernel_on_chunk(name, inputs, outputs=outputs, ncol=ncol)

        # 878: the RRTMG state, alive until 1192
        H.rstate_create(lchnk); log("rrtmg_state_create")

        if dosw:                                                     # 890
            # 898, 906: the two optics branches this configuration takes
            self.ice_optics_sw(st, lchnk); log("get_ice_optics_sw")
            self.liquid_optics_sw(st, lchnk); log("get_liquid_optics_sw")
            # 912-915
            K("rad_combine_cld_optics_sw",
              {"ncol": ncol,
               "liq_tau": None, "liq_tau_w": None, "liq_tau_w_g": None, "liq_tau_w_f": None,
               "ice_tau": None, "ice_tau_w": None, "ice_tau_w_g": None, "ice_tau_w_f": None},
              outputs={"cld_tau": None, "cld_tau_w": None,
                       "cld_tau_w_g": None, "cld_tau_w_f": None})
            log("rad_combine_cld_optics_sw")
            # 917-945, with 919's optics call made first
            if has_snow:
                self.snow_optics_sw(st, lchnk)
            log("get_snow_optics_sw")
            K("rad_snow_blend_sw",
              {"ncol": ncol, "has_snow": int(has_snow), "cld": cld, "cldfsnow": cldfsnow,
               "snow_tau": None, "snow_tau_w": None,
               "snow_tau_w_g": None, "snow_tau_w_f": None,
               "cld_tau": None, "cld_tau_w": None,
               "cld_tau_w_g": None, "cld_tau_w_f": None},
              outputs={"cldfprime": None, "c_cld_tau": None, "c_cld_tau_w": None,
                       "c_cld_tau_w_g": None, "c_cld_tau_w_f": None})
            log("rad_snow_blend_sw")

        if dolw:                                                     # 948
            self.ice_props_lw(st, lchnk); log("ice_cloud_get_rad_props_lw")
            self.liquid_props_lw(st, lchnk); log("liquid_cloud_get_rad_props_lw")
            # 968
            K("rad_combine_cld_optics_lw",
              {"ncol": ncol, "liq_lw_abs": None, "ice_lw_abs": None},
              outputs={"cld_lw_abs": None})
            log("rad_combine_cld_optics_lw")
            if has_snow:
                self.snow_props_lw(st, lchnk)
            log("snow_cloud_get_rad_props_lw")

        # 974-995: the LW blend, and the cldfprime default when there is no
        # snow field, which the driver writes outside the dolw branch
        K("rad_snow_blend_lw",
          {"ncol": ncol, "has_snow": int(has_snow), "cld": cld, "cldfsnow": cldfsnow,
           "snow_lw_abs": None, "cld_lw_abs": None},
          outputs={"cldfprime": None, "c_cld_lw_abs": None})
        log("rad_snow_blend_lw")

        # 1000: cgs pressures and the earth-sun distance factor
        K("rad_inp",
          {"ncol": ncol, "pmid": S["state_pmid"], "pint": S["state_pint"]},
          outputs={"pmidrd": None, "pintrd": None, "eccf": None})
        log("radinp")
        eccf = float(L["eccf"][()])

        # 1005-1012
        K("rad_interface_temperature",
          {"ncol": ncol, "t": S["state_t"], "lnpint": S["state_lnpint"],
           "lnpmid": S["state_lnpmid"], "lwup": cam_in["lwup"], "stebol": STEBOL},
          outputs={"tint": None})
        log("rad_interface_temperature")

        if dosw:                                                     # 1016
            K("get_variability", {}, outputs={"sfac": None}); log("get_variability")
            # 1026: only the climate call is active, asserted at attach
            H.rstate_update(lchnk, 0); log("rrtmg_state_update")
            self.aer_props_sw(st, lchnk, nnite, idxnite); log("aer_rad_props_sw")

            # 1034-1051: the shortwave core
            self._call_sw(st, lchnk, ncol, dt, eccf, nday, nnite, idxday, idxnite,
                          S, cam_in, cam_out, flux, qrs, coszrs)
            log(SW)

            # 1052-1055: FSNR is off, asserted at attach, so only the two
            # 200 mb interpolations run
            for source, target in (("fcns", "fsn200c"), ("fns", "fsn200")):
                K("vertinterp",
                  {"ncol": ncol, "ncold": pcols, "nlev": pverp,
                   "pmid": S["state_pint"], "pout": 20000.0, "arrin": L[source]},
                  outputs={"arrout": L[target]})
            log("vertinterp*")

            # 1057-1059
            K("rad_sw_cloud_forcing",
              {"ncol": ncol, "fsntoa": None, "fsntoac": None}, outputs={"swcf": None})
            log("rad_sw_cloud_forcing")

            # 1061-1090: the shortwave history block
            K("rad_scale_by_cpair", {"ncol": ncol, "field": qrs, "cpair": CPAIR},
              outputs={"ftem": None})
            log("rad_scale_by_cpair")
            H.outfld("QRS     ", L["ftem"], pcols, lchnk)
            K("rad_scale_by_cpair", {"ncol": ncol, "field": L["qrsc"], "cpair": CPAIR},
              outputs={"ftem": None})
            log("rad_scale_by_cpair")
            for name, value in (
                ("QRSC    ", L["ftem"]), ("SOLIN   ", L["solin"]),
                ("FSDS    ", flux["fsds"]), ("FSNIRTOA", L["fsnirt"]),
                ("FSNRTOAC", L["fsnrtc"]), ("FSNRTOAS", L["fsnirtsq"]),
                ("FSNT    ", flux["fsnt"]), ("FSNS    ", flux["fsns"]),
                ("FSNTC   ", L["fsntc"]), ("FSNSC   ", L["fsnsc"]),
                ("FSDSC   ", L["fsdsc"]), ("FSNTOA  ", L["fsntoa"]),
                ("FSUTOA  ", L["fsutoa"]), ("FSNTOAC ", L["fsntoac"]),
                ("SOLS    ", cam_out["sols"]), ("SOLL    ", cam_out["soll"]),
                ("SOLSD   ", cam_out["solsd"]), ("SOLLD   ", cam_out["solld"]),
                ("FSN200  ", L["fsn200"]), ("FSN200C ", L["fsn200c"]),
                ("SWCF    ", L["swcf"]), ("FSNR    ", L["fsnr"]),
            ):
                H.outfld(name, value, pcols, lchnk)
            log("outfld*")

            # 1092-1110
            K("rad_visible_tau",
              {"ncol": ncol, "Nnite": nnite, "IdxNite": idxnite,
               "has_snow": int(has_snow), "idx_sw_diag": IDX_SW_DIAG,
               "fillvalue": FILLVALUE, "c_cld_tau": None, "liq_tau": None,
               "ice_tau": None, "snow_tau": None, "cldfprime": None},
              outputs={"tot_cld_vistau": None, "tot_icld_vistau": None,
                       "liq_icld_vistau": None, "ice_icld_vistau": None,
                       "snow_icld_vistau": None})
            log("rad_visible_tau")
            for name, key in (("TOT_CLD_VISTAU", "tot_cld_vistau"),
                              ("TOT_ICLD_VISTAU", "tot_icld_vistau"),
                              ("LIQ_ICLD_VISTAU", "liq_icld_vistau"),
                              ("ICE_ICLD_VISTAU", "ice_icld_vistau")):
                H.outfld(f"{name:8s}"[:16], L[key], pcols, lchnk)
            if has_snow:
                H.outfld("SNOW_ICLD_VISTAU", L["snow_icld_vistau"], pcols, lchnk)
            log("outfld*")

        # 1122: aerosol mixing ratios, outside the dosw branch
        H.cnst_out(lchnk, 0); log("rad_cnst_out")

        if dolw:                                                     # 1126
            # 1130-1134
            K("rad_lwup_cgs",
              {"ncol": ncol, "lwup": cam_in["lwup"], "refused_scm": 0},
              outputs={"lwupcgs": None})
            log("rad_lwup_cgs")
            H.rstate_update(lchnk, 0); log("rrtmg_state_update")
            self.aer_props_lw(st, lchnk); log("aer_rad_props_lw")

            # 1148-1154: the longwave core
            self._call_lw(st, lchnk, ncol, dt, S, cam_out, flux, qrl, cld)
            log(LW)

            # 1156-1158
            K("rad_lw_cloud_forcing",
              {"ncol": ncol, "flutc": None, "flut": None}, outputs={"lwcf": None})
            log("rad_lw_cloud_forcing")

            # 1160-1161: FLNR is off, asserted at attach
            for source, target in (("fnl", "fln200"), ("fcnl", "fln200c")):
                K("vertinterp",
                  {"ncol": ncol, "ncold": pcols, "nlev": pverp,
                   "pmid": S["state_pint"], "pout": 20000.0, "arrin": L[source]},
                  outputs={"arrout": L[target]})
            log("vertinterp*")

            # 1170-1171: the division and the (:ncol,:) shape are one
            # expression outfld is given with idim = ncol, so the handles
            # module keeps the whole line rather than splitting it
            H.outfld_scaled(lchnk, ncol, "QRL     ", qrl, CPAIR); log("outfld_scaled")
            H.outfld_scaled(lchnk, ncol, "QRLC    ", L["qrlc"], CPAIR); log("outfld_scaled")
            for name, value in (
                ("FLNT    ", flux["flnt"]), ("FLUT    ", L["flut"]),
                ("FLUTC   ", L["flutc"]), ("FLNTC   ", L["flntc"]),
                ("FLNS    ", flux["flns"]), ("FLDSC   ", L["fldsc"]),
                ("FLNSC   ", L["flnsc"]), ("LWCF    ", L["lwcf"]),
                ("FLN200  ", L["fln200"]), ("FLN200C ", L["fln200c"]),
                ("FLDS    ", cam_out["flwds"]), ("FLNR    ", L["flnr"]),
            ):
                H.outfld(name, value, pcols, lchnk)
            log("outfld*")

        # 1192
        H.rstate_destroy(lchnk); log("rrtmg_state_destroy")

        # 1241-1243
        K("rad_emissivity",
          {"ncol": ncol, "rrtmg_lw_cloudsim_band": RRTMG_LW_CLOUDSIM_BAND,
           "cld_lw_abs": None}, outputs={"emis": None})
        log("rad_emissivity")
        H.outfld("EMIS    ", L["emis"], pcols, lchnk); log("outfld")

        # 1246-1257: computed for COSP, which is refused, but the driver
        # writes these unconditionally and a gate notices a missing write
        K("rad_snow_gridbox",
          {"ncol": ncol, "has_snow": int(has_snow),
           "rrtmg_sw_cloudsim_band": RRTMG_SW_CLOUDSIM_BAND,
           "rrtmg_lw_cloudsim_band": RRTMG_LW_CLOUDSIM_BAND,
           "cldfsnow": cldfsnow, "snow_tau": None, "snow_lw_abs": None},
          outputs={"gb_snow_tau": None, "gb_snow_lw": None})
        log("rad_snow_gridbox")

    # -- the two cores, assembled ------------------------------------------------

    def _call_sw(self, st, lchnk, ncol, dt, eccf, nday, nnite, idxday, idxnite,
                 S, cam_in, cam_out, flux, qrs, coszrs) -> None:
        """``radiation.F90:1034-1051``, argument for argument.

        ``scalars`` and ``arrays`` are the wrapper's dummy list in order;
        ``inputs`` and ``outputs`` are the same values by the name a model
        would see, which is also what the trace records.
        """

        L = st.local
        scalars = (ncol, st.constants.num_rrtmg_levs, nday, nnite, eccf)
        arrays = [
            idxday, idxnite,
            S["state_pmid"], L["cldfprime"],
            L["aer_tau"], L["aer_tau_w"], L["aer_tau_w_g"], L["aer_tau_w_f"],
            coszrs, cam_in["asdir"], cam_in["asdif"], cam_in["aldir"], cam_in["aldif"],
            L["sfac"],
            L["c_cld_tau"], L["c_cld_tau_w"], L["c_cld_tau_w_g"], L["c_cld_tau_w_f"],
            L["solin"], qrs, L["qrsc"], flux["fsnt"], L["fsntc"], L["fsntoa"],
            L["fsutoa"], L["fsntoac"], L["fsnirt"], L["fsnrtc"], L["fsnirtsq"],
            flux["fsns"], L["fsnsc"], L["fsdsc"], flux["fsds"],
            cam_out["sols"], cam_out["soll"], cam_out["solsd"], cam_out["solld"],
            L["fns"], L["fcns"],
        ]
        inputs = {
            "ncol": ncol, "rrtmg_levs": st.constants.num_rrtmg_levs,
            "Nday": nday, "Nnite": nnite, "IdxDay": idxday, "IdxNite": idxnite,
            "pmid": S["state_pmid"], "cld": L["cldfprime"], "eccf": eccf,
            "aer_tau": L["aer_tau"], "aer_tau_w": L["aer_tau_w"],
            "aer_tau_w_g": L["aer_tau_w_g"], "aer_tau_w_f": L["aer_tau_w_f"],
            "coszrs": coszrs, "asdir": cam_in["asdir"], "asdif": cam_in["asdif"],
            "aldir": cam_in["aldir"], "aldif": cam_in["aldif"], "sfac": L["sfac"],
            "cld_tau": L["c_cld_tau"], "cld_tau_w": L["c_cld_tau_w"],
            "cld_tau_w_g": L["c_cld_tau_w_g"], "cld_tau_w_f": L["c_cld_tau_w_f"],
            "rstate_pmidmb": st.handles.view(lchnk, VIEW["rstate_pmidmb"]),
            "rstate_pintmb": st.handles.view(lchnk, VIEW["rstate_pintmb"]),
            "rstate_tlay": st.handles.view(lchnk, VIEW["rstate_tlay"]),
            "rstate_tlev": st.handles.view(lchnk, VIEW["rstate_tlev"]),
            "rstate_h2ovmr": st.handles.view(lchnk, VIEW["rstate_h2ovmr"]),
            "rstate_o3vmr": st.handles.view(lchnk, VIEW["rstate_o3vmr"]),
            "rstate_co2vmr": st.handles.view(lchnk, VIEW["rstate_co2vmr"]),
        }
        outputs = {
            "solin": L["solin"], "qrs": qrs, "qrsc": L["qrsc"],
            "fsnt": flux["fsnt"], "fsntc": L["fsntc"], "fsntoa": L["fsntoa"],
            "fsutoa": L["fsutoa"], "fsntoac": L["fsntoac"], "fsnirtoa": L["fsnirt"],
            "fsnrtoac": L["fsnrtc"], "fsnrtoaq": L["fsnirtsq"], "fsns": flux["fsns"],
            "fsnsc": L["fsnsc"], "fsdsc": L["fsdsc"], "fsds": flux["fsds"],
            "sols": cam_out["sols"], "soll": cam_out["soll"],
            "solsd": cam_out["solsd"], "solld": cam_out["solld"],
            "fns": L["fns"], "fcns": L["fcns"],
        }
        self.rad_rrtmg_sw(st, lchnk, ncol, dt, inputs, outputs, scalars, arrays)

    def _call_lw(self, st, lchnk, ncol, dt, S, cam_out, flux, qrl, cld) -> None:
        """``radiation.F90:1148-1154``, argument for argument."""

        L = st.local
        scalars = (ncol, st.constants.num_rrtmg_levs)
        arrays = [
            S["state_pmid"], L["aer_lw_abs"], L["cldfprime"], L["c_cld_lw_abs"],
            qrl, L["qrlc"],
            flux["flns"], flux["flnt"], L["flnsc"], L["flntc"], cam_out["flwds"],
            L["flut"], L["flutc"], L["fnl"], L["fcnl"], L["fldsc"],
        ]
        inputs = {
            "ncol": ncol, "rrtmg_levs": st.constants.num_rrtmg_levs,
            "pmid": S["state_pmid"], "aer_lw_abs": L["aer_lw_abs"],
            "cld": L["cldfprime"], "tauc_lw": L["c_cld_lw_abs"],
            "rstate_pmidmb": st.handles.view(lchnk, VIEW["rstate_pmidmb"]),
            "rstate_pintmb": st.handles.view(lchnk, VIEW["rstate_pintmb"]),
            "rstate_tlay": st.handles.view(lchnk, VIEW["rstate_tlay"]),
            "rstate_tlev": st.handles.view(lchnk, VIEW["rstate_tlev"]),
            "rstate_h2ovmr": st.handles.view(lchnk, VIEW["rstate_h2ovmr"]),
            "rstate_o3vmr": st.handles.view(lchnk, VIEW["rstate_o3vmr"]),
            "rstate_co2vmr": st.handles.view(lchnk, VIEW["rstate_co2vmr"]),
        }
        outputs = {
            "qrl": qrl, "qrlc": L["qrlc"],
            "flns": flux["flns"], "flnt": flux["flnt"], "flnsc": L["flnsc"],
            "flntc": L["flntc"], "flwds": cam_out["flwds"], "flut": L["flut"],
            "flutc": L["flutc"], "fnl": L["fnl"], "fcnl": L["fcnl"],
            "fldsc": L["fldsc"],
        }
        self.rad_rrtmg_lw(st, lchnk, ncol, dt, inputs, outputs, scalars, arrays)


__all__ = ["FIRST_HALF", "KERNELS", "Radiation", "SECOND_HALF", "SEQUENCE_QUIET_STEP",
           "SEQUENCE_RADIATION_STEP", "STAGE", "SW", "LW", "VIEW"]

#: The stage's place in the workflow, for callers that ask before constructing one.
STAGE = Radiation.STAGE
FIRST_HALF = Radiation.FIRST_HALF
SECOND_HALF = Radiation.SECOND_HALF
