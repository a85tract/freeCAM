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
RRTMG_SW_CLOUDSIM_BAND, RRTMG_LW_CLOUDSIM_BAND = 9, 7
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
    "calday", "pbuf_get_field*", "outfld", "latlon", "zenith",
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
    "rrtmg_state_destroy", "rad_emissivity", "outfld", "rad_snow_gridbox",
    "rad_data_write", "radheat_tend", "rad_theta_heating", "outfld",
    "rad_heating_scale",
)

#: What the driver does on a step where neither shortwave nor longwave runs.
SEQUENCE_QUIET_STEP = (
    "calday", "pbuf_get_field*", "outfld", "latlon", "zenith",
    "rad_gather_day_night", "radiation_do:sw", "radiation_do:lw",
    "rad_heating_unscale",
    "rad_data_write", "radheat_tend", "rad_theta_heating", "outfld",
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
        ("troplev_r", ("pcols", "chunks")), ("p_trop", ("pcols", "chunks")),
        ("pbr", ("pcols", "pver", "chunks")), ("pnm", ("pcols", "pverp", "chunks")),
        ("liq_lw_abs", ("nbndlw", "pcols", "pver", "chunks")),
        ("ice_lw_abs", ("nbndlw", "pcols", "pver", "chunks")),
        ("snow_lw_abs", ("nbndlw", "pcols", "pver", "chunks")),
        ("liq_tau_w", ("nbndsw", "pcols", "pver", "chunks")),
        ("aer_tau", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_tau_w", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_tau_w_g", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_tau_w_f", ("pcols", "pverp", "nbndsw", "chunks")),
        ("aer_lw_abs", ("pcols", "pver", "nbndlw", "chunks")),
        ("sfac", ("nbndsw", "chunks")),
        ("qrsc", ("pcols", "pver", "chunks")), ("qrlc", ("pcols", "pver", "chunks")),
        ("fns", ("pcols", "pverp", "chunks")), ("fcns", ("pcols", "pverp", "chunks")),
        ("fnl", ("pcols", "pverp", "chunks")), ("fcnl", ("pcols", "pverp", "chunks")),
    ) + tuple(
        (name, ("pcols", "chunks")) for name in (
            "solin", "fsntoa", "fsutoa", "fsntoac", "fsnirt", "fsnrtc", "fsnirtsq",
            "fsntc", "fsnsc", "fsdsc", "flut", "flutc", "flntc", "flnsc", "fldsc",
            "fsn200", "fsn200c", "fln200", "fln200c", "fsnr", "flnr",
            "lwupcgs", "eccf_out",
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
        return {"nbndsw": NBNDSW, "nbndlw": NBNDLW,
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
