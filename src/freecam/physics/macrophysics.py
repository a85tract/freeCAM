"""CAM5's Park macrophysics as one Python class, in two hosts.

``Macrophysics`` is the macrophysics stage of the PI-atm workflow as its
supervisor asked for it: a Python object whose numerical kernels are methods
that call the original Fortran, and whose control flow -- fetching inputs,
sequencing, bookkeeping, history output -- is Python.  It works standalone
(:meth:`mmacro_pcond` runs the kernel from the reviewed standalone image with
no model at all) and inside the driver (:meth:`attach` puts :meth:`tend`
between the two halves of the macrophysics stage, where it reproduces
``macrop_driver_tend`` statement for statement).

The rule that makes the in-model path testable is that **Python computes no
floating-point number**.  Every arithmetic statement of the Fortran driver
is one of the seven routines lifted into ``pycam_macro_kernels``; every
derived-type call goes through ``pycam_macro_handles``; every buffer field
is CAM's own storage reached through ``pycam_pbuf_field_v1``.  Python
orders the calls and passes addresses.  A run driven this way is therefore
expected to be bit-for-bit with the oracle, and Gate B2 asserts it.

Two single IEEE divisions and one two-term sum are the exceptions, each
marked ``[exact]`` where it occurs with the reason it cannot differ.

Swapping the kernel for a model is then one assignment: ``scheme.kernel``.

What is not about macrophysics -- binding the handles module, CAM's host
services, the kernel-argument audit, the scratch arrays and the exact
copies into and out of them, and installing a stage between two halves --
lives in :mod:`freecam.physics.stage`.  This module is the stage's own
part: its tables, its module constants, its two extra handle calls, and
the transliteration itself.
"""


from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..pi_cam.errors import PICAMConfigurationError
from ..pi_cam.pbuf import PBuf, MACROP_FIELDS, macrop_fields
from .errors import PhysicsError
from .image import module_view
from .spec import load_function_spec
from .stage import (
    HOST_ENTRIES,
    HostEntries,
    HostServices,
    NativeStage,
    StageRuntime,
    check as _check,
    fortran as _f,
    pointer_of as _ptr,
)

REPO = Path(__file__).resolve().parents[3]

REPO = Path(__file__).resolve().parents[3]

FUNCTION = "mmacro_pcond"

# water_types.F90: named constants, so not symbols in the image.  A test pins
# them against the pinned source.
IWTVAP, IWTLIQ, IWTICE, PWTYPE = 1, 2, 3, 7
WTRC_MAX_CNST = 700

# physconst.F90 declares these three as parameters of shr_const_mod, so the
# image has no symbol for them either.  The decimal literals below are the
# source's; Python and the Fortran compiler round the same literal to the
# same double.  A test pins them against the pinned shr_const_mod.F90.
CPAIR, LATICE, LATVAP = 1.00464e3, 3.337e5, 2.501e6

# pycam_macro_handles.F90 view codes.  A test keeps this table equal to the
# Fortran one.
VIEW = {
    "state_t": 1, "state_q": 2, "state_pmid": 3, "state_pdel": 4, "state_pint": 5,
    "state_omega": 6, "state_phis": 7,
    "ptend_loc_s": 11, "ptend_loc_q": 12, "ptend_s": 21, "ptend_q": 22,
    "det_s": 31, "det_ice": 32, "process_rates": 33,
}
RECORD_PTEND_LOC, RECORD_PTEND = 1, 2

# physpkg's pycam_macro_forcing_v1 codes: the driver's forcing arguments.
FORCING = {"zdu": 1, "cmfmc": 2, "cmfmc2": 3, "dlf": 4, "dlf2": 5, "rliq": 6, "wtdlf": 7}

#: cldfrc_fice is promoted from the catalog, and a catalog-promoted descriptor
#: carries no extents -- it sizes itself from the pool's contracts.  These are
#: the routine's declared shapes (cloud_fraction.F90: t, fice, fsnow are
#: (pcols,pver)), used only where a descriptor says nothing.
FALLBACK_EXTENTS = {
    "ncol": ("chunks",), "t": ("pcols", "pver", "chunks"),
    "fice": ("pcols", "pver", "chunks"), "fsnow": ("pcols", "pver", "chunks"),
}

#: The direct kernels tend() runs; every one must be in the image and in
#: the reviewed descriptors with the same field list.
KERNELS = (
    "macrop_detrain_partition", "macrop_clear_fraction", "macrop_advective_forcing",
    "macrop_kernel_to_ptend", "macrop_tracer_rate_split", "macrop_cloud_mixing_ratio",
    "macrop_save_equilibrium", "mmacro_pcond", "wtrc_init_rates", "wtrc_add_rates",
    "cloud_fraction_fice",
)

#: The order in which the Fortran driver calls things; ``tend`` follows it
#: and a test compares the two.  Names are the routine or handle entry.
SEQUENCE = (
    "physics_state_copy", "pbuf_get_field*", "physics_ptend_init:pcwdetrain",
    "macrop_detrain_partition", "outfld*", "physics_ptend_init:macrop",
    "physics_ptend_sum", "physics_update", "macrop_clear_fraction", "cldfrc",
    "cldfrc_fice", "physics_ptend_init:macro_park", "macrop_advective_forcing",
    "mmacro_pcond", "macrop_kernel_to_ptend", "wtrc_init_rates",
    "macrop_tracer_rate_split", "wtrc_add_rates*", "wtrc_apply_rates",
    "physics_ptend_sum", "physics_update", "outfld*", "macrop_cloud_mixing_ratio",
    "outfld*", "macrop_save_equilibrium", "outfld", "physics_state_dealloc",
)



# -- the image, seen through ctypes ---------------------------------------------


class _MacroEntries(HostEntries):
    """The shared handles-module entries, plus macrophysics' own three.

    ``cldfrc`` and ``wtrc_apply_rates`` take the physics buffer, so neither
    can be promoted as a direct kernel; each gets a wrapper in the handles
    module instead.  ``forcing`` reaches the convection tendencies that
    ``tphysbc`` holds in its own private buffers.
    """

    TABLE = {
        **HOST_ENTRIES,
        "forcing": ("pycam_{prefix}_forcing_v1",
                    [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64)], False),
        "wtrc_apply": ("pycam_{prefix}_wtrc_apply_v1",
                       [ctypes.c_int, ctypes.c_int, ctypes.c_double,
                        ctypes.POINTER(ctypes.c_double)], False),
        # cldfrc's 27 array pointers are passed positionally; ctypes is not
        # told the signature, so the wrapper's own dummy list is the contract.
        "cldfrc": ("pycam_{prefix}_cldfrc_v1", None, False),
    }


class _MacroHandles(HostServices):
    """CAM's host services, plus the three calls only this stage makes."""

    def forcing(self, lchnk: int, name: str) -> np.ndarray:
        """A zero-copy view of one of tphysbc's convection forcing buffers."""

        return self._deref(self.e.forcing, f"pycam_macro_forcing_v1({name}, chunk {lchnk})",
                           lchnk, FORCING[name], ndims_max=4)

    def wtrc_apply(self, lchnk: int, top_lev: int, dt: float, prelat: np.ndarray) -> None:
        _check(self.e.wtrc_apply(lchnk, top_lev, float(dt), _ptr(prelat)), "wtrc_apply_rates")

    def cldfrc(self, lchnk: int, ncol: int, arrays: Sequence[np.ndarray],
               use_shfrc: bool, dindex: int) -> None:
        pointers = [_ptr(a) for a in arrays]
        status = self.e.cldfrc(
            ctypes.c_int(lchnk), ctypes.c_int(ncol),
            *pointers[:6], ctypes.c_int(int(use_shfrc)), *pointers[6:], ctypes.c_int(dindex),
        )
        _check(status, "cldfrc")


# -- module constants --------------------------------------------------------------


@dataclass(frozen=True)
class _Constants:
    top_lev: int
    ixcldliq: int
    ixcldice: int
    ixnumliq: int
    ixnumice: int
    do_cldice: bool
    do_cldliq: bool
    do_detrain: bool
    micro_do_icesupersat: bool
    use_shfrc: bool
    cpair: float
    latice: float
    latvap: float
    gravit: float
    tmelt: float
    trace_water: bool
    wtrc_detrain_in_macrop: bool
    wtrc_nwset: int
    wtrc_ncnst: int
    wtrc_iatype: np.ndarray = field(repr=False)
    wtrc_indices: np.ndarray = field(repr=False)

    @classmethod
    def read(cls, library: Any) -> "_Constants":
        def i(symbol):
            return int(module_view(library, symbol, "int32", ()))

        def b(symbol):
            return bool(int(module_view(library, symbol, "int32", ())))

        def r(symbol):
            return float(module_view(library, symbol, "float64", ()))

        return cls(
            top_lev=i("ref_pres_mp_trop_cloud_top_lev_"),
            ixcldliq=i("macrop_driver_mp_ixcldliq_"), ixcldice=i("macrop_driver_mp_ixcldice_"),
            ixnumliq=i("macrop_driver_mp_ixnumliq_"), ixnumice=i("macrop_driver_mp_ixnumice_"),
            do_cldice=b("macrop_driver_mp_do_cldice_"), do_cldliq=b("macrop_driver_mp_do_cldliq_"),
            do_detrain=b("macrop_driver_mp_do_detrain_"),
            micro_do_icesupersat=b("macrop_driver_mp_micro_do_icesupersat_"),
            use_shfrc=b("macrop_driver_mp_use_shfrc_"),
            cpair=CPAIR, latice=LATICE, latvap=LATVAP,
            gravit=r("physconst_mp_gravit_"), tmelt=r("physconst_mp_tmelt_"),
            trace_water=b("water_tracer_vars_mp_trace_water_"),
            wtrc_detrain_in_macrop=b("water_tracer_vars_mp_wtrc_detrain_in_macrop_"),
            wtrc_nwset=i("water_tracer_vars_mp_wtrc_nwset_"),
            wtrc_ncnst=i("water_tracer_vars_mp_wtrc_ncnst_"),
            wtrc_iatype=np.array(module_view(library, "water_tracer_vars_mp_wtrc_iatype_", "int32",
                                             (WTRC_MAX_CNST, PWTYPE))),
            wtrc_indices=np.array(module_view(library, "water_tracer_vars_mp_wtrc_indices_", "int32",
                                              (WTRC_MAX_CNST,))),
        )

    def refuse_unsupported(self) -> None:
        """The paths the admitted configuration never takes are not ported."""

        if self.micro_do_icesupersat:
            raise PICAMConfigurationError(
                "micro_do_icesupersat is on; the Python macrophysics does not carry ice_macro_tend")


# -- the class ---------------------------------------------------------------------


class Macrophysics(NativeStage):
    """CAM5 Park macrophysics: cldfrc + detrainment + mmacro_pcond, as Python.

    ``kernel`` is what computes mmacro_pcond's 23 returned values inside the
    model: ``None`` runs the original through its direct kernel; a callable
    taking the batch mapping ``{name: (ncol, ...) array}`` and returning the
    23 by name -- a ``torch.nn.Module`` wrapped that way, say -- takes its
    place.  Nothing else in :meth:`tend_chunk` changes when it does.
    """

    STAGE = "cam_run1.cloud_macro_microphysics"
    FIRST_HALF = "cam_run1.macro_tend_pre_leaf"
    SECOND_HALF = "cam_run1.macro_tend_post_leaf"
    PREFIX = "macro"
    PROCESS_NAME = "macro_tend"
    TRACE_ENV = "FREECAM_MACRO_TRACE"

    KERNELS = KERNELS
    #: cldfrc_fice comes from the catalog, so its fields are named after
    #: StatePool entries rather than this stage's locals.
    UNSCRATCHED = ("cloud_fraction_fice",)
    FALLBACK_EXTENTS = FALLBACK_EXTENTS
    CAM_IN = ("landfrac", "ocnfrac", "snowhland", "ts", "sst")
    SWAPPABLE = (FUNCTION,)
    #: What the driver keeps that no kernel declares.
    EXTRA_SCRATCH = tuple(
        (name, ("pcols", "pver", "chunks")) for name in (
            "concld_old", "rhcloud", "cldst", "rhu00", "icecldf", "liqcldf",
            "relhum", "shfrc_zero")
    ) + (("clc", ("pcols", "chunks")),)

    entries_class = _MacroEntries
    services_class = _MacroHandles

    def __init__(self, *, kernel: Callable[..., Mapping[str, np.ndarray]] | None = None) -> None:
        super().__init__(kernel=kernel)
        self.spec = load_function_spec(FUNCTION)
        self._standalone: Any = None

    # -- standalone ------------------------------------------------------------

    def example_input(self, name: str = "captured-anchor"):
        return self._function().example_input(name)

    def mmacro_pcond(self, inputs: Mapping[str, Any], parameters: Mapping[str, Any] | None = None):
        """One column through the reviewed standalone image; no model needed."""

        return self._function().run(inputs, parameters)

    def _function(self):
        if self._standalone is None:
            from .function import load_function

            self._standalone = load_function(FUNCTION)
        return self._standalone

    def close(self) -> None:
        if self._standalone is not None:
            self._standalone.close()
            self._standalone = None

    # -- what the runtime asks of this stage -------------------------------------

    def read_constants(self, library: Any) -> "_Constants":
        return _Constants.read(library)

    def refuse_unsupported(self, constants: "_Constants") -> None:
        constants.refuse_unsupported()

    def extra_extents(self, constants: "_Constants") -> Mapping[str, int]:
        return {"pwtype": PWTYPE, "wtrc_nwset": constants.wtrc_nwset}

    def build_pbuf(self, library: Any, runtime: StageRuntime) -> PBuf:
        indices = {symbol: int(module_view(library, f"macrop_driver_mp_{symbol}_", "int32", ()))
                   for _, symbol, _ in MACROP_FIELDS}
        buffer = PBuf(library, macrop_fields(indices))
        lchnk, _ = runtime.native.chunks
        buffer.verify(int(lchnk[0]), pcols=runtime.pcols, pver=runtime.pver)
        return buffer

    def after_runtime(self, runtime: StageRuntime) -> None:
        # cldfrc_fice's promoted descriptor names its fields after StatePool
        # entries; the routine's dummies are ncol, t, fice, fsnow in that order.
        runtime.fice_fields = dict(zip(
            ("ncol", "t", "fice", "fsnow"),
            (a.field for a in runtime.descriptors["cloud_fraction_fice"].arguments)))

    # -- the transliteration -----------------------------------------------------

    def tend_chunk(self, st: StageRuntime, lchnk: int, ncol: int, index: int,
                   dt: float, nstep: int) -> None:
        """``macrop_driver_tend``, statement for statement, for one chunk."""

        H, C, pb = st.handles, st.constants, st.pbuf
        L = st.local

        def K(name, inputs, *, outputs, fields=None):
            st.kernel_on_chunk(name, inputs, outputs=outputs, fields=fields, ncol=ncol)
        top = C.top_lev
        pcols, pver, pverp = st.pcols, st.pver, st.pverp
        cols = slice(0, ncol)
        lev = slice(top - 1, pver)
        log = self.calls.append

        # 618-621: lchnk, ncol; copy the state into the driver's local
        H.state_copy(lchnk); log("physics_state_copy")
        S = {name: H.view(lchnk, code) for name, code in VIEW.items() if name.startswith("state_")}

        # 626-648: the physics-buffer fields, older time sample where the source says so
        pbv = {name: pb.view(name, lchnk) for name in (
            "QCWAT", "TCWAT", "LCWAT", "ICCWAT", "NLWAT", "NIWAT", "CC_T", "CC_qv", "CC_ql",
            "CC_qi", "CC_nl", "CC_ni", "CC_qlst", "CLD", "CONCLD", "AST", "AIST", "ALST",
            "QIST", "QLST", "CMELIQ", "FICE")}
        log("pbuf_get_field*")

        # 659-664: the detrainment forcings start at zero
        for name in ("dlf_T", "dlf_qv", "dlf_ql", "dlf_qi", "dlf_nl", "dlf_ni"):
            st.scratch[name][...] = 0.0

        # 666-688: the first tendency record, with the constituents it carries
        lq = np.zeros(st.pcnst, dtype=np.int32)
        for ix in (C.ixcldliq, C.ixcldice, C.ixnumliq, C.ixnumice):
            lq[ix - 1] = 1
        if C.trace_water and C.wtrc_detrain_in_macrop:
            for m in range(C.wtrc_nwset):
                lq[C.wtrc_iatype[m, IWTLIQ - 1] - 1] = 1
                lq[C.wtrc_iatype[m, IWTICE - 1] - 1] = 1
        H.ptend_init(lchnk, RECORD_PTEND_LOC, "pcwdetrain", ls=True, lq=lq); log("physics_ptend_init:pcwdetrain")
        pl_s, pl_q = H.view(lchnk, VIEW["ptend_loc_s"]), H.view(lchnk, VIEW["ptend_loc_q"])
        det_s, det_ice = H.view(lchnk, VIEW["det_s"]), H.view(lchnk, VIEW["det_ice"])
        forcing = {name: H.forcing(lchnk, name) for name in ("dlf", "dlf2", "cmfmc", "cmfmc2", "zdu", "wtdlf")}

        # 706-797: detrainment partition
        K("macrop_detrain_partition", {
            "ncol": ncol, "top_lev": top, "ixcldliq": C.ixcldliq, "ixcldice": C.ixcldice,
            "ixnumliq": C.ixnumliq, "ixnumice": C.ixnumice, "nwset": C.wtrc_nwset,
            "iatype_liq": C.wtrc_iatype[:C.wtrc_nwset, IWTLIQ - 1],
            "iatype_ice": C.wtrc_iatype[:C.wtrc_nwset, IWTICE - 1],
            "do_detrain": C.do_detrain, "cu_det_st": False,
            "do_wtrc_detrain": C.trace_water and C.wtrc_detrain_in_macrop,
            "gravit": C.gravit, "latice": C.latice, "cpair": C.cpair,
            "t": S["state_t"], "pdel": S["state_pdel"], "dlf": forcing["dlf"], "dlf2": forcing["dlf2"],
            "wtdlf": forcing["wtdlf"], "ptend_s": pl_s, "ptend_q": pl_q,
        }, outputs={"ptend_s": pl_s, "ptend_q": pl_q, "det_s": det_s, "det_ice": det_ice,
                    "dlf_T": None, "dlf_qv": None, "dlf_ql": None, "dlf_qi": None, "dlf_nl": None,
                    "dlf_ni": None, "dpdlfliq": None, "dpdlfice": None, "shdlfliq": None,
                    "shdlfice": None, "dpdlft": None, "shdlft": None})
        log("macrop_detrain_partition")

        # 799-806
        for name, key in (("DPDLFLIQ ", "dpdlfliq"), ("DPDLFICE ", "dpdlfice"), ("SHDLFLIQ ", "shdlfliq"),
                          ("SHDLFICE ", "shdlfice"), ("DPDLFT   ", "dpdlft"), ("SHDLFT   ", "shdlft")):
            H.outfld(name, L[key], pcols, lchnk)
        H.outfld("ZMDLF", forcing["dlf"], pcols, lchnk)
        log("outfld*")

        # 808  [exact] one IEEE division per element, correctly rounded in
        # both Fortran (-fp-model source forbids a reciprocal) and NumPy.
        det_ice[cols] = det_ice[cols] / 1000.0

        # 810-812
        H.ptend_init(lchnk, RECORD_PTEND, "macrop"); log("physics_ptend_init:macrop")
        H.ptend_sum(lchnk, ncol); log("physics_ptend_sum")
        H.update(lchnk, dt); log("physics_update")
        # 813-837: ice supersaturation -- refused at attach; nothing to do here

        # 841
        concld_old = L["concld_old"]
        concld_old[cols, lev] = pbv["CONCLD"][cols, lev]

        # 895-902
        K("macrop_clear_fraction", {"ncol": ncol, "top_lev": top, "concld": pbv["CONCLD"],
                                    "alst": pbv["ALST"], "ast": pbv["AST"]},
          outputs={"clrw_old": None, "clri_old": None})
        log("macrop_clear_fraction")

        # 904-909
        shfrc = pb.view("SHFRC", lchnk) if C.use_shfrc else L["shfrc_zero"]

        # 918-925
        cam_in = st.cam_in(index)
        H.cldfrc(lchnk, ncol, [
            S["state_pmid"], S["state_t"], _f(S["state_q"][:, :, 0]), S["state_omega"], S["state_phis"],
            _f(shfrc),
            pbv["CLD"], L["rhcloud"], L["clc"], S["state_pdel"],
            forcing["cmfmc"], forcing["cmfmc2"], cam_in["landfrac"], cam_in["snowhland"],
            pbv["CONCLD"], L["cldst"], cam_in["ts"], cam_in["sst"],
            _f(S["state_pint"][:, pverp - 1]), forcing["zdu"], cam_in["ocnfrac"], L["rhu00"],
            _f(S["state_q"][:, :, C.ixcldice - 1]), L["icecldf"], L["liqcldf"],
            L["relhum"],
        ], C.use_shfrc, 0)
        log("cldfrc")

        # 927-929  [exact] one IEEE division
        rdtime = 1.0 / dt

        # 930
        K("cloud_fraction_fice", {"ncol": ncol, "t": S["state_t"]}, outputs={"fice": None, "fsnow": None},
          fields=st.fice_fields)
        log("cldfrc_fice")

        # 932-938
        lq = np.zeros(st.pcnst, dtype=np.int32)
        lq[0] = 1
        for ix in (C.ixcldice, C.ixcldliq, C.ixnumliq, C.ixnumice):
            lq[ix - 1] = 1
        for m in range(C.wtrc_ncnst):
            lq[C.wtrc_indices[m] - 1] = 1
        H.ptend_init(lchnk, RECORD_PTEND_LOC, "macro_park", ls=True, lq=lq); log("physics_ptend_init:macro_park")
        pl_s, pl_q = H.view(lchnk, VIEW["ptend_loc_s"]), H.view(lchnk, VIEW["ptend_loc_q"])

        # 966-970: copies
        q = S["state_q"]
        L["qc"][cols, lev] = q[cols, lev, C.ixcldliq - 1]
        L["qi"][cols, lev] = q[cols, lev, C.ixcldice - 1]
        L["nc"][cols, lev] = q[cols, lev, C.ixnumliq - 1]
        L["ni"][cols, lev] = q[cols, lev, C.ixnumice - 1]

        # 978-1019
        wat = {n: pbv[n.upper()] for n in ("tcwat", "qcwat", "lcwat", "iccwat", "nlwat", "niwat")}
        cc = {n: pbv[n] for n in ("CC_T", "CC_qv", "CC_ql", "CC_qi", "CC_nl", "CC_ni", "CC_qlst")}
        K("macrop_advective_forcing", {
            "ncol": ncol, "top_lev": top, "nstep": nstep, "rdtime": rdtime,
            "t": S["state_t"], "q": q, "qc": None, "qi": None, "nc": None, "ni": None, **cc, **wat,
        }, outputs={**{n: v for n, v in cc.items()}, **{n: v for n, v in wat.items()},
                    "ttend": None, "qtend": None, "ltend": None, "itend": None, "nltend": None,
                    "nitend": None, "lmitend": None, "t_inout": None, "qv_inout": None,
                    "ql_inout": None, "qi_inout": None, "nl_inout": None, "ni_inout": None})
        log("macrop_advective_forcing")

        # 1028-1037: the kernel -- the original, or whatever `kernel` is
        self._kernel_call(st, lchnk, ncol, dt, pbv, forcing, cam_in, index)
        log("mmacro_pcond")

        # 1042-1046: copies and one maximum [exact: selection, no rounding]
        fice_ql = pbv["FICE"]
        fice_ql[cols, :top - 1] = 0.0
        fice_ql[cols, lev] = L["fice"][cols, lev]
        ast = pbv["AST"]
        ast[cols, :top - 1] = 0.0
        ast[cols, lev] = np.maximum(pbv["ALST"][cols, lev], pbv["AIST"][cols, lev])

        # 1051-1080
        status = K("macrop_kernel_to_ptend", {
            "ncol": ncol, "top_lev": top, "ixcldliq": C.ixcldliq, "ixcldice": C.ixcldice,
            "ixnumliq": C.ixnumliq, "ixnumice": C.ixnumice, "do_cldice": C.do_cldice,
            "do_cldliq": C.do_cldliq, "tlat": None, "qvlat": None, "qcten": None, "qiten": None,
            "ncten": None, "niten": None, "ptend_s": pl_s, "ptend_q": pl_q,
        }, outputs={"ptend_s": pl_s, "ptend_q": pl_q, "status": None})
        log("macrop_kernel_to_ptend")
        code = int(L["status"])
        if code != 0:
            raise PhysicsError(
                "macrop_driver:ERROR - Cldwat is configured not to prognose a species, "
                f"but mmacro_pcond returned tendencies for it (code {code})")

        # 1100-1133
        if C.trace_water:
            rates = H.view(lchnk, VIEW["process_rates"])
            K("wtrc_init_rates", {"top_lev": top}, outputs={"process_rates": rates}); log("wtrc_init_rates")
            K("macrop_tracer_rate_split", {"ncol": ncol, "top_lev": top, "qcten": None, "qiten": None},
              outputs={"pqctn": None, "nqctn": None, "pqitn": None, "nqitn": None})
            log("macrop_tracer_rate_split")
            s = L
            # [exact] two IEEE additions, left to right in both languages
            vapour = (s["qvlat"] + s["qcten"]) + s["qiten"]
            for src, dst, rtype, rate in ((IWTVAP, IWTVAP, IWTVAP, vapour), (IWTVAP, IWTLIQ, IWTVAP, s["pqctn"]),
                                          (IWTVAP, IWTLIQ, IWTLIQ, s["nqctn"]), (IWTVAP, IWTICE, IWTVAP, s["pqitn"]),
                                          (IWTVAP, IWTICE, IWTICE, s["nqitn"])):
                K("wtrc_add_rates", {"process_rates": rates, "ncol": ncol, "top_lev": top, "isrctype": src,
                                     "idsttype": dst, "rtype": rtype, "rate": rate},
                  outputs={"process_rates": rates})
            log("wtrc_add_rates*")
            H.wtrc_apply(lchnk, top, dt, _f(s["tlat"])); log("wtrc_apply_rates")

        # 1136-1137
        H.ptend_sum(lchnk, ncol); log("physics_ptend_sum")
        H.update(lchnk, dt); log("physics_update")

        # 1141-1160
        s = L
        for name, array in (("CLR_LIQ", s["clrw_old"]), ("CLR_ICE", s["clri_old"]),
                            ("MACPDT ", s["tlat"]), ("MACPDQ ", s["qvlat"]), ("MACPDLIQ ", s["qcten"]),
                            ("MACPDICE ", s["qiten"]), ("CLDVAPADJ", s["qvadj"]), ("CLDLIQADJ", s["qladj"]),
                            ("CLDICEADJ", s["qiadj"]), ("CLDLIQDET", s["dlf_ql"]), ("CLDICEDET", s["dlf_qi"]),
                            ("CLDLIQLIM", s["qllim"]), ("CLDICELIM", s["qilim"]), ("ICECLDF ", pbv["AIST"]),
                            ("LIQCLDF ", pbv["ALST"]), ("AST", ast), ("CONCLD ", pbv["CONCLD"]),
                            ("CLDST ", s["cldst"]), ("CMELIQ", pbv["CMELIQ"])):
            H.outfld(name, array, pcols, lchnk)
        log("outfld*")

        # 1182-1197
        K("macrop_cloud_mixing_ratio", {"ncol": ncol, "top_lev": top, "ixcldliq": C.ixcldliq,
                                        "ixcldice": C.ixcldice, "q": q, "cld": pbv["CLD"]},
          outputs={"mr_ccliq": None, "mr_ccice": None, "mr_lsliq": None, "mr_lsice": None})
        log("macrop_cloud_mixing_ratio")
        for name, key in (("CLDLIQSTR ", "mr_lsliq"), ("CLDICESTR ", "mr_lsice"),
                          ("CLDLIQCON ", "mr_ccliq"), ("CLDICECON ", "mr_ccice")):
            H.outfld(name, s[key], pcols, lchnk)
        log("outfld*")

        # 1208-1218
        K("macrop_save_equilibrium", {"ncol": ncol, "top_lev": top, "ixcldliq": C.ixcldliq,
                                      "ixcldice": C.ixcldice, "ixnumliq": C.ixnumliq, "ixnumice": C.ixnumice,
                                      "tmelt": C.tmelt, "t": S["state_t"], "q": q},
          outputs={**wat, "cldsice": None})
        log("macrop_save_equilibrium")
        H.outfld("CLDSICE", s["cldsice"], pcols, lchnk); log("outfld")

        # 1222
        H.state_dealloc(lchnk); log("physics_state_dealloc")


    def _kernel_call(self, st, lchnk, ncol, dt, pbv, forcing, cam_in, index) -> None:
        C, s = st.constants, st.local
        S = st.handles

        state_pmid = S.view(lchnk, VIEW["state_pmid"])
        state_pdel = S.view(lchnk, VIEW["state_pdel"])
        inputs = {
            "lchnk": lchnk, "ncol": ncol, "dt": dt, "p": state_pmid, "dp": state_pdel,
            "t0": None, "qv0": None, "ql0": None, "qi0": None, "nl0": None, "ni0": None,
            "a_t": None, "a_qv": None, "a_ql": None, "a_qi": None, "a_nl": None, "a_ni": None,
            "c_t": pbv["CC_T"], "c_qv": pbv["CC_qv"], "c_ql": pbv["CC_ql"], "c_qi": pbv["CC_qi"],
            "c_nl": pbv["CC_nl"], "c_ni": pbv["CC_ni"], "c_qlst": pbv["CC_qlst"],
            "d_t": None, "d_qv": None, "d_ql": None, "d_qi": None, "d_nl": None, "d_ni": None,
            "a_cud": None, "a_cu0": pbv["CONCLD"], "clrw_old": None, "clri_old": None,
            "landfrac": cam_in["landfrac"], "snowh": cam_in["snowhland"],
            "tke": None, "qtl_flx": None, "qti_flx": None, "cmfr_det": None, "qlr_det": None, "qir_det": None,
            "do_cldice": C.do_cldice,
        }
        # the driver's names for the kernel's: the same mapping the capture hook used
        alias = {"t0": "t_inout", "qv0": "qv_inout", "ql0": "ql_inout", "qi0": "qi_inout",
                 "nl0": "nl_inout", "ni0": "ni_inout", "a_t": "ttend", "a_qv": "qtend", "a_ql": "lmitend",
                 "a_qi": "itend", "a_nl": "nltend", "a_ni": "nitend", "d_t": "dlf_T", "d_qv": "dlf_qv",
                 "d_ql": "dlf_ql", "d_qi": "dlf_qi", "d_nl": "dlf_nl", "d_ni": "dlf_ni",
                 "a_cud": "concld_old", "clrw_old": "clrw_old", "clri_old": "clri_old",
                 "tke": "tke", "qtl_flx": "qtl_flx", "qti_flx": "qti_flx", "cmfr_det": "cmfr_det",
                 "qlr_det": "qlr_det", "qir_det": "qir_det"}
        for name, source in alias.items():
            if inputs[name] is None:
                inputs[name] = s[source]
        outputs = {"s_tendout": s["tlat"], "qv_tendout": s["qvlat"], "ql_tendout": s["qcten"],
                   "qi_tendout": s["qiten"], "nl_tendout": s["ncten"], "ni_tendout": s["niten"],
                   "qme": pbv["CMELIQ"], "qvadj": s["qvadj"], "qladj": s["qladj"], "qiadj": s["qiadj"],
                   "qllim": s["qllim"], "qilim": s["qilim"], "cld": pbv["CLD"], "al_st_star": pbv["ALST"],
                   "ai_st_star": pbv["AIST"], "ql_st_star": pbv["QLST"], "qi_st_star": pbv["QIST"],
                   "t0": s["t_inout"], "qv0": s["qv_inout"], "ql0": s["ql_inout"], "qi0": s["qi_inout"],
                   "nl0": s["nl_inout"], "ni0": s["ni_inout"]}
        # The one kernel a model may replace.  The runtime runs the
        # original or the model, and traces both the same way.
        st.swappable_kernel(FUNCTION, inputs, outputs=outputs, ncol=ncol,
                            lchnk=lchnk, dt=dt)
#: The stage's place in the workflow, for callers that ask before constructing one.
STAGE = Macrophysics.STAGE
FIRST_HALF = Macrophysics.FIRST_HALF
SECOND_HALF = Macrophysics.SECOND_HALF

__all__ = ["FIRST_HALF", "FORCING", "KERNELS", "Macrophysics", "SECOND_HALF", "SEQUENCE",
           "STAGE", "VIEW"]
