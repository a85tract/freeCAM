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
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..pi_cam.errors import PICAMConfigurationError
from ..pi_cam.facade import Physics
from ..pi_cam.pbuf import PBuf, macrop_fields
from .errors import PhysicsError
from .image import module_view
from .spec import load_function_spec

FUNCTION = "mmacro_pcond"
STAGE = "cam_run1.cloud_macro_microphysics"
FIRST_HALF = "cam_run1.macro_tend_pre_leaf"
SECOND_HALF = "cam_run1.macro_tend_post_leaf"

# water_types.F90: named constants, so not symbols in the image.  A test pins
# them against the pinned source.
IWTVAP, IWTLIQ, IWTICE, PWTYPE = 1, 2, 3, 7
WTRC_MAX_CNST = 700

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


class _Entries:
    """The bind(C) entries of the handles module and the boundary, bound once."""

    def __init__(self, library: Any) -> None:
        self.library = library
        c = ctypes
        self.set_owner = self._bind("pycam_macro_set_owner_v1", [c.c_int])
        self.bind_hosts = self._bind("pycam_macro_bind_hosts_v1", [])
        self.state_copy = self._bind("pycam_macro_state_copy_v1", [c.c_int])
        self.state_dealloc = self._bind("pycam_macro_state_dealloc_v1", [c.c_int])
        self.ptend_init = self._bind("pycam_macro_ptend_init_v1", [
            c.c_int, c.c_int, c.c_char_p, c.c_int, c.c_int, c.c_int, c.POINTER(c.c_int32)])
        self.ptend_sum = self._bind("pycam_macro_ptend_sum_v1", [c.c_int, c.c_int])
        self.update = self._bind("pycam_macro_update_v1", [c.c_int, c.c_double])
        self.view = self._bind("pycam_macro_view_v1", [
            c.c_int, c.c_int, c.POINTER(c.c_void_p), c.POINTER(c.c_int), c.POINTER(c.c_int64)])
        self.forcing = self._bind("pycam_macro_forcing_v1", [
            c.c_int, c.c_int, c.POINTER(c.c_void_p), c.POINTER(c.c_int), c.POINTER(c.c_int64)])
        self.wtrc_apply = self._bind("pycam_macro_wtrc_apply_v1", [
            c.c_int, c.c_int, c.c_double, c.POINTER(c.c_double)])
        self.outfld = self._bind("pycam_outfld_v1", [
            c.c_char_p, c.c_int, c.POINTER(c.c_double), c.c_int, c.c_int])
        self.cldfrc = self._bind("pycam_macro_cldfrc_v1", None)

    def _bind(self, name: str, argtypes: list | None):
        try:
            function = getattr(self.library, name)
        except AttributeError as error:
            raise PICAMConfigurationError(
                f"the image exposes no {name}; it predates the macrophysics boundary"
            ) from error
        function.restype = ctypes.c_int
        if argtypes is not None:
            function.argtypes = argtypes
        return function


def _check(status: int, what: str) -> None:
    if status != 0:
        raise PICAMConfigurationError(f"{what} refused with status {status}")


def _as_view(pointer: ctypes.c_void_p, ndims: int, extents: Sequence[int]) -> np.ndarray:
    shape = tuple(int(extents[i]) for i in range(ndims))
    count = int(np.prod(shape)) if shape else 1
    buffer = (ctypes.c_double * count).from_address(pointer.value)
    return np.ndarray(shape, dtype=np.float64, buffer=buffer, order="F")


def _f(array: np.ndarray) -> np.ndarray:
    """An F-contiguous float64 array, itself if already so."""

    return np.asfortranarray(array, dtype=np.float64)


def _ptr(array: np.ndarray):
    assert array.flags.f_contiguous and array.dtype == np.float64
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class _Handles:
    """One chunk's worth of the handles module, as NumPy views."""

    def __init__(self, entries: _Entries, pcnst: int) -> None:
        self.e = entries
        self.pcnst = pcnst

    def view(self, lchnk: int, code: int) -> np.ndarray:
        pointer = ctypes.c_void_p()
        ndims = ctypes.c_int()
        extents = (ctypes.c_int64 * 5)()
        _check(self.e.view(lchnk, code, ctypes.byref(pointer), ctypes.byref(ndims), extents),
               f"pycam_macro_view_v1(chunk {lchnk}, code {code})")
        return _as_view(pointer, ndims.value, extents)

    def forcing(self, lchnk: int, name: str) -> np.ndarray:
        pointer = ctypes.c_void_p()
        ndims = ctypes.c_int()
        extents = (ctypes.c_int64 * 4)()
        _check(self.e.forcing(lchnk, FORCING[name], ctypes.byref(pointer), ctypes.byref(ndims), extents),
               f"pycam_macro_forcing_v1({name}, chunk {lchnk})")
        return _as_view(pointer, ndims.value, extents)

    def state_copy(self, lchnk: int) -> None:
        _check(self.e.state_copy(lchnk), "physics_state_copy")

    def state_dealloc(self, lchnk: int) -> None:
        _check(self.e.state_dealloc(lchnk), "physics_state_dealloc")

    def ptend_init(self, lchnk: int, which: int, name: str, *, ls: bool | None = None,
                   lq: np.ndarray | None = None) -> None:
        with_flags = lq is not None
        flags = np.zeros(self.pcnst, dtype=np.int32) if lq is None else np.asarray(lq, dtype=np.int32)
        assert flags.shape == (self.pcnst,)
        _check(self.e.ptend_init(
            lchnk, which, name.encode("ascii"), len(name), int(with_flags),
            int(bool(ls)), flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ), f"physics_ptend_init({name!r})")

    def ptend_sum(self, lchnk: int, ncol: int) -> None:
        _check(self.e.ptend_sum(lchnk, ncol), "physics_ptend_sum")

    def update(self, lchnk: int, dt: float) -> None:
        _check(self.e.update(lchnk, float(dt)), "physics_update")

    def wtrc_apply(self, lchnk: int, top_lev: int, dt: float, prelat: np.ndarray) -> None:
        _check(self.e.wtrc_apply(lchnk, top_lev, float(dt), _ptr(prelat)), "wtrc_apply_rates")

    def outfld(self, name: str, array: np.ndarray, idim: int, lchnk: int) -> None:
        array = _f(array)
        _check(self.e.outfld(name.encode("ascii"), len(name), _ptr(array), idim, lchnk),
               f"outfld({name!r})")

    def cldfrc(self, lchnk: int, ncol: int, arrays: Sequence[np.ndarray], use_shfrc: bool, dindex: int) -> None:
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
            cpair=r("physconst_mp_cpair_"), latice=r("physconst_mp_latice_"),
            latvap=r("physconst_mp_latvap_"), gravit=r("physconst_mp_gravit_"),
            tmelt=r("physconst_mp_tmelt_"),
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


class Macrophysics:
    """CAM5 Park macrophysics: cldfrc + detrainment + mmacro_pcond, as Python.

    ``kernel`` is what computes mmacro_pcond's 23 returned values inside the
    model: ``None`` runs the original through its direct kernel; a callable
    taking the batch mapping ``{name: (ncol, ...) array}`` and returning the
    23 by name -- a ``torch.nn.Module`` wrapped that way, say -- takes its
    place.  Nothing else in :meth:`tend` changes when it does.
    """

    def __init__(self, *, kernel: Callable[..., Mapping[str, np.ndarray]] | None = None) -> None:
        self.kernel = kernel
        self.spec = load_function_spec(FUNCTION)
        self._standalone: Any = None
        self._process: Any = None
        self.calls: list[str] = []          # what tend() did last, for the sequence test

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

    # -- in the driver -----------------------------------------------------------

    def attach(self, run: Any) -> Any:
        """Put :meth:`tend` between the two halves of the macrophysics stage."""

        run.workflow.process(STAGE)          # fail early if this is not a PI-CAM workflow
        process = _MacroTendProcess(self)
        run.workflow.process(STAGE).disable()
        for name in (FIRST_HALF, SECOND_HALF):
            run.workflow.process(name).enable()
        self._process = run.workflow.insert_after(FIRST_HALF, process)
        return self._process

    # -- the transliteration -----------------------------------------------------

    def tend(self, fields: Any, context: Any) -> None:
        """``macrop_driver_tend``, statement for statement, for every chunk."""

        native = context.native
        if native is None:
            raise PhysicsError("Macrophysics.tend must run as a native process")
        state = _RankState.get(native, self)
        dt = float(context.timestep_seconds)
        nstep = int(context.step)
        self.calls = []
        for index, (lchnk, ncol) in enumerate(zip(*native.chunks)):
            self._tend_chunk(state, int(lchnk), int(ncol), index, dt, nstep)

    def _tend_chunk(self, st: "_RankState", lchnk: int, ncol: int, index: int, dt: float, nstep: int) -> None:
        H, C, pb, K = st.handles, st.constants, st.pbuf, st.kernel_on_chunk
        L = st.local
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
        C, s, K = st.constants, st.local, st.kernel_on_chunk
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
        if self.kernel is None:
            K("mmacro_pcond", inputs, outputs=outputs)
            return
        # A model in the kernel's place: the live columns as (ncol, ...) arrays
        # in, the 23 returned values by name out, written to the live columns.
        batch = {}
        for name, value in inputs.items():
            array = np.asarray(value)
            batch[name] = array[:ncol].copy() if array.ndim else array
        answer = self.kernel(batch)
        missing = [name for name in outputs if name not in answer]
        if missing:
            raise PhysicsError(f"kernel returned {len(answer)} of {len(outputs)} values; missing {missing}")
        for name, target in outputs.items():
            target[:ncol, ...] = np.asarray(answer[name], dtype=np.float64)


class _MacroTendProcess(Physics):
    """The installed process: no pool fields, native access, transactional off.

    It reads and writes nothing through the StatePool -- every array it
    touches is CAM's own storage reached by handle -- so there is nothing
    for the transactional snapshot to protect, and the process must say so.
    """

    name = "macro_tend"
    reads = ()
    writes = ()
    native = True
    transactional = False

    def __init__(self, scheme: Macrophysics) -> None:
        self.scheme = scheme

    def run(self, state: Any, context: Any) -> None:
        self.scheme.tend(state, context)


class _RankState:
    """What tend() needs on this rank, built once and kept on the native object."""

    _cache: dict[int, "_RankState"] = {}

    @classmethod
    def get(cls, native: Any, scheme: Macrophysics) -> "_RankState":
        key = id(native.pool)
        try:
            return cls._cache[key]
        except KeyError:
            state = cls(native, scheme)
            cls._cache[key] = state
            return state

    def __init__(self, native: Any, scheme: Macrophysics) -> None:
        self.native = native
        library = native.library
        self.entries = _Entries(library)
        _check(self.entries.bind_hosts(), "pycam_macro_bind_hosts_v1")
        dims = native.pool.dimensions
        self.pcols, self.pver, self.pverp, self.pcnst = (int(dims[k]) for k in ("pcols", "pver", "pverp", "pcnst"))
        self.constants = _Constants.read(library)
        self.constants.refuse_unsupported()
        self.handles = _Handles(self.entries, self.pcnst)
        indices = {symbol: int(module_view(library, f"macrop_driver_mp_{symbol}_", "int32", ()))
                   for _, symbol, _ in __import__("freecam.pi_cam.pbuf", fromlist=["MACROP_FIELDS"]).MACROP_FIELDS}
        self.pbuf = PBuf(library, macrop_fields(indices))
        lchnk, _ = native.chunks
        self.pbuf.verify(int(lchnk[0]), pcols=self.pcols, pver=self.pver)
        self.scratch = self._allocate()
        # cldfrc_fice's promoted descriptor names its fields after StatePool
        # entries; the routine's dummies are ncol, t, fice, fsnow in that order.
        self.fice_fields = dict(zip(("ncol", "t", "fice", "fsnow"),
                                    (a["field"] for a in native.kernel_arguments("cloud_fraction_fice"))))
        _check(self.entries.set_owner(1), "pycam_macro_set_owner_v1")

    def _allocate(self) -> dict[str, np.ndarray]:
        """One (…, 1) F-ordered array per kernel field, plus the driver's locals."""

        dims = {"pcols": self.pcols, "pver": self.pver, "pverp": self.pverp, "pcnst": self.pcnst,
                "pwtype": PWTYPE, "wtrc_nwset": self.constants.wtrc_nwset, "chunks": 1}
        scratch: dict[str, np.ndarray] = {}
        for name in ("macrop_detrain_partition", "macrop_clear_fraction", "macrop_advective_forcing",
                     "macrop_kernel_to_ptend", "macrop_tracer_rate_split", "macrop_cloud_mixing_ratio",
                     "macrop_save_equilibrium", "mmacro_pcond", "wtrc_init_rates", "wtrc_add_rates"):
            for argument in self.native.kernel_arguments(name):
                key = argument["field"].removeprefix("macro.")
                if key in scratch:
                    continue
                shape = tuple(dims[e] if isinstance(e, str) else int(e) for e in argument["extents"])
                scratch[key] = np.zeros(shape, dtype=np.dtype(argument["dtype"]), order="F")
        for name in ("concld_old", "rhcloud", "cldst", "rhu00", "icecldf", "liqcldf", "relhum", "shfrc_zero"):
            scratch.setdefault(name, np.zeros((self.pcols, self.pver, 1), dtype=np.float64, order="F"))
        scratch.setdefault("clc", np.zeros((self.pcols, 1), dtype=np.float64, order="F"))
        return scratch

    @property
    def local(self) -> "_Local":
        """The scratch arrays with the chunk axis dropped: live views, never copies."""

        return _Local(self.scratch)


    def cam_in(self, index: int) -> dict[str, np.ndarray]:
        pool = self.native.pool
        return {name: _f(np.asarray(pool[f"cam_in.{name}"])[:, index])
                for name in ("landfrac", "ocnfrac", "snowhland", "ts", "sst")}

    def kernel_on_chunk(self, name: str, inputs: Mapping[str, Any], *, outputs: Mapping[str, Any],
                        fields: Mapping[str, str] | None = None) -> None:
        """Run one direct kernel on one chunk, copying views in and out exactly.

        Inputs that are handle or buffer views are copied into the kernel's
        scratch slice before the call; outputs mapped to a view are copied
        back after it.  ``None`` means "the scratch array of the same name".
        Every copy is a bit-exact move of doubles; no arithmetic happens here.
        """

        arrays: dict[str, np.ndarray] = {}
        inverse = {} if fields is None else {field: local for local, field in fields.items()}
        for argument in self.native.kernel_arguments(name):
            field = argument["field"]
            local = field.removeprefix("macro.") if fields is None else inverse[field]
            scratch = self.scratch[local] if local in self.scratch else self._scratch_for(argument, local)
            value = inputs.get(local)
            if value is not None:
                self._copy_in(scratch, value)
            arrays[field] = scratch
        self.native.run_kernel(name, arrays)
        for local, target in outputs.items():
            if target is None:
                continue
            self._copy_out(target, self.scratch[local])

    def _scratch_for(self, argument: Mapping[str, Any], key: str) -> np.ndarray:
        dims = {"pcols": self.pcols, "pver": self.pver, "pverp": self.pverp, "pcnst": self.pcnst,
                "pwtype": PWTYPE, "wtrc_nwset": self.constants.wtrc_nwset, "chunks": 1}
        shape = tuple(dims[e] if isinstance(e, str) else int(e) for e in argument["extents"])
        self.scratch[key] = np.zeros(shape, dtype=np.dtype(argument["dtype"]), order="F")
        return self.scratch[key]

    @staticmethod
    def _copy_in(scratch: np.ndarray, value: Any) -> None:
        array = np.asarray(value)
        if array.ndim == 0:
            scratch[...] = array.astype(scratch.dtype)
        else:
            scratch[..., 0] = array.astype(scratch.dtype, copy=False)

    @staticmethod
    def _copy_out(target: np.ndarray, scratch: np.ndarray) -> None:
        if target.ndim == 0 or target.ndim == scratch.ndim:
            target[...] = scratch
        else:
            target[...] = scratch[..., 0]


class _Local(Mapping[str, np.ndarray]):
    """``scratch[name][..., 0]`` on every access, so late allocations are seen."""

    def __init__(self, scratch: dict[str, np.ndarray]) -> None:
        self._scratch = scratch

    def __getitem__(self, name: str) -> np.ndarray:
        return self._scratch[name][..., 0]

    def __iter__(self):
        return iter(self._scratch)

    def __len__(self) -> int:
        return len(self._scratch)


__all__ = ["FIRST_HALF", "FORCING", "Macrophysics", "SECOND_HALF", "SEQUENCE", "STAGE", "VIEW"]
