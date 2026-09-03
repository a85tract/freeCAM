"""MG1.0 cloud microphysics -- ``micro_mg_cam_tend`` -- as a Python class.

``microp_driver_tend`` is one call, ``micro_mg_cam_tend``
(``micro_mg_cam.F90:997-3184`` in the pinned source): 2188 lines that read
sixty physics-buffer fields, copy the state, pack the cloudy columns, run
the MG core, unpack, and turn the outputs into diagnostics, water-tracer
rates, cloud sizes and a hundred history fields.

:class:`Microphysics` is that routine statement for statement, in Python,
with every floating-point number still Fortran's:

* the packer section (lines 1768-2286: ``micro_mg_get_cols``, the
  ``MGPacker``/``MGPostProc`` objects, 117 packed arrays, the substep
  loop, the 116-argument core call, the unpacks) is lifted **verbatim**
  into ``pycam_micro_handles`` as five procedures over chunk-local module
  state, because it is bookkeeping around one call and carries no
  Python-visible semantics;
* the routine's 63 live arithmetic statements outside it are the twelve
  lifted ``micro_*`` kernels (``tools/generate_pi_cam_micro_kernels.py``),
  plus the water-tracer rate kernels macrophysics already promoted;
* copies, zeroings, pointer aliases, the sixty buffer reads and the
  hundred history writes are Python, on zero-copy views.

The core, ``micro_mg_tend``, is the stage's swappable kernel: with no model
installed the original runs inside the lifted section and the stage is
bit-for-bit.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..pi_cam.errors import PICAMConfigurationError
from ..pi_cam.pbuf import PBuf, load_pbuf_table
from .image import module_view
from .macrophysics import CPAIR, IWTICE, IWTLIQ, IWTVAP, PWTYPE
from .errors import PhysicsError
from .result import FunctionResult
from .stage import (
    CORE_ENTRIES,
    HostEntries,
    HostServices,
    NativeStage,
    StageRuntime,
    check as _check,
    pointer_of as _ptr,
)

REPO = Path(__file__).resolve().parents[3]
SOURCE = "physics/cam/micro_mg_cam.F90"

#: The handles module's view codes and input codes, as the generator wrote
#: them (tools/generate_pi_cam_micro_handles.py).
VIEWS_TABLE = REPO / "native/pi_cam/micro_views.yaml"
#: The sixty physics-buffer fields the routine reads
#: (tools/generate_pi_cam_pbuf_table.py).
PBUF_TABLE = REPO / "native/pi_cam/pbuf_fields_micro.yaml"

CORE = "micro_mg_tend"

# water_types.F90: named constants, so not symbols in the image; the
# remaining two the macrophysics stage did not need.  A test pins them.
IWTSTRAIN, IWTSTSNOW = 4, 5
# physconst.F90: rhoh2o = shr_const_rhofw, a parameter.  A test pins the
# literal against the pinned shr_const_mod.F90.
RHOH2O = 1.000e3
# micro_mg_utils.F90:108,111: parameters, so not symbols either.
QSMALL, MINCLD = 1.e-18, 0.0001


def _load_views() -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    import yaml

    payload = yaml.safe_load(VIEWS_TABLE.read_text())
    views = {row["name"]: int(row["code"]) for row in payload["views"]}
    ranks = {row["name"]: int(row["rank"]) for row in payload["views"]}
    inputs = {row["name"]: int(row["code"]) for row in payload["inputs"]}
    return views, ranks, inputs


#: ``pycam_micro_view_v1`` codes by name, their ranks, and the
#: ``pycam_micro_bind_input_v1`` codes by name.
VIEW, VIEW_RANK, INPUT = _load_views()
#: What ``pycam_micro_configure_v1`` takes, in order.
CONFIGURATION = ("micro_mg_version", "micro_mg_sub_version", "num_steps", "microp_uniform",
                 "do_cldice", "do_cldliq", "ixcldliq", "ixcldice", "ixnumliq", "ixnumice",
                 "ixrain", "ixsnow", "ixnumrain", "ixnumsnow")

#: The direct kernels tend() runs.  The two water-tracer kernels are the
#: macrophysics stage's (``macro.`` fields), used through a field map.
KERNELS = (
    "micro_initial_water_paths", "micro_precip_diagnostics", "micro_macro_feedback",
    "micro_in_cloud_quantities", "micro_split_signs", "micro_air_density",
    "micro_liquid_size", "micro_ice_and_precip_size", "micro_precip_efficiency",
    "micro_autoconversion_ratio", "micro_effective_outputs", "micro_history_tendencies",
    "wtrc_init_rates", "wtrc_add_rates",
)
UNSCRATCHED = ("wtrc_init_rates", "wtrc_add_rates")
WTRC_FIELDS = {"ncol": "macro.ncol", "top_lev": "macro.top_lev", "isrctype": "macro.isrctype",
               "idsttype": "macro.idsttype", "rtype": "macro.rtype", "rate": "macro.rate"}

#: Routine locals the walk keeps that no kernel declares.
EXTRA_SCRATCH = (("sed_rates_grid", ("pcols", "pver", "pwtype", "chunks")),)

#: The ``_grid`` arrays that are pointers into, or copies into, the physics
#: buffer (2467-2489, 1700-1713): name -> buffer field.  Every other
#: ``_grid`` array is a routine local, i.e. kernel scratch.
GRID_PBUF = {
    "lambdac_grid": "LAMBDAC", "mu_grid": "MU", "rel_grid": "REL", "rei_grid": "REI",
    "dei_grid": "DEI", "des_grid": "DES", "prec_str_grid": "PREC_STR", "iclwpst_grid": "ICLWPST",
    "cvreffliq_grid": "CV_REFFLIQ", "cvreffice_grid": "CV_REFFICE", "mgflxprc_grid": "LS_FLXPRC",
    "mgflxsnw_grid": "LS_FLXSNW", "qme_grid": "QME", "nevapr_grid": "NEVAPR", "prain_grid": "PRAIN",
    "mgreffrain_grid": "LS_REFFRAIN", "mgreffsnow_grid": "LS_REFFSNOW", "acprecl_grid": "ACPRECL",
    "acgcme_grid": "ACGCME", "acnum_grid": "ACNUM", "cmeliq_grid": "CMELIQ", "ast_grid": "AST",
    "evprain_st_grid": "evprain_st", "evpsnow_st_grid": "evpsnow_st", "am_evp_st_grid": "am_evp_st",
}
#: 2490-2519: ``x_grid = x`` -- copies of MG outputs into routine locals.
GRID_COPIES = (
    "qrout", "qsout", "nsout", "nrout", "qcreso", "melto", "mnuccco", "mnuccto", "bergo",
    "homoo", "msacwio", "psacwso", "bergso", "cmeiout", "qireso", "prcio", "praio",
    "prao", "prco",
)

#: 3040-3060: the six budget terms the history kernel forms.
HISTORY_TENDENCIES = (("MPDW2V", "mpdw2v"), ("MPDW2I", "mpdw2i"), ("MPDW2P", "mpdw2p"),
                      ("MPDI2V", "mpdi2v"), ("MPDI2W", "mpdi2w"), ("MPDI2P", "mpdi2p"))
#: 3063-3103: history of the MG outputs, in order (mg1.0: no QRSEDTEN,
#: QSSEDTEN, UMR, UMS, QCRAT).
HISTORY_MG = (
    ("MPICLWPI", "iclwpi"), ("MPICIWPI", "iciwpi"), ("REFL", "refl"), ("AREFL", "arefl"),
    ("AREFLZ", "areflz"), ("FREFL", "frefl"), ("CSRFL", "csrfl"), ("ACSRFL", "acsrfl"),
    ("FCSRFL", "fcsrfl"), ("RERCLD", "rercld"), ("NCAL", "ncal"), ("NCAI", "ncai"),
    ("AQRAIN", "qrout2"), ("AQSNOW", "qsout2"), ("ANRAIN", "nrout2"), ("ANSNOW", "nsout2"),
    ("FREQR", "freqr"), ("FREQS", "freqs"), ("MPDT", "tlat"), ("MPDQ", "qvlat"),
    ("MPDLIQ", "qcten"), ("MPDICE", "qiten"), ("EVAPSNOW", "evapsnow"), ("QCSEVAP", "qcsevap"),
    ("QISEVAP", "qisevap"), ("QVRES", "qvres"), ("VTRMC", "vtrmc"), ("VTRMI", "vtrmi"),
    ("QCSEDTEN", "qcsedten"), ("QISEDTEN", "qisedten"), ("MNUCCDO", "mnuccdo"),
    ("MNUCCDOhet", "mnuccdohet"), ("MNUCCRO", "mnuccro"), ("PRACSO", "pracso"),
    ("MELTSDT", "meltsdt"), ("FRZRDT", "frzrdt"), ("FICE", "nfice"),
)
#: 3120-3176: history of the grid arrays, in order.
HISTORY_GRID = (
    ("QRAIN", "qrout_grid"), ("QSNOW", "qsout_grid"), ("NRAIN", "nrout_grid"), ("NSNOW", "nsout_grid"),
    ("CV_REFFLIQ", "cvreffliq_grid"), ("CV_REFFICE", "cvreffice_grid"), ("LS_FLXPRC", "mgflxprc_grid"),
    ("LS_FLXSNW", "mgflxsnw_grid"), ("CME", "qme_grid"), ("PRODPREC", "prain_grid"),
    ("EVAPPREC", "nevapr_grid"), ("QCRESO", "qcreso_grid"), ("LS_REFFRAIN", "mgreffrain_grid"),
    ("LS_REFFSNOW", "mgreffsnow_grid"), ("DSNOW", "des_grid"), ("ADRAIN", "drout2_grid"),
    ("ADSNOW", "dsout2_grid"), ("PE", "pe_grid"), ("PEFRAC", "pefrac_grid"), ("APRL", "tpr_grid"),
    ("VPRAO", "vprao_grid"), ("VPRCO", "vprco_grid"), ("RACAU", "racau_grid"), ("AREL", "efcout_grid"),
    ("AREI", "efiout_grid"), ("AWNC", "ncout_grid"), ("AWNI", "niout_grid"), ("FREQL", "freql_grid"),
    ("FREQI", "freqi_grid"), ("ACTREL", "ctrel_grid"), ("ACTREI", "ctrei_grid"), ("ACTNL", "ctnl_grid"),
    ("ACTNI", "ctni_grid"), ("FCTL", "fctl_grid"), ("FCTI", "fcti_grid"), ("ICINC", "icinc_grid"),
    ("ICWNC", "icwnc_grid"), ("EFFLIQ_IND", "rel_fn_grid"), ("CDNUMC", "cdnumc_grid"), ("REL", "rel_grid"),
    ("REI", "rei_grid"), ("ICIMRST", "icimrst_grid_out"), ("ICWMRST", "icwmrst_grid_out"),
    ("CMEIOUT", "cmeiout_grid"), ("PRAO", "prao_grid"), ("PRCO", "prco_grid"), ("MNUCCCO", "mnuccco_grid"),
    ("MNUCCTO", "mnuccto_grid"), ("MSACWIO", "msacwio_grid"), ("PSACWSO", "psacwso_grid"),
    ("BERGSO", "bergso_grid"), ("BERGO", "bergo_grid"), ("MELTO", "melto_grid"), ("HOMOO", "homoo_grid"),
    ("PRCIO", "prcio_grid"), ("PRAIO", "praio_grid"), ("QIRESO", "qireso_grid"),
)

#: The core's arguments (micro_mg1_0.F90 ``micro_mg_tend``) as the packed
#: arrays the lifted section hands it, by intent: what a model in its place
#: receives, and what it must answer.  ``qc, qi, nc, ni`` are inout.  The
#: five ``*_dum`` actuals (effc_fn, reff_rain, reff_snow, drout2, dsout2)
#: are outputs the driver discards and are not part of the contract.
PACKED_INPUTS = ("t", "q", "qc", "qi", "nc", "ni", "p", "pdel", "cldn", "liqcldf", "relvar",
                 "accre_enhan", "icecldf", "naai", "npccn", "rndst", "nacon")
PACKED_INPUTS_NO_CLDICE = ("tnd_qsnow", "tnd_nsnow", "re_ice")
PACKED_INPUTS_HETFRZ = ("frzimm", "frzcnt", "frzdep")
PACKED_OUTPUTS = (
    "qc", "qi", "nc", "ni", "rate1ord_cw2pr_st", "tlat", "qvlat", "qctend", "qitend", "nctend",
    "nitend", "rel", "rei", "prect", "preci", "nevapr", "evapsnow", "am_evp_st", "prain",
    "prodsnow", "cmeout", "dei", "mu", "lambdac", "qsout", "des", "rflx", "sflx", "qrout",
    "qcsevap", "qisevap", "qvres", "cmei", "vtrmc", "vtrmi", "qcsedten", "qisedten", "pra", "prc",
    "mnuccc", "mnucct", "msacwi", "psacws", "bergs", "berg", "melt", "homo", "qcres", "prci",
    "prai", "qires", "mnuccr", "pracs", "meltsdt", "frzrdt", "mnuccd", "nrout", "nsout", "refl",
    "arefl", "areflz", "frefl", "csrfl", "acsrfl", "fcsrfl", "rercld", "ncai", "ncal", "qrout2",
    "qsout2", "nrout2", "nsout2", "freqs", "freqr", "nfice", "prer_evap", "preo", "prdso",
    "frzro", "meltso", "wtfc", "wtfi", "wtprelat", "wtpostlat",
)

#: The driver's packed arrays and the core's dummies are named differently:
#: `packed_t` is the routine's `tn`, `packed_rel` its `effc`.  The pairing is
#: positional in the lifted call, and a test derives this table from it.
PACKED_TO_DUMMY = {
    "t": "tn", "q": "qn", "npccn": "npccnin", "rel": "effc", "rei": "effi",
    "dei": "deffi", "mu": "pgamrad", "lambdac": "lamcrad", "des": "dsout",
    "cmei": "cmeiout", "pra": "prao", "prc": "prco", "mnuccc": "mnuccco",
    "mnucct": "mnuccto", "msacwi": "msacwio", "psacws": "psacwso", "bergs": "bergso",
    "berg": "bergo", "melt": "melto", "homo": "homoo", "qcres": "qcreso",
    "prci": "prcio", "prai": "praio", "qires": "qireso", "mnuccr": "mnuccro",
    "pracs": "pracso", "mnuccd": "mnuccdo",
}
#: What the core reads that is not a packed array: the configuration it
#: branches on and the substep's length.
PACKED_SCALARS = ("microp_uniform", "do_cldice", "deltatin")
#: Everything the core answers, under its own dummy names.  A model in its
#: place is given a column under those names and must answer all of them.
RETURNED = tuple(PACKED_TO_DUMMY.get(name, name) for name in PACKED_OUTPUTS)

#: The order in which the Fortran routine does things under the admitted
#: configuration; ``tend`` follows it per chunk and a test compares the two
#: against the pinned source.
SEQUENCE = (
    "pbuf_get_field*", "micro_initial_water_paths", "cldo=ast", "physics_state_copy",
    "physics_ptend_init:cldwat", "pack_prelude", "substep_pack", CORE, "substep_unpack",
    "post_proc", "micro_precip_diagnostics", "cvreff=const", "rate1ord=rate1cld",
    "micro_macro_feedback", "zero6", "micro_in_cloud_quantities", "grid_aliases", "grid_copies",
    "wtrc_aliases", "wtrc_init_rates", "micro_split_signs", "wtrc_add_rates*", "sed_rates",
    "wtrc_init_rates", "wtrc_add_rates*", "wtrc_apply_rates", "micro_air_density", "rho_grid=rho",
    "micro_liquid_size", "micro_ice_and_precip_size", "micro_precip_efficiency",
    "micro_autoconversion_ratio", "micro_effective_outputs", "grid_ptr=grid",
    "micro_history_tendencies", "outfld*", "wtrc_output_precip", "physics_state_dealloc",
)


# -- the image, seen through ctypes ---------------------------------------------

_INT, _DBL = ctypes.c_int, ctypes.c_double
_P_DBL = ctypes.POINTER(ctypes.c_double)


class _MicroEntries(HostEntries):
    """The core entries plus the lifted section's and the routine's own calls."""

    TABLE = {
        **CORE_ENTRIES,
        "set_core_owner": ("pycam_{prefix}_set_core_owner_v1", [_INT], False),
        "configure": ("pycam_{prefix}_configure_v1", [_INT] * len(CONFIGURATION), False),
        "begin": ("pycam_{prefix}_begin_v1", [_INT, _INT, _DBL], False),
        "init_ptend": ("pycam_{prefix}_ptend_init_v1", [_INT], False),
        "bind_input": ("pycam_{prefix}_bind_input_v1",
                       [_INT, _INT, ctypes.c_void_p, _INT, _INT, _INT], False),
        "pack_prelude": ("pycam_{prefix}_pack_prelude_v1", [_INT], False),
        "substep_pack": ("pycam_{prefix}_substep_pack_v1", [_INT, _INT], False),
        "core": ("pycam_{prefix}_core_v1", [_INT], False),
        "substep_unpack": ("pycam_{prefix}_substep_unpack_v1", [_INT], False),
        "post_proc": ("pycam_{prefix}_post_proc_v1", [_INT], False),
        "wtrc_apply": ("pycam_{prefix}_wtrc_apply_v1",
                       [_INT, _P_DBL, _P_DBL, _P_DBL, _P_DBL, _P_DBL], False),
        "wtrc_add_sum": ("pycam_{prefix}_wtrc_add_sum_v1",
                         [_P_DBL, _INT, _INT, _INT, _INT, _INT, _P_DBL, _P_DBL, _P_DBL, _INT], False),
        "output_precip": ("pycam_{prefix}_output_precip_v1", [_INT], False),
        "end": ("pycam_{prefix}_end_v1", [_INT], False),
    }


class _MicroHandles(HostServices):
    """CAM's host services, plus the lifted section and the routine's calls."""

    def configure(self, constants: "_Constants") -> None:
        values = [int(getattr(constants, name)) for name in CONFIGURATION]
        _check(self.e.configure(*values), "pycam_micro_configure_v1")

    def set_core_owner(self, owns: bool) -> None:
        _check(self.e.set_core_owner(int(owns)), "pycam_micro_set_core_owner_v1")

    def begin(self, lchnk: int, ncol: int, dtime: float) -> None:
        """1556-1559 and 1741: the chunk's sizes and the state copy."""

        _check(self.e.begin(lchnk, ncol, float(dtime)), "micro begin (physics_state_copy)")

    def init_ptend(self, lchnk: int) -> None:
        """1743-1766: the constituent flags and the cldwat ptend."""

        _check(self.e.init_ptend(lchnk), "physics_ptend_init('cldwat')")

    def bind_input(self, lchnk: int, name: str, array: np.ndarray) -> None:
        """Hand the lifted section the buffer storage it packs, by address."""

        assert array.flags.f_contiguous and array.dtype == np.float64, name
        extents = list(array.shape) + [1] * (3 - array.ndim)
        _check(self.e.bind_input(lchnk, INPUT[name], array.ctypes.data, *extents),
               f"bind_input({name!r})")

    def pack_prelude(self, lchnk: int) -> None:
        _check(self.e.pack_prelude(lchnk), "micro pack_prelude (1768-2069)")

    def substep_pack(self, lchnk: int, it: int) -> None:
        _check(self.e.substep_pack(lchnk, it), "micro substep_pack (2074-2086)")

    def core(self, lchnk: int) -> None:
        _check(self.e.core(lchnk), "micro_mg_tend (2087-2206)")

    def substep_unpack(self, lchnk: int) -> None:
        _check(self.e.substep_unpack(lchnk), "micro substep_unpack (2210-2247)")

    def post_proc(self, lchnk: int) -> None:
        _check(self.e.post_proc(lchnk), "micro post_proc (2252-2286)")

    def wtrc_apply(self, lchnk: int, pre: np.ndarray, sed: np.ndarray, post: np.ndarray,
                   alst_mic: np.ndarray, aist_mic: np.ndarray) -> None:
        _check(self.e.wtrc_apply(lchnk, _ptr(pre), _ptr(sed), _ptr(post), _ptr(alst_mic),
                                 _ptr(aist_mic)), "wtrc_apply_rates")

    def wtrc_add_sum(self, rates: np.ndarray, ncol: int, top_lev: int, isrc: int, idst: int,
                     rtype: int, terms: Sequence[np.ndarray]) -> None:
        """``wtrc_add_rates`` with a rate that is the sum of two or three arrays;
        the sum is formed in Fortran."""

        assert len(terms) in (2, 3)
        arrays = list(terms) + [terms[0]] * (3 - len(terms))
        _check(self.e.wtrc_add_sum(_ptr(rates), ncol, top_lev, isrc, idst, rtype,
                                   *[_ptr(a) for a in arrays], len(terms)), "wtrc_add_rates(sum)")

    def output_precip(self, lchnk: int) -> None:
        _check(self.e.output_precip(lchnk), "wtrc_output_precip")

    def end(self, lchnk: int) -> None:
        _check(self.e.end(lchnk), "micro end (physics_state_dealloc)")


# -- module constants --------------------------------------------------------------


@dataclass(frozen=True)
class _Constants:
    micro_mg_version: int
    micro_mg_sub_version: int
    num_steps: int
    microp_uniform: bool
    do_cldice: bool
    do_cldliq: bool
    ixcldliq: int
    ixcldice: int
    ixnumliq: int
    ixnumice: int
    ixrain: int
    ixsnow: int
    ixnumrain: int
    ixnumsnow: int
    qrain_idx: int
    qsnow_idx: int
    nrain_idx: int
    nsnow_idx: int
    rate1_cw2pr_st_idx: int
    am_evp_st_idx: int
    use_hetfrz_classnuc: bool
    use_subcol_microp: bool
    trace_water: bool
    top_lev: int
    gravit: float
    rair: float
    cpair: float
    rhoh2o: float
    mincld: float
    qsmall: float

    @classmethod
    def read(cls, library: Any) -> "_Constants":
        def i(symbol):
            return int(module_view(library, symbol, "int32", ()))

        def b(symbol):
            return bool(int(module_view(library, symbol, "int32", ())))

        def r(symbol):
            return float(module_view(library, symbol, "float64", ()))

        m = "micro_mg_cam_mp_{}_".format
        return cls(
            micro_mg_version=i(m("micro_mg_version")), micro_mg_sub_version=i(m("micro_mg_sub_version")),
            num_steps=i(m("num_steps")), microp_uniform=b(m("microp_uniform")),
            do_cldice=b(m("do_cldice")), do_cldliq=b(m("do_cldliq")),
            ixcldliq=i(m("ixcldliq")), ixcldice=i(m("ixcldice")),
            ixnumliq=i(m("ixnumliq")), ixnumice=i(m("ixnumice")),
            ixrain=i(m("ixrain")), ixsnow=i(m("ixsnow")),
            ixnumrain=i(m("ixnumrain")), ixnumsnow=i(m("ixnumsnow")),
            qrain_idx=i(m("qrain_idx")), qsnow_idx=i(m("qsnow_idx")),
            nrain_idx=i(m("nrain_idx")), nsnow_idx=i(m("nsnow_idx")),
            rate1_cw2pr_st_idx=i(m("rate1_cw2pr_st_idx")), am_evp_st_idx=i(m("am_evp_st_idx")),
            use_hetfrz_classnuc=b("phys_control_mp_use_hetfrz_classnuc_"),
            use_subcol_microp=b("phys_control_mp_use_subcol_microp_"),
            trace_water=b("water_tracer_vars_mp_trace_water_"),
            top_lev=i("ref_pres_mp_trop_cloud_top_lev_"),
            gravit=r("physconst_mp_gravit_"), rair=r("physconst_mp_rair_"),
            cpair=CPAIR, rhoh2o=RHOH2O, mincld=MINCLD, qsmall=QSMALL,
        )

    def refuse_unsupported(self) -> None:
        """The paths the admitted configuration never takes are not ported."""

        def refuse(what: str) -> None:
            raise PICAMConfigurationError(
                f"{what}; the Python microphysics does not carry that path")

        if (self.micro_mg_version, self.micro_mg_sub_version) != (1, 0):
            refuse(f"micro_mg_version {self.micro_mg_version}.{self.micro_mg_sub_version} is not 1.0")
        if self.use_subcol_microp:
            refuse("use_subcol_microp is on (subcolumn microphysics)")
        if self.num_steps < 1:
            refuse(f"micro_mg_num_steps is {self.num_steps}")
        if self.top_lev != 1:
            # 2666-2667 assign (pcols,pver) arrays into (:,top_lev:,:) sections;
            # the source conforms only when the cloud top is the model top
            refuse(f"trop_cloud_top_lev is {self.top_lev}, not 1")


# -- the class ---------------------------------------------------------------------


class Microphysics(NativeStage):
    """``micro_mg_cam_tend`` as Python; ``micro_mg_tend`` is the swappable core.

    A sub-walk of :class:`CloudMacroMicrophysics` in ``microp_driver_tend``'s
    place; it has no workflow action of its own.
    """

    STAGE = "cam_run1.cloud_macro_microphysics"
    PREFIX = "micro"
    PROCESS_NAME = "micro_mg_cam_tend"
    TRACE_ENV = "FREECAM_MICRO_TRACE"
    PROFILE_ENV = "FREECAM_MICRO_PROFILE"

    KERNELS = KERNELS
    UNSCRATCHED = UNSCRATCHED
    EXTRA_SCRATCH = EXTRA_SCRATCH
    SWAPPABLE = (CORE,)

    entries_class = _MicroEntries
    services_class = _MicroHandles

    def __init__(self, *, kernel=None, kernels=None, standalone_core: bool = False) -> None:
        super().__init__(kernel=kernel, kernels=kernels)
        self._standalone: Any = None
        self._scalars: dict[str, Any] = {}
        self._discarded: dict[str, np.ndarray] | None = None
        #: Run the core through its standalone image rather than the copy the
        #: model holds.  A flag, not the callable: the stage is cloudpickled to
        #: every rank when it is installed, and a loaded image is not picklable.
        #: Each rank opens its own on first use.
        self.use_standalone_core = bool(standalone_core)

    # -- standalone ------------------------------------------------------------

    def micro_mg_tend(self, inputs: Mapping[str, Any],
                      parameters: Mapping[str, Any] | None = None):
        """The stage's numerical core: one column in, one column out.

        This is the only place in the class that says what computes
        ``micro_mg_tend``.  :meth:`tend_chunk` calls it for every packed
        column the driver gathered, and a notebook calls it for a column of
        its own; installing a model under :attr:`kernels` changes both.
        """

        model = self.kernels[CORE]
        if model is None:
            return self._function().run(inputs, parameters)
        answer = dict(model(inputs, parameters)
                      if getattr(model, "takes_parameters", False)
                      else model(inputs))
        missing = [name for name in RETURNED if name not in answer]
        if missing:
            raise PhysicsError(
                f"the model in {CORE}'s place returned {len(answer)} of {len(RETURNED)} "
                f"values; missing {missing}")
        return FunctionResult(outputs=answer, updated_inputs={})

    def example_input(self, name: str = "captured-anchor"):
        return self._function().example_input(name)

    def _function(self):
        if self._standalone is None:
            from .function import load_function

            self._standalone = load_function(CORE)
        return self._standalone

    def close(self) -> None:
        if self._standalone is not None:
            self._standalone.close()
            self._standalone = None

    def standalone_core(self):
        """The original core, as a model the stage may run in its own place.

        Gate M-5's form: ``stage.kernels["micro_mg_tend"] =
        micro.standalone_core()`` sends every packed column through the
        standalone image instead of the routine inside the lifted section.
        The answer must be the same to the bit -- it is the same machine
        code, on the same numbers -- which is what makes the packed contract
        worth trusting before a model takes its place.
        """

        function = self._function()
        dummy = {item.name: item for item in function.spec.arguments}
        # The driver hands five outputs to locals it discards (*_dum), and two
        # of them are intent(inout): the routine zeroes both at 995-996 before
        # it writes them, so what goes in cannot be read.  Zeros stand in.
        discarded = {item.name: np.zeros(item.public_extent(function.spec.dimensions))
                     for item in function.spec.arguments
                     if item.role == "inout" and item.name not in PACKED_TO_DUMMY.values()
                     and item.name not in PACKED_OUTPUTS}

        def model(batch: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
            packed = {name: np.asarray(value) for name, value in batch.items()}
            rows = int(packed["t"].shape[0])
            levels = int(packed["t"].shape[1])
            # rflx and sflx carry one more level than the rest: the shape is
            # the routine's own, not this batch's
            def shape(name: str) -> tuple[int, ...]:
                item = dummy[PACKED_TO_DUMMY.get(name, name)]
                extent = item.public_extent(function.spec.dimensions)
                return (rows, *(levels + 1 if axis > levels else axis for axis in extent))

            answer = {name: np.zeros(shape(name)) for name in PACKED_OUTPUTS
                      if PACKED_TO_DUMMY.get(name, name) in dummy}
            for row in range(rows):
                one: dict[str, Any] = dict(discarded)
                for name, value in packed.items():
                    key = PACKED_TO_DUMMY.get(name, name)
                    if key not in dummy:
                        continue
                    one[key] = value if name in PACKED_SCALARS else value[row]
                result = function.run(one)
                values = {**result.outputs, **result.updated_inputs}
                for name in answer:
                    answer[name][row] = np.asarray(values[PACKED_TO_DUMMY.get(name, name)])
            return answer

        model.takes_packed_batch = True   # packed names, batch in, batch out
        return model

    # -- what the runtime asks of this stage -------------------------------------

    def read_constants(self, library: Any) -> "_Constants":
        return _Constants.read(library)

    def refuse_unsupported(self, constants: "_Constants") -> None:
        constants.refuse_unsupported()

    def extra_extents(self, constants: "_Constants") -> Mapping[str, int]:
        return {"pwtype": PWTYPE}

    def build_pbuf(self, library: Any, runtime: StageRuntime) -> PBuf:
        import yaml

        symbols = [row["symbol"] for row in yaml.safe_load(PBUF_TABLE.read_text())["fields"]]
        indices = {symbol: int(module_view(library, symbol, "int32", ())) for symbol in symbols}
        buffer = PBuf(library, load_pbuf_table(PBUF_TABLE, indices))
        lchnk, _ = runtime.native.chunks
        buffer.verify(int(lchnk[0]), pcols=runtime.pcols, pver=runtime.pver)
        return buffer

    def after_runtime(self, runtime: StageRuntime) -> None:
        # the lifted section branches on the module's configuration; it is
        # module state there, set once from what the image holds
        runtime.handles.configure(runtime.constants)
        runtime.handles.set_core_owner(self.kernels.get(CORE) is not None)

    # -- the transliteration -----------------------------------------------------

    def tend_chunk(self, st: StageRuntime, lchnk: int, ncol: int, index: int,
                   dt: float, nstep: int) -> None:
        """``micro_mg_cam_tend`` (997-3184) under the admitted configuration.

        ``dt`` is the driver's ``dtime``.  Line numbers are the pinned
        source's.  Every kernel is run with every output view also copied
        in, so the scratch holds the Fortran array in every lane and level
        before the call and the copy back is exact whatever the kernel
        assigns -- whole arrays included.
        """

        H, C, pb = st.handles, st.constants, st.pbuf
        L = st.local
        log = self.calls.append
        top = C.top_lev
        pcols, pver = st.pcols, st.pver
        cols, lev = slice(0, ncol), slice(top - 1, pver)
        ngrdcol = ncol                          # no subcolumns: 1559

        def V(name: str) -> np.ndarray:
            return H.view(lchnk, VIEW[name])

        def K(name, inputs, *, outputs, fields=None):
            # A field the walk does not name keeps whatever its scratch holds,
            # which is right for a routine local the previous kernel wrote and
            # wrong for anything living in the physics buffer or a handle view:
            # that storage is never the scratch.  Refuse rather than read zeros.
            if fields is None:
                named = set(inputs) | set(outputs)
                missing = [a.field.removeprefix(f"{self.PREFIX}.")
                           for a in st.descriptors[name].arguments
                           if a.field.removeprefix(f"{self.PREFIX}.") in GRID_PBUF
                           and a.field.removeprefix(f"{self.PREFIX}.") not in named]
                if missing:
                    raise PhysicsError(
                        f"{name} takes {missing} from the physics buffer; the walk "
                        f"passes neither the value nor a target, so the kernel would "
                        f"read this stage's scratch instead")
            merged = {**{k: v for k, v in outputs.items() if v is not None}, **inputs}
            st.kernel_on_chunk(name, merged, outputs=outputs, fields=fields, ncol=None)

        def G(name: str) -> np.ndarray:
            """A ``_grid`` array: buffer storage where the source points into
            it, the walk's scratch otherwise."""

            return pbv[GRID_PBUF[name]] if name in GRID_PBUF else L[name]

        # 1556-1559, 1741: sizes and the state copy (the copy is logged where
        # the source makes it)
        H.begin(lchnk, ncol, dt)
        # both the host state and the driver's copy: the routine reads the
        # host's before the copy and again after the substep updated the copy
        S = {name: V(name) for name in (
            "state_t", "state_q", "state_pmid", "state_pdel",
            "state_loc_t", "state_loc_q", "state_loc_pmid", "state_loc_pdel")}
        q_loc = S["state_loc_q"]

        # 1573-1644, 1700-1714: the buffer fields, older time sample where the
        # source says so; the lifted section packs the same storage
        names = [name for name in pb.fields if name in pb]
        pbv = {name: pb.view(name, lchnk) for name in names}
        for name in ("naai", "npccn", "rndst", "nacon", "relvar", "accre_enhan"):
            H.bind_input(lchnk, name, pbv[name.upper()])
        H.bind_input(lchnk, "ast", pbv["AST"])
        if not C.do_cldice:                                    # 1589-1593
            for name in ("tnd_qsnow", "tnd_nsnow", "re_ice"):
                H.bind_input(lchnk, name, pbv[name.upper()])
        if C.use_hetfrz_classnuc:                              # 1595-1599
            for name in ("frzimm", "frzcnt", "frzdep"):
                H.bind_input(lchnk, name, pbv[name.upper()])
        for name in ("rel", "rei", "dei", "des", "mu", "lambdac", "prain", "nevapr", "prer_evap"):
            H.bind_input(lchnk, name, pbv[name.upper()])
        if C.rate1_cw2pr_st_idx > 0:                           # 1642-1644
            H.bind_input(lchnk, "rate1ord_cw2pr_st", pbv["RATE1_CW2PR_ST"])
        # 1719-1720: alst_mic => ast; aist_mic => ast
        H.bind_input(lchnk, "alst_mic", pbv["AST"])
        H.bind_input(lchnk, "aist_mic", pbv["AST"])
        alst_mic = aist_mic = pbv["AST"]
        log("pbuf_get_field*")

        # 1724-1736: the host state, before the copy
        K("micro_initial_water_paths",
          {"ncol": ncol, "top_lev": top, "ixcldliq": C.ixcldliq, "ixcldice": C.ixcldice,
           "mincld": C.mincld, "gravit": C.gravit, "q": S["state_q"], "ast": pbv["AST"],
           "pdel": S["state_pdel"]},
          outputs={"iclwpi": None, "iciwpi": None})
        log("micro_initial_water_paths")
        # 1738 [exact: a copy]
        pbv["CLDO"][cols, lev] = pbv["AST"][cols, lev]
        log("cldo=ast")
        log("physics_state_copy")                              # 1741, done in begin
        # 1743-1766
        H.init_ptend(lchnk); log("physics_ptend_init:cldwat")

        # 1768-2286: the lifted section
        H.pack_prelude(lchnk); log("pack_prelude")
        for it in range(1, C.num_steps + 1):                   # 2070
            H.substep_pack(lchnk, it); log("substep_pack")
            self._core_call(st, lchnk, ncol, dt)
            H.substep_unpack(lchnk); log("substep_unpack")
        H.post_proc(lchnk); log("post_proc")

        # 2287-2299
        K("micro_precip_diagnostics",
          {"ncol": ncol, "top_lev": top, "naai": pbv["NAAI"], "naai_hom": pbv["NAAI_HOM"],
           "mnuccdo": V("mnuccdo"), "qrout": V("qrout"), "qsout": V("qsout"),
           "rflx": V("rflx"), "sflx": V("sflx")},
          outputs={"mnuccdohet": None, "mgmrprc": pbv["LS_MRPRC"], "mgmrsnw": pbv["LS_MRSNW"],
                   "mgflxprc": pbv["LS_FLXPRC"], "mgflxsnw": pbv["LS_FLXSNW"]})
        log("micro_precip_diagnostics")
        # 2303-2304 [exact: literals]
        pbv["CV_REFFLIQ"][cols, lev] = 9.0
        pbv["CV_REFFICE"][cols, lev] = 37.0
        log("cvreff=const")
        # 2307-2309 [exact: a copy]
        if C.rate1_cw2pr_st_idx > 0:
            pbv["RATE1_CW2PR_ST"][cols, lev] = V("rate1cld")[cols, lev]
        log("rate1ord=rate1cld")
        # 2312-2338
        K("micro_macro_feedback",
          {"ncol": ncol, "top_lev": top, "cpair": C.cpair, "vtrmc": V("vtrmc"), "tlat": V("tlat"),
           "qvlat": V("qvlat"), "qcten": V("qcten"), "qiten": V("qiten"), "ncten": V("ncten"),
           "niten": V("niten"), "alst_mic": alst_mic, "cmeliq": pbv["CMELIQ"],
           "cmeiout": V("cmeiout"), "ast": pbv["AST"], "prect": V("prect"), "preci": V("preci")},
          outputs={"wsedl": pbv["WSEDL"], "cc_t": pbv["CC_T"], "cc_qv": pbv["CC_qv"],
                   "cc_ql": pbv["CC_ql"], "cc_qi": pbv["CC_qi"], "cc_nl": pbv["CC_nl"],
                   "cc_ni": pbv["CC_ni"], "cc_qlst": pbv["CC_qlst"], "qme": pbv["QME"],
                   "icecldf": None, "liqcldf": None, "prec_pcw": pbv["PREC_PCW"],
                   "snow_pcw": pbv["SNOW_PCW"], "prec_sed": pbv["PREC_SED"],
                   "snow_sed": pbv["SNOW_SED"], "prec_str": pbv["PREC_STR"],
                   "snow_str": pbv["SNOW_STR"]})
        log("micro_macro_feedback")
        # 2345-2350 [exact: zeroing]; icinc and icwnc are routine locals
        for name in ("icinc", "icwnc"):
            st.scratch[name][...] = 0.0
        for name in ("ICIWPST", "ICLWPST", "ICSWP", "CLDFSNOW"):
            pbv[name][...] = 0.0
        log("zero6")
        # 2352-2384
        K("micro_in_cloud_quantities",
          {"ncol": ncol, "top_lev": top, "ixcldliq": C.ixcldliq, "ixcldice": C.ixcldice,
           "ixnumliq": C.ixnumliq, "ixnumice": C.ixnumice, "mincld": C.mincld, "gravit": C.gravit,
           "q_loc": q_loc, "pmid_loc": S["state_loc_pmid"], "t_loc": S["state_loc_t"],
           "pdel_loc": S["state_loc_pdel"], "icecldf": None, "liqcldf": None, "ast": pbv["AST"],
           "cld": pbv["CLD"], "concld": pbv["CONCLD"], "qsout": V("qsout")},
          outputs={"icimrst": None, "icwmrst": None, "icinc": None, "icwnc": None,
                   "iciwpst": pbv["ICIWPST"], "iclwpst": pbv["ICLWPST"],
                   "cldfsnow": pbv["CLDFSNOW"], "icswp": pbv["ICSWP"]})
        log("micro_in_cloud_quantities")
        # 2387-2398: micro_mg_version > 1 -- dead; 2407-2464: subcolumns -- dead
        # 2467-2483: pointers into the buffer -- GRID_PBUF, nothing to do
        log("grid_aliases")
        # 2485-2519 [exact: copies]
        pbv["am_evp_st"][...] = V("am_evp_st")                 # mg1.0
        pbv["evpsnow_st"][...] = V("evapsnow")
        for name in GRID_COPIES:
            st.scratch[f"{name}_grid"][..., 0] = V(name)
        st.scratch["cld_grid"][..., 0] = pbv["CLD"]
        for name in ("icwmrst", "icimrst", "liqcldf", "icecldf", "icwnc", "icinc"):
            st.scratch[f"{name}_grid"][..., 0] = L[name]
        st.scratch["pdel_grid"][..., 0] = S["state_loc_pdel"]
        st.scratch["nc_grid"][..., 0] = q_loc[:, :, C.ixnumliq - 1]
        st.scratch["ni_grid"][..., 0] = q_loc[:, :, C.ixnumice - 1]
        log("grid_copies")

        # 2567-2700: the water tracers
        if C.trace_water:
            self._water_tracers(st, lchnk, ncol, alst_mic, aist_mic, V, K, G)

        # 2712-2718: the host state again, not the copy the substep updated
        K("micro_air_density", {"ncol": ncol, "top_lev": top, "rair": C.rair,
                                "pmid": S["state_pmid"], "t": S["state_t"]},
          outputs={"rho": None})
        log("micro_air_density")
        st.scratch["rho_grid"][..., 0] = L["rho"]              # 2717 [exact: a copy]
        log("rho_grid=rho")
        # 2721-2759
        K("micro_liquid_size",
          {"ngrdcol": ngrdcol, "top_lev": top, "mincld": C.mincld, "qsmall": C.qsmall},
          outputs={"mu_grid": G("mu_grid"), "lambdac_grid": G("lambdac_grid"), "rel_fn_grid": None,
                   "ncic_grid": None, "rel_grid": G("rel_grid")})
        log("micro_liquid_size")
        # 2762-2854
        K("micro_ice_and_precip_size",
          {"ngrdcol": ngrdcol, "top_lev": top, "mincld": C.mincld, "qsmall": C.qsmall,
           "ast_grid": G("ast_grid")},
          outputs={"drout2_grid": None, "reff_rain_grid": None, "des_grid": G("des_grid"),
                   "dsout2_grid": None, "reff_snow_grid": None, "rei_grid": G("rei_grid"),
                   "niic_grid": None, "dei_grid": G("dei_grid"), "mu_grid": G("mu_grid"),
                   "lambdac_grid": G("lambdac_grid"), "mgreffrain_grid": G("mgreffrain_grid"),
                   "mgreffsnow_grid": G("mgreffsnow_grid")})
        log("micro_ice_and_precip_size")
        # 2864-2927
        K("micro_precip_efficiency",
          {"ngrdcol": ngrdcol, "top_lev": top, "gravit": C.gravit, "rhoh2o": C.rhoh2o,
           "cmeliq_grid": G("cmeliq_grid"), "prec_str_grid": G("prec_str_grid"),
           "iclwpst_grid": G("iclwpst_grid")},
          outputs={"acgcme_grid": G("acgcme_grid"), "acprecl_grid": G("acprecl_grid"),
                   "acnum_grid": G("acnum_grid"), "tgliqwp_grid": None, "tgcmeliq_grid": None,
                   "pe_grid": None, "tpr_grid": None, "pefrac_grid": None})
        log("micro_precip_efficiency")
        # 2933-2964
        K("micro_autoconversion_ratio", {"ngrdcol": ngrdcol, "top_lev": top, "gravit": C.gravit},
          outputs={"vprao_grid": None, "vprco_grid": None, "racau_grid": None, "cdnumc_grid": None})
        log("micro_autoconversion_ratio")
        # 2967-3024
        K("micro_effective_outputs",
          {"ngrdcol": ngrdcol, "top_lev": top, "rel_grid": G("rel_grid"), "rei_grid": G("rei_grid"),
           "nevapr_grid": G("nevapr_grid")},
          outputs={"evpsnow_st_grid": G("evpsnow_st_grid"), "efcout_grid": None, "efiout_grid": None,
                   "ncout_grid": None, "niout_grid": None, "freql_grid": None, "freqi_grid": None,
                   "icwmrst_grid_out": None, "icimrst_grid_out": None,
                   "evprain_st_grid": G("evprain_st_grid"), "fcti_grid": None, "fctl_grid": None,
                   "ctrel_grid": None, "ctrei_grid": None, "ctnl_grid": None, "ctni_grid": None})
        log("micro_effective_outputs")
        # 3027-3030 [exact: copies]
        for idx, field, name in ((C.qrain_idx, "QRAIN", "qrout_grid"), (C.qsnow_idx, "QSNOW", "qsout_grid"),
                                 (C.nrain_idx, "NRAIN", "nrout_grid"), (C.nsnow_idx, "NSNOW", "nsout_grid")):
            if idx > 0:
                pbv[field][...] = L[name]
        log("grid_ptr=grid")
        # 3037-3060
        K("micro_history_tendencies", {"ngrdcol": ngrdcol, "top_lev": top},
          outputs={"mpdw2v": None, "mpdw2i": None, "mpdw2p": None, "mpdi2v": None,
                   "mpdi2w": None, "mpdi2p": None})
        log("micro_history_tendencies")
        for name, key in HISTORY_TENDENCIES:
            H.outfld(name, L[key], pcols, lchnk)
        # 3063-3103: psetcols is pcols; avg_subcol_field is off
        for name, key in HISTORY_MG:
            array = L[key] if key in ("iclwpi", "iciwpi", "mnuccdohet") else V(key)
            H.outfld(name, array, pcols, lchnk)
        # 3120-3176
        for name, key in HISTORY_GRID:
            H.outfld(name, G(key), pcols, lchnk)
        log("outfld*")
        # 3179
        H.output_precip(lchnk); log("wtrc_output_precip")
        # 3182
        H.end(lchnk); log("physics_state_dealloc")

    def _core_call(self, st: StageRuntime, lchnk: int, ncol: int, dt: float) -> None:
        """2087-2209: the core inside the lifted section, or a model in its place.

        With nothing in the slot the original runs where it always did --
        inside the lifted section, over every packed column at once, the
        call the driver made.  A model is handed the packed inputs as
        ``(mgncol, nlev)`` batches and answers every packed output, which is
        written back into the packed storage the unpack reads.  Either way
        the trace, when on, hashes every argument before and after.
        """

        H, C = st.handles, st.constants
        model = self.kernels.get(CORE)
        if model is None and self.use_standalone_core:
            model = self.kernels[CORE] = self.standalone_core()
        owns = model is not None
        if getattr(st, "core_owner", None) != owns:
            H.set_core_owner(owns)
            st.core_owner = owns
        if not owns and st.trace is None:
            H.core(lchnk)                      # 2087-2206, the original
            self.calls.append(CORE)
            return
        names = list(PACKED_INPUTS)
        if not C.do_cldice:
            names += PACKED_INPUTS_NO_CLDICE
        if C.use_hetfrz_classnuc:
            names += PACKED_INPUTS_HETFRZ
        inputs = {name: H.view(lchnk, VIEW[f"packed_{name}"]) for name in names}
        outputs = {name: H.view(lchnk, VIEW[f"packed_{name}"]) for name in PACKED_OUTPUTS}
        mgncol = int(inputs["t"].shape[0])
        # 2087-2095: what the core reads beside the packed columns
        self._scalars = {"microp_uniform": int(C.microp_uniform),
                         "do_cldice": int(C.do_cldice),
                         "deltatin": dt / C.num_steps}
        inputs.update(self._scalars)
        st.swappable_kernel(CORE, inputs, outputs=outputs, ncol=mgncol, lchnk=lchnk, dt=dt,
                            kernel=self._chunk_model(model, mgncol) if owns else None,
                            original=lambda: H.core(lchnk))
        self.calls.append(CORE)

    def _discarded_outputs(self) -> dict[str, np.ndarray]:
        """The two intent(inout) outputs the driver hands to locals it discards.

        The routine zeroes both before writing them, so what goes in cannot
        be read; zeros stand in.  Their names come from the reviewed spec.
        """

        if self._discarded is None:
            spec = self._function().spec
            self._discarded = {item.name: np.zeros(item.public_extent(spec.dimensions))
                               for item in spec.arguments
                               if item.role == "inout" and item.name not in PACKED_OUTPUTS}
        return self._discarded

    def _chunk_model(self, model, rows: int):
        """The installed model, answering a chunk's packed columns in one call.

        Three shapes of model reach the slot.  :meth:`standalone_core`'s
        callable already takes the packed batch under the packed names and
        passes straight through.  A kernel with a ``batched`` method (a
        trained surrogate) is given the batch under the routine's own dummy
        names, with the discarded outputs and the scalars beside them.  A
        callable that only knows one column, as a notebook's own function
        does, is walked column by column and its answers stacked -- the
        contract :meth:`micro_mg_tend` has always offered.
        """

        if getattr(model, "takes_packed_batch", False):
            return model

        def call(batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
            packed = {name: np.asarray(value) for name, value in batch.items()}
            if hasattr(model, "batched"):
                columns = {PACKED_TO_DUMMY.get(name, name): value for name, value in packed.items()}
                for name, zeros in self._discarded_outputs().items():
                    columns[name] = np.broadcast_to(zeros, (rows, *zeros.shape))
                answer = dict(model.batched(columns))
                return {name: np.asarray(answer[PACKED_TO_DUMMY.get(name, name)])
                        for name in PACKED_OUTPUTS}
            pieces = [self._column({name: (value if name in PACKED_SCALARS else value[row])
                                    for name, value in packed.items()})
                      for row in range(rows)]
            return {name: np.stack([np.asarray(piece.outputs[name]) for piece in pieces])
                    for name in PACKED_OUTPUTS}

        return call

    def _column(self, column: Mapping[str, Any]):
        """One packed column, under the names the core's own dummies use."""

        renamed = {PACKED_TO_DUMMY.get(name, name): value for name, value in column.items()}
        renamed.update(self._discarded_outputs())
        answer = self.micro_mg_tend({**renamed, **self._scalars})
        values = {**answer.outputs, **answer.updated_inputs}
        return FunctionResult(
            outputs={name: values[PACKED_TO_DUMMY.get(name, name)] for name in PACKED_OUTPUTS},
            updated_inputs={})

    def _water_tracers(self, st, lchnk, ncol, alst_mic, aist_mic, V, K, G) -> None:
        """2567-2700 under trace_water, without subcolumns."""

        C, H, L = st.constants, st.handles, st.local
        top = C.top_lev
        log = self.calls.append
        # 2586-2600: pointers to the MG outputs -- the views themselves
        preo, prdso, meltso = V("preo"), V("prdso"), V("meltso")
        mnuccro, pracso = V("mnuccro"), V("pracso")
        qcsedten, qisedten = V("qcsedten"), V("qisedten")
        log("wtrc_aliases")

        def init(rates: str) -> None:
            K("wtrc_init_rates", {"top_lev": top}, outputs={rates: None},
              fields={"top_lev": "macro.top_lev", rates: "macro.process_rates"})
            log("wtrc_init_rates")

        def add(rates: str, isrc: int, idst: int, rtype: int, rate: np.ndarray) -> None:
            K("wtrc_add_rates",
              {"ncol": ncol, "top_lev": top, "isrctype": isrc, "idsttype": idst, "rtype": rtype,
               "rate": rate},
              outputs={}, fields={rates: "macro.process_rates", **WTRC_FIELDS})

        def add_sum(rates: str, isrc: int, idst: int, rtype: int, terms) -> None:
            H.wtrc_add_sum(L[rates], ncol, top, isrc, idst, rtype, terms)

        # 2604
        init("pre_rates_grid")
        # 2607-2626
        K("micro_split_signs", {"ncol": ncol, "top_lev": top, "meltso": meltso, "meltso_grid": meltso},
          outputs={"cmeiout_grid": None, "pcmei_grid": None, "ncmei_grid": None,
                   "pmelts_grid": None, "nmelts_grid": None})
        log("micro_split_signs")
        # 2630-2660
        pre = "pre_rates_grid"
        add(pre, IWTVAP, IWTICE, IWTVAP, L["pcmei_grid"])
        add(pre, IWTVAP, IWTICE, IWTICE, L["ncmei_grid"])
        add(pre, IWTVAP, IWTSTRAIN, IWTSTRAIN, preo)
        add(pre, IWTVAP, IWTSTSNOW, IWTSTSNOW, prdso)
        add_sum(pre, IWTLIQ, IWTICE, IWTLIQ, [G("mnuccco_grid"), G("mnuccto_grid"), G("msacwio_grid")])
        add_sum(pre, IWTLIQ, IWTSTRAIN, IWTLIQ, [G("prao_grid"), G("prco_grid")])
        add(pre, IWTLIQ, IWTSTSNOW, IWTLIQ, G("psacwso_grid"))
        add(pre, IWTLIQ, IWTLIQ, IWTLIQ, G("bergo_grid"))
        add(pre, IWTICE, IWTICE, IWTICE, G("bergso_grid"))
        add_sum(pre, IWTICE, IWTSTSNOW, IWTICE, [G("praio_grid"), G("prcio_grid")])
        add_sum(pre, IWTSTRAIN, IWTSTSNOW, IWTSTRAIN, [pracso, mnuccro])
        log("wtrc_add_rates*")
        # 2665-2667 [exact: zeroing and copies]
        sed = L["sed_rates_grid"]
        sed[:, top - 1:, :] = 0.0
        sed[:, top - 1:, IWTLIQ - 1] = qcsedten[:, top - 1:]
        sed[:, top - 1:, IWTICE - 1] = qisedten[:, top - 1:]
        log("sed_rates")
        # 2670-2692
        init("post_rates_grid")
        post = "post_rates_grid"
        add(post, IWTVAP, IWTLIQ, IWTVAP, G("qcreso_grid"))
        add(post, IWTVAP, IWTICE, IWTVAP, G("qireso_grid"))
        add(post, IWTLIQ, IWTICE, IWTLIQ, G("homoo_grid"))
        add(post, IWTICE, IWTLIQ, IWTICE, G("melto_grid"))
        log("wtrc_add_rates*")
        # 2695-2698
        H.wtrc_apply(lchnk, L["pre_rates_grid"], sed, L["post_rates_grid"], alst_mic, aist_mic)
        log("wtrc_apply_rates")


__all__ = ["CORE", "CONFIGURATION", "GRID_COPIES", "GRID_PBUF", "HISTORY_GRID", "HISTORY_MG",
           "HISTORY_TENDENCIES", "INPUT", "KERNELS", "Microphysics", "PACKED_INPUTS",
           "PACKED_OUTPUTS", "PACKED_SCALARS", "PACKED_TO_DUMMY", "RETURNED", "SEQUENCE",
           "VIEW"]
