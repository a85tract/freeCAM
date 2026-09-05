"""Aerosol activation -- ``microp_aero_run`` -- as a Python class.

``microp_aero_run`` (``microp_aero.F90:345-713``) is what tphysbc calls
before the microphysics: it reads the modal aerosol fields, forms the
sub-grid vertical velocity from the PBL's turbulent kinetic energy, and
calls three cores -- ice nucleation, droplet activation and, when it is
on, classical heterogeneous freezing -- leaving the activation tendencies
in the physics buffer for the microphysics to pack.

:class:`MicropAero` is that routine statement for statement, with every
floating-point number still Fortran's: the twenty live arithmetic
statements through the lifted ``aero_*`` kernels
(``tools/generate_pi_cam_aero_kernels.py``) and everything that takes a
derived type through ``pycam_aero_handles``.  A sub-walk of
:class:`CloudMacroMicrophysics`; it has no workflow action of its own.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..pi_cam.errors import PICAMConfigurationError
from ..pi_cam.pbuf import PBuf, load_pbuf_table
from .image import module_view
from freecam.pi_cam.tables import load_table
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

#: The seven physics-buffer fields the routine reads under this
#: configuration (tools/generate_pi_cam_pbuf_table.py).
PBUF_TABLE = REPO / "native/pi_cam/pbuf_fields_aero.yaml"

#: pycam_aero_handles.F90 view codes.  A test keeps this table equal to the
#: Fortran one.
VIEW = {
    "state_t": 1, "state_q": 2, "state_pmid": 3,
    "ptend_aero_s": 11, "ptend_aero_q": 12,
    "nctend_mixnuc": 21, "factnum": 22,
    "num_coarse": 31, "coarse_dust": 32, "coarse_nacl": 33,
}

#: The direct kernels tend() runs (tools/generate_pi_cam_aero_kernels.py).
KERNELS = ("aero_initial_bins", "aero_air_density", "aero_subgrid_velocity",
           "aero_liquid_fraction", "aero_cloud_fraction_split",
           "aero_npccn_from_mixnuc", "aero_contact_freezing")

# microp_aero.F90:70,73 -- parameters, so not symbols in the image.  A test
# pins them against the pinned source.
QSMALL, MINCLD = 1.e-18, 0.0001

#: The order in which the Fortran routine does things under the admitted
#: configuration; ``tend`` follows it per chunk and a test compares the two.
SEQUENCE = (
    "pbuf_get_field*", "rad_cnst_get_info", "aero_initial_bins",
    "hetfrz_classnuc_cam_save_cbaero", "aero_air_density",
    "rad_cnst_get_mode_num", "rad_cnst_get_aer_mmr*", "aero_subgrid_velocity", "outfld*",
    "nucleate_ice_cam_calc", "aero_liquid_fraction", "aero_cloud_fraction_split", "outfld",
    "dropmixnuc", "aero_npccn_from_mixnuc", "aero_contact_freezing",
    "hetfrz_classnuc_cam_calc", "end",
)


# -- the image, seen through ctypes ---------------------------------------------

_INT, _DBL = ctypes.c_int, ctypes.c_double
_P_DBL = ctypes.POINTER(ctypes.c_double)


class _AeroEntries(HostEntries):
    """The core entries plus the routine's own calls into CAM."""

    TABLE = {
        **CORE_ENTRIES,
        "nmodes": ("pycam_{prefix}_nmodes_v1", [], False),
        "begin": ("pycam_{prefix}_begin_v1", [_INT], False),
        "modal_fields": ("pycam_{prefix}_modal_fields_v1", [_INT, _INT, _INT, _INT], False),
        "save_cbaero": ("pycam_{prefix}_save_cbaero_v1", [], False),
        "nucleate_ice": ("pycam_{prefix}_nucleate_ice_v1", [_P_DBL], False),
        "dropmixnuc": ("pycam_{prefix}_dropmixnuc_v1",
                       [_DBL, _INT, _P_DBL, _P_DBL, _P_DBL], False),
        "hetfrz": ("pycam_{prefix}_hetfrz_v1", [_DBL], False),
        "end": ("pycam_{prefix}_end_v1", [], False),
    }


class _AeroHandles(HostServices):
    """CAM's host services, plus the calls only this routine makes."""

    def nmodes(self) -> int:
        """``rad_cnst_get_info(0, nmodes=nmodes)`` (441)."""

        value = self.e.nmodes()
        if value < 1:
            raise PICAMConfigurationError(f"rad_cnst_get_info returned {value} aerosol modes")
        return int(value)

    def begin(self, lchnk: int) -> None:
        _check(self.e.begin(lchnk), "aero begin")

    def modal_fields(self, dust_mode: int, salt_mode: int, dust: int, salt: int) -> None:
        """``rad_cnst_get_mode_num`` and the two ``rad_cnst_get_aer_mmr`` (473-477)."""

        _check(self.e.modal_fields(dust_mode, salt_mode, dust, salt), "rad_cnst_get_* (coarse mode)")

    def save_cbaero(self) -> None:
        _check(self.e.save_cbaero(), "hetfrz_classnuc_cam_save_cbaero")

    def nucleate_ice(self, wsubi: np.ndarray) -> None:
        _check(self.e.nucleate_ice(_ptr(wsubi)), "nucleate_ice_cam_calc")

    def dropmixnuc(self, dt: float, nmodes: int, wsub: np.ndarray, lcldn: np.ndarray,
                   lcldo: np.ndarray) -> None:
        _check(self.e.dropmixnuc(float(dt), nmodes, _ptr(wsub), _ptr(lcldn), _ptr(lcldo)),
               "dropmixnuc")

    def hetfrz(self, dt: float) -> None:
        _check(self.e.hetfrz(float(dt)), "hetfrz_classnuc_cam_calc")

    def end(self) -> None:
        _check(self.e.end(), "aero end")


# -- module constants --------------------------------------------------------------


@dataclass(frozen=True)
class _Constants:
    clim_modal_aero: bool
    micro_do_icesupersat: bool
    separate_dust: bool
    use_hetfrz_classnuc: bool
    use_preexisting_ice: bool
    eddy_scheme: str
    cldliq: int
    cldice: int
    mode_coarse_dst_idx: int
    mode_coarse_slt_idx: int
    coarse_dust_idx: int
    coarse_nacl_idx: int
    top_lev: int
    rair: float
    mincld: float
    qsmall: float

    @classmethod
    def read(cls, library: Any) -> "_Constants":
        def i(symbol):
            return int(module_view(library, symbol, "int32", ()))

        def b(symbol):
            return bool(int(module_view(library, symbol, "int32", ())))

        m = "microp_aero_mp_{}_".format
        return cls(
            clim_modal_aero=b(m("clim_modal_aero")),
            micro_do_icesupersat=b(m("micro_do_icesupersat")),
            separate_dust=b(m("separate_dust")),
            use_hetfrz_classnuc=b("phys_control_mp_use_hetfrz_classnuc_"),
            use_preexisting_ice=b("nucleate_ice_cam_mp_use_preexisting_ice_"),
            eddy_scheme=module_view(library, "microp_aero_mp_eddy_scheme_", "S16", ()
                                    ).item().decode("ascii").strip(),
            cldliq=i(m("cldliq_idx")), cldice=i(m("cldice_idx")),
            mode_coarse_dst_idx=i(m("mode_coarse_dst_idx")),
            mode_coarse_slt_idx=i(m("mode_coarse_slt_idx")),
            coarse_dust_idx=i(m("coarse_dust_idx")), coarse_nacl_idx=i(m("coarse_nacl_idx")),
            top_lev=i("ref_pres_mp_trop_cloud_top_lev_"),
            rair=float(module_view(library, "physconst_mp_rair_", "float64", ())),
            mincld=MINCLD, qsmall=QSMALL,
        )

    def refuse_unsupported(self) -> None:
        """The paths the admitted configuration never takes are not ported."""

        def refuse(what: str) -> None:
            raise PICAMConfigurationError(
                f"{what}; the Python aerosol activation does not carry that path")

        if not self.clim_modal_aero:
            refuse("clim_modal_aero is off (the bulk-aerosol path)")
        if self.micro_do_icesupersat:
            refuse("micro_do_icesupersat is on (the activation reads CLDO, not AST)")
        if self.eddy_scheme != "diag_TKE":
            refuse(f"eddy_scheme is {self.eddy_scheme!r}, not 'diag_TKE'")
        if self.mode_coarse_dst_idx < 1 or self.mode_coarse_slt_idx < 1:
            refuse("the coarse dust or sea-salt mode is not registered")
        if self.cldliq < 1 or self.cldice < 1:
            refuse("cloud liquid or ice is not a constituent")


# -- the class ---------------------------------------------------------------------


class MicropAero(NativeStage):
    """``microp_aero_run`` as Python; the three cores stay Fortran."""

    STAGE = "cam_run1.cloud_macro_microphysics"
    PREFIX = "aero"
    PROCESS_NAME = "microp_aero_run"
    TRACE_ENV = "FREECAM_AERO_TRACE"
    PROFILE_ENV = "FREECAM_AERO_PROFILE"

    KERNELS = KERNELS
    #: The routine's `wsub`/`wsubi` are locals of its own; every other array
    #: a kernel names is a buffer field or a handle view.
    EXTRA_SCRATCH = ()

    entries_class = _AeroEntries
    services_class = _AeroHandles

    # -- what the runtime asks of this stage -------------------------------------

    def read_constants(self, library: Any) -> "_Constants":
        return _Constants.read(library)

    def refuse_unsupported(self, constants: "_Constants") -> None:
        constants.refuse_unsupported()

    def extra_extents(self, constants: "_Constants") -> Mapping[str, int]:
        # the mode count the coarse-mode diameters carry, from the image
        return {"nmodes": self._nmodes}

    def build_pbuf(self, library: Any, runtime: StageRuntime) -> PBuf:

        symbols = [row["symbol"] for row in load_table(PBUF_TABLE)["fields"]]
        indices = {symbol: int(module_view(library, symbol, "int32", ())) for symbol in symbols}
        buffer = PBuf(library, load_pbuf_table(PBUF_TABLE, indices))
        lchnk, _ = runtime.native.chunks
        buffer.verify(int(lchnk[0]), pcols=runtime.pcols, pver=runtime.pver)
        return buffer

    def runtime(self, native: Any) -> StageRuntime:
        # the mode count has to be known before the scratch is sized, and it
        # is a call, not a symbol: ask the image once, through its own entry
        key = id(native.pool)
        if key not in self._runtimes:
            entries = self.entries_class(native.library, self.PREFIX)
            _check(entries.bind_hosts(), "pycam_aero_bind_hosts_v1")
            self._nmodes = self.services_class(entries, 0).nmodes()
        return super().runtime(native)

    # -- the transliteration -----------------------------------------------------

    def tend_chunk(self, st: StageRuntime, lchnk: int, ncol: int, index: int,
                   dt: float, nstep: int) -> None:
        """``microp_aero_run`` (345-713) under the admitted configuration.

        ``dt`` is the driver's ``deltatin``.  Line numbers are the pinned
        source's.
        """

        H, C, pb = st.handles, st.constants, st.pbuf
        L = st.local
        log = self.calls.append
        top = C.top_lev
        pcols = st.pcols

        def K(name, inputs, *, outputs):
            st.kernel_on_chunk(name, inputs, outputs=outputs, ncol=None)

        H.begin(lchnk)
        S = {name: H.view(lchnk, VIEW[name]) for name in ("state_t", "state_q", "state_pmid")}
        # 416-426, 430-442: the buffer fields, older time sample where the
        # source says so; micro_do_icesupersat is off, so `ast` is AST
        pbv = {name: pb.view(name, lchnk) for name in
               ("AST", "NPCCN", "NACON", "RNDST", "CLDO", "DGNUMWET", "tke")}
        ast, cldn = pbv["AST"], pbv["AST"]        # 420, 435: both are AST
        log("pbuf_get_field*")
        log("rad_cnst_get_info")                  # 441, asked once when the runtime was built

        # 449-457
        K("aero_initial_bins", {"ncol": ncol},
          outputs={"npccn": pbv["NPCCN"], "nacon": pbv["NACON"], "rndst": pbv["RNDST"]})
        log("aero_initial_bins")
        # 460-462
        if C.use_hetfrz_classnuc:
            H.save_cbaero(); log("hetfrz_classnuc_cam_save_cbaero")
        # 465-469
        K("aero_air_density",
          {"ncol": ncol, "top_lev": top, "rair": C.rair,
           "pmid": S["state_pmid"], "t": S["state_t"]},
          outputs={"rho": None})
        log("aero_air_density")
        # 473-477
        H.modal_fields(C.mode_coarse_dst_idx, C.mode_coarse_slt_idx,
                       C.coarse_dust_idx, C.coarse_nacl_idx)
        log("rad_cnst_get_mode_num"); log("rad_cnst_get_aer_mmr*")
        coarse = {name: H.view(lchnk, VIEW[name])
                  for name in ("num_coarse", "coarse_dust", "coarse_nacl")}
        # 501-549: eddy_scheme is diag_TKE (refused otherwise)
        K("aero_subgrid_velocity",
          {"ncol": ncol, "top_lev": top, "use_preexisting_ice": C.use_preexisting_ice,
           "tke": pbv["tke"]},
          outputs={"wsub": None, "wsubi": None})
        log("aero_subgrid_velocity")
        # 551-552
        H.outfld("WSUB", L["wsub"], pcols, lchnk)
        H.outfld("WSUBI", L["wsubi"], pcols, lchnk)
        log("outfld*")
        # 559
        H.nucleate_ice(L["wsubi"]); log("nucleate_ice_cam_calc")
        # 564-568
        K("aero_liquid_fraction", {"ncol": ncol, "top_lev": top, "mincld": C.mincld, "ast": ast},
          outputs={"lcldm": None})
        log("aero_liquid_fraction")
        # 578-588: clim_modal_aero (refused otherwise)
        K("aero_cloud_fraction_split",
          {"ncol": ncol, "top_lev": top, "qsmall": C.qsmall,
           "qc": S["state_q"][:, :, C.cldliq - 1], "qi": S["state_q"][:, :, C.cldice - 1],
           "cldn": cldn, "cldo": pbv["CLDO"]},
          outputs={"lcldn": None, "lcldo": None})
        log("aero_cloud_fraction_split")
        # 590
        H.outfld("LCLOUD", L["lcldn"], pcols, lchnk); log("outfld")
        # 592-594
        H.dropmixnuc(dt, self._nmodes, L["wsub"], L["lcldn"], L["lcldo"])
        log("dropmixnuc")
        # 596
        K("aero_npccn_from_mixnuc",
          {"ncol": ncol, "nctend_mixnuc": H.view(lchnk, VIEW["nctend_mixnuc"])},
          outputs={"npccn": pbv["NPCCN"]})
        log("aero_npccn_from_mixnuc")
        # 632-684
        K("aero_contact_freezing",
          {"ncol": ncol, "top_lev": top, "separate_dust": C.separate_dust,
           "t": S["state_t"], "coarse_dust": coarse["coarse_dust"],
           "coarse_nacl": coarse["coarse_nacl"], "num_coarse": coarse["num_coarse"],
           "rho": None,
           # 662 indexes one plane of the mode-resolved diameters
           "dgnumwet_coarse": pbv["DGNUMWET"][:, :, C.mode_coarse_dst_idx - 1]},
          outputs={"nacon": pbv["NACON"], "rndst": pbv["RNDST"]})
        log("aero_contact_freezing")
        # 701-705
        if C.use_hetfrz_classnuc:
            H.hetfrz(dt); log("hetfrz_classnuc_cam_calc")
        H.end(); log("end")


__all__ = ["KERNELS", "MicropAero", "MINCLD", "QSMALL", "SEQUENCE", "VIEW"]
