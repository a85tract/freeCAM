"""Macrophysics: the driver's statements, in the driver's order, from Python."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from freecam.physics import macrophysics as M
from freecam.physics.macrophysics import Macrophysics, SEQUENCE, VIEW, FORCING

REPO = Path(__file__).resolve().parents[2]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/macrop_driver.F90"
HANDLES = REPO / "native/pi_cam/support/pycam_macro_handles.F90"
BOUNDARY = REPO / "native/pi_cam/control_patches/0039-macro-tend-boundary.patch"

pinned = pytest.mark.skipif(not PINNED.is_file(), reason="the pinned iCESM submodule is not checked out")


# -- tables that mirror Fortran --------------------------------------------------


def test_view_codes_equal_the_handles_module_s_table() -> None:
    fortran = {m.group(1).removeprefix("view_"): int(m.group(2))
               for m in re.finditer(r"parameter, public :: (view_\w+) = (\d+)", HANDLES.read_text())}
    assert VIEW == fortran
    records = dict(re.findall(r"parameter, public :: (record_\w+) = (\d+)", HANDLES.read_text()))
    assert int(records["record_ptend_loc"]) == M.RECORD_PTEND_LOC
    assert int(records["record_ptend"]) == M.RECORD_PTEND


@pinned
def test_water_type_constants_equal_the_pinned_source() -> None:
    types = (REPO / "external/iCESM1.3.1_fzhu/cime/src/share/util/water_types.F90").read_text()
    values = {m.group(1): int(m.group(2)) for m in re.finditer(r"parameter, public :: (\w+)\s*=\s*(\d+)", types)}
    assert (values["iwtvap"], values["iwtliq"], values["iwtice"], values["pwtype"]) == (
        M.IWTVAP, M.IWTLIQ, M.IWTICE, M.PWTYPE)
    tracers = (REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/water_tracer_vars.F90").read_text()
    assert int(re.search(r"WTRC_MAX_CNST\s*=\s*(\d+)", tracers).group(1)) == M.WTRC_MAX_CNST


@pinned
def test_the_three_physconst_parameters_equal_the_pinned_shr_const_literals() -> None:
    """cpair, latice and latvap are parameters: no symbol, so they are pinned.

    Gate B2 failed once on `image has no symbol physconst_mp_cpair_`.
    """

    text = (REPO / "external/iCESM1.3.1_fzhu/cime/src/share/util/shr_const_mod.F90").read_text()
    literal = {m.group(1): float(m.group(2).replace("_R8", "").replace("_r8", ""))
               for m in re.finditer(r"SHR_CONST_(CPDAIR|LATICE|LATVAP)\s*=\s*([0-9.eE+-]+_[Rr]8)", text)}
    assert literal == {"CPDAIR": M.CPAIR, "LATICE": M.LATICE, "LATVAP": M.LATVAP}
    # and physconst really does declare them as parameters, not variables
    physconst = next(REPO.glob("external/iCESM1.3.1_fzhu/components/cam/src/**/physconst.F90")).read_text()
    for name in ("cpair", "latice", "latvap"):
        assert re.search(rf"parameter\s*::\s*{name}\s*=\s*shr_const_", physconst), name
    for name in ("gravit", "tmelt"):
        assert not re.search(rf"parameter\s*::\s*{name}\s*=", physconst), f"{name} is a parameter now; read it that way"


def test_forcing_codes_equal_the_boundary_accessor_s_cases() -> None:
    added = "\n".join(l[1:] for l in BOUNDARY.read_text().splitlines() if l.startswith("+"))
    cases = re.findall(r"case \((\d)\)\n\s*call pycam_macro_address\d\(pycesm_bc_(\w+)\(", added)
    assert {name: int(code) for code, name in cases} == FORCING


# -- the sequence -------------------------------------------------------------------


def _fortran_sequence() -> list[str]:
    """The driver's calls, 612-1224, as tend() names them, runs of like calls collapsed."""

    lines = PINNED.read_text().splitlines()
    names: list[str] = []
    carved = {  # first line of each lifted block -> its routine
        706: "macrop_detrain_partition", 895: "macrop_clear_fraction", 978: "macrop_advective_forcing",
        1051: "macrop_kernel_to_ptend", 1107: "macrop_tracer_rate_split", 1182: "macrop_cloud_mixing_ratio",
        1208: "macrop_save_equilibrium",
    }
    skip = False
    for number in range(612, 1225):
        text = lines[number - 1].split("!")[0].strip()
        if text.startswith("if (micro_do_icesupersat)"):   # refused at attach
            skip = True
        if skip:
            if text.startswith("endif") or text.startswith("end if"):
                skip = False
            continue
        if number in carved:
            names.append(carved[number])
            continue
        m = re.match(r"call (\w+)\(", text)
        if not m:
            continue
        name = m.group(1)
        if name in ("t_startf", "t_stopf", "endrun"):
            continue
        if name == "physics_ptend_init":
            label = re.search(r"'(\w+)'", text).group(1)
            name = f"physics_ptend_init:{label}"
        if name == "pbuf_get_field" and number > 860:
            continue                 # the six UNICON pointers: unregistered here, zeros in Python
        if name in ("outfld", "pbuf_get_field", "wtrc_add_rates"):
            if names and names[-1] == name + "*":
                continue
            names.append(name + "*")
            continue
        names.append(name)
    # a lone outfld is written without the star
    return [n if not (n == "outfld*" and i == len(names) - 2) else "outfld" for i, n in enumerate(names)]


@pinned
def test_the_sequence_is_the_fortran_driver_s_call_order() -> None:
    assert list(SEQUENCE) == _fortran_sequence()


# -- tend() end to end, against a fake image -------------------------------------


class _Lib:
    """A fake image: every entry succeeds, every view is a real array."""

    PCOLS, PVER, PVERP, PCNST = 16, 30, 31, 57

    def __init__(self) -> None:
        self.views: dict[tuple[int, int], np.ndarray] = {}
        self.calls: list[str] = []
        self.owner = 0

    def _array(self, key, shape):
        return self.views.setdefault(key, np.zeros(shape, order="F"))

    def _fill(self, ptr, ndims, extents, array):
        ptr._obj.value = array.ctypes.data
        ndims._obj.value = array.ndim
        for i, e in enumerate(array.shape):
            extents[i] = e

    def __getattr__(self, name):
        if not name.startswith("pycam_"):
            raise AttributeError(name)
        lib = self

        def entry(*args):
            lib.calls.append(name)
            if name == "pycam_macro_view_v1":
                lchnk, code, ptr, ndims, extents = args
                shape = {1: (16, 30), 2: (16, 30, 57), 3: (16, 30), 4: (16, 30), 5: (16, 31), 6: (16, 30),
                         7: (16,), 11: (16, 30), 12: (16, 30, 57), 21: (16, 30), 22: (16, 30, 57),
                         31: (16,), 32: (16,), 33: (16, 30, 7, 7, 7)}[code]
                lib._fill(ptr, ndims, extents, lib._array((lchnk, code), shape))
            elif name == "pycam_macro_forcing_v1":
                lchnk, code, ptr, ndims, extents = args
                shape = {1: (16, 30), 2: (16, 31), 3: (16, 31), 4: (16, 30), 5: (16, 30), 6: (16,), 7: (16, 30, 4)}[code]
                lib._fill(ptr, ndims, extents, lib._array((lchnk, 100 + code), shape))
            elif name == "pycam_macro_set_owner_v1":
                lib.owner = args[0]
            elif name == "pycam_pbuf_field_v1":
                lchnk, index, sliced, ptr, extents = args
                array = lib._array((lchnk, 200 + index), (16, 30))
                ptr._obj.value = array.ctypes.data
                extents[0], extents[1] = array.shape
            return 0
        return entry


class _Native:
    def __init__(self, lib):
        self.library = lib
        self.fcomm = 0
        self.pool = {"grid.chunk_id": np.array([1540, 1541]), "grid.chunk_ncols": np.array([14, 13])}
        self.pool_dims = {"pcols": 16, "pver": 30, "pverp": 31, "pcnst": 57}
        self.kernels: list[str] = []
        import yaml
        text = yaml.safe_load((REPO / "native/pi_cam/direct_kernels_macrophysics.yaml").read_text())
        self._args = {k["name"]: k["arguments"] for k in text["kernels"]}
        self._args["cloud_fraction_fice"] = [
            {"field": "cldfrc_fice.ncol", "dtype": "int32", "rank": 1, "extents": ["chunks"]},
            {"field": "cldfrc_fice.t", "dtype": "float64", "rank": 3, "extents": ["pcols", "pver", "chunks"]},
            {"field": "cldfrc_fice.fice", "dtype": "float64", "rank": 3, "extents": ["pcols", "pver", "chunks"]},
            {"field": "cldfrc_fice.fsnow", "dtype": "float64", "rank": 3, "extents": ["pcols", "pver", "chunks"]},
        ]

    @property
    def chunks(self):
        return self.pool["grid.chunk_id"], self.pool["grid.chunk_ncols"]

    def kernel_arguments(self, name):
        return tuple(self._args[name])

    def run_kernel(self, name, arrays):
        self.kernels.append(name)
        for a in self._args[name]:
            assert a["field"] in arrays, (name, a["field"])
            assert arrays[a["field"]].flags.f_contiguous
            assert arrays[a["field"]].shape[-1] == 1


class _Pool(dict):
    @property
    def dimensions(self):
        return {"pcols": 16, "pver": 30, "pverp": 31, "pcnst": 57, "chunks": 2}


class _Context:
    def __init__(self, native):
        self.native = native
        self.timestep_seconds = 1800
        self.step = 5


@pytest.fixture
def fake(monkeypatch):
    lib = _Lib()
    native = _Native(lib)
    pool = _Pool(native.pool)
    for name in ("landfrac", "ocnfrac", "snowhland", "ts", "sst"):
        pool[f"cam_in.{name}"] = np.zeros((16, 2), order="F")
    native.pool = pool
    constants = M._Constants(
        top_lev=1, ixcldliq=2, ixcldice=3, ixnumliq=4, ixnumice=5, do_cldice=True, do_cldliq=True,
        do_detrain=True, micro_do_icesupersat=False, use_shfrc=True, cpair=1004.64, latice=3.337e5,
        latvap=2.501e6, gravit=9.80616, tmelt=273.15, trace_water=True, wtrc_detrain_in_macrop=True,
        wtrc_nwset=4, wtrc_ncnst=12,
        wtrc_iatype=np.tile(np.arange(6, 13)[None, :], (700, 1)).astype(np.int32),
        wtrc_indices=np.arange(6, 706).astype(np.int32),
    )
    monkeypatch.setattr(M._Constants, "read", classmethod(lambda cls, library: constants))
    monkeypatch.setattr(M, "module_view", lambda library, symbol, dtype, shape: np.array(3, dtype=np.int32))
    M._RankState._cache.clear()
    return native


def test_tend_walks_the_driver_in_its_order_on_every_chunk(fake) -> None:
    scheme = Macrophysics()
    scheme.tend(None, _Context(fake))
    per_chunk = len(SEQUENCE)
    assert len(scheme.calls) == 2 * per_chunk
    assert scheme.calls[:per_chunk] == list(SEQUENCE)
    assert scheme.calls[per_chunk:] == list(SEQUENCE)
    # every kernel ran once per chunk, all on the native path
    kernels = [k for k in fake.kernels]
    assert kernels.count("mmacro_pcond") == 2
    assert kernels.count("wtrc_add_rates") == 10
    assert fake.library.owner == 1


def test_a_model_in_the_kernel_s_place_sees_live_columns_and_must_answer_all_23(fake) -> None:
    seen = {}

    def model(batch):
        seen.update({k: np.asarray(v).shape for k, v in batch.items()})
        return {name: np.zeros((batch["t0"].shape[0], 30)) for name in (
            "s_tendout", "qv_tendout", "ql_tendout", "qi_tendout", "nl_tendout", "ni_tendout", "qme",
            "qvadj", "qladj", "qiadj", "qllim", "qilim", "cld", "al_st_star", "ai_st_star", "ql_st_star",
            "qi_st_star", "t0", "qv0", "ql0", "qi0", "nl0", "ni0")}

    scheme = Macrophysics(kernel=model)
    scheme.tend(None, _Context(fake))
    assert seen["t0"] == (13, 30) and seen["landfrac"] == (13,)      # the last chunk has 13 columns
    assert "mmacro_pcond" not in fake.kernels                          # the original was not called

    def partial(batch):
        return {"cld": np.zeros((batch["t0"].shape[0], 30))}

    M._RankState._cache.clear()
    with pytest.raises(M.PhysicsError, match="missing"):
        Macrophysics(kernel=partial).tend(None, _Context(fake))


def test_the_installed_process_is_native_and_owns_no_pool_field() -> None:
    process = M._MacroTendProcess(Macrophysics())
    assert process.native is True
    assert process.reads == () and process.writes == ()
    assert process.transactional is False
    assert process.name == "macro_tend"


def test_attach_swaps_the_stage_for_its_halves_and_sits_between_them() -> None:
    class Action:
        def __init__(self): self.enabled = None
        def enable(self, **_): self.enabled = True
        def disable(self, **_): self.enabled = False

    class Workflow:
        def __init__(self):
            self.items = {M.STAGE: Action(), M.FIRST_HALF: Action(), M.SECOND_HALF: Action()}
            self.inserted = []
        def process(self, name): return self.items[name]
        def insert_after(self, anchor, process):
            self.inserted.append((anchor, process)); return process

    class Run:
        workflow = Workflow()

    scheme = Macrophysics()
    handle = scheme.attach(Run)
    assert Run.workflow.items[M.STAGE].enabled is False
    assert Run.workflow.items[M.FIRST_HALF].enabled is True
    assert Run.workflow.items[M.SECOND_HALF].enabled is True
    assert Run.workflow.inserted == [(M.FIRST_HALF, handle)]
    assert isinstance(handle, M._MacroTendProcess)


def test_tend_refuses_to_run_as_an_ordinary_process() -> None:
    class Context:
        native = None

    with pytest.raises(M.PhysicsError, match="native"):
        Macrophysics().tend(None, Context())
