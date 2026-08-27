"""MicropAero: microp_aero_run as Python, checked against the pinned source.

The generated modules are what the generators write; the class binds what
its handles module offers; the carve drops exactly the arms this
configuration never takes; ``tend``'s per-chunk sequence is the routine's
live statements; and a fake image walks two chunks.
"""

from __future__ import annotations

import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import freecam.physics.microp_aero as A  # noqa: E402
from freecam.physics.microp_aero import KERNELS, SEQUENCE, VIEW, MicropAero  # noqa: E402
from freecam.pi_cam.errors import PICAMConfigurationError  # noqa: E402

KERNEL_MODULE = REPO / "native/pi_cam/support/pycam_aero_kernels.F90"
HANDLES = REPO / "native/pi_cam/support/pycam_aero_handles.F90"
SOURCE = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/microp_aero.F90"
LINES = SOURCE.read_text().splitlines()


# -- the generated modules -------------------------------------------------------


def test_the_committed_modules_are_what_the_generators_write() -> None:
    import generate_pi_cam_aero_handles as handles
    import generate_pi_cam_aero_kernels as kernels

    assert kernels.render_module() == KERNEL_MODULE.read_text()
    assert handles.render_module() == HANDLES.read_text()


def test_every_carved_line_is_the_pinned_source_s() -> None:
    """Each block's body is the source's text; only the dropped lines differ."""

    import generate_pi_cam_aero_kernels as gen

    module = KERNEL_MODULE.read_text()
    for block in gen.BLOCKS:
        body = re.search(rf"subroutine {block.name}\(.*?\n\n(.*?)\n\n  end subroutine",
                         module, re.S)
        assert body, block.name
        carved = [line.strip() for line in body.group(1).splitlines() if line.strip()]
        pinned = [LINES[n - 1].strip() for n in range(block.first, block.last + 1)
                  if n not in block.skip and LINES[n - 1].strip()]
        declared = {line for line in carved if re.match(
            r"(integer|real\(r8\)|logical)[ ,]", line)}
        assert [line for line in carved if line not in declared] == pinned, block.name


def test_the_drops_are_exactly_the_arms_this_configuration_never_takes() -> None:
    import generate_pi_cam_aero_kernels as gen

    dropped = {block.name: sorted(block.skip) for block in gen.BLOCKS if block.skip}
    assert set(dropped) == {"aero_subgrid_velocity", "aero_contact_freezing"}
    # the velocity block drops the select-case scaffolding, the eddy-diffusivity
    # arm and the CLUBB arm of the wsubi test -- and nothing that computes
    for name, numbers in dropped.items():
        for n in numbers:
            line = re.sub(r"!.*", "", LINES[n - 1]).strip()
            if not line:
                continue
            assert (re.match(r"(select case|case |end select|if |else|endif|end if)", line)
                    or "kvh" in line or "CLUBB" in line or "wsubi(i,k) =" in line
                    or "naer2" in line or "dum" in line), (name, n, line)
    # what remains computes the diag_TKE velocity and the modal dust bins
    module = KERNEL_MODULE.read_text()
    assert "wsub(i,k) = sqrt(0.5_r8*(tke(i,k) + tke(i,k+1))*(2._r8/3._r8))" in module
    assert "nacon(i,k,3) = wght*num_coarse(i,k)*rho(i,k)" in module
    assert "kvh" not in module and "naer2" not in module


def test_the_class_binds_exactly_the_entries_the_module_offers() -> None:
    offered = set(re.findall(r"bind\(C, name='(pycam_aero_\w+)'\)", HANDLES.read_text()))
    bound = {template.format(prefix="aero") for template, _, _ in A._AeroEntries.TABLE.values()
             if template.startswith("pycam_{prefix}")}
    assert bound - offered == {"pycam_aero_bind_hosts_v1"}
    assert offered - bound == set()


def test_view_codes_match_the_module() -> None:
    codes = {name: int(code) for name, code in
             re.findall(r"parameter, public :: view_(\w+) = (\d+)", HANDLES.read_text())}
    assert codes == VIEW


def test_the_named_constants_are_the_pinned_source_s() -> None:
    text = SOURCE.read_text()
    assert float(re.search(r"qsmall = ([0-9.e-]+)_r8", text).group(1)) == A.QSMALL
    assert float(re.search(r"mincld = ([0-9.e-]+)_r8", text).group(1)) == A.MINCLD
    # the fixed dust radii are copied into the kernel module, not retyped
    for n in (62, 63, 64, 65):
        assert LINES[n - 1].strip() in KERNEL_MODULE.read_text()


def test_the_cores_are_called_in_the_driver_s_form() -> None:
    module = HANDLES.read_text()
    for call, pinned in (
        ("nucleate_ice_cam_calc", "call nucleate_ice_cam_calc(state, wsubi, pbuf)"),
        ("hetfrz_classnuc_cam_save_cbaero", "call hetfrz_classnuc_cam_save_cbaero(state, pbuf)"),
        ("hetfrz_classnuc_cam_calc", "call hetfrz_classnuc_cam_calc(state, deltatin, factnum, pbuf)"),
    ):
        assert pinned in " ".join(LINES), pinned
        assert pinned.replace("state,", "state,") in re.sub(r"\s+", " ", module), call
    # dropmixnuc keeps the driver's argument order, with the walk's ptend
    ours = re.search(r"call dropmixnuc\((.*?)\)\s*\n", re.sub(r"&\s*\n\s*", "", module), re.S)
    assert ours, "dropmixnuc"
    arguments = [a.strip() for a in ours.group(1).split(",")]
    assert arguments == ["state", "aero_ptend(lchnk)", "deltatin", "pbuf", "wsub",
                         "lcldn", "lcldo", "nctend_mixnuc", "factnum"]


# -- the sequence, against the pinned source ------------------------------------

CONDITIONS = {
    "micro_do_icesupersat": False,
    "clim_modal_aero": True,
    ".not. clim_modal_aero": False,
    "use_hetfrz_classnuc": True,
    "separate_dust": None,
    "t(i,k) < 269.15_r8": None,
    "dmc > 0.0_r8": None,
    "rndst(i,k,3) <= 0._r8": None,
    "qcld > qsmall": None,
    "qc(i,k) >= qsmall": None,
    "idxdst2 > 0": None, "idxdst3 > 0": None, "idxdst4 > 0": None,
    "eddy_scheme == 'CLUBB_SGS'": False,
    ".not. use_preexisting_ice": None,
}


def test_tend_s_sequence_is_the_pinned_routine_s_live_statements() -> None:
    """Walk the routine following its branches under this configuration; every
    call it makes and every carved block it enters must be in ``tend``'s
    order, once."""

    import generate_pi_cam_aero_kernels as gen

    blocks = {block.first: block.name for block in gen.BLOCKS}
    ends = {block.first: block.last for block in gen.BLOCKS}
    events: list[str] = []
    stack = [[True, True]]
    n = 408
    while n <= 713:
        if n in blocks and all(a and b for a, b in stack):
            events.append(blocks[n])
            n = ends[n] + 1
            continue
        line = re.sub(r"!.*", "", LINES[n - 1]).strip()
        n += 1
        if not line:
            continue
        if re.match(r"if\s*\(", line) and line.endswith("then"):
            parent = all(a and b for a, b in stack)
            if not parent:                       # inside a branch this run never takes
                stack.append([False, True])
                continue
            condition = re.match(r"if\s*\((.*)\)\s*then$", line).group(1).strip()
            assert condition in CONDITIONS, condition
            live = CONDITIONS[condition]
            stack.append([parent, True if live is None else live])
            continue
        if re.match(r"else\b", line):
            stack[-1][1] = not stack[-1][1]
            continue
        if re.match(r"end\s*if\b|endif\b", line):
            stack.pop()
            continue
        if not all(a and b for a, b in stack):
            continue
        call = re.match(r"call (\w+)", line)
        if call and call.group(1) not in ("t_startf", "t_stopf", "pbuf_get_field",
                                          "physics_ptend_init"):
            name = call.group(1)
            if name in ("outfld", "rad_cnst_get_aer_mmr"):
                if events and events[-1] == f"{name}*":
                    continue                     # a run of writes, logged once
                if name == "outfld" and events and events[-1] != "aero_subgrid_velocity":
                    events.append(name)          # the single LCLOUD write
                    continue
                name = f"{name}*"
            events.append(name)
    # the walk reads the buffer first and closes the chunk last; the getters
    # and the mode count it asks for are the source's, in its order
    assert ["pbuf_get_field*"] + events + ["end"] == list(SEQUENCE)


# -- the fake image -------------------------------------------------------------


class _Lib:
    PCOLS, PVER, PVERP, PCNST, NMODES = 16, 30, 31, 57, 3

    def __init__(self) -> None:
        self.views: dict[tuple, np.ndarray] = {}
        self.calls: list[str] = []
        self.owner = 0
        self.history: list[str] = []
        self.cores: list[tuple[str, tuple]] = []
        self.tke_index = -1
        self.dgnumwet_index = -1

    def _shape(self, code: int) -> tuple[int, ...]:
        name = next(k for k, v in VIEW.items() if v == code)
        if name == "state_q":
            return (self.PCOLS, self.PVER, self.PCNST)
        if name == "factnum":
            return (self.PCOLS, self.PVER, self.NMODES)
        if name == "ptend_aero_q":
            return (self.PCOLS, self.PVER, self.PCNST)
        return (self.PCOLS, self.PVER)

    def _serve(self, key, shape, ptr, ndims, extents, dtype=np.float64):
        array = self.views.setdefault(key, np.zeros(shape, dtype=dtype, order="F"))
        ptr._obj.value = array.ctypes.data
        if ndims is not None:
            ndims._obj.value = array.ndim
        for i, e in enumerate(array.shape):
            extents[i] = e

    def __getattr__(self, name):
        if not name.startswith("pycam_"):
            raise AttributeError(name)
        lib = self

        def entry(*args):
            lib.calls.append(name)
            if name == "pycam_aero_view_v1":
                lchnk, code, ptr, ndims, extents = args
                lib._serve((lchnk, "view", code), lib._shape(code), ptr, ndims, extents)
            elif name == "pycam_pbuf_field_v1":
                lchnk, index, sliced, ptr, extents = args
                # the PBL's turbulent kinetic energy is on the interfaces
                levels = lib.PVERP if index == lib.tke_index else lib.PVER
                lib._serve((lchnk, "pbuf", index), (lib.PCOLS, levels), ptr, None, extents)
            elif name == "pycam_pbuf_field_v2":
                lchnk, index, sliced, rank, is_int, ptr, ndims, extents = args
                third = lib.NMODES if index == lib.dgnumwet_index else 4
                lib._serve((lchnk, "pbuf", index), (lib.PCOLS, lib.PVER, third),
                           ptr, ndims, extents)
            elif name == "pycam_aero_set_owner_v1":
                lib.owner = args[0]
            elif name == "pycam_aero_nstep_v1":
                return 5
            elif name == "pycam_aero_dt_v1":
                return 1800
            elif name == "pycam_aero_nmodes_v1":
                return lib.NMODES
            elif name == "pycam_outfld_v1":
                lib.history.append(args[0].decode())
            else:
                lib.cores.append((name.removeprefix("pycam_aero_").removesuffix("_v1"),
                                  tuple(a for a in args if isinstance(a, (int, float)))))
            return 0
        return entry


class _Pool(dict):
    @property
    def dimensions(self):
        return {"pcols": 16, "pver": 30, "pverp": 31, "pcnst": 57, "chunks": 2}


class _Native:
    def __init__(self, lib):
        self.library = lib
        self.pool = _Pool({"grid.chunk_id": np.array([1540, 1541]),
                           "grid.chunk_ncols": np.array([14, 13])})
        self.kernels: list[str] = []
        from freecam.pi_cam.kernel_codegen import load_direct_kernels

        self._args = {k.name: [{"field": a.field, "dtype": a.dtype, "rank": a.rank}
                               for a in k.arguments]
                      for k in load_direct_kernels(MicropAero.DESCRIPTORS)}

    @property
    def chunks(self):
        return self.pool["grid.chunk_id"], self.pool["grid.chunk_ncols"]

    def kernel_arguments(self, name):
        return tuple(self._args[name])

    def run_kernel(self, name, arrays):
        self.kernels.append(name)


class _Context:
    def __init__(self, native):
        self.native = native
        self.timestep_seconds = 1800
        self.step = 5
        self.rank = 3


def _constants(**overrides) -> A._Constants:
    base = A._Constants(
        clim_modal_aero=True, micro_do_icesupersat=False, separate_dust=False,
        use_hetfrz_classnuc=True, use_preexisting_ice=False, eddy_scheme="diag_TKE",
        cldliq=2, cldice=3, mode_coarse_dst_idx=3, mode_coarse_slt_idx=3,
        coarse_dust_idx=1, coarse_nacl_idx=2, top_lev=1, rair=287.04,
        mincld=A.MINCLD, qsmall=A.QSMALL)
    return replace(base, **overrides)


@pytest.fixture
def fake(monkeypatch):
    native = _Native(_Lib())
    constants = _constants()
    monkeypatch.setattr(A._Constants, "read", classmethod(lambda cls, library: constants))
    indices: dict[str, int] = {}
    monkeypatch.setattr(A, "module_view", lambda library, symbol, dtype, shape: np.array(
        indices.setdefault(symbol, len(indices) + 1), dtype=np.int32))
    monkeypatch.setattr(A.PBuf, "verify", lambda self, chunk, **kw: None)
    native.library.tke_index = indices.setdefault("microp_aero_mp_tke_idx_", len(indices) + 1)
    native.library.dgnumwet_index = indices.setdefault(
        "microp_aero_mp_dgnumwet_idx_", len(indices) + 1)
    return native


def test_tend_walks_the_routine_in_its_order_on_every_chunk(fake) -> None:
    scheme = MicropAero()
    scheme.tend(None, _Context(fake))
    assert scheme.calls == list(SEQUENCE) * 2
    lib = fake.library
    assert lib.owner == 1
    for name in KERNELS:
        assert fake.kernels.count(name) == 2, name
    names = [n for n, _ in lib.cores]
    per_chunk = ["begin", "save_cbaero", "modal_fields", "nucleate_ice",
                 "dropmixnuc", "hetfrz", "end"]
    # bind_hosts is called once when the runtime is built, and again by the
    # runtime itself; nmodes is asked once before the scratch is sized
    assert [n for n in names if n not in ("bind_hosts", "nmodes")] == per_chunk * 2
    # the cores got the driver's dtime and the mode count the image reported
    assert [a for n, a in lib.cores if n == "dropmixnuc"] == [(1800.0, 3)] * 2
    assert [a for n, a in lib.cores if n == "hetfrz"] == [(1800.0,)] * 2
    assert lib.history == ["WSUB", "WSUBI", "LCLOUD"] * 2


def test_without_classical_freezing_neither_freezing_call_runs(fake, monkeypatch) -> None:
    constants = _constants(use_hetfrz_classnuc=False)
    monkeypatch.setattr(A._Constants, "read", classmethod(lambda cls, library: constants))
    scheme = MicropAero()
    scheme.tend(None, _Context(fake))
    assert [n for n, _ in fake.library.cores if "hetfrz" in n or "cbaero" in n] == []
    assert scheme.calls == [c for c in SEQUENCE if "hetfrz" not in c] * 2


@pytest.mark.parametrize("bad, what", [
    ({"clim_modal_aero": False}, "bulk-aerosol"),
    ({"micro_do_icesupersat": True}, "icesupersat"),
    ({"eddy_scheme": "HB"}, "diag_TKE"),
    ({"mode_coarse_dst_idx": 0}, "coarse dust"),
    ({"cldliq": 0}, "constituent"),
])
def test_the_paths_the_configuration_never_takes_are_refused(bad, what) -> None:
    with pytest.raises(PICAMConfigurationError, match=what):
        _constants(**bad).refuse_unsupported()
    _constants().refuse_unsupported()
