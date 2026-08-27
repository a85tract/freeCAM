"""Microphysics: micro_mg_cam_tend as Python, checked against the pinned source.

Without a model image: the class's tables are the generators' and the
source's (view and input codes, history names in order, the grid copies,
the named constants); ``tend``'s per-chunk sequence equals the live
statements of the pinned routine, found by following its branch nesting
under the admitted configuration with every lifted range atomic; and a
fake image walks two chunks end to end.
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

import freecam.physics.microphysics as M  # noqa: E402
from freecam.physics.microphysics import (  # noqa: E402
    CORE, GRID_COPIES, HISTORY_GRID, HISTORY_MG, HISTORY_TENDENCIES, INPUT, KERNELS,
    SEQUENCE, VIEW, Microphysics,
)
from freecam.pi_cam.errors import PICAMConfigurationError  # noqa: E402

HANDLES = REPO / "native/pi_cam/support/pycam_micro_handles.F90"
SOURCE = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/micro_mg_cam.F90"
LINES = SOURCE.read_text().splitlines()


def _line(n: int) -> str:
    return LINES[n - 1]


# -- tables against the generators and the source -------------------------------


def test_the_codes_are_the_generator_s() -> None:
    import generate_pi_cam_micro_handles as gen

    assert INPUT == {name: i + 1 for i, (name, _) in enumerate(gen.INPUT_ALIASES)}
    assert list(M.CONFIGURATION) == [name for name, _ in gen.CONFIGURATION]
    text = HANDLES.read_text()
    codes = {name: int(code) for name, code in
             re.findall(r"parameter, public :: view_(\w+) = (\d+)", text)}
    assert codes == VIEW


def test_the_class_binds_exactly_the_entries_the_module_offers() -> None:
    text = HANDLES.read_text()
    offered = set(re.findall(r"bind\(C, name='(pycam_micro_\w+)'\)", text))
    bound = {template.format(prefix="micro") for template, _, _ in M._MicroEntries.TABLE.values()
             if template.startswith("pycam_{prefix}")}
    assert bound - offered == {"pycam_micro_bind_hosts_v1"}
    assert offered - bound == set()


def test_the_named_constants_are_the_pinned_source_s() -> None:
    water_types = (REPO / "external/iCESM1.3.1_fzhu/cime/src/share/util/water_types.F90").read_text()
    for name, value in (("iwtstrain", M.IWTSTRAIN), ("iwtstsnow", M.IWTSTSNOW)):
        assert re.search(rf"parameter, public :: {name}\s*=\s*{value}\b", water_types), name
    shr = (REPO / "external/iCESM1.3.1_fzhu/cime/src/share/util/shr_const_mod.F90").read_text()
    literal = re.search(r"SHR_CONST_RHOFW\s*=\s*([0-9.e]+)_R8", shr).group(1)
    assert float(literal) == M.RHOH2O
    utils = (REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/micro_mg_utils.F90").read_text()
    assert float(re.search(r"qsmall = ([0-9.e-]+)_r8", utils).group(1)) == M.QSMALL
    assert float(re.search(r"mincld = ([0-9.e-]+)_r8", utils).group(1)) == M.MINCLD


def _live_outfld_calls() -> list[tuple[str, str]]:
    """(field, argument) of every outfld in 3040-3176 outside the dead branches."""

    calls = []
    dead = 0
    for n in range(3037, 3177):
        line = _line(n).strip()
        if re.match(r"if \((micro_mg_version > 1|\.not\. \(micro_mg_version == 1|use_subcol_microp)", line):
            dead += 1
            continue
        if line.startswith("end if") and dead:
            dead -= 1
            continue
        if dead:
            continue
        m = re.match(r"call outfld\(\s*'(\w+)'\s*,\s*(\w+)", line)
        if m:
            calls.append(m.groups())
    return calls


def test_history_is_the_source_s_in_order() -> None:
    pinned = _live_outfld_calls()
    ours = [(name, "ftem_grid") for name, _ in HISTORY_TENDENCIES] + list(HISTORY_MG) + list(HISTORY_GRID)
    assert [name for name, _ in ours] == [name for name, _ in pinned]
    # the MG-output arguments are the source's names (views or kernel scratch)
    assert [arg for _, arg in ours[6:]] == [arg for _, arg in pinned[6:]]
    # every history argument the class reads is a view, a kernel field or a grid array
    fields = set()
    from freecam.pi_cam.kernel_codegen import load_direct_kernels
    for kernel in load_direct_kernels(Microphysics.DESCRIPTORS):
        fields.update(a.field.removeprefix("micro.") for a in kernel.arguments)
    for _, key in HISTORY_MG:
        assert key in VIEW or key in fields, key
    for _, key in HISTORY_GRID:
        assert key in M.GRID_PBUF or key in fields, key


def test_grid_copies_are_the_source_s() -> None:
    copies = []
    for n in range(2485, 2520):
        m = re.match(r"\s*(\w+)_grid\s*=\s*(\w+)\s*$", _line(n))
        if m and m.group(1) == m.group(2):
            copies.append(m.group(1))
    # am_evp_st, evpsnow_st (buffer targets), cld, icwmrst..icinc, pdel, nc/ni
    # are handled by name in the walk; the rest are GRID_COPIES in order
    named = {"am_evp_st", "cld", "icwmrst", "icimrst", "liqcldf", "icecldf", "icwnc", "icinc"}
    assert [c for c in copies if c not in named] == list(GRID_COPIES)
    assert "evpsnow_st_grid = evapsnow" in _line(2489)
    assert "pdel_grid       = state_loc%pdel" in _line(2514)
    assert "nc_grid = state_loc%q(:,:,ixnumliq)" in _line(2518)


# -- the sequence, against the pinned source ------------------------------------

CONDITIONS = {
    ".not. do_cldice": False,
    "use_hetfrz_classnuc": False,
    "rate1_cw2pr_st_idx > 0": True,
    "qrain_idx > 0": True, "qsnow_idx > 0": True, "nrain_idx > 0": True, "nsnow_idx > 0": True,
    "use_subcol_microp": False,
    "micro_mg_version == 1 .and. micro_mg_sub_version == 0": True,
    "micro_mg_version > 1": False,
    "trace_water": True,
    ".not. (micro_mg_version == 1 .and. micro_mg_sub_version == 0)": False,
}

#: Statements that are the first of a group the walk does by hand, to the
#: name it logs.
GROUPS = {
    "cldo(:ncol,top_lev:pver)=ast(:ncol,top_lev:pver)": "cldo=ast",
    "cvreffliq(:ncol,top_lev:pver) = 9.0_r8": "cvreff=const",
    "rate1ord_cw2pr_st(:ncol,top_lev:pver) = rate1cld(:ncol,top_lev:pver)": "rate1ord=rate1cld",
    "icinc = 0._r8": "zero6",
    "lambdac_grid    => lambdac": "grid_aliases",
    "am_evp_st_grid  = am_evp_st": "grid_copies",
    "preo_grid      => preo": "wtrc_aliases",
    "sed_rates_grid(:, top_lev:, :)        = 0._r8": "sed_rates",
    "rho_grid = rho": "rho_grid=rho",
    "qrout_grid_ptr = qrout_grid": "grid_ptr=grid",
}
GROUPS_NORMALISED = {re.sub(r"\s+", " ", k): v for k, v in GROUPS.items()}
IGNORED_CALLS = {"phys_getopts", "pbuf_col_type_index", "t_startf", "t_stopf"}
IGNORED_ASSIGNMENTS = re.compile(
    r"^(nlev|lchnk|ncol|psetcols|ngrdcol|itim_old)\s*=|^\w+\s*=>\s*\w+$|^\w+_grid(_ptr)?\s*=\s*\w+$"
    r"|^\w+_grid\s*= state_loc%|^(cvreffice|cldfsnow|icswp|iclwpst|iciwpst|icwnc)\(?.*= 0\._r8$"
    r"|^cvreffice\(|^sed_rates_grid\(")


def _lifted_ranges() -> list[tuple[int, int, str]]:
    import generate_pi_cam_micro_handles as handles
    import generate_pi_cam_micro_kernels as kernels

    ranges = [(b.first, b.last, b.name) for b in kernels.BLOCKS]
    names = {"micro_pack_prelude": "pack_prelude", "micro_substep_pack": "substep_pack",
             "micro_core": CORE, "micro_substep_unpack": "substep_unpack",
             "micro_post_proc": "post_proc"}
    ranges += [(first, last, names[name]) for name, first, last in handles.VERBATIM]
    ranges.append((1743, 1766, "physics_ptend_init:cldwat"))
    return sorted(ranges)


def _live_events() -> list[str]:
    ranges = _lifted_ranges()
    stack = [[True, True]]
    events: list[str] = []
    n = 1554
    continued = False
    while n <= 3182:
        lifted = next((r for r in ranges if r[0] <= n <= r[1]), None)
        if lifted:
            if stack[-1][0] and stack[-1][1]:
                events.append(lifted[2])
            n = lifted[1] + 1
            continued = False
            continue
        raw = _line(n)
        n += 1
        line = re.sub(r"!.*", "", raw).strip()
        if not line:
            continue
        tail, continued = continued, line.endswith("&")
        if tail:
            continue                      # the rest of a continued statement
        line = line.rstrip("&").strip()
        one = re.match(r"if \(([^()]*)\)\s+(call \w+.*|\w+ = .*)$", line)
        if one:
            assert one.group(1) in CONDITIONS, one.group(1)
            if not (stack[-1][0] and stack[-1][1] and CONDITIONS[one.group(1)]):
                continue
            line = one.group(2)
        elif re.match(r"if\s*\(", line) and line.endswith("then"):
            condition = re.match(r"if\s*\((.*)\)\s*then$", line).group(1).strip()
            assert condition in CONDITIONS, f"unlisted branch condition: {condition!r}"
            stack.append([stack[-1][0] and stack[-1][1], CONDITIONS[condition]])
            continue
        elif re.match(r"else\b", line):
            stack[-1][1] = not stack[-1][1]
            continue
        elif re.match(r"end\s*if\b|endif\b", line):
            stack.pop()
            continue
        if not (stack[-1][0] and stack[-1][1]):
            continue
        if re.match(r"(do |end do|integer|real|logical|character|type\(|&)", line) or line.startswith("&"):
            continue
        call = re.match(r"call (\w+)", line)
        if call:
            name = call.group(1)
            if name in IGNORED_CALLS:
                continue
            if name in ("pbuf_get_field", "outfld", "wtrc_add_rates"):
                name = f"{name}*"
                if events and events[-1] == name:
                    continue
            events.append(name)
            continue
        key = re.sub(r"\s+", " ", line)
        kernel = GROUPS_NORMALISED.get(key)
        if kernel:
            events.append(kernel)
            continue
        assert IGNORED_ASSIGNMENTS.match(line), f"unclassified live statement at {n - 1}: {line!r}"
    assert len(stack) == 1
    return events


def test_tend_s_sequence_is_the_pinned_routine_s_live_statements() -> None:
    assert _live_events() == list(SEQUENCE)


# -- the fake image -------------------------------------------------------------


class _Lib:
    PCOLS, PVER, PVERP, PCNST = 16, 30, 31, 57

    def __init__(self) -> None:
        self.views: dict[tuple, np.ndarray] = {}
        self.calls: list[str] = []
        self.owner = 0
        self.core_owner = None
        self.configured = None
        self.bound: list[tuple[int, int, tuple]] = []
        self.history: list[str] = []
        self.lifecycle: list[tuple[str, tuple]] = []
        self.interface_fields: set[int] = set()

    def _view_shape(self, code: int) -> tuple[int, ...]:
        name = next(k for k, v in VIEW.items() if v == code)
        rank = M.VIEW_RANK[name]
        if rank == 1:
            return (self.PCOLS,)
        if rank == 3:
            return (self.PCOLS, self.PVER, self.PCNST)
        return (self.PCOLS, self.PVERP if name in ("rflx", "sflx") else self.PVER)

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
            if name == "pycam_micro_view_v1":
                lchnk, code, ptr, ndims, extents = args
                lib._serve((lchnk, "view", code), lib._view_shape(code), ptr, ndims, extents)
            elif name == "pycam_pbuf_field_v1":
                lchnk, index, sliced, ptr, extents = args
                levels = lib.PVERP if index in lib.interface_fields else lib.PVER
                lib._serve((lchnk, "pbuf", index), (lib.PCOLS, levels), ptr, None, extents)
            elif name == "pycam_pbuf_field_v2":
                lchnk, index, sliced, rank, is_int, ptr, ndims, extents = args
                shape = {1: (lib.PCOLS,), 3: (lib.PCOLS, lib.PVER, 4)}[rank]
                lib._serve((lchnk, "pbuf", index), shape, ptr, ndims, extents,
                           dtype=np.int32 if is_int else np.float64)
            elif name == "pycam_micro_set_owner_v1":
                lib.owner = args[0]
            elif name == "pycam_micro_set_core_owner_v1":
                lib.core_owner = args[0]
            elif name == "pycam_micro_configure_v1":
                lib.configured = tuple(args)
            elif name == "pycam_micro_nstep_v1":
                return 5
            elif name == "pycam_micro_dt_v1":
                return 1800
            elif name == "pycam_micro_bind_input_v1":
                lib.bound.append((args[0], args[1], tuple(args[3:])))
            elif name == "pycam_outfld_v1":
                lib.history.append(args[0].decode())
            elif name.startswith("pycam_micro_"):
                lib.lifecycle.append((name.removeprefix("pycam_micro_").removesuffix("_v1"),
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
                      for k in load_direct_kernels(Microphysics.DESCRIPTORS)}

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


def _constants(**overrides) -> M._Constants:
    base = M._Constants(
        micro_mg_version=1, micro_mg_sub_version=0, num_steps=1, microp_uniform=False,
        do_cldice=True, do_cldliq=True, ixcldliq=2, ixcldice=3, ixnumliq=4, ixnumice=5,
        ixrain=-1, ixsnow=-1, ixnumrain=-1, ixnumsnow=-1, qrain_idx=7, qsnow_idx=8,
        nrain_idx=9, nsnow_idx=10, rate1_cw2pr_st_idx=11, am_evp_st_idx=12,
        use_hetfrz_classnuc=False, use_subcol_microp=False, trace_water=True, top_lev=1,
        gravit=9.80616, rair=287.04, cpair=M.CPAIR, rhoh2o=M.RHOH2O, mincld=M.MINCLD,
        qsmall=M.QSMALL)
    return replace(base, **overrides)


@pytest.fixture
def fake(monkeypatch):
    native = _Native(_Lib())
    constants = _constants()
    monkeypatch.setattr(M._Constants, "read", classmethod(lambda cls, library: constants))
    indices: dict[str, int] = {}
    monkeypatch.setattr(M, "module_view", lambda library, symbol, dtype, shape: np.array(
        indices.setdefault(symbol, len(indices) + 1), dtype=np.int32))
    monkeypatch.setattr(M.PBuf, "verify", lambda self, chunk, **kw: None)
    # the two flux fields are (pcols, pverp) in the buffer
    native.library.interface_fields = {
        indices.setdefault(f"micro_mg_cam_mp_{n}_idx_", len(indices) + 1)
        for n in ("ls_flxprc", "ls_flxsnw")}
    return native


def test_tend_walks_the_routine_in_its_order_on_every_chunk(fake) -> None:
    scheme = Microphysics()
    scheme.tend(None, _Context(fake))
    assert scheme.calls == list(SEQUENCE) * 2
    lib = fake.library
    assert lib.owner == 1 and lib.core_owner == 0
    assert lib.configured == (1, 0, 1, 0, 1, 1, 2, 3, 4, 5, -1, -1, -1, -1)
    for name in KERNELS:
        expected = {"wtrc_init_rates": 4, "wtrc_add_rates": 22}.get(name, 2)
        assert fake.kernels.count(name) == expected, name
    # the lifted section, once per chunk in its order, with the driver's dtime
    names = [n for n, _ in lib.lifecycle]
    per_chunk = ["begin", "ptend_init", "pack_prelude", "substep_pack", "core", "substep_unpack",
                 "post_proc", "wtrc_add_sum", "wtrc_add_sum", "wtrc_add_sum", "wtrc_add_sum",
                 "wtrc_apply", "output_precip", "end"]
    assert names == ["bind_hosts"] + per_chunk * 2
    assert [a for n, a in lib.lifecycle if n == "begin"] == [(1540, 14, 1800.0), (1541, 13, 1800.0)]
    # the buffer storage the lifted section packs was bound by code, once per
    # chunk: the eighteen fields of the admitted configuration
    codes = sorted(code for lchnk, code, _ in lib.bound if lchnk == 1540)
    expected = sorted(INPUT[n] for n in (
        "naai", "npccn", "rndst", "nacon", "relvar", "accre_enhan", "ast", "alst_mic", "aist_mic",
        "rel", "rei", "dei", "des", "mu", "lambdac", "prain", "nevapr", "prer_evap",
        "rate1ord_cw2pr_st"))
    assert codes == expected
    assert (1540, INPUT["rndst"], (16, 30, 4)) in lib.bound
    # history, in the source's order, per chunk
    per_chunk_history = ([n for n, _ in HISTORY_TENDENCIES] + [n for n, _ in HISTORY_MG]
                         + [n for n, _ in HISTORY_GRID])
    assert lib.history == per_chunk_history * 2


def test_the_flags_the_walk_branches_on(fake, monkeypatch) -> None:
    constants = _constants(use_hetfrz_classnuc=True, rate1_cw2pr_st_idx=0, trace_water=False)
    monkeypatch.setattr(M._Constants, "read", classmethod(lambda cls, library: constants))
    scheme = Microphysics()
    scheme.tend(None, _Context(fake))
    lib = fake.library
    codes = {code for lchnk, code, _ in lib.bound if lchnk == 1540}
    assert {INPUT["frzimm"], INPUT["frzcnt"], INPUT["frzdep"]} <= codes
    assert INPUT["rate1ord_cw2pr_st"] not in codes
    assert "wtrc_apply_rates" not in scheme.calls and "wtrc_apply" not in [n for n, _ in lib.lifecycle]
    assert "micro_split_signs" not in fake.kernels


@pytest.mark.parametrize("bad, what", [
    ({"micro_mg_version": 2}, "not 1.0"),
    ({"micro_mg_sub_version": 5}, "not 1.0"),
    ({"use_subcol_microp": True}, "subcol"),
    ({"num_steps": 0}, "num_steps"),
    ({"top_lev": 3}, "trop_cloud_top_lev"),
])
def test_the_paths_the_configuration_never_takes_are_refused(bad, what) -> None:
    with pytest.raises(PICAMConfigurationError, match=what):
        _constants(**bad).refuse_unsupported()
    _constants().refuse_unsupported()


def test_the_packed_contract_is_the_core_s_argument_list() -> None:
    """Every packed input and output is a view the handles module serves, and
    together they are micro_mg_tend1_0's array arguments minus the five the
    driver discards."""

    for name in M.PACKED_INPUTS + M.PACKED_INPUTS_NO_CLDICE + M.PACKED_INPUTS_HETFRZ + M.PACKED_OUTPUTS:
        assert f"packed_{name}" in VIEW, name
    text = HANDLES.read_text()
    call = re.search(r"call micro_mg_tend1_0\((.*?)\)\s*\n\s*case \(5\)", text, re.S).group(1)
    actuals = [a.strip() for a in re.sub(r"[&\n]", " ", call).split(",")]
    packed = [a.removeprefix("packed_") for a in actuals if a.startswith("packed_")]
    contract = set(M.PACKED_INPUTS + M.PACKED_INPUTS_NO_CLDICE + M.PACKED_INPUTS_HETFRZ + M.PACKED_OUTPUTS)
    assert set(packed) == contract
    assert [a for a in actuals if a.endswith("_dum")] == [
        "rel_fn_dum", "reff_rain_dum", "reff_snow_dum", "drout_dum", "dsout2_dum"]
    # inout arguments are in both lists
    assert {"qc", "qi", "nc", "ni"} <= set(M.PACKED_INPUTS) & set(M.PACKED_OUTPUTS)


def test_a_model_in_the_core_s_place_sees_the_packed_batch_and_answers_it(fake) -> None:
    seen = {}

    def model(batch):
        seen.update({k: np.asarray(v).shape for k, v in batch.items()})
        return {name: np.full(batch["t"].shape if name not in ("prect", "preci") else batch["t"].shape[:1],
                              7.0) for name in M.PACKED_OUTPUTS}

    scheme = Microphysics(kernel=model)
    scheme.tend(None, _Context(fake))
    lib = fake.library
    assert lib.core_owner == 1
    assert "core" not in [n for n, _ in lib.lifecycle]
    assert scheme.calls == list(SEQUENCE) * 2
    assert set(seen) == set(M.PACKED_INPUTS)
    assert seen["t"] == (16, 30) and seen["rndst"] == (16, 30, 57)
    # the answer landed in the packed storage the unpack reads
    tlat = lib.views[(1540, "view", VIEW["packed_tlat"])]
    assert np.all(tlat == 7.0)


def test_a_model_must_answer_every_output(fake) -> None:
    scheme = Microphysics(kernel=lambda batch: {"tlat": batch["t"]})
    from freecam.physics.errors import PhysicsError

    with pytest.raises(PhysicsError, match="missing"):
        scheme.tend(None, _Context(fake))


def test_a_kernel_field_that_lives_in_the_buffer_must_be_named_by_the_walk(fake, monkeypatch) -> None:
    """Scratch is not the physics buffer: a `_grid` field the walk forgets to
    pass would be read as zeros.  The walk refuses instead."""

    from freecam.physics.errors import PhysicsError

    # cld_grid is a routine local the walk fills by copy, so the walk passes it
    # to no kernel; calling it buffer-backed makes the guard fire on the real
    # call that takes it
    monkeypatch.setitem(M.GRID_PBUF, "cld_grid", "CLD")
    scheme = Microphysics()
    with pytest.raises(PhysicsError, match="cld_grid.*physics buffer"):
        scheme.tend(None, _Context(fake))


def test_each_kernel_reads_the_state_the_source_reads() -> None:
    """The routine reads `state%` before it copies and again after the substep
    updated the copy, and `state_loc%` in between; from the substep on they
    are different arrays.  A kernel field named `q`/`t`/`pmid`/`pdel` is the
    host's, one named `*_loc` the copy's, and the walk must pass each from the
    matching view."""

    import generate_pi_cam_micro_kernels as kernels
    from freecam.pi_cam.kernel_codegen import load_direct_kernels

    host = {v: k for k, v in kernels.COMMON.items() if k.startswith("state%")}
    copy = {v: k for k, v in kernels.COMMON.items() if k.startswith("state_loc%")}
    assert set(host) == {"q", "pmid", "t", "pdel"} and set(copy) == {
        "q_loc", "pmid_loc", "t_loc", "pdel_loc"}

    source = (REPO / "src/freecam/physics/microphysics.py").read_text().split("def tend_chunk", 1)[1]
    calls = {m.group(1): m.group(2) for m in
             re.finditer(r'K\("(\w+)",\s*(.*?)\n\s*log\(', source, re.S)}
    described = {k.name: [a.field.removeprefix("micro.") for a in k.arguments]
                 for k in load_direct_kernels(Microphysics.DESCRIPTORS)}
    checked = 0
    for name, fields in described.items():
        if name not in calls:
            continue
        for field in fields:
            if field in host:
                assert f'"{field}": S["state_{field}"]' in calls[name], (name, field)
                checked += 1
            elif field in copy:
                stem = field.removesuffix("_loc")
                assert f'"{field}": S["state_loc_{stem}"]' in calls[name] or (
                    field == "q_loc" and '"q_loc": q_loc' in calls[name]), (name, field)
                checked += 1
    assert checked >= 6
    # and both sets of views exist, at distinct codes
    for stem in ("t", "q", "pmid", "pdel"):
        assert VIEW[f"state_{stem}"] != VIEW[f"state_loc_{stem}"]


def test_every_buffer_field_the_routine_reads_is_reached_by_the_walk() -> None:
    """The generated table is what the pinned routine reads; a field in it that
    the walk never binds, reads or writes is a statement the walk dropped."""

    import yaml

    source = (REPO / "src/freecam/physics/microphysics.py").read_text().split("def tend_chunk", 1)[1]
    named = set(re.findall(r'pbv\["(\w+)"\]', source))
    named |= set(M.GRID_PBUF.values())                        # reached through G()
    for group in re.findall(r'for name in \(([^)]*)\):\n\s*H\.bind_input', source):
        named |= {n.strip().strip('"').upper() for n in group.split(",") if n.strip()}
    # named singly, or through a loop variable
    named |= {"AST", "RATE1_CW2PR_ST", "QRAIN", "QSNOW", "NRAIN", "NSNOW"}
    table = {row["name"] for row in
             yaml.safe_load((REPO / "native/pi_cam/pbuf_fields_micro.yaml").read_text())["fields"]}
    assert table - named == set(), sorted(table - named)
