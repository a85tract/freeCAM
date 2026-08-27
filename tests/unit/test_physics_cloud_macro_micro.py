"""CloudMacroMicrophysics: tphysbc stage 7 as Python, the drivers called whole.

Three things are checked without a model image: that the handles module
the generator writes is the committed one and wraps tphysbc's calls in
tphysbc's form; that the class binds exactly what the module offers; and
that ``tend`` walks the stage in the pinned source's order on every chunk.
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

import freecam.physics.cloud_macro_microphysics as M  # noqa: E402
from freecam.physics.cloud_macro_microphysics import (  # noqa: E402
    KERNELS, MACROP_ARGUMENTS, SEQUENCE, SEQUENCE_WHOLE, SEQUENCE_WHOLE_MICRO, VIEW,
    CloudMacroMicrophysics,
)
from freecam.physics.macrophysics import Macrophysics  # noqa: E402
from freecam.physics.microphysics import Microphysics  # noqa: E402
from freecam.pi_cam.errors import PICAMConfigurationError  # noqa: E402

HANDLES = REPO / "native/pi_cam/support/pycam_mm_handles.F90"
PHYSPKG = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/physpkg.F90"
# from the first statement of the MG branch through the end if that closes
# the mass fixer's: 1-based line numbers found by their text
_SOURCE = PHYSPKG.read_text().splitlines()
LAST = 2 + _SOURCE.index("     call wtrc_mass_fixer(state)")
FIRST = 2 + max(i for i, line in enumerate(_SOURCE[:LAST])
                if "elseif( microp_scheme == 'MG' )" in line)


def _lines(path: Path, first: int, last: int) -> list[str]:
    return path.read_text().splitlines()[first - 1:last]


# -- the handles module ---------------------------------------------------------


def test_the_committed_module_is_what_the_generator_writes() -> None:
    import generate_pi_cam_mm_handles as gen

    assert gen.render_module() == HANDLES.read_text()


def test_the_class_binds_exactly_the_entries_the_module_offers() -> None:
    text = HANDLES.read_text()
    offered = set(re.findall(r"bind\(C, name='(pycam_mm_\w+)'\)", text))
    bound = {template.format(prefix="mm") for template, _, _ in M._MMEntries.TABLE.values()
             if template.startswith("pycam_{prefix}")}
    # bind_hosts is emitted by state_codegen into cam_comp's include
    assert bound - offered == {"pycam_mm_bind_hosts_v1"}
    assert offered - bound == set()
    # the two the module does not own: history and tphysbc's forcing buffers
    others = {template for template, _, _ in M._MMEntries.TABLE.values()
              if not template.startswith("pycam_{prefix}")}
    assert others == {"pycam_outfld_v1", "pycam_macro_forcing_v1"}


def test_view_codes_match_the_module() -> None:
    text = HANDLES.read_text()
    codes = {name: int(code) for name, code in
             re.findall(r"parameter, public :: view_(\w+) = (\d+)", text)}
    assert codes == VIEW


def _call_arguments(text: str, routine: str) -> list[str]:
    """The comma-separated arguments of the first ``call routine(...)``, joined
    across continuation lines, comments and blanks removed."""

    match = re.search(rf"call {routine}\((.*?)\)\s*\n(?!\s*&)", text, re.S)
    assert match, routine
    body = re.sub(r"!.*", "", match.group(1))
    body = body.replace("&", "").replace("\n", " ")
    arguments, depth, current = [], 0, ""
    for char in body:                      # split on commas outside parentheses
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            arguments.append(current.strip()); current = ""
        else:
            current += char
    if current.strip():
        arguments.append(current.strip())
    return arguments


def test_the_macrophysics_wrapper_passes_tphysbc_s_arguments_in_its_order() -> None:
    pinned = _call_arguments("\n".join(_lines(PHYSPKG, 2244, 2252)), "macrop_driver_tend")
    ours = _call_arguments(HANDLES.read_text(), "macrop_driver_tend")
    rename = {"host_state(lchnk)": "state", "mm_ptend(lchnk)": "ptend",
              "mm_det_s(:,lchnk)": "det_s", "mm_det_ice(:,lchnk)": "det_ice"}
    ours = [rename.get(a, a) for a in ours]
    pinned = [a.replace("cam_in%", "") for a in pinned]
    assert ours == pinned
    # and the class hands the eleven arrays in the same order
    assert list(MACROP_ARGUMENTS) == pinned[3:14]


def test_the_glue_s_expressions_are_formed_in_fortran_as_tphysbc_forms_them() -> None:
    text = re.sub(r"!.*", "", HANDLES.read_text())
    pinned = "\n".join(_lines(PHYSPKG, FIRST, LAST))
    # the scale factor, the substep-divided fluxes, the driver's substep length
    assert text.count("1._r8/cld_macmic_num_steps") == 2
    assert "1._r8/cld_macmic_num_steps" in pinned
    for expression in ("flx_cnd/cld_macmic_num_steps", "flx_ice/cld_macmic_num_steps",
                       "flx_sen/cld_macmic_num_steps"):
        assert expression in text
    assert "det_ice/cld_macmic_num_steps" in pinned and "flx_heat/cld_macmic_num_steps" in pinned
    assert "prec_str/cld_macmic_num_steps" in pinned
    # every other statement of the glue is a call or a lifted kernel: no
    # arithmetic in Python
    source = (REPO / "src/freecam/physics/cloud_macro_microphysics.py").read_text()
    body = source.split("def tend_chunk", 1)[1]
    assert not re.search(r"[-+*/]\s*(dt|n|ncol|zero|sub_dt)\b", body.replace("n + 1", ""))


def test_the_module_sits_below_the_control_layer_and_is_registered() -> None:
    text = HANDLES.read_text()
    assert "use cam_comp" not in text and "use physpkg" not in text
    from apply_pi_cam_source_patches import SUPPORT_SOURCES
    from build_pi_cam_devices import SUPPORT_MODULES

    for name in ("pycam_mm_handles.F90", "pycam_micro_handles.F90"):
        assert name in SUPPORT_MODULES
        assert any(source.endswith(name) for source, _ in SUPPORT_SOURCES)
    from freecam.pi_cam import state_codegen

    binding = "\n".join(state_codegen._mm_host_binding())
    assert "call pycam_mm_bind_hosts(phys_state, phys_tend, pbuf2d)" in binding
    assert "call pycam_micro_bind_hosts(phys_state, pbuf2d)" in "\n".join(
        state_codegen._micro_host_binding())
    codegen = Path(state_codegen.__file__).read_text()
    assert "*_mm_host_binding()," in codegen and "*_micro_host_binding()," in codegen


# -- the sequence, against the pinned source ------------------------------------

#: How each branch condition in stage 7 reads under the admitted
#: configuration.  A condition not listed here is a source change to look at.
CONDITIONS = {
    "micro_do_icesupersat": False,
    "macrop_scheme .ne. 'CLUBB_SGS'": True,
    "is_subcol_on()": False,
    ".not. micro_do_icesupersat": True,
    "use_subcol_microp": False,
    "carma_do_cldice .or. carma_do_cldliq": False,
    "trace_water": True,
}

#: The first statement of each lifted arithmetic group, to the kernel's name.
GROUPS = {
    "cld_macmic_ztodt = ztodt/cld_macmic_num_steps": "mm_substep_dt",
    "prec_sed_macmic = 0._r8": "macmic_zero",
    "flx_cnd(:ncol) = -1._r8*rliq(:ncol)": "mm_flux_terms",
    "prec_sed_macmic(:ncol) = prec_sed_macmic(:ncol) + prec_sed(:ncol)": "mm_precip_accumulate",
    "prec_sed(:ncol) = prec_sed_macmic(:ncol)/cld_macmic_num_steps": "mm_precip_average",
}


def _live_events() -> list[str]:
    """Walk the pinned stage, following if/else nesting on the configuration."""

    # [parent live, own branch live]: the routine, then the MG branch the
    # range starts inside of (its endif is the last one the walk sees)
    stack = [[True, True], [True, True]]
    events: list[str] = []
    for raw in _lines(PHYSPKG, FIRST, LAST):
        line = re.sub(r"!.*", "", raw).strip()
        if not line:
            continue
        if re.match(r"if\s*\(", line) and line.endswith("then"):
            condition = re.match(r"if\s*\((.*)\)\s*then$", line).group(1).strip()
            assert condition in CONDITIONS, f"unlisted branch condition: {condition!r}"
            parent = stack[-1][0] and stack[-1][1]
            stack.append([parent, CONDITIONS[condition]])
            continue
        if re.match(r"else\b", line):
            stack[-1][1] = not stack[-1][1]
            continue
        if re.match(r"end\s*if\b|endif\b", line):
            stack.pop()
            continue
        if not (stack[-1][0] and stack[-1][1]):
            continue
        call = re.match(r"call (\w+)\s*\((.*)", line)
        if call:
            name, rest = call.groups()
            if name in ("t_startf", "t_stopf", "physics_ptend_dealloc"):
                continue
            if name == "check_energy_chng":
                label = re.search(r'"(\w+)"', rest).group(1)
                events.append(f"{name}:{label}")
            elif name == "physics_ptend_sum":
                events.append(f"{name}:{rest.split(',')[0].strip()}")
            else:
                events.append(name)
            continue
        kernel = GROUPS.get(re.sub(r"\s+", " ", line))
        if kernel:
            events.append(kernel)
    assert len(stack) == 1
    return events


def test_tend_s_sequence_is_the_pinned_stage_s_live_statements() -> None:
    assert ["pbuf_get_field*"] + _live_events() == list(SEQUENCE_WHOLE)
    # composed: each driver is its sub-walk plus the hand-over
    def undo(sequence, walk, driver):
        at = list(sequence).index(f"{walk}.tend_chunk")
        assert sequence[at + 1] == f"take_{driver.split('_')[0].removesuffix('p')}"
        return tuple(sequence[:at]) + (driver,) + tuple(sequence[at + 2:])

    assert undo(SEQUENCE_WHOLE_MICRO, "Macrophysics", "macrop_driver_tend") == SEQUENCE_WHOLE
    assert undo(SEQUENCE, "Microphysics", "microp_driver_tend") == SEQUENCE_WHOLE_MICRO


# -- the fake image -------------------------------------------------------------


class _Lib:
    PCOLS, PVER, PVERP, PCNST, NWSET = 16, 30, 31, 57, 3

    def __init__(self) -> None:
        self.views: dict[tuple, np.ndarray] = {}
        self.calls: list[str] = []
        self.owner = 0
        self.energy: list[tuple[str, int, int]] = []
        self.drivers: list[tuple[str, int, float, int]] = []

    def _shape(self, code: int) -> tuple[int, ...]:
        return {21: (self.PCOLS, self.PVER), 23: (self.PCOLS, self.PVER),
                22: (self.PCOLS, self.PVER, self.PCNST), 24: (self.PCOLS, self.PVER, self.PCNST),
                31: (self.PCOLS,), 32: (self.PCOLS,)}[code]

    def _forcing_shape(self, code: int) -> tuple[int, ...]:
        return {1: (self.PCOLS, self.PVER), 2: (self.PCOLS, self.PVERP), 3: (self.PCOLS, self.PVERP),
                4: (self.PCOLS, self.PVER), 5: (self.PCOLS, self.PVER), 6: (self.PCOLS,),
                7: (self.PCOLS, self.PVER, self.NWSET)}[code]

    def _serve(self, key, shape, ptr, ndims, extents):
        array = self.views.setdefault(key, np.zeros(shape, order="F"))
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
            if name == "pycam_mm_view_v1":
                lchnk, code, ptr, ndims, extents = args
                lib._serve((lchnk, "view", code), lib._shape(code), ptr, ndims, extents)
            elif name == "pycam_macro_forcing_v1":
                lchnk, code, ptr, ndims, extents = args
                lib._serve((lchnk, "forcing", code), lib._forcing_shape(code), ptr, ndims, extents)
            elif name == "pycam_pbuf_field_v2":
                lchnk, index, sliced, rank, is_int, ptr, ndims, extents = args
                assert rank == 1 and not is_int
                lib._serve((lchnk, "pbuf", index), (lib.PCOLS,), ptr, ndims, extents)
            elif name == "pycam_mm_set_owner_v1":
                lib.owner = args[0]
            elif name == "pycam_mm_nstep_v1":
                return 5
            elif name == "pycam_mm_dt_v1":
                return 1800
            elif name == "pycam_mm_check_energy_v1":
                lib.energy.append((args[1].decode(), args[5], args[10]))
            elif name in ("pycam_mm_macrop_driver_tend_v1", "pycam_mm_microp_aero_run_v1",
                          "pycam_mm_microp_driver_tend_v1"):
                lib.drivers.append((name, args[0], args[1], len(args) - 2))
            elif name in ("pycam_mm_take_macro_v1", "pycam_mm_take_micro_v1"):
                lib.drivers.append((name, args[0], None, 0))
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
        for name in CloudMacroMicrophysics.CAM_IN:
            self.pool[f"cam_in.{name}"] = np.zeros((16, 2), order="F")
        self.kernels: list[str] = []
        from freecam.pi_cam.kernel_codegen import load_direct_kernels

        self._args = {k.name: [{"field": a.field, "dtype": a.dtype, "rank": a.rank}
                               for a in k.arguments]
                      for k in load_direct_kernels(CloudMacroMicrophysics.DESCRIPTORS)}

    @property
    def chunks(self):
        return self.pool["grid.chunk_id"], self.pool["grid.chunk_ncols"]

    def kernel_arguments(self, name):
        return tuple(self._args[name])

    def run_kernel(self, name, arrays):
        self.kernels.append(name)
        if name == "mm_substep_dt":
            arrays["mm.cld_macmic_ztodt"][...] = 900.0        # a stand-in, not arithmetic


class _Context:
    def __init__(self, native):
        self.native = native
        self.timestep_seconds = 1800
        self.step = 5
        self.rank = 3


def _constants(**overrides) -> M._Constants:
    base = M._Constants(cld_macmic_num_steps=1, macrop_scheme="park", microp_scheme="MG",
                        micro_do_icesupersat=False, use_subcol_microp=False,
                        carma_do_cldice=False, carma_do_cldliq=False, trace_water=True)
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
    return native


def test_tend_walks_stage_seven_in_its_order_on_every_chunk(fake) -> None:
    scheme = CloudMacroMicrophysics(whole_drivers=True)
    assert scheme.macro is None and scheme.kernels == {}
    scheme.tend(None, _Context(fake))
    assert scheme.calls == list(SEQUENCE_WHOLE) * 2
    assert fake.library.owner == 1
    for name in KERNELS:
        assert fake.kernels.count(name) == 2, name
    lib = fake.library
    # the drivers, whole, with the substep length Fortran formed, once per chunk
    assert [d[0].removeprefix("pycam_mm_").removesuffix("_v1") for d in lib.drivers] == [
        "macrop_driver_tend", "microp_aero_run", "microp_driver_tend"] * 2
    assert {d[2] for d in lib.drivers} == {900.0}
    assert [d[3] for d in lib.drivers if "macrop" in d[0]] == [len(MACROP_ARGUMENTS)] * 2
    assert [d[1] for d in lib.drivers] == [1540] * 3 + [1541] * 3
    # both energy checks, in the substep-scaled form, with the count passed in
    assert lib.energy == [("macrop_tend", 1, 1), ("microp_tend", 1, 1)] * 2
    assert lib.calls.count("pycam_mm_ptend_scale_v1") == 4
    assert lib.calls.count("pycam_mm_update_v1") == 4
    assert lib.calls.count("pycam_mm_ptend_sum_aero_v1") == 2
    assert lib.calls.count("pycam_mm_wtrc_mass_fixer_v1") == 2


class _SubWalk:
    """Stands in for a sub-stage's tend_chunk: records what it was handed."""

    def __init__(self, monkeypatch, *classes) -> None:
        self.calls: list[tuple] = []
        self.runtimes: list = []
        walk = self

        def runtime(self, native):
            walk.runtimes.append(native)
            return type("RT", (), {"rank": 0, "nstep": 0})()

        def tend_chunk(self, st, lchnk, ncol, index, dt, nstep):
            self.calls.append(type(self).__name__)
            walk.calls.append((type(self).__name__, lchnk, ncol, index, dt, nstep))

        for cls in classes:
            monkeypatch.setattr(cls, "runtime", runtime)
            monkeypatch.setattr(cls, "tend_chunk", tend_chunk)


def test_composed_both_walks_run_in_their_drivers_places(fake, monkeypatch) -> None:
    walk = _SubWalk(monkeypatch, Macrophysics, Microphysics)
    scheme = CloudMacroMicrophysics()
    assert isinstance(scheme.macro, Macrophysics) and isinstance(scheme.micro, Microphysics)
    assert scheme.components == {"macro": scheme.macro, "micro": scheme.micro}
    # one kernels mapping: every sub-walk's swappable core is reachable from the stage
    assert scheme.kernels == {"mmacro_pcond": None, "micro_mg_tend": None}
    assert scheme.macro.kernels is scheme.kernels and scheme.micro.kernels is scheme.kernels
    scheme.tend(None, _Context(fake))
    assert scheme.calls == list(SEQUENCE) * 2
    # each sub-walk got the driver's dtime -- the substep length Fortran formed --
    # and the chunk it was asked for; the hand-over followed on the same chunk
    assert walk.calls == [("Macrophysics", 1540, 14, 0, 900.0, 5), ("Microphysics", 1540, 14, 0, 900.0, 5),
                          ("Macrophysics", 1541, 13, 1, 900.0, 5), ("Microphysics", 1541, 13, 1, 900.0, 5)]
    lib = fake.library
    names = [d[0].removeprefix("pycam_mm_").removesuffix("_v1") for d in lib.drivers]
    assert names == ["take_macro", "microp_aero_run", "take_micro"] * 2
    assert "pycam_mm_macrop_driver_tend_v1" not in lib.calls
    assert "pycam_mm_microp_driver_tend_v1" not in lib.calls
    # a sub-walk's own call log is per chunk, not per run
    assert scheme.macro.calls == ["Macrophysics"] and scheme.micro.calls == ["Microphysics"]


def test_whole_micro_keeps_gate_m2_s_form(fake, monkeypatch) -> None:
    walk = _SubWalk(monkeypatch, Macrophysics)
    scheme = CloudMacroMicrophysics(whole_micro=True)
    assert scheme.micro is None and scheme.kernels == {"mmacro_pcond": None}
    scheme.tend(None, _Context(fake))
    assert scheme.calls == list(SEQUENCE_WHOLE_MICRO) * 2
    names = [d[0].removeprefix("pycam_mm_").removesuffix("_v1") for d in fake.library.drivers]
    assert names == ["take_macro", "microp_aero_run", "microp_driver_tend"] * 2
    assert len(walk.calls) == 2


def test_a_model_assigned_on_the_stage_reaches_the_sub_walk() -> None:
    model = object()
    scheme = CloudMacroMicrophysics(kernels={"mmacro_pcond": model})
    assert scheme.macro.kernels["mmacro_pcond"] is model
    assert scheme.micro.kernels["mmacro_pcond"] is model      # one mapping
    scheme = CloudMacroMicrophysics(kernels={"micro_mg_tend": model})
    assert scheme.micro.kernels["micro_mg_tend"] is model
    with pytest.raises(PICAMConfigurationError, match="no swappable kernel"):
        CloudMacroMicrophysics(kernels={"rad_rrtmg_sw": model})
    with pytest.raises(PICAMConfigurationError, match="no swappable kernel"):
        CloudMacroMicrophysics(whole_drivers=True, kernels={"mmacro_pcond": model})
    with pytest.raises(PICAMConfigurationError, match="no swappable kernel"):
        CloudMacroMicrophysics(whole_micro=True, kernels={"micro_mg_tend": model})


def test_it_replaces_the_whole_action_and_has_no_swappable_kernel_yet() -> None:
    scheme = CloudMacroMicrophysics(whole_drivers=True)
    assert scheme.replaces_whole_action
    assert scheme.STAGE == "cam_run1.cloud_macro_microphysics"
    assert scheme.kernels == {}
    from freecam.pi_cam.plan import PICAMStepPlan

    action = PICAMStepPlan.default().select("cloud_macro_microphysics", phase="cam_run1")
    assert action.native_id == 427 and action.qualified_name == scheme.STAGE


@pytest.mark.parametrize("bad, what", [
    ({"microp_scheme": "RK"}, "stratiform_tend"),
    ({"macrop_scheme": "CLUBB_SGS"}, "clubb"),
    ({"micro_do_icesupersat": True}, "icesupersat"),
    ({"use_subcol_microp": True}, "subcol"),
    ({"carma_do_cldice": True}, "CARMA"),
    ({"cld_macmic_num_steps": 0}, "cld_macmic_num_steps"),
])
def test_the_paths_the_configuration_never_takes_are_refused(bad, what) -> None:
    with pytest.raises(PICAMConfigurationError, match=what):
        _constants(**bad).refuse_unsupported()
    _constants().refuse_unsupported()


def test_without_water_tracers_the_mass_fixer_is_not_called(fake, monkeypatch) -> None:
    constants = _constants(trace_water=False)
    monkeypatch.setattr(M._Constants, "read", classmethod(lambda cls, library: constants))
    scheme = CloudMacroMicrophysics(whole_drivers=True)
    scheme.tend(None, _Context(fake))
    assert "wtrc_mass_fixer" not in scheme.calls
    assert scheme.calls == [c for c in SEQUENCE_WHOLE if c != "wtrc_mass_fixer"] * 2
