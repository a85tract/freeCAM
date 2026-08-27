"""Radiation: the driver's statements, in the driver's order, from Python."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from freecam.physics import radiation as R
from freecam.physics.radiation import (
    LW, SEQUENCE_QUIET_STEP, SEQUENCE_RADIATION_STEP, SW, VIEW, Radiation,
)

REPO = Path(__file__).resolve().parents[2]
PINNED = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/rrtmg/radiation.F90"
HANDLES = REPO / "native/pi_cam/support/pycam_rad_handles.F90"

pinned = pytest.mark.skipif(not PINNED.is_file(),
                            reason="the pinned iCESM submodule is not checked out")


# -- tables that mirror Fortran --------------------------------------------------


def test_view_codes_equal_the_handles_module_s_table() -> None:
    fortran = {m.group(1).removeprefix("view_"): int(m.group(2))
               for m in re.finditer(r"parameter, public :: (view_\w+) = (\d+)",
                                    HANDLES.read_text())}
    assert VIEW == fortran


@pinned
def test_the_pinned_parameters_are_what_the_class_holds() -> None:
    """These are named constants, so the image has no symbol for them and the
    class carries the numbers.  A drift here is silent until a gate."""

    src = REPO / "external/iCESM1.3.1_fzhu/components/cam/src"
    # radconstants.F90 exists twice; the RRTMG one is what this image links
    constants = (src / "physics/rrtmg/radconstants.F90").read_text()
    for name, value in (("nbndsw", R.NBNDSW), ("nbndlw", R.NBNDLW),
                        ("idx_sw_diag", R.IDX_SW_DIAG),
                        ("rrtmg_sw_cloudsim_band", R.RRTMG_SW_CLOUDSIM_BAND),
                        ("rrtmg_lw_cloudsim_band", R.RRTMG_LW_CLOUDSIM_BAND)):
        found = re.search(rf"parameter, public :: {name}\s*=\s*(\d+)", constants)
        assert found, name
        assert int(found.group(1)) == value, (name, found.group(1), value)
    # the CAM-RT radconstants gives idx_sw_diag a different value; taking the
    # wrong file would be silent, so the two are held apart here
    camrt = (src / "physics/cam/radconstants.F90").read_text()
    assert int(re.search(r"idx_sw_diag\s*=\s*(\d+)", camrt).group(1)) != R.IDX_SW_DIAG

    # nbndsw and nbndlw are also in the RRTMG parameter modules the lifted
    # kernels and the handles use-associate, and the two definitions must
    # agree or an array would be shaped one way and read the other
    ext = src / "physics/rrtmg/ext"
    for path, name, value in ((ext / "rrtmg_sw/parrrsw.f90", "nbndsw", R.NBNDSW),
                              (ext / "rrtmg_lw/parrrtm.f90", "nbndlw", R.NBNDLW)):
        found = re.search(rf"{name}\s*=\s*(\d+)", path.read_text())
        assert found and int(found.group(1)) == value, (name, found and found.group(1))

    physconst = (src / "utils/physconst.F90").read_text()
    for name in ("cpair", "stebol"):
        assert re.search(rf"parameter\s*::\s*{name}\s*=\s*shr_const_", physconst), name
    assert not re.search(r"parameter\s*::\s*cappa\s*=", physconst)
    assert "physconst_mp_cappa_" in (REPO / "src/freecam/physics/radiation.py").read_text()


# -- the sequence ----------------------------------------------------------------


#: Regions the admitted configuration never enters, each refused at attach
#: with the module state that decides it.
DEAD = (
    (830, 835),      # spectralflux: the four spectral-flux pbuf fields
    (838, 842),      # single_column .and. scm_crm_mode: the CRM cloud override
    (871, 873),      # hist_fld_active('FSNR'/'FLNR'): the tropopause lookup
    (891, 893),      # oldcldoptics, shortwave
    (896, 897),      # icecldoptics == 'ebertcurry'
    (904, 905),      # liqcldoptics == 'slingo'
    (949, 951),      # oldcldoptics, longwave
    (954, 955),      # ice ebertcurry, longwave
    (962, 963),      # liquid slingo, longwave
    (1052, 1056),    # hist_fld_active('FSNR'): the per-column interpolation
    (1163, 1167),    # hist_fld_active('FLNR'): the same, longwave
    (1201, 1236),    # dohirs: hirsrtm and the gases it needs
    (1259, 1273),    # docosp: the COSP simulator
)

def _carved() -> dict[int, tuple[list[str], str]]:
    """first line -> (calls Python makes itself first, the routine's name).

    A lifted block may enclose a call Python has to make before the block
    runs -- the snow optics fill an array the blend then reads -- so the
    generator records those lines in ``Block.skip``.  Reading that here is
    what keeps this test and the lift from drifting apart.
    """

    import sys

    sys.path.insert(0, str(REPO / "tools"))
    import generate_pi_cam_rad_kernels as gen

    lines = PINNED.read_text().splitlines()
    out: dict[int, tuple[list[str], str]] = {}
    for block in gen.blocks(lines):
        if block.name == "rad_inp":
            continue                       # a routine, not driver-body lines
        first = []
        for number in sorted(block.skip):
            match = re.match(r"call\s+(\w+)", lines[number - 1].split("!")[0].strip())
            if match:
                first.append(match.group(1))
        out[block.first] = (first, block.name)
    # 1063 is rad_scale_by_cpair a second time, over qrsc
    out[1063] = ([], "rad_scale_by_cpair")
    return out


#: Calls tend_chunk does not make per chunk, with the reason.
SILENT = {
    "t_startf", "t_stopf", "endrun",
    # the active-call list is module state that cannot change within a run,
    # so Python reads it once at attach and asserts only the climate call is
    # on, rather than asking again for every chunk of every step
    "rad_cnst_get_call_list",
}

#: What Python calls each of the driver's calls, where the names differ.
RENAMED = {
    "get_curr_calday": "calday",
    "get_rlat_all_p": "latlon", "get_rlon_all_p": None,   # one entry does both
    "rrtmg_state_create": "rrtmg_state_create",
    "rrtmg_state_update": "rrtmg_state_update",
    "rrtmg_state_destroy": "rrtmg_state_destroy",
}


def _dead(number: int) -> bool:
    return any(a <= number <= b for a, b in DEAD)


def _fortran_sequence(*, radiative: bool) -> list[str]:
    """The driver's calls, 807-1320, as tend_chunk names them."""

    lines = PINNED.read_text().splitlines()
    carved = _carved()
    spans = {first: first for first in carved}
    import sys

    sys.path.insert(0, str(REPO / "tools"))
    import generate_pi_cam_rad_kernels as gen

    for block in gen.blocks(lines):
        if block.name != "rad_inp":
            spans[block.first] = block.last

    names: list[str] = []
    skip_to = 0
    # the branch a step of this kind does NOT take
    skip = (1275, 1288) if radiative else (875, 1274)
    for number in range(807, 1321):
        if number <= skip_to:
            continue
        if _dead(number) or skip[0] <= number <= skip[1]:
            continue
        if number in carved:
            before, name = carved[number]
            names.extend(before)
            names.append(name)
            skip_to = spans.get(number, number)
            continue
        text = lines[number - 1].split("!")[0].strip()
        if re.match(r"dosw\s*=\s*radiation_do", text):
            names.append("radiation_do:sw"); continue
        if re.match(r"dolw\s*=\s*radiation_do", text):
            names.append("radiation_do:lw"); continue
        if re.match(r"calday\s*=\s*get_curr_calday", text):
            names.append("calday"); continue
        if "=> rrtmg_state_create(" in text:
            names.append("rrtmg_state_create"); continue
        match = re.match(r"call\s+(\w+)\s*\(", text)
        if not match:
            continue
        name = match.group(1)
        if name in SILENT:
            continue
        if name in RENAMED:
            renamed = RENAMED[name]
            if renamed is None:
                continue
            name = renamed
        if name == "outfld" and "/cpair" in text:
            names.append("outfld_scaled")
            continue
        if name in ("outfld", "pbuf_get_field", "vertinterp"):
            star = name + "*"
            if names and names[-1] == star:
                continue
            names.append(star)
            continue
        names.append(name)
    return names


@pinned
def test_the_radiation_step_sequence_is_the_fortran_driver_s_call_order() -> None:
    assert list(SEQUENCE_RADIATION_STEP) == _fortran_sequence(radiative=True)


@pinned
def test_the_quiet_step_sequence_is_the_fortran_driver_s_other_branch() -> None:
    assert list(SEQUENCE_QUIET_STEP) == _fortran_sequence(radiative=False)


@pinned
def test_both_sequences_share_the_work_outside_the_branch() -> None:
    """Everything before `if (dosw .or. dolw)` and after its `end if` runs on
    every step, so the two sequences must agree there."""

    head = SEQUENCE_RADIATION_STEP[:SEQUENCE_RADIATION_STEP.index("rrtmg_state_create")]
    assert list(head) == list(SEQUENCE_QUIET_STEP[:len(head)])
    tail = ("rad_data_write", "radheat_tend", "rad_theta_heating", "outfld*",
            "rad_heating_scale")
    assert tuple(SEQUENCE_RADIATION_STEP[-len(tail):]) == tail
    assert tuple(SEQUENCE_QUIET_STEP[-len(tail):]) == tail


# -- the class's shape -----------------------------------------------------------


def test_the_two_cores_are_the_swappable_kernels_and_nothing_else_is() -> None:
    scheme = Radiation()
    assert list(scheme.kernels) == [SW, LW]
    assert all(v is None for v in scheme.kernels.values())
    # a stage with two refuses the singular attribute rather than guessing
    from freecam.physics.errors import PhysicsError

    with pytest.raises(PhysicsError, match="assign into .kernels"):
        scheme.kernel


def test_the_ten_numerical_routines_are_public_methods() -> None:
    for name in ("ice_optics_sw", "liquid_optics_sw", "snow_optics_sw",
                 "ice_props_lw", "liquid_props_lw", "snow_props_lw",
                 "aer_props_sw", "aer_props_lw", "rad_rrtmg_sw", "rad_rrtmg_lw"):
        assert callable(getattr(Radiation, name)), name
        assert not name.startswith("_"), name


def test_the_stage_declares_no_ptend_services_it_does_not_have() -> None:
    """radheat_tend builds the ptend, so radiation copies no physics state."""

    from freecam.physics.stage import CORE_ENTRIES, PTEND_ENTRIES

    table = R._RadEntries.TABLE
    for name in CORE_ENTRIES:
        assert name in table, name
    for name in PTEND_ENTRIES:
        assert name not in table, name


def test_every_kernel_the_stage_runs_is_in_the_reviewed_descriptors() -> None:
    from freecam.pi_cam.kernel_codegen import load_direct_kernels

    described = {k.name for k in load_direct_kernels(Radiation.DESCRIPTORS)}
    for name in Radiation.KERNELS:
        assert name in described, name
    # zenith is reached through a handle, not a kernel, because it is a bare
    # external subroutine
    assert "zenith" not in Radiation.KERNELS
    assert "zenith" in R._RadEntries.TABLE


def test_attach_swaps_the_stage_for_its_halves_and_sits_between_them() -> None:
    class Action:
        def __init__(self): self.enabled = None
        def enable(self, **_): self.enabled = True
        def disable(self, **_): self.enabled = False

    class Workflow:
        def __init__(self):
            self.items = {R.STAGE: Action(), R.FIRST_HALF: Action(),
                          R.SECOND_HALF: Action()}
            self.inserted = []
        def process(self, name): return self.items[name]
        def insert_after(self, anchor, process):
            self.inserted.append((anchor, process)); return process

    class Run:
        workflow = Workflow()

    handle = Radiation().attach(Run)
    assert Run.workflow.items[R.STAGE].enabled is False
    assert Run.workflow.items[R.FIRST_HALF].enabled is True
    assert Run.workflow.items[R.SECOND_HALF].enabled is True
    assert Run.workflow.inserted == [(R.FIRST_HALF, handle)]
    assert handle.name == "rad_tend"
    assert handle.native is True and handle.transactional is False


def test_tend_refuses_to_run_as_an_ordinary_process() -> None:
    from freecam.physics.errors import PhysicsError

    class Context:
        native = None

    with pytest.raises(PhysicsError, match="native"):
        Radiation().tend(None, Context())


# -- the refusals ----------------------------------------------------------------


def _constants(**overrides):
    base = dict(
        qrs_idx=1, qrl_idx=2, cld_idx=3, cldfsnow_idx=4, su_idx=0, sd_idx=0,
        lu_idx=0, ld_idx=0, iradsw=-1, iradlw=-1, irad_always=0,
        spectralflux=False, dohirs=False, docosp=False, num_rrtmg_levs=29,
        cappa=0.2857,
    )
    base.update(overrides)
    return R._Constants(**base)


def test_the_admitted_configuration_is_accepted() -> None:
    _constants().refuse_unsupported()


@pytest.mark.parametrize("overrides,message", [
    ({"spectralflux": True}, "spectralflux"),
    ({"docosp": True}, "COSP"),
    ({"dohirs": True}, "hirsrtm"),
    ({"oldcldoptics": True}, "oldcldoptics"),
    ({"icecldoptics_mitchell": False}, "icecldoptics"),
    ({"liqcldoptics_gammadist": False}, "liqcldoptics"),
    ({"active_calls": 3}, "radiation calls are active"),
    ({"fsnr_active": True}, "FSNR or FLNR"),
    ({"flnr_active": True}, "FSNR or FLNR"),
])
def test_every_path_the_transliteration_cannot_carry_is_refused(overrides, message) -> None:
    from freecam.pi_cam.errors import PICAMConfigurationError

    with pytest.raises(PICAMConfigurationError, match=message):
        _constants(**overrides).refuse_unsupported()


# -- the scratch the transliteration reads ---------------------------------------


def test_every_scratch_name_tend_chunk_reads_is_one_the_runtime_allocates() -> None:
    """A name that is read but never allocated is a KeyError 512 ranks deep,
    and the shapes only exist in two places -- the descriptors and
    EXTRA_SCRATCH -- so the two can be compared here instead."""

    from freecam.pi_cam.kernel_codegen import load_direct_kernels

    described = {k.name: k for k in load_direct_kernels(Radiation.DESCRIPTORS)}
    allocated = {a.field.removeprefix("rad.")
                 for name in Radiation.KERNELS for a in described[name].arguments}
    allocated |= {name for name, _ in Radiation.EXTRA_SCRATCH}

    source = (REPO / "src/freecam/physics/radiation.py").read_text()
    read = set(re.findall(r'L\["(\w+)"\]', source))
    read |= set(re.findall(r'st\.local\["(\w+)"\]', source))
    assert not read - allocated, sorted(read - allocated)


def test_no_extra_scratch_is_declared_that_nothing_reads() -> None:
    """EXTRA_SCRATCH is for what no kernel declares, so an entry nothing reads
    is either a leftover or a name that drifted."""

    source = (REPO / "src/freecam/physics/radiation.py").read_text()
    read = set(re.findall(r'L\["(\w+)"\]', source))
    read |= set(re.findall(r'st\.local\["(\w+)"\]', source))
    unread = sorted(name for name, _ in Radiation.EXTRA_SCRATCH if name not in read)
    assert not unread, unread


def test_extra_scratch_never_shadows_a_field_a_kernel_already_declares() -> None:
    """Two shapes for one name is a silent mismatch: the allocation wins and
    the kernel reads it as something else."""

    from freecam.pi_cam.kernel_codegen import load_direct_kernels

    described = {k.name: k for k in load_direct_kernels(Radiation.DESCRIPTORS)}
    from_kernels = {a.field.removeprefix("rad.")
                    for name in Radiation.KERNELS for a in described[name].arguments}
    clashes = sorted(name for name, _ in Radiation.EXTRA_SCRATCH if name in from_kernels)
    assert not clashes, clashes


def test_every_extent_a_kernel_declares_is_one_the_stage_can_resolve() -> None:
    """An extent name the runtime cannot resolve is a ValueError when the
    scratch is allocated, 512 ranks deep.  radconstants and the RRTMG
    parameter modules name the same band counts differently, which is exactly
    how sfac's `nswbands` slipped past a first reading."""

    from freecam.pi_cam.kernel_codegen import load_direct_kernels

    class _C:
        num_rrtmg_levs = 29

    described = {k.name: k for k in load_direct_kernels(Radiation.DESCRIPTORS)}
    resolvable = {"pcols", "pver", "pverp", "pcnst", "chunks"}
    resolvable |= set(Radiation().extra_extents(_C()))
    declared = {extent for name in Radiation.KERNELS
                for a in described[name].arguments for extent in a.extents}
    unresolved = sorted(e for e in declared if e not in resolvable and not e.isdigit())
    assert not unresolved, unresolved


def test_the_two_names_for_the_band_counts_agree() -> None:
    extents = Radiation().extra_extents(type("C", (), {"num_rrtmg_levs": 29})())
    assert extents["nswbands"] == extents["nbndsw"] == R.NBNDSW
    assert extents["nlwbands"] == extents["nbndlw"] == R.NBNDLW


def test_a_scalar_scratch_is_never_indexed_as_if_it_had_a_lane() -> None:
    """A dummy declared without dimensions has extents ("chunks",), so the
    local view drops to 0-d and `[0]` raises.  Gate R-B2 spent a 512-rank run
    finding that for Nday."""

    from freecam.pi_cam.kernel_codegen import load_direct_kernels

    described = {k.name: k for k in load_direct_kernels(Radiation.DESCRIPTORS)}
    extents: dict[str, tuple[str, ...]] = {}
    for name in Radiation.KERNELS:
        for a in described[name].arguments:
            extents.setdefault(a.field.removeprefix("rad."), tuple(a.extents))
    scalars = {name for name, e in extents.items() if e == ("chunks",)}
    assert scalars, "the descriptors declare no scalar dummies at all"

    source = (REPO / "src/freecam/physics/radiation.py").read_text()
    wrong = [(m.group(1), m.group(2))
             for m in re.finditer(r'L\["(\w+)"\]\s*\[([^\]]*)\]', source)
             if m.group(1) in scalars and m.group(2) != "()"]
    assert not wrong, wrong
