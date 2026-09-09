"""The pausable runners: generated from their specs, complete over the pinned text, decodable by Python."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import re

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import pi_cam_pausable as pausable  # noqa: E402
import generate_pi_cam_pausable_runners as generator  # noqa: E402

SPECS = sorted(pausable.SPECS.glob("*.yaml"))


@pytest.mark.parametrize("path", SPECS, ids=[p.stem for p in SPECS])
def test_the_committed_modules_are_what_the_generator_writes_and_the_source_is_where_it_was(path) -> None:
    spec = pausable.load_spec(path)
    recorded = yaml.safe_load(path.read_text())["anchors"]
    assert recorded == pausable.spec_digest(spec), "the pinned source moved under the spec"
    for target, text in pausable.render_all(spec).items():
        assert target.read_text() == text, f"{target.name} is stale; run tools/generate_pi_cam_pausable_runners.py"


@pytest.mark.parametrize("path", SPECS, ids=[p.stem for p in SPECS])
def test_every_executable_line_of_a_unit_is_a_piece_a_skeleton_line_or_a_pause(path) -> None:
    spec = pausable.load_spec(path)
    for unit in spec.units.values():
        assert pausable.coverage_gaps(unit) == [], unit.key


def test_the_frame_table_is_current_and_names_every_argument_but_the_character_ones() -> None:
    table = yaml.safe_load(pausable.FRAMES.read_text())["kernels"]
    expected: dict = {}
    for path in SPECS:
        expected.update(pausable.frame_descriptors(pausable.load_spec(path)))
    assert table == expected
    dadadj = [s["name"] for s in table["dadadj"]]
    assert dadadj == ["lchnk", "ncol", "pmid", "pint", "pdel", "t", "q"]
    assert [s["intent"] for s in table["dadadj"]] == ["in", "in", "in", "in", "in", "inout", "inout"]
    assert table["dadadj"][6]["actual"] == "ptend%q(1,1,1)"       # sequence association, served with the callee's shape
    uw = table["compute_uwshcu_inv"]
    assert len(uw) == 54 and uw[0]["name"] == "mix" and uw[0]["actual"] == "pcols"
    assert [s for s in uw if s["name"] == "tke_inv"][0]["kind"] == "pointer"
    assert [s for s in uw if s["name"] == "tr0_inv"][0]["rank"] == 3


def test_the_runner_modules_carry_the_abi_and_the_original_entry() -> None:
    for prefix in ("dadadj", "shcu"):
        text = (pausable.SUPPORT / f"pycam_{prefix}_runner.F90").read_text()
        for suffix in ("create", "start", "frame", "resume", "original", "error", "reset", "destroy"):
            assert f"bind(C, name='pycam_{prefix}_{suffix}_v1')" in text, (prefix, suffix)
        # no callbacks into Python: the one procedure pointer a hooked runner takes is its own
        # fiber body, handed to the C context switch
        assert "c_f_procpointer" not in text.lower()
        if "c_funptr" in text.lower():
            assert "c_funloc(fiber_body)" in text and "hook_of(nkernels)" in text
            assert text.lower().count("c_funloc(") == 1


def test_the_manifest_names_the_pausable_runners_and_python_decodes_their_frames() -> None:
    from freecam.pi_cam import segment_runner as runners

    spec = runners.runner_spec("cam_run1.dry_adjustment")
    assert spec is not None and spec.original and spec.kernel_names == ("dadadj",)
    assert spec.kernel("dadadj").frame == "native/pi_cam/segment_frames.yaml"
    assert runners.frame_names_from_descriptor(REPO / spec.kernel("dadadj").frame, "dadadj") == (
        "lchnk", "ncol", "pmid", "pint", "pdel", "t", "q")
    shallow = runners.runner_spec("cam_run1.shallow_convection")
    assert shallow is not None and shallow.kernel_names == ("compute_uwshcu_inv", "fluxbelowinv")
    assert set(runners.bindable_kernels()) >= {"mmacro_pcond", "micro_mg_tend", "dadadj", "compute_uwshcu_inv"}
    # an image without the original entry is refused when the manifest promises it
    lib = SimpleNamespace(**{f"pycam_dadadj_{s}_v1": object() for s in runners.ENTRY_SUFFIXES})
    with pytest.raises(Exception, match="exports no pycam_dadadj_original_v1"):
        runners.ImageSegmentRunner(lib, spec)


def test_the_original_at_the_pause_exercises_the_write_back() -> None:
    from freecam.physics.segments import FrameArgument, KernelFrame, OriginalAtPause

    t = np.full((8, 4), 1.5, order="F")
    q = np.full((8, 4), 2.5, order="F")
    pmid = np.ones((8, 4), order="F")
    frame = KernelFrame(kernel="dadadj", call_index=1, lchnk=1, ncol=6, substep=1, token=3, arguments=(
        FrameArgument("pmid", pmid, "in"), FrameArgument("t", t, "inout"), FrameArgument("q", q, "inout")))
    ran = []

    class Runner:
        def run_original(self, context, kernel):
            ran.append((context, kernel))
            t[:6] += 1.0                                   # what the original does

    answer = OriginalAtPause()(frame, Runner(), 7)
    assert ran == [(7, "dadadj")]
    assert set(answer) == {"t", "q"} and np.all(answer["t"] == 2.5) and np.all(answer["q"] == 2.5)
    assert np.all(t[:6] == 0.0) and np.all(t[6:] == 1.5)   # zeroed, so the write-back is what restores it
    frame.write_back(answer)
    assert np.all(t[:6] == 2.5) and np.all(q[:6] == 2.5)


def test_a_pausable_stage_runs_whole_when_nothing_is_replaced_and_refuses_the_walk() -> None:
    from freecam.physics.errors import PhysicsError
    from freecam.physics.pausable import STAGES, DryAdjustment, InertStage, ShallowConvection

    stage = DryAdjustment()
    assert stage.WHOLE_ACTION and stage.SWAPPABLE == ("dadadj",) and stage.STAGE == "cam_run1.dry_adjustment"
    assert stage.select_mode(None) == "native-whole"
    stage.kernels["dadadj"] = lambda batch: {}
    covering = SimpleNamespace(segment_runner=lambda stage: SimpleNamespace(kernels=("dadadj",)))
    assert stage.select_mode(covering) == "segmented"
    with pytest.raises(PhysicsError, match="no walk to fall back to"):
        stage.select_mode(SimpleNamespace(segment_runner=lambda stage: None))
    stage.execution_policy = "legacy-python"
    with pytest.raises(PhysicsError, match="no statement-by-statement Python walk"):
        stage.select_mode(None)
    assert ShallowConvection().SWAPPABLE == ("compute_uwshcu_inv", "fluxbelowinv")
    inert = [cls for cls in STAGES.values() if issubclass(cls, InertStage)]
    assert len(inert) == 11 and all(cls.SWAPPABLE == () for cls in inert)
    assert {cls.STAGE for cls in inert} == {
        "cam_run2.rayleigh_friction", "cam_run2.charge_neutrality", "cam_run2.qbo_relaxation", "cam_run2.ion_drag",
        "cam_run1.sea_salt_rebin", "cam_run1.modal_aerosol_preparation_leaf", "cam_run1.carma_wet_deposition_leaf",
        "cam_run2.carma_aerosol_tendencies_leaf", "cam_run2.carma_statistics_leaf", "cam_run2.tracer_tendencies_leaf",
        "cam_run2.age_of_air_tendencies_leaf"}
    rows = {r["kernel"]: r for r in stage.describe_kernels()}
    assert rows["dadadj"]["bindable"] and rows["dadadj"]["validated"]      # gates 7333952 and 7333955
    assert rows["dadadj"]["contract"]["path"] == "native/pi_cam/functions/dadadj.yaml"


def test_a_pausable_stage_drives_a_fake_runner_with_the_original_at_the_pause() -> None:
    from freecam.physics.pausable import DryAdjustment
    from freecam.physics.segments import OriginalKernel, SegmentEvent

    class FakeRunner:
        kernels = ("dadadj",)
        runs_original = True

        def __init__(self):
            self.log = []
            self.t = np.full((8, 4), 3.0, order="F")

        def create(self, stage): self.log.append("create"); return 1
        def start(self, cid, mask): self.log.append(("start", dict(mask))); self.pending = 2; return SegmentEvent.NEEDS_PYTHON_KERNEL
        def frame(self, cid):
            from freecam.physics.segments import FrameArgument, KernelFrame
            return KernelFrame(kernel="dadadj", call_index=1, lchnk=1, ncol=6, substep=1, token=self.pending,
                               arguments=(FrameArgument("t", self.t, "inout"),))
        def resume(self, cid, kernel, token):
            self.log.append(("resume", kernel, token)); self.pending -= 1
            return SegmentEvent.NEEDS_PYTHON_KERNEL if self.pending > 0 else SegmentEvent.DONE
        def run_original(self, cid, kernel): self.log.append(("original", kernel)); self.t[:6] += 1.0
        def error(self, cid): return ""
        def reset(self, cid): pass
        def destroy(self, cid): pass

    runner = FakeRunner()
    library = SimpleNamespace(pycam_stagehost_bind_v1=lambda: 0)
    native = SimpleNamespace(segment_runner=lambda stage: runner, library=library, run_action=lambda *a, **k: None)
    stage = DryAdjustment()
    stage.kernels["dadadj"] = OriginalKernel()
    stage.tend(None, SimpleNamespace(native=native))
    assert stage.execution.mode == "segmented"
    assert stage.execution.describe()["python_model_calls_by_kernel"] == {"dadadj": 2}
    assert [e for e in runner.log if e[0] == "original"] == [("original", "dadadj")] * 2
    assert np.all(runner.t[:6] == 5.0) and np.all(runner.t[6:] == 3.0)     # both pauses ran and wrote back


def test_the_radiation_frames_serve_the_rrtmg_state_by_component_and_arrays_from_their_lower_bounds() -> None:
    frames = yaml.safe_load(pausable.FRAMES.read_text())["kernels"]
    sw = {s["name"]: s for s in frames["rad_rrtmg_sw"]}
    lw = {s["name"]: s for s in frames["rad_rrtmg_lw"]}
    # the derived-type dummy is served component by component, each the core's body names
    assert {n for n in sw if n.startswith("r_state.")} == {f"r_state.{c}" for c in (
        "h2ovmr", "o3vmr", "co2vmr", "ch4vmr", "o2vmr", "n2ovmr", "pmidmb", "pintmb", "tlay", "tlev")}
    assert sum(n.startswith("r_state.") for n in lw) == 14 and lw["r_state.cfc11vmr"]["intent"] == "in"
    # a literal actual is a scalar slot; the spectral-flux pointers are slots served empty while null
    assert sw["old_convert"] == {"name": "old_convert", "actual": ".false.", "rank": 0, "dtype": "int32",
                                 "intent": "in", "kind": "scalar"}
    assert sw["su"]["kind"] == "pointer" and lw["lu"]["kind"] == "pointer" and sw["qrs"]["intent"] == "out"
    assert len(frames["rad_rrtmg_sw"]) == 58 and len(frames["rad_rrtmg_lw"]) == 35
    driver = (pausable.SUPPORT / "pycam_radt_driver.F90").read_text()
    # aer_tau(pcols,0:pver,nbndsw) starts at its declared lower bounds
    assert "c_loc(aer_tau(1,0,1))" in driver and "c_loc(aer_lw_abs(1,1,1))" in driver
    assert "if (associated(su)) then" in driver and "merge(1, 0, .false.)" in driver
    # the module's private variables and procedures the body needs, verbatim
    assert "logical :: spectralflux  = .false." in driver and "subroutine radinp(" in driver
    runner = (pausable.SUPPORT / "pycam_radt_runner.F90").read_text()
    assert "use rad_constituents, only: N_DIAG" in runner and "if (su_idx > 0) then" in runner
    assert runner.index("call driver_resolve_indices()") < runner.index("if (su_idx > 0) then")


def test_the_radiation_class_runs_between_its_halves_through_the_runner() -> None:
    import ctypes

    from freecam.physics.errors import PhysicsError
    from freecam.physics.radiation import Radiation
    from freecam.physics.segments import OriginalKernel

    calls: list = []

    class Library:
        def pycam_rad_set_owner_v1(self, owns):
            calls.append(("owner", owns.value if isinstance(owns, ctypes.c_int) else owns)); return 0

        def pycam_rad_bind_hosts_v1(self):
            calls.append(("rad_hosts",)); return 0

        def pycam_stagehost_bind_v1(self):
            calls.append(("stage_hosts",)); return 0

    class Native:
        library = Library()

        def segment_runner(self, stage):
            return SimpleNamespace(kernels=("rad_rrtmg_sw", "rad_rrtmg_lw"), runs_original=True) \
                if stage == "cam_run1.radiation" else None

        def run_action(self, stage):
            raise AssertionError("a split stage has no whole action to run")

    stage = Radiation()
    assert stage.SPLIT_RUNNER and not stage.WHOLE_ACTION
    native = Native()
    # nothing replaced: the step is left to the resume half, which calls the driver itself
    assert stage.select_mode(native) == "native-whole"
    stage.tend(None, SimpleNamespace(native=native))
    assert calls == [("owner", 0)] and stage.execution.native_stage_calls == 1
    # a replaced core the runner pauses at: segmented, and the runner needs both host bindings
    stage.kernels["rad_rrtmg_sw"] = OriginalKernel()
    assert stage.select_mode(native) == "segmented"
    stage.execution_policy = "segmented"
    assert stage.select_mode(native) == "segmented"          # allowed for a split stage with a runner
    calls.clear()
    stage.prepare_segmented(native)
    assert calls == [("stage_hosts",), ("rad_hosts",)]
    calls.clear()
    stage.after_segmented(native)
    assert calls == [("owner", 1)]
    # native-whole is refused while a core is replaced, as for any stage
    stage.execution_policy = "native-whole"
    with pytest.raises(PhysicsError):
        stage.select_mode(native)


def test_the_deep_convection_frames_address_the_module_state_and_answerable_scalars() -> None:
    frames = yaml.safe_load(pausable.FRAMES.read_text())["kernels"]
    zm = {s["name"]: s for s in frames["zm_convr"]}
    tran = {s["name"]: s for s in frames["convtran"]}
    assert len(frames["zm_convr"]) == 70 and len(frames["zm_conv_evap"]) == 22
    assert len(frames["momtran"]) == 25 and len(frames["convtran"]) == 21
    # the gathered column count is an intent(out) scalar of zm_convr: served where it lives
    assert zm["lengath"] == {"name": "lengath", "actual": "lengath(lchnk)", "rank": 0, "dtype": "int32",
                             "intent": "inout", "kind": "scalar"}
    # an expression actual stays a copy; the organisation pointers are empty slots while unused
    assert zm["delt"]["actual"] == ".5_r8*ztodt" and zm["org"]["kind"] == "pointer"
    assert tran["mu"]["actual"] == "mu(:,:,lchnk)" and tran["doconvtran"]["actual"] == "ptend%lq"
    deep = (pausable.SUPPORT / "pycam_zmdeep_zm.F90").read_text()
    leaf = (pausable.SUPPORT / "pycam_zmtran_zm2.F90").read_text()
    # zm_conv_intr's per-chunk arrays (control patch 0044) are addressed through a TARGET dummy
    assert "use zm_conv_intr, only: mu, eu, du, md, ed, dp, dsubcld, jt, maxg, ideep, lengath" in deep
    # a section goes to an assumed-shape helper of its rank; an element (momtran's mu(1,1,lchnk)) and a
    # scalar (the gathered count) to the sequence-association helper
    assert "call slot_address_r8_2(mu(:,:,lchnk), address)" in deep and "call slot_address_i4(lengath(lchnk), address)" in deep
    assert "call slot_address_r8(mu(1,1,lchnk), address)" in deep
    assert "call slot_address_r8_2(mu(:,:,lchnk), address)" in leaf and "call slot_address_i4_1(jt(:,lchnk), address)" in leaf
    # a `:pcols` section starts at its lower bound; an automatic array is allocated once and sized by itself
    assert "c_loc(state1%q(1,1,1))" in deep
    assert "allocatable, save, target, public :: rwt(:,:,:,:)" in deep.lower()
    assert "if (.not. allocated(rwt)) allocate(rwt(pcols,pver,wtrc_nwset,2))" in deep.lower()
    assert "int(size(rwt,3), c_int64_t)" in leaf.lower()
    # a routine without an ncol local reports the state's
    assert "ncol_out = int(state%ncol, c_int)" in leaf
    glue = (pausable.SUPPORT / "pycam_zmdeep_glue.F90").read_text()
    # the block's timer is closed after the last piece; the sub-column switch is a typed option
    assert "subroutine glue_leave()" in glue and "call t_stopf('moist_convection')" in glue
    assert "logical, save, public :: use_subcol_microp = .false." in glue
    assert "call phys_getopts(use_subcol_microp_out=use_subcol_microp)" in glue
    runner = (pausable.SUPPORT / "pycam_zmdeep_runner.F90").read_text()
    assert "use phys_control, only: cam_physpkg_is" in runner and "call glue_leave()" in runner
    tran_runner = (pausable.SUPPORT / "pycam_zmtran_runner.F90").read_text()
    assert "if (deep_scheme_does_scav_trans()) then" in tran_runner


def test_the_deep_convection_classes_own_their_actions_and_kernels() -> None:
    from freecam.physics.pausable import STAGES, ConvectiveTracerTransport, DeepConvection

    assert STAGES["deep_convection"] is DeepConvection and STAGES["convective_tracer_transport_leaf"] is ConvectiveTracerTransport
    assert DeepConvection.SWAPPABLE == ("zm_convr", "zm_conv_evap", "momtran", "cldfrc_fice") and DeepConvection.WHOLE_ACTION
    assert ConvectiveTracerTransport.SWAPPABLE == ("convtran",)
    assert ConvectiveTracerTransport.STAGE == "cam_run1.convective_tracer_transport_leaf"
    rows = {r["kernel"]: r for r in DeepConvection().describe_kernels()}
    assert set(rows) == set(DeepConvection.SWAPPABLE) and all(r["bindable"] for r in rows.values())
    # gates 7334212, 7334213 and 7335519 answered each kernel through its pause; 7335520 all three at once;
    # cldfrc_fice, hooked inside the compiled zm_conv_evap, is validated through gate 7343708
    assert all(rows[name]["validated"] for name in ("zm_convr", "zm_conv_evap", "momtran"))
    assert rows["cldfrc_fice"]["validated"] and rows["cldfrc_fice"]["validated_by"][:2] == [
        "validation/pi_cam_pausable_fice-hook_50step.json", "validation/pi_cam_pausable_fice-hook_vs_oracle_50step_bfb.json"]
    assert len(rows["cldfrc_fice"]["validated_by"]) == 6         # + the capture run and the everything run
    leaf = {r["kernel"]: r for r in ConvectiveTracerTransport().describe_kernels()}
    assert leaf["convtran"]["bindable"] and leaf["convtran"]["validated"]        # gates 7335521, 7335522


def test_the_tphysac_frames_serve_every_site_alike_and_size_automatic_arrays_per_chunk() -> None:
    frames = yaml.safe_load(pausable.FRAMES.read_text())["kernels"]
    vdiff = {s["name"]: s for s in frames["compute_vdiff"]}
    gw = {s["name"]: s for s in frames["gw_drag_prof"]}
    assert len(frames["compute_vdiff"]) == 42 and len(frames["gw_drag_prof"]) == 33
    # the solver's two sites serve one frame: the optional the dry site omits is an empty slot,
    # a selector with private components is opaque, procedure arguments are not slots
    assert vdiff["kvt"]["kind"] == "absent" and vdiff["fieldlist"]["kind"] == "opaque"
    assert "compute_molec_diff" not in vdiff and "vd_lu_qdecomp" not in vdiff and "errstring" not in vdiff
    driver = (pausable.SUPPORT / "pycam_vdiff_driver.F90").read_text()
    assert "subroutine compute_vdiff_1_frame" in driver and "subroutine compute_vdiff_2_original" in driver
    # a module pointer array's section is addressed through an assumed-shape helper of its rank
    assert "call slot_address_r8_2(cpairv(:,:,state%lchnk), address)" in driver
    assert "use shr_kind_mod, only: i4=> shr_kind_i4" in driver
    runner = (pausable.SUPPORT / "pycam_vdiff_runner.F90").read_text()
    assert "case (pc_at_compute_vdiff_1)" in runner and "case (pc_at_compute_vdiff_2)" in runner
    assert "if (trim(eddy_scheme) /= 'diag_TKE') then" in runner
    glue = (pausable.SUPPORT / "pycam_vdiff_glue.F90").read_text()
    assert "pycam_ac_carry_v1" in glue and "logical, save, public :: do_clubb_sgs = .false." in glue
    # the band and the coordinates by component; the optional adjustment an empty slot
    assert {n for n in gw if n.startswith("band.")} == {"band.ngwv", "band.kwv", "band.effkwv"}
    assert {n for n in gw if n.startswith("p.")} == {"p.del", "p.rdel"} and gw["ro_adjust"]["kind"] == "absent"
    gwd = (pausable.SUPPORT / "pycam_gwd_driver.F90").read_text()
    assert "allocatable, save, target, public :: ttgw(:,:)" in gwd
    assert "if (any(shape(ttgw) /= (/ state%ncol, pver /))) deallocate(ttgw)" in gwd
    assert "use gw_drag, only: band_oro" in gwd and "gw_spec_outflds" in gwd
    # the frame ABI carries five extents: the tracer ratio is rank four
    assert "shapes(5, count)" in runner and "shapes(5, count)" in (pausable.SUPPORT / "pycam_stage7_runner.F90").read_text()


def test_the_tphysac_classes_own_their_actions_and_kernels() -> None:
    from freecam.physics.pausable import STAGES, GravityWaveDrag, VerticalDiffusion

    assert STAGES["vertical_diffusion"] is VerticalDiffusion and STAGES["gravity_wave_drag"] is GravityWaveDrag
    assert VerticalDiffusion.SWAPPABLE == ("compute_tms", "compute_eddy_diff", "compute_vdiff", "virtem")
    assert GravityWaveDrag.SWAPPABLE == ("gw_drag_prof",) and GravityWaveDrag.STAGE == "cam_run2.gravity_wave_drag"
    rows = {r["kernel"]: r for r in VerticalDiffusion().describe_kernels()}
    assert set(rows) == set(VerticalDiffusion.SWAPPABLE) and all(r["bindable"] for r in rows.values())


def test_the_leaf_frames_and_the_flow_statements_the_runner_carries_out() -> None:
    frames = yaml.safe_load(pausable.FRAMES.read_text())["kernels"]
    wet = {s["name"]: s for s in frames["wetdepa_v2"]}
    chem = {s["name"]: s for s in frames["gas_phase_chemdr"]}
    assert len(frames["wetdepa_v2"]) == 32 and len(frames["modal_aero_depvel_part"]) == 13 and len(frames["gas_phase_chemdr"]) == 34
    # the cloud-borne site omits the interstitial optionals; the buffer is opaque to the frame
    assert wet["qqcw"]["kind"] == "absent" and wet["f_act_conv"]["kind"] == "absent"
    assert chem["pbuf"]["kind"] == "opaque" and "chem_name" not in chem
    awet = (pausable.SUPPORT / "pycam_awet_driver.F90").read_text()
    runner = (pausable.SUPPORT / "pycam_awet_runner.F90").read_text()
    # a species the driver skips: the piece reports the cycle, the runner moves its loop
    assert "flow = 1   ! cycle at the routine's level: the runner moves its loop" in awet
    assert "if (mm <= 0) then   ! cycle at the routine's level" in awet
    # the driver's early return when nothing is wet-deposited: the piece reports it, the runner leaves the routine
    assert "if (nwetdep<1) then   ! return at the routine's level: the runner leaves the routine" in awet
    assert "select case (driver_flow)" in runner and "driver_flow => flow" in runner
    # the else-if chain of the species loop is the runner's, with both pause sites inside it
    assert "else if ((lphase == 1) .and. (lspec == nspec_amode(m)+1)) then" in runner
    assert "case (pc_at_wetdepa_v2_1)" in runner and "case (pc_at_wetdepa_v2_2)" in runner
    assert "do lspec" not in runner and "lspec = lspec + (1)" in runner        # the loop is re-expressed, not copied
    adry = (pausable.SUPPORT / "pycam_adry_runner.F90").read_text()
    assert all(f"case (pc_at_modal_aero_depvel_part_{n})" in adry for n in (1, 2, 3, 4))
    chem_driver = (pausable.SUPPORT / "pycam_chem_driver.F90").read_text()
    assert "if ( .not. chem_step ) then   ! return at the routine's level" in chem_driver
    assert "use chemistry, only: chem_name" in chem_driver and "chem_freq" in chem_driver


def test_the_leaf_classes_own_their_actions_and_kernels() -> None:
    from freecam.physics.pausable import STAGES, AerosolDryDeposition, AerosolWetDeposition, ChemistryTendencies

    assert STAGES["aerosol_wet_deposition_leaf"] is AerosolWetDeposition
    assert STAGES["aerosol_dry_deposition_leaf"] is AerosolDryDeposition
    assert STAGES["chemistry_tendencies_leaf"] is ChemistryTendencies
    assert AerosolWetDeposition.SWAPPABLE == ("wetdepa_v2",) and AerosolDryDeposition.SWAPPABLE == ("modal_aero_depvel_part",)
    assert ChemistryTendencies.SWAPPABLE == ("gas_phase_chemdr",) and ChemistryTendencies.STAGE == "cam_run2.chemistry_tendencies_leaf"
    for cls in (AerosolWetDeposition, AerosolDryDeposition, ChemistryTendencies):
        rows = {r["kernel"]: r for r in cls().describe_kernels()}
        assert set(rows) == set(cls.SWAPPABLE) and all(r["bindable"] for r in rows.values())
    # every exposed numerical process now has a class: nine pausable, the energy fixer and eleven inert
    # here, radiation and the cloud stage in their own modules
    assert len(STAGES) == 21


def test_the_energy_fixer_is_owned_whole_and_says_why_it_has_no_pause() -> None:
    from freecam.physics.errors import PhysicsError
    from freecam.physics.pausable import STAGES, EnergyFixer

    assert STAGES["energy_fixer"] is EnergyFixer and EnergyFixer.SWAPPABLE == ()
    stage = EnergyFixer()
    assert stage.select_mode(None) == "native-whole" and stage.describe_kernels() == ()
    assert "check_energy_fix" in EnergyFixer.NO_PAUSE_BECAUSE
    stage.execution_policy = "segmented"
    with pytest.raises(PhysicsError):
        stage.select_mode(None)


def test_a_function_inside_an_assignment_pauses_with_its_sections_and_its_result() -> None:
    """virtem: an elemental function applied to sections, served as arrays; the left-hand side is the result slot."""

    spec = pausable.load_spec(pausable.SPECS / "vertical_diffusion.yaml")
    kernel = spec.kernels["virtem"]
    assert kernel.kind == "function" and kernel.elemental and kernel.result == "virtem"
    (pause,) = [p for u in spec.units.values() for p in u.pauses if p.kernel == "virtem"]
    assert pause.form == "assign" and pause.lhs == "thvs(:ncol)"
    slots = pausable.frame_slots(pause, kernel)
    assert [(s.dummy, s.rank, s.intent, s.shape) for s in slots] == [
        ("t", 1, "in", ["ncol"]), ("q", 1, "in", ["ncol"]), ("virtem", 1, "out", ["ncol"])]
    assert slots[0].expression == "th(1,pver)" and slots[1].expression == "state%q(1,pver,1)"
    assert slots[2].expression == "thvs(1)"
    driver = pausable.render_unit(spec, spec.units["driver"])
    assert "thvs(:ncol) = virtem(th(:ncol,pver),state%q(:ncol,pver,1))" in driver   # the original, verbatim
    # a partial range before another ranged axis is not contiguous and is refused
    with pytest.raises(SystemExit):
        pausable._section_extents("x(1:ncol,:)", None)
    assert pausable._section_extents("x(:ncol,k)", None) == (1, ["ncol"])
    assert pausable._section_extents("x(:,2:n,j)", None) == (2, ["size(x,1)", "(n)-(2)+1"])
    assert pausable._section_extents("x(i,k)", None) == (0, [])


def test_a_kernel_inside_a_compiled_kernel_is_reached_through_its_hook() -> None:
    """cldfrc_fice and fluxbelowinv: hooked, so the runner runs on the fiber when they alone are replaced."""

    import subprocess

    from freecam.pi_cam.hooks import load_hooks
    from freecam.pi_cam.segment_runner import load_manifest

    table = load_hooks()
    assert table.kernel_names == ("cldfrc_fice", "fluxbelowinv", "instratus_condensate")
    fice, flux = table.hook("cldfrc_fice"), table.hook("fluxbelowinv")
    assert fice.redirect == "rename-references" and fice.symbol == "pycam_hook_cldfrc_fice_"
    assert flux.redirect == "weaken-definition" and flux.symbol == "uwshcu_mp_fluxbelowinv_"
    assert flux.original_symbol == "uwshcu_mp_fluxbelowinv_original_"
    for stem, kernel, ids in (("deep_convection", "cldfrc_fice", "0, 0, 0, 1"), ("shallow_convection", "fluxbelowinv", "0, 2")):
        spec = pausable.load_spec(pausable.SPECS / f"{stem}.yaml")
        assert spec.kernels[kernel].hook and spec.kernels[kernel].within
        assert not any(p.kernel == kernel for u in spec.units.values() for p in u.pauses)
        runner = pausable.render_runner(spec)
        assert f"hook_of(nkernels) = (/ {ids} /)" in runner
        for needle in ("call run_from_start(event)", "pycam_hooks_frame_v1(count, ptrs, ndims, shapes, dtypes, intents, ncol_out)",
                       "status = pycam_hooks_original_v1()", "call continue_fiber(event)", "call abandon_fiber()"):
            assert needle in runner, needle
    # a runner without hooked kernels renders exactly as before: no fiber, no hooks
    plain = pausable.render_runner(pausable.load_spec(pausable.SPECS / "dry_adjustment.yaml"))
    assert "hook_of" not in plain and "fiber" not in plain and "pycam_hooks" not in plain
    # the hook module is what its generator writes, and the manifest ties the kernels to their contracts
    check = subprocess.run([sys.executable, str(pausable.REPO / "tools/generate_pi_cam_hooks.py"), "--check"],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr[-800:]
    manifest = {s.stage: s for s in load_manifest()}
    assert manifest["cam_run1.deep_convection"].kernel("cldfrc_fice").contract == "native/pi_cam/functions/cldfrc_fice.yaml"
    assert manifest["cam_run1.deep_convection"].replacement_conflicts({"zm_conv_evap": True, "cldfrc_fice": True}) == [("cldfrc_fice", "zm_conv_evap")]
    assert manifest["cam_run1.shallow_convection"].replacement_conflicts({"compute_uwshcu_inv": True, "fluxbelowinv": True}) == [("fluxbelowinv", "compute_uwshcu_inv")]


def test_a_runner_error_event_names_the_pauses_made_and_the_runner_message() -> None:
    from freecam.physics.errors import PhysicsError
    from freecam.physics.pausable import DryAdjustment
    from freecam.physics.segments import OriginalKernel, SegmentEvent

    class FailingRunner:
        kernels = ("dadadj",)
        runs_original = True

        def __init__(self):
            self.t = np.zeros((8, 4), order="F")
            self.destroyed = False

        def create(self, stage): return 1
        def start(self, cid, mask): return SegmentEvent.NEEDS_PYTHON_KERNEL
        def frame(self, cid):
            from freecam.physics.segments import FrameArgument, KernelFrame
            return KernelFrame(kernel="dadadj", call_index=1, lchnk=1, ncol=6, substep=1, token=7,
                               arguments=(FrameArgument("t", self.t, "inout"),))
        def resume(self, cid, kernel, token): return SegmentEvent.ERROR
        def run_original(self, cid, kernel): pass
        def error(self, cid): return "dadadj: the fiber ended with an error event and no message"
        def reset(self, cid): pass
        def destroy(self, cid): self.destroyed = True

    runner = FailingRunner()
    library = SimpleNamespace(pycam_stagehost_bind_v1=lambda: 0)
    native = SimpleNamespace(segment_runner=lambda stage: runner, library=library, run_action=lambda *a, **k: None)
    stage = DryAdjustment()
    stage.kernels["dadadj"] = OriginalKernel()
    with pytest.raises(PhysicsError, match=r"failed after 1 pause\(s\) this run: dadadj: the fiber ended"):
        stage.tend(None, SimpleNamespace(native=native))
    assert runner.destroyed                      # the context is gone and the stage is tainted


def test_every_c_bound_procedure_of_a_generated_support_module_carries_its_own_name() -> None:
    # a bare bind(C) takes the procedure's own name as the global symbol; two runners doing that
    # for their fiber bodies shared one symbol and ran each other's state machine (gate 7343594)
    bare = {}
    names = {}
    for path in sorted((REPO / "native/pi_cam/support").glob("pycam_*.F90")):
        in_interface = False
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("!", 1)[0].strip().lower()
            if code.startswith("interface") or code.startswith("abstract interface"):
                in_interface = True
            if code.startswith("end interface"):
                in_interface = False
            if in_interface or not code:
                continue                         # interfaces declare other objects' names
            if re.search(r"bind\(\s*c\s*\)", code):
                bare[f"{path.name}:{number}"] = line.strip()
            match = re.search(r"bind\(\s*c\s*,\s*name\s*=\s*'([^']+)'", code)
            if match:
                names.setdefault(match.group(1), []).append(path.name)
    assert bare == {}, bare
    shared = {name: owners for name, owners in names.items() if len(set(owners)) > 1}
    assert shared == {}, shared
    assert names["pycam_zmdeep_fiber_body_v1"] == ["pycam_zmdeep_runner.F90"]
    assert names["pycam_shcu_fiber_body_v1"] == ["pycam_shcu_runner.F90"]
