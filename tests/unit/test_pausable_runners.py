"""The pausable runners: generated from their specs, complete over the pinned text, decodable by Python."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

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
        assert "c_funptr" not in text.lower() and "c_f_procpointer" not in text.lower()


def test_the_manifest_names_the_pausable_runners_and_python_decodes_their_frames() -> None:
    from freecam.pi_cam import segment_runner as runners

    spec = runners.runner_spec("cam_run1.dry_adjustment")
    assert spec is not None and spec.original and spec.kernel_names == ("dadadj",)
    assert spec.kernel("dadadj").frame == "native/pi_cam/segment_frames.yaml"
    assert runners.frame_names_from_descriptor(REPO / spec.kernel("dadadj").frame, "dadadj") == (
        "lchnk", "ncol", "pmid", "pint", "pdel", "t", "q")
    shallow = runners.runner_spec("cam_run1.shallow_convection")
    assert shallow is not None and shallow.kernel_names == ("compute_uwshcu_inv",)
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
    assert ShallowConvection().SWAPPABLE == ("compute_uwshcu_inv",)
    inert = [cls for cls in STAGES.values() if issubclass(cls, InertStage)]
    assert len(inert) == 11 and all(cls.SWAPPABLE == () for cls in inert)
    assert {cls.STAGE for cls in inert} == {
        "cam_run2.rayleigh_friction", "cam_run2.charge_neutrality", "cam_run2.qbo_relaxation", "cam_run2.ion_drag",
        "cam_run1.sea_salt_rebin", "cam_run1.modal_aerosol_preparation_leaf", "cam_run1.carma_wet_deposition_leaf",
        "cam_run2.carma_aerosol_tendencies_leaf", "cam_run2.carma_statistics_leaf", "cam_run2.tracer_tendencies_leaf",
        "cam_run2.age_of_air_tendencies_leaf"}
    rows = {r["kernel"]: r for r in stage.describe_kernels()}
    assert rows["dadadj"]["bindable"] and not rows["dadadj"]["validated"]
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
