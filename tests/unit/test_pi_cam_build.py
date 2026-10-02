"""Builds for compile-time options: the default is the installed model, any other a chain of jobs.

The jobs themselves run on Derecho; here a fake submitter and a fake qstat
drive the build's state machine -- what it submits, after what, and what it
does when it is called again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import freecam as fc
from freecam.pi_cam import Driver
from freecam.pi_cam.build import Build, BuildError, BuildOptions
from freecam.pi_cam.config import PICAMConfig

STAGES = ["cases", "cases-build", "runs", "image", "coupler", "online-gate"]


class Queue:
    """A fake PBS: submissions get ids in order, and qstat answers what the test sets."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, str | None, dict]] = []
        self.states: dict[str, tuple[str, int | None]] = {}

    def submit(self, repo, job, environment, after):
        job_id = f"{len(self.submitted) + 1}.pbs"
        self.submitted.append((Path(job).name, after, dict(environment)))
        self.states[job_id] = ("queued", None)
        return job_id

    def inspect(self, job_id):
        return self.states.get(job_id, ("unknown", None))


def _build(tmp_path: Path, queue: Queue, **options) -> Build:
    return fc.build(scratch=tmp_path, submitter=queue.submit, inspector=queue.inspect, **options)


def _set_up(build: Build, case: str, *actions: str) -> None:
    status = build.case(case) / "CaseStatus"
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text("".join(f"2026-10-01 00:00:00: {action} starting\n2026-10-01 00:00:01: {action} success\n"
                              for action in actions))


def test_the_default_options_are_the_installed_model_and_build_nothing(tmp_path: Path) -> None:
    def refuse(*_):
        raise AssertionError("the installed model is never rebuilt")

    made = fc.build(scratch=tmp_path, submitter=refuse)

    assert made.installed and made.validated and made.stages() == []
    assert made.status()["build"] == "installed"
    assert made.config_for("configs/pi_cam_icesm131.yaml") == Path("configs/pi_cam_icesm131.yaml").resolve()


@pytest.mark.parametrize("options", [{"pcols": 0}, {"pcols": True}, {"device": "tpu"}])
def test_impossible_options_are_refused(options: dict) -> None:
    with pytest.raises(BuildError):
        BuildOptions(**options)


def test_the_fingerprint_follows_the_options() -> None:
    repo = Path(__file__).resolve().parents[2]

    assert BuildOptions(pcols=32).fingerprint(repo) == BuildOptions(pcols=32).fingerprint(repo)
    assert len({BuildOptions(pcols=32).fingerprint(repo), BuildOptions(pcols=24).fingerprint(repo),
                BuildOptions(pcols=32, device="cuda").fingerprint(repo)}) == 3
    assert BuildOptions(pcols=32, device="cuda").label == "pcols32-devicecuda"


def test_a_build_submits_every_stage_after_the_one_before(tmp_path: Path) -> None:
    queue = Queue()
    made = _build(tmp_path, queue, pcols=32)

    assert [job for job, _, _ in queue.submitted] == [
        "pi_cam_cases_prepare.pbs", "pi_cam_cases_build.pbs", "pi_cam_cases_run.pbs",
        "pi_cam_promoted_statepool_build.pbs", "pi_cam_online_coupler_build.pbs", "pi_cam_exact_cesm_online_50step.pbs"]
    assert [after for _, after, _ in queue.submitted] == [None, "1.pbs", "2.pbs", "3.pbs", "4.pbs", "5.pbs"]
    assert queue.submitted[0][2]["FREECAM_BUILD_PCOLS"] == "32"
    assert made.root.parent == tmp_path / "freeCAM/builds" and made.root.name.startswith("pcols32-")
    record = json.loads(made.record_path.read_text())
    assert list(record["stages"]) == STAGES and record["options"] == {"pcols": 32, "device": "cpu"}
    # everything a build writes stays in its root
    for _, _, environment in queue.submitted:
        for value in environment.values():
            assert not value.startswith("/") or value.startswith(str(made.root)) or "build/ftorch" in value


def test_the_gate_runs_on_a_configuration_with_the_builds_pcols_and_image(tmp_path: Path) -> None:
    made = _build(tmp_path, Queue(), pcols=32)
    gate = Path(made.stages()[-1].environment["PYCAM_CONFIG"])

    config = PICAMConfig.from_yaml(gate)

    assert config.pcols == 32 and config.native_manifest == made.native_manifest
    assert config.source_root.is_dir()


def test_calling_again_resumes(tmp_path: Path) -> None:
    queue = Queue()
    made = _build(tmp_path, queue, pcols=32)
    _build(tmp_path, queue, pcols=32)
    assert len(queue.submitted) == 6                       # every stage is still waiting: nothing new

    # the first two stages' products exist; the runs ended without theirs, and PBS dropped what waited on them
    for case in ("oracle", "state", "pycesm"):
        _set_up(made, case, "case.setup", "case.build")
    queue.states.update({"3.pbs": ("finished", 1), "4.pbs": ("unknown", None), "5.pbs": ("finished", None),
                         "6.pbs": ("finished", None)})
    assert {name: entry["state"] for name, entry in made.status()["stages"].items()} == {
        "cases": "done", "cases-build": "done", "runs": "failed", "image": "submitted", "coupler": "failed",
        "online-gate": "failed"}

    made.submit()

    # from the runs on, every stage again: what waited on the failed runs will not run
    resubmitted = queue.submitted[6:]
    assert [job for job, _, _ in resubmitted] == ["pi_cam_cases_run.pbs", "pi_cam_promoted_statepool_build.pbs",
                                                  "pi_cam_online_coupler_build.pbs", "pi_cam_exact_cesm_online_50step.pbs"]
    assert [after for _, after, _ in resubmitted] == [None, "7.pbs", "8.pbs", "9.pbs"]


def test_a_build_is_validated_by_its_own_online_gate(tmp_path: Path) -> None:
    queue = Queue()
    made = _build(tmp_path, queue, pcols=32)
    assert not made.validated

    made.gate_record.parent.mkdir(parents=True)
    made.gate_record.write_text(json.dumps({"bfb": False}))
    assert made.status()["stages"]["online-gate"]["state"] == "failed"
    with pytest.raises(BuildError, match="online-gate"):
        made.wait(poll_seconds=0)

    made.gate_record.write_text(json.dumps({"bfb": True}))
    assert made.validated


def test_a_driver_runs_an_unvalidated_build_only_as_an_exploration(tmp_path: Path) -> None:
    made = _build(tmp_path, Queue(), pcols=32)

    with pytest.raises(ValueError, match="online gate"):
        Driver(case="PI-atm", build=made, scratch=tmp_path)
    driver = Driver(case="PI-atm", build=made, scratch=tmp_path, exploratory=True)

    assert driver.config.pcols == 32 and driver.config.native_manifest == made.native_manifest
    assert driver.reference_case == made.case("oracle") and driver.reference_run == made.run_dir("oracle")
    assert driver._resolve_online_library() == made.online_library
    assert driver.diagnose()["build"] == made.root.name


def test_a_driver_refuses_an_image_named_twice(tmp_path: Path, monkeypatch) -> None:
    made = _build(tmp_path, Queue(), pcols=32)
    monkeypatch.setenv("FREECAM_NATIVE_MANIFEST", str(tmp_path / "another/native_cam_manifest.json"))

    with pytest.raises(ValueError, match="FREECAM_NATIVE_MANIFEST"):
        Driver(case="PI-atm", build=made, scratch=tmp_path, exploratory=True)


def test_a_replay_case_keeps_the_installed_build(tmp_path: Path) -> None:
    made = _build(tmp_path, Queue(), pcols=32)

    with pytest.raises(ValueError, match="installed build"):
        Driver(case="PI-atm-replay", build=made, scratch=tmp_path, exploratory=True)


def _products(made: Build, *, linked_pcols: int | None, image_pcols: int = 32) -> Path:
    made.native_manifest.parent.mkdir(parents=True, exist_ok=True)
    made.native_manifest.write_text(json.dumps({"dimensions": {"pcols": image_pcols, "pver": 30, "pcnst": 57}}))
    made.online_library.parent.mkdir(parents=True, exist_ok=True)
    made.online_library.write_bytes(b"library")
    if linked_pcols is not None:
        (made.root / "provider/external_atm_build.json").write_text(
            json.dumps({"cam_dimensions": {"pcols": linked_pcols, "pver": 30, "pcnst": 57}}))
    return made.online_library.resolve()


def test_the_coupler_library_runs_only_with_an_image_of_its_grid(tmp_path: Path) -> None:
    made = _build(tmp_path, Queue(), pcols=32)
    driver = Driver(case="PI-atm", build=made, scratch=tmp_path, exploratory=True)

    driver._check_pairing(_products(made, linked_pcols=32))
    with pytest.raises(ValueError, match="pcols=16; this run's image has pcols=32"):
        driver._check_pairing(_products(made, linked_pcols=16))


def test_a_builds_coupler_library_needs_its_record(tmp_path: Path) -> None:
    made = _build(tmp_path, Queue(), pcols=32)
    driver = Driver(case="PI-atm", build=made, scratch=tmp_path, exploratory=True)

    with pytest.raises(ValueError, match="no build record"):
        driver._check_pairing(_products(made, linked_pcols=None))
