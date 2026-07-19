from pathlib import Path
from types import SimpleNamespace

import pytest

from pycam_sima.full_driver import FULL_CAM_PHASES, FullCAMDriver


class FakeBackend:
    def __init__(self, library: str | Path) -> None:
        self.library = Path(library)
        self.calls: list[str] = []
        self.nstep = 0

    def initialize(self, comm) -> None:
        self.calls.append("initialize")

    def timestep_init(self) -> None:
        self.calls.append("cam_timestep_init")

    def attach_state(self, pool) -> None:
        self.calls.append("attach_state")

    def run1(self) -> None:
        self.calls.append("cam_run1")

    def run2(self) -> None:
        self.calls.append("cam_run2")

    def run3(self) -> None:
        self.calls.append("cam_run3")

    def run4(self) -> None:
        self.calls.append("cam_run4")

    def timestep_final(self) -> None:
        self.calls.append("cam_timestep_final")

    def advance_timestep(self) -> None:
        self.calls.append("advance_timestep")
        self.nstep += 1

    def finalize(self) -> None:
        self.calls.append("finalize")


@pytest.fixture
def driver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FullCAMDriver:
    monkeypatch.setattr("pycam_sima.full_driver.FullNativeBackend", FakeBackend)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "atm_in").write_text("&cam_initfiles_nl /\n")
    config = SimpleNamespace(dt_seconds=1800, steps=1, mode="interactive")
    comm = SimpleNamespace(rank=0, size=24)
    result = FullCAMDriver(config, comm, library=tmp_path / "fake.so", run_dir=run_dir)
    result.initialize()
    yield result
    if result._initialized:
        result.finalize()


def test_each_phase_can_pause_and_preserves_default_order(driver: FullCAMDriver) -> None:
    assert driver.phase_names == FULL_CAM_PHASES
    assert driver.next_phase == "cam_run2"
    assert driver.phase_status.cycle_kind == "initial_send"

    for index, phase in enumerate(FULL_CAM_PHASES):
        status = driver.run_phase(phase)
        assert status.last_phase == phase
        assert status.next_phase == FULL_CAM_PHASES[(index + 1) % len(FULL_CAM_PHASES)]

    assert driver.clock.step == 0
    assert driver.backend.nstep == 1
    assert driver.phase_status.cycle_complete

    for phase in FULL_CAM_PHASES:
        driver.run_phase(phase)

    assert driver.clock.step == 1
    assert driver.backend.nstep == 2
    assert driver.next_phase == "cam_run2"


def test_step_uses_the_same_phase_state_machine(driver: FullCAMDriver) -> None:
    driver.run(1)
    assert driver.clock.step == 1
    assert driver.backend.nstep == 2
    cycle = [
        "cam_run2",
        "cam_run3",
        "cam_run4",
        "cam_timestep_final",
        "advance_timestep",
        "cam_timestep_init",
        "attach_state",
        "cam_run1",
    ]
    assert driver.backend.calls[4:] == cycle + cycle


def test_invalid_order_is_rejected_unless_explicitly_unsafe(driver: FullCAMDriver) -> None:
    with pytest.raises(RuntimeError, match="expected cam_run2, got cam_run3"):
        driver.run_phase("cam_run3")

    status = driver.run_phase("cam_run3", allow_unsafe_order=True)
    assert not status.sequence_safe
    assert status.next_phase is None
    assert status.cycle_kind == "unsafe"

    with pytest.raises(RuntimeError, match="unsafe phase ordering"):
        driver.run(1)
    with pytest.raises(RuntimeError, match="already in unsafe-order mode"):
        driver.run_phase("cam_run2")

    driver.run_phase("cam_run2", allow_unsafe_order=True)


def test_unknown_phase_fails_before_native_call(driver: FullCAMDriver) -> None:
    calls = list(driver.backend.calls)
    with pytest.raises(ValueError, match="unknown CAM phase"):
        driver.run_phase("not_a_phase")
    assert driver.backend.calls == calls
