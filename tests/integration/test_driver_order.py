from pycam_sima.config import CaseConfig
from pycam_sima.driver import FKesslerDriver
from pycam_sima.mpi_runtime import SerialComm
from pycam_sima.native import RecordingBackend
from pycam_sima.suites.kessler import AFTER_SCHEMES, BEFORE_SCHEMES


def test_data_initialize_and_model_advance_order():
    config = CaseConfig.from_yaml("configs/fkessler_ne3pg3.yaml")
    backend = RecordingBackend()
    driver = FKesslerDriver(config, SerialComm(), backend=backend)
    seen = []
    driver.observe("before:*", lambda ctx: seen.append((ctx.step, f"before:{ctx.task_name}")))
    driver.observe("after:*", lambda ctx: seen.append((ctx.step, f"after:{ctx.task_name}")))

    driver.initialize()
    assert backend.calls[:2] == ["lifecycle:register", "lifecycle:initialize"]
    assert backend.calls[2] == "lifecycle:timestep_initial"
    assert tuple(backend.calls[3 : 3 + len(BEFORE_SCHEMES)]) == BEFORE_SCHEMES

    driver.run(1)
    start = 3 + len(BEFORE_SCHEMES)
    assert tuple(backend.calls[start : start + len(AFTER_SCHEMES)]) == AFTER_SCHEMES
    assert backend.calls[start + len(AFTER_SCHEMES)] == "lifecycle:timestep_final"
    assert backend.calls[start + len(AFTER_SCHEMES) + 1] == "lifecycle:timestep_initial"
    assert tuple(backend.calls[-len(BEFORE_SCHEMES) :]) == BEFORE_SCHEMES
    assert driver.clock.step == 1
    assert (0, "before:kessler") in seen
    assert (0, "after:kessler") in seen
