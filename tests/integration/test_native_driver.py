from pathlib import Path

import numpy as np
import pytest

from pycam_sima.config import CaseConfig
from pycam_sima.driver import FKesslerDriver
from pycam_sima.mpi_runtime import SerialComm
from pycam_sima.native import NativeKesslerBackend
from pycam_sima.suites.kessler import AFTER_SCHEMES, BEFORE_SCHEMES


LIBRARY = Path("build/native/libpycam_sima_kessler.so")


@pytest.mark.skipif(not LIBRARY.is_file(), reason="native library has not been built")
def test_all_suite_calls_run_on_python_owned_state():
    config = CaseConfig.from_yaml("configs/fkessler_ne3pg3.yaml")
    backend = NativeKesslerBackend(LIBRARY)
    driver = FKesslerDriver(config, SerialComm(), backend=backend)
    driver.allocate_minimal_state(ncol=2)
    pointers = {name: driver.pool.pointer(name) for name in driver.pool}

    for scheme in BEFORE_SCHEMES + AFTER_SCHEMES:
        assert backend.lib.pycam_kessler_has_scheme(scheme.encode()) == 1

    driver.initialize()
    driver.run(2)
    driver.finalize()

    assert pointers == {name: driver.pool.pointer(name) for name in pointers}
    assert np.isfinite(driver.pool["air_temperature"]).all()
    assert np.isfinite(driver.pool["ccpp_constituents"]).all()
