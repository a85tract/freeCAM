import numpy as np
import pytest

from pycam_sima.state_pool import FieldSpec, StatePool


def test_python_owned_fortran_array_and_readonly_callback():
    pool = StatePool()
    array = pool.allocate(
        FieldSpec("air_temperature", np.float64, ("ncol", "pver")), (2, 3)
    )
    assert array.flags.f_contiguous
    pointer = pool.pointer("air_temperature")

    with pool.callback_access(writable=False):
        with pytest.raises(ValueError):
            array[0, 0] = 1.0

    array[0, 0] = 2.0
    assert pool.pointer("air_temperature") == pointer
    assert array[0, 0] == 2.0
