import numpy as np
import pytest

from pycam_sima.pi_cam import (
    PICAMFieldContract,
    PICAMStateError,
    PICAMStatePool,
)


def test_pi_cam_state_is_rank_local_python_owned_fortran_storage() -> None:
    pool = PICAMStatePool({"column": 27, "level": 30})
    values = pool.create(
        PICAMFieldContract(
            "air_temperature",
            ("column", "level"),
            aliases=("T",),
        )
    )

    assert values.shape == (27, 30)
    assert values.flags.f_contiguous
    assert pool["T"].ctypes.data == values.ctypes.data


def test_replay_field_cannot_change_shape_between_steps() -> None:
    pool = PICAMStatePool({})
    pool.ensure_from_array("cam_in.sst", np.ones((2, 3)), category="boundary")

    with pytest.raises(PICAMStateError, match="changed shape"):
        pool.ensure_from_array(
            "cam_in.sst", np.ones((3, 2)), category="boundary"
        )
