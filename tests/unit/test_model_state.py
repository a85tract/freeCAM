import numpy as np
import pytest

from pycam_sima.model.errors import StateOwnershipError
from pycam_sima.model.grid import (
    _global_dof_map,
    dimensions_for_rank,
    global_elements,
    local_elements,
)
from pycam_sima.model.state import StatePool


def test_ne3_sfc_partition_and_global_dofs_are_complete():
    elements = global_elements(24)
    assert [item.sfc for item in elements] == list(range(54))
    assert [len(local_elements(rank)) for rank in range(24)] == [3] * 6 + [2] * 18
    assert sorted(item.global_id for item in elements) == list(range(1, 55))
    arrays, owners = _global_dof_map()
    assert len(arrays) == 54
    assert len(owners) == 488
    assert all(value.dtype == np.int64 and value.flags.f_contiguous for value in arrays.values())


def test_constituent_aliases_are_zero_copy_and_static_fields_seal():
    pool = StatePool(dimensions_for_rank(0))
    assert np.shares_memory(
        pool.get("water_vapor"), pool.get("constituent_mixing_ratio")
    )
    pool.get("water_vapor")[0, 0, 0, 0, 0] = 4.0
    assert pool.get("constituent_mixing_ratio")[0, 0, 0, 0, 2, 0] == 4.0
    pool.seal_static()
    with pytest.raises(StateOwnershipError):
        pool.set("gll_node", 0.0)
    pool.set("gll_node", 0.0, unsafe=True)
