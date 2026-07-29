from __future__ import annotations

import numpy as np

from pycam_sima.model.backend import FVMKernelConfig
from pycam_sima.model.fvm_mapping import (
    synchronize_min_owned_gll,
    tensor_lagrange_interp,
)


def test_pg3_to_gll_simple_limiter_matches_cam_surrounding_cell_bounds() -> None:
    logical_index = np.arange(-2, 7, dtype=np.float64)
    pg3_coordinate = logical_index * np.float64(2.0 / 3.0) - np.float64(
        4.0 / 3.0
    )
    coordinates = np.empty((2, 9, 9), dtype=np.float64, order="F")
    coordinates[0, ...] = pg3_coordinate[:, None]
    coordinates[1, ...] = pg3_coordinate[None, :]

    field = np.zeros((9, 9, 1, 1), dtype=np.float64, order="F")
    field[4, :, 0, 0] = 100.0

    unlimited = tensor_lagrange_interp(
        0, field, coordinates, limiter=False
    )
    limited = tensor_lagrange_interp(0, field, coordinates, limiter=True)

    # Cubic interpolation undershoots at the outer GLL nodes, while CAM's
    # llimiter=.true. topography path clips to the four surrounding PG3
    # values in source order.
    assert np.all(unlimited[(0, 3), :, 0, 0] < 0.0)
    assert np.array_equal(
        limited[(0, 3), :, 0, 0],
        np.zeros((2, 4), dtype=np.float64),
    )
    assert np.array_equal(
        limited[1:3, :, 0, 0],
        unlimited[1:3, :, 0, 0],
    )


def test_min_owned_gll_synchronization_copies_only_unique_occurrence() -> None:
    class Pool:
        dimensions = {"np": 2, "nelem_local": 2}

        fields = {
            "spectral_element_count": np.array(2),
            "global_element_id": np.array((1, 2), dtype=np.int32),
            "gll_global_dof": np.empty((2, 2, 2), dtype=np.int64, order="F"),
        }
        fields["gll_global_dof"][:, :, 0] = ((1, 3), (2, 4))
        fields["gll_global_dof"][:, :, 1] = ((2, 7), (6, 8))

        def get(self, name):
            return self.fields[name]

    class LocalComm:
        @staticmethod
        def Allreduce(send, receive):
            receive[...] = send

    field = np.empty((2, 2, 2), dtype=np.float64, order="F")
    field[:, :, 0] = ((10.0, 20.0), (30.0, 40.0))
    field[:, :, 1] = ((99.0, 60.0), (70.0, 80.0))
    synchronized = synchronize_min_owned_gll(Pool(), LocalComm(), field)

    # Element 2's first occurrence shares DOF 2 with element 1's second
    # occurrence.  Only the MIN-owned element-1 value may survive.
    assert synchronized[0, 0, 1] == 30.0
    assert synchronized[1, 1, 1] == 80.0


def test_fvm_kernel_uses_advected_tracer_count_not_storage_count() -> None:
    class Pool:
        dimensions = {
            "fv_nphys": 3,
            "pver": 26,
            "ntrac": 7,
            "nconst": 10,
            "fvm_reconstruction": 3,
            "np": 4,
            "fvm_internal": 9,
            "fvm_interp_span": 9,
            "fvm_stretch": 8,
            "fvm_halo": 9,
        }

    config = FVMKernelConfig.from_pool(Pool())

    assert config.ntrac == 7
    assert config.abi().ntrac == 7
