from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from freecam.model.fvm_geometry import generate_fvm_geometry


ROOT = Path(__file__).resolve().parents[2]


def test_fvm_geometry_is_generated_without_a_packaged_grid_file() -> None:
    assert not (
        ROOT
        / "src/freecam/model/data/fkessler_ne3pg3_l30_fvm_grid.nc"
    ).exists()

    interfaces = np.linspace(0.0, 1.0, 31, dtype=np.float64)
    geometry = generate_fvm_geometry(
        interfaces, np.zeros_like(interfaces), 100000.0
    )

    assert geometry["global_element_id"].shape == (54,)
    assert geometry["vertex_cartesian"].shape == (4, 2, 9, 9, 54)
    assert geometry["sphere_centroid"].shape == (5, 9, 9, 54)
    assert geometry["reconstruction_metric"].shape == (3, 5, 5, 54)
    assert np.array_equal(
        geometry["dp_ref"],
        np.broadcast_to(np.diff(interfaces)[:, None] * 100000.0, (30, 54)),
    )

    digest = hashlib.sha256()
    for name in sorted(geometry):
        if name in {"dp_ref", "dp_ref_inverse"}:
            continue
        value = np.ascontiguousarray(geometry[name])
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    assert digest.hexdigest() == (
        "891a7677ba27023e88fc2fc419cb4c37c5206c4eca33a2ed260955daaab84c41"
    )


def test_fvm_geometry_follows_nonreference_ne_and_cell_count() -> None:
    interfaces = np.linspace(0.01, 1.0, 13, dtype=np.float64)
    geometry = generate_fvm_geometry(
        np.zeros_like(interfaces),
        interfaces,
        100000.0,
        ne=2,
        nc=4,
    )

    assert geometry["global_element_id"].shape == (24,)
    assert geometry["vertex_cartesian"].shape == (4, 2, 10, 10, 24)
    assert geometry["reconstruction_metric"].shape == (3, 6, 6, 24)
    assert geometry["halo_interpolation_weight"].shape == (4, 8, 2, 2, 24)
    assert all(np.isfinite(value).all() for value in geometry.values())
