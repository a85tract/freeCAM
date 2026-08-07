import hashlib

import numpy as np
import pytest

from freecam.model.contracts import (
    model_alias_rules,
    model_ccpp_field_aliases,
)
from freecam.model.errors import StateOwnershipError
from freecam.model.grid import (
    _derivative_matrix,
    _gll_nodes_weights,
    _global_dof_map,
    dimensions_for_rank,
    global_elements,
    homme_space_curve,
    local_elements,
    populate_grid,
    sfc_partition_counts,
    sfc_partition_owner,
)
from freecam.model.state import StatePool


def test_ne3_sfc_partition_and_global_dofs_are_complete():
    elements = global_elements(24)
    assert [item.sfc for item in elements] == list(range(54))
    assert [len(local_elements(rank)) for rank in range(24)] == [3] * 6 + [2] * 18
    assert sorted(item.global_id for item in elements) == list(range(1, 55))
    arrays, owners = _global_dof_map()
    assert len(arrays) == 54
    assert len(owners) == 488
    assert all(value.dtype == np.int64 and value.flags.f_contiguous for value in arrays.values())


@pytest.mark.parametrize(
    ("ne", "fortran_sha256"),
    (
        (2, "8641d3a6c53b6fbc3f6575d6703eb583fd3ca0d22027f654d759005f561142ac"),
        (3, "4e8f51aa30d7e210b500de673c6fc498ea51310edf1e4e3dd1c5dee7e72fa5ef"),
        (5, "c2839d28a1778cb5b90e838ce2f3075359c90ca323eff784135625f95754f0ee"),
        (6, "3274e70e51910976765af73c82e82880c3a08fc814dcb89fa8e2bdc8d42cef10"),
        (7, "1d4c3d77f291b6e26830b72b856aa6c5f588794bc1355a1f3f2ef23a81f7f3d6"),
        (10, "15cfc6c0559c8598b5f9e158ba5a1df2cb4eec8f28689f8aec7e88e3002a2978"),
        (11, "54718593594003a7b3ae0b9864caa1e5ecd537a7fb365fe91a79e8e983773232"),
        (30, "c6fe02c575285ebeb7f75a66d2475feee9b74d9d2900e726b52b29c1dbaf173a"),
        (31, "5f30f35de859e4c0b0c0ae139ab0771043b0ad182feae985238ba6d838849863"),
    ),
)
def test_python_sfc_matches_homme_fortran(ne, fortran_sha256):
    mesh = np.asarray(homme_space_curve(ne), dtype="<i8")
    assert sorted(mesh.flat) == list(range(ne * ne))
    assert (
        hashlib.sha256(mesh.tobytes(order="F")).hexdigest()
        == fortran_sha256
    )


def test_generic_cubed_sphere_sfc_and_partition_are_complete():
    ne = 7
    size = 13
    elements = global_elements(size, ne)
    assert [item.sfc for item in elements] == list(range(6 * ne * ne))
    assert sorted(item.global_id for item in elements) == list(
        range(1, 6 * ne * ne + 1)
    )
    assert [
        len(local_elements(rank, size, ne)) for rank in range(size)
    ] == list(sfc_partition_counts(6 * ne * ne, size))


def test_homme_space_curve_rejects_nonpositive_side():
    assert homme_space_curve(1) == ((0,),)
    with pytest.raises(ValueError, match="side must be positive"):
        homme_space_curve(0)


@pytest.mark.parametrize(
    ("size", "counts"),
    (
        (1, [54]),
        (7, [8, 8, 8, 8, 8, 7, 7]),
        (12, [5] * 6 + [4] * 6),
        (18, [3] * 18),
        (24, [3] * 6 + [2] * 18),
        (27, [2] * 27),
        (53, [2] + [1] * 52),
        (54, [1] * 54),
    ),
)
def test_sfc_partition_matches_homme_remainder_first_distribution(
    size,
    counts,
):
    assert list(sfc_partition_counts(54, size)) == counts
    elements = global_elements(size)
    assert [len(local_elements(rank, size)) for rank in range(size)] == counts
    assert [item.owner for item in elements] == [
        sfc_partition_owner(index, 54, size) for index in range(54)
    ]
    for rank in range(size):
        indices = [item.sfc for item in elements if item.owner == rank]
        assert indices == list(range(indices[0], indices[-1] + 1))


def test_sfc_partition_rejects_empty_partitions_and_invalid_indices():
    with pytest.raises(ValueError, match="cannot exceed element_count"):
        sfc_partition_counts(54, 55)
    with pytest.raises(ValueError, match="partition_count must be positive"):
        sfc_partition_counts(54, 0)
    with pytest.raises(ValueError, match="sfc_index"):
        sfc_partition_owner(54, 54, 24)
    with pytest.raises(ValueError, match="rank must be between"):
        local_elements(18, 18)


@pytest.mark.parametrize("size", (1, 7, 12, 18, 24, 27, 53, 54))
def test_ne3_dimensions_and_halo_inventory_follow_runtime_mpi_size(size):
    dimensions = [dimensions_for_rank(rank, size) for rank in range(size)]
    assert sum(item["nelem_local"] for item in dimensions) == 54
    assert sum(item["nphys_local"] for item in dimensions) == 54 * 9
    arrays, owners = _global_dof_map(size)
    assert len(arrays) == 54
    assert all(
        0 <= owner < size
        for dof_owners in owners.values()
        for owner in dof_owners
    )


@pytest.mark.parametrize(("size", "rank"), ((18, 0), (27, 26), (54, 53)))
def test_grid_generation_uses_the_runtime_sfc_partition(size, rank):
    pool = StatePool(dimensions_for_rank(rank, size))
    populate_grid(pool, rank, size)
    expected = [item.global_id for item in local_elements(rank, size)]
    assert pool.get("global_element_id").tolist() == expected
    peers = pool.get("halo_peer_rank")
    assert np.all((peers == -1) | ((0 <= peers) & (peers < size)))


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


def test_nonreference_grid_and_constituent_dimensions_are_allocated() -> None:
    dimensions = dimensions_for_rank(
        0,
        8,
        ne=2,
        np_value=5,
        fv_nphys=4,
        pver=12,
        constituent_count=5,
    )
    pool = StatePool(dimensions)
    populate_grid(pool, 0, 8, ne=2)

    assert pool.get("gll_cartesian").shape == (3, 5, 5, 3)
    assert pool.get("physics_latitude").shape == (3 * 4 * 4,)
    assert pool.get("constituent_mixing_ratio").shape == (
        5,
        5,
        12,
        3,
        5,
        3,
    )
    assert np.isfinite(pool.get("metric_jacobian")).all()


@pytest.mark.parametrize("node_count", (3, 5, 6))
def test_generic_gll_derivative_uses_homme_input_output_layout(
    node_count: int,
) -> None:
    nodes, _weights = _gll_nodes_weights(node_count)
    derivative = _derivative_matrix(nodes)

    assert np.allclose(
        np.sum(derivative, axis=0),
        0.0,
        rtol=0.0,
        atol=3.0e-14,
    )
    assert np.allclose(
        nodes @ derivative,
        1.0,
        rtol=0.0,
        atol=3.0e-14,
    )


def test_reduced_constituent_pool_omits_unavailable_water_aliases() -> None:
    pool = StatePool(
        dimensions_for_rank(
            0,
            1,
            ne=1,
            pver=4,
            constituent_count=1,
        )
    )
    assert pool.get("cloud_liquid_water").shape[-1] == 3
    with pytest.raises(KeyError, match="water_vapor"):
        pool.get("water_vapor")


def test_constituent_standard_names_follow_configured_species_order():
    dimensions = dimensions_for_rank(
        0,
        constituent_count=1,
    )
    pool = StatePool(
        dimensions,
        alias_rules=model_alias_rules(("water_vapor",)),
        ccpp_aliases=model_ccpp_field_aliases(("water_vapor",)),
    )

    water_name = pool.ccpp_field_name(
        "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water"
    )
    water = pool.get(water_name)
    constituents = pool.get("physics_constituent_mixing_ratio")

    assert water.shape == constituents[..., 0].shape
    assert np.shares_memory(water, constituents)
    water[...] = 0.25
    assert np.array_equal(constituents[..., 0], water)
    with pytest.raises(KeyError):
        pool.get("physics_cloud_liquid_water")


def test_cloud_ice_standard_name_is_a_zero_copy_constituent_view():
    dimensions = dimensions_for_rank(0, constituent_count=2)
    pool = StatePool(
        dimensions,
        alias_rules=model_alias_rules(("cloud_ice", "water_vapor")),
        ccpp_aliases=model_ccpp_field_aliases(
            ("cloud_ice", "water_vapor")
        ),
        constituent_names=("cloud_ice", "water_vapor"),
    )

    ice = pool.get(
        pool.ccpp_field_name(
            "cloud_ice_mixing_ratio_wrt_moist_air_and_condensed_water"
        )
    )
    constituents = pool.get("physics_constituent_mixing_ratio")

    assert np.shares_memory(ice, constituents[..., 0])


def test_state_pool_tracks_distinct_physics_and_advected_orders():
    dimensions = dimensions_for_rank(
        0,
        constituent_count=7,
        advected_constituent_count=4,
        thermodynamic_constituent_count=2,
    )
    pool = StatePool(
        dimensions,
        constituent_names=(
            "cloud_liquid_water",
            "water_vapor",
            "cl",
            "cl2",
            "o3",
            "air",
            "o2",
        ),
        advected_constituent_indices=(0, 2, 3, 1),
    )

    assert pool.advected_constituent_names == (
        "cloud_liquid_water",
        "cl",
        "cl2",
        "water_vapor",
    )
    assert pool.advected_slot(1) == 3


def test_tracer_limiter_bounds_use_active_qsize_stride():
    dimensions = dimensions_for_rank(
        0,
        1,
        constituent_count=7,
        advected_constituent_count=4,
        thermodynamic_constituent_count=2,
    )
    pool = StatePool(dimensions)

    expected = (
        dimensions["pver"],
        dimensions["qsize"],
        dimensions["nelem_local"],
    )
    assert pool.get("tracer_stage_minimum").shape == expected
    assert pool.get("tracer_stage_maximum").shape == expected
