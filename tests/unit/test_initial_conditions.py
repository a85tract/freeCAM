from __future__ import annotations

import numpy as np

from freecam.model.initial_conditions import (
    populate_held_suarez_1994_initial_state,
    populate_resting_isothermal_initial_state,
)


class _Pool:
    constituent_names = (
        "cloud_liquid_water",
        "water_vapor",
        "cl",
        "cl2",
    )
    advected_constituent_indices = (0, 2, 3, 1)
    contracts: dict[str, object] = {}
    dimensions = {
        "ntime": 3,
        "pver": 2,
        "nconst": 4,
        "nphys_local": 1,
    }

    def __init__(self) -> None:
        state = (1, 1, 2, 1, 3)
        physics = (1, 2)
        self.fields = {
            "reference_pressure": np.asarray(100000.0),
            "hybrid_a_interface": np.asarray((0.0, 0.0, 0.0)),
            "hybrid_b_interface": np.asarray((0.0, 0.5, 1.0)),
            "hybrid_a_midpoint": np.asarray((0.0, 0.0)),
            "hybrid_b_midpoint": np.asarray((0.25, 0.75)),
            "constituent_minimum": np.asarray((1.0, 2.0, 3.0, 4.0))
            * 1.0e-12,
            "dry_air_gas_constant": np.asarray(287.0),
            "dry_air_specific_heat": np.asarray(1004.0),
            "zonal_wind": np.empty(state),
            "meridional_wind": np.empty(state),
            "air_temperature": np.empty(state),
            "surface_pressure": np.empty((1, 1, 1, 3)),
            "layer_pressure_thickness": np.empty(state),
            "constituent_mixing_ratio": np.empty((1, 1, 2, 1, 4, 3)),
            "constituent_mass": np.empty((1, 1, 2, 1, 10, 3)),
            "physics_zonal_wind": np.empty(physics),
            "physics_meridional_wind": np.empty(physics),
            "physics_air_temperature": np.empty(physics),
            "physics_surface_pressure": np.empty(1),
            "physics_layer_pressure_thickness": np.empty(physics),
            "physics_constituent_mixing_ratio": np.empty((1, 2, 4)),
            "fvm_tracer": np.empty((3, 3, 2, 1, 4)),
        }

    def get(self, name: str):
        return self.fields[name]

    def advected_slot(self, constituent: int) -> int:
        return self.advected_constituent_indices.index(constituent)


def test_resting_initial_state_keeps_qdp_and_fvm_registry_orders_distinct():
    pool = _Pool()
    populate_resting_isothermal_initial_state(pool)

    dp = np.asarray((50000.0, 50000.0))
    qdp = pool.get("constituent_mass")[0, 0, :, 0, :, 0]
    assert np.array_equal(qdp[:, 0], dp * 1.0e-12)
    assert np.array_equal(qdp[:, 1], dp * 2.0e-12)
    assert np.count_nonzero(qdp[:, 2:]) == 0

    fvm = pool.get("fvm_tracer")[0, 0, :, 0, :]
    assert np.array_equal(
        fvm[0],
        np.asarray((1.0, 3.0, 4.0, 2.0)) * 1.0e-12,
    )


def test_held_suarez_zeros_vapor_in_both_independent_orders():
    pool = _Pool()
    populate_held_suarez_1994_initial_state(
        pool,
        constituent_names=pool.constituent_names,
    )

    assert np.count_nonzero(
        pool.get("constituent_mass")[..., 1, :]
    ) == 0
    assert np.count_nonzero(pool.get("fvm_tracer")[..., 3]) == 0
    assert np.count_nonzero(
        pool.get("physics_constituent_mixing_ratio")[..., 1]
    ) == 0
