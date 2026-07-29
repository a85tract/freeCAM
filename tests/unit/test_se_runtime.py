from __future__ import annotations

import numpy as np

from pycam_sima.model.se_runtime import apply_cam_forcing


def test_native_tracer_update_does_not_skip_python_owned_forcing() -> None:
    class Pool:
        dimensions = {
            "pver": 1,
            "nelem_local": 1,
            "qsize": 1,
            "ntrac": 1,
            "np": 1,
            "nc": 1,
            "nhc": 0,
        }
        fields = {
            "dynamics_forcing_type": np.asarray(2, dtype=np.int32),
            "dynamics_time_level_n0": np.asarray(0, dtype=np.int32),
            "dynamics_internal_step": np.asarray(0, dtype=np.int64),
            "dynamics_qsplit": np.asarray(1, dtype=np.int32),
            "vertical_remap_timestep": np.asarray(1.0),
            "model_timestep": np.asarray(2.0),
            "constituent_mass": np.zeros((1, 1, 1, 1, 1, 3), order="F"),
            "constituent_forcing": np.ones((1, 1, 1, 1, 1), order="F"),
            "fvm_tracer": np.zeros((1, 1, 1, 1, 1), order="F"),
            "fvm_layer_pressure_thickness": np.ones(
                (1, 1, 1, 1), order="F"
            ),
            "fvm_constituent_mass_forcing": np.full(
                (1, 1, 1, 1, 1), 0.25, order="F"
            ),
            "air_temperature": np.full(
                (1, 1, 1, 1, 3), 250.0, order="F"
            ),
            "zonal_wind": np.zeros((1, 1, 1, 1, 3), order="F"),
            "meridional_wind": np.zeros((1, 1, 1, 1, 3), order="F"),
            "temperature_forcing": np.full((1, 1, 1, 1), 0.5, order="F"),
            "zonal_wind_forcing": np.full((1, 1, 1, 1), 2.0, order="F"),
            "meridional_wind_forcing": np.full(
                (1, 1, 1, 1), -3.0, order="F"
            ),
            "layer_pressure_thickness": np.ones(
                (1, 1, 1, 1, 3), order="F"
            ),
            "forcing_full_layer_pressure_thickness": np.full(
                (1, 1, 1, 1), 2.0, order="F"
            ),
        }

        def get(self, name):
            return self.fields[name]

    class Backend:
        @staticmethod
        def apply_tracer_forcing(*, timestep, qdp, forcing):
            qdp[...] = qdp + timestep * forcing

    pool = Pool()
    apply_cam_forcing(pool, Backend())

    assert pool.fields["constituent_mass"][0, 0, 0, 0, 0, 0] == 1.0
    assert pool.fields["fvm_tracer"][0, 0, 0, 0, 0] == 0.5
    assert pool.fields["air_temperature"][0, 0, 0, 0, 0] == 250.5
    assert pool.fields["zonal_wind"][0, 0, 0, 0, 0] == 2.0
    assert pool.fields["meridional_wind"][0, 0, 0, 0, 0] == -3.0
