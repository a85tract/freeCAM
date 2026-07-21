"""Vertical-coordinate input distributed by Python."""

from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from .errors import ConfigurationError


def load_vertical_coordinate(pool, ncdata: str, comm) -> None:
    values = None
    if comm.rank == 0:
        with Dataset(ncdata, "r") as dataset:
            values = {name: np.asarray(dataset.variables[name][...], dtype=np.float64) for name in ("hyai", "hybi", "hyam", "hybm")}
            values["P0"] = np.float64(dataset.variables["P0"].getValue())
    values = comm.bcast(values, root=0)
    if values["hyai"].shape != (31,) or values["hyam"].shape != (30,):
        raise ConfigurationError("ncdata vertical coordinate is not L30")
    pool.set("hybrid_a_interface", values["hyai"])
    pool.set("hybrid_b_interface", values["hybi"])
    pool.set("hybrid_a_midpoint", values["hyam"])
    pool.set("hybrid_b_midpoint", values["hybm"])
    pool.set("reference_pressure", values["P0"])
    pool.set("reference_interface_pressure", values["hyai"] * values["P0"] + values["hybi"] * values["P0"])
    pool.set("reference_midpoint_pressure", values["hyam"] * values["P0"] + values["hybm"] * values["P0"])
