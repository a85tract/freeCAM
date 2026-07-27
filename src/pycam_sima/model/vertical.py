"""Vertical-coordinate input distributed by Python."""

from __future__ import annotations

import numpy as np
from netCDF4 import Dataset

from .errors import ConfigurationError


def load_vertical_coordinate(
    pool,
    ncdata: str,
    comm,
    *,
    expected_levels: int | None = None,
) -> None:
    values = None
    if comm.rank == 0:
        with Dataset(ncdata, "r") as dataset:
            values = {name: np.asarray(dataset.variables[name][...], dtype=np.float64) for name in ("hyai", "hybi", "hyam", "hybm")}
            values["P0"] = np.float64(dataset.variables["P0"].getValue())
    values = comm.bcast(values, root=0)
    nlev = (
        int(pool.dimensions["pver"])
        if expected_levels is None
        else int(expected_levels)
    )
    expected = {
        "hyai": (nlev + 1,),
        "hybi": (nlev + 1,),
        "hyam": (nlev,),
        "hybm": (nlev,),
    }
    mismatches = {
        name: (np.asarray(values[name]).shape, shape)
        for name, shape in expected.items()
        if np.asarray(values[name]).shape != shape
    }
    if mismatches:
        raise ConfigurationError(
            f"ncdata vertical coordinate does not match pver={nlev}: "
            f"{mismatches}"
        )
    pool.set("hybrid_a_interface", values["hyai"])
    pool.set("hybrid_b_interface", values["hybi"])
    pool.set("hybrid_a_midpoint", values["hyam"])
    pool.set("hybrid_b_midpoint", values["hybm"])
    pool.set("reference_pressure", values["P0"])
    pool.set("reference_interface_pressure", values["hyai"] * values["P0"] + values["hybi"] * values["P0"])
    pool.set("reference_midpoint_pressure", values["hyam"] * values["P0"] + values["hybm"] * values["P0"])
