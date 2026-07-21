"""Read and validate the fixed CAM namelist without invoking CAM code."""

from __future__ import annotations

from pathlib import Path

import f90nml

from .errors import ConfigurationError


def read_atm_in(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"atm_in does not exist: {path}")
    nml = f90nml.read(path)
    checks = {
        ("analytic_ic_nl", "analytic_ic_type"): "moist_baroclinic_wave_dcmip2016",
        ("cam_initfiles_nl", "pertlim"): 0.0,
        ("dyn_se_nl", "se_ne"): 3,
        ("dyn_se_nl", "se_fv_nphys"): 3,
        ("physics_nl", "physics_suite"): "kessler",
        ("vert_coord_nl", "pver"): 30,
    }
    errors = []
    for (section, key), expected in checks.items():
        actual = nml.get(section, {}).get(key)
        if actual != expected:
            errors.append(f"{section}.{key}={actual!r}, required {expected!r}")
    ncdata = nml.get("cam_initfiles_nl", {}).get("ncdata")
    if not ncdata or not Path(ncdata).is_file():
        errors.append(f"cam_initfiles_nl.ncdata is not readable: {ncdata!r}")
    if errors:
        raise ConfigurationError("unsupported atm_in: " + "; ".join(errors))
    return {"namelist": nml, "ncdata": str(ncdata)}
