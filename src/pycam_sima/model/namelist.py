"""Read a CAM namelist and verify it against the selected case configuration."""

from __future__ import annotations

from pathlib import Path

import f90nml

from .errors import ConfigurationError


def read_atm_in(path: str | Path, config) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"atm_in does not exist: {path}")
    nml = f90nml.read(path)
    checks = {
        ("analytic_ic_nl", "analytic_ic_type"): config.analytic_ic_type,
        ("cam_initfiles_nl", "pertlim"): config.pertlim,
        ("dyn_se_nl", "se_ne"): config.ne,
        ("dyn_se_nl", "se_fv_nphys"): config.fv_nphys,
        ("physics_nl", "physics_suite"): config.physics_suite,
        ("vert_coord_nl", "pver"): config.pver,
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
        raise ConfigurationError(
            "atm_in differs from ModelConfig: " + "; ".join(errors)
        )
    return {"namelist": nml, "ncdata": str(ncdata)}
