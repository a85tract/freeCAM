"""Infer suite dimensions from the selected case and pinned source data."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from netCDF4 import Dataset


def _namelist_value(
    namelist: Mapping[str, Any],
    bindings: Iterable[Any],
) -> Any | None:
    for binding in bindings:
        group = namelist.get(binding.group, {})
        if binding.local_name in group:
            return group[binding.local_name]
    return None


def _netcdf_extent(path: str | Path, dimension: str) -> int:
    resolved = Path(path).expanduser().resolve()
    with Dataset(resolved, "r") as dataset:
        try:
            return int(len(dataset.dimensions[dimension]))
        except KeyError as exc:
            raise ValueError(
                f"{resolved}: missing NetCDF dimension {dimension!r}"
            ) from exc


def _source_integer(
    path: str | Path,
    local_name: str,
) -> int:
    resolved = Path(path).resolve()
    text = resolved.read_text(encoding="utf-8")
    match = re.search(
        rf"\b{re.escape(local_name)}\s*=\s*(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"{resolved}: cannot find integer definition {local_name!r}"
        )
    return int(match.group(1))


def infer_suite_dimensions(
    *,
    required: Iterable[str],
    existing: Mapping[str, int],
    namelist: Mapping[str, Any],
    namelist_bindings: Mapping[str, Iterable[Any]],
    source_root: str | Path,
) -> dict[str, int]:
    """Return missing dimensions without embedding one case in ModelConfig.

    Scalar CCPP dimensions come directly from the generated ``atm_in``.
    File-sized arrays use the dimensions of the exact NetCDF files selected
    by that namelist.  The two temporary MUSICA extents and CAM ridge count
    are read from their pinned upstream host-provider source definitions.
    """

    wanted = {str(name).lower() for name in required}
    result: dict[str, int] = {}

    aliases = {
        "daytime_columns_dimension": "nphys_local",
        # RRTMGP may insert one layer above the CAM model top.  Reserve the
        # maximum extents up front; the active nlay/nlayp values remain
        # runtime StatePool scalars produced by rrtmgp_inputs_setup.
        "number_of_vertical_layers_in_rrtmgp": "pverp",
    }
    for name, target in aliases.items():
        if name in wanted and target in existing:
            result[name] = int(existing[target])
    if (
        "number_of_vertical_interfaces_in_rrtmgp" in wanted
        and "pverp" in existing
    ):
        result["number_of_vertical_interfaces_in_rrtmgp"] = (
            int(existing["pverp"]) + 1
        )

    for name in wanted:
        if name in existing or name in result:
            continue
        value = _namelist_value(
            namelist,
            namelist_bindings.get(name, ()),
        )
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            result[name] = len(value)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, int):
            result[name] = int(value)

    if {
        "number_of_wavelength_samples_of_spectrum",
        "number_of_wavelength_samples_of_spectrum_plus_one",
    } & wanted:
        solar = namelist.get("solar_data", {})
        path = solar.get("solar_irrad_data_file")
        if path:
            extent = _netcdf_extent(path, "wlen")
            result["number_of_wavelength_samples_of_spectrum"] = extent
            result[
                "number_of_wavelength_samples_of_spectrum_plus_one"
            ] = extent + 1

    coefficient_files = {
        "number_of_longwave_g_point_intervals": (
            "rrtmgp_lw_gas_optics",
            "rrtmgp_coefs_lw_file",
        ),
        "number_of_shortwave_g_point_intervals": (
            "rrtmgp_sw_gas_optics",
            "rrtmgp_coefs_sw_file",
        ),
    }
    for name, (group_name, key) in coefficient_files.items():
        if name not in wanted:
            continue
        path = namelist.get(group_name, {}).get(key)
        if path:
            result[name] = _netcdf_extent(path, "gpt")

    if (
        "number_of_time_slices_in_tropopause_climatology_dataset"
        in wanted
    ):
        path = namelist.get("tropopause_nl", {}).get(
            "tropopause_climo_file"
        )
        if path:
            result[
                "number_of_time_slices_in_tropopause_climatology_dataset"
            ] = _netcdf_extent(path, "time")

    source = Path(source_root)
    source_dimensions = {
        "number_of_ridges_in_ridge_gravity_wave_drag": (
            source
            / "src/physics/utils/gravity_wave_drag_ridge_read.F90",
            "prdg",
        ),
        "photolysis_wavelength_grid_section_dimension": (
            source / "src/physics/utils/musica_ccpp_dependencies.F90",
            "photolysis_wavelength_grid_section_dimension",
        ),
        "photolysis_wavelength_grid_interface_dimension": (
            source / "src/physics/utils/musica_ccpp_dependencies.F90",
            "photolysis_wavelength_grid_interface_dimension",
        ),
    }
    for name, (path, local_name) in source_dimensions.items():
        if name in wanted:
            result[name] = _source_integer(path, local_name)

    return result
