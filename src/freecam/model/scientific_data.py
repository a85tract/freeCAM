"""Python-owned readers for CAM scientific datasets used before CCPP run."""

from __future__ import annotations

import json
from pathlib import Path
import re

from netCDF4 import Dataset
import numpy as np

from .errors import ConfigurationError


_NO_LEAP_MONTH_START = np.asarray(
    (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334),
    dtype=np.int64,
)
_TROPOPAUSE_MONTH_DAY = (
    (1, 16),
    (2, 14),
    (3, 16),
    (4, 15),
    (5, 16),
    (6, 15),
    (7, 16),
    (8, 16),
    (9, 15),
    (10, 16),
    (11, 15),
    (12, 16),
)

_FORTRAN_REAL_LITERAL = re.compile(
    r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[deDE][-+]?\d+)?"
)


def _fortran_real(text: str) -> np.float64:
    """Convert a Fortran real literal without accepting expressions."""

    match = _FORTRAN_REAL_LITERAL.fullmatch(text.strip())
    if match is None:
        raise ConfigurationError(f"unsupported Fortran real literal {text!r}")
    return np.float64(match.group(0).replace("D", "e").replace("d", "e"))


def _source_real_assignment(source: str, name: str) -> np.float64:
    """Read one scalar placeholder assignment from pinned CAM source."""

    match = re.search(
        rf"(?im)^\s*{re.escape(name)}\s*(?:\(:\))?\s*=\s*"
        rf"(?P<value>{_FORTRAN_REAL_LITERAL.pattern})_kind_phys\s*$",
        source,
    )
    if match is None:
        raise ConfigurationError(
            f"MUSICA placeholder source does not assign {name}"
        )
    return _fortran_real(match.group("value"))


def read_musica_placeholder_data(
    source_root: str | Path,
    *,
    horizontal_dimension: int,
    wavelength_interface_dimension: int,
    wavelength_section_dimension: int,
) -> dict[str, np.ndarray]:
    """Read the pinned upstream MUSICA placeholder host data.

    CAM-SIMA explicitly labels these values as placeholder data.  Reading the
    original source keeps Python's host-service contract aligned with the
    pinned revision without copying a second numerical table into freeCAM.
    """

    path = (
        Path(source_root).expanduser().resolve()
        / "src/physics/utils/musica_ccpp_dependencies.F90"
    )
    if not path.is_file():
        raise ConfigurationError(
            f"missing MUSICA placeholder source: {path}"
        )
    source = path.read_text(encoding="utf-8")
    match = re.search(
        r"(?is)photolysis_wavelength_grid_interfaces\s*=\s*"
        r"\(/\s*&(?P<body>.*?)\s*/\)",
        source,
    )
    if match is None:
        raise ConfigurationError(
            f"{path}: missing photolysis wavelength interface table"
        )
    literals = re.findall(
        rf"({_FORTRAN_REAL_LITERAL.pattern})_kind_phys",
        match.group("body"),
        flags=re.IGNORECASE,
    )
    interfaces = np.asfortranarray(
        np.asarray([_fortran_real(item) for item in literals])
    )
    if interfaces.size != wavelength_interface_dimension:
        raise ConfigurationError(
            f"{path}: source contains {interfaces.size} wavelength "
            f"interfaces, expected {wavelength_interface_dimension}"
        )
    if wavelength_section_dimension != wavelength_interface_dimension - 1:
        raise ConfigurationError(
            "MUSICA wavelength section dimension must be one less than the "
            "interface dimension"
        )
    if horizontal_dimension < 1:
        raise ConfigurationError(
            "MUSICA horizontal dimension must be positive"
        )

    surface_albedo = _source_real_assignment(source, "surface_albedo")
    surface_temperature = _source_real_assignment(
        source, "blackbody_temperature_at_surface"
    )
    extraterrestrial_flux = _source_real_assignment(
        source, "extraterrestrial_radiation_flux"
    )
    return {
        "photolysis_wavelength_grid_interfaces": interfaces,
        "extraterrestrial_radiation_flux": np.full(
            wavelength_section_dimension,
            extraterrestrial_flux,
            dtype=np.float64,
            order="F",
        ),
        "surface_albedo_due_to_uv_and_vis_direct": np.full(
            horizontal_dimension,
            surface_albedo,
            dtype=np.float64,
            order="F",
        ),
        "blackbody_temperature_at_surface": np.full(
            horizontal_dimension,
            surface_temperature,
            dtype=np.float64,
            order="F",
        ),
    }


def read_musica_initial_concentrations(
    source_root: str | Path,
    micm_configuration: str | Path,
) -> dict[str, np.float64]:
    """Read MUSICA's temporary startup concentrations from pinned inputs."""

    path = (
        Path(source_root).expanduser().resolve()
        / "src/physics/utils/musica_ccpp_dependencies.F90"
    )
    if not path.is_file():
        raise ConfigurationError(
            f"missing MUSICA placeholder source: {path}"
        )
    source = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?is)species_t\s*\(\s*&?\s*"
        r"['\"](?P<name>[^'\"]+)['\"]\s*,\s*"
        rf"(?P<value>{_FORTRAN_REAL_LITERAL.pattern})_kind_phys\s*\)"
    )
    concentrations = {
        match.group("name").strip().lower(): _fortran_real(
            match.group("value")
        )
        for match in pattern.finditer(source)
    }
    if not concentrations:
        raise ConfigurationError(
            f"{path}: missing temporary MUSICA species concentrations"
        )

    configuration_path = Path(micm_configuration).expanduser().resolve()
    try:
        configuration = json.loads(
            configuration_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"cannot read MUSICA MICM configuration: {configuration_path}"
        ) from exc
    files = configuration.get("camp-files")
    if not isinstance(files, list):
        raise ConfigurationError(
            f"{configuration_path}: missing camp-files array"
        )
    for relative in files:
        data_path = configuration_path.parent / str(relative)
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"cannot read MUSICA CAMP data: {data_path}"
            ) from exc
        entries = data.get("camp-data", [])
        if not isinstance(entries, list):
            raise ConfigurationError(
                f"{data_path}: camp-data must be an array"
            )
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            value = entry.get("__default mixing ratio [kg kg-1]")
            if name is not None and value is not None:
                concentrations[str(name).strip().lower()] = np.float64(value)
    return concentrations


def stage_musica_tuvx_configuration(
    path: str | Path,
    *,
    run_dir: str | Path,
    rank: int,
) -> Path:
    """Write a rank-local TUV-x config with absolute chemistry-data paths."""

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise ConfigurationError(
            f"missing MUSICA TUV-x configuration: {source_path}"
        )
    try:
        configuration = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(
            f"cannot read MUSICA TUV-x configuration: {source_path}"
        ) from exc

    try:
        mechanisms_root = source_path.parents[2]
    except IndexError as exc:
        raise ConfigurationError(
            f"unexpected MUSICA configuration layout: {source_path}"
        ) from exc

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str) and value.startswith(
            "musica_configurations/"
        ):
            relative = value.removeprefix("musica_configurations/")
            resolved = (mechanisms_root / relative).resolve()
            if not resolved.is_file():
                raise ConfigurationError(
                    "MUSICA TUV-x configuration references missing file: "
                    f"{resolved}"
                )
            return str(resolved)
        return value

    staged = resolve(configuration)
    target_dir = Path(run_dir).expanduser().resolve() / ".pycam_musica"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"tuvx-config-rank-{int(rank):06d}.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(staged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _linear_weights(
    source: np.ndarray,
    target: np.ndarray,
    *,
    cyclic: bool,
    cyclic_min: float = 0.0,
    cyclic_max: float = 360.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Port CAM ``interpolate_data::lininterp_init`` to zero-based indices."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or source.size < 2:
        raise ValueError("linear interpolation coordinates must be 1-D")
    differences = np.diff(source)
    increasing = bool(np.all(differences >= 0.0))
    decreasing = bool(np.all(differences <= 0.0))
    if not (increasing or decreasing):
        raise ValueError("source interpolation coordinates are not monotonic")

    lower = np.full(target.size, -1, dtype=np.int64)
    upper = np.full(target.size, -1, dtype=np.int64)
    weight_lower = np.empty(target.size, dtype=np.float64)
    weight_upper = np.empty(target.size, dtype=np.float64)

    if cyclic:
        span = np.float64(cyclic_max) - np.float64(cyclic_min)
        wrap = np.float64(source[0] + span - source[-1])
        average = np.float64(
            abs(source[-1] - source[0]) / np.float64(source.size - 1)
        )
        ratio = np.float64(wrap / average)
        if ratio < np.float64(0.9) or ratio > np.float64(1.1):
            raise ValueError(
                f"cyclic coordinate wrap {wrap} is inconsistent with "
                f"average spacing {average}"
            )
        for index, value in enumerate(target):
            if increasing:
                if value <= source[0]:
                    lower[index], upper[index] = source.size - 1, 0
                    weight_lower[index] = np.float64(
                        (source[0] - value) / wrap
                    )
                    weight_upper[index] = np.float64(
                        (value + span - source[-1]) / wrap
                    )
                elif value > source[-1]:
                    lower[index], upper[index] = source.size - 1, 0
                    weight_lower[index] = np.float64(
                        (source[0] + span - value) / wrap
                    )
                    weight_upper[index] = np.float64(
                        (value - source[-1]) / wrap
                    )
            else:
                if value > source[0]:
                    lower[index], upper[index] = source.size - 1, 0
                    weight_lower[index] = np.float64(
                        (source[0] - value) / wrap
                    )
                    weight_upper[index] = np.float64(
                        (value + span - source[-1]) / wrap
                    )
                elif value <= source[-1]:
                    lower[index], upper[index] = source.size - 1, 0
                    weight_lower[index] = np.float64(
                        (source[0] + span - value) / wrap
                    )
                    weight_upper[index] = np.float64(
                        (value + span - source[-1]) / wrap
                    )
    else:
        for index, value in enumerate(target):
            if increasing:
                if value <= source[0]:
                    lower[index] = upper[index] = 0
                    weight_lower[index], weight_upper[index] = 1.0, 0.0
                elif value > source[-1]:
                    lower[index] = upper[index] = source.size - 1
                    weight_lower[index], weight_upper[index] = 1.0, 0.0
            else:
                if value > source[0]:
                    lower[index] = upper[index] = 0
                    weight_lower[index], weight_upper[index] = 1.0, 0.0
                elif value <= source[-1]:
                    lower[index] = upper[index] = source.size - 1
                    weight_lower[index], weight_upper[index] = 1.0, 0.0

    for output_index, value in enumerate(target):
        for input_index in range(source.size - 1):
            if increasing:
                bracketed = (
                    value > source[input_index]
                    and value <= source[input_index + 1]
                )
            else:
                bracketed = (
                    value <= source[input_index]
                    and value > source[input_index + 1]
                )
            if not bracketed:
                continue
            lower[output_index] = input_index
            upper[output_index] = input_index + 1
            denominator = np.float64(
                source[input_index + 1] - source[input_index]
            )
            weight_lower[output_index] = np.float64(
                (source[input_index + 1] - value) / denominator
            )
            weight_upper[output_index] = np.float64(
                (value - source[input_index]) / denominator
            )
            break

    if np.any(lower < 0) or np.any(upper < 0):
        raise ValueError("failed to bracket every interpolation target")
    return lower, upper, weight_lower, weight_upper


def _bilinear_columns(
    source: np.ndarray,
    longitude_weights,
    latitude_weights,
) -> np.ndarray:
    """Port the expression order in CAM ``lininterp2d1d``."""

    lon_lower, lon_upper, west, east = longitude_weights
    lat_lower, lat_upper, south, north = latitude_weights
    output = np.empty((lon_lower.size, source.shape[2]), order="F")
    for column in range(lon_lower.size):
        for time_index in range(source.shape[2]):
            south_west = np.float64(
                np.float64(
                    source[lon_lower[column], lat_lower[column], time_index]
                    * west[column]
                )
                * south[column]
            )
            south_east = np.float64(
                np.float64(
                    source[lon_upper[column], lat_lower[column], time_index]
                    * east[column]
                )
                * south[column]
            )
            north_west = np.float64(
                np.float64(
                    source[lon_lower[column], lat_upper[column], time_index]
                    * west[column]
                )
                * north[column]
            )
            north_east = np.float64(
                np.float64(
                    source[lon_upper[column], lat_upper[column], time_index]
                    * east[column]
                )
                * north[column]
            )
            value = np.float64(south_west + south_east)
            value = np.float64(value + north_west)
            output[column, time_index] = np.float64(value + north_east)
    return output


def read_tropopause_climatology(
    path: str | Path,
    *,
    target_longitude: np.ndarray,
    target_latitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and regrid the exact monthly CAM tropopause climatology."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(
            f"missing tropopause climatology dataset: {resolved}"
        )
    with Dataset(resolved, "r") as dataset:
        dataset.set_auto_mask(False)
        dataset.set_auto_scale(False)
        try:
            longitude = np.asarray(
                dataset.variables["lon"][:], dtype=np.float64
            )
            latitude = np.asarray(
                dataset.variables["lat"][:], dtype=np.float64
            )
            pressure = np.asarray(
                dataset.variables["trop_p"][:], dtype=np.float64
            )
            time_count = len(dataset.dimensions["time"])
        except KeyError as exc:
            raise ConfigurationError(
                f"{resolved}: incomplete tropopause climatology dataset"
            ) from exc

    if time_count != 12:
        raise ConfigurationError(
            f"{resolved}: expected 12 climatology slices, got {time_count}"
        )
    if pressure.shape != (time_count, latitude.size, longitude.size):
        raise ConfigurationError(
            f"{resolved}: trop_p shape {pressure.shape} does not match "
            f"(time, lat, lon)"
        )

    pi = np.float64(np.pi)
    degrees_to_radians = np.float64(pi / np.float64(180.0))
    longitude = longitude * degrees_to_radians
    latitude = latitude * degrees_to_radians
    longitude_weights = _linear_weights(
        longitude,
        target_longitude,
        cyclic=True,
        cyclic_min=0.0,
        cyclic_max=np.float64(pi * np.float64(2.0)),
    )
    latitude_weights = _linear_weights(
        latitude,
        target_latitude,
        cyclic=False,
    )
    local_pressure = _bilinear_columns(
        np.asfortranarray(pressure.transpose(2, 1, 0)),
        longitude_weights,
        latitude_weights,
    )
    calendar_days = np.asarray(
        [
            _NO_LEAP_MONTH_START[month - 1] + day
            for month, day in _TROPOPAUSE_MONTH_DAY
        ],
        dtype=np.float64,
        order="F",
    )
    return local_pressure, calendar_days


def read_ridge_gravity_wave_data(
    path: str | Path,
    *,
    global_columns: np.ndarray,
    earth_radius: float,
    ridge_count: int,
) -> dict[str, np.ndarray]:
    """Read rank-local meso-beta ridge fields from CAM's PG3 topo file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(
            f"missing ridge gravity-wave dataset: {resolved}"
        )
    columns = np.asarray(global_columns, dtype=np.int64).reshape(
        -1, order="F"
    )
    indices = columns - 1
    if np.any(indices < 0):
        raise ConfigurationError(
            "physics global columns must use positive one-based indices"
        )

    one_dimensional = {
        "grid_box_area_for_beta_ridge_gravity_wave_drag": "GBXAR",
        "isotropic_variance_for_beta_ridge_gravity_wave_drag": "ISOVAR",
        "isotropic_weight_for_beta_ridge_gravity_wave_drag": "ISOWGT",
    }
    two_dimensional = {
        "ridge_half_width_for_beta_ridge_gravity_wave_drag": "HWDTH",
        "ridge_length_for_beta_ridge_gravity_wave_drag": "CLNGT",
        "ridge_obstacle_height_for_beta_ridge_gravity_wave_drag": "MXDIS",
        "ridge_anisotropy_for_beta_ridge_gravity_wave_drag": "ANIXY",
        "ridge_clockwise_angle_from_north_for_beta_ridge_gravity_wave_drag": "ANGLL",
    }
    result: dict[str, np.ndarray] = {}
    with Dataset(resolved, "r") as dataset:
        dataset.set_auto_mask(False)
        dataset.set_auto_scale(False)
        try:
            file_columns = len(dataset.dimensions["ncol"])
            file_ridges = len(dataset.dimensions["nrdg"])
        except KeyError as exc:
            raise ConfigurationError(
                f"{resolved}: ridge file needs ncol and nrdg dimensions"
            ) from exc
        if file_ridges < ridge_count:
            raise ConfigurationError(
                f"{resolved}: requires {ridge_count} ridges, has "
                f"{file_ridges}"
            )
        if np.any(indices >= file_columns):
            raise ConfigurationError(
                f"{resolved}: physics column exceeds ncol={file_columns}"
            )
        for standard_name, variable_name in one_dimensional.items():
            if variable_name not in dataset.variables:
                if variable_name in {"ISOVAR", "ISOWGT"}:
                    result[standard_name] = np.zeros(
                        columns.size, dtype=np.float64, order="F"
                    )
                    continue
                raise ConfigurationError(
                    f"{resolved}: missing required ridge variable "
                    f"{variable_name}"
                )
            values = np.asarray(
                dataset.variables[variable_name][:], dtype=np.float64
            )
            result[standard_name] = np.asfortranarray(values[indices])
        for standard_name, variable_name in two_dimensional.items():
            try:
                variable = dataset.variables[variable_name]
            except KeyError as exc:
                raise ConfigurationError(
                    f"{resolved}: missing required ridge variable "
                    f"{variable_name}"
                ) from exc
            values = np.asarray(variable[:], dtype=np.float64)
            if variable.dimensions == ("nrdg", "ncol"):
                local = values[:ridge_count, indices].T
            elif variable.dimensions == ("ncol", "nrdg"):
                local = values[indices, :ridge_count]
            else:
                raise ConfigurationError(
                    f"{resolved}: {variable_name} dimensions "
                    f"{variable.dimensions} are not nrdg/ncol"
                )
            result[standard_name] = np.asfortranarray(local)

    radius_km = np.float64(earth_radius) / np.float64(1000.0)
    area_name = "grid_box_area_for_beta_ridge_gravity_wave_drag"
    result[area_name] = np.asfortranarray(
        np.multiply(
            np.multiply(result[area_name], radius_km),
            radius_km,
        )
    )
    return result


def read_physics_topography(
    path: str | Path,
    *,
    global_columns: np.ndarray,
) -> np.ndarray:
    """Read rank-local ``PHIS`` values from CAM's physics-grid topo file.

    The file and ``physics_global_column`` use the same one-based,
    face-major PG ordering.  No regridding occurs here; CAM's PG-to-GLL
    interpolation is applied separately after every rank has read its local
    physics columns.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"missing CAM topography dataset: {resolved}")
    columns = np.asarray(global_columns, dtype=np.int64).reshape(
        -1, order="F"
    )
    indices = columns - 1
    if np.any(indices < 0):
        raise ConfigurationError(
            "physics global columns must use positive one-based indices"
        )
    with Dataset(resolved, "r") as dataset:
        dataset.set_auto_mask(False)
        dataset.set_auto_scale(False)
        try:
            file_columns = len(dataset.dimensions["ncol"])
            source = np.asarray(dataset.variables["PHIS"][:], dtype=np.float64)
        except KeyError as exc:
            raise ConfigurationError(
                f"{resolved}: topography file needs ncol and PHIS"
            ) from exc
    source = np.squeeze(source)
    if source.ndim != 1 or source.size != file_columns:
        raise ConfigurationError(
            f"{resolved}: PHIS shape {source.shape} does not match ncol="
            f"{file_columns}"
        )
    if indices.size and int(indices.max()) >= file_columns:
        raise ConfigurationError(
            f"{resolved}: physics column {int(indices.max()) + 1} exceeds "
            f"ncol={file_columns}"
        )
    return np.asfortranarray(source[indices])


def _ccpp_text(pool, standard_name: str) -> str:
    """Return one scalar CCPP character field as ordinary Python text."""

    value = np.asarray(pool.get_ccpp(standard_name)).reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="strict").strip().strip("\0")
    return str(value).strip().strip("\0")


def _set_ccpp(pool, standard_name: str, value) -> None:
    """Set an optional CCPP field without exposing its generated local name."""

    try:
        field_name = pool.ccpp_field_name(standard_name)
    except KeyError:
        return
    pool.set(field_name, value, unsafe=True)


def solar_irradiance_data_register(pool) -> None:
    """Provide CAM's file-sized solar dimensions from Python.

    Dimension discovery has already sized the StatePool before this CCPP
    lifecycle call.  Re-reading the file here preserves the register-phase
    validation performed by the original host-dependent Fortran scheme.
    """

    path_text = _ccpp_text(pool, "filename_of_solar_irradiance_data")
    if path_text == "NONE":
        sample_count = 1
    else:
        path = Path(path_text).expanduser().resolve()
        if not path.is_file():
            raise ConfigurationError(
                f"missing solar irradiance dataset: {path}"
            )
        with Dataset(path, "r") as dataset:
            wavelength_name = (
                "wavelength"
                if "wavelength" in dataset.variables
                else "wvl"
                if "wvl" in dataset.variables
                else None
            )
            if wavelength_name is None:
                raise ConfigurationError(
                    f"{path}: missing wavelength or wvl variable"
                )
            sample_count = int(dataset.variables[wavelength_name].size)

    expected = int(
        pool.dimensions["number_of_wavelength_samples_of_spectrum"]
    )
    if sample_count != expected:
        raise ConfigurationError(
            "solar wavelength dimension changed between StatePool allocation "
            f"and register: expected {expected}, found {sample_count}"
        )
    _set_ccpp(
        pool,
        "number_of_wavelength_samples_of_spectrum",
        sample_count,
    )
    _set_ccpp(
        pool,
        "number_of_wavelength_samples_of_spectrum_plus_one",
        sample_count + 1,
    )


def solar_irradiance_data_initialize(pool) -> None:
    """Read fixed CAM solar forcing while retaining native radiation kernels.

    This is the host-I/O part of ``solar_irradiance_data_init``.  CAM4's
    pinned case uses ``solar_data_type='FIXED'``; time-varying forcing fails
    closed rather than silently selecting the wrong record.
    """

    path_text = _ccpp_text(pool, "filename_of_solar_irradiance_data")
    solar_constant = np.float64(
        pool.get_ccpp("constant_total_solar_irradiance").item()
    )
    spectral_scaling_requested = bool(
        pool.get_ccpp(
            "do_solar_radiation_heating_spectral_scaling"
        ).item()
    )
    sample_count = int(
        pool.dimensions["number_of_wavelength_samples_of_spectrum"]
    )

    _set_ccpp(pool, "total_solar_irradiance", solar_constant)
    _set_ccpp(
        pool,
        "solar_irradiance_file_has_spectrum_information",
        False,
    )
    _set_ccpp(
        pool,
        "do_spectral_scaling_of_solar_irradiance_data",
        False,
    )
    if path_text == "NONE":
        _set_ccpp(
            pool,
            "wavelength_endpoints",
            np.zeros(sample_count + 1, dtype=np.float64, order="F"),
        )
        return

    data_type = _ccpp_text(pool, "type_of_solar_irradiance_data")
    if data_type != "FIXED":
        raise ConfigurationError(
            "Python solar irradiance host service currently supports the "
            f"pinned FIXED forcing contract, got {data_type!r}"
        )
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"missing solar irradiance dataset: {path}")

    with Dataset(path, "r") as dataset:
        dataset.set_auto_mask(False)
        dataset.set_auto_scale(False)
        wavelength_name = (
            "wavelength"
            if "wavelength" in dataset.variables
            else "wvl"
            if "wvl" in dataset.variables
            else None
        )
        if wavelength_name is None:
            raise ConfigurationError(
                f"{path}: missing wavelength or wvl variable"
            )
        wavelength = np.asarray(
            dataset.variables[wavelength_name][:],
            dtype=np.float64,
        ).reshape(-1)
        if wavelength.size != sample_count:
            raise ConfigurationError(
                f"{path}: wavelength size {wavelength.size} does not match "
                f"registered size {sample_count}"
            )
        try:
            band_width = np.asarray(
                dataset.variables["band_width"][:],
                dtype=np.float64,
            ).reshape(-1)
        except KeyError as exc:
            raise ConfigurationError(
                f"{path}: spectrum requires band_width"
            ) from exc
        if band_width.size != sample_count:
            raise ConfigurationError(
                f"{path}: band_width size {band_width.size} does not match "
                f"registered size {sample_count}"
            )

        has_spectrum = "ssi" in dataset.variables
        if has_spectrum:
            variable = dataset.variables["ssi"]
            values = np.asarray(variable[:], dtype=np.float64)
            wavelength_axis = next(
                (
                    index
                    for index, name in enumerate(variable.dimensions)
                    if name in {"wavelength", "wlen", "wvl"}
                ),
                None,
            )
            if wavelength_axis is None:
                raise ConfigurationError(
                    f"{path}: cannot identify SSI wavelength dimension "
                    f"{variable.dimensions}"
                )
            values = np.moveaxis(values, wavelength_axis, 0)
            spectrum = values.reshape(sample_count, -1, order="F")[:, 0]
        else:
            spectrum = np.zeros(sample_count, dtype=np.float64)

        total_solar_irradiance = solar_constant
        if solar_constant < np.float64(0.0) and "tsi" in dataset.variables:
            # The pinned upstream routine reads TSI through its integer
            # buffer. Preserve that conversion before assigning the real
            # CCPP output.
            tsi = np.asarray(dataset.variables["tsi"][:]).reshape(-1)
            if not tsi.size:
                raise ConfigurationError(f"{path}: empty tsi variable")
            total_solar_irradiance = np.float64(
                np.asarray(tsi[0], dtype=np.int32)
            )

    endpoints = np.empty(sample_count + 1, dtype=np.float64, order="F")
    endpoints[:sample_count] = (
        wavelength
        - np.float64(0.5) * band_width
    )
    endpoints[sample_count] = (
        wavelength[-1]
        + np.float64(0.5) * band_width[-1]
    )
    solar_irradiance = np.asfortranarray(
        spectrum * np.float64(1.0e-3)
    )
    _set_ccpp(
        pool,
        "solar_irradiance_file_has_spectrum_information",
        has_spectrum,
    )
    _set_ccpp(
        pool,
        "do_spectral_scaling_of_solar_irradiance_data",
        has_spectrum and spectral_scaling_requested,
    )
    _set_ccpp(pool, "wavelength_endpoints", endpoints)
    _set_ccpp(pool, "solar_irradiance", solar_irradiance)
    _set_ccpp(
        pool,
        "total_solar_irradiance",
        total_solar_irradiance,
    )


def solar_irradiance_data_timestep_initial(pool) -> None:
    """Honor the fixed-forcing lifecycle without re-reading the dataset."""

    data_type = _ccpp_text(pool, "type_of_solar_irradiance_data")
    if data_type != "FIXED":
        raise ConfigurationError(
            "time-varying solar irradiance requires a time-coordinate host "
            "service; the current scientific oracle is FIXED"
        )


def solar_irradiance_data_finalize(pool) -> None:
    """Solar file resources are scoped to initialization and already closed."""
