"""Python host service backed by CESM's unmodified orbital utility."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Iterable

import numpy as np

from .device_codegen import _validate_elf
from .errors import DeviceBuildError, DeviceContractError


def build_orbital_host_library(
    project_root: str | Path,
    *,
    compiler: str,
    fflags: Iterable[str],
    ldflags: Iterable[str] = (),
) -> Path:
    """Build the host bridge from the pinned original ``shr_orb_mod``."""

    root = Path(project_root).resolve()
    output = root / "build/host_services/orbital"
    module_dir = output / "mod"
    output.mkdir(parents=True, exist_ok=True)
    module_dir.mkdir(parents=True, exist_ok=True)
    sources = (
        root / "native/devices/support/shr_kind_mod.F90",
        root / "native/devices/support/shr_log_mod.F90",
        root / "native/devices/support/shr_sys_mod.F90",
        root / "external/CAM-SIMA/share/src/shr_const_mod.F90",
        root / "external/CAM-SIMA/share/src/shr_orb_mod.F90",
        root / "native/host/orbital_host_adapter.F90",
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise DeviceBuildError(
            f"orbital host sources are missing: {missing}"
        )
    library = output / "libpycam_orbital_host.so"
    command = [
        str(Path(compiler).absolute()),
        *fflags,
        "-shared",
        *ldflags,
        "-J",
        str(module_dir),
        "-I",
        str(module_dir),
        "-o",
        str(library),
        *(str(path) for path in sources),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": os.environ.get("HOME", ""),
        "LC_ALL": "C",
    }
    try:
        subprocess.run(command, check=True, env=environment, cwd=root)
        _validate_elf(library)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DeviceBuildError(
            f"failed to compile orbital host library: {exc}"
        ) from exc

    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.read_bytes())
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "library": library.name,
                "source_sha256": digest.hexdigest(),
                "sources": [
                    str(path.relative_to(root)) for path in sources
                ],
                "forbidden_runtime_dependencies": ["MPI", "ESMF", "PIO"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return library


class OrbitalHostService:
    """Evaluate CAM orbital state without reimplementing its formulas."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        self.library_path = (
            self.project_root
            / "build/host_services/orbital/libpycam_orbital_host.so"
        )
        self._library = None
        self._function = None

    def _load(self):
        if self._function is not None:
            return self._function
        if not self.library_path.is_file():
            raise DeviceContractError(
                "orbital host library is missing; run "
                "`python -m freecam.cli build-catalog-devices --strict`"
            )
        self._library = ctypes.CDLL(
            str(self.library_path),
            mode=getattr(ctypes, "RTLD_LOCAL", 0),
        )
        function = self._library.pycam_orbital_advance_v1
        array = np.ctypeslib.ndpointer(
            dtype=np.float64,
            ndim=1,
            flags=("F_CONTIGUOUS", "ALIGNED"),
        )
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            array,
            array,
            ctypes.c_double,
            ctypes.c_bool,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            array,
            array,
        ]
        function.restype = ctypes.c_int
        self._function = function
        return function

    @staticmethod
    def _optional(pool, standard_name: str):
        try:
            return pool.get_ccpp(standard_name, unsafe=True)
        except KeyError:
            return None

    def update(self, pool, clock, *, orbital_year: int) -> bool:
        """Update all orbital fields present in the selected suite."""

        solar_zenith = self._optional(pool, "solar_zenith_angle")
        cosine_zenith = self._optional(
            pool, "cosine_of_solar_zenith_angle_for_radiation"
        )
        earth_sun_distance = self._optional(pool, "earth_sun_distance")
        solar_declination = self._optional(pool, "solar_declination_angle")
        if all(
            value is None
            for value in (
                solar_zenith,
                cosine_zenith,
                earth_sun_distance,
                solar_declination,
            )
        ):
            return False

        latitude = np.asfortranarray(
            pool.get_ccpp("latitude"), dtype=np.float64
        )
        longitude = np.asfortranarray(
            pool.get_ccpp("longitude"), dtype=np.float64
        )
        columns = latitude.size
        computed_zenith = np.empty(columns, dtype=np.float64, order="F")
        computed_cosine = np.empty(columns, dtype=np.float64, order="F")
        declination = ctypes.c_double()
        distance = ctypes.c_double()
        averaging = self._optional(
            pool,
            "averaging_time_interval_for_solar_zenith_angle_calculation",
        )
        averaging_seconds = (
            float(averaging.item()) if averaging is not None else 0.0
        )
        uniform_flag = self._optional(
            pool,
            "use_radiation_uniform_angle_in_solar_zenith_angle_calculation",
        )
        uniform_angle = self._optional(
            pool,
            "radiation_uniform_angle_in_solar_zenith_angle_calculation",
        )
        status = self._load()(
            int(orbital_year),
            int(columns),
            float(clock.fractional_calendar_day()),
            latitude,
            longitude,
            averaging_seconds,
            bool(uniform_flag.item()) if uniform_flag is not None else False,
            float(uniform_angle.item()) if uniform_angle is not None else -99.0,
            ctypes.byref(declination),
            ctypes.byref(distance),
            computed_zenith,
            computed_cosine,
        )
        if status:
            raise RuntimeError(
                f"pinned CESM orbital host returned status {status}"
            )
        if solar_zenith is not None:
            solar_zenith[...] = computed_zenith
            pool.mark_initialized(
                pool.ccpp_field_name("solar_zenith_angle")
            )
        if cosine_zenith is not None:
            cosine_zenith[...] = computed_cosine
            pool.mark_initialized(
                pool.ccpp_field_name(
                    "cosine_of_solar_zenith_angle_for_radiation"
                )
            )
        if earth_sun_distance is not None:
            earth_sun_distance[...] = distance.value
            pool.mark_initialized(
                pool.ccpp_field_name("earth_sun_distance")
            )
        if solar_declination is not None:
            solar_declination[...] = declination.value
            pool.mark_initialized(
                pool.ccpp_field_name("solar_declination_angle")
            )
        return True
