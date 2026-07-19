from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .state_pool import FieldSpec, StatePool


@dataclass(frozen=True)
class NativeField:
    field_id: int
    standard_name: str
    dimensions: tuple[str, ...]


FULL_CAM_FIELDS = (
    NativeField(1, "air_temperature", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(2, "eastward_wind", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(3, "northward_wind", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(4, "surface_air_pressure", ("horizontal_dimension",)),
    NativeField(5, "air_pressure_thickness", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(6, "air_pressure_thickness_of_dry_air", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(7, "air_pressure", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(8, "air_pressure_of_dry_air", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(9, "air_pressure_at_interface", ("horizontal_dimension", "vertical_interface_dimension")),
    NativeField(10, "air_pressure_of_dry_air_at_interface", ("horizontal_dimension", "vertical_interface_dimension")),
    NativeField(11, "surface_pressure_of_dry_air", ("horizontal_dimension",)),
    NativeField(12, "surface_geopotential", ("horizontal_dimension",)),
    NativeField(13, "geopotential_height_wrt_surface", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(14, "geopotential_height_wrt_surface_at_interface", ("horizontal_dimension", "vertical_interface_dimension")),
    NativeField(15, "lagrangian_tendency_of_air_pressure", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(16, "reciprocal_of_dimensionless_exner_function_wrt_surface_air_pressure", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(17, "dry_static_energy", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(18, "tendency_of_air_temperature_due_to_model_physics", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(19, "tendency_of_eastward_wind_due_to_model_physics", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(20, "tendency_of_northward_wind_due_to_model_physics", ("horizontal_dimension", "vertical_layer_dimension")),
    NativeField(21, "ccpp_constituents", ("horizontal_dimension", "vertical_layer_dimension", "number_of_ccpp_constituents")),
)


class FullNativeBackend:
    """Typed CFFI bridge to a complete CAM-SIMA/SE runtime.

    CAM owns the long-lived allocations required by its derived types.  This
    class exposes them as writable, zero-copy NumPy arrays in ``StatePool``.
    """

    def __init__(self, library: str | Path) -> None:
        from cffi import FFI

        self.library = Path(library).resolve()
        if not self.library.is_file():
            raise FileNotFoundError(f"full CAM-SIMA library not found: {self.library}")
        self.ffi = FFI()
        self.ffi.cdef(
            """
            int pycam_full_abi_version(void);
            int pycam_full_initialize(int comm, int timestep_seconds);
            int pycam_full_timestep_init(void);
            int pycam_full_run1(void);
            int pycam_full_run2(void);
            int pycam_full_run3(void);
            int pycam_full_run4(void);
            int pycam_full_timestep_final(void);
            int pycam_full_advance_timestep(void);
            int pycam_full_finalize(void);
            int pycam_full_get_nstep(void);
            int pycam_full_get_field(int field_id, void **data, int *rank, int dims[4]);
            """
        )
        self.lib = self.ffi.dlopen(str(self.library))
        version = int(self.lib.pycam_full_abi_version())
        if version != 3:
            raise RuntimeError(f"unsupported full CAM-SIMA ABI version {version}")
        self._buffers: dict[str, Any] = {}
        self._initialized = False

    @property
    def nstep(self) -> int:
        return int(self.lib.pycam_full_get_nstep())

    def initialize(self, comm: Any, timestep_seconds: int) -> None:
        if self._initialized:
            raise RuntimeError("full CAM-SIMA backend is already initialized")
        if timestep_seconds <= 0:
            raise ValueError("timestep_seconds must be positive")
        try:
            fortran_comm = int(comm.py2f())
        except AttributeError as exc:
            raise TypeError("full CAM-SIMA requires an mpi4py communicator") from exc
        self._call(
            "initialize",
            self.lib.pycam_full_initialize,
            fortran_comm,
            int(timestep_seconds),
        )
        self._initialized = True

    def timestep_init(self) -> None:
        self._call("timestep_init", self.lib.pycam_full_timestep_init)

    def run1(self) -> None:
        self._call("run1", self.lib.pycam_full_run1)

    def run2(self) -> None:
        self._call("run2", self.lib.pycam_full_run2)

    def run3(self) -> None:
        self._call("run3", self.lib.pycam_full_run3)

    def run4(self) -> None:
        self._call("run4", self.lib.pycam_full_run4)

    def timestep_final(self) -> None:
        self._call("timestep_final", self.lib.pycam_full_timestep_final)

    def advance_timestep(self) -> None:
        self._call("advance_timestep", self.lib.pycam_full_advance_timestep)

    def finalize(self) -> None:
        self._call("finalize", self.lib.pycam_full_finalize)
        self._buffers.clear()
        self._initialized = False

    def attach_state(self, pool: StatePool) -> None:
        for field in FULL_CAM_FIELDS:
            data = self.ffi.new("void **")
            rank = self.ffi.new("int *")
            dims = self.ffi.new("int[4]")
            ierr = int(self.lib.pycam_full_get_field(field.field_id, data, rank, dims))
            if ierr:
                raise RuntimeError(
                    f"cannot expose full CAM field {field.standard_name}: error {ierr}"
                )
            shape = tuple(int(dims[index]) for index in range(int(rank[0])))
            size = int(np.prod(shape, dtype=np.int64))
            buffer = self.ffi.buffer(data[0], size * np.dtype(np.float64).itemsize)
            array = np.ndarray(shape, dtype=np.float64, buffer=buffer, order="F")
            spec = FieldSpec(
                field.standard_name,
                np.dtype(np.float64),
                field.dimensions,
                owner="native_view",
            )
            pool.register(spec, array)
            self._buffers[field.standard_name] = buffer

    @staticmethod
    def _check_initialized(initialized: bool) -> None:
        if not initialized:
            raise RuntimeError("full CAM-SIMA backend is not initialized")

    def _call(self, name: str, function: Any, *args: Any) -> None:
        if name != "initialize":
            self._check_initialized(self._initialized)
        ierr = int(function(*args))
        if ierr:
            raise RuntimeError(f"full CAM-SIMA {name} failed with error code {ierr}")
