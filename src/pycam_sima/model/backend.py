"""Stateless numerical-kernel backend used by the Python model."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .devices import DeviceRegistry
from .errors import MissingKernelError, StateOwnershipError


_F64_F = np.ctypeslib.ndpointer(dtype=np.float64, flags=("F_CONTIGUOUS", "ALIGNED"))
_I32_F = np.ctypeslib.ndpointer(dtype=np.int32, flags=("F_CONTIGUOUS", "ALIGNED"))


class _FVMDimensionsABI(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_int)
        for name in (
            "nc", "nlev", "ntrac", "np", "ngpc", "irecons",
            "nhe", "nhr", "nht", "ns", "nhc", "kmin_jet",
            "kmax_jet", "large_courant", "level_begin", "level_end",
        )
    ]


@dataclass(frozen=True, slots=True)
class FVMKernelConfig:
    """Python-owned dimensions and controls passed into each FVM ABI call."""

    nc: int
    nlev: int
    ntrac: int
    np: int
    ngpc: int
    irecons: int
    nhe: int
    nhr: int
    nht: int
    ns: int
    nhc: int
    kmin_jet: int
    kmax_jet: int
    large_courant: bool
    level_begin: int
    level_end: int
    irecons_levels: np.ndarray

    @classmethod
    def from_pool(cls, pool) -> "FVMKernelConfig":
        dimensions = pool.dimensions
        nc = dimensions["fv_nphys"]
        nlev = dimensions["pver"]
        irecons = dimensions["fvm_reconstruction"] + 1
        return cls(
            nc=nc,
            nlev=nlev,
            ntrac=dimensions["nconst"],
            np=dimensions["np"],
            ngpc=dimensions["fv_nphys"],
            irecons=irecons,
            nhe=(dimensions["fvm_internal"] - nc) // 2,
            nhr=(dimensions["fvm_interp_span"] - nc) // 2,
            nht=dimensions["fvm_stretch"] - nc - 1,
            ns=dimensions["fv_nphys"],
            nhc=(dimensions["fvm_halo"] - nc) // 2,
            kmin_jet=1,
            kmax_jet=nlev,
            large_courant=True,
            level_begin=1,
            level_end=nlev,
            irecons_levels=np.full(nlev, irecons, dtype=np.int32, order="F"),
        )

    def abi(self) -> _FVMDimensionsABI:
        return _FVMDimensionsABI(
            self.nc, self.nlev, self.ntrac, self.np, self.ngpc,
            self.irecons, self.nhe, self.nhr, self.nht, self.ns, self.nhc,
            self.kmin_jet, self.kmax_jet, int(self.large_courant),
            self.level_begin, self.level_end,
        )


class KernelBackend:
    def __init__(
        self,
        library: str | Path,
        *,
        device_root: str | Path | None = None,
    ):
        self.path = Path(library).resolve()
        if not self.path.is_file():
            raise MissingKernelError(f"kernel library does not exist: {self.path}")
        self.lib = ctypes.CDLL(str(self.path), mode=ctypes.RTLD_LOCAL)
        if device_root is None:
            build_root = next(
                (
                    parent
                    for parent in self.path.parents
                    if parent.name == "build"
                ),
                self.path.parent,
            )
            roots = tuple(
                dict.fromkeys(
                    (
                        self.path.parent / "devices",
                        self.path.parent / "catalog_devices",
                        build_root / "devices",
                        build_root / "catalog_devices",
                    )
                )
            )
        else:
            roots = (device_root,)
        self.devices = DeviceRegistry(roots)
        self.call_count = 0
        self.lib.pycam_sima_abi_version.restype = ctypes.c_int
        self._kernel_specialization = (
            self.lib.pycam_sima_kernel_specialization_v1
        )
        self._kernel_specialization.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._kernel_specialization.restype = None
        self._validate_se_dimensions = self.lib.pycam_sima_validate_se_dimensions_v2
        self._validate_se_dimensions.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)
        ]
        self._validate_se_dimensions.restype = None
        self._abi_checked = False
        self._wind_tendency = self.lib.pycam_sima_wind_tendency_v2
        self._wind_tendency.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ] + [_F64_F] * 12
        self._wind_tendency.restype = None
        self._scalar_laplace_weak = self.lib.pycam_sima_laplace_weak_v2
        self._scalar_laplace_weak.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
        ] + [_F64_F] * 5
        self._scalar_laplace_weak.restype = None
        self._vector_laplace_weak = self.lib.pycam_sima_vector_laplace_weak_v2
        self._vector_laplace_weak.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
        ] + [_F64_F] * 10
        self._vector_laplace_weak.restype = None
        self._hypervis_reference = self.lib.pycam_sima_hypervis_reference_v2
        self._hypervis_reference.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ] + [ctypes.c_double] * 7 + [_F64_F] * 9
        self._hypervis_reference.restype = None
        self._limiter_optim = self.lib.pycam_sima_limiter_optim_v2
        self._limiter_optim.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ] + [_F64_F] * 5
        self._limiter_optim.restype = None
        self._remap_fv3 = self.lib.pycam_sima_remap_fv3_v1
        self._remap_fv3.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_double,
        ] + [_F64_F] * 3
        self._remap_fv3.restype = None
        self._fvm_transport = self.lib.pycam_sima_fvm_transport_v2
        self._fvm_transport.argtypes = (
            [ctypes.POINTER(_FVMDimensionsABI), _I32_F, ctypes.c_double]
            + [_F64_F] * 9
            + [ctypes.c_int, _F64_F, _I32_F, _F64_F, _F64_F, _I32_F, _I32_F]
            + [_F64_F] * 3
            + [_I32_F] * 5
            + [_F64_F] * 3
            + [ctypes.POINTER(ctypes.c_int)]
        )
        self._fvm_transport.restype = None
        self._fvm_displacement = self.lib.pycam_sima_fvm_displacement_v2
        self._fvm_displacement.argtypes = [
            ctypes.POINTER(_FVMDimensionsABI), _I32_F,
        ] + [_F64_F] * 4 + [ctypes.POINTER(ctypes.c_int)]
        self._fvm_displacement.restype = None
        self._physics_diagnostics = self.lib.pycam_sima_physics_diagnostics_v1
        self._physics_diagnostics.argtypes = [ctypes.c_int] * 4 + [
            ctypes.c_double,
            ctypes.c_double,
        ] + [_F64_F] * 10
        self._physics_diagnostics.restype = None

    @property
    def available_phases(self) -> frozenset[str]:
        return self.devices.process_names

    @property
    def specialization(self) -> dict[str, int]:
        values = [ctypes.c_int() for _ in range(4)]
        self._kernel_specialization(
            *(ctypes.byref(value) for value in values)
        )
        return dict(
            zip(
                ("np", "fv_nphys", "pver", "constituent_count"),
                (int(value.value) for value in values),
            )
        )

    def validate_specialization(self, config) -> None:
        expected = config.kernel_specialization
        actual = self.specialization
        if actual != expected:
            raise MissingKernelError(
                f"kernel library {self.path} is specialized for {actual}, "
                f"but the model requires {expected}; build it with "
                f"`pycam-sima build-kernels --config <config.yaml>`"
            )

    def run_phase(self, phase: str, pool) -> None:
        if phase not in self.available_phases:
            raise MissingKernelError(
                f"phase {phase!r} has no generated device implementation"
            )
        before = pool.pointer_records()
        self.devices.invoke(phase, pool)
        self.call_count += 1
        pool.assert_pointer_stability(before)

    def _ensure_abi(self) -> None:
        if not self._abi_checked:
            if self.lib.pycam_sima_abi_version() != 2:
                raise MissingKernelError("unsupported pycam_sima kernel ABI")
            self._abi_checked = True

    def _require_arrays(self, label: str, arrays) -> None:
        self._ensure_abi()
        if not all(value.flags.f_contiguous and value.dtype == np.float64 for value in arrays):
            raise StateOwnershipError(
                f"{label} ABI requires Fortran-contiguous float64 arrays"
            )

    def _require_se_dimensions(self, np_value: int, ngp_value: int) -> None:
        error = ctypes.c_int()
        self._validate_se_dimensions(
            int(np_value), int(ngp_value), ctypes.byref(error)
        )
        if error.value:
            raise StateOwnershipError(
                f"SE ABI dimensions np={np_value}, ngp={ngp_value} are unsupported"
            )

    def scalar_laplace_weak(
        self, *, inverse_radius, derivative, inverse_metric, mass, scalar, output
    ) -> None:
        arrays = (derivative, inverse_metric, mass, scalar, output)
        self._require_arrays("scalar-laplace", arrays)
        self._require_se_dimensions(scalar.shape[0], scalar.shape[0] ** 2)
        self._scalar_laplace_weak(
            scalar.shape[3],
            scalar.shape[2],
            scalar.shape[0],
            float(inverse_radius),
            *arrays,
        )
        self.call_count += 1

    def vector_laplace_weak(
        self,
        *,
        inverse_radius,
        divergence_ratio,
        derivative,
        metric,
        inverse_metric,
        inverse_metric_tensor,
        metric_jacobian,
        inverse_metric_jacobian,
        mass,
        reference_mass,
        vector,
        output,
    ) -> None:
        arrays = (
            derivative,
            metric,
            inverse_metric,
            inverse_metric_tensor,
            metric_jacobian,
            inverse_metric_jacobian,
            mass,
            reference_mass,
            vector,
            output,
        )
        self._require_arrays("vector-laplace", arrays)
        self._require_se_dimensions(vector.shape[0], vector.shape[0] ** 2)
        self._vector_laplace_weak(
            vector.shape[4],
            vector.shape[3],
            vector.shape[0],
            float(inverse_radius),
            float(divergence_ratio),
            *arrays,
        )
        self.call_count += 1

    def hypervis_reference(
        self,
        *,
        reference_pressure,
        dry_air_gas_constant,
        dry_air_specific_heat,
        gravity,
        reference_temperature,
        lapse_rate,
        kappa,
        hybrid_a_interface,
        hybrid_b_interface,
        hybrid_a_midpoint,
        hybrid_b_midpoint,
        surface_geopotential,
        pressure_thickness,
        temperature,
        surface_pressure,
        sponge_scale,
    ) -> None:
        arrays = (
            hybrid_a_interface,
            hybrid_b_interface,
            hybrid_a_midpoint,
            hybrid_b_midpoint,
            surface_geopotential,
            pressure_thickness,
            temperature,
            surface_pressure,
            sponge_scale,
        )
        self._require_arrays("hypervis-reference", arrays)
        self._require_se_dimensions(
            surface_geopotential.shape[0], surface_geopotential.shape[0] ** 2
        )
        self._hypervis_reference(
            surface_geopotential.shape[2],
            pressure_thickness.shape[2],
            surface_geopotential.shape[0],
            float(reference_pressure),
            float(dry_air_gas_constant),
            float(dry_air_specific_heat),
            float(gravity),
            float(reference_temperature),
            float(lapse_rate),
            float(kappa),
            *arrays,
        )
        self.call_count += 1

    def wind_tendency(
        self,
        *,
        inverse_radius,
        reference_pressure,
        dry_specific_heat,
        derivative,
        inverse_metric,
        coriolis,
        zonal,
        meridional,
        virtual_temperature,
        pressure,
        geopotential,
        kappa,
        vorticity,
        zonal_tendency,
        meridional_tendency,
    ) -> None:
        """Evaluate the FP-sensitive primitive-equation wind terms statelessly."""

        self._ensure_abi()
        arrays = (
            derivative,
            inverse_metric,
            coriolis,
            zonal,
            meridional,
            virtual_temperature,
            pressure,
            geopotential,
            kappa,
            vorticity,
            zonal_tendency,
            meridional_tendency,
        )
        if not all(value.flags.f_contiguous and value.dtype == np.float64 for value in arrays):
            raise StateOwnershipError(
                "wind-tendency ABI requires Fortran-contiguous float64 arrays"
            )
        nlev = zonal.shape[2]
        nelem = zonal.shape[3]
        self._require_se_dimensions(zonal.shape[0], zonal.shape[0] ** 2)
        self._wind_tendency(
            nelem,
            nlev,
            zonal.shape[0],
            float(inverse_radius),
            float(reference_pressure),
            float(dry_specific_heat),
            *arrays,
        )
        self.call_count += 1

    def limiter_optim(self, *, tracer_mass, mass, minimum, maximum, dry_mass) -> None:
        arrays = (tracer_mass, mass, minimum, maximum, dry_mass)
        self._require_arrays("tracer-limiter", arrays)
        np_value = tracer_mass.shape[0]
        ngp_value = tracer_mass.shape[0] * tracer_mass.shape[1]
        self._require_se_dimensions(np_value, ngp_value)
        self._limiter_optim(
            tracer_mass.shape[4],
            tracer_mass.shape[2],
            tracer_mass.shape[3],
            *arrays,
        )
        self.call_count += 1

    def remap_fv3(
        self,
        *,
        field,
        source_pressure_thickness,
        target_pressure_thickness,
        pressure_top,
        identifier,
        mass_field,
        method=-9,
    ) -> None:
        arrays = (field, source_pressure_thickness, target_pressure_thickness)
        self._require_arrays("FV3 vertical-remap", arrays)
        if field.ndim != 4 or source_pressure_thickness.ndim != 3:
            raise StateOwnershipError("FV3 vertical-remap ABI requires 4-D field and 3-D pressure arrays")
        self._remap_fv3(
            field.shape[0],
            field.shape[2],
            field.shape[3],
            int(identifier),
            bool(mass_field),
            int(method),
            float(pressure_top),
            field,
            source_pressure_thickness,
            target_pressure_thickness,
        )
        self.call_count += 1

    def fvm_transport(
        self,
        *,
        config: FVMKernelConfig,
        dt,
        subelement_flux,
        tracer,
        pressure_thickness,
        surface_pressure,
        swept_flux,
        reference_pressure_thickness,
        inverse_reference_pressure_thickness,
        cell_area,
        inverse_cell_area,
        cube_boundary,
        displacement_maximum,
        flux_vector,
        vertex_cartesian,
        flux_orientation,
        cell_indicator,
        rotation_matrix,
        sphere_centroid,
        reconstruction_metric,
        reconstruction_metric_integral,
        jx_min,
        jx_max,
        jy_min,
        jy_max,
        interpolation_base,
        halo_interpolation_weight,
        centroid_stretch,
        vertex_reconstruction_weight,
    ) -> None:
        arguments = (
            subelement_flux, tracer, pressure_thickness, surface_pressure,
            swept_flux, reference_pressure_thickness,
            inverse_reference_pressure_thickness, cell_area, inverse_cell_area,
            displacement_maximum, flux_vector, vertex_cartesian,
            flux_orientation, cell_indicator, rotation_matrix, sphere_centroid,
            reconstruction_metric, reconstruction_metric_integral, jx_min,
            jx_max, jy_min, jy_max, interpolation_base,
            halo_interpolation_weight, centroid_stretch,
            vertex_reconstruction_weight,
        )
        for value in arguments:
            if not value.flags.f_contiguous:
                raise StateOwnershipError("FVM transport ABI requires Fortran-contiguous arrays")
        native_config = config.abi()
        error = ctypes.c_int()
        self._fvm_transport(
            ctypes.byref(native_config),
            config.irecons_levels,
            float(dt),
            subelement_flux,
            tracer,
            pressure_thickness,
            surface_pressure,
            swept_flux,
            reference_pressure_thickness,
            inverse_reference_pressure_thickness,
            cell_area,
            inverse_cell_area,
            int(cube_boundary),
            displacement_maximum,
            flux_vector,
            vertex_cartesian,
            flux_orientation,
            cell_indicator,
            rotation_matrix,
            sphere_centroid,
            reconstruction_metric,
            reconstruction_metric_integral,
            jx_min,
            jx_max,
            jy_min,
            jy_max,
            interpolation_base,
            halo_interpolation_weight,
            centroid_stretch,
            vertex_reconstruction_weight,
            ctypes.byref(error),
        )
        if error.value:
            raise RuntimeError(f"stateless FVM transport kernel failed with code {error.value}")
        self.call_count += 1

    def fvm_displacement(
        self, *, config: FVMKernelConfig, pressure_thickness, swept_flux,
        displacement_maximum, vertex_cartesian
    ) -> None:
        arrays = (
            pressure_thickness,
            swept_flux,
            displacement_maximum,
            vertex_cartesian,
        )
        self._require_arrays("FVM displacement", arrays)
        native_config = config.abi()
        error = ctypes.c_int()
        self._fvm_displacement(
            ctypes.byref(native_config), config.irecons_levels, *arrays,
            ctypes.byref(error),
        )
        if error.value:
            raise RuntimeError(f"stateless FVM displacement kernel failed with code {error.value}")
        self.call_count += 1

    def physics_diagnostics(
        self,
        *,
        update_static_energy,
        gravity,
        virtual_temperature_coefficient,
        temperature,
        constituent,
        pressure_thickness,
        midpoint_pressure,
        gas_constant,
        heat_capacity,
        surface_geopotential,
        interface_height,
        midpoint_height,
        static_energy,
    ) -> None:
        arrays = (
            temperature,
            constituent,
            pressure_thickness,
            midpoint_pressure,
            gas_constant,
            heat_capacity,
            surface_geopotential,
            interface_height,
            midpoint_height,
            static_energy,
        )
        self._require_arrays("physics diagnostics", arrays)
        self._physics_diagnostics(
            temperature.shape[0],
            temperature.shape[1],
            constituent.shape[2],
            int(bool(update_static_energy)),
            float(gravity),
            float(virtual_temperature_coefficient),
            *arrays,
        )
        self.call_count += 1
