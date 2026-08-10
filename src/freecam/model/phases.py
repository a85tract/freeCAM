"""Explicit model phase boundaries controlled by Python.

Some routines in this module are transcribed from CESM/CAM Fortran sources
and preserve their upstream expression order; each is marked with a ``Port
...`` docstring naming its upstream routine.  Those routines are
Copyright (c) 2017, University Corporation for Atmospheric Research (UCAR)
and are redistributed under the BSD 3-Clause license in
LICENSES/UCAR-CESM-BSD-3-Clause.txt.  See NOTICE section 2.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .constituents import (
    is_water_constituent,
    is_water_vapor,
    water_constituent_indices,
    water_species_flags,
    water_vapor_index,
)
from .grid import _physics_reference_nodes


def _integrate_subcells(sample: np.ndarray, metdet: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scalar-order equivalent of derivative_mod:subcell_integration."""
    np_value = sample.shape[0]
    nc = weights.shape[0]
    val = np.empty((np_value, np_value), dtype=np.float64, order="F")
    tmp = np.empty((np_value, nc), dtype=np.float64, order="F")
    result = np.empty((nc, nc), dtype=np.float64, order="F")
    for j in range(np_value):
        for i in range(np_value):
            val[i, j] = np.float64(sample[i, j] * metdet[i, j])
    # MATMUL(val, TRANSPOSE(weights))
    for j in range(nc):
        for i in range(np_value):
            value = np.float64(0.0)
            for k in range(np_value):
                value = np.float64(value + np.float64(val[i, k] * weights[j, k]))
            tmp[i, j] = value
    # MATMUL(weights, tmp)
    for j in range(nc):
        for i in range(nc):
            value = np.float64(0.0)
            for k in range(np_value):
                value = np.float64(value + np.float64(weights[i, k] * tmp[k, j]))
            result[i, j] = value
    return result


def _interpolate_tensor_point(field: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.float64:
    np_value = field.shape[0]
    intermediate = np.empty(np_value, dtype=np.float64)
    for j in range(np_value):
        value = np.float64(0.0)
        for i in range(np_value):
            value = np.float64(value + np.float64(wx[i] * field[i, j]))
        intermediate[j] = value
    value = np.float64(0.0)
    for j in range(np_value):
        value = np.float64(value + np.float64(wy[j] * intermediate[j]))
    return value


def _interpolate_legendre_2d(field: np.ndarray, x: float, y: float, matrix: np.ndarray) -> np.float64:
    if field.shape[0] != 4:
        return _interpolate_legendre_2d_generic(field, x, y, matrix)
    vtemp = np.empty(4, dtype=np.float64)
    for l in range(0, 4, 2):
        pk = np.float64(1.0)
        fk0 = np.float64(0.0)
        fk1 = np.float64(0.0)
        for j in range(4):
            fk0 = np.float64(fk0 + np.float64(matrix[j, 0] * field[j, l]))
            fk1 = np.float64(fk1 + np.float64(matrix[j, 0] * field[j, l + 1]))
        vtemp[l] = np.float64(pk * fk0)
        vtemp[l + 1] = np.float64(pk * fk1)
        tmp2 = pk
        pk = np.float64(x)
        fk0 = np.float64(0.0)
        fk1 = np.float64(0.0)
        for j in range(4):
            fk0 = np.float64(fk0 + np.float64(matrix[j, 1] * field[j, l]))
            fk1 = np.float64(fk1 + np.float64(matrix[j, 1] * field[j, l + 1]))
        vtemp[l] = np.float64(vtemp[l] + np.float64(pk * fk0))
        vtemp[l + 1] = np.float64(vtemp[l + 1] + np.float64(pk * fk1))
        for k in range(2, 4):
            tmp1 = tmp2
            tmp2 = pk
            pk = np.float64(((2 * k - 1) * x * tmp2 - (k - 1) * tmp1) * (1.0 / k))
            fk0 = np.float64(0.0)
            fk1 = np.float64(0.0)
            for j in range(4):
                fk0 = np.float64(fk0 + np.float64(matrix[j, k] * field[j, l]))
                fk1 = np.float64(fk1 + np.float64(matrix[j, k] * field[j, l + 1]))
            vtemp[l] = np.float64(vtemp[l] + np.float64(pk * fk0))
            vtemp[l + 1] = np.float64(vtemp[l + 1] + np.float64(pk * fk1))
    pk = np.float64(1.0)
    fk0 = np.float64(0.0)
    for j in range(4):
        fk0 = np.float64(fk0 + np.float64(matrix[j, 0] * vtemp[j]))
    value = np.float64(pk * fk0)
    tmp2 = pk
    pk = np.float64(y)
    fk0 = np.float64(0.0)
    for j in range(4):
        fk0 = np.float64(fk0 + np.float64(matrix[j, 1] * vtemp[j]))
    value = np.float64(value + np.float64(pk * fk0))
    for k in range(2, 4):
        tmp1 = tmp2
        tmp2 = pk
        pk = np.float64(((2 * k - 1) * y * tmp2 - (k - 1) * tmp1) * (1.0 / k))
        fk0 = np.float64(0.0)
        for j in range(4):
            fk0 = np.float64(fk0 + np.float64(matrix[j, k] * vtemp[j]))
        value = np.float64(value + np.float64(pk * fk0))
    return value


def _interpolate_legendre_2d_generic(
    field: np.ndarray,
    x: float,
    y: float,
    matrix: np.ndarray,
) -> np.float64:
    """Dimension-independent Legendre interpolation."""

    count = field.shape[0]

    def polynomials(value: float) -> np.ndarray:
        result = np.empty(count, dtype=np.float64)
        result[0] = 1.0
        if count > 1:
            result[1] = value
        for degree in range(2, count):
            result[degree] = (
                (2 * degree - 1) * value * result[degree - 1]
                - (degree - 1) * result[degree - 2]
            ) / degree
        return result

    px = polynomials(x)
    py = polynomials(y)
    coefficients = matrix.T @ field @ matrix
    return np.float64(px @ coefficients @ py)


def initialize_fvm_state(pool, time_level: int = 0) -> None:
    """Port dyn2fvm_mass_vars for the startup GLL state."""
    weights = pool.get("mapping_subcell_integration")
    nlev, ntrac = pool.dimensions["pver"], pool.dimensions["ntrac"]
    np_value = pool.dimensions["np"]
    nc = pool.dimensions["fv_nphys"]
    nhc = (pool.dimensions["fvm_halo"] - nc) // 2
    ones = np.ones((np_value, np_value), dtype=np.float64, order="F")
    for le in range(pool.dimensions["nelem_local"]):
        metdet = pool.get("metric_jacobian")[:, :, le]
        area = _integrate_subcells(ones, metdet, weights)
        inv_area = np.float64(1.0) / area
        ps = pool.get("surface_pressure")[:, :, le, time_level]
        pool.get("fvm_surface_dry_air_pressure")[:, :, le] = _integrate_subcells(ps, metdet, weights) * inv_area
        for lev in range(nlev):
            dp_gll = pool.get("layer_pressure_thickness")[:, :, lev, le, time_level]
            dp_fvm = _integrate_subcells(dp_gll, metdet, weights) * inv_area
            pool.get("fvm_layer_pressure_thickness")[
                nhc : nhc + nc,
                nhc : nhc + nc,
                lev,
                le,
            ] = dp_fvm
            inv_darea_dp = inv_area / dp_fvm
            for advected_slot, physics_index in enumerate(
                pool.advected_constituent_indices
            ):
                q_gll = pool.get("constituent_mixing_ratio")[
                    :, :, lev, le, physics_index, time_level
                ]
                q_fvm = _integrate_subcells(q_gll * dp_gll, metdet, weights) * inv_darea_dp
                minimum = np.min(q_gll)
                maximum = np.max(q_gll)
                for j in range(nc):
                    for i in range(nc):
                        q_fvm[i, j] = max(minimum, min(maximum, q_fvm[i, j]))
                pool.get("fvm_tracer")[
                    nhc : nhc + nc,
                    nhc : nhc + nc,
                    lev,
                    le,
                    advected_slot,
                ] = q_fvm


def _derive_pressure_and_geopotential(pool, backend=None) -> None:
    """Port the FKESSLER subset of SE ``derived_phys_dry``.

    Dynamics supplies dry pressure thickness and dry water mixing ratios. CAM
    physics owns both dry and moist pressure diagnostics and stores water on a
    moist basis outside the individual Kessler calls.
    """

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    ps0 = np.float64(pool.get("reference_pressure"))
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * ps0
    dpdry = pool.get("physics_dry_layer_pressure_thickness")
    dpdry[...] = pool.get("physics_layer_pressure_thickness")
    q = pool.get("physics_constituent_mixing_ratio")
    factor = np.empty((ncol, nlev), dtype=np.float64, order="F")
    for k in range(nlev):
        for i in range(ncol):
            value = np.float64(1.0)
            for constituent in water_constituent_indices(
                pool.constituent_names
            ):
                value = np.float64(value + q[i, k, constituent])
            factor[i, k] = value

    dp = pool.get("physics_layer_pressure_thickness")
    pintdry = pool.get("physics_dry_interface_pressure")
    pint = pool.get("physics_interface_pressure")
    pmiddry = pool.get("physics_dry_midpoint_pressure")
    pmid = pool.get("physics_midpoint_pressure")
    rpdeldry = pool.get("physics_reciprocal_dry_layer_pressure_thickness")
    rpdel = pool.get("physics_reciprocal_layer_pressure_thickness")
    lnpintdry = pool.get("physics_log_dry_interface_pressure")
    lnpint = pool.get("physics_log_interface_pressure")
    lnpmiddry = pool.get("physics_log_dry_midpoint_pressure")
    lnpmid = pool.get("physics_log_midpoint_pressure")
    psdry = pool.get("physics_surface_dry_air_pressure")
    ps = pool.get("physics_surface_pressure")
    for i in range(ncol):
        pintdry[i, 0] = ptop
        pint[i, 0] = ptop
        lnpintdry[i, 0] = np.log(pintdry[i, 0])
        lnpint[i, 0] = np.log(pint[i, 0])
        psdry[i] = ptop
        ps[i] = ptop
    for k in range(nlev):
        for i in range(ncol):
            dp[i, k] = np.float64(dpdry[i, k] * factor[i, k])
            pintdry[i, k + 1] = np.float64(pintdry[i, k] + dpdry[i, k])
            pint[i, k + 1] = np.float64(pint[i, k] + dp[i, k])
            pmiddry[i, k] = np.float64(0.5) * np.float64(pintdry[i, k + 1] + pintdry[i, k])
            pmid[i, k] = np.float64(0.5) * np.float64(pint[i, k + 1] + pint[i, k])
            rpdeldry[i, k] = np.float64(1.0) / dpdry[i, k]
            rpdel[i, k] = np.float64(1.0) / dp[i, k]
            lnpintdry[i, k + 1] = np.log(pintdry[i, k + 1])
            lnpint[i, k + 1] = np.log(pint[i, k + 1])
            lnpmiddry[i, k] = np.log(pmiddry[i, k])
            lnpmid[i, k] = np.log(pmid[i, k])
            psdry[i] = np.float64(psdry[i] + dpdry[i, k])
            ps[i] = np.float64(ps[i] + dp[i, k])

    # qneg_run is deliberately after wet-pressure construction and before the
    # dry-to-wet constituent conversion in derived_phys_dry.  Thus clamping a
    # transported undershoot must not feed back into pdel/pint for this phase.
    minima = pool.get("constituent_minimum")
    for constituent in range(pool.dimensions["nconst"]):
        for k in range(nlev):
            for i in range(ncol):
                if q[i, k, constituent] < minima[constituent]:
                    q[i, k, constituent] = minima[constituent]

    # derived_phys_dry inverts factor_array first and then multiplies each
    # water species.  Direct division changes the final bit.
    for k in range(nlev):
        for i in range(ncol):
            factor[i, k] = np.float64(1.0) / factor[i, k]
    for constituent in range(pool.dimensions["nconst"]):
        for k in range(nlev):
            for i in range(ncol):
                q[i, k, constituent] = np.float64(factor[i, k] * q[i, k, constituent])

    cpair = pool.get("column_dry_air_specific_heat")
    rair = pool.get("column_dry_air_gas_constant")
    inv_exner = pool.get("physics_inverse_surface_exner")
    for k in range(nlev):
        for i in range(ncol):
            inv_exner[i, k] = (ps[i] / pmid[i, k]) ** (rair[i, k] / cpair[i, k])

    _update_geopotential_and_static_energy(
        pool, update_static_energy=True, backend=backend
    )


def _update_geopotential_and_static_energy(
    pool, *, update_static_energy: bool, backend=None
) -> None:
    ncol, nlev, nconst = pool.dimensions["nphys_local"], pool.dimensions["pver"], pool.dimensions["nconst"]
    q = pool.get("physics_constituent_mixing_ratio")
    temp = pool.get("physics_air_temperature")
    pmid = pool.get("physics_midpoint_pressure")
    pdel = pool.get("physics_layer_pressure_thickness")
    rair = pool.get("column_dry_air_gas_constant")
    gravity = np.float64(pool.get("gravitational_acceleration"))
    zvir = np.float64(pool.get("virtual_temperature_coefficient"))
    zi = pool.get("physics_interface_geopotential_height")
    zm = pool.get("thermodynamic_level_height")
    try:
        lagrangian_vertical = bool(
            pool.get_ccpp("do_lagrangian_vertical_coordinate")
        )
    except KeyError:
        # Suites without gravity-wave metadata do not expose this optional
        # CAM control.  The pinned SE/FVM configurations use the Eulerian
        # pressure-coordinate branch in that case.
        lagrangian_vertical = False
    if backend is not None:
        backend.physics_diagnostics(
            update_static_energy=update_static_energy,
            lagrangian_vertical=lagrangian_vertical,
            gravity=gravity,
            virtual_temperature_coefficient=zvir,
            water_vapor_index=water_vapor_index(pool.constituent_names),
            water_species=water_species_flags(pool.constituent_names),
            temperature=temp,
            constituent=q,
            pressure_thickness=pdel,
            midpoint_pressure=pmid,
            reciprocal_pressure_thickness=pool.get(
                "physics_reciprocal_layer_pressure_thickness"
            ),
            interface_pressure=pool.get("physics_interface_pressure"),
            log_interface_pressure=pool.get(
                "physics_log_interface_pressure"
            ),
            gas_constant=rair,
            heat_capacity=pool.get("column_dry_air_specific_heat"),
            surface_geopotential=pool.get("physics_surface_geopotential"),
            interface_height=zi,
            midpoint_height=zm,
            static_energy=pool.get("static_energy"),
        )
        pool.mark_initialized("physics_interface_geopotential_height")
        pool.mark_initialized("thermodynamic_level_height")
        if update_static_energy:
            pool.mark_initialized("static_energy")
        return
    for i in range(ncol):
        zi[i, nlev] = np.float64(0.0)
    rpdel = pool.get("physics_reciprocal_layer_pressure_thickness")
    pint = pool.get("physics_interface_pressure")
    lnpint = pool.get("physics_log_interface_pressure")
    # Match geopotential_temp_run's bottom-up hydrostatic elements for either
    # the Lagrangian or Eulerian vertical coordinate.
    for k in range(nlev - 1, -1, -1):
        for i in range(ncol):
            if lagrangian_vertical:
                hkl = np.float64(lnpint[i, k + 1] - lnpint[i, k])
                hkk = np.float64(1.0) - np.float64(
                    pint[i, k] * hkl * rpdel[i, k]
                )
            else:
                hkl = np.float64(pdel[i, k] / pmid[i, k])
                hkk = np.float64(0.5) * hkl
            qfac_denom = np.float64(1.0)
            for constituent in water_constituent_indices(
                pool.constituent_names
            ):
                qfac_denom = np.float64(qfac_denom - q[i, k, constituent])
            qfac = np.float64(1.0) / qfac_denom
            sum_dry = np.float64(1.0)
            for constituent in water_constituent_indices(
                pool.constituent_names
            ):
                sum_dry = np.float64(sum_dry + np.float64(q[i, k, constituent] * qfac))
            sum_dry = np.float64(1.0) / sum_dry
            vapor = next(
                (
                    q[i, k, index]
                    for index, name in enumerate(pool.constituent_names)
                    if is_water_vapor(name)
                ),
                np.float64(0.0),
            )
            tvfac = np.float64(
                1.0
                + np.float64(
                    (zvir + np.float64(1.0)) * vapor * qfac
                )
            ) * sum_dry
            tv = np.float64(temp[i, k] * tvfac)
            rog = np.float64(rair[i, k] / gravity)
            zm[i, k] = np.float64(zi[i, k + 1] + np.float64(rog * tv * hkk))
            zi[i, k] = np.float64(zi[i, k + 1] + np.float64(rog * tv * hkl))
    if update_static_energy:
        phis = pool.get("physics_surface_geopotential")
        dse = pool.get("static_energy")
        cpair = pool.get("column_dry_air_specific_heat")
        for k in range(nlev):
            for i in range(ncol):
                dse[i, k] = np.float64(temp[i, k] * cpair[i, k]) + np.float64(gravity * zm[i, k]) + phis[i]
    pool.mark_initialized("physics_interface_geopotential_height")
    pool.mark_initialized("thermodynamic_level_height")
    if update_static_energy:
        pool.mark_initialized("static_energy")


def calc_exner(pool) -> None:
    """Implement the ``calc_exner`` CCPP scheme."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    pmid = pool.get("physics_midpoint_pressure")
    cpair = pool.get("column_dry_air_specific_heat")
    rair = pool.get("column_dry_air_gas_constant")
    ps0 = np.float64(pool.get("reference_pressure"))
    exner = pool.get("exner_function")
    for k in range(nlev):
        for i in range(ncol):
            exner[i, k] = (pmid[i, k] / ps0) ** (rair[i, k] / cpair[i, k])


def temp_to_potential_temp(pool) -> None:
    """Implement the ``temp_to_potential_temp`` CCPP scheme."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    exner = pool.get("exner_function")
    theta = pool.get("potential_temperature")
    temp = pool.get("physics_air_temperature")
    for k in range(nlev):
        for i in range(ncol):
            theta[i, k] = np.float64(temp[i, k] / exner[i, k])


def calc_dry_air_ideal_gas_density(pool) -> None:
    """Implement the ``calc_dry_air_ideal_gas_density`` CCPP scheme."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    pmiddry = pool.get("physics_dry_midpoint_pressure")
    rair = pool.get("column_dry_air_gas_constant")
    rho = pool.get("dry_air_density")
    temp = pool.get("physics_air_temperature")
    for k in range(nlev):
        for i in range(ncol):
            rho[i, k] = np.float64(pmiddry[i, k] / np.float64(rair[i, k] * temp[i, k]))


def _wet_to_dry(pool, constituent: int) -> None:
    ratio = pool.get("physics_layer_pressure_thickness") / pool.get("physics_dry_layer_pressure_thickness")
    q = pool.get("physics_constituent_mixing_ratio")
    for k in range(pool.dimensions["pver"]):
        for i in range(pool.dimensions["nphys_local"]):
            q[i, k, constituent] = np.float64(q[i, k, constituent] * ratio[i, k])


def wet_to_dry_water_vapor(pool) -> None:
    _wet_to_dry(pool, 2)


def wet_to_dry_cloud_liquid_water(pool) -> None:
    _wet_to_dry(pool, 0)


def wet_to_dry_rain(pool) -> None:
    _wet_to_dry(pool, 1)


def _dry_to_wet(pool, constituent: int) -> None:
    ratio = pool.get("physics_dry_layer_pressure_thickness") / pool.get("physics_layer_pressure_thickness")
    q = pool.get("physics_constituent_mixing_ratio")
    for k in range(pool.dimensions["pver"]):
        for i in range(pool.dimensions["nphys_local"]):
            q[i, k, constituent] = np.float64(q[i, k, constituent] * ratio[i, k])


def dry_to_wet_water_vapor(pool) -> None:
    _dry_to_wet(pool, 2)


def dry_to_wet_cloud_liquid_water(pool) -> None:
    _dry_to_wet(pool, 0)


def dry_to_wet_rain(pool) -> None:
    _dry_to_wet(pool, 1)


def qneg(pool) -> None:
    """Apply the registered constituent lower bounds."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    q = pool.get("physics_constituent_mixing_ratio")
    minima = pool.get("constituent_minimum")
    for constituent in range(pool.dimensions["nconst"]):
        for k in range(nlev):
            for i in range(ncol):
                q[i, k, constituent] = max(q[i, k, constituent], minima[constituent])


def geopotential_temp(pool, backend=None) -> None:
    """Refresh the geopotential diagnostics after Kessler."""

    # suite_kessler.xml runs geopotential_temp after Kessler, but it does not
    # run update_dry_static_energy in the before-coupler group.  History at
    # nstep=0 therefore retains the pre-Kessler DSE while ZM/ZI are refreshed.
    _update_geopotential_and_static_energy(
        pool, update_static_energy=False, backend=backend
    )


def _populate_gravity_wave_vorticity(
    pool, comm, *, time_level: int
) -> None:
    """Port SE ``gws_src_vort`` into the Python-owned dycore coupling."""

    try:
        target_name = pool.ccpp_field_name("relative_vorticity")
    except KeyError:
        return

    from .dynamics import _edge_sum, _vorticity_sphere

    np_value = pool.dimensions["np"]
    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    raw = np.empty(
        (np_value, np_value, nlev, nelem),
        dtype=np.float64,
        order="F",
    )
    derivative = pool.get("gll_derivative")
    inverse_radius = np.float64(1.0) / np.float64(
        pool.get("earth_radius")
    )
    for le in range(nelem):
        metric = pool.get("metric_derivative")[:, :, :, :, le]
        inverse_jacobian = pool.get("inverse_metric_jacobian")[:, :, le]
        for level in range(nlev):
            velocity = np.empty(
                (np_value, np_value, 2),
                dtype=np.float64,
                order="F",
            )
            velocity[:, :, 0] = pool.get("zonal_wind")[
                :, :, level, le, time_level
            ]
            velocity[:, :, 1] = pool.get("meridional_wind")[
                :, :, level, le, time_level
            ]
            raw[:, :, level, le] = _vorticity_sphere(
                velocity,
                derivative,
                metric,
                inverse_jacobian,
                inverse_radius,
            )

    mass_weighted = raw * pool.get("spectral_mass_matrix")[
        :, :, np.newaxis, :
    ]
    if hasattr(comm, "Allreduce"):
        assembled = _edge_sum(pool, comm, mass_weighted)
    elif np.count_nonzero(mass_weighted) == 0:
        # Metadata-only unit communicators do not implement edge exchange.
        assembled = mass_weighted
    else:
        raise RuntimeError(
            "gravity-wave vorticity requires an MPI communicator with "
            "rank-edge exchange"
        )
    continuous = assembled * pool.get("inverse_spectral_mass_matrix")[
        :, :, np.newaxis, :
    ]

    integration = pool.get("mapping_subcell_integration")
    output = pool.get(target_name, unsafe=True)
    columns_per_element = pool.dimensions["fv_nphys"] ** 2
    ones = np.ones(
        (np_value, np_value), dtype=np.float64, order="F"
    )
    for le in range(nelem):
        columns = slice(
            le * columns_per_element,
            (le + 1) * columns_per_element,
        )
        metric_jacobian = pool.get("metric_jacobian")[:, :, le]
        inverse_area = np.float64(1.0) / _integrate_subcells(
            ones, metric_jacobian, integration
        )
        for level in range(nlev):
            mapped = (
                _integrate_subcells(
                    continuous[:, :, level, le],
                    metric_jacobian,
                    integration,
                )
                * inverse_area
            )
            output[columns, level] = mapped.reshape(
                columns_per_element, order="F"
            )
    pool.set(target_name, output)


def _vertical_pressure_derivative(
    interface_pressure: np.ndarray,
    midpoint_pressure: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Port ``gravity_waves_sources::compute_vertical_derivative``."""

    levels = midpoint_pressure.shape[2]
    derivative = np.empty_like(values, order="F")
    for level in range(levels):
        if level == 0:
            pressure_above = midpoint_pressure[:, :, level]
            pressure_below = interface_pressure[:, :, level + 1]
            value_above = values[:, :, level]
            value_below = np.float64(0.5) * (
                values[:, :, level + 1] + values[:, :, level]
            )
        elif level == levels - 1:
            pressure_above = interface_pressure[:, :, level]
            pressure_below = midpoint_pressure[:, :, level]
            value_above = np.float64(0.5) * (
                values[:, :, level - 1] + values[:, :, level]
            )
            value_below = values[:, :, level]
        else:
            pressure_above = interface_pressure[:, :, level]
            pressure_below = interface_pressure[:, :, level + 1]
            value_above = np.float64(0.5) * (
                values[:, :, level - 1] + values[:, :, level]
            )
            value_below = np.float64(0.5) * (
                values[:, :, level + 1] + values[:, :, level]
            )
        derivative[:, :, level] = (
            value_above - value_below
        ) / (pressure_above - pressure_below)
    return derivative


def _sphere_to_cartesian_vectors(
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    """Return HOMME ``vec_sphere2cart(np,np,3,2)`` in NumPy order."""

    result = np.empty(
        (*longitude.shape, 3, 2), dtype=np.float64, order="F"
    )
    result[:, :, 0, 0] = -np.sin(longitude)
    result[:, :, 1, 0] = np.cos(longitude)
    result[:, :, 2, 0] = 0.0
    result[:, :, 0, 1] = -np.sin(latitude) * np.cos(longitude)
    result[:, :, 1, 1] = -np.sin(latitude) * np.sin(longitude)
    result[:, :, 2, 1] = np.cos(latitude)
    return result


def _populate_gravity_wave_frontogenesis(
    pool, comm, *, time_level: int
) -> None:
    """Port SE ``gws_src_fnct`` into the Python-owned dycore coupling."""

    try:
        function_name = pool.ccpp_field_name("frontogenesis_function")
    except KeyError:
        return
    try:
        angle_name = pool.ccpp_field_name("frontogenesis_angle")
    except KeyError:
        angle_name = None

    from .dynamics import _edge_sum, _gradient_sphere

    np_value = pool.dimensions["np"]
    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    derivative_matrix = pool.get("gll_derivative")
    inverse_radius = np.float64(1.0) / np.float64(
        pool.get("earth_radius")
    )
    reference_pressure = np.float64(pool.get("reference_pressure"))
    top_pressure = np.float64(
        pool.get("hybrid_a_interface")[0] * reference_pressure
    )
    dry_kappa = np.float64(pool.get("dry_air_kappa"))
    water_indices = tuple(
        index
        for index, name in enumerate(pool.constituent_names)
        if name == "water_vapor"
    )
    function_gll = np.empty(
        (np_value, np_value, nlev, nelem),
        dtype=np.float64,
        order="F",
    )
    gradient_theta = np.empty(
        (np_value, np_value, 2, nlev, nelem),
        dtype=np.float64,
        order="F",
    )

    for le in range(nelem):
        inverse_metric = pool.get("inverse_metric")[:, :, :, :, le]
        longitude = pool.get("gll_longitude")[:, :, le]
        latitude = pool.get("gll_latitude")[:, :, le]
        sphere_to_cartesian = _sphere_to_cartesian_vectors(
            longitude, latitude
        )
        pressure_interface = np.empty(
            (np_value, np_value, nlev + 1),
            dtype=np.float64,
            order="F",
        )
        pressure_midpoint = np.empty(
            (np_value, np_value, nlev),
            dtype=np.float64,
            order="F",
        )
        potential_temperature = np.empty_like(
            pressure_midpoint, order="F"
        )
        pressure_interface[:, :, 0] = top_pressure
        for level in range(nlev):
            water_factor = np.ones(
                (np_value, np_value),
                dtype=np.float64,
                order="F",
            )
            for constituent in water_indices:
                water_factor = water_factor + pool.get(
                    "constituent_mixing_ratio"
                )[:, :, level, le, constituent, time_level]
            dry_pressure_thickness = pool.get(
                "layer_pressure_thickness"
            )[:, :, level, le, time_level]
            pressure_midpoint[:, :, level] = (
                pressure_interface[:, :, level]
                + np.float64(0.5)
                * water_factor
                * dry_pressure_thickness
            )
            pressure_interface[:, :, level + 1] = (
                pressure_interface[:, :, level]
                + dry_pressure_thickness
            )
            potential_temperature[:, :, level] = pool.get(
                "air_temperature"
            )[:, :, level, le, time_level] * (
                reference_pressure / pressure_midpoint[:, :, level]
            ) ** dry_kappa

        theta_pressure_derivative = _vertical_pressure_derivative(
            pressure_interface,
            pressure_midpoint,
            potential_temperature,
        )
        for level in range(nlev):
            theta_gradient = _gradient_sphere(
                potential_temperature[:, :, level],
                derivative_matrix,
                inverse_metric,
                inverse_radius,
            )
            pressure_gradient = _gradient_sphere(
                pressure_midpoint[:, :, level],
                derivative_matrix,
                inverse_metric,
                inverse_radius,
            )
            for component in range(2):
                gradient_theta[:, :, component, level, le] = (
                    theta_gradient[:, :, component]
                    - theta_pressure_derivative[:, :, level]
                    * pressure_gradient[:, :, component]
                )

        cartesian_wind = np.empty(
            (np_value, np_value, 3, nlev),
            dtype=np.float64,
            order="F",
        )
        for level in range(nlev):
            zonal = pool.get("zonal_wind")[
                :, :, level, le, time_level
            ]
            meridional = pool.get("meridional_wind")[
                :, :, level, le, time_level
            ]
            for component in range(3):
                cartesian_wind[:, :, component, level] = (
                    sphere_to_cartesian[:, :, component, 0] * zonal
                    + sphere_to_cartesian[:, :, component, 1]
                    * meridional
                )
        cartesian_wind_derivative = np.empty_like(
            cartesian_wind, order="F"
        )
        for component in range(3):
            cartesian_wind_derivative[:, :, component, :] = (
                _vertical_pressure_derivative(
                    pressure_interface,
                    pressure_midpoint,
                    cartesian_wind[:, :, component, :],
                )
            )

        for level in range(nlev):
            pressure_gradient = _gradient_sphere(
                pressure_midpoint[:, :, level],
                derivative_matrix,
                inverse_metric,
                inverse_radius,
            )
            contracted_cartesian = np.empty(
                (np_value, np_value, 3),
                dtype=np.float64,
                order="F",
            )
            for cartesian_component in range(3):
                wind_gradient = _gradient_sphere(
                    cartesian_wind[
                        :, :, cartesian_component, level
                    ],
                    derivative_matrix,
                    inverse_metric,
                    inverse_radius,
                )
                for spherical_component in range(2):
                    wind_gradient[:, :, spherical_component] = (
                        wind_gradient[:, :, spherical_component]
                        - cartesian_wind_derivative[
                            :, :, cartesian_component, level
                        ]
                        * pressure_gradient[:, :, spherical_component]
                    )
                contracted_cartesian[:, :, cartesian_component] = (
                    gradient_theta[:, :, 0, level, le]
                    * wind_gradient[:, :, 0]
                    + gradient_theta[:, :, 1, level, le]
                    * wind_gradient[:, :, 1]
                )

            contracted_spherical = np.empty(
                (np_value, np_value, 2),
                dtype=np.float64,
                order="F",
            )
            for spherical_component in range(2):
                contracted_spherical[:, :, spherical_component] = (
                    contracted_cartesian[:, :, 0]
                    * sphere_to_cartesian[
                        :, :, 0, spherical_component
                    ]
                    + contracted_cartesian[:, :, 1]
                    * sphere_to_cartesian[
                        :, :, 1, spherical_component
                    ]
                    + contracted_cartesian[:, :, 2]
                    * sphere_to_cartesian[
                        :, :, 2, spherical_component
                    ]
                )
            function_gll[:, :, level, le] = -(
                contracted_spherical[:, :, 0]
                * gradient_theta[:, :, 0, level, le]
                + contracted_spherical[:, :, 1]
                * gradient_theta[:, :, 1, level, le]
            )

        mass = pool.get("spectral_mass_matrix")[:, :, le]
        function_gll[:, :, :, le] *= mass[:, :, np.newaxis]
        gradient_theta[:, :, :, :, le] *= mass[
            :, :, np.newaxis, np.newaxis
        ]

    if hasattr(comm, "Allreduce"):
        function_gll = _edge_sum(pool, comm, function_gll)
        packed_gradient = gradient_theta.reshape(
            np_value, np_value, 2 * nlev, nelem, order="F"
        )
        packed_gradient = _edge_sum(pool, comm, packed_gradient)
        gradient_theta = packed_gradient.reshape(
            np_value, np_value, 2, nlev, nelem, order="F"
        )
    inverse_mass = pool.get("inverse_spectral_mass_matrix")
    function_gll *= inverse_mass[:, :, np.newaxis, :]
    gradient_theta *= inverse_mass[
        :, :, np.newaxis, np.newaxis, :
    ]

    function_output = pool.get(function_name, unsafe=True)
    angle_output = (
        pool.get(angle_name, unsafe=True)
        if angle_name is not None
        else None
    )
    integration = pool.get("mapping_subcell_integration")
    interpolation = pool.get("mapping_interpolation_matrix")
    physics_nodes = _physics_reference_nodes(
        pool.dimensions["fv_nphys"]
    )
    columns_per_element = pool.dimensions["fv_nphys"] ** 2
    ones = np.ones(
        (np_value, np_value), dtype=np.float64, order="F"
    )
    for le in range(nelem):
        columns = slice(
            le * columns_per_element,
            (le + 1) * columns_per_element,
        )
        metric_jacobian = pool.get("metric_jacobian")[:, :, le]
        inverse_area = np.float64(1.0) / _integrate_subcells(
            ones, metric_jacobian, integration
        )
        inverse_metric = pool.get("inverse_metric")[:, :, :, :, le]
        physics_derivative = pool.get("mapping_derivative_pg3")[
            :, :, columns
        ]
        for level in range(nlev):
            mapped_function = (
                _integrate_subcells(
                    function_gll[:, :, level, le],
                    metric_jacobian,
                    integration,
                )
                * inverse_area
            )
            function_output[columns, level] = mapped_function.reshape(
                columns_per_element, order="F"
            )
            if angle_output is None:
                continue
            for physics_j, y in enumerate(physics_nodes):
                for physics_i, x in enumerate(physics_nodes):
                    local_column = (
                        physics_i
                        + pool.dimensions["fv_nphys"] * physics_j
                    )
                    contravariant_1 = np.empty(
                        (np_value, np_value),
                        dtype=np.float64,
                        order="F",
                    )
                    contravariant_2 = np.empty_like(
                        contravariant_1, order="F"
                    )
                    for j in range(np_value):
                        for i in range(np_value):
                            eastward = gradient_theta[
                                i, j, 0, level, le
                            ]
                            northward = gradient_theta[
                                i, j, 1, level, le
                            ]
                            contravariant_1[i, j] = np.float64(
                                inverse_metric[0, 0, i, j] * eastward
                            ) + np.float64(
                                inverse_metric[0, 1, i, j] * northward
                            )
                            contravariant_2[i, j] = np.float64(
                                inverse_metric[1, 0, i, j] * eastward
                            ) + np.float64(
                                inverse_metric[1, 1, i, j] * northward
                            )
                    first = _interpolate_legendre_2d(
                        contravariant_1, x, y, interpolation
                    )
                    second = _interpolate_legendre_2d(
                        contravariant_2, x, y, interpolation
                    )
                    eastward = np.float64(
                        physics_derivative[0, 0, local_column] * first
                    ) + np.float64(
                        physics_derivative[0, 1, local_column] * second
                    )
                    northward = np.float64(
                        physics_derivative[1, 0, local_column] * first
                    ) + np.float64(
                        physics_derivative[1, 1, local_column] * second
                    )
                    angle_output[
                        columns.start + local_column, level
                    ] = np.arctan2(
                        northward,
                        eastward + np.float64(1.0e-10),
                    )
    pool.set(function_name, function_output)
    if angle_name is not None:
        pool.set(angle_name, angle_output)


def dynamics_to_physics(
    pool,
    time_level: int | None = None,
    comm=None,
    backend=None,
    *,
    canonicalize_resting_wind_zero: bool = True,
) -> None:
    if time_level is None:
        time_level = int(pool.get("dynamics_time_level_n0"))
    if comm is not None:
        _populate_gravity_wave_vorticity(
            pool, comm, time_level=time_level
        )
        _populate_gravity_wave_frontogenesis(
            pool, comm, time_level=time_level
        )
    w = pool.get("mapping_weights_gll_to_pg3")
    integration = pool.get("mapping_subcell_integration")
    interpolation_matrix = pool.get("mapping_interpolation_matrix")
    nlev, ntrac = pool.dimensions["pver"], pool.dimensions["ntrac"]
    np_value = pool.dimensions["np"]
    nc = pool.dimensions["fv_nphys"]
    nhc = (pool.dimensions["fvm_halo"] - nc) // 2
    columns_per_element = nc * nc
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(pool.get("reference_pressure"))
    physics_nodes = _physics_reference_nodes(nc)
    ones = np.ones((np_value, np_value), dtype=np.float64, order="F")
    if backend is not None:
        nelem = pool.dimensions["nelem_local"]
        output_shape = (columns_per_element, nelem, nlev)
        backend.dynamics_to_physics_thermo_vector(
            temperature=pool.get("air_temperature")[
                :, :, :, :, time_level
            ],
            pressure_thickness=pool.get("layer_pressure_thickness")[
                :, :, :, :, time_level
            ],
            zonal_wind=pool.get("zonal_wind")[:, :, :, :, time_level],
            meridional_wind=pool.get("meridional_wind")[
                :, :, :, :, time_level
            ],
            metric_jacobian=pool.get("metric_jacobian"),
            inverse_metric=pool.get("inverse_metric"),
            integration=integration,
            interpolation=interpolation_matrix,
            physics_nodes=np.asfortranarray(physics_nodes),
            physics_derivative=pool.get("mapping_derivative_pg3").reshape(
                (2, 2, columns_per_element, nelem),
                order="F",
            ),
            temperature_physics=pool.get(
                "physics_air_temperature"
            ).reshape(output_shape, order="F"),
            zonal_physics=pool.get("physics_zonal_wind").reshape(
                output_shape, order="F"
            ),
            meridional_physics=pool.get(
                "physics_meridional_wind"
            ).reshape(output_shape, order="F"),
            pressure_physics=pool.get(
                "physics_layer_pressure_thickness"
            ).reshape(output_shape, order="F"),
        )
        if int(pool.get("model_step")) == 0:
            native_pressure = pool.get(
                "physics_layer_pressure_thickness"
            )
            for le in range(nelem):
                columns = slice(
                    le * columns_per_element,
                    (le + 1) * columns_per_element,
                )
                pool.get("fvm_layer_pressure_thickness")[
                    nhc : nhc + nc,
                    nhc : nhc + nc,
                    :,
                    le,
                ] = native_pressure[columns].reshape(
                    (nc, nc, nlev), order="F"
                )
        if (
            canonicalize_resting_wind_zero
            and not np.any(
                pool.get("zonal_wind")[:, :, :, :, time_level]
            )
            and not np.any(
                pool.get("meridional_wind")[:, :, :, :, time_level]
            )
        ):
            # CAM's resting analytic initial conditions expose canonical
            # positive zero on the physics grid.  Metric multiplication can
            # otherwise manufacture negative zero without changing the
            # numerical value.
            pool.get("physics_zonal_wind")[...] = np.float64(0.0)
            pool.get("physics_meridional_wind")[...] = np.float64(0.0)
    for le in range(pool.dimensions["nelem_local"]):
        columns = slice(
            le * columns_per_element,
            (le + 1) * columns_per_element,
        )
        metdet = pool.get("metric_jacobian")[:, :, le]
        area = _integrate_subcells(ones, metdet, integration)
        inv_area = np.float64(1.0) / area
        psdry = np.full((nc, nc), ptop, dtype=np.float64, order="F")
        dinv = pool.get("inverse_metric")[:, :, :, :, le]
        dphys = pool.get("mapping_derivative_pg3")[:, :, columns]
        if backend is None:
            for lev in range(nlev):
                u = pool.get("zonal_wind")[:, :, lev, le, time_level]
                v = pool.get("meridional_wind")[:, :, lev, le, time_level]
                contra1 = np.empty(
                    (np_value, np_value),
                    dtype=np.float64,
                    order="F",
                )
                contra2 = np.empty_like(contra1, order="F")
                for j in range(np_value):
                    for i in range(np_value):
                        contra1[i, j] = np.float64(dinv[0, 0, i, j] * u[i, j]) + np.float64(dinv[0, 1, i, j] * v[i, j])
                        contra2[i, j] = np.float64(dinv[1, 0, i, j] * u[i, j]) + np.float64(dinv[1, 1, i, j] * v[i, j])
                for pj in range(nc):
                    for pi in range(nc):
                        local_col = pi + nc * pj
                        x = physics_nodes[pi]
                        y = physics_nodes[pj]
                        v1 = _interpolate_legendre_2d(contra1, x, y, interpolation_matrix)
                        v2 = _interpolate_legendre_2d(contra2, x, y, interpolation_matrix)
                        target_column = le * columns_per_element + local_col
                        pool.get("physics_zonal_wind")[target_column, lev] = np.float64(dphys[0, 0, local_col] * v1) + np.float64(dphys[0, 1, local_col] * v2)
                        pool.get("physics_meridional_wind")[target_column, lev] = np.float64(dphys[1, 0, local_col] * v1) + np.float64(dphys[1, 1, local_col] * v2)
        for lev in range(nlev):
            dp_fvm = pool.get("fvm_layer_pressure_thickness")[
                nhc : nhc + nc,
                nhc : nhc + nc,
                lev,
                le,
            ]
            psdry = psdry + dp_fvm
            pool.get("physics_layer_pressure_thickness")[columns, lev] = dp_fvm.reshape(columns_per_element, order="F")
            if backend is None:
                t_gll = pool.get("air_temperature")[
                    :, :, lev, le, time_level
                ]
                dp_gll = pool.get("layer_pressure_thickness")[
                    :, :, lev, le, time_level
                ]
                # dyn2phys_all_vars derives the temperature denominator from
                # an independent GLL dry-mass integration.
                dp3d_tmp = (
                    _integrate_subcells(dp_gll, metdet, integration)
                    * inv_area
                )
                inv_darea_dp_phys = inv_area / dp3d_tmp
                t_phys = _integrate_subcells(
                    t_gll * dp_gll, metdet, integration
                ) * inv_darea_dp_phys
                pool.get("physics_air_temperature")[
                    columns, lev
                ] = t_phys.reshape(columns_per_element, order="F")
            omega = _integrate_subcells(pool.get("vertical_pressure_velocity")[:, :, lev, le], metdet, integration) * inv_area
            pool.get("physics_vertical_pressure_velocity")[columns, lev] = omega.reshape(columns_per_element, order="F")
            for advected_slot, physics_index in enumerate(
                pool.advected_constituent_indices
            ):
                q_fvm = pool.get("fvm_tracer")[
                    nhc : nhc + nc,
                    nhc : nhc + nc,
                    lev,
                    le,
                    advected_slot,
                ]
                pool.get("physics_constituent_mixing_ratio")[
                    columns, lev, physics_index
                ] = q_fvm.reshape(columns_per_element, order="F")
        # d_p_coupling writes dry surface pressure.  Moist PS is formed later
        # by derived_phys_dry in physics_timestep_initial.
        pool.get("physics_surface_dry_air_pressure")[columns] = psdry.reshape(columns_per_element, order="F")
        pool.get("physics_surface_geopotential")[columns] = pool.get(
            "fvm_surface_geopotential"
        )[:, :, le].reshape(columns_per_element, order="F")
    # d_p_coupling saves the dry tracer state before derived_phys_dry converts
    # water to the moist physics convention.
    pool.get("physics_constituent_previous")[...] = pool.get(
        "physics_constituent_mixing_ratio"
    )
    thermo_water_update(pool, constituents_are_dry=True)


def physics_timestep_initial(pool, backend=None) -> None:
    debug = bool(os.environ.get("PYCAM_DEBUG_PROCESS_TRACE"))
    rank = int(pool.get("mpi_rank", unsafe=True)) if debug else -1
    if debug:
        print(f"PYCAM_TIMESTEP_INITIAL rank={rank} derive_begin", flush=True)
    _derive_pressure_and_geopotential(pool, backend=backend)
    if debug:
        print(f"PYCAM_TIMESTEP_INITIAL rank={rank} derive_done", flush=True)
    # ``d_p_coupling`` updates the SE energy heat capacity from the mapped
    # water state before CCPP sees the first physics column.  Recompute here
    # after qneg/dry-to-moist conversion so the initial full-physics path also
    # observes the constituent floor rather than retaining dry-air cp.
    thermo_water_update(pool)
    if debug:
        print(f"PYCAM_TIMESTEP_INITIAL rank={rank} thermo_done", flush=True)
    if "air_temperature_previous_timestep" in pool.contracts:
        pool.set(
            "air_temperature_previous_timestep",
            pool.get("physics_air_temperature"),
        )
    for name in ("physics_air_temperature_tendency", "physics_zonal_wind_tendency", "physics_meridional_wind_tendency", "physics_constituent_tendency"):
        pool.get(name)[...] = 0.0
    if debug:
        print(f"PYCAM_TIMESTEP_INITIAL rank={rank} zero_done", flush=True)


def physics_timestep_final(pool) -> None:
    pool.get("physics_air_temperature")[:] += float(pool.get("model_timestep")) * pool.get("physics_air_temperature_tendency")
    if "static_energy" in pool.contracts:
        pool.set(
            "static_energy",
            pool.get("physics_air_temperature")
            * pool.get("column_dry_air_specific_heat")
            + float(pool.get("gravitational_acceleration"))
            * pool.get("thermodynamic_level_height"),
        )


def thermo_water_update(
    pool, *, constituents_are_dry: bool = False
) -> None:
    """Update the SE heat capacity for the current water composition."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    q = pool.get("physics_constituent_mixing_ratio")
    factor = (
        np.float64(1.0)
        if constituents_are_dry
        else pool.get("physics_layer_pressure_thickness")
        / pool.get("physics_dry_layer_pressure_thickness")
    )
    cpair = pool.get("column_dry_air_specific_heat")
    cpdy = pool.get("dycore_heat_capacity")
    liquid_cp = np.float64(pool.get("liquid_water_specific_heat"))
    vapor_cp = np.float64(pool.get("water_vapor_specific_heat"))
    species_cp = tuple(
        (
            index,
            vapor_cp if is_water_vapor(name) else liquid_cp,
        )
        for index, name in enumerate(pool.constituent_names)
        if is_water_constituent(name)
    )
    for k in range(nlev):
        for i in range(ncol):
            sum_species = np.float64(1.0)
            sum_cp = cpair[i, k]
            for constituent, heat_capacity in species_cp:
                conversion = (
                    factor if np.isscalar(factor) else factor[i, k]
                )
                dry = np.float64(q[i, k, constituent] * conversion)
                sum_species = np.float64(sum_species + dry)
                term = np.float64(
                    np.float64(heat_capacity * q[i, k, constituent])
                    * conversion
                )
                sum_cp = np.float64(sum_cp + term)
            cpdy[i, k] = np.float64(sum_cp / sum_species)


def check_energy_scaling(pool) -> None:
    """Compute the temperature-increment scaling used by the SE dycore."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    cpair = pool.get("column_dry_air_specific_heat")
    cpdy = pool.get("dycore_heat_capacity")
    scaling = pool.get("dycore_energy_scaling")
    for k in range(nlev):
        for i in range(ncol):
            scaling[i, k] = np.float64(cpair[i, k] / cpdy[i, k])


def dycore_energy_consistency_adjust(pool) -> None:
    """Form the local temperature tendency needed for SE energy consistency."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    scaling = pool.get("dycore_energy_scaling")
    local_tendency = pool.get("temperature_consistency_tendency")
    total_tendency = pool.get("physics_air_temperature_tendency")
    for k in range(nlev):
        for i in range(ncol):
            local_tendency[i, k] = np.float64(
                np.float64(scaling[i, k] - np.float64(1.0))
                * total_tendency[i, k]
            )


def apply_tendency_of_air_temperature(pool) -> None:
    """Apply and accumulate the scheme-local temperature tendency."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    local_tendency = pool.get("temperature_consistency_tendency")
    total_tendency = pool.get("physics_air_temperature_tendency")
    temperature = pool.get("physics_air_temperature")
    dt = pool.get("model_timestep")
    for k in range(nlev):
        for i in range(ncol):
            temperature[i, k] = np.float64(
                temperature[i, k]
                + np.float64(local_tendency[i, k] * dt)
            )
            total_tendency[i, k] = np.float64(
                total_tendency[i, k] + local_tendency[i, k]
            )
    local_tendency[...] = 0.0


def check_energy_zero_fluxes(pool) -> None:
    """Represent the zero-flux energy-check scheme for this closed ATM case.

    The current ATM-only capability has no surface/coupler energy fluxes and does not retain
    CAM's message-only conservation bookkeeping in model state.  The scheme is
    still an explicit control boundary so it can be enabled, disabled, moved,
    and observed independently.
    """


def check_energy_scaling_before_coupler(pool) -> None:
    """Represent the pre-coupler energy-check boundary for this capability.

    This occurrence of ``check_energy_scaling`` in the CCPP suite performs
    conservation bookkeeping.  It is distinct from the after-coupler scheme
    with the same name, which calculates the SE temperature scaling array.
    """


def check_energy_chng(pool, backend=None) -> None:
    """Update CAM's column energy bookkeeping after one suite process."""

    values = _hydrostatic_energy(pool, backend=backend)
    energy_physics, energy_dycore, total_water = values
    current_physics = pool.get_ccpp(
        "vertically_integrated_total_energy_using_physics_energy_formula"
    )
    current_dycore = pool.get_ccpp(
        "vertically_integrated_total_energy_using_dycore_energy_formula"
    )
    current_water = pool.get_ccpp("vertically_integrated_total_water")
    energy_at_end = pool.get_ccpp(
        "vertically_integrated_total_energy_using_dycore_energy_formula_at_"
        "end_of_physics_timestep"
    )
    energy_at_start = pool.get_ccpp(
        "vertically_integrated_total_energy_using_dycore_energy_formula_at_"
        "start_of_physics_timestep"
    )
    energy_flux = pool.get_ccpp(
        "cumulative_total_energy_boundary_flux_using_physics_energy_formula"
    )
    water_flux = pool.get_ccpp("cumulative_total_water_boundary_flux")
    vapor_flux = pool.get_ccpp(
        "net_water_vapor_fluxes_through_top_and_bottom_of_atmosphere_column"
    )
    condensate_flux = pool.get_ccpp(
        "net_liquid_and_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column"
    )
    ice_flux = pool.get_ccpp(
        "net_lwe_ice_fluxes_through_top_and_bottom_of_atmosphere_column"
    )
    sensible_flux = pool.get_ccpp(
        "net_sensible_heat_flux_through_top_and_bottom_of_atmosphere_column"
    )
    latent_vapor = pool.get("latent_heat_of_vaporization").item()
    latent_ice = pool.get("latent_heat_of_fusion").item()
    for column in range(pool.dimensions["nphys_local"]):
        energy_tendency = np.float64(
            np.float64(vapor_flux[column] * (latent_vapor + latent_ice))
            - np.float64(
                np.float64(condensate_flux[column] - ice_flux[column])
                * np.float64(1000.0)
                * latent_ice
            )
            + sensible_flux[column]
        )
        water_tendency = np.float64(
            vapor_flux[column]
            - np.float64(condensate_flux[column] * np.float64(1000.0))
        )
        energy_flux[column] = np.float64(
            energy_flux[column] + energy_tendency
        )
        water_flux[column] = np.float64(
            water_flux[column] + water_tendency
        )
        current_physics[column] = energy_physics[column]
        current_dycore[column] = energy_dycore[column]
        current_water[column] = total_water[column]
    # The pinned check_energy_chng_run resets ``teout`` after every process
    # during the first model timestep.  Keeping only the lifecycle-time
    # initialization makes the following gmean call apply a spurious global
    # fixer to the analytic initial state.
    if bool(pool.get_ccpp("is_first_timestep").item()):
        energy_at_end[...] = energy_at_start


def check_energy_timestep_initial(pool, backend=None) -> None:
    """Compute the initial CAM column integrals for one physics timestep."""

    # CAM passes the current temperature directly to both initial energy
    # integrals and then saves that same field in temp_ini.  The common
    # run-phase helper below reconstructs dycore temperature from temp_ini,
    # so refresh it before the call instead of reusing the previous step.
    initial_temperature = pool.get_ccpp(
        "air_temperature_at_start_of_physics_timestep"
    )
    initial_temperature[...] = pool.get_ccpp("air_temperature")
    energy_physics, energy_dycore, total_water = _hydrostatic_energy(
        pool, backend=backend
    )
    initial_physics = pool.get_ccpp(
        "vertically_integrated_total_energy_using_physics_energy_formula_at_"
        "start_of_physics_timestep"
    )
    initial_dycore = pool.get_ccpp(
        "vertically_integrated_total_energy_using_dycore_energy_formula_at_"
        "start_of_physics_timestep"
    )
    initial_water = pool.get_ccpp(
        "vertically_integrated_total_water_at_start_of_physics_timestep"
    )
    current_physics = pool.get_ccpp(
        "vertically_integrated_total_energy_using_physics_energy_formula"
    )
    current_dycore = pool.get_ccpp(
        "vertically_integrated_total_energy_using_dycore_energy_formula"
    )
    current_water = pool.get_ccpp("vertically_integrated_total_water")
    initial_height = pool.get_ccpp(
        "geopotential_height_wrt_surface_at_start_of_physics_timestep"
    )
    energy_flux = pool.get_ccpp(
        "cumulative_total_energy_boundary_flux_using_physics_energy_formula"
    )
    water_flux = pool.get_ccpp("cumulative_total_water_boundary_flux")
    initial_physics[...] = energy_physics
    initial_dycore[...] = energy_dycore
    initial_water[...] = total_water
    current_physics[...] = energy_physics
    current_dycore[...] = energy_dycore
    current_water[...] = total_water
    initial_height[...] = pool.get_ccpp("geopotential_height_wrt_surface")
    energy_flux[...] = 0.0
    water_flux[...] = 0.0
    pool.set(
        pool.ccpp_field_name(
            "number_of_atmosphere_columns_with_significant_energy_or_water_"
            "imbalances"
        ),
        0,
    )
    if bool(pool.get_ccpp("is_first_timestep").item()):
        pool.get_ccpp(
            "vertically_integrated_total_energy_using_dycore_energy_formula_"
            "at_end_of_physics_timestep"
        )[...] = energy_dycore


def _hydrostatic_energy(
    pool,
    backend=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port the active loop in CAM ``get_hydrostatic_energy_1hd``.

    This is a host service because the original routine obtains constituent
    registry state from CAM modules.  Inputs and outputs remain explicit
    StatePool arrays and the loop/operation order follows the pinned source.
    """

    q = pool.get_ccpp("ccpp_constituents")
    pressure = pool.get_ccpp("air_pressure_thickness")
    zonal = pool.get_ccpp("eastward_wind")
    meridional = pool.get_ccpp("northward_wind")
    temperature = pool.get_ccpp("air_temperature")
    cp_physics = pool.get_ccpp(
        "composition_dependent_specific_heat_of_dry_air_at_constant_pressure"
    )
    cp_dycore = pool.get_ccpp("specific_heat_of_air_used_in_dycore")
    dry_interface = pool.get_ccpp("air_pressure_of_dry_air_at_interface")
    surface_geopotential = pool.get_ccpp("surface_geopotential")
    scaling = pool.get_ccpp(
        "ratio_of_specific_heat_of_air_used_in_physics_energy_formula_to_"
        "specific_heat_of_air_used_in_dycore_energy_formula"
    )
    initial_temperature = pool.get_ccpp(
        "air_temperature_at_start_of_physics_timestep"
    )
    reciprocal_gravity = pool.get(
        "reciprocal_gravitational_acceleration"
    ).item()
    latent_vapor = pool.get("latent_heat_of_vaporization").item()
    latent_ice = pool.get("latent_heat_of_fusion").item()
    ncol, nlev, nconst = q.shape

    if backend is not None:
        liquid_names = {
            "cloud_liquid_water",
            "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
            "rain_water",
            "rain_water_mixing_ratio_wrt_moist_air_and_condensed_water",
        }
        ice_names = {
            "cloud_ice",
            "cloud_ice_mixing_ratio_wrt_moist_air_and_condensed_water",
        }
        liquid_species = np.asfortranarray(
            np.asarray(
                [int(name in liquid_names) for name in pool.constituent_names],
                dtype=np.int32,
            )
        )
        ice_species = np.asfortranarray(
            np.asarray(
                [int(name in ice_names) for name in pool.constituent_names],
                dtype=np.int32,
            )
        )
        return backend.hydrostatic_energy(
            water_vapor_index=water_vapor_index(pool.constituent_names),
            liquid_species=liquid_species,
            ice_species=ice_species,
            reciprocal_gravity=reciprocal_gravity,
            latent_vapor=latent_vapor,
            latent_ice=latent_ice,
            constituent=q,
            pressure_thickness=pressure,
            zonal_wind=zonal,
            meridional_wind=meridional,
            temperature=temperature,
            initial_temperature=initial_temperature,
            physics_heat_capacity=cp_physics,
            dycore_heat_capacity=cp_dycore,
            dycore_scaling=scaling,
            dry_interface_pressure=dry_interface,
            surface_geopotential=surface_geopotential,
        )

    water_vapor = None
    liquid_fields: list[np.ndarray] = []
    ice_fields: list[np.ndarray] = []
    for standard_name, category in (
        (
            "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water",
            "vapor",
        ),
        (
            "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
            "liquid",
        ),
        (
            "rain_mixing_ratio_wrt_moist_air_and_condensed_water",
            "liquid",
        ),
        (
            "cloud_ice_mixing_ratio_wrt_moist_air_and_condensed_water",
            "ice",
        ),
    ):
        try:
            field = pool.get_ccpp(standard_name)
        except KeyError:
            continue
        if category == "vapor":
            water_vapor = field
        elif category == "liquid":
            if not any(np.shares_memory(field, item) for item in liquid_fields):
                liquid_fields.append(field)
        else:
            ice_fields.append(field)
    if water_vapor is None:
        # All current CAM-SIMA suites register water vapor.  Keep a clear
        # failure at the host/device boundary rather than silently selecting
        # an arbitrary constituent.
        raise KeyError("CCPP constituent registry has no water-vapor field")

    def integrate(cp: np.ndarray, temp: np.ndarray) -> np.ndarray:
        result = np.empty(ncol, dtype=np.float64, order="F")
        for column in range(ncol):
            kinetic = np.float64(0.0)
            static = np.float64(0.0)
            potential = np.float64(dry_interface[column, 0])
            vapor = np.float64(0.0)
            liquid = np.float64(0.0)
            ice = np.float64(0.0)
            for level in range(nlev):
                dp = pressure[column, level]
                kinetic = np.float64(
                    kinetic
                    + np.float64(
                        np.float64(
                            dp
                            * np.float64(0.5)
                            * np.float64(
                                np.float64(zonal[column, level] ** 2)
                                + np.float64(meridional[column, level] ** 2)
                            )
                        )
                        * reciprocal_gravity
                    )
                )
                static = np.float64(
                    static
                    + np.float64(
                        np.float64(
                            temp[column, level]
                            * cp[column, level]
                            * dp
                        )
                        * reciprocal_gravity
                    )
                )
                potential = np.float64(potential + dp)
                vapor = np.float64(
                    vapor
                    + np.float64(
                        np.float64(water_vapor[column, level] * dp)
                        * reciprocal_gravity
                    )
                )
                for field in liquid_fields:
                    liquid = np.float64(
                        liquid
                        + np.float64(
                            np.float64(field[column, level] * dp)
                            * reciprocal_gravity
                        )
                    )
                for field in ice_fields:
                    ice = np.float64(
                        ice
                        + np.float64(
                            np.float64(field[column, level] * dp)
                            * reciprocal_gravity
                        )
                    )
            potential = np.float64(
                surface_geopotential[column]
                * potential
                * reciprocal_gravity
            )
            # The pinned CAM-SIMA component configures ice as the enthalpy
            # reference state.
            result[column] = np.float64(
                static
                + potential
                + kinetic
                + np.float64(
                    np.float64(latent_vapor + latent_ice) * vapor
                )
                + np.float64(latent_ice * liquid)
            )
        return result

    physics_energy = integrate(cp_physics, temperature)
    dycore_temperature = np.empty_like(temperature, order="F")
    for level in range(nlev):
        for column in range(ncol):
            dycore_temperature[column, level] = np.float64(
                initial_temperature[column, level]
                + np.float64(
                    scaling[column, level]
                    * np.float64(
                        temperature[column, level]
                        - initial_temperature[column, level]
                    )
                )
            )
    dycore_energy = integrate(cp_dycore, dycore_temperature)
    total_water = np.empty(ncol, dtype=np.float64, order="F")
    for column in range(ncol):
        total = np.float64(0.0)
        for level in range(nlev):
            for constituent in range(nconst):
                total = np.float64(
                    total
                    + np.float64(
                        np.float64(q[column, level, constituent]
                                   * pressure[column, level])
                        * reciprocal_gravity
                    )
                )
        total_water[column] = total
    return physics_energy, dycore_energy, total_water


def _fixed_point_reproducible_sums(
    rank_values: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Match CAM ``shr_reprosum_calc`` fixed-point REAL64 reconstruction.

    The CAM routine does not simply round the exact mathematical sum once.
    It decomposes each summand into base-2 integer levels, reduces those
    integers, and reconstructs the result from the smallest level upward.
    The reconstruction deliberately performs several REAL64 additions and
    can therefore differ by an ulp from a correctly-rounded exact rational
    sum.  Preserve those operations here so the Python host service is BFB
    with the original Fortran component service.
    """

    if not rank_values:
        raise ValueError("reproducible sum requires at least one rank")
    if any(values.ndim != 2 for values in rank_values):
        raise ValueError("reproducible sum inputs must be rank-2 arrays")
    fields = rank_values[0].shape[1]
    if any(values.shape[1] != fields for values in rank_values):
        raise ValueError("reproducible sum field counts do not match")

    # CAM is built without OpenMP for this validation profile.  These are the
    # corresponding values from shr_reprosum_calc:
    #   arr_max_shift = digits(int64) - (exponent(max_summands) + 1)
    tasks = len(rank_values)
    local_extent = max(values.shape[0] for values in rank_values)
    maximum_partial_summands = max(local_extent + 1, tasks)
    maximum_shift = 63 - (
        math.frexp(float(maximum_partial_summands))[1] + 1
    )
    if maximum_shift < 2:
        raise OverflowError("too many summands for fixed-point reduction")
    integer_radix_level = 2**maximum_shift

    def exponent(value: float | np.float64) -> int:
        return math.frexp(float(value))[1]

    def fraction(value: float | np.float64) -> np.float64:
        return np.float64(math.frexp(float(value))[0])

    def scale(value: float | np.float64, shift: int) -> np.float64:
        return np.float64(np.ldexp(np.float64(value), int(shift)))

    def carry_levels(levels: list[int]) -> None:
        for level in range(len(levels) - 1, 0, -1):
            represented = np.float64(levels[level])
            carry = int(scale(represented, -maximum_shift))
            if carry:
                levels[level - 1] += carry
                levels[level] -= carry * integer_radix_level

    results = np.empty(fields, dtype=np.float64)
    for field_index in range(fields):
        nonzero: list[np.float64] = []
        for values in rank_values:
            for raw in values[:, field_index]:
                value = np.float64(raw)
                if not math.isfinite(value):
                    raise FloatingPointError(
                        "non-finite input to reproducible sum"
                    )
                if value != 0.0:
                    nonzero.append(value)
        if not nonzero:
            results[field_index] = 0.0
            continue

        global_maximum_exponent = max(exponent(value) for value in nonzero)
        global_minimum_exponent = min(exponent(value) for value in nonzero)
        maximum_levels = 2 + (
            53 + global_maximum_exponent - global_minimum_exponent
        ) // maximum_shift

        # Build one integer expansion per MPI rank, including the extra
        # overflow level at index zero, then emulate MPI_INTEGER8/SUM.
        rank_levels: list[list[int]] = []
        for values in rank_values:
            levels = [0] * (maximum_levels + 1)
            for raw in values[:, field_index]:
                value = np.float64(raw)
                if value == 0.0:
                    continue
                value_exponent = exponent(value)
                remainder = np.float64(0.0)
                level_shift = maximum_shift - (
                    global_maximum_exponent - value_exponent
                )
                if level_shift < 1:
                    level = (
                        1 + global_maximum_exponent - value_exponent
                    ) // maximum_shift
                    level_shift = (
                        level * maximum_shift
                        - (global_maximum_exponent - value_exponent)
                    )
                    while level_shift < 1:
                        level_shift += maximum_shift
                        level += 1
                else:
                    level = 1

                if level <= maximum_levels:
                    remainder = scale(fraction(value), level_shift)
                    integer_part = int(remainder)
                    levels[level] += integer_part
                    remainder = np.float64(remainder - integer_part)
                    while remainder != 0.0 and level < maximum_levels:
                        level += 1
                        remainder = scale(remainder, maximum_shift)
                        integer_part = int(remainder)
                        levels[level] += integer_part
                        remainder = np.float64(remainder - integer_part)

                if remainder != 0.0:
                    raise ArithmeticError(
                        "fixed-point expansion lost significant digits"
                    )

            carry_levels(levels)
            rank_levels.append(levels)

        global_levels = [
            sum(levels[level] for levels in rank_levels)
            for level in range(maximum_levels + 1)
        ]
        carry_levels(global_levels)

        # Make all nonzero components have the same sign before converting
        # them back to binary64.  This is the cancellation-avoidance pass in
        # shr_reprosum_int.
        first_nonzero = next(
            (
                level
                for level, value in enumerate(global_levels)
                if value != 0
            ),
            maximum_levels,
        )
        if first_nonzero < maximum_levels:
            sign = 1 if global_levels[first_nonzero] > 0 else -1
            for level in range(first_nonzero, maximum_levels):
                current_sign = 1 if global_levels[level] >= 0 else -1
                next_sign = 1 if global_levels[level + 1] >= 0 else -1
                if current_sign != next_sign:
                    global_levels[level] -= sign
                    global_levels[level + 1] += (
                        sign * integer_radix_level
                    )

        # Reconstruct from the smallest fixed-point level to the largest.
        level_shift = (
            global_maximum_exponent
            - maximum_levels * maximum_shift
        )
        current_exponent = 0
        total = np.float64(0.0)
        first = True
        for level in range(maximum_levels, -1, -1):
            integer_value = global_levels[level]
            if integer_value:
                pieces: list[tuple[np.float64, int]] = []
                represented = np.float64(integer_value)
                represented_integer = int(represented)
                pieces.append((represented, exponent(represented)))
                remainder = np.float64(
                    integer_value - represented_integer
                )
                while remainder != 0.0:
                    pieces.append((remainder, exponent(remainder)))
                    represented_integer += int(remainder)
                    remainder = np.float64(
                        integer_value - represented_integer
                    )

                for piece, piece_exponent in reversed(pieces):
                    if first:
                        current_exponent = (
                            piece_exponent + level_shift
                        )
                        total = fraction(piece)
                        first = False
                    else:
                        correction_exponent = current_exponent - (
                            piece_exponent + level_shift
                        )
                        total = np.float64(
                            fraction(piece)
                            + scale(total, correction_exponent)
                        )
                        current_exponent = (
                            piece_exponent + level_shift
                        )
            level_shift += maximum_shift

        if first:
            results[field_index] = 0.0
            continue
        correction_exponent = current_exponent + exponent(total)
        results[field_index] = scale(
            fraction(total), correction_exponent
        )
    return results


def check_energy_gmean(pool, comm) -> None:
    """Compute CAM's area-weighted global energy means in Python.

    ``gmean_mod`` is a component service because its only non-local operation
    is a communicator-wide reproducible sum.  Keeping that operation here
    avoids linking MPI into a numerical device while preserving the original
    scheme formulas and the rank-local physics-grid weights.
    """

    def field(standard_name: str) -> np.ndarray:
        return pool.get(pool.ccpp_field_name(standard_name))

    interface_pressure = field("air_pressure_at_interface")
    energy_in = field(
        "vertically_integrated_total_energy_using_dycore_energy_formula_at_"
        "start_of_physics_timestep"
    )
    energy_out = field(
        "vertically_integrated_total_energy_using_dycore_energy_formula_at_"
        "end_of_physics_timestep"
    )
    local_values = np.empty(
        (pool.dimensions["nphys_local"], 4),
        dtype=np.float64,
        order="F",
    )
    local_values[:, 0] = energy_in
    local_values[:, 1] = energy_out
    local_values[:, 2] = interface_pressure[:, -1]
    local_values[:, 3] = interface_pressure[:, 0]
    weights = pool.get("physics_cell_area")
    weighted = np.empty_like(local_values, order="F")
    for index in range(4):
        for column in range(local_values.shape[0]):
            weighted[column, index] = np.float64(
                local_values[column, index] * weights[column]
            )

    gathered = tuple(comm.allgather(weighted))
    normalization = np.float64(
        np.float64(4.0) * pool.get("circle_constant").item()
    )
    means = _fixed_point_reproducible_sums(gathered)
    means[...] = means / normalization

    output_names = (
        "global_mean_vertically_integrated_total_energy_using_dycore_energy_"
        "formula_at_start_of_physics_timestep",
        "global_mean_vertically_integrated_total_energy_using_dycore_energy_"
        "formula_at_end_of_physics_timestep",
        "global_mean_surface_air_pressure",
        "global_mean_air_pressure_at_top_of_atmosphere_model",
    )
    for standard_name, value in zip(output_names, means):
        pool.set(pool.ccpp_field_name(standard_name), value)

    correction = np.float64(means[0] - means[1])
    pool.set(
        pool.ccpp_field_name(
            "global_mean_total_energy_correction_for_energy_conservation"
        ),
        correction,
    )
    timestep = pool.get("model_timestep").item()
    gravity = pool.get("gravitational_acceleration").item()
    # Preserve the original Fortran evaluation order exactly:
    #
    #   -tedif_glob / dtime * gravit / (psurf_glob - ptopb_glob)
    #
    # Grouping ``gravit / (psurf_glob - ptopb_glob)`` first changes the
    # rounded REAL64 result and eventually shifts CAM7 temperature by one or
    # two ULPs.
    heating = np.float64(-correction / timestep)
    heating = np.float64(heating * gravity)
    heating = np.float64(
        heating / np.float64(means[2] - means[3])
    )
    pool.set(
        pool.ccpp_field_name(
            "global_mean_heating_rate_correction_for_energy_conservation"
        ),
        heating,
    )


def sima_state_diagnostics(pool) -> None:
    """Mark the state-diagnostic boundary owned by :class:`HistoryWriter`."""


def kessler_diagnostics(pool) -> None:
    """Mark the Kessler precipitation-diagnostic boundary."""


def sima_tend_diagnostics(pool) -> None:
    """Mark the physics-tendency diagnostic boundary."""


def potential_temperature_to_temperature(pool) -> None:
    pool.set("physics_air_temperature", pool.get("potential_temperature") * pool.get("exner_function"))


def physics_to_dynamics(pool, time_level: int = 0) -> None:
    w = pool.get("mapping_weights_pg3_to_gll")
    nlev, nconst = pool.dimensions["pver"], pool.dimensions["nconst"]
    nc = pool.dimensions["fv_nphys"]
    columns_per_element = nc * nc
    for le in range(pool.dimensions["nelem_local"]):
        columns = slice(
            le * columns_per_element,
            (le + 1) * columns_per_element,
        )
        for source, target in (("physics_zonal_wind", "zonal_wind"), ("physics_meridional_wind", "meridional_wind"), ("physics_air_temperature", "air_temperature"), ("physics_layer_pressure_thickness", "layer_pressure_thickness")):
            src, dst = pool.get(source), pool.get(target)
            for lev in range(nlev):
                dst[:,:,lev,le,time_level] = w @ src[columns,lev].reshape((nc,nc), order="F") @ w.T
        pool.get("surface_pressure")[:,:,le,time_level] = w @ pool.get("physics_surface_pressure")[columns].reshape((nc,nc), order="F") @ w.T
        for constituent in range(nconst):
            for lev in range(nlev):
                pool.get("constituent_mixing_ratio")[:,:,lev,le,constituent,time_level] = w @ pool.get("physics_constituent_mixing_ratio")[columns,lev,constituent].reshape((nc,nc), order="F") @ w.T
