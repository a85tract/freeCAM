"""Explicit model phase boundaries controlled by Python."""

from __future__ import annotations

import numpy as np

from .grid import _pg3_reference_nodes


def _integrate_subcells(sample: np.ndarray, metdet: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scalar-order equivalent of derivative_mod:subcell_integration."""
    val = np.empty((4, 4), dtype=np.float64, order="F")
    tmp = np.empty((4, 3), dtype=np.float64, order="F")
    result = np.empty((3, 3), dtype=np.float64, order="F")
    for j in range(4):
        for i in range(4):
            val[i, j] = np.float64(sample[i, j] * metdet[i, j])
    # MATMUL(val, TRANSPOSE(weights))
    for j in range(3):
        for i in range(4):
            value = np.float64(0.0)
            for k in range(4):
                value = np.float64(value + np.float64(val[i, k] * weights[j, k]))
            tmp[i, j] = value
    # MATMUL(weights, tmp)
    for j in range(3):
        for i in range(3):
            value = np.float64(0.0)
            for k in range(4):
                value = np.float64(value + np.float64(weights[i, k] * tmp[k, j]))
            result[i, j] = value
    return result


def _interpolate_tensor_point(field: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.float64:
    intermediate = np.empty(4, dtype=np.float64)
    for j in range(4):
        value = np.float64(0.0)
        for i in range(4):
            value = np.float64(value + np.float64(wx[i] * field[i, j]))
        intermediate[j] = value
    value = np.float64(0.0)
    for j in range(4):
        value = np.float64(value + np.float64(wy[j] * intermediate[j]))
    return value


def _interpolate_legendre_2d(field: np.ndarray, x: float, y: float, matrix: np.ndarray) -> np.float64:
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


def initialize_fvm_state(pool, time_level: int = 0) -> None:
    """Port dyn2fvm_mass_vars for the startup GLL state."""
    weights = pool.get("mapping_subcell_integration")
    nlev, nconst = pool.dimensions["pver"], pool.dimensions["nconst"]
    ones = np.ones((4, 4), dtype=np.float64, order="F")
    for le in range(pool.dimensions["nelem_local"]):
        metdet = pool.get("metric_jacobian")[:, :, le]
        area = _integrate_subcells(ones, metdet, weights)
        inv_area = np.float64(1.0) / area
        ps = pool.get("surface_pressure")[:, :, le, time_level]
        pool.get("fvm_surface_dry_air_pressure")[:, :, le] = _integrate_subcells(ps, metdet, weights) * inv_area
        for lev in range(nlev):
            dp_gll = pool.get("layer_pressure_thickness")[:, :, lev, le, time_level]
            dp_fvm = _integrate_subcells(dp_gll, metdet, weights) * inv_area
            pool.get("fvm_layer_pressure_thickness")[3:6, 3:6, lev, le] = dp_fvm
            inv_darea_dp = inv_area / dp_fvm
            for constituent in range(nconst):
                q_gll = pool.get("constituent_mixing_ratio")[:, :, lev, le, constituent, time_level]
                q_fvm = _integrate_subcells(q_gll * dp_gll, metdet, weights) * inv_darea_dp
                minimum = np.min(q_gll)
                maximum = np.max(q_gll)
                for j in range(3):
                    for i in range(3):
                        q_fvm[i, j] = max(minimum, min(maximum, q_fvm[i, j]))
                pool.get("fvm_tracer")[3:6, 3:6, lev, le, constituent] = q_fvm


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
            for constituent in range(pool.dimensions["nconst"]):
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
    if backend is not None:
        backend.physics_diagnostics(
            update_static_energy=update_static_energy,
            gravity=gravity,
            virtual_temperature_coefficient=zvir,
            temperature=temp,
            constituent=q,
            pressure_thickness=pdel,
            midpoint_pressure=pmid,
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
    # The current SE/FVM capability has lagrangian_vertical=.false.; use the Eulerian
    # hydrostatic elements from geopotential_temp_run, bottom upward.
    for k in range(nlev - 1, -1, -1):
        for i in range(ncol):
            hkl = np.float64(pdel[i, k] / pmid[i, k])
            hkk = np.float64(0.5) * hkl
            qfac_denom = np.float64(1.0)
            for constituent in range(nconst):
                qfac_denom = np.float64(qfac_denom - q[i, k, constituent])
            qfac = np.float64(1.0) / qfac_denom
            sum_dry = np.float64(1.0)
            for constituent in range(nconst):
                sum_dry = np.float64(sum_dry + np.float64(q[i, k, constituent] * qfac))
            sum_dry = np.float64(1.0) / sum_dry
            tvfac = np.float64(1.0 + np.float64((zvir + np.float64(1.0)) * q[i, k, 2] * qfac)) * sum_dry
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
    """Apply the configured constituent lower bounds."""

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


def dynamics_to_physics(pool, time_level: int | None = None) -> None:
    if time_level is None:
        time_level = int(pool.get("dynamics_time_level_n0"))
    w = pool.get("mapping_weights_gll_to_pg3")
    integration = pool.get("mapping_subcell_integration")
    interpolation_matrix = pool.get("mapping_interpolation_matrix")
    nlev, nconst = pool.dimensions["pver"], pool.dimensions["nconst"]
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(pool.get("reference_pressure"))
    pg3_nodes = _pg3_reference_nodes()
    ones = np.ones((4, 4), dtype=np.float64, order="F")
    for le in range(pool.dimensions["nelem_local"]):
        columns = slice(le * 9, (le + 1) * 9)
        metdet = pool.get("metric_jacobian")[:, :, le]
        area = _integrate_subcells(ones, metdet, integration)
        inv_area = np.float64(1.0) / area
        psdry = np.full((3, 3), ptop, dtype=np.float64, order="F")
        dinv = pool.get("inverse_metric")[:, :, :, :, le]
        dphys = pool.get("mapping_derivative_pg3")[:, :, columns]
        for lev in range(nlev):
            u = pool.get("zonal_wind")[:, :, lev, le, time_level]
            v = pool.get("meridional_wind")[:, :, lev, le, time_level]
            contra1 = np.empty((4, 4), dtype=np.float64, order="F")
            contra2 = np.empty((4, 4), dtype=np.float64, order="F")
            for j in range(4):
                for i in range(4):
                    contra1[i, j] = np.float64(dinv[0, 0, i, j] * u[i, j]) + np.float64(dinv[0, 1, i, j] * v[i, j])
                    contra2[i, j] = np.float64(dinv[1, 0, i, j] * u[i, j]) + np.float64(dinv[1, 1, i, j] * v[i, j])
            for pj in range(3):
                for pi in range(3):
                    local_col = pi + 3 * pj
                    x = pg3_nodes[pi]
                    y = pg3_nodes[pj]
                    v1 = _interpolate_legendre_2d(contra1, x, y, interpolation_matrix)
                    v2 = _interpolate_legendre_2d(contra2, x, y, interpolation_matrix)
                    pool.get("physics_zonal_wind")[le * 9 + local_col, lev] = np.float64(dphys[0, 0, local_col] * v1) + np.float64(dphys[0, 1, local_col] * v2)
                    pool.get("physics_meridional_wind")[le * 9 + local_col, lev] = np.float64(dphys[1, 0, local_col] * v1) + np.float64(dphys[1, 1, local_col] * v2)
        for lev in range(nlev):
            dp_fvm = pool.get("fvm_layer_pressure_thickness")[3:6, 3:6, lev, le]
            psdry = psdry + dp_fvm
            pool.get("physics_layer_pressure_thickness")[columns, lev] = dp_fvm.reshape(9, order="F")
            t_gll = pool.get("air_temperature")[:, :, lev, le, time_level]
            dp_gll = pool.get("layer_pressure_thickness")[:, :, lev, le, time_level]
            # dyn2phys_all_vars derives the temperature denominator from an
            # independent GLL dry-mass integration.  The prognostic PG3 dp is
            # copied to physics, but substituting it here changes low bits
            # after the first vertical remap.
            dp3d_tmp = _integrate_subcells(dp_gll, metdet, integration) * inv_area
            inv_darea_dp_phys = inv_area / dp3d_tmp
            t_phys = _integrate_subcells(
                t_gll * dp_gll, metdet, integration
            ) * inv_darea_dp_phys
            pool.get("physics_air_temperature")[columns, lev] = t_phys.reshape(9, order="F")
            omega = _integrate_subcells(pool.get("vertical_pressure_velocity")[:, :, lev, le], metdet, integration) * inv_area
            pool.get("physics_vertical_pressure_velocity")[columns, lev] = omega.reshape(9, order="F")
            for constituent in range(nconst):
                q_fvm = pool.get("fvm_tracer")[3:6, 3:6, lev, le, constituent]
                pool.get("physics_constituent_mixing_ratio")[columns, lev, constituent] = q_fvm.reshape(9, order="F")
        # d_p_coupling writes dry surface pressure.  Moist PS is formed later
        # by derived_phys_dry in physics_timestep_initial.
        pool.get("physics_surface_dry_air_pressure")[columns] = psdry.reshape(9, order="F")
        pool.get("physics_surface_geopotential")[columns] = 0.0
    # d_p_coupling saves the dry tracer state before derived_phys_dry converts
    # water to the moist physics convention.
    pool.get("physics_constituent_previous")[...] = pool.get(
        "physics_constituent_mixing_ratio"
    )


def physics_timestep_initial(pool, backend=None) -> None:
    _derive_pressure_and_geopotential(pool, backend=backend)
    if "air_temperature_previous_timestep" in pool.contracts:
        pool.set(
            "air_temperature_previous_timestep",
            pool.get("physics_air_temperature"),
        )
    for name in ("physics_air_temperature_tendency", "physics_zonal_wind_tendency", "physics_meridional_wind_tendency", "physics_constituent_tendency"):
        pool.get(name)[...] = 0.0


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


def thermo_water_update(pool) -> None:
    """Update the SE heat capacity for the current water composition."""

    ncol, nlev = pool.dimensions["nphys_local"], pool.dimensions["pver"]
    q = pool.get("physics_constituent_mixing_ratio")
    factor = pool.get("physics_layer_pressure_thickness") / pool.get(
        "physics_dry_layer_pressure_thickness"
    )
    cpair = pool.get("column_dry_air_specific_heat")
    cpdy = pool.get("dycore_heat_capacity")
    species_cp = (
        np.float64(pool.get("liquid_water_specific_heat")),
        np.float64(pool.get("liquid_water_specific_heat")),
        np.float64(pool.get("water_vapor_specific_heat")),
    )
    # air_composition orders the active species as vapor, cloud liquid, rain,
    # while the CAM constituent array is cloud liquid, rain, vapor.
    for k in range(nlev):
        for i in range(ncol):
            sum_species = np.float64(1.0)
            sum_cp = cpair[i, k]
            for constituent in (2, 0, 1):
                dry = np.float64(q[i, k, constituent] * factor[i, k])
                sum_species = np.float64(sum_species + dry)
                term = np.float64(
                    np.float64(species_cp[constituent] * q[i, k, constituent])
                    * factor[i, k]
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


def check_energy_chng(pool) -> None:
    """Represent CAM's message-only conservation check for this capability."""


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
    for le in range(pool.dimensions["nelem_local"]):
        columns = slice(le * 9, (le + 1) * 9)
        for source, target in (("physics_zonal_wind", "zonal_wind"), ("physics_meridional_wind", "meridional_wind"), ("physics_air_temperature", "air_temperature"), ("physics_layer_pressure_thickness", "layer_pressure_thickness")):
            src, dst = pool.get(source), pool.get(target)
            for lev in range(nlev):
                dst[:,:,lev,le,time_level] = w @ src[columns,lev].reshape((3,3), order="F") @ w.T
        pool.get("surface_pressure")[:,:,le,time_level] = w @ pool.get("physics_surface_pressure")[columns].reshape((3,3), order="F") @ w.T
        for constituent in range(nconst):
            for lev in range(nlev):
                pool.get("constituent_mixing_ratio")[:,:,lev,le,constituent,time_level] = w @ pool.get("physics_constituent_mixing_ratio")[columns,lev,constituent].reshape((3,3), order="F") @ w.T
