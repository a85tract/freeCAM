"""Spectral-element timestep implementation for the CAM SE/FVM capability.

Some routines in this module are transcribed from CESM/CAM Fortran sources
and preserve their upstream expression order; each is marked with a ``Port
...`` docstring naming its upstream routine.  Those routines are
Copyright (c) 2017, University Corporation for Atmospheric Research (UCAR)
and are redistributed under the BSD 3-Clause license in
LICENSES/UCAR-CESM-BSD-3-Clause.txt.  See NOTICE section 2.
"""

from __future__ import annotations

import numpy as np

from .constituents import is_water_vapor, water_constituent_indices
from .dynamics import (
    _divergence_sphere,
    _edge_sum,
    _gradient_sphere,
    _vorticity_sphere,
)


def _matmul_source(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty((left.shape[0], right.shape[1]), dtype=np.float64, order="F")
    for j in range(right.shape[1]):
        for i in range(left.shape[0]):
            value = np.float64(0.0)
            for k in range(left.shape[1]):
                value = np.float64(value + np.float64(left[i, k] * right[k, j]))
            result[i, j] = value
    return result


def _subcell_integrate(pool, sampled: np.ndarray, metdet: np.ndarray) -> np.ndarray:
    weights = pool.get("mapping_subcell_integration")
    np_value = sampled.shape[0]
    weighted = np.empty((np_value, np_value), dtype=np.float64, order="F")
    for j in range(np_value):
        for i in range(np_value):
            weighted[i, j] = np.float64(sampled[i, j] * metdet[i, j])
    return _matmul_source(weights, _matmul_source(weighted, weights.T))


def _subcell_div_fluxes(pool, contravariant: np.ndarray, metdet: np.ndarray) -> np.ndarray:
    weights = pool.get("mapping_subcell_integration")
    boundary = pool.get("mapping_boundary_interpolation")
    inverse_radius = np.float64(1.0) / np.float64(pool.get("earth_radius"))
    vector = np.empty_like(contravariant, order="F")
    np_value = contravariant.shape[0]
    nc = weights.shape[0]
    for component in range(2):
        for j in range(np_value):
            for i in range(np_value):
                vector[i, j, component] = np.float64(
                    contravariant[i, j, component] * metdet[i, j]
                )
    top_bottom = _matmul_source(weights, vector[:, :, 1])
    flux_bottom = _matmul_source(top_bottom, boundary[:, 0, :].T)
    flux_top = _matmul_source(top_bottom, boundary[:, 1, :].T)
    left_right = _matmul_source(vector[:, :, 0], weights.T)
    flux_left = _matmul_source(boundary[:, 0, :], left_right)
    flux_right = _matmul_source(boundary[:, 1, :], left_right)
    result = np.empty((nc, nc, 4), dtype=np.float64, order="F")
    for j in range(nc):
        for i in range(nc):
            result[i, j, 0] = np.float64(-flux_bottom[i, j] * inverse_radius)
            result[i, j, 1] = np.float64(flux_right[i, j] * inverse_radius)
            result[i, j, 2] = np.float64(flux_top[i, j] * inverse_radius)
            result[i, j, 3] = np.float64(-flux_left[i, j] * inverse_radius)
    return result


def _distribute_corner_flux(
    corners: np.ndarray,
    has_diagonal: tuple[bool, bool, bool, bool],
) -> np.ndarray:
    result = np.zeros((2, 2, 2), dtype=np.float64, order="F")
    outside = corners.shape[0] - 1
    inside = outside - 1
    # southwest, southeast, northwest, northeast in HOMME getmapP order.
    definitions = (
        (0, 0, 1, 1, 0, 1, 1, 0),
        (outside, 0, inside, 1, outside, 1, inside, 0),
        (0, outside, 1, inside, 0, inside, 1, outside),
        (
            outside,
            outside,
            inside,
            inside,
            outside,
            inside,
            inside,
            outside,
        ),
    )
    for corner, (ox, oy, ix, iy, ax, ay, bx, by) in enumerate(definitions):
        ci, cj = (corner % 2, corner // 2)
        horizontal = np.float64(corners[ox, ay] - corners[ix, iy])
        vertical = np.float64(corners[bx, oy] - corners[ix, iy])
        if has_diagonal[corner]:
            horizontal = np.float64(
                horizontal
                + np.float64(np.float64(corners[ox, oy] - corners[ix, iy]) / np.float64(2.0))
                + np.float64(np.float64(corners[ox, ay] - corners[bx, oy]) / np.float64(2.0))
            )
            vertical = np.float64(
                vertical
                + np.float64(np.float64(corners[ox, oy] - corners[ix, iy]) / np.float64(2.0))
                + np.float64(np.float64(corners[bx, oy] - corners[ox, ay]) / np.float64(2.0))
            )
        # The source's first component is the x-normal correction and the
        # second is y-normal; signs are already encoded by each corner's
        # outside-minus-inside expressions above.
        result[ci, cj, 0] = horizontal
        result[ci, cj, 1] = vertical
    return result


def _subcell_dss_fluxes(
    pool,
    dss: np.ndarray,
    metdet: np.ndarray,
    corner_flux: np.ndarray,
) -> np.ndarray:
    np_value = dss.shape[0]
    nc = pool.get("mapping_subcell_integration").shape[0]
    last = np_value - 1
    bottom_p = np.zeros((np_value, np_value), dtype=np.float64, order="F")
    top_p = np.zeros((np_value, np_value), dtype=np.float64, order="F")
    left_p = np.zeros((np_value, np_value), dtype=np.float64, order="F")
    right_p = np.zeros((np_value, np_value), dtype=np.float64, order="F")
    bottom_p[:, 0] = dss[:, 0]
    top_p[:, last] = dss[:, last]
    right_p[last, :] = dss[last, :]
    left_p[0, :] = dss[0, :]
    bottom_p[0, 0], left_p[0, 0] = corner_flux[0, 0, 1], corner_flux[0, 0, 0]
    bottom_p[last, 0], right_p[last, 0] = corner_flux[1, 0, 1], corner_flux[1, 0, 0]
    top_p[0, last], left_p[0, last] = corner_flux[0, 1, 1], corner_flux[0, 1, 0]
    top_p[last, last], right_p[last, last] = (
        corner_flux[1, 1, 1],
        corner_flux[1, 1, 0],
    )
    bottom = _subcell_integrate(pool, bottom_p, metdet)
    top = _subcell_integrate(pool, top_p, metdet)
    left = _subcell_integrate(pool, left_p, metdet)
    right = _subcell_integrate(pool, right_p, metdet)
    for i in range(nc):
        for j in range(nc):
            if j > 0:
                top[i, j] = np.float64(top[i, j] + top[i, j - 1])
            if i > 0:
                right[i, j] = np.float64(right[i, j] + right[i - 1, j])
    for i in range(nc - 1, -1, -1):
        for j in range(nc - 1, -1, -1):
            if j < nc - 1:
                bottom[i, j] = np.float64(bottom[i, j] + bottom[i, j + 1])
            if i < nc - 1:
                left[i, j] = np.float64(left[i, j] + left[i + 1, j])
    result = np.zeros((nc, nc, 4), dtype=np.float64, order="F")
    for i in range(nc):
        for j in range(nc):
            result[i, j, 0] = bottom[i, j] if j == 0 else np.float64(bottom[i, j] - top[i, j - 1])
            result[i, j, 1] = right[i, j] if i == nc - 1 else np.float64(right[i, j] - left[i + 1, j])
            result[i, j, 2] = top[i, j] if j == nc - 1 else np.float64(top[i, j] - bottom[i, j + 1])
            result[i, j, 3] = left[i, j] if i == 0 else np.float64(left[i, j] - right[i - 1, j])
    return result


def _subcell_laplace_fluxes(pool, scalar: np.ndarray, le: int) -> np.ndarray:
    """Port ``subcell_Laplace_fluxes`` for one GLL element field."""

    dvv = pool.get("gll_derivative")
    dinv = pool.get("inverse_metric")[:, :, :, :, le]
    mass = pool.get("spectral_mass_matrix")[:, :, le]
    metdet = pool.get("metric_jacobian")[:, :, le]
    inverse_radius = np.float64(1.0) / np.float64(pool.get("earth_radius"))
    gradient = _gradient_sphere(scalar, dvv, dinv, inverse_radius)
    np_value = scalar.shape[0]
    nc = pool.get("mapping_subcell_integration").shape[0]
    vector = np.empty((np_value, np_value, 2), dtype=np.float64, order="F")
    divergence = np.empty_like(vector, order="F")
    for j in range(np_value):
        for i in range(np_value):
            vector[i, j, 0] = np.float64(
                np.float64(dinv[0, 0, i, j] * gradient[i, j, 0])
                + np.float64(dinv[0, 1, i, j] * gradient[i, j, 1])
            )
            vector[i, j, 1] = np.float64(
                np.float64(dinv[1, 0, i, j] * gradient[i, j, 0])
                + np.float64(dinv[1, 1, i, j] * gradient[i, j, 1])
            )
    for j in range(np_value):
        for i in range(np_value):
            first = np.float64(0.0)
            second = np.float64(0.0)
            for l in range(np_value):
                first = np.float64(
                    first
                    - np.float64(
                        np.float64(mass[l, j] * vector[l, j, 0]) * dvv[i, l]
                    )
                )
                second = np.float64(
                    second
                    - np.float64(
                        np.float64(mass[i, l] * vector[i, l, 1]) * dvv[j, l]
                    )
                )
            divergence[i, j, 0] = np.float64(first * inverse_radius)
            divergence[i, j, 1] = np.float64(second * inverse_radius)
            divergence[i, j, 0] = np.float64(divergence[i, j, 0] / mass[i, j])
            divergence[i, j, 1] = np.float64(divergence[i, j, 1] / mass[i, j])
    integrated = np.empty((nc, nc, 2), dtype=np.float64, order="F")
    integrated[:, :, 0] = _subcell_integrate(pool, divergence[:, :, 0], metdet)
    integrated[:, :, 1] = _subcell_integrate(pool, divergence[:, :, 1], metdet)
    for i in range(nc):
        for j in range(1, nc):
            integrated[j, i, 0] = np.float64(
                integrated[j, i, 0] + integrated[j - 1, i, 0]
            )
            integrated[i, j, 1] = np.float64(
                integrated[i, j, 1] + integrated[i, j - 1, 1]
            )
    flux = np.zeros((nc, nc, 4), dtype=np.float64, order="F")
    for i in range(nc):
        for j in range(nc):
            if j > 0:
                flux[i, j, 0] = np.float64(-integrated[i, j - 1, 1])
            if i < nc - 1:
                flux[i, j, 1] = integrated[i, j, 0]
            if j < nc - 1:
                flux[i, j, 2] = integrated[i, j, 1]
            if i > 0:
                flux[i, j, 3] = np.float64(-integrated[i - 1, j, 0])
    return flux


def _dg_halo(pool, comm, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exchange unsummed element-edge values as ``edgeDGVunpack`` does."""

    ids = np.asarray(pool.get("global_element_id"), dtype=np.int32)
    dofs = np.asarray(pool.get("gll_global_dof"), dtype=np.int64)
    gathered = comm.allgather((ids, dofs, np.asarray(field)))
    np_value = field.shape[0]
    halo_width = np_value + 2
    last = np_value - 1
    global_dofs: dict[int, np.ndarray] = {}
    global_values: dict[int, np.ndarray] = {}
    for rank_ids, rank_dofs, rank_values in gathered:
        for le, gid_value in enumerate(np.asarray(rank_ids)):
            gid = int(gid_value)
            global_dofs[gid] = np.asarray(rank_dofs)[:, :, le]
            global_values[gid] = np.asarray(rank_values)[..., le]
    halo = np.zeros(
        (halo_width, halo_width, field.shape[2], len(ids)),
        dtype=np.float64,
        order="F",
    )
    diagonal = np.zeros((4, len(ids)), dtype=bool, order="F")
    side_points = {
        "east": tuple((last, j) for j in range(np_value)),
        "south": tuple((i, 0) for i in range(np_value)),
        "north": tuple((i, last) for i in range(np_value)),
        "west": tuple((0, j) for j in range(np_value)),
    }
    for le, gid_value in enumerate(ids):
        gid = int(gid_value)
        halo[1 : np_value + 1, 1 : np_value + 1, :, le] = field[..., le]
        neighbors: dict[str, int] = {}
        for side, points in side_points.items():
            edge = frozenset(int(global_dofs[gid][i, j]) for i, j in points)
            matches = []
            for other_gid, other_dofs in global_dofs.items():
                if other_gid == gid:
                    continue
                other_edges = (
                    frozenset(int(other_dofs[i, j]) for i, j in values)
                    for values in side_points.values()
                )
                if edge in other_edges:
                    matches.append(other_gid)
            if len(matches) != 1:
                raise RuntimeError(f"element {gid} {side} DG edge has {len(matches)} neighbors")
            neighbors[side] = matches[0]
        destinations = {
            "east": tuple((halo_width - 1, j + 1) for j in range(np_value)),
            "south": tuple((i + 1, 0) for i in range(np_value)),
            "north": tuple((i + 1, halo_width - 1) for i in range(np_value)),
            "west": tuple((0, j + 1) for j in range(np_value)),
        }
        for side, points in side_points.items():
            neighbor = neighbors[side]
            for (i, j), (di, dj) in zip(points, destinations[side]):
                dof = int(global_dofs[gid][i, j])
                location = np.argwhere(global_dofs[neighbor] == dof)[0]
                halo[di, dj, :, le] = global_values[neighbor][int(location[0]), int(location[1]), :]
        corners = (
            (0, 0, ("west", "south")),
            (last, 0, ("east", "south")),
            (0, last, ("west", "north")),
            (last, last, ("east", "north")),
        )
        for corner, (i, j, sides) in enumerate(corners):
            dof = int(global_dofs[gid][i, j])
            excluded = {gid, neighbors[sides[0]], neighbors[sides[1]]}
            remaining = [
                other_gid for other_gid, other_dofs in global_dofs.items()
                if other_gid not in excluded and np.any(other_dofs == dof)
            ]
            if remaining:
                neighbor = sorted(remaining)[0]
                location = np.argwhere(global_dofs[neighbor] == dof)[0]
                di, dj = (
                    0 if i == 0 else halo_width - 1,
                    0 if j == 0 else halo_width - 1,
                )
                halo[di, dj, :, le] = global_values[neighbor][int(location[0]), int(location[1]), :]
                diagonal[corner, le] = True
    return halo, diagonal


def tracer_time_levels(pool) -> tuple[int, int]:
    """Return HOMME's two alternating Qdp time levels as zero-based indices."""

    internal_step = int(pool.get("dynamics_internal_step"))
    qsplit = int(pool.get("dynamics_qsplit"))
    quotient = internal_step // qsplit
    return (0, 1) if quotient % 2 == 0 else (1, 0)


def update_time_levels(pool) -> None:
    """Apply ``TimeLevel_update(..., 'leapfrog')`` in source order."""

    nm1 = int(pool.get("dynamics_time_level_nm1"))
    n0 = int(pool.get("dynamics_time_level_n0"))
    np1 = int(pool.get("dynamics_time_level_np1"))
    pool.set("dynamics_time_level_np1", nm1)
    pool.set("dynamics_time_level_nm1", n0)
    pool.set("dynamics_time_level_n0", np1)
    pool.set("dynamics_internal_step", int(pool.get("dynamics_internal_step")) + 1)


def scale_physics_forcing(pool, backend=None) -> None:
    """Port the forcing normalization at the start of ``dyn_comp:stepon``.

    The PG3-to-GLL coupling boundary stores tracer mixing-ratio adjustments.
    HOMME converts those adjustments to dry-pressure mass tendencies before
    ``ApplyCAMForcing``.  Its FVM mass adjustments are divided by the same
    physics timestep.  These arrays remain Python-owned throughout.
    """

    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    qsize = pool.dimensions["qsize"]
    ntrac = pool.dimensions["ntrac"]
    np_value = pool.dimensions["np"]
    nc = pool.dimensions["nc"]
    n0 = int(pool.get("dynamics_time_level_n0"))
    qn0, _ = tracer_time_levels(pool)
    dtime = np.float64(pool.get("model_timestep"))
    reciprocal_timestep = np.float64(np.float64(1.0) / dtime)
    fq = pool.get("constituent_forcing")
    dp = pool.get("layer_pressure_thickness")

    if backend is not None:
        packed_forcing = np.empty(
            (np_value, np_value, nlev, qsize, nelem),
            dtype=np.float64,
            order="F",
        )
        packed_pressure = np.empty(
            (np_value, np_value, nlev, nelem),
            dtype=np.float64,
            order="F",
        )
        for le in range(nelem):
            packed_pressure[..., le] = dp[..., le, n0]
            for constituent in range(qsize):
                packed_forcing[:, :, :, constituent, le] = fq[
                    :, :, :, le, constituent
                ]
        backend.scale_tracer_forcing(
            reciprocal_timestep=reciprocal_timestep,
            forcing=packed_forcing,
            pressure_thickness=packed_pressure,
        )
        for le in range(nelem):
            for constituent in range(qsize):
                fq[:, :, :, le, constituent] = packed_forcing[
                    :, :, :, constituent, le
                ]
    else:
        # Preserve dyn_comp.F90 loop and expression order: ie,m,k,j,i and
        # (FQ*rec2dt)*dp.
        for le in range(nelem):
            for constituent in range(qsize):
                for level in range(nlev):
                    for j in range(np_value):
                        for i in range(np_value):
                            fq[i, j, level, le, constituent] = np.float64(
                                np.float64(
                                    fq[i, j, level, le, constituent]
                                    * reciprocal_timestep
                                )
                                * dp[i, j, level, le, n0]
                            )

    # The registry and Qdp storage use the same configured constituent order.
    active_species_order = range(qsize)
    qdp = pool.get("constituent_mass")
    fdp = pool.get("forcing_full_layer_pressure_thickness")
    for le in range(nelem):
        for level in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    pdel = np.float64(dp[i, j, level, le, n0])
                    for constituent in active_species_order:
                        pdel = np.float64(
                            pdel
                            + np.float64(
                                qdp[i, j, level, le, constituent, qn0]
                                + np.float64(
                                    fq[i, j, level, le, constituent] * dtime
                                )
                            )
                        )
                    fdp[i, j, level, le] = pdel

    fc = pool.get("fvm_constituent_mass_forcing")
    for le in range(nelem):
        for constituent in range(ntrac):
            for level in range(nlev):
                for j in range(nc):
                    for i in range(nc):
                        fc[i, j, level, le, constituent] = np.float64(
                            fc[i, j, level, le, constituent]
                            * reciprocal_timestep
                        )


def apply_cam_forcing(pool, backend=None, *, nsubstep: int = 1) -> None:
    """Port the SE/FVM v1 ``prim_advance_mod:ApplyCAMForcing`` path exactly."""

    if int(pool.get("dynamics_forcing_type")) != 2:
        raise RuntimeError("the fixed model requires se_ftype=2")
    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    qsize = pool.dimensions["qsize"]
    ntrac = pool.dimensions["ntrac"]
    np_value = pool.dimensions["np"]
    nc = pool.dimensions["nc"]
    nhc = pool.dimensions["nhc"]
    n0 = int(pool.get("dynamics_time_level_n0"))
    qn0, _ = tracer_time_levels(pool)
    dt_local = np.float64(pool.get("vertical_remap_timestep"))
    dt_local_tracer = dt_local
    dt_local_tracer_fvm = (
        np.float64(pool.get("model_timestep"))
        if nsubstep == 1
        else np.float64(0.0)
    )

    qdp = pool.get("constituent_mass")
    fq = pool.get("constituent_forcing")
    fvm_tracer = pool.get("fvm_tracer")
    fvm_dp = pool.get("fvm_layer_pressure_thickness")
    fc = pool.get("fvm_constituent_mass_forcing")
    temperature = pool.get("air_temperature")
    zonal = pool.get("zonal_wind")
    meridional = pool.get("meridional_wind")
    ft = pool.get("temperature_forcing")
    fu = pool.get("zonal_wind_forcing")
    fv = pool.get("meridional_wind_forcing")
    dry_dp = pool.get("layer_pressure_thickness")
    post_physics_dp = pool.get("forcing_full_layer_pressure_thickness")

    if backend is not None:
        packed_qdp = np.empty(
            (np_value, np_value, nlev, qsize, nelem),
            dtype=np.float64,
            order="F",
        )
        packed_forcing = np.empty_like(packed_qdp, order="F")
        for le in range(nelem):
            for constituent in range(qsize):
                packed_qdp[:, :, :, constituent, le] = qdp[
                    :, :, :, le, constituent, qn0
                ]
                packed_forcing[:, :, :, constituent, le] = fq[
                    :, :, :, le, constituent
                ]
        backend.apply_tracer_forcing(
            timestep=dt_local_tracer,
            qdp=packed_qdp,
            forcing=packed_forcing,
        )
        for le in range(nelem):
            for constituent in range(qsize):
                qdp[:, :, :, le, constituent, qn0] = packed_qdp[
                    :, :, :, constituent, le
                ]
    else:
        for le in range(nelem):
            # qsize tracer update, preserving q,k,j,i source order.
            for constituent in range(qsize):
                for level in range(nlev):
                    for j in range(np_value):
                        for i in range(np_value):
                            v1 = np.float64(
                                dt_local_tracer
                                * fq[i, j, level, le, constituent]
                            )
                            old = qdp[i, j, level, le, constituent, qn0]
                            if old + v1 < 0.0 and v1 < 0.0:
                                if old < 0.0:
                                    v1 = np.float64(0.0)
                                else:
                                    v1 = np.float64(-old)
                            qdp[
                                i, j, level, le, constituent, qn0
                            ] = np.float64(old + v1)

    # Only the compact GLL Qdp update above is delegated to the native
    # kernel.  CSLAM state and T/U/V remain Python-owned and must be updated
    # for both paths.
    for le in range(nelem):
        if dt_local_tracer_fvm > 0.0:
            for constituent in range(ntrac):
                for level in range(nlev):
                    for j in range(nc):
                        for i in range(nc):
                            hi, hj = i + nhc, j + nhc
                            tmp = np.float64(
                                np.float64(
                                    dt_local_tracer_fvm
                                    * fc[i, j, level, le, constituent]
                                )
                                / fvm_dp[hi, hj, level, le]
                            )
                            v1 = tmp
                            old = fvm_tracer[hi, hj, level, le, constituent]
                            if old + v1 < 0.0 and v1 < 0.0:
                                if old < 0.0:
                                    v1 = np.float64(0.0)
                                else:
                                    v1 = np.float64(-old)
                            fvm_tracer[hi, hj, level, le, constituent] = np.float64(
                                old + v1
                            )

        # ftype_conserve=1 is the HOMME default in this fixed configuration.
        for level in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    pdel = np.float64(dry_dp[i, j, level, le, n0])
                    for constituent in range(qsize):
                        pdel = np.float64(
                            pdel + qdp[i, j, level, le, constituent, qn0]
                        )
                    pdel = np.float64(
                        post_physics_dp[i, j, level, le] / pdel
                    )
                    temperature[i, j, level, le, n0] = np.float64(
                        temperature[i, j, level, le, n0]
                        + np.float64(
                            np.float64(dt_local * ft[i, j, level, le]) * pdel
                        )
                    )
                    zonal[i, j, level, le, n0] = np.float64(
                        zonal[i, j, level, le, n0]
                        + np.float64(
                            np.float64(dt_local * fu[i, j, level, le]) * pdel
                        )
                    )
                    meridional[i, j, level, le, n0] = np.float64(
                        meridional[i, j, level, le, n0]
                        + np.float64(
                            np.float64(dt_local * fv[i, j, level, le]) * pdel
                        )
                    )


def initialize_prim_step(pool) -> None:
    """Initialize the persistent derived arrays at a ``prim_step`` boundary."""

    n0 = int(pool.get("dynamics_time_level_n0"))
    pool.get("mean_horizontal_mass_flux")[...] = 0.0
    pool.get("vertical_pressure_velocity")[...] = 0.0
    pool.get("pressure_dissipation_average")[...] = 0.0
    pool.get("pressure_dissipation_biharmonic")[...] = 0.0
    pool.get("pressure_at_step_start")[...] = pool.get("layer_pressure_thickness")[..., n0]


def _thermodynamic_coefficients(pool, n0: int, qn0: int, backend=None):
    """Port the active-water subset of CAM's thermodynamic helper calls."""

    dp_dry = pool.get("layer_pressure_thickness")[..., n0]
    constituent_mass = pool.get("constituent_mass")[..., qn0]
    qdp = constituent_mass[..., : pool.dimensions["qsize"]]
    qwater = pool.get("rk_water_mixing_ratio")
    sum_water = np.empty_like(dp_dry, order="F")
    inv_cp = pool.get("rk_inverse_heat_capacity")
    kappa = pool.get("rk_kappa")
    rair = np.float64(pool.get("dry_air_gas_constant"))
    rh2o = np.float64(pool.get("water_vapor_gas_constant"))
    cpair = np.float64(pool.get("dry_air_specific_heat"))
    cpwv = np.float64(pool.get("water_vapor_specific_heat"))
    cpliq = np.float64(pool.get("liquid_water_specific_heat"))
    np_value = pool.dimensions["np"]
    thermodynamic_indices = water_constituent_indices(
        pool.constituent_names
    )
    qsize = pool.dimensions["qsize"]
    if backend is not None:
        backend.prepare_qwater(
            constituent_mass=constituent_mass,
            pressure_thickness=dp_dry,
            qwater=qwater,
            qsize=qsize,
        )
    else:
        qwater[...] = np.float64(0.0)
        for le in range(pool.dimensions["nelem_local"]):
            for k in range(pool.dimensions["pver"]):
                for j in range(np_value):
                    for i in range(np_value):
                        for constituent in range(qsize):
                            qwater[i, j, k, le, constituent] = np.float64(
                                qdp[i, j, k, le, constituent]
                                / dp_dry[i, j, k, le]
                            )
    for le in range(pool.dimensions["nelem_local"]):
        for k in range(pool.dimensions["pver"]):
            for j in range(np_value):
                for i in range(np_value):
                    species_sum = np.float64(1.0)
                    for constituent in range(qsize):
                        species_sum = np.float64(
                            species_sum + qwater[i, j, k, le, constituent]
                        )
                    sum_water[i, j, k, le] = species_sum
                    heat_sum = cpair
                    for constituent in range(qsize):
                        specific_heat = (
                            cpwv
                            if is_water_vapor(
                                pool.constituent_names[
                                    thermodynamic_indices[constituent]
                                ]
                            )
                            else cpliq
                        )
                        heat_sum = np.float64(
                            heat_sum
                            + np.float64(
                                specific_heat
                                * qwater[i, j, k, le, constituent]
                            )
                        )
                    inv_cp[i, j, k, le] = np.float64(species_sum / heat_sum)
    kappa[...] = np.float64(rair / cpair)
    return qwater, sum_water, inv_cp, kappa, rair, rh2o, cpair


def _hydrostatic_state(
    pool,
    n0: int,
    qwater: np.ndarray,
    sum_water: np.ndarray,
    rair: np.float64,
    rh2o: np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return virtual temperature, moist midpoint pressure, and geopotential."""

    temperature = pool.get("air_temperature")[..., n0]
    dp_dry = pool.get("layer_pressure_thickness")[..., n0]
    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    np_value = pool.dimensions["np"]
    thermodynamic_indices = water_constituent_indices(
        pool.constituent_names
    )
    vapor_indexes = tuple(
        packed_index
        for packed_index, constituent_index in enumerate(
            thermodynamic_indices
        )
        if is_water_vapor(
            pool.constituent_names[constituent_index]
        )
    )
    if len(vapor_indexes) > 1:
        raise ValueError("constituent registry contains multiple water vapors")
    vapor_index = vapor_indexes[0] if vapor_indexes else None
    virtual_temperature = np.empty_like(dp_dry, order="F")
    pressure = np.empty_like(dp_dry, order="F")
    geopotential = np.empty_like(dp_dry, order="F")
    dp_full = np.empty_like(dp_dry, order="F")
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(
        pool.get("reference_pressure")
    )
    for le in range(nelem):
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    qv = (
                        qwater[i, j, k, le, vapor_index]
                        if vapor_index is not None
                        else np.float64(0.0)
                    )
                    gas_sum = rair
                    gas_sum = np.float64(gas_sum + np.float64(rh2o * qv))
                    virtual_temperature[i, j, k, le] = np.float64(
                        np.float64(gas_sum * temperature[i, j, k, le])
                        / np.float64(rair * sum_water[i, j, k, le])
                    )
                    dp_full[i, j, k, le] = np.float64(
                        sum_water[i, j, k, le] * dp_dry[i, j, k, le]
                    )
        for j in range(np_value):
            for i in range(np_value):
                interfaces = np.empty(nlev + 1, dtype=np.float64)
                interfaces[0] = ptop
                for k in range(1, nlev + 1):
                    interfaces[k] = np.float64(
                        dp_full[i, j, k - 1, le] + interfaces[k - 1]
                    )
                for k in range(nlev):
                    pressure[i, j, k, le] = np.float64(
                        dp_full[i, j, k, le]
                        / np.float64(
                            np.log(interfaces[k + 1]) - np.log(interfaces[k])
                        )
                    )
                half_geopotential = pool.get("surface_geopotential_gll")[i, j, le]
                for k in range(nlev - 1, -1, -1):
                    rdry_tv = np.float64(
                        rair * virtual_temperature[i, j, k, le]
                    )
                    geopotential[i, j, k, le] = np.float64(
                        half_geopotential
                        + np.float64(
                            rdry_tv
                            * np.float64(
                                np.float64(1.0)
                                - np.float64(interfaces[k] / pressure[i, j, k, le])
                            )
                        )
                    )
                    half_geopotential = np.float64(
                        half_geopotential
                        + np.float64(
                            rdry_tv
                            * np.float64(
                                np.log(interfaces[k + 1]) - np.log(interfaces[k])
                            )
                        )
                    )
    return virtual_temperature, pressure, geopotential


def compute_and_apply_rhs(
    pool,
    comm,
    backend,
    *,
    output_level: int,
    base_level: int,
    rhs_level: int,
    dt2: np.float64,
    eta_average_weight: np.float64,
    qwater: np.ndarray,
    sum_water: np.ndarray,
    inv_cp: np.ndarray,
    kappa: np.ndarray,
    rair: np.float64,
    rh2o: np.float64,
    cpair: np.float64,
) -> None:
    """Port ``prim_advance_mod:compute_and_apply_rhs`` for configured dimensions."""

    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    np_value = pool.dimensions["np"]
    dvv = pool.get("gll_derivative")
    inverse_radius = np.float64(1.0) / np.float64(pool.get("earth_radius"))
    ps0 = np.float64(pool.get("reference_pressure"))
    temperature = pool.get("air_temperature")
    zonal = pool.get("zonal_wind")
    meridional = pool.get("meridional_wind")
    dp = pool.get("layer_pressure_thickness")
    tv, pressure, geopotential = _hydrostatic_state(
        pool, rhs_level, qwater, sum_water, rair, rh2o
    )
    pool.get("virtual_temperature")[...] = tv
    pool.get("pressure_midpoint_gll")[...] = pressure
    pool.get("rk_geopotential")[...] = geopotential

    mass_temperature = np.empty(
        (np_value, np_value, nlev, nelem), dtype=np.float64, order="F"
    )
    mass_zonal = np.empty_like(mass_temperature, order="F")
    mass_meridional = np.empty_like(mass_temperature, order="F")
    mass_dp = np.empty_like(mass_temperature, order="F")
    for le in range(nelem):
        dinv = pool.get("inverse_metric")[:, :, :, :, le]
        metric = pool.get("metric_derivative")[:, :, :, :, le]
        metdet = pool.get("metric_jacobian")[:, :, le]
        rmetdet = pool.get("inverse_metric_jacobian")[:, :, le]
        spectral_mass = pool.get("spectral_mass_matrix")[:, :, le]
        fcor = pool.get("coriolis_parameter")[:, :, le]
        div_dry = np.empty(
            (np_value, np_value, nlev), dtype=np.float64, order="F"
        )
        div_full = np.empty_like(div_dry, order="F")
        vgrad_pressure = np.empty_like(div_dry, order="F")
        vorticity = np.empty_like(div_dry, order="F")
        for k in range(nlev):
            pressure_gradient = _gradient_sphere(
                pressure[:, :, k, le], dvv, dinv, inverse_radius
            )
            dry_flux = np.empty(
                (np_value, np_value, 2), dtype=np.float64, order="F"
            )
            full_flux = np.empty_like(dry_flux, order="F")
            velocity = np.empty_like(dry_flux, order="F")
            for j in range(np_value):
                for i in range(np_value):
                    u = zonal[i, j, k, le, rhs_level]
                    v = meridional[i, j, k, le, rhs_level]
                    velocity[i, j, 0] = u
                    velocity[i, j, 1] = v
                    dry_flux[i, j, 0] = np.float64(u * dp[i, j, k, le, rhs_level])
                    dry_flux[i, j, 1] = np.float64(v * dp[i, j, k, le, rhs_level])
                    full_dp = np.float64(
                        sum_water[i, j, k, le] * dp[i, j, k, le, rhs_level]
                    )
                    full_flux[i, j, 0] = np.float64(u * full_dp)
                    full_flux[i, j, 1] = np.float64(v * full_dp)
                    vgrad_pressure[i, j, k] = np.float64(
                        np.float64(u * pressure_gradient[i, j, 0])
                        + np.float64(v * pressure_gradient[i, j, 1])
                    )
                    if eta_average_weight != 0.0:
                        pool.get("mean_horizontal_mass_flux")[i, j, 0, k, le] = np.float64(
                            pool.get("mean_horizontal_mass_flux")[i, j, 0, k, le]
                            + np.float64(eta_average_weight * dry_flux[i, j, 0])
                        )
                        pool.get("mean_horizontal_mass_flux")[i, j, 1, k, le] = np.float64(
                            pool.get("mean_horizontal_mass_flux")[i, j, 1, k, le]
                            + np.float64(eta_average_weight * dry_flux[i, j, 1])
                        )
            div_dry[:, :, k] = _divergence_sphere(
                dry_flux, dvv, dinv, metdet, rmetdet, inverse_radius
            )
            div_full[:, :, k] = _divergence_sphere(
                full_flux, dvv, dinv, metdet, rmetdet, inverse_radius
            )
            vorticity[:, :, k] = _vorticity_sphere(
                velocity, dvv, metric, rmetdet, inverse_radius
            )
        pool.get("rk_vorticity")[:, :, :, le] = vorticity
        pool.get("rk_dry_mass_flux_divergence")[:, :, :, le] = div_dry
        pool.get("rk_full_mass_flux_divergence")[:, :, :, le] = div_full
        pool.get("rk_horizontal_pressure_advection")[:, :, :, le] = vgrad_pressure

        native_zonal_tendency = np.empty(
            (np_value, np_value, nlev, 1), dtype=np.float64, order="F"
        )
        native_meridional_tendency = np.empty_like(native_zonal_tendency, order="F")
        backend.wind_tendency(
            inverse_radius=inverse_radius,
            reference_pressure=ps0,
            dry_specific_heat=cpair,
            derivative=dvv,
            inverse_metric=pool.get("inverse_metric")[:, :, :, :, le : le + 1],
            coriolis=pool.get("coriolis_parameter")[:, :, le : le + 1],
            zonal=zonal[:, :, :, le : le + 1, rhs_level],
            meridional=meridional[:, :, :, le : le + 1, rhs_level],
            virtual_temperature=tv[:, :, :, le : le + 1],
            pressure=pressure[:, :, :, le : le + 1],
            geopotential=geopotential[:, :, :, le : le + 1],
            kappa=kappa[:, :, :, le : le + 1],
            vorticity=vorticity[:, :, :, None],
            zonal_tendency=native_zonal_tendency,
            meridional_tendency=native_meridional_tendency,
        )

        omega_full = np.empty_like(div_full, order="F")
        cumulative = np.zeros(
            (np_value, np_value), dtype=np.float64, order="F"
        )
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    term = np.float64(-div_full[i, j, k])
                    omega_full[i, j, k] = np.float64(
                        cumulative[i, j]
                        + np.float64(np.float64(0.5) * term)
                        + vgrad_pressure[i, j, k]
                    )
                    cumulative[i, j] = np.float64(cumulative[i, j] + term)
                    if eta_average_weight != 0.0:
                        pool.get("vertical_pressure_velocity")[i, j, k, le] = np.float64(
                            pool.get("vertical_pressure_velocity")[i, j, k, le]
                            + np.float64(eta_average_weight * omega_full[i, j, k])
                        )

        for k in range(nlev):
            grad_temperature = _gradient_sphere(
                temperature[:, :, k, le, rhs_level], dvv, dinv, inverse_radius
            )
            for j in range(np_value):
                for i in range(np_value):
                    u = zonal[i, j, k, le, rhs_level]
                    v = meridional[i, j, k, le, rhs_level]
                    zonal_tendency = native_zonal_tendency[i, j, k, 0]
                    meridional_tendency = native_meridional_tendency[i, j, k, 0]
                    vgrad_temperature = np.float64(
                        np.float64(u * grad_temperature[i, j, 0])
                        + np.float64(v * grad_temperature[i, j, 1])
                    )
                    density_inverse = np.float64(
                        np.float64(rair * tv[i, j, k, le]) / pressure[i, j, k, le]
                    )
                    temperature_tendency = np.float64(
                        -vgrad_temperature
                        + np.float64(
                            np.float64(density_inverse * omega_full[i, j, k])
                            * inv_cp[i, j, k, le]
                        )
                    )
                    pool.get("rk_zonal_wind_tendency")[i, j, k, le] = zonal_tendency
                    pool.get("rk_meridional_wind_tendency")[i, j, k, le] = meridional_tendency
                    pool.get("rk_temperature_tendency")[i, j, k, le] = temperature_tendency
                    mass_temperature[i, j, k, le] = np.float64(
                        spectral_mass[i, j]
                        * np.float64(
                            temperature[i, j, k, le, base_level]
                            + np.float64(dt2 * temperature_tendency)
                        )
                    )
                    mass_zonal[i, j, k, le] = np.float64(
                        spectral_mass[i, j]
                        * np.float64(
                            zonal[i, j, k, le, base_level]
                            + np.float64(dt2 * zonal_tendency)
                        )
                    )
                    mass_meridional[i, j, k, le] = np.float64(
                        spectral_mass[i, j]
                        * np.float64(
                            meridional[i, j, k, le, base_level]
                            + np.float64(dt2 * meridional_tendency)
                        )
                    )
                    mass_dp[i, j, k, le] = np.float64(
                        spectral_mass[i, j]
                        * np.float64(
                            dp[i, j, k, le, base_level]
                            - np.float64(dt2 * div_dry[i, j, k])
                        )
                    )

        if eta_average_weight != 0.0:
            subflux = pool.get("subelement_mass_flux")
            nc = pool.dimensions["nc"]
            for k in range(nlev):
                contravariant = np.empty(
                    (np_value, np_value, 2),
                    dtype=np.float64,
                    order="F",
                )
                for j in range(np_value):
                    for i in range(np_value):
                        dry_u = np.float64(
                            zonal[i, j, k, le, rhs_level]
                            * dp[i, j, k, le, rhs_level]
                        )
                        dry_v = np.float64(
                            meridional[i, j, k, le, rhs_level]
                            * dp[i, j, k, le, rhs_level]
                        )
                        contravariant[i, j, 0] = np.float64(
                            np.float64(dinv[0, 0, i, j] * dry_u)
                            + np.float64(dinv[0, 1, i, j] * dry_v)
                        )
                        contravariant[i, j, 1] = np.float64(
                            np.float64(dinv[1, 0, i, j] * dry_u)
                            + np.float64(dinv[1, 1, i, j] * dry_v)
                        )
                flux = _subcell_div_fluxes(
                    pool, contravariant, metdet
                )
                for edge in range(4):
                    for j in range(nc):
                        for i in range(nc):
                            subflux[i, j, edge, k, le] = np.float64(
                                subflux[i, j, edge, k, le]
                                - np.float64(eta_average_weight * flux[i, j, edge])
                            )

    dg_halo, diagonal_corner = _dg_halo(pool, comm, mass_dp)
    packed = np.concatenate(
        (mass_temperature, mass_zonal, mass_meridional, mass_dp), axis=2
    )
    assembled = _edge_sum(pool, comm, packed)
    for le in range(nelem):
        inverse_mass = pool.get("inverse_spectral_mass_matrix")[:, :, le]
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    scale = inverse_mass[i, j]
                    temperature[i, j, k, le, output_level] = np.float64(
                        scale * assembled[i, j, k, le]
                    )
                    zonal[i, j, k, le, output_level] = np.float64(
                        scale * assembled[i, j, nlev + k, le]
                    )
                    meridional[i, j, k, le, output_level] = np.float64(
                        scale * assembled[i, j, 2 * nlev + k, le]
                    )
                    dp[i, j, k, le, output_level] = np.float64(
                        scale * assembled[i, j, 3 * nlev + k, le]
                    )

    if eta_average_weight != 0.0:
        subflux = pool.get("subelement_mass_flux")
        nc = pool.dimensions["nc"]
        halo_width = np_value + 2
        last = np_value - 1
        for le in range(nelem):
            spectral_mass = pool.get("spectral_mass_matrix")[:, :, le]
            inverse_mass = pool.get("inverse_spectral_mass_matrix")[:, :, le]
            metdet = pool.get("metric_jacobian")[:, :, le]
            flags = tuple(bool(diagonal_corner[corner, le]) for corner in range(4))
            for k in range(nlev):
                corners = np.empty(
                    (halo_width, halo_width), dtype=np.float64, order="F"
                )
                for j in range(halo_width):
                    for i in range(halo_width):
                        corners[i, j] = np.float64(dg_halo[i, j, k, le] / dt2)
                corner_flux = _distribute_corner_flux(corners, flags)
                corner_flux[0, 0, :] = np.float64(inverse_mass[0, 0]) * corner_flux[0, 0, :]
                corner_flux[1, 0, :] = np.float64(inverse_mass[last, 0]) * corner_flux[1, 0, :]
                corner_flux[0, 1, :] = np.float64(inverse_mass[0, last]) * corner_flux[0, 1, :]
                corner_flux[1, 1, :] = np.float64(inverse_mass[last, last]) * corner_flux[1, 1, :]
                dss = np.empty(
                    (np_value, np_value), dtype=np.float64, order="F"
                )
                for j in range(np_value):
                    for i in range(np_value):
                        stash = np.float64(mass_dp[i, j, k, le] / spectral_mass[i, j])
                        dss[i, j] = np.float64(
                            dp[i, j, k, le, output_level] - stash
                        )
                        dss[i, j] = np.float64(dss[i, j] / dt2)
                flux = _subcell_dss_fluxes(
                    pool, dss, metdet, corner_flux
                )
                for edge in range(4):
                    for j in range(nc):
                        for i in range(nc):
                            subflux[i, j, edge, k, le] = np.float64(
                                subflux[i, j, edge, k, le]
                                + np.float64(eta_average_weight * flux[i, j, edge])
                            )


def _hypervis_dss_update(
    pool,
    comm,
    *,
    level_count: int,
    level_offset: int,
    time_level: int,
    dt: np.float64,
    eta_weight: np.float64,
    inverse_cp: np.ndarray,
    temperature_tendency: np.ndarray,
    zonal_tendency: np.ndarray,
    meridional_tendency: np.ndarray,
    pressure_tendency: np.ndarray,
) -> None:
    """DSS hyperviscosity fields, update state, and close CSLAM fluxes."""

    nelem = pool.dimensions["nelem_local"]
    np_value = pool.dimensions["np"]
    nc = pool.dimensions["nc"]
    halo_width = np_value + 2
    last = np_value - 1
    temperature = pool.get("air_temperature")
    zonal = pool.get("zonal_wind")
    meridional = pool.get("meridional_wind")
    pressure = pool.get("layer_pressure_thickness")
    mass_pressure = np.empty(
        (np_value, np_value, level_count, nelem),
        dtype=np.float64,
        order="F",
    )
    for le in range(nelem):
        mass = pool.get("spectral_mass_matrix")[:, :, le]
        for local_k in range(level_count):
            k = level_offset + local_k
            for j in range(np_value):
                for i in range(np_value):
                    mass_pressure[i, j, local_k, le] = np.float64(
                        np.float64(pressure[i, j, k, le, time_level] * mass[i, j])
                        + np.float64(dt * pressure_tendency[i, j, local_k, le])
                    )

    dg_halo, diagonal_corner = _dg_halo(pool, comm, mass_pressure)
    packed = np.concatenate(
        (
            temperature_tendency,
            zonal_tendency,
            meridional_tendency,
            mass_pressure,
        ),
        axis=2,
    )
    assembled = _edge_sum(pool, comm, packed)
    for le in range(nelem):
        inverse_mass = pool.get("inverse_spectral_mass_matrix")[:, :, le]
        for local_k in range(level_count):
            k = level_offset + local_k
            for j in range(np_value):
                for i in range(np_value):
                    scale = inverse_mass[i, j]
                    temperature_tendency[i, j, local_k, le] = np.float64(
                        np.float64(dt * assembled[i, j, local_k, le]) * scale
                    )
                    zonal_tendency[i, j, local_k, le] = np.float64(
                        np.float64(dt * assembled[i, j, level_count + local_k, le])
                        * scale
                    )
                    meridional_tendency[i, j, local_k, le] = np.float64(
                        np.float64(dt * assembled[i, j, 2 * level_count + local_k, le])
                        * scale
                    )
                    pressure[i, j, k, le, time_level] = np.float64(
                        assembled[i, j, 3 * level_count + local_k, le] * scale
                    )

    subflux = pool.get("subelement_mass_flux")
    for le in range(nelem):
        mass = pool.get("spectral_mass_matrix")[:, :, le]
        inverse_mass = pool.get("inverse_spectral_mass_matrix")[:, :, le]
        metdet = pool.get("metric_jacobian")[:, :, le]
        flags = tuple(bool(diagonal_corner[corner, le]) for corner in range(4))
        for local_k in range(level_count):
            k = level_offset + local_k
            corners = np.empty(
                (halo_width, halo_width), dtype=np.float64, order="F"
            )
            for j in range(halo_width):
                for i in range(halo_width):
                    corners[i, j] = np.float64(dg_halo[i, j, local_k, le] / dt)
            corner_flux = _distribute_corner_flux(corners, flags)
            corner_flux[0, 0, :] = np.float64(inverse_mass[0, 0]) * corner_flux[0, 0, :]
            corner_flux[1, 0, :] = np.float64(inverse_mass[last, 0]) * corner_flux[1, 0, :]
            corner_flux[0, 1, :] = np.float64(inverse_mass[0, last]) * corner_flux[0, 1, :]
            corner_flux[1, 1, :] = np.float64(inverse_mass[last, last]) * corner_flux[1, 1, :]
            dss = np.empty(
                (np_value, np_value), dtype=np.float64, order="F"
            )
            for j in range(np_value):
                for i in range(np_value):
                    before_dss = np.float64(
                        mass_pressure[i, j, local_k, le] / mass[i, j]
                    )
                    dss[i, j] = np.float64(
                        pressure[i, j, k, le, time_level] - before_dss
                    )
                    dss[i, j] = np.float64(dss[i, j] / dt)
            flux = _subcell_dss_fluxes(
                pool, dss, metdet, corner_flux
            )
            for edge in range(4):
                for j in range(nc):
                    for i in range(nc):
                        subflux[i, j, edge, k, le] = np.float64(
                            subflux[i, j, edge, k, le]
                            + np.float64(eta_weight * flux[i, j, edge])
                        )

    for le in range(nelem):
        for local_k in range(level_count):
            k = level_offset + local_k
            for j in range(np_value):
                for i in range(np_value):
                    new_u = np.float64(
                        zonal[i, j, k, le, time_level]
                        + zonal_tendency[i, j, local_k, le]
                    )
                    new_v = np.float64(
                        meridional[i, j, k, le, time_level]
                        + meridional_tendency[i, j, local_k, le]
                    )
                    zonal[i, j, k, le, time_level] = new_u
                    meridional[i, j, k, le, time_level] = new_v
                    temperature[i, j, k, le, time_level] = np.float64(
                        temperature[i, j, k, le, time_level]
                        + temperature_tendency[i, j, local_k, le]
                    )
                    # Preserve CAM's source order: the pre-update velocity is
                    # reconstructed from the rounded updated velocity, rather
                    # than retained in a Python temporary.
                    old_u = np.float64(
                        new_u - zonal_tendency[i, j, local_k, le]
                    )
                    old_v = np.float64(
                        new_v - meridional_tendency[i, j, local_k, le]
                    )
                    heating = np.float64(
                        np.float64(0.5)
                        * np.float64(
                            np.float64(new_u * new_u)
                            + np.float64(new_v * new_v)
                            - np.float64(
                                np.float64(old_u * old_u) + np.float64(old_v * old_v)
                            )
                        )
                    )
                    temperature[i, j, k, le, time_level] = np.float64(
                        temperature[i, j, k, le, time_level]
                        - np.float64(heating * inverse_cp[i, j, k, le])
                    )


def advance_hyperviscosity(pool, comm, backend) -> None:
    """Port SE/FVM v1 biharmonic diffusion and its three-level sponge."""

    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    np_value = pool.dimensions["np"]
    nc = pool.dimensions["nc"]
    np1 = int(pool.get("dynamics_time_level_np1"))
    n0 = int(pool.get("dynamics_time_level_n0"))
    qn0, _ = tracer_time_levels(pool)
    _qwater, _sum_water, inverse_cp, _kappa, rair, _rh2o, cpair = (
        _thermodynamic_coefficients(pool, n0, qn0, backend)
    )
    inverse_radius = np.float64(1.0) / np.float64(pool.get("earth_radius"))
    dvv = pool.get("gll_derivative")
    dinv = pool.get("inverse_metric")
    metric = pool.get("metric_derivative")
    metinv = pool.get("inverse_metric_tensor")
    metdet = pool.get("metric_jacobian")
    rmetdet = pool.get("inverse_metric_jacobian")
    mass = pool.get("spectral_mass_matrix")
    weights = pool.get("gll_weight")
    reference_mass = np.empty(
        (np_value, np_value), dtype=np.float64, order="F"
    )
    for j in range(np_value):
        for i in range(np_value):
            reference_mass[i, j] = np.float64(weights[i] * weights[j])

    reference_dp = np.empty(
        (np_value, np_value, nlev, nelem), dtype=np.float64, order="F"
    )
    reference_temperature = np.empty_like(reference_dp, order="F")
    reference_surface_pressure = np.empty(
        (np_value, np_value, nelem), dtype=np.float64, order="F"
    )
    native_sponge_scale = np.empty(nlev, dtype=np.float64, order="F")
    backend.hypervis_reference(
        reference_pressure=pool.get("reference_pressure"),
        dry_air_gas_constant=rair,
        dry_air_specific_heat=cpair,
        gravity=pool.get("gravitational_acceleration"),
        reference_temperature=np.float64(288.0),
        lapse_rate=np.float64(0.0065),
        kappa=np.float64(rair / cpair),
        hybrid_a_interface=pool.get("hybrid_a_interface"),
        hybrid_b_interface=pool.get("hybrid_b_interface"),
        hybrid_a_midpoint=pool.get("hybrid_a_midpoint"),
        hybrid_b_midpoint=pool.get("hybrid_b_midpoint"),
        surface_geopotential=pool.get("surface_geopotential_gll"),
        pressure_thickness=reference_dp,
        temperature=reference_temperature,
        surface_pressure=reference_surface_pressure,
        sponge_scale=native_sponge_scale,
    )
    configured_sponge_scale = pool.get("sponge_viscosity_scale")
    if not np.array_equal(
        native_sponge_scale.view(np.uint64), configured_sponge_scale.view(np.uint64)
    ):
        raise RuntimeError("Python-owned sponge scale differs from the CAM compiler result")

    pressure_hyper = np.float64(pool.get("pressure_hyperviscosity"))
    temperature_hyper = np.float64(pool.get("temperature_hyperviscosity"))
    velocity_hyper = np.float64(pool.get("velocity_hyperviscosity"))
    divergence_hyper = np.float64(pool.get("divergence_hyperviscosity"))
    divergence_ratio = np.sqrt(
        np.float64(divergence_hyper / velocity_hyper)
    )
    subcycles = int(pool.get("hyperviscosity_subcycles"))
    eta = np.float64(np.float64(1.0) / np.float64(pool.get("dynamics_qsplit")))
    dt = np.float64(pool.get("dynamics_timestep")) / np.float64(subcycles)
    reciprocal_subcycles = np.float64(np.float64(1.0) / np.float64(subcycles))
    temperature_state = pool.get("air_temperature")
    pressure_state = pool.get("layer_pressure_thickness")
    zonal_state = pool.get("zonal_wind")
    meridional_state = pool.get("meridional_wind")

    for _subcycle in range(subcycles):
        scalar_temperature = np.empty_like(reference_temperature, order="F")
        scalar_pressure = np.empty_like(reference_dp, order="F")
        vector = np.empty(
            (np_value, np_value, 2, nlev, nelem),
            dtype=np.float64,
            order="F",
        )
        for le in range(nelem):
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        scalar_temperature[i, j, k, le] = np.float64(
                            temperature_state[i, j, k, le, np1]
                            - reference_temperature[i, j, k, le]
                        )
                        scalar_pressure[i, j, k, le] = np.float64(
                            pressure_state[i, j, k, le, np1]
                            - reference_dp[i, j, k, le]
                        )
                        vector[i, j, 0, k, le] = zonal_state[i, j, k, le, np1]
                        vector[i, j, 1, k, le] = meridional_state[i, j, k, le, np1]

        first_temperature = np.empty_like(scalar_temperature, order="F")
        first_pressure = np.empty_like(scalar_pressure, order="F")
        first_vector = np.empty_like(vector, order="F")
        backend.scalar_laplace_weak(
            inverse_radius=inverse_radius,
            derivative=dvv,
            inverse_metric=dinv,
            mass=mass,
            scalar=scalar_temperature,
            output=first_temperature,
        )
        backend.scalar_laplace_weak(
            inverse_radius=inverse_radius,
            derivative=dvv,
            inverse_metric=dinv,
            mass=mass,
            scalar=scalar_pressure,
            output=first_pressure,
        )
        backend.vector_laplace_weak(
            inverse_radius=inverse_radius,
            divergence_ratio=divergence_ratio,
            derivative=dvv,
            metric=metric,
            inverse_metric=dinv,
            inverse_metric_tensor=metinv,
            metric_jacobian=metdet,
            inverse_metric_jacobian=rmetdet,
            mass=mass,
            reference_mass=reference_mass,
            vector=vector,
            output=first_vector,
        )
        first_temperature = _edge_sum(pool, comm, first_temperature)
        first_pressure = _edge_sum(pool, comm, first_pressure)
        first_vector = _edge_sum(pool, comm, first_vector)

        normalized_temperature = np.empty_like(first_temperature, order="F")
        normalized_pressure = np.empty_like(first_pressure, order="F")
        normalized_vector = np.empty_like(first_vector, order="F")
        dpflux = np.empty(
            (nc, nc, 4, nlev, nelem), dtype=np.float64, order="F"
        )
        for le in range(nelem):
            inverse_mass = pool.get("inverse_spectral_mass_matrix")[:, :, le]
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        normalized_temperature[i, j, k, le] = np.float64(
                            inverse_mass[i, j] * first_temperature[i, j, k, le]
                        )
                        normalized_pressure[i, j, k, le] = np.float64(
                            inverse_mass[i, j] * first_pressure[i, j, k, le]
                        )
                        normalized_vector[i, j, 0, k, le] = np.float64(
                            inverse_mass[i, j] * first_vector[i, j, 0, k, le]
                        )
                        normalized_vector[i, j, 1, k, le] = np.float64(
                            inverse_mass[i, j] * first_vector[i, j, 1, k, le]
                        )
                dpflux[:, :, :, k, le] = _subcell_laplace_fluxes(
                    pool, normalized_pressure[:, :, k, le], le
                )

        second_temperature = np.empty_like(first_temperature, order="F")
        second_pressure = np.empty_like(first_pressure, order="F")
        second_vector = np.empty_like(first_vector, order="F")
        backend.scalar_laplace_weak(
            inverse_radius=inverse_radius,
            derivative=dvv,
            inverse_metric=dinv,
            mass=mass,
            scalar=normalized_temperature,
            output=second_temperature,
        )
        backend.scalar_laplace_weak(
            inverse_radius=inverse_radius,
            derivative=dvv,
            inverse_metric=dinv,
            mass=mass,
            scalar=normalized_pressure,
            output=second_pressure,
        )
        backend.vector_laplace_weak(
            inverse_radius=inverse_radius,
            divergence_ratio=divergence_ratio,
            derivative=dvv,
            metric=metric,
            inverse_metric=dinv,
            inverse_metric_tensor=metinv,
            metric_jacobian=metdet,
            inverse_metric_jacobian=rmetdet,
            mass=mass,
            reference_mass=reference_mass,
            vector=normalized_vector,
            output=second_vector,
        )
        for le in range(nelem):
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        pool.get("pressure_dissipation_average")[i, j, k, le] = np.float64(
                            pool.get("pressure_dissipation_average")[i, j, k, le]
                            + np.float64(
                                np.float64(reciprocal_subcycles * eta)
                                * pressure_state[i, j, k, le, np1]
                            )
                        )
                        pool.get("pressure_dissipation_biharmonic")[i, j, k, le] = np.float64(
                            pool.get("pressure_dissipation_biharmonic")[i, j, k, le]
                            + np.float64(
                                np.float64(reciprocal_subcycles * eta)
                                * second_pressure[i, j, k, le]
                            )
                        )
                        second_temperature[i, j, k, le] = np.float64(
                            -temperature_hyper * second_temperature[i, j, k, le]
                        )
                        second_pressure[i, j, k, le] = np.float64(
                            -pressure_hyper * second_pressure[i, j, k, le]
                        )
                        second_vector[i, j, 0, k, le] = np.float64(
                            -velocity_hyper * second_vector[i, j, 0, k, le]
                        )
                        second_vector[i, j, 1, k, le] = np.float64(
                            -velocity_hyper * second_vector[i, j, 1, k, le]
                        )
                for edge in range(4):
                    for j in range(nc):
                        for i in range(nc):
                            pool.get("subelement_mass_flux")[i, j, edge, k, le] = np.float64(
                                pool.get("subelement_mass_flux")[i, j, edge, k, le]
                                - np.float64(
                                    np.float64(
                                        np.float64(reciprocal_subcycles * eta)
                                        * pressure_hyper
                                    )
                                    * dpflux[i, j, edge, k, le]
                                )
                            )

        _hypervis_dss_update(
            pool,
            comm,
            level_count=nlev,
            level_offset=0,
            time_level=np1,
            dt=dt,
            eta_weight=np.float64(reciprocal_subcycles * eta),
            inverse_cp=inverse_cp,
            temperature_tendency=second_temperature,
            zonal_tendency=second_vector[:, :, 0],
            meridional_tendency=second_vector[:, :, 1],
            pressure_tendency=second_pressure,
        )

    sponge_levels = int(pool.get("sponge_level_count"))
    sponge_temperature = np.empty(
        (np_value, np_value, sponge_levels, nelem),
        dtype=np.float64,
        order="F",
    )
    sponge_pressure = np.empty_like(sponge_temperature, order="F")
    sponge_vector = np.empty(
        (np_value, np_value, 2, sponge_levels, nelem),
        dtype=np.float64,
        order="F",
    )
    for le in range(nelem):
        for k in range(sponge_levels):
            for j in range(np_value):
                for i in range(np_value):
                    sponge_temperature[i, j, k, le] = temperature_state[i, j, k, le, np1]
                    sponge_pressure[i, j, k, le] = pressure_state[i, j, k, le, np1]
                    sponge_vector[i, j, 0, k, le] = zonal_state[i, j, k, le, np1]
                    sponge_vector[i, j, 1, k, le] = meridional_state[i, j, k, le, np1]
    sponge_temperature_tendency = np.empty_like(sponge_temperature, order="F")
    sponge_pressure_tendency = np.empty_like(sponge_pressure, order="F")
    sponge_vector_tendency = np.empty_like(sponge_vector, order="F")
    backend.scalar_laplace_weak(
        inverse_radius=inverse_radius,
        derivative=dvv,
        inverse_metric=dinv,
        mass=mass,
        scalar=sponge_temperature,
        output=sponge_temperature_tendency,
    )
    backend.scalar_laplace_weak(
        inverse_radius=inverse_radius,
        derivative=dvv,
        inverse_metric=dinv,
        mass=mass,
        scalar=sponge_pressure,
        output=sponge_pressure_tendency,
    )
    backend.vector_laplace_weak(
        inverse_radius=inverse_radius,
        divergence_ratio=np.float64(1.0),
        derivative=dvv,
        metric=metric,
        inverse_metric=dinv,
        inverse_metric_tensor=metinv,
        metric_jacobian=metdet,
        inverse_metric_jacobian=rmetdet,
        mass=mass,
        reference_mass=reference_mass,
        vector=sponge_vector,
        output=sponge_vector_tendency,
    )
    sponge_scale = pool.get("sponge_viscosity_scale")
    nu_top = np.float64(pool.get("sponge_top_viscosity"))
    for le in range(nelem):
        for k in range(sponge_levels):
            coefficient = np.float64(sponge_scale[k] * nu_top)
            laplace_flux = _subcell_laplace_fluxes(
                pool, pressure_state[:, :, k, le, np1], le
            )
            for j in range(np_value):
                for i in range(np_value):
                    sponge_temperature_tendency[i, j, k, le] = np.float64(
                        coefficient * sponge_temperature_tendency[i, j, k, le]
                    )
                    sponge_pressure_tendency[i, j, k, le] = np.float64(
                        coefficient * sponge_pressure_tendency[i, j, k, le]
                    )
                    sponge_vector_tendency[i, j, 0, k, le] = np.float64(
                        coefficient * sponge_vector_tendency[i, j, 0, k, le]
                    )
                    sponge_vector_tendency[i, j, 1, k, le] = np.float64(
                        coefficient * sponge_vector_tendency[i, j, 1, k, le]
                    )
            for edge in range(4):
                for j in range(nc):
                    for i in range(nc):
                        pool.get("subelement_mass_flux")[i, j, edge, k, le] = np.float64(
                            pool.get("subelement_mass_flux")[i, j, edge, k, le]
                            + np.float64(
                                np.float64(eta * coefficient)
                                * laplace_flux[i, j, edge]
                            )
                        )
    _hypervis_dss_update(
        pool,
        comm,
        level_count=sponge_levels,
        level_offset=0,
        time_level=np1,
        dt=np.float64(pool.get("dynamics_timestep")),
        eta_weight=eta,
        inverse_cp=inverse_cp,
        temperature_tendency=sponge_temperature_tendency,
        zonal_tendency=sponge_vector_tendency[:, :, 0],
        meridional_tendency=sponge_vector_tendency[:, :, 1],
        pressure_tendency=sponge_pressure_tendency,
    )


def update_surface_dry_air_pressure(pool) -> None:
    """Update ``psdry`` after a primitive-equation dynamics advance."""

    np1 = int(pool.get("dynamics_time_level_np1"))
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(
        pool.get("reference_pressure")
    )
    pressure = pool.get("layer_pressure_thickness")
    surface = pool.get("surface_dry_air_pressure")
    np_value = pool.dimensions["np"]
    for le in range(pool.dimensions["nelem_local"]):
        for j in range(np_value):
            for i in range(np_value):
                value = ptop
                for k in range(pool.dimensions["pver"]):
                    value = np.float64(value + pressure[i, j, k, le, np1])
                surface[i, j, le] = value


def _neighbor_element_minmax(pool, comm, minimum, maximum) -> None:
    """Apply HOMME ``edgeSunpackMIN/MAX`` over side and corner neighbors."""

    ids = np.asarray(pool.get("global_element_id"), dtype=np.int32)
    dofs = np.asarray(pool.get("gll_global_dof"), dtype=np.int64)
    gathered = comm.allgather((ids, dofs, minimum, maximum))
    global_dofs = {}
    global_minimum = {}
    global_maximum = {}
    for rank_ids, rank_dofs, rank_minimum, rank_maximum in gathered:
        for le, gid_value in enumerate(np.asarray(rank_ids)):
            gid = int(gid_value)
            global_dofs[gid] = np.asarray(rank_dofs)[:, :, le]
            global_minimum[gid] = np.asarray(rank_minimum)[:, :, le]
            global_maximum[gid] = np.asarray(rank_maximum)[:, :, le]
    for le, gid_value in enumerate(ids):
        gid = int(gid_value)
        owned = frozenset(int(value) for value in global_dofs[gid].ravel())
        for other_gid in sorted(global_dofs):
            if other_gid == gid:
                continue
            other = frozenset(int(value) for value in global_dofs[other_gid].ravel())
            if owned.isdisjoint(other):
                continue
            minimum[:, :, le] = np.minimum(
                minimum[:, :, le], global_minimum[other_gid]
            )
            maximum[:, :, le] = np.maximum(
                maximum[:, :, le], global_maximum[other_gid]
            )
    minimum[...] = np.maximum(minimum, np.float64(0.0))


def _tracer_biharmonic(pool, comm, backend, scalar: np.ndarray) -> np.ndarray:
    """Return the weak biharmonic of all tracer fields."""

    nlev = pool.dimensions["pver"]
    nq = pool.dimensions["qsize"]
    nelem = pool.dimensions["nelem_local"]
    np_value = pool.dimensions["np"]
    packed = np.empty(
        (np_value, np_value, nlev * nq, nelem),
        dtype=np.float64,
        order="F",
    )
    for le in range(nelem):
        for q in range(nq):
            for k in range(nlev):
                packed[:, :, k + nlev * q, le] = scalar[:, :, k, q, le]
    first = np.empty_like(packed, order="F")
    common = dict(
        inverse_radius=np.float64(1.0) / np.float64(pool.get("earth_radius")),
        derivative=pool.get("gll_derivative"),
        inverse_metric=pool.get("inverse_metric"),
        mass=pool.get("spectral_mass_matrix"),
    )
    backend.scalar_laplace_weak(scalar=packed, output=first, **common)
    first = _edge_sum(pool, comm, first)
    normalized = np.empty_like(first, order="F")
    for le in range(nelem):
        inverse_mass = pool.get("inverse_spectral_mass_matrix")[:, :, le]
        for level in range(nlev * nq):
            for j in range(np_value):
                for i in range(np_value):
                    normalized[i, j, level, le] = np.float64(
                        inverse_mass[i, j] * first[i, j, level, le]
                    )
    second = np.empty_like(first, order="F")
    backend.scalar_laplace_weak(scalar=normalized, output=second, **common)
    result = np.empty_like(scalar, order="F")
    for le in range(nelem):
        for q in range(nq):
            for k in range(nlev):
                result[:, :, k, q, le] = second[:, :, k + nlev * q, le]
    return result


def _euler_tracer_stage(pool, comm, backend, *, source_level, output_level, dt, rhs_multiplier, dss_field) -> None:
    """Port one limiter-8 ``euler_step`` from SE tracer advection."""

    nlev = pool.dimensions["pver"]
    nq = pool.dimensions["qsize"]
    nelem = pool.dimensions["nelem_local"]
    np_value = pool.dimensions["np"]
    pressure_start = pool.get("pressure_at_step_start")
    divdp = pool.get("mass_flux_divergence")
    divdp_projected = pool.get("projected_mass_flux_divergence")
    qdp = pool.get("constituent_mass")
    vn0 = pool.get("mean_horizontal_mass_flux")
    minimum = pool.get("tracer_stage_minimum")
    maximum = pool.get("tracer_stage_maximum")
    stage_dp = np.empty_like(pressure_start, order="F")
    mixing = np.empty(
        (np_value, np_value, nlev, nq, nelem),
        dtype=np.float64,
        order="F",
    )
    for le in range(nelem):
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    stage_dp[i, j, k, le] = np.float64(
                        pressure_start[i, j, k, le]
                        - np.float64(
                            np.float64(rhs_multiplier) * np.float64(dt)
                            * divdp_projected[i, j, k, le]
                        )
                    )
                    for q in range(nq):
                        mixing[i, j, k, q, le] = np.float64(
                            qdp[i, j, k, le, q, source_level]
                            / stage_dp[i, j, k, le]
    )
    for le in range(nelem):
        for q in range(nq):
            for k in range(nlev):
                local_minimum = mixing[0, 0, k, q, le]
                local_maximum = local_minimum
                for j in range(np_value):
                    for i in range(np_value):
                        local_minimum = min(local_minimum, mixing[i, j, k, q, le])
                        local_maximum = max(local_maximum, mixing[i, j, k, q, le])
                if rhs_multiplier == 1:
                    minimum[k, q, le] = min(minimum[k, q, le], local_minimum)
                    maximum[k, q, le] = max(maximum[k, q, le], local_maximum)
                else:
                    minimum[k, q, le] = local_minimum
                    maximum[k, q, le] = local_maximum
    if rhs_multiplier in (0, 2):
        _neighbor_element_minmax(pool, comm, minimum, maximum)

    biharmonic = None
    if rhs_multiplier == 2:
        dp0 = np.empty(nlev, dtype=np.float64)
        hyai = pool.get("hybrid_a_interface")
        hybi = pool.get("hybrid_b_interface")
        ps0 = np.float64(pool.get("reference_pressure"))
        for k in range(nlev):
            dp0[k] = np.float64(
                np.float64(hyai[k + 1] - hyai[k]) * ps0
                + np.float64(hybi[k + 1] - hybi[k]) * ps0
            )
        scaled = np.empty_like(mixing, order="F")
        average = pool.get("pressure_dissipation_average")
        for le in range(nelem):
            for q in range(nq):
                for k in range(nlev):
                    for j in range(np_value):
                        for i in range(np_value):
                            scaled[i, j, k, q, le] = np.float64(
                                mixing[i, j, k, q, le]
                                * np.float64(average[i, j, k, le] / dp0[k])
                            )
        biharmonic = _tracer_biharmonic(pool, comm, backend, scaled)
        nuq = np.float64(pool.get("tracer_hyperviscosity"))
        mass = pool.get("spectral_mass_matrix")
        for le in range(nelem):
            for q in range(nq):
                for k in range(nlev):
                    for j in range(np_value):
                        for i in range(np_value):
                            biharmonic[i, j, k, q, le] = np.float64(
                                np.float64(
                                    -np.float64(3.0) * np.float64(dt) * nuq * dp0[k]
                                )
                                * biharmonic[i, j, k, q, le]
                                / mass[i, j, le]
                            )

    qtens = np.empty_like(mixing, order="F")
    dry_mass = np.empty_like(stage_dp, order="F")
    dvv = pool.get("gll_derivative")
    inverse_radius = np.float64(1.0) / np.float64(pool.get("earth_radius"))
    for le in range(nelem):
        dinv = pool.get("inverse_metric")[:, :, :, :, le]
        metdet = pool.get("metric_jacobian")[:, :, le]
        rmetdet = pool.get("inverse_metric_jacobian")[:, :, le]
        for k in range(nlev):
            dry_mass[:, :, k, le] = np.float64(0.0)
            for j in range(np_value):
                for i in range(np_value):
                    dry_mass[i, j, k, le] = np.float64(
                        stage_dp[i, j, k, le]
                        - np.float64(dt * divdp[i, j, k, le])
                        - np.float64(
                            np.float64(3.0) * np.float64(dt)
                            * np.float64(pool.get("tracer_hyperviscosity"))
                            * pool.get("pressure_dissipation_biharmonic")[i, j, k, le]
                            / pool.get("spectral_mass_matrix")[i, j, le]
                        )
                    ) if rhs_multiplier == 2 else np.float64(
                        stage_dp[i, j, k, le] - np.float64(
                            dt * divdp[i, j, k, le]
                        )
                    )
            for q in range(nq):
                flux = np.empty(
                    (np_value, np_value, 2),
                    dtype=np.float64,
                    order="F",
                )
                for j in range(np_value):
                    for i in range(np_value):
                        flux[i, j, 0] = np.float64(
                            np.float64(
                                vn0[i, j, 0, k, le]
                                / stage_dp[i, j, k, le]
                            )
                            * qdp[i, j, k, le, q, source_level]
                        )
                        flux[i, j, 1] = np.float64(
                            np.float64(
                                vn0[i, j, 1, k, le]
                                / stage_dp[i, j, k, le]
                            )
                            * qdp[i, j, k, le, q, source_level]
                        )
                divergence = _divergence_sphere(
                    flux,
                    dvv,
                    dinv,
                    metdet,
                    rmetdet,
                    inverse_radius,
                )
                for j in range(np_value):
                    for i in range(np_value):
                        qtens[i, j, k, q, le] = np.float64(
                            qdp[i, j, k, le, q, source_level]
                            - np.float64(dt * divergence[i, j])
                        )
                        if biharmonic is not None:
                            qtens[i, j, k, q, le] = np.float64(
                                qtens[i, j, k, q, le]
                                + biharmonic[i, j, k, q, le]
                            )
    minimum[...] = np.maximum(minimum, np.float64(0.0))
    backend.limiter_optim(
        tracer_mass=qtens,
        mass=pool.get("spectral_mass_matrix"),
        minimum=minimum,
        maximum=maximum,
        dry_mass=dry_mass,
    )

    extra = 1 if dss_field is not None else 0
    packed = np.empty(
        (np_value, np_value, nlev, nq + extra, nelem),
        dtype=np.float64,
        order="F",
    )
    for le in range(nelem):
        mass = pool.get("spectral_mass_matrix")[:, :, le]
        for q in range(nq):
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        packed[i, j, k, q, le] = np.float64(
                            mass[i, j] * qtens[i, j, k, q, le]
                        )
        if dss_field is not None:
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        packed[i, j, k, nq, le] = np.float64(
                            mass[i, j] * dss_field[i, j, k, le]
                        )
    assembled = _edge_sum(pool, comm, packed)
    for le in range(nelem):
        inverse_mass = pool.get("inverse_spectral_mass_matrix")[:, :, le]
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    for q in range(nq):
                        qdp[i, j, k, le, q, output_level] = np.float64(
                            inverse_mass[i, j] * assembled[i, j, k, q, le]
                        )
                    if dss_field is not None:
                        dss_field[i, j, k, le] = np.float64(
                            inverse_mass[i, j] * assembled[i, j, k, nq, le]
                        )
def advance_se_tracers(pool, comm, backend) -> None:
    """Run the configured limiter-8 three-stage SE tracer advection."""

    nlev = pool.dimensions["pver"]
    nelem = pool.dimensions["nelem_local"]
    np_value = pool.dimensions["np"]
    dvv = pool.get("gll_derivative")
    inverse_radius = np.float64(1.0) / np.float64(pool.get("earth_radius"))
    vn0 = pool.get("mean_horizontal_mass_flux")
    divdp = pool.get("mass_flux_divergence")
    for le in range(nelem):
        dinv = pool.get("inverse_metric")[:, :, :, :, le]
        metdet = pool.get("metric_jacobian")[:, :, le]
        rmetdet = pool.get("inverse_metric_jacobian")[:, :, le]
        for k in range(nlev):
            divdp[:, :, k, le] = _divergence_sphere(
                vn0[:, :, :, k, le], dvv, dinv, metdet, rmetdet, inverse_radius
            )
    pool.get("projected_mass_flux_divergence")[...] = divdp
    n0_qdp, np1_qdp = tracer_time_levels(pool)
    half_dt = np.float64(pool.get("dynamics_timestep")) / np.float64(2.0)
    _euler_tracer_stage(pool, comm, backend, source_level=n0_qdp, output_level=np1_qdp, dt=half_dt, rhs_multiplier=0, dss_field=pool.get("projected_mass_flux_divergence"))
    _euler_tracer_stage(pool, comm, backend, source_level=np1_qdp, output_level=np1_qdp, dt=half_dt, rhs_multiplier=1, dss_field=None)
    _euler_tracer_stage(pool, comm, backend, source_level=np1_qdp, output_level=np1_qdp, dt=half_dt, rhs_multiplier=2, dss_field=pool.get("vertical_pressure_velocity"))
    qdp = pool.get("constituent_mass")
    reciprocal_stage = np.float64(1.0) / np.float64(3.0)
    for le in range(nelem):
        for q in range(pool.dimensions["qsize"]):
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        qdp[i, j, k, le, q, np1_qdp] = np.float64(
                            reciprocal_stage
                            * np.float64(
                                qdp[i, j, k, le, q, n0_qdp]
                                + np.float64(
                                    np.float64(2.0)
                                    * qdp[i, j, k, le, q, np1_qdp]
                                )
                            )
                        )


def vertical_remap_se(pool, backend) -> None:
    """Remap the Python-owned SE state back to configured hybrid eta levels."""

    nlev = pool.dimensions["pver"]
    qsize = pool.dimensions["qsize"]
    thermodynamic_indices = water_constituent_indices(
        pool.constituent_names
    )
    np_value = pool.dimensions["np"]
    np1 = int(pool.get("dynamics_time_level_np1"))
    _n0_qdp, np1_qdp = tracer_time_levels(pool)
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(
        pool.get("reference_pressure")
    )
    hyai = pool.get("hybrid_a_interface")
    hybi = pool.get("hybrid_b_interface")
    ps0 = np.float64(pool.get("reference_pressure"))
    cp_liquid = np.float64(pool.get("liquid_water_specific_heat"))
    cp_vapor = np.float64(pool.get("water_vapor_specific_heat"))
    cp = tuple(
        (
            cp_vapor
            if is_water_vapor(
                pool.constituent_names[
                    thermodynamic_indices[constituent]
                ]
            )
            else cp_liquid
        )
        for constituent in range(qsize)
    )
    cpair = np.float64(pool.get("dry_air_specific_heat"))
    pressure = pool.get("layer_pressure_thickness")
    temperature = pool.get("air_temperature")
    zonal = pool.get("zonal_wind")
    meridional = pool.get("meridional_wind")
    qdp = pool.get("constituent_mass")
    psdry = pool.get("surface_dry_air_pressure")
    reference_pressure = np.empty(
        (
            np_value,
            np_value,
            nlev,
            pool.dimensions["nelem_local"],
        ),
        dtype=np.float64,
        order="F",
    )
    backend.reference_pressure_thickness(
        hybrid_a_interface=hyai,
        hybrid_b_interface=hybi,
        reference_pressure=ps0,
        source_pressure_thickness=pressure[:, :, :, :, np1],
        surface_dry_air_pressure=psdry,
        pressure_thickness=reference_pressure,
    )

    for le in range(pool.dimensions["nelem_local"]):
        source_dry = np.array(
            pressure[:, :, :, le, np1], dtype=np.float64, order="F", copy=True
        )
        target_dry = np.empty_like(source_dry, order="F")
        source_moist = np.empty_like(source_dry, order="F")
        target_moist = np.empty_like(source_dry, order="F")
        tracer = np.empty(
            (np_value, np_value, nlev, qsize),
            dtype=np.float64,
            order="F",
        )
        enthalpy = np.empty(
            (np_value, np_value, nlev, 1), dtype=np.float64, order="F"
        )
        heat_mass = np.empty_like(source_dry, order="F")
        wind = np.empty(
            (np_value, np_value, nlev, 1), dtype=np.float64, order="F"
        )

        target_dry[...] = reference_pressure[..., le]
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    source_moist[i, j, k] = source_dry[i, j, k]
                    heat = np.float64(cpair * source_dry[i, j, k])
                    for q in range(qsize):
                        value = qdp[i, j, k, le, q, np1_qdp]
                        tracer[i, j, k, q] = value
                        source_moist[i, j, k] = np.float64(
                            source_moist[i, j, k] + value
                        )
                        heat = np.float64(heat + np.float64(cp[q] * value))
                    enthalpy[i, j, k, 0] = np.float64(
                        heat * temperature[i, j, k, le, np1]
                    )

        backend.remap_fv3(
            field=tracer,
            source_pressure_thickness=source_dry,
            target_pressure_thickness=target_dry,
            pressure_top=ptop,
            identifier=0,
            mass_field=True,
        )
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    pressure[i, j, k, le, np1] = target_dry[i, j, k]
                    target_moist[i, j, k] = target_dry[i, j, k]
                    heat = np.float64(cpair * target_dry[i, j, k])
                    for q in range(qsize):
                        value = tracer[i, j, k, q]
                        qdp[i, j, k, le, q, np1_qdp] = value
                        target_moist[i, j, k] = np.float64(
                            target_moist[i, j, k] + value
                        )
                        heat = np.float64(heat + np.float64(cp[q] * value))
                    heat_mass[i, j, k] = heat

        backend.remap_fv3(
            field=enthalpy,
            source_pressure_thickness=source_dry,
            target_pressure_thickness=target_dry,
            pressure_top=ptop,
            identifier=1,
            mass_field=True,
        )
        for k in range(nlev):
            for j in range(np_value):
                for i in range(np_value):
                    temperature[i, j, k, le, np1] = np.float64(
                        enthalpy[i, j, k, 0] / heat_mass[i, j, k]
                    )

        wind[:, :, :, 0] = zonal[:, :, :, le, np1]
        backend.remap_fv3(
            field=wind,
            source_pressure_thickness=source_moist,
            target_pressure_thickness=target_moist,
            pressure_top=ptop,
            identifier=-1,
            mass_field=False,
        )
        zonal[:, :, :, le, np1] = wind[:, :, :, 0]
        wind[:, :, :, 0] = meridional[:, :, :, le, np1]
        backend.remap_fv3(
            field=wind,
            source_pressure_thickness=source_moist,
            target_pressure_thickness=target_moist,
            pressure_top=ptop,
            identifier=-1,
            mass_field=False,
        )
        meridional[:, :, :, le, np1] = wind[:, :, :, 0]


def vertical_remap_fvm(pool, backend) -> None:
    """Remap Python-owned FVM tracers to configured hybrid eta levels."""

    nlev = pool.dimensions["pver"]
    ntrac = pool.dimensions["ntrac"]
    nc = pool.dimensions["nc"]
    nhc = pool.dimensions["nhc"]
    interior = slice(nhc, nhc + nc)
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(
        pool.get("reference_pressure")
    )
    hyai = pool.get("hybrid_a_interface")
    hybi = pool.get("hybrid_b_interface")
    ps0 = np.float64(pool.get("reference_pressure"))
    pressure = pool.get("fvm_layer_pressure_thickness")
    tracer_state = pool.get("fvm_tracer")
    surface_pressure = pool.get("fvm_surface_dry_air_pressure")
    before = pool.pointer_records()

    for le in range(pool.dimensions["nelem_local"]):
        source = np.array(
            pressure[interior, interior, :, le],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        target = np.empty_like(source, order="F")
        tracer = np.array(
            tracer_state[interior, interior, :, le, :ntrac],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        for k in range(nlev):
            delta_a = np.float64(hyai[k + 1] - hyai[k])
            delta_b = np.float64(hybi[k + 1] - hybi[k])
            for j in range(nc):
                for i in range(nc):
                    target[i, j, k] = np.float64(
                        np.float64(delta_a * ps0)
                        + np.float64(delta_b * surface_pressure[i, j, le])
                    )
        backend.remap_fv3(
            field=tracer,
            source_pressure_thickness=source,
            target_pressure_thickness=target,
            pressure_top=ptop,
            identifier=0,
            mass_field=False,
            method=-9,
        )
        pressure[interior, interior, :, le] = target
        tracer_state[interior, interior, :, le, :ntrac] = tracer

    pool.assert_pointer_stability(before)


def advance_fvm_tracers(pool, comm, backend) -> None:
    """Run configured CSLAM transport with mpi4py-owned halos."""

    from .backend import FVMKernelConfig
    from .fvm_mapping import gather_physgrid_halo

    nelem = pool.dimensions["nelem_local"]
    nlev = pool.dimensions["pver"]
    ntrac = pool.dimensions["ntrac"]
    nc = pool.dimensions["nc"]
    halo_width = pool.dimensions["fvm_halo"]
    internal_width = pool.dimensions["fvm_internal"]
    kernel_config = FVMKernelConfig.from_pool(pool)
    nhc = kernel_config.nhc
    nhe = kernel_config.nhe
    interior = slice(nhc, nhc + nc)
    local = np.empty(
        (nc, nc, nlev, ntrac + 1, nelem),
        dtype=np.float64,
        order="F",
    )
    dp_state = pool.get("fvm_layer_pressure_thickness")
    tracer_state = pool.get("fvm_tracer")
    for le in range(nelem):
        local[:, :, :, 0, le] = dp_state[interior, interior, :, le]
        for q in range(ntrac):
            local[:, :, :, q + 1, le] = tracer_state[
                interior, interior, :, le, q
            ]
    halo = gather_physgrid_halo(pool, comm, local)
    schedule = pool.get("pg3_halo_global_column")
    for le in range(nelem):
        for j in range(halo_width):
            for i in range(halo_width):
                if schedule[i, j, le] <= 0:
                    halo[i, j, :, :, le] = np.float64(1.11e100)

    dt = np.float64(pool.get("dynamics_timestep"))
    ptop = np.float64(pool.get("hybrid_a_interface")[0]) * np.float64(
        pool.get("reference_pressure")
    )
    before = pool.pointer_records()
    prepared_dp = np.empty(
        (halo_width, halo_width, nlev, nelem),
        dtype=np.float64,
        order="F",
    )
    prepared_tracer = np.empty(
        (halo_width, halo_width, nlev, ntrac, nelem),
        dtype=np.float64,
        order="F",
    )
    transformed_flux = np.empty(
        (nc, nc, nlev, 4, nelem), dtype=np.float64, order="F"
    )

    # CAM converts the local mass flux to geometric swept displacement before
    # exchanging its halo.  Preserve that ordering: exchanging raw mass flux
    # gives correct element interiors but incorrect edge and corner updates.
    for le in range(nelem):
        inverse_reference = pool.get("fvm_inverse_reference_pressure_thickness")[:, le]
        dp = np.array(halo[:, :, :, 0, le], dtype=np.float64, order="F", copy=True)
        for k in range(nlev):
            dp[:, :, k] = np.float64(dp[:, :, k] * inverse_reference[k])
        # CAM's ghost exchange leaves the nonexistent cubed-sphere corner
        # cells at the fixed sentinel value.  They must not be normalized:
        # reconstruction uses their exact sentinel to recognize that the
        # corresponding cross-panel stencil does not exist.
        for j in range(halo_width):
            for i in range(halo_width):
                if schedule[i, j, le] <= 0:
                    dp[i, j, :] = np.float64(1.11e100)
        prepared_tracer[:, :, :, :, le] = halo[
            :, :, :, 1 : ntrac + 1, le
        ]

        swept_local = np.empty(
            (nc, nc, 4, nlev), dtype=np.float64, order="F"
        )
        for k in range(nlev):
            for edge in range(4):
                for j in range(nc):
                    for i in range(nc):
                        swept_local[i, j, edge, k] = np.float64(
                            np.float64(
                                dt
                                * pool.get("subelement_mass_flux")[
                                    i, j, edge, k, le
                                ]
                            )
                            * inverse_reference[k]
                        )
        backend.fvm_displacement(
            config=kernel_config,
            pressure_thickness=dp,
            swept_flux=swept_local,
            displacement_maximum=pool.get("fvm_displacement_maximum")[:, :, :, le],
            vertex_cartesian=pool.get("fvm_vertex_cartesian")[:, :, :, :, le],
        )
        # The transport kernel repeats CAM's interior normalization.  Give it
        # the original dimensional interior while retaining the already
        # exchanged, normalized halo used by the displacement calculation.
        dp[interior, interior, :] = local[:, :, :, 0, le]
        prepared_dp[:, :, :, le] = dp
        for k in range(nlev):
            for edge in range(4):
                transformed_flux[:, :, k, edge, le] = swept_local[:, :, edge, k]

    flux_halo = gather_physgrid_halo(pool, comm, transformed_flux)
    # The original ``se_flux`` allocation retains the edge-buffer sentinel in
    # nonexistent corner cells; ``ghost_flux_unpack`` subsequently zeros the
    # same cells inside the native kernel.
    for le in range(nelem):
        for j in range(halo_width):
            for i in range(halo_width):
                if schedule[i, j, le] <= 0:
                    flux_halo[i, j, :, :, le] = np.float64(1.11e100)
    for le in range(nelem):
        inverse_reference = pool.get("fvm_inverse_reference_pressure_thickness")[:, le]
        dp = np.array(
            prepared_dp[:, :, :, le], dtype=np.float64, order="F", copy=True
        )
        tracer = np.array(
            prepared_tracer[:, :, :, :, le],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        swept = np.empty(
            (internal_width, internal_width, 4, nlev),
            dtype=np.float64,
            order="F",
        )
        flux_start = nhc - nhe
        flux_end = flux_start + internal_width
        for k in range(nlev):
            for edge in range(4):
                swept[:, :, edge, k] = flux_halo[
                    flux_start:flux_end,
                    flux_start:flux_end,
                    k,
                    edge,
                    le,
                ]
        subflux = np.array(
            pool.get("subelement_mass_flux")[:, :, :, :, le],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        surface = np.array(
            pool.get("fvm_surface_dry_air_pressure")[:, :, le],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        backend.fvm_transport(
            config=kernel_config,
            dt=dt,
            subelement_flux=subflux,
            tracer=tracer,
            pressure_thickness=dp,
            surface_pressure=surface,
            swept_flux=swept,
            reference_pressure_thickness=pool.get("fvm_reference_pressure_thickness")[:, le],
            inverse_reference_pressure_thickness=inverse_reference,
            cell_area=pool.get("fvm_cell_area")[:, :, le],
            inverse_cell_area=pool.get("fvm_inverse_cell_area")[:, :, le],
            cube_boundary=pool.get("fvm_cube_boundary")[le],
            displacement_maximum=pool.get("fvm_displacement_maximum")[:, :, :, le],
            flux_vector=pool.get("fvm_flux_vector")[:, :, :, :, le],
            vertex_cartesian=pool.get("fvm_vertex_cartesian")[:, :, :, :, le],
            flux_orientation=pool.get("fvm_flux_orientation")[:, :, :, le],
            cell_indicator=pool.get("fvm_cell_indicator")[:, :, le],
            rotation_matrix=pool.get("fvm_rotation_matrix")[:, :, :, :, le],
            sphere_centroid=pool.get("fvm_sphere_centroid")[:, :, :, le],
            reconstruction_metric=pool.get("fvm_reconstruction_metric")[:, :, :, le],
            reconstruction_metric_integral=pool.get("fvm_reconstruction_metric_integral")[:, :, :, le],
            jx_min=pool.get("fvm_jx_min")[:, le],
            jx_max=pool.get("fvm_jx_max")[:, le],
            jy_min=pool.get("fvm_jy_min")[:, le],
            jy_max=pool.get("fvm_jy_max")[:, le],
            interpolation_base=pool.get("fvm_interpolation_base")[:, :, :, le],
            halo_interpolation_weight=pool.get("fvm_halo_interpolation_weight")[:, :, :, :, le],
            centroid_stretch=pool.get("fvm_centroid_stretch")[:, :, :, le],
            vertex_reconstruction_weight=pool.get("fvm_vertex_reconstruction_weight")[:, :, :, :, le],
        )
        pool.get("subelement_mass_flux")[:, :, :, :, le] = subflux
        dp_state[:, :, :, le] = dp
        tracer_state[:, :, :, le, :ntrac] = tracer
        pool.get("fvm_swept_flux")[:, :, :, :, le] = swept

    # ``run_consistent_se_cslam`` performs a second ``fill_halo_fvm`` after
    # swept-flux reconstruction and immediately before its large-Courant
    # correction.  The native device is deliberately MPI-free, so its first
    # stage stops at that exact boundary.  Exchange the newly updated mass
    # fields here, then let the second native stage preserve the original
    # Fortran arithmetic for the correction and mixing-ratio conversion.
    post_swept_local = np.empty(
        (nc, nc, nlev, ntrac + 1, nelem),
        dtype=np.float64,
        order="F",
    )
    for le in range(nelem):
        post_swept_local[:, :, :, 0, le] = dp_state[
            interior, interior, :, le
        ]
        for q in range(ntrac):
            post_swept_local[:, :, :, q + 1, le] = tracer_state[
                interior, interior, :, le, q
            ]
    post_swept_halo = gather_physgrid_halo(pool, comm, post_swept_local)
    for le in range(nelem):
        for j in range(halo_width):
            for i in range(halo_width):
                if schedule[i, j, le] <= 0:
                    post_swept_halo[i, j, :, :, le] = np.float64(
                        1.11e100
                    )
    for le in range(nelem):
        dp = np.array(
            dp_state[:, :, :, le],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        tracer = np.array(
            tracer_state[:, :, :, le, :ntrac],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        dp[...] = post_swept_halo[:, :, :, 0, le]
        tracer[...] = post_swept_halo[
            :, :, :, 1 : ntrac + 1, le
        ]
        swept = np.array(
            pool.get("fvm_swept_flux")[:, :, :, :, le],
            dtype=np.float64,
            order="F",
            copy=True,
        )
        surface = np.empty((nc, nc), dtype=np.float64, order="F")
        backend.fvm_large_courant_finalize(
            config=kernel_config,
            tracer=tracer,
            pressure_thickness=dp,
            swept_flux=swept,
            reference_pressure_thickness=pool.get(
                "fvm_reference_pressure_thickness"
            )[:, le],
            inverse_cell_area=pool.get("fvm_inverse_cell_area")[:, :, le],
            surface_pressure=surface,
            pressure_top=ptop,
        )
        dp_state[:, :, :, le] = dp
        tracer_state[:, :, :, le, :ntrac] = tracer
        pool.get("fvm_swept_flux")[:, :, :, :, le] = swept
        pool.get("subelement_mass_flux")[:, :, :, :, le] = 0.0
        pool.get("fvm_surface_dry_air_pressure")[:, :, le] = surface
    pool.assert_pointer_stability(before)


def compute_final_omega(pool, comm) -> None:
    """Compute the post-remap omega diagnostic on the final time levels."""

    from .dynamics import initialize_vertical_pressure_velocity

    _n0_qdp, np1_qdp = tracer_time_levels(pool)
    initialize_vertical_pressure_velocity(
        pool,
        comm,
        time_level=int(pool.get("dynamics_time_level_np1")),
        q_time_level=np1_qdp,
        apply_hyperviscosity=True,
    )


def prim_advance_type4_rk(pool, comm, backend) -> None:
    """Run the five RK stages, stopping before horizontal diffusion."""

    if int(pool.get("dynamics_timestep_type")) != 4:
        raise RuntimeError("the fixed model requires SE timestep type 4")
    nm1 = int(pool.get("dynamics_time_level_nm1"))
    n0 = int(pool.get("dynamics_time_level_n0"))
    np1 = int(pool.get("dynamics_time_level_np1"))
    qn0, _ = tracer_time_levels(pool)
    np_value = pool.dimensions["np"]
    qwater, sum_water, inv_cp, kappa, rair, rh2o, cpair = _thermodynamic_coefficients(
        pool, n0, qn0, backend
    )
    dt = np.float64(pool.get("dynamics_timestep"))
    common = dict(
        pool=pool,
        comm=comm,
        backend=backend,
        qwater=qwater,
        sum_water=sum_water,
        inv_cp=inv_cp,
        kappa=kappa,
        rair=rair,
        rh2o=rh2o,
        cpair=cpair,
    )
    compute_and_apply_rhs(output_level=nm1, base_level=n0, rhs_level=n0, dt2=np.float64(dt / np.float64(5.0)), eta_average_weight=np.float64(0.25), **common)
    compute_and_apply_rhs(output_level=np1, base_level=n0, rhs_level=nm1, dt2=np.float64(dt / np.float64(5.0)), eta_average_weight=np.float64(0.0), **common)
    compute_and_apply_rhs(output_level=np1, base_level=n0, rhs_level=np1, dt2=np.float64(dt / np.float64(3.0)), eta_average_weight=np.float64(0.0), **common)
    compute_and_apply_rhs(output_level=np1, base_level=n0, rhs_level=np1, dt2=np.float64(np.float64(2.0) * dt / np.float64(3.0)), eta_average_weight=np.float64(0.0), **common)
    for le in range(pool.dimensions["nelem_local"]):
        for k in range(pool.dimensions["pver"]):
            for j in range(np_value):
                for i in range(np_value):
                    pool.get("zonal_wind")[i, j, k, le, nm1] = np.float64(
                        np.float64(np.float64(5.0) * pool.get("zonal_wind")[i, j, k, le, nm1])
                        - pool.get("zonal_wind")[i, j, k, le, n0]
                    ) / np.float64(4.0)
                    pool.get("meridional_wind")[i, j, k, le, nm1] = np.float64(
                        np.float64(np.float64(5.0) * pool.get("meridional_wind")[i, j, k, le, nm1])
                        - pool.get("meridional_wind")[i, j, k, le, n0]
                    ) / np.float64(4.0)
                    pool.get("air_temperature")[i, j, k, le, nm1] = np.float64(
                        np.float64(np.float64(5.0) * pool.get("air_temperature")[i, j, k, le, nm1])
                        - pool.get("air_temperature")[i, j, k, le, n0]
                    ) / np.float64(4.0)
                    pool.get("layer_pressure_thickness")[i, j, k, le, nm1] = np.float64(
                        np.float64(np.float64(5.0) * pool.get("layer_pressure_thickness")[i, j, k, le, nm1])
                        - pool.get("layer_pressure_thickness")[i, j, k, le, n0]
                    ) / np.float64(4.0)
    compute_and_apply_rhs(output_level=np1, base_level=nm1, rhs_level=np1, dt2=np.float64(np.float64(3.0) * dt / np.float64(4.0)), eta_average_weight=np.float64(0.75), **common)


def prim_advance_first_rhs(pool, comm, backend) -> None:
    """Execute only the first type-4 RK RHS for boundary validation."""

    if int(pool.get("dynamics_timestep_type")) != 4:
        raise RuntimeError("the fixed model requires SE timestep type 4")
    nm1 = int(pool.get("dynamics_time_level_nm1"))
    n0 = int(pool.get("dynamics_time_level_n0"))
    qn0, _ = tracer_time_levels(pool)
    qwater, sum_water, inv_cp, kappa, rair, rh2o, cpair = _thermodynamic_coefficients(
        pool, n0, qn0, backend
    )
    compute_and_apply_rhs(
        pool,
        comm,
        backend,
        output_level=nm1,
        base_level=n0,
        rhs_level=n0,
        dt2=np.float64(np.float64(pool.get("dynamics_timestep")) / np.float64(5.0)),
        eta_average_weight=np.float64(0.25),
        qwater=qwater,
        sum_water=sum_water,
        inv_cp=inv_cp,
        kappa=kappa,
        rair=rair,
        rh2o=rh2o,
        cpair=cpair,
    )
