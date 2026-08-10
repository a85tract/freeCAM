"""Startup diagnostics and rank-local SE differential kernels.

Some routines in this module are transcribed from CESM/CAM Fortran sources
and preserve their upstream expression order; each is marked with a ``Port
...`` docstring naming its upstream routine.  Those routines are
Copyright (c) 2017, University Corporation for Atmospheric Research (UCAR)
and are redistributed under the BSD 3-Clause license in
LICENSES/UCAR-CESM-BSD-3-Clause.txt.  See NOTICE section 2.
"""

from __future__ import annotations

import numpy as np


def _edge_sum(pool, comm, field: np.ndarray) -> np.ndarray:
    """Reproduce ``edgeVunpack``'s deterministic E/S/N/W/corner sum order.

    A dense MPI reduction is numerically equivalent, but its rank-ordered
    reduction tree does not reproduce HOMME at element corners.  The fixed
    ne3 runtime is small enough to exchange the rank-local element slabs with
    mpi4py and then perform the additions in the source routine's order.
    """

    element_ids = np.asarray(pool.get("global_element_id"), dtype=np.int32)
    local_dofs = np.asarray(pool.get("gll_global_dof"), dtype=np.int64)
    gathered = comm.allgather((element_ids, local_dofs, np.asarray(field)))
    global_dofs: dict[int, np.ndarray] = {}
    global_fields: dict[int, np.ndarray] = {}
    for rank_ids, rank_dofs, rank_field in gathered:
        for le, gid_value in enumerate(np.asarray(rank_ids)):
            gid = int(gid_value)
            global_dofs[gid] = np.asarray(rank_dofs)[:, :, le]
            global_fields[gid] = np.asarray(rank_field)[..., le]

    np_value = local_dofs.shape[0]
    last = np_value - 1
    side_indices = {
        "east": ((last, j) for j in range(np_value)),
        "south": ((i, 0) for i in range(np_value)),
        "north": ((i, last) for i in range(np_value)),
        "west": ((0, j) for j in range(np_value)),
    }
    # Materialize the generators because each element needs all four lists.
    side_indices = {name: tuple(indices) for name, indices in side_indices.items()}
    side_order = ("east", "south", "north", "west")
    side_sets = {
        gid: {
            name: frozenset(int(dofs[i, j]) for i, j in indices)
            for name, indices in side_indices.items()
        }
        for gid, dofs in global_dofs.items()
    }

    result = np.array(field, dtype=np.float64, order="F", copy=True)
    for le, gid_value in enumerate(element_ids):
        gid = int(gid_value)
        dofs = global_dofs[gid]
        neighbors: dict[str, int] = {}
        for name in side_order:
            edge_dofs = side_sets[gid][name]
            matches = [
                other_gid
                for other_gid, other_sides in side_sets.items()
                if other_gid != gid and edge_dofs in other_sides.values()
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"element {gid} {name} edge has {len(matches)} neighbors"
                )
            neighbors[name] = matches[0]

        for j in range(np_value):
            for i in range(np_value):
                dof = int(dofs[i, j])
                value = result[i, j, ..., le]
                added_gids = {gid}
                for name in side_order:
                    on_side = (
                        (name == "east" and i == last)
                        or (name == "south" and j == 0)
                        or (name == "north" and j == last)
                        or (name == "west" and i == 0)
                    )
                    if not on_side:
                        continue
                    neighbor_gid = neighbors[name]
                    neighbor_dofs = global_dofs[neighbor_gid]
                    locations = np.argwhere(neighbor_dofs == dof)
                    if locations.shape != (1, 2):
                        raise RuntimeError(
                            f"shared dof {dof} not unique in edge neighbor {neighbor_gid}"
                        )
                    ni, nj = (int(locations[0, 0]), int(locations[0, 1]))
                    value = np.float64(
                        value + global_fields[neighbor_gid][ni, nj, ...]
                    )
                    added_gids.add(neighbor_gid)

                # edgeVunpack adds corner buffers after all four edge buffers.
                if i in (0, last) and j in (0, last):
                    for corner_gid in sorted(global_dofs):
                        if corner_gid in added_gids:
                            continue
                        locations = np.argwhere(global_dofs[corner_gid] == dof)
                        if locations.shape == (1, 2):
                            ni, nj = (int(locations[0, 0]), int(locations[0, 1]))
                            value = np.float64(
                                value + global_fields[corner_gid][ni, nj, ...]
                            )
                result[i, j, ..., le] = value
    return result


def assemble_inverse_spectral_mass(pool, comm) -> None:
    """Build HOMME ``rspheremp`` from the global assembled mass matrix."""

    mass = pool.get("spectral_mass_matrix")
    assembled = _edge_sum(pool, comm, mass[:, :, np.newaxis, :])[:, :, 0, :]
    inverse = pool.get("inverse_spectral_mass_matrix")
    np_value = pool.dimensions["np"]
    for le in range(pool.dimensions["nelem_local"]):
        for j in range(np_value):
            for i in range(np_value):
                inverse[i, j, le] = np.float64(1.0) / assembled[i, j, le]


def _source_dvv(nodes: np.ndarray) -> np.ndarray:
    """Reproduce ``derivative_mod:dvvinit`` for configured GLL nodes."""

    count = len(nodes)
    leg = np.empty(count, dtype=np.float64)
    for i in range(count):
        x = np.float64(nodes[i])
        p_2 = np.float64(1.0)
        p_3 = x
        for k in range(2, count):
            p_1 = p_2
            p_2 = p_3
            p_3 = np.float64(
                np.float64(np.float64(2 * k - 1) * x * p_2)
                - np.float64(np.float64(k - 1) * p_1)
            ) / np.float64(k)
        leg[i] = p_3
    dvv = np.zeros((count, count), dtype=np.float64, order="F")
    for j in range(count):
        for i in range(count):
            if i != j:
                dvv[j, i] = (
                    np.float64(1.0) / np.float64(nodes[i] - nodes[j])
                ) * np.float64(leg[i] / leg[j])
    endpoint = np.float64(count * (count - 1)) / np.float64(4.0)
    dvv[count - 1, count - 1] = endpoint
    dvv[0, 0] = -endpoint
    return dvv


def _gradient_sphere(
    scalar: np.ndarray,
    dvv: np.ndarray,
    inverse_metric: np.ndarray,
    inverse_radius: np.float64,
) -> np.ndarray:
    count = scalar.shape[0]
    v1 = np.empty((count, count), dtype=np.float64, order="F")
    v2 = np.empty((count, count), dtype=np.float64, order="F")
    result = np.empty((count, count, 2), dtype=np.float64, order="F")
    for j in range(count):
        for l in range(count):
            dsdx = np.float64(0.0)
            dsdy = np.float64(0.0)
            for i in range(count):
                dsdx = np.float64(dsdx + np.float64(dvv[i, l] * scalar[i, j]))
                dsdy = np.float64(dsdy + np.float64(dvv[i, l] * scalar[j, i]))
            v1[l, j] = np.float64(dsdx * inverse_radius)
            v2[j, l] = np.float64(dsdy * inverse_radius)
    for j in range(count):
        for i in range(count):
            result[i, j, 0] = np.float64(
                np.float64(inverse_metric[0, 0, i, j] * v1[i, j])
                + np.float64(inverse_metric[1, 0, i, j] * v2[i, j])
            )
            result[i, j, 1] = np.float64(
                np.float64(inverse_metric[0, 1, i, j] * v1[i, j])
                + np.float64(inverse_metric[1, 1, i, j] * v2[i, j])
            )
    return result


def _divergence_sphere(
    vector: np.ndarray,
    dvv: np.ndarray,
    inverse_metric: np.ndarray,
    metdet: np.ndarray,
    inverse_metdet: np.ndarray,
    inverse_radius: np.float64,
) -> np.ndarray:
    count = vector.shape[0]
    gv = np.empty((count, count, 2), dtype=np.float64, order="F")
    temp = np.empty((count, count), dtype=np.float64, order="F")
    result = np.empty((count, count), dtype=np.float64, order="F")
    for j in range(count):
        for i in range(count):
            gv[i, j, 0] = np.float64(metdet[i, j] * np.float64(
                np.float64(inverse_metric[0, 0, i, j] * vector[i, j, 0])
                + np.float64(inverse_metric[0, 1, i, j] * vector[i, j, 1])
            ))
            gv[i, j, 1] = np.float64(metdet[i, j] * np.float64(
                np.float64(inverse_metric[1, 0, i, j] * vector[i, j, 0])
                + np.float64(inverse_metric[1, 1, i, j] * vector[i, j, 1])
            ))
    for j in range(count):
        for l in range(count):
            dudx = np.float64(0.0)
            dvdy = np.float64(0.0)
            for i in range(count):
                dudx = np.float64(dudx + np.float64(dvv[i, l] * gv[i, j, 0]))
                dvdy = np.float64(dvdy + np.float64(dvv[i, l] * gv[j, i, 1]))
            result[l, j] = dudx
            temp[j, l] = dvdy
    for j in range(count):
        for i in range(count):
            result[i, j] = np.float64(
                np.float64(result[i, j] + temp[i, j])
                * np.float64(inverse_metdet[i, j] * inverse_radius)
            )
    return result


def _vorticity_sphere(
    vector: np.ndarray,
    dvv: np.ndarray,
    metric: np.ndarray,
    inverse_metdet: np.ndarray,
    inverse_radius: np.float64,
) -> np.ndarray:
    """Scalar-order equivalent of ``derivative_mod:vorticity_sphere``."""

    count = vector.shape[0]
    covariant_1 = np.empty((count, count), dtype=np.float64, order="F")
    covariant_2 = np.empty((count, count), dtype=np.float64, order="F")
    temporary = np.empty((count, count), dtype=np.float64, order="F")
    result = np.empty((count, count), dtype=np.float64, order="F")
    for j in range(count):
        for i in range(count):
            v1 = vector[i, j, 0]
            v2 = vector[i, j, 1]
            covariant_1[i, j] = np.float64(
                np.float64(metric[0, 0, i, j] * v1)
                + np.float64(metric[1, 0, i, j] * v2)
            )
            covariant_2[i, j] = np.float64(
                np.float64(metric[0, 1, i, j] * v1)
                + np.float64(metric[1, 1, i, j] * v2)
            )
    for j in range(count):
        for l in range(count):
            dvdx = np.float64(0.0)
            dudy = np.float64(0.0)
            for i in range(count):
                dvdx = np.float64(dvdx + np.float64(dvv[i, l] * covariant_2[i, j]))
                dudy = np.float64(dudy + np.float64(dvv[i, l] * covariant_1[j, i]))
            result[l, j] = dvdx
            temporary[j, l] = dudy
    for j in range(count):
        for i in range(count):
            result[i, j] = np.float64(
                np.float64(result[i, j] - temporary[i, j])
                * np.float64(inverse_metdet[i, j] * inverse_radius)
            )
    return result


def _divergence_sphere_weak(
    vector: np.ndarray,
    dvv: np.ndarray,
    inverse_metric: np.ndarray,
    mass: np.ndarray,
    inverse_radius: np.float64,
) -> np.ndarray:
    count = vector.shape[0]
    contra = np.empty((count, count, 2), dtype=np.float64, order="F")
    result = np.empty((count, count), dtype=np.float64, order="F")
    for j in range(count):
        for i in range(count):
            contra[i, j, 0] = np.float64(
                np.float64(inverse_metric[0, 0, i, j] * vector[i, j, 0])
                + np.float64(inverse_metric[0, 1, i, j] * vector[i, j, 1])
            )
            contra[i, j, 1] = np.float64(
                np.float64(inverse_metric[1, 0, i, j] * vector[i, j, 0])
                + np.float64(inverse_metric[1, 1, i, j] * vector[i, j, 1])
            )
    for n in range(count):
        for m in range(count):
            value = np.float64(0.0)
            for j in range(count):
                term = np.float64(
                    np.float64(mass[j, n] * contra[j, n, 0] * dvv[m, j])
                    + np.float64(mass[m, j] * contra[m, j, 1] * dvv[n, j])
                )
                value = np.float64(value - np.float64(term * inverse_radius))
            result[m, n] = value
    return result


def _laplace_sphere_weak(
    scalar: np.ndarray,
    dvv: np.ndarray,
    inverse_metric: np.ndarray,
    mass: np.ndarray,
    inverse_radius: np.float64,
) -> np.ndarray:
    gradient = _gradient_sphere(scalar, dvv, inverse_metric, inverse_radius)
    return _divergence_sphere_weak(gradient, dvv, inverse_metric, mass, inverse_radius)


def _dss(pool, comm, field: np.ndarray, *, multiply_mass: bool, divide_mass: bool) -> np.ndarray:
    """Sum a ``(np,np,nlev,nelem)`` field by Python-owned global DOF."""

    mass = pool.get("spectral_mass_matrix")
    inverse_mass = pool.get("inverse_spectral_mass_matrix")
    packed = np.empty_like(field, order="F")
    np_value = field.shape[0]
    for le in range(field.shape[3]):
        for k in range(field.shape[2]):
            for j in range(np_value):
                for i in range(np_value):
                    value = field[i, j, k, le]
                    packed[i, j, k, le] = (
                        np.float64(mass[i, j, le] * value)
                        if multiply_mass
                        else value
                    )
    result = _edge_sum(pool, comm, packed)
    if divide_mass:
        for le in range(field.shape[3]):
            for k in range(field.shape[2]):
                for j in range(np_value):
                    for i in range(np_value):
                        result[i, j, k, le] = np.float64(
                            inverse_mass[i, j, le] * result[i, j, k, le]
                        )
    return result


def initialize_vertical_pressure_velocity(
    pool,
    comm,
    time_level: int = 0,
    *,
    q_time_level: int | None = None,
    apply_hyperviscosity: bool = True,
) -> None:
    """Port ``compute_omega`` including DSS and its startup biharmonic filter."""

    dvv = pool.get("gll_derivative")
    inverse_radius = np.float64(1.0) / np.float64(pool.get("earth_radius"))
    ptop = np.float64(pool.get("hybrid_a_interface")[0] * pool.get("reference_pressure"))
    nelem = pool.dimensions["nelem_local"]
    nlev = pool.dimensions["pver"]
    np_value = pool.dimensions["np"]
    dpdry = pool.get("layer_pressure_thickness")[..., time_level]
    if q_time_level is None:
        q_time_level = time_level
    qdp = pool.get("constituent_mass")[..., q_time_level]
    velocity_u = pool.get("zonal_wind")[..., time_level]
    velocity_v = pool.get("meridional_wind")[..., time_level]
    omega = pool.get("vertical_pressure_velocity")
    for le in range(nelem):
        dinv = pool.get("inverse_metric")[:, :, :, :, le]
        metdet = pool.get("metric_jacobian")[:, :, le]
        inverse_metdet = pool.get("inverse_metric_jacobian")[:, :, le]
        pressure = np.empty(
            (np_value, np_value),
            dtype=np.float64,
            order="F",
        )
        cumulative = np.zeros_like(pressure, order="F")
        previous_dp = None
        for k in range(nlev):
            dp = np.empty_like(pressure, order="F")
            for j in range(np_value):
                for i in range(np_value):
                    value = dpdry[i, j, k, le]
                    for constituent in range(pool.dimensions["qsize"]):
                        value = np.float64(value + qdp[i, j, k, le, constituent])
                    dp[i, j] = value
                    if k == 0:
                        pressure[i, j] = np.float64(ptop + np.float64(value / np.float64(2.0)))
                    else:
                        pressure[i, j] = np.float64(
                            pressure[i, j]
                            + np.float64(previous_dp[i, j] / np.float64(2.0))
                            + np.float64(value / np.float64(2.0))
                        )
            gradient = _gradient_sphere(pressure, dvv, dinv, inverse_radius)
            mass_flux = np.empty(
                (np_value, np_value, 2),
                dtype=np.float64,
                order="F",
            )
            vgrad = np.empty_like(pressure, order="F")
            for j in range(np_value):
                for i in range(np_value):
                    u = velocity_u[i, j, k, le]
                    v = velocity_v[i, j, k, le]
                    mass_flux[i, j, 0] = np.float64(dp[i, j] * u)
                    mass_flux[i, j, 1] = np.float64(dp[i, j] * v)
                    vgrad[i, j] = np.float64(
                        np.float64(u * gradient[i, j, 0])
                        + np.float64(v * gradient[i, j, 1])
                    )
            divergence = _divergence_sphere(
                mass_flux, dvv, dinv, metdet, inverse_metdet, inverse_radius
            )
            for j in range(np_value):
                for i in range(np_value):
                    term = np.float64(-divergence[i, j])
                    omega[i, j, k, le] = np.float64(
                        cumulative[i, j]
                        + np.float64(np.float64(0.5) * term)
                        + vgrad[i, j]
                    )
                    cumulative[i, j] = np.float64(cumulative[i, j] + term)
            previous_dp = dp

    pool.get("vertical_pressure_velocity_raw")[...] = omega
    omega[...] = _dss(pool, comm, omega, multiply_mass=True, divide_mass=True)
    pool.get("vertical_pressure_velocity_after_dss")[...] = omega

    if not apply_hyperviscosity:
        return

    subcycles = int(pool.get("hyperviscosity_subcycles"))
    dt_hyper = np.float64(pool.get("vertical_remap_timestep")) / np.float64(subcycles)
    nu_p = np.float64(pool.get("pressure_hyperviscosity"))
    for subcycle in range(subcycles):
        weak = np.empty_like(omega, order="F")
        for le in range(nelem):
            dinv = pool.get("inverse_metric")[:, :, :, :, le]
            mass = pool.get("spectral_mass_matrix")[:, :, le]
            for k in range(nlev):
                weak[:, :, k, le] = _laplace_sphere_weak(
                    omega[:, :, k, le], dvv, dinv, mass, inverse_radius
                )
        weak = _dss(pool, comm, weak, multiply_mass=False, divide_mass=True)
        biharmonic = np.empty_like(omega, order="F")
        for le in range(nelem):
            dinv = pool.get("inverse_metric")[:, :, :, :, le]
            mass = pool.get("spectral_mass_matrix")[:, :, le]
            for k in range(nlev):
                biharmonic[:, :, k, le] = _laplace_sphere_weak(
                    weak[:, :, k, le], dvv, dinv, mass, inverse_radius
                )
        pool.get("omega_biharmonic_stage")[..., subcycle] = biharmonic
        for le in range(nelem):
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        biharmonic[i, j, k, le] = np.float64(
                            np.float64(-dt_hyper * nu_p) * biharmonic[i, j, k, le]
                        )
        correction = _dss(pool, comm, biharmonic, multiply_mass=False, divide_mass=True)
        for le in range(nelem):
            for k in range(nlev):
                for j in range(np_value):
                    for i in range(np_value):
                        omega[i, j, k, le] = np.float64(
                            omega[i, j, k, le] + correction[i, j, k, le]
                        )
        pool.get("omega_after_hypervis_stage")[..., subcycle] = omega
