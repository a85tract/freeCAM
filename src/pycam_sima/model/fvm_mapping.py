"""Halo-aware PG3-to-GLL forcing mapping orchestrated by Python."""

from __future__ import annotations

import math

import numpy as np

from .dynamics import _edge_sum


def _lagrange_1d(
    source_grid: np.ndarray,
    source_value: np.ndarray,
    destination: np.float64,
    width: int = 2,
) -> np.float64:
    """Scalar-order port of ``fvm_mapping:lagrange_1d``."""

    ngrid = len(source_grid)
    if destination <= source_grid[0]:
        reference = 0
    else:
        reference = 0
        while reference < ngrid and destination > source_grid[reference]:
            reference += 1
        reference -= 1
    # Convert the Fortran one-based clamp to zero-based indexing.
    reference = min(max(reference, width - 1), ngrid - width - 1)
    weights = np.ones(ngrid, dtype=np.float64)
    first = reference - (width - 1)
    last = reference + width
    for j in range(first, last + 1):
        for k in range(first, last + 1):
            if k != j:
                weights[j] = np.float64(
                    np.float64(weights[j] * np.float64(destination - source_grid[k]))
                    / np.float64(source_grid[j] - source_grid[k])
                )
    value = np.float64(0.0)
    for j in range(first, last + 1):
        value = np.float64(value + np.float64(weights[j] * source_value[j]))
    return value


def gather_physgrid_halo(pool, comm, local_field: np.ndarray) -> np.ndarray:
    """Fill three PG3 halo cells through an explicit mpi4py exchange."""

    nlev, nfields = local_field.shape[2:4]
    nelem = pool.dimensions["nelem_local"]
    ids = np.empty(nelem * 9, dtype=np.int64)
    values = np.empty((nelem * 9, nlev, nfields), dtype=np.float64, order="F")
    offset = 0
    for le in range(nelem):
        for j in range(3):
            for i in range(3):
                ids[offset] = pool.get("physics_global_column")[i, j, le]
                values[offset, :, :] = local_field[i, j, :, :, le]
                offset += 1
    gathered = comm.allgather((ids, values))
    global_values = np.empty((6 * 9 * 9, nlev, nfields), dtype=np.float64, order="F")
    for rank_ids, rank_values in gathered:
        for row, column in enumerate(np.asarray(rank_ids, dtype=np.int64)):
            global_values[int(column) - 1, :, :] = np.asarray(rank_values)[row, :, :]

    halo = np.full(
        (9, 9, nlev, nfields, nelem),
        np.float64(-9.99e99),
        dtype=np.float64,
        order="F",
    )
    schedule = pool.get("pg3_halo_global_column")
    for le in range(nelem):
        for j in range(9):
            for i in range(9):
                column = int(schedule[i, j, le])
                if column > 0:
                    halo[i, j, :, :, le] = global_values[column - 1, :, :]
    return halo


def _special_corner(
    boundary: int,
    psi: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply CAM's cross-panel corner interpolation before tensor mapping."""

    imin = np.full(9, -2, dtype=np.int32)
    imax = np.full(9, 6, dtype=np.int32)
    nlev, nfields = psi.shape[2:4]
    temporary = np.empty((9, 9), dtype=np.float64, order="F")
    if boundary == 5:  # swest
        for field in range(nfields):
            for level in range(nlev):
                for jrow in range(1, 6):
                    for irow in range(-1, 1):
                        temporary[irow + 2, jrow + 2] = _lagrange_1d(
                            coordinates[1, irow + 2, 3:9],
                            psi[irow + 2, 3:9, level, field],
                            coordinates[1, 3, jrow + 2],
                        )
                psi[1:3, 3:8, level, field] = temporary[1:3, 3:8]
        imin[0:3] = 1
    elif boundary == 7:  # nwest
        for field in range(nfields):
            for level in range(nlev):
                for jrow in range(-1, 4):
                    for irow in range(-1, 1):
                        temporary[irow + 2, jrow + 2] = _lagrange_1d(
                            coordinates[1, irow + 2, 0:6],
                            psi[irow + 2, 0:6, level, field],
                            coordinates[1, 3, jrow + 2],
                        )
                psi[1:3, 1:6, level, field] = temporary[1:3, 1:6]
        imin[6:9] = 1
    elif boundary == 6:  # seast
        for field in range(nfields):
            for level in range(nlev):
                for jrow in range(1, 6):
                    for irow in range(4, 6):
                        temporary[irow + 2, jrow + 2] = _lagrange_1d(
                            coordinates[1, irow + 2, 3:9],
                            psi[irow + 2, 3:9, level, field],
                            coordinates[1, 3, jrow + 2],
                        )
                psi[6:8, 3:8, level, field] = temporary[6:8, 3:8]
        imax[0:3] = 3
    elif boundary == 8:  # neast
        for field in range(nfields):
            for level in range(nlev):
                for jrow in range(-1, 4):
                    for irow in range(4, 6):
                        temporary[irow + 2, jrow + 2] = _lagrange_1d(
                            coordinates[1, irow + 2, 0:6],
                            psi[irow + 2, 0:6, level, field],
                            coordinates[1, 3, jrow + 2],
                        )
                psi[6:8, 1:6, level, field] = temporary[6:8, 1:6]
        imax[6:9] = 3
    return imin, imax


def tensor_lagrange_interp(
    boundary: int,
    psi: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Port the un-limited nc=3, np=4 tensor interpolation path."""

    psi = np.array(psi, dtype=np.float64, order="F", copy=True)
    nlev, nfields = psi.shape[2:4]
    output = np.empty((4, 4, nlev, nfields), dtype=np.float64, order="F")
    imin, imax = _special_corner(boundary, psi, coordinates)
    gll = np.array(
        (-1.0, -math.sqrt(1.0 / 5.0), math.sqrt(1.0 / 5.0), 1.0),
        dtype=np.float64,
    )
    value = np.empty(9, dtype=np.float64)
    if boundary not in (1, 2):
        for field in range(nfields):
            for level in range(nlev):
                for igll in range(4):
                    for jrow in range(-1, 6):
                        first = int(imin[jrow + 2])
                        last = int(imax[jrow + 2])
                        value[jrow + 2] = _lagrange_1d(
                            coordinates[0, first + 2:last + 3, jrow + 2],
                            psi[first + 2:last + 3, jrow + 2, level, field],
                            gll[igll],
                        )
                    for jgll in range(4):
                        output[igll, jgll, level, field] = _lagrange_1d(
                            coordinates[1, 3, 1:8], value[1:8], gll[jgll]
                        )
    else:
        for field in range(nfields):
            for level in range(nlev):
                for jgll in range(4):
                    for irow in range(-1, 6):
                        value[irow + 2] = _lagrange_1d(
                            coordinates[1, irow + 2, 0:9],
                            psi[irow + 2, 0:9, level, field],
                            gll[jgll],
                        )
                    for igll in range(4):
                        output[igll, jgll, level, field] = _lagrange_1d(
                            coordinates[0, 1:8, 3], value[1:8], gll[igll]
                        )
    return output


def physgrid_to_gll(pool, comm, local_field: np.ndarray, *, vector_start: int | None = None) -> np.ndarray:
    """Fill halos, map scalars/vectors, and return rank-local GLL fields."""

    halo = gather_physgrid_halo(pool, comm, local_field)
    nelem = pool.dimensions["nelem_local"]
    if vector_start is not None:
        inverse_metric = pool.get("fvm_inverse_metric_physgrid")
        for le in range(nelem):
            for level in range(local_field.shape[2]):
                for j in range(9):
                    for i in range(9):
                        v1 = halo[i, j, level, vector_start, le]
                        v2 = halo[i, j, level, vector_start + 1, le]
                        halo[i, j, level, vector_start, le] = np.float64(
                            np.float64(inverse_metric[0, 0, i, j, le] * v1)
                            + np.float64(inverse_metric[0, 1, i, j, le] * v2)
                        )
                        halo[i, j, level, vector_start + 1, le] = np.float64(
                            np.float64(inverse_metric[1, 0, i, j, le] * v1)
                            + np.float64(inverse_metric[1, 1, i, j, le] * v2)
                        )
    mapped = np.empty(
        (4, 4, local_field.shape[2], local_field.shape[3], nelem),
        dtype=np.float64,
        order="F",
    )
    for le in range(nelem):
        mapped[..., le] = tensor_lagrange_interp(
            int(pool.get("fvm_cube_boundary")[le]),
            halo[..., le],
            pool.get("fvm_normalized_element_coordinate")[..., le],
        )
    if vector_start is not None:
        metric = pool.get("metric_derivative")
        for le in range(nelem):
            for level in range(local_field.shape[2]):
                for j in range(4):
                    for i in range(4):
                        v1 = mapped[i, j, level, vector_start, le]
                        v2 = mapped[i, j, level, vector_start + 1, le]
                        mapped[i, j, level, vector_start, le] = np.float64(
                            np.float64(metric[0, 0, i, j, le] * v1)
                            + np.float64(metric[0, 1, i, j, le] * v2)
                        )
                        mapped[i, j, level, vector_start + 1, le] = np.float64(
                            np.float64(metric[1, 0, i, j, le] * v1)
                            + np.float64(metric[1, 1, i, j, le] * v2)
                        )
    return mapped


def physics_to_dynamics_forcing(pool, comm) -> None:
    """Port the CAM SE/FVM ``p_d_coupling`` forcing boundary."""

    nlev = pool.dimensions["pver"]
    nconst = pool.dimensions["nconst"]
    nelem = pool.dimensions["nelem_local"]
    current_q = pool.get("physics_constituent_mixing_ratio")
    wet_to_dry = pool.get("physics_layer_pressure_thickness") / pool.get(
        "physics_dry_layer_pressure_thickness"
    )
    for constituent in range(nconst):
        for level in range(nlev):
            for column in range(pool.dimensions["nphys_local"]):
                current_q[column, level, constituent] = np.float64(
                    wet_to_dry[column, level]
                    * current_q[column, level, constituent]
                )
    adjustment = current_q - pool.get("physics_constituent_previous")
    local = np.empty((3, 3, nlev, 3 + nconst, nelem), dtype=np.float64, order="F")
    for le in range(nelem):
        columns = slice(le * 9, (le + 1) * 9)
        local[:, :, :, 0, le] = pool.get("physics_air_temperature_tendency")[columns, :].reshape(
            (3, 3, nlev), order="F"
        )
        local[:, :, :, 1, le] = pool.get("physics_zonal_wind_tendency")[columns, :].reshape(
            (3, 3, nlev), order="F"
        )
        local[:, :, :, 2, le] = pool.get("physics_meridional_wind_tendency")[columns, :].reshape(
            (3, 3, nlev), order="F"
        )
        local[:, :, :, 3:, le] = (
            pool.get("fvm_tracer")[3:6, 3:6, :, le, :] +
            adjustment[columns, :, :].reshape((3, 3, nlev, nconst), order="F")
        )
        pool.get("fvm_temperature_forcing")[:, :, :, le] = local[:, :, :, 0, le]
        pool.get("fvm_momentum_forcing")[:, :, 0, :, le] = local[:, :, :, 1, le]
        pool.get("fvm_momentum_forcing")[:, :, 1, :, le] = local[:, :, :, 2, le]
        pool.get("fvm_constituent_adjustment")[:, :, :, le, :] = adjustment[
            columns, :, :
        ].reshape((3, 3, nlev, nconst), order="F")
        pool.get("fvm_constituent_mass_forcing")[:, :, :, le, :] = np.float64(0.0)
        for constituent in range(nconst):
            pool.get("fvm_constituent_mass_forcing")[:, :, :, le, constituent] = (
                pool.get("fvm_constituent_adjustment")[:, :, :, le, constituent]
                * pool.get("fvm_layer_pressure_thickness")[3:6, 3:6, :, le]
            )
        pool.get("fvm_dry_pressure_from_physics")[:, :, :, le] = pool.get(
            "physics_dry_layer_pressure_thickness"
        )[columns, :].reshape((3, 3, nlev), order="F")

    mapped = physgrid_to_gll(pool, comm, local, vector_start=1)
    n0 = int(pool.get("dynamics_time_level_n0"))
    qn0 = 0 if int(pool.get("dynamics_internal_step")) % 2 == 0 else 1
    qold = np.empty((4, 4, nlev, nconst, nelem), dtype=np.float64, order="F")
    for le in range(nelem):
        for constituent in range(nconst):
            for level in range(nlev):
                for j in range(4):
                    for i in range(4):
                        qold[i, j, level, constituent, le] = np.float64(
                            pool.get("constituent_mass")[i, j, level, le, constituent, qn0]
                            / pool.get("layer_pressure_thickness")[i, j, level, le, n0]
                        )

    forcing = np.empty((4, 4, nlev, 3 + nconst, nelem), dtype=np.float64, order="F")
    forcing[:, :, :, 0:3, :] = mapped[:, :, :, 0:3, :]
    for constituent in range(nconst):
        forcing[:, :, :, 3 + constituent, :] = np.float64(0.0)
        for le in range(nelem):
            for level in range(nlev):
                for j in range(4):
                    for i in range(4):
                        forcing[i, j, level, 3 + constituent, le] = np.float64(
                            mapped[i, j, level, 3 + constituent, le]
                            - qold[i, j, level, constituent, le]
                        )

    mass = pool.get("spectral_mass_matrix")
    packed = np.empty_like(forcing, order="F")
    for le in range(nelem):
        for field in range(3 + nconst):
            for level in range(nlev):
                for j in range(4):
                    for i in range(4):
                        packed[i, j, level, field, le] = np.float64(
                            forcing[i, j, level, field, le] * mass[i, j, le]
                        )
    assembled = _edge_sum(pool, comm, packed)
    inverse_mass = pool.get("inverse_spectral_mass_matrix")
    for le in range(nelem):
        for field in range(3 + nconst):
            for level in range(nlev):
                for j in range(4):
                    for i in range(4):
                        forcing[i, j, level, field, le] = np.float64(
                            assembled[i, j, level, field, le]
                            * inverse_mass[i, j, le]
                        )
    pool.get("temperature_forcing")[...] = forcing[:, :, :, 0, :]
    pool.get("zonal_wind_forcing")[...] = forcing[:, :, :, 1, :]
    pool.get("meridional_wind_forcing")[...] = forcing[:, :, :, 2, :]
    pool.get("constituent_forcing")[...] = forcing[:, :, :, 3:, :].transpose(0, 1, 2, 4, 3)
