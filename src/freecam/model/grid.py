"""ne3 cubed-sphere topology and rank decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import math

import numpy as np


# (source face, source edge) -> (neighbor face, neighbor edge, reverse along edge)
# for the equiangular HOMME cubed sphere.  Edge names use W/E/S/N.
_FACE_EDGE = {
    (1, "W"): (4, "E", False), (1, "E"): (2, "W", False),
    (1, "S"): (5, "N", False), (1, "N"): (6, "S", False),
    (2, "W"): (1, "E", False), (2, "E"): (3, "W", False),
    (2, "S"): (5, "E", True),  (2, "N"): (6, "E", False),
    (3, "W"): (2, "E", False), (3, "E"): (4, "W", False),
    (3, "S"): (5, "S", True),  (3, "N"): (6, "N", True),
    (4, "W"): (3, "E", False), (4, "E"): (1, "W", False),
    (4, "S"): (5, "W", False), (4, "N"): (6, "W", True),
    (5, "W"): (4, "S", False), (5, "E"): (2, "S", True),
    (5, "S"): (3, "S", True),  (5, "N"): (1, "S", False),
    (6, "W"): (4, "N", True),  (6, "E"): (2, "N", False),
    (6, "S"): (1, "N", False), (6, "N"): (3, "N", True),
}


@dataclass(frozen=True, slots=True)
class Element:
    global_id: int
    face: int
    i: int
    j: int
    sfc: int
    owner: int


def _sfc_factors(side: int) -> tuple[int, ...] | None:
    """Return HOMME's ordered 2/3/5 factorization, or ``None``."""

    remaining = side
    factors: list[int] = []
    for factor in (2, 3, 5):
        while remaining % factor == 0:
            factors.append(factor)
            remaining //= factor
    return tuple(factors) if remaining == 1 else None


def _factorable_space_curve(side: int) -> tuple[tuple[int, ...], ...]:
    """Port ``spacecurve_mod::GenSpaceCurve`` for a 2/3/5 grid."""

    factors = _sfc_factors(side)
    if not factors:
        raise ValueError(f"side={side} is not factorable by 2, 3, and 5")

    ordered = [[-1 for _ in range(side)] for _ in range(side)]
    position = [0, 0]
    visitation_count = 0

    def increment(join_axis: int, join_direction: int) -> None:
        nonlocal visitation_count
        ordered[position[0]][position[1]] = visitation_count
        visitation_count += 1
        position[join_axis] += join_direction

    def curve(
        level: int,
        curve_type: int,
        main_axis: int,
        main_direction: int,
        join_axis: int,
        join_direction: int,
    ) -> None:
        other_axis = (main_axis + 1) % 2
        if curve_type == 2:
            sections = (
                (other_axis, main_direction, other_axis, main_direction),
                (main_axis, main_direction, main_axis, main_direction),
                (main_axis, main_direction, other_axis, -main_direction),
                (
                    other_axis,
                    -main_direction,
                    join_axis,
                    join_direction,
                ),
            )
        elif curve_type == 3:
            sections = (
                (other_axis, main_direction, other_axis, main_direction),
                (other_axis, main_direction, other_axis, main_direction),
                (main_axis, main_direction, main_axis, main_direction),
                (main_axis, main_direction, main_axis, main_direction),
                (main_axis, main_direction, other_axis, -main_direction),
                (main_axis, -main_direction, main_axis, -main_direction),
                (other_axis, -main_direction, other_axis, -main_direction),
                (other_axis, -main_direction, main_axis, main_direction),
                (
                    main_axis,
                    main_direction,
                    join_axis,
                    join_direction,
                ),
            )
        elif curve_type == 5:
            sections = (
                (main_axis, main_direction, main_axis, main_direction),
                (main_axis, main_direction, main_axis, main_direction),
                (other_axis, main_direction, other_axis, main_direction),
                (other_axis, main_direction, other_axis, main_direction),
                (other_axis, main_direction, main_axis, -main_direction),
                (other_axis, -main_direction, other_axis, -main_direction),
                (main_axis, -main_direction, main_axis, -main_direction),
                (main_axis, -main_direction, other_axis, main_direction),
                (other_axis, main_direction, other_axis, main_direction),
                (other_axis, main_direction, other_axis, main_direction),
                (main_axis, main_direction, main_axis, main_direction),
                (main_axis, main_direction, other_axis, -main_direction),
                (other_axis, -main_direction, main_axis, main_direction),
                (other_axis, main_direction, other_axis, main_direction),
                (main_axis, main_direction, main_axis, main_direction),
                (main_axis, main_direction, main_axis, main_direction),
                (main_axis, main_direction, other_axis, -main_direction),
                (main_axis, -main_direction, main_axis, -main_direction),
                (other_axis, -main_direction, other_axis, -main_direction),
                (other_axis, -main_direction, main_axis, main_direction),
                (main_axis, main_direction, other_axis, -main_direction),
                (main_axis, -main_direction, main_axis, -main_direction),
                (other_axis, -main_direction, other_axis, -main_direction),
                (other_axis, -main_direction, main_axis, main_direction),
                (
                    main_axis,
                    main_direction,
                    join_axis,
                    join_direction,
                ),
            )
        else:
            raise ValueError(f"unsupported HOMME SFC factor {curve_type}")

        if level == 1:
            for _, _, next_join_axis, next_join_direction in sections:
                increment(next_join_axis, next_join_direction)
            return

        next_type = factors[level - 2]
        for (
            next_main_axis,
            next_main_direction,
            next_join_axis,
            next_join_direction,
        ) in sections:
            curve(
                level - 1,
                next_type,
                next_main_axis,
                next_main_direction,
                next_join_axis,
                next_join_direction,
            )

    curve(len(factors), factors[-1], 0, 1, 0, 1)
    if visitation_count != side * side:
        raise RuntimeError(
            f"HOMME SFC visited {visitation_count} of {side * side} cells"
        )
    return tuple(tuple(column) for column in ordered)


@cache
def homme_space_curve(side: int) -> tuple[tuple[int, ...], ...]:
    """Return HOMME's zero-based SFC index as ``mesh[i][j]``.

    This is a Python port of ``spacecurve_mod::GenSpaceCurve`` and the
    non-factorable fallback in ``cube_mod::CubeTopology``.  It therefore
    supports every positive element count per cubed-sphere edge, not only
    the reference case's ``ne=3``.
    """

    side = int(side)
    if side <= 0:
        raise ValueError("side must be positive")
    if side == 1:
        return ((0,),)

    factors = _sfc_factors(side)
    if factors is not None:
        return _factorable_space_curve(side)

    enclosing_side = 1 << (side - 1).bit_length()
    enclosing = _factorable_space_curve(enclosing_side)

    def enclosing_index(index: int) -> int:
        # This is Fortran NINT for a positive, one-based CubeTopology index.
        coordinate = (
            ((index + 0.5) / side) * enclosing_side + 0.5
        )
        one_based = math.floor(coordinate + 0.5)
        return min(max(one_based - 1, 0), enclosing_side - 1)

    mapped: dict[int, tuple[int, int]] = {}
    for j in range(side):
        for i in range(side):
            enclosing_i = enclosing_index(i)
            enclosing_j = enclosing_index(j)
            mapped[enclosing[enclosing_i][enclosing_j]] = (i, j)

    ordered = [[-1 for _ in range(side)] for _ in range(side)]
    for sfc_index, (_, (i, j)) in enumerate(sorted(mapped.items())):
        ordered[i][j] = sfc_index
    if len(mapped) != side * side:
        raise RuntimeError(
            f"HOMME fallback mapped {len(mapped)} of {side * side} cells"
        )
    return tuple(tuple(column) for column in ordered)


def _face_sfc(face: int, i: int, j: int, ne: int = 3) -> int:
    if not 0 <= i < ne or not 0 <= j < ne:
        raise ValueError(f"element indices ({i}, {j}) are outside ne={ne}")
    mesh = homme_space_curve(ne)
    if face in (1, 2):
        key, face_offset = (i, ne - 1 - j), (face - 1) * ne * ne
    elif face == 6:
        key, face_offset = (ne - 1 - i, ne - 1 - j), 2 * ne * ne
    elif face == 4:
        key, face_offset = (ne - 1 - j, i), 3 * ne * ne
    elif face == 5:
        key, face_offset = (i, j), 4 * ne * ne
    elif face == 3:
        key, face_offset = (i, j), 5 * ne * ne
    else:
        raise ValueError(f"invalid cube face {face}")
    return face_offset + mesh[key[0]][key[1]]


def _validate_sfc_partition(
    element_count: int,
    partition_count: int,
) -> tuple[int, int]:
    element_count = int(element_count)
    partition_count = int(partition_count)
    if element_count <= 0:
        raise ValueError("element_count must be positive")
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if partition_count > element_count:
        raise ValueError(
            "partition_count cannot exceed element_count because HOMME "
            "does not support empty SE partitions"
        )
    return element_count, partition_count


def sfc_partition_counts(
    element_count: int,
    partition_count: int,
) -> tuple[int, ...]:
    """Return HOMME's contiguous SFC element count for every partition."""

    element_count, partition_count = _validate_sfc_partition(
        element_count,
        partition_count,
    )
    elements_per_partition, extra = divmod(
        element_count, partition_count
    )
    return tuple(
        elements_per_partition + int(rank < extra)
        for rank in range(partition_count)
    )


def sfc_partition_owner(
    sfc_index: int,
    element_count: int,
    partition_count: int,
) -> int:
    """Return the zero-based owner of one zero-based SFC element index."""

    sfc_index = int(sfc_index)
    element_count, partition_count = _validate_sfc_partition(
        element_count,
        partition_count,
    )
    if not 0 <= sfc_index < element_count:
        raise ValueError(
            f"sfc_index must be between 0 and {element_count - 1}"
        )
    elements_per_partition, extra = divmod(
        element_count, partition_count
    )
    larger_partition = elements_per_partition + 1
    larger_span = extra * larger_partition
    if sfc_index < larger_span:
        return sfc_index // larger_partition
    return extra + (sfc_index - larger_span) // elements_per_partition


def global_elements(size: int, ne: int = 3) -> tuple[Element, ...]:
    element_count = 6 * ne * ne
    sfc_partition_counts(element_count, size)
    elements: list[Element] = []
    for face in range(1, 7):
        for j in range(ne):
            for i in range(ne):
                sfc = _face_sfc(face, i, j, ne)
                owner = sfc_partition_owner(sfc, element_count, size)
                global_id = 1 + i + ne * j + ne * ne * (face - 1)
                elements.append(
                    Element(global_id, face, i, j, sfc, owner)
                )
    return tuple(sorted(elements, key=lambda item: item.sfc))


def local_elements(
    rank: int,
    size: int = 24,
    ne: int = 3,
) -> tuple[Element, ...]:
    if not 0 <= rank < size:
        raise ValueError(f"rank must be between 0 and {size - 1}")
    # HOMME stores each rank's contiguous SFC assignment in ascending global
    # element id, not traversal order within the SFC segment.
    return tuple(
        sorted(
            (
                item
                for item in global_elements(size, ne)
                if item.owner == rank
            ),
            key=lambda item: item.global_id,
        )
    )


def dimensions_for_rank(
    rank: int,
    size: int = 24,
    *,
    pver: int = 30,
    np_value: int = 4,
    fv_nphys: int = 3,
    constituent_count: int = 3,
    advected_constituent_count: int | None = None,
    thermodynamic_constituent_count: int | None = None,
    ne: int = 3,
    extra_dimensions: dict[str, int] | None = None,
) -> dict[str, int]:
    nelem = len(local_elements(rank, size, ne))
    # Halo sizes are finalized before allocation from the global topology.
    peers, shared = _halo_inventory(
        rank,
        size,
        ne=ne,
        np_value=np_value,
    )
    ntrac = (
        constituent_count
        if advected_constituent_count is None
        else int(advected_constituent_count)
    )
    qsize = (
        constituent_count
        if thermodynamic_constituent_count is None
        else int(thermodynamic_constituent_count)
    )
    dimensions = {
        "pver": pver,
        "pverp": pver + 1,
        "np": np_value,
        "nc": fv_nphys,
        "fv_nphys": fv_nphys,
        "nelem_local": nelem,
        "nphys_local": nelem * fv_nphys * fv_nphys,
        # HOMME keeps at least a ten-slot Qdp buffer. Preserve that ABI-sized
        # Python-owned state; active constituent aliases use nconst.
        "ntime": 3,
        "ntracer_time": 3,
        "nconst": constituent_count,
        "ntrac": ntrac,
        "number_of_ccpp_constituents": constituent_count,
        "qsize": qsize,
        "qsize_storage": max(10, constituent_count),
        "ccpp_constant_one": 1,
        "ccpp_constant_two": 2,
        "nhypervis": min(3, pver),
        "edge_count": 4,
        # CSLAM reconstruction controls remain explicit, while their array
        # extents follow the configured number of finite-volume cells.
        "fvm_halo": fv_nphys + 6,
        "nhc": 3,
        "fvm_internal": fv_nphys + 2,
        "fvm_interp_span": fv_nphys + 4,
        "fvm_reconstruction": 5,
        "fvm_reconstruction_terms": 3,
        "fvm_halo_range": 3,
        "fvm_stretch": fv_nphys + 4,
        "cartesian": 3, "metric_i": 2, "metric_j": 2,
        "nhalo_peer": max(1, len(peers)), "nhalo_peerp": max(1, len(peers)) + 1,
        "nhalo_dof": max(1, sum(len(value) for value in shared.values())),
        "mapping_fields": 7, "coupler_fields": 8,
    }
    for name, value in (extra_dimensions or {}).items():
        if name in dimensions and dimensions[name] != int(value):
            raise ValueError(
                f"extra dimension {name!r}={value} conflicts with "
                f"the grid-derived value {dimensions[name]}"
            )
        dimensions[str(name)] = int(value)
    return dimensions


def _unit_sphere(face: int, x: float, y: float) -> tuple[float, float, tuple[float, float, float]]:
    r = math.sqrt(1.0 + x * x + y * y)
    if face == 1:
        lat, lon, xyz = math.asin(y / r), math.atan2(x, 1.0), (1.0 / r, x / r, y / r)
    elif face == 2:
        lat, lon, xyz = math.asin(y / r), math.atan2(1.0, -x), (-x / r, 1.0 / r, y / r)
    elif face == 3:
        lat, lon, xyz = math.asin(y / r), math.atan2(-x, -1.0), (-1.0 / r, -x / r, y / r)
    elif face == 4:
        lat, lon, xyz = math.asin(y / r), math.atan2(-1.0, x), (x / r, -1.0 / r, y / r)
    elif face == 5:
        lon = math.atan2(x, y) if abs(x) > 1.0e-14 or abs(y) > 1.0e-14 else 0.0
        lat, xyz = math.asin(-1.0 / r), (y / r, x / r, -1.0 / r)
    else:
        lon = math.atan2(x, -y) if abs(x) > 1.0e-14 or abs(y) > 1.0e-14 else 0.0
        lat, xyz = math.asin(1.0 / r), (-y / r, x / r, 1.0 / r)
    if lon < 0.0:
        lon += 2.0 * math.pi
    return lon, lat, xyz


def _cartesian_to_face_angles(face: int, xyz: tuple[float, float, float]) -> tuple[float, float]:
    """Scalar-order port of ``cart2cubedsphere``."""

    x, y, z = xyz
    if face == 1:
        first, second = y / x, z / x
    elif face == 2:
        first, second = -x / y, z / y
    elif face == 3:
        first, second = y / x, -z / x
    elif face == 4:
        first, second = -x / y, -z / y
    elif face == 5:
        first, second = -y / z, -x / z
    elif face == 6:
        first, second = y / z, -x / z
    else:
        raise ValueError(f"invalid cube face {face}")
    return math.atan(first), math.atan(second)


def _fortran_cubedsphere_to_cart(
    face: int, alpha: np.float64, beta: np.float64
) -> tuple[float, float, float]:
    """Port ``cubedsphere2cart`` including its spherical round trip."""

    x = math.tan(alpha)
    y = math.tan(beta)
    radius = math.sqrt(np.float64(1.0) + x * x + y * y)
    if face == 1:
        latitude, longitude = math.asin(y / radius), math.atan2(x, 1.0)
    elif face == 2:
        latitude, longitude = math.asin(y / radius), math.atan2(1.0, -x)
    elif face == 3:
        latitude, longitude = math.asin(y / radius), math.atan2(-x, -1.0)
    elif face == 4:
        latitude, longitude = math.asin(y / radius), math.atan2(-1.0, x)
    elif face == 5:
        longitude = math.atan2(x, y) if abs(y) > 1.0e-9 or abs(x) > 1.0e-9 else 0.0
        latitude = math.asin(-1.0 / radius)
    elif face == 6:
        longitude = math.atan2(x, -y) if abs(y) > 1.0e-9 or abs(x) > 1.0e-9 else 0.0
        latitude = math.asin(1.0 / radius)
    else:
        raise ValueError(f"invalid cube face {face}")
    if longitude < 0.0:
        longitude = longitude + 2.0 * math.pi
    cos_latitude = math.cos(latitude)
    return (
        cos_latitude * math.cos(longitude),
        cos_latitude * math.sin(longitude),
        math.sin(latitude),
    )


def _cross_face_cell(face: int, gx: int, gy: int, width: int = 9) -> tuple[int, int, int] | None:
    """Map a single-edge halo cell to its owning panel and cell indexes."""

    outside_x = gx < 0 or gx >= width
    outside_y = gy < 0 or gy >= width
    if outside_x and outside_y:
        return None
    if not outside_x and not outside_y:
        return face, gx, gy
    if gx < 0:
        edge, depth, along = "W", -gx, gy
    elif gx >= width:
        edge, depth, along = "E", gx - width + 1, gy
    elif gy < 0:
        edge, depth, along = "S", -gy, gx
    else:
        edge, depth, along = "N", gy - width + 1, gx
    target_face, target_edge, reverse = _FACE_EDGE[(face, edge)]
    if reverse:
        along = width - 1 - along
    if target_edge == "W":
        tx, ty = depth - 1, along
    elif target_edge == "E":
        tx, ty = width - depth, along
    elif target_edge == "S":
        tx, ty = along, depth - 1
    else:
        tx, ty = along, width - depth
    return target_face, tx, ty


def _fvm_cube_boundary(element: Element, ne: int = 3) -> int:
    if element.i == 0 and element.j == 0:
        return 5
    if element.i == ne - 1 and element.j == 0:
        return 6
    if element.i == 0 and element.j == ne - 1:
        return 7
    if element.i == ne - 1 and element.j == ne - 1:
        return 8
    if element.i == 0:
        return 1
    if element.i == ne - 1:
        return 2
    if element.j == 0:
        return 3
    if element.j == ne - 1:
        return 4
    return 0


def _initialize_physgrid_halo(
    pool,
    elements: tuple[Element, ...],
    size: int,
    ne: int = 3,
) -> None:
    """Build the persistent finite-volume ghost schedule and mapping metric."""

    nc = pool.dimensions["fv_nphys"]
    nhc = (pool.dimensions["fvm_halo"] - nc) // 2
    panel_width = ne * nc
    by_id = {
        element.global_id: element for element in global_elements(size, ne)
    }
    schedule = pool.get("pg3_halo_global_column", unsafe=True)
    norm = pool.get("fvm_normalized_element_coordinate", unsafe=True)
    dinv = pool.get("fvm_inverse_metric_physgrid", unsafe=True)
    boundary = pool.get("fvm_cube_boundary", unsafe=True)
    for le, element in enumerate(elements):
        boundary[le] = _fvm_cube_boundary(element, ne)
        source_corners = pool.get("cube_corner_angle", unsafe=True)[:, :, le]
        source_x0 = source_corners[0, 0]
        source_y0 = source_corners[1, 0]
        source_dx = np.float64(
            abs(np.float64(source_corners[0, 0] - source_corners[0, 1]))
            / np.float64(nc)
        )
        denominator = np.float64(0.5) * np.float64(nc) * source_dx
        u2q = _u2qmap(
            element,
            pool.get("gll_node", unsafe=True),
            ne=ne,
        )
        for hj in range(nc + 2 * nhc):
            local_j = hj - nhc
            gy = element.j * nc + local_j
            for hi in range(nc + 2 * nhc):
                local_i = hi - nhc
                gx = element.i * nc + local_i
                mapped = _cross_face_cell(element.face, gx, gy, panel_width)
                if mapped is None:
                    schedule[hi, hj, le] = -1
                    x1 = np.float64(1.0e9)
                    x2 = np.float64(1.0e9)
                else:
                    target_face, tx, ty = mapped
                    target_element_i, pi = divmod(tx, nc)
                    target_element_j, pj = divmod(ty, nc)
                    gid = 1 + target_element_i + ne * target_element_j + ne * ne * (target_face - 1)
                    schedule[hi, hj, le] = (gid - 1) * nc * nc + pi + nc * pj + 1
                    target_element = by_id[gid]
                    target_corners = _element_corners(target_element, ne)
                    target_dx = np.float64(
                        abs(np.float64(target_corners[0][0] - target_corners[1][0]))
                        / np.float64(nc)
                    )
                    target_dy = np.float64(
                        abs(np.float64(target_corners[0][1] - target_corners[3][1]))
                        / np.float64(nc)
                    )
                    target_alpha = np.float64(
                        np.float64(target_corners[0][0])
                        + np.float64(np.float64(pi + 1) - np.float64(0.5)) * target_dx
                    )
                    target_beta = np.float64(
                        np.float64(target_corners[0][1])
                        + np.float64(np.float64(pj + 1) - np.float64(0.5)) * target_dy
                    )
                    if target_face != element.face:
                        xyz = _fortran_cubedsphere_to_cart(
                            target_face, target_alpha, target_beta
                        )
                        target_alpha, target_beta = _cartesian_to_face_angles(element.face, xyz)
                    x1 = np.float64(
                        np.float64(target_alpha - source_x0) / denominator
                        - np.float64(1.0)
                    )
                    x2 = np.float64(
                        np.float64(target_beta - source_y0) / denominator
                        - np.float64(1.0)
                    )
                norm[0, hi, hj, le] = x1
                norm[1, hi, hj, le] = x2
                d = _raw_d_at(element, x1, x2, u2q, ne=ne)
                determinant = np.float64(
                    np.float64(d[0, 0] * d[1, 1])
                    - np.float64(d[0, 1] * d[1, 0])
                )
                dinv[0, 0, hi, hj, le] = np.float64(d[1, 1] / determinant)
                dinv[0, 1, hi, hj, le] = np.float64(-d[0, 1] / determinant)
                dinv[1, 0, hi, hj, le] = np.float64(-d[1, 0] / determinant)
                dinv[1, 1, hi, hj, le] = np.float64(d[0, 0] / determinant)


def _element_point(element: Element, xi: float, eta: float, ne: int = 3):
    cube_start = -0.25 * math.pi
    cube_end = 0.25 * math.pi
    delta = (cube_end - cube_start) / ne
    start_alpha = cube_start + element.i * delta
    start_beta = cube_start + element.j * delta
    alpha = start_alpha + 0.5 * (xi + 1.0) * delta
    beta = start_beta + 0.5 * (eta + 1.0) * delta
    x, y = math.tan(alpha), math.tan(beta)
    return _unit_sphere(element.face, x, y)


def _element_corners(element: Element, ne: int = 3) -> tuple[tuple[float, float], ...]:
    cube_start = -0.25 * math.pi
    cube_end = 0.25 * math.pi
    dx = (cube_end - cube_start) / ne
    dy = (cube_end - cube_start) / ne
    startx = cube_start + element.i * dx
    starty = cube_start + element.j * dy
    return ((startx, starty), (startx + dx, starty),
            (startx + dx, starty + dy), (startx, starty + dy))


def _bilinear(corners, a: float, b: float) -> tuple[float, float]:
    p_i = (1.0 - a) / 2.0
    p_j = (1.0 - b) / 2.0
    q_i = (1.0 + a) / 2.0
    q_j = (1.0 + b) / 2.0
    x = (p_i * p_j * corners[0][0] + q_i * p_j * corners[1][0]
         + q_i * q_j * corners[2][0] + p_i * q_j * corners[3][0])
    y = (p_i * p_j * corners[0][1] + q_i * p_j * corners[1][1]
         + q_i * q_j * corners[2][1] + p_i * q_j * corners[3][1])
    return x, y


def _physics_reference_nodes(count: int) -> np.ndarray:
    """Return finite-volume cell centers in HOMME's operation order.

    Values are formed from the same scalar expression as ``dmap`` rather than
    from pre-rounded fractions, preserving the existing PG3 BFB path.
    """
    dx = np.float64(2.0) / np.float64(count)
    result = np.empty(count, dtype=np.float64)
    for i in range(count):
        result[i] = np.float64(-1.0) + (
            np.float64(i + 1) - np.float64(0.5)
        ) * dx
    return result


def _pg3_reference_nodes() -> np.ndarray:
    """Backward-compatible name for the BFB reference PG3 nodes."""

    return _physics_reference_nodes(3)


def _reference_point(
    element: Element,
    a: float,
    b: float,
    *,
    ne: int = 3,
):
    x, y = _bilinear(_element_corners(element, ne), a, b)
    return _unit_sphere(element.face, math.tan(x), math.tan(y))


def _u2qmap(
    element: Element,
    nodes: np.ndarray,
    *,
    ne: int = 3,
) -> np.ndarray:
    corners = _element_corners(element, ne)
    last = len(nodes) - 1
    c11 = _bilinear(corners, float(nodes[0]), float(nodes[0]))
    c41 = _bilinear(corners, float(nodes[last]), float(nodes[0]))
    c44 = _bilinear(corners, float(nodes[last]), float(nodes[last]))
    c14 = _bilinear(corners, float(nodes[0]), float(nodes[last]))
    result = np.empty((4, 2), dtype=np.float64, order="F")
    for component in range(2):
        result[0, component] = (c11[component] + c41[component] + c44[component] + c14[component]) / 4.0
        result[1, component] = (-c11[component] + c41[component] + c44[component] - c14[component]) / 4.0
        result[2, component] = (-c11[component] - c41[component] + c44[component] + c14[component]) / 4.0
        result[3, component] = (c11[component] - c41[component] + c44[component] - c14[component]) / 4.0
    return result


def _vmap(face: int, x1: float, x2: float) -> np.ndarray:
    r = math.sqrt(1.0 + math.tan(x1) ** 2 + math.tan(x2) ** 2)
    result = np.empty((2, 2), dtype=np.float64, order="F")
    if 1 <= face <= 4:
        result[0, 0] = 1.0 / (r * math.cos(x1))
        result[0, 1] = 0.0
        result[1, 0] = -math.tan(x1) * math.tan(x2) / (math.cos(x1) * r * r)
        result[1, 1] = 1.0 / (r * r * math.cos(x1) * math.cos(x2) * math.cos(x2))
        return result
    poledist = math.sqrt(math.tan(x1) ** 2 + math.tan(x2) ** 2)
    if poledist <= 1.0e-9:
        result[...] = np.eye(2)
        return result
    sign = 1.0 if face == 5 else -1.0
    result[0, 0] = sign * math.tan(x2) / (poledist * math.cos(x1) * math.cos(x1) * r)
    result[0, 1] = -sign * math.tan(x1) / (poledist * math.cos(x2) * math.cos(x2) * r)
    result[1, 0] = sign * math.tan(x1) / (poledist * math.cos(x1) * math.cos(x1) * r * r)
    result[1, 1] = sign * math.tan(x2) / (poledist * math.cos(x2) * math.cos(x2) * r * r)
    return result


def _raw_d_at(
    element: Element,
    a: float,
    b: float,
    u2q: np.ndarray | None = None,
    *,
    ne: int = 3,
) -> np.ndarray:
    corners = _element_corners(element, ne)
    if u2q is None:
        nodes, _weights = _gll_nodes_weights(4)
        u2q = _u2qmap(element, nodes, ne=ne)
    jp11 = u2q[1, 0] + u2q[3, 0] * b
    jp12 = u2q[2, 0] + u2q[3, 0] * a
    jp21 = u2q[1, 1] + u2q[3, 1] * b
    jp22 = u2q[2, 1] + u2q[3, 1] * a
    x1, x2 = _bilinear(corners, a, b)
    tmp = _vmap(element.face, x1, x2)
    d = np.empty((2, 2), dtype=np.float64, order="F")
    d[0, 0] = tmp[0, 0] * jp11 + tmp[0, 1] * jp21
    d[0, 1] = tmp[0, 0] * jp12 + tmp[0, 1] * jp22
    d[1, 0] = tmp[1, 0] * jp11 + tmp[1, 1] * jp21
    d[1, 1] = tmp[1, 0] * jp12 + tmp[1, 1] * jp22
    return d


def _raw_metric(
    element: Element,
    nodes: np.ndarray,
    *,
    ne: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    np_value = len(nodes)
    u2q = _u2qmap(element, nodes, ne=ne)
    d = np.empty(
        (2, 2, np_value, np_value),
        dtype=np.float64,
        order="F",
    )
    metdet = np.empty((np_value, np_value), dtype=np.float64, order="F")
    for j in range(np_value):
        b = float(nodes[j])
        for i in range(np_value):
            a = float(nodes[i])
            d[:, :, i, j] = _raw_d_at(
                element,
                a,
                b,
                u2q,
                ne=ne,
            )
            det = d[0, 0, i, j] * d[1, 1, i, j] - d[0, 1, i, j] * d[1, 0, i, j]
            metdet[i, j] = abs(det)
    return d, metdet


def _subcell_integration_weights(
    nodes: np.ndarray,
    weights: np.ndarray,
    intervals: int = 3,
) -> np.ndarray:
    np_value = len(nodes)
    lagrange = np.empty(
        (intervals, np_value, np_value),
        dtype=np.float64,
        order="F",
    )
    denominator = np.empty(np_value, dtype=np.float64)
    for j in range(np_value):
        value = 1.0
        for m in range(np_value):
            if m != j:
                value = value * (nodes[j] - nodes[m])
        denominator[j] = value
    for cell in range(intervals):
        a = -1.0 + cell * 2.0 / intervals
        b = -1.0 + (cell + 1) * 2.0 / intervals
        for n in range(np_value):
            x = (a + b) / 2.0 + nodes[n] / intervals
            for j in range(np_value):
                value = 1.0
                for m in range(np_value):
                    if m != j:
                        value = value * (x - nodes[m])
                lagrange[cell, n, j] = value / denominator[j]
    result = np.empty((intervals, np_value), dtype=np.float64, order="F")
    for cell in range(intervals):
        for j in range(np_value):
            value = 0.0
            for n in range(np_value):
                value = value + weights[n] * lagrange[cell, n, j]
            result[cell, j] = value / intervals
    return result


def _subcell_boundary_weights(
    nodes: np.ndarray,
    intervals: int = 3,
) -> np.ndarray:
    """Return HOMME's cached ``boundary_interp_matrix`` in source order."""

    np_value = len(nodes)
    lagrange = np.empty(
        (intervals, np_value, np_value),
        dtype=np.float64,
        order="F",
    )
    denominator = np.empty(np_value, dtype=np.float64)
    for j in range(np_value):
        value = np.float64(1.0)
        for m in range(np_value):
            if m != j:
                value = np.float64(value * np.float64(nodes[j] - nodes[m]))
        denominator[j] = value
    for cell in range(intervals):
        a = np.float64(-1.0 + cell * 2.0 / intervals)
        b = np.float64(-1.0 + (cell + 1) * 2.0 / intervals)
        for n in range(np_value):
            x = np.float64(np.float64(a + b) / np.float64(2.0) + np.float64(nodes[n] / intervals))
            for j in range(np_value):
                value = np.float64(1.0)
                for m in range(np_value):
                    if m != j:
                        value = np.float64(value * np.float64(x - nodes[m]))
                lagrange[cell, n, j] = np.float64(value / denominator[j])
    result = np.empty(
        (intervals, 2, np_value),
        dtype=np.float64,
        order="F",
    )
    result[:, 0, :] = lagrange[:, 0, :]
    result[:, 1, :] = lagrange[:, np_value - 1, :]
    return result


def _interpolation_matrix(nodes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    np_value = len(nodes)

    def legendre(x: float) -> np.ndarray:
        values = np.empty(np_value, dtype=np.float64)
        p3 = 1.0
        values[0] = p3
        if np_value == 1:
            return values
        p2 = p3
        p3 = x
        values[1] = p3
        for k in range(2, np_value):
            p1 = p2
            p2 = p3
            p3 = ((2 * k - 1) * x * p2 - (k - 1) * p1) / k
            values[k] = p3
        return values

    gamma = np.zeros(np_value, dtype=np.float64)
    legs: list[np.ndarray] = []
    for i in range(np_value):
        values = legendre(float(nodes[i]))
        legs.append(values)
        for k in range(np_value):
            gamma[k] = gamma[k] + values[k] * values[k] * weights[i]
    result = np.empty(
        (np_value, np_value),
        dtype=np.float64,
        order="F",
    )
    for j in range(np_value):
        for k in range(np_value):
            result[j, k] = legs[j][k] * weights[j] / gamma[k]
    return result


def _physics_point(
    element: Element,
    pi: int,
    pj: int,
    ne: int = 3,
    fv_nphys: int = 3,
):
    """Reproduce compute_basic_coordinate_vars in its scalar FP order."""
    cube_start = -0.25 * math.pi
    cube_end = 0.25 * math.pi
    delta = (cube_end - cube_start) / ne
    start_alpha = cube_start + element.i * delta
    start_beta = cube_start + element.j * delta
    corner2_alpha = start_alpha + delta
    corner4_beta = start_beta + delta
    dalpha = abs(start_alpha - corner2_alpha) / fv_nphys
    dbeta = abs(start_beta - corner4_beta) / fv_nphys
    alpha = start_alpha + ((pi + 1) - 0.5) * dalpha
    beta = start_beta + ((pj + 1) - 0.5) * dbeta
    return _unit_sphere(element.face, math.tan(alpha), math.tan(beta))


@cache
def _gll_nodes_weights(
    np_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss-Lobatto-Legendre nodes and weights.

    The np=4 constants retain HOMME's existing BFB representation. Other
    orders use roots of the derivative of ``P_(np-1)``.
    """

    np_value = int(np_value)
    if np_value < 2:
        raise ValueError("np must be at least 2")
    if np_value == 4:
        nodes = np.array(
            (-1.0, -math.sqrt(0.2), math.sqrt(0.2), 1.0),
            dtype=np.float64,
        )
        weights = np.array(
            (1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0),
            dtype=np.float64,
        )
        return nodes, weights

    polynomial = np.polynomial.legendre.Legendre.basis(np_value - 1)
    interior = np.sort(polynomial.deriv().roots())
    nodes = np.concatenate(([-1.0], interior, [1.0])).astype(np.float64)
    values = polynomial(nodes)
    weights = (
        np.float64(2.0)
        / np.float64(np_value * (np_value - 1))
        / (values * values)
    )
    return nodes, weights.astype(np.float64)


def _global_dof_map(
    size: int = 24,
    ne: int = 3,
    np_value: int = 4,
):
    nodes, _weights = _gll_nodes_weights(np_value)
    key_to_dof: dict[tuple[int, int, int], int] = {}
    owners: dict[int, set[int]] = {}
    element_dofs: dict[int, np.ndarray] = {}
    # global_dof starts with (global_element-1)*np*np+local_node and
    # performs an edge MIN reduction.  These deliberately sparse IDs are
    # also how CreateUniqueIndex decides which occurrence initializes ICs.
    for element in global_elements(size, ne):
        for j, eta in enumerate(nodes):
            for i, xi in enumerate(nodes):
                _lon, _lat, xyz = _element_point(element, xi, eta, ne)
                key = tuple(int(round(value * 10**13)) for value in xyz)
                local_dof = (
                    (element.global_id - 1) * np_value * np_value
                    + j * np_value
                    + i
                    + 1
                )
                key_to_dof[key] = min(key_to_dof.get(key, local_dof), local_dof)
    for element in global_elements(size, ne):
        dofs = np.empty(
            (np_value, np_value),
            dtype=np.int64,
            order="F",
        )
        for j, eta in enumerate(nodes):
            for i, xi in enumerate(nodes):
                _lon, _lat, xyz = _element_point(element, xi, eta, ne)
                key = tuple(int(round(value * 10**13)) for value in xyz)
                dof = key_to_dof[key]
                owners.setdefault(dof, set()).add(element.owner)
                dofs[i, j] = dof
        element_dofs[element.global_id] = dofs
    return element_dofs, owners


def _halo_inventory(
    rank: int,
    size: int = 24,
    *,
    ne: int = 3,
    np_value: int = 4,
):
    element_dofs, dof_owners = _global_dof_map(size, ne, np_value)
    local = {
        int(dof)
        for elem in local_elements(rank, size, ne)
        for dof in element_dofs[elem.global_id].flat
    }
    shared: dict[int, list[int]] = {}
    for dof in sorted(local):
        for peer in sorted(dof_owners[dof] - {rank}):
            shared.setdefault(peer, []).append(dof)
    return sorted(shared), shared


def _derivative_matrix(nodes: np.ndarray) -> np.ndarray:
    """Return HOMME ``deriv%Dvv`` for the configured GLL grid.

    HOMME obtains the interior quadrature nodes through Newton iteration.
    Replacing those nodes with analytical ``sqrt(0.2)`` changes four Dvv
    coefficients by one bit, so retain the exact np=4 coefficients.
    """

    if len(nodes) == 4:
        h = float.fromhex
        return np.array(
            (
                (h("-0x1.8000000000000p+1"), h("-0x1.9e3779b97f4a7p-1"), h("0x1.3c6ef372fe950p-2"), h("-0x1.0000000000000p-1")),
                (h("0x1.02e2ac13ef8e8p+2"), 0.0, h("-0x1.1e3779b97f4a8p+0"), h("0x1.8b8ab04fbe3a4p+0")),
                (h("-0x1.8b8ab04fbe3a4p+0"), h("0x1.1e3779b97f4a8p+0"), 0.0, h("-0x1.02e2ac13ef8e8p+2")),
                (h("0x1.0000000000000p-1"), h("-0x1.3c6ef372fe950p-2"), h("0x1.9e3779b97f4a7p-1"), h("0x1.8000000000000p+1")),
            ),
            dtype=np.float64,
            order="F",
        )

    count = len(nodes)
    barycentric = np.ones(count, dtype=np.float64)
    for j in range(count):
        for k in range(count):
            if j != k:
                barycentric[j] /= nodes[j] - nodes[k]
    result = np.empty((count, count), dtype=np.float64, order="F")
    for i in range(count):
        for j in range(count):
            if i != j:
                result[i, j] = barycentric[j] / (
                    barycentric[i] * (nodes[i] - nodes[j])
                )
        result[i, i] = -sum(
            result[i, j] for j in range(count) if j != i
        )
    # The construction above is the conventional D[output, input] layout.
    # HOMME stores Dvv(input, output), and every scalar-order Python port
    # follows that Fortran indexing.  The np=4 table above is already in Dvv
    # layout, so transpose only the generic construction.
    return np.asfortranarray(result.T)


def _dmap(face: int, alpha: float, beta: float, ne: int = 3) -> np.ndarray:
    x, y = math.tan(alpha), math.tan(beta)
    r = math.sqrt(1.0 + x*x + y*y)
    if face <= 4:
        matrix = np.array(((1.0/(r*math.cos(alpha)), 0.0),
                           (-x*y/(math.cos(alpha)*r*r), 1.0/(r*r*math.cos(alpha)*math.cos(beta)**2))), dtype=np.float64)
    else:
        pole = math.sqrt(x*x + y*y)
        if pole <= 1.0e-14:
            matrix = np.eye(2, dtype=np.float64)
        elif face == 6:
            matrix = np.array(((-y/(pole*math.cos(alpha)**2*r), x/(pole*math.cos(beta)**2*r)),
                               (-x/(pole*math.cos(alpha)**2*r*r), -y/(pole*math.cos(beta)**2*r*r))), dtype=np.float64)
        else:
            matrix = np.array(((y/(pole*math.cos(alpha)**2*r), -x/(pole*math.cos(beta)**2*r)),
                               (x/(pole*math.cos(alpha)**2*r*r), y/(pole*math.cos(beta)**2*r*r))), dtype=np.float64)
    return matrix * (math.pi / (4.0 * ne))


def _cell_area(
    element: Element,
    pi: int,
    pj: int,
    *,
    ne: int = 3,
    fv_nphys: int = 3,
) -> float:
    """Reproduce the irecons=6 analytic I_00 area expression."""
    cube_start = -0.25 * math.pi
    cube_end = 0.25 * math.pi
    delta = (cube_end - cube_start) / ne
    start_alpha = cube_start + element.i * delta
    start_beta = cube_start + element.j * delta
    dalpha = abs(start_alpha - (start_alpha + delta)) / fv_nphys
    dbeta = abs(start_beta - (start_beta + delta)) / fv_nphys
    x0 = math.tan(start_alpha + pi * dalpha)
    x1 = math.tan(start_alpha + (pi + 1) * dalpha)
    y0 = math.tan(start_beta + pj * dbeta)
    y1 = math.tan(start_beta + (pj + 1) * dbeta)

    def i00(x: float, y: float) -> float:
        return math.atan(x * y / math.sqrt(1.0 + x * x + y * y))

    return i00(x1, y1) - i00(x0, y1) + i00(x0, y0) - i00(x1, y0)


def populate_grid(
    pool,
    rank: int,
    size: int,
    *,
    ne: int = 3,
) -> None:
    np_value = pool.dimensions["np"]
    fv_nphys = pool.dimensions["fv_nphys"]
    elements = local_elements(rank, size, ne)
    all_dofs, _owners = _global_dof_map(size, ne, np_value)
    nodes, weights = _gll_nodes_weights(np_value)
    pool.set("gll_node", nodes)
    pool.set("gll_weight", weights)
    pool.set("gll_derivative", _derivative_matrix(nodes))
    raw_metrics: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    element_areas: list[float] = []
    for global_element in global_elements(size, ne):
        d, metdet = _raw_metric(global_element, nodes, ne=ne)
        raw_metrics[global_element.global_id] = d, metdet
        area = 0.0
        for j in range(np_value):
            for i in range(np_value):
                area = area + weights[i] * weights[j] * metdet[i, j]
        element_areas.append(area)
    metric_normalization = 4.0 * math.pi / math.fsum(element_areas)
    metric_scale = math.sqrt(metric_normalization)
    for le, element in enumerate(elements):
        corners = _element_corners(element, ne)
        u2q = _u2qmap(element, nodes, ne=ne)
        pool.get("global_element_id")[le] = element.global_id
        pool.get("cube_face")[le] = element.face
        pool.get("cube_element_i")[le] = element.i + 1
        pool.get("cube_element_j")[le] = element.j + 1
        pool.get("space_filling_curve_index")[le] = element.sfc
        pool.get("element_owner_rank")[le] = rank
        pool.get("cube_corner_angle")[:, :, le] = np.asarray(corners, dtype=np.float64).T
        pool.get("mapping_uniform_to_quadrilateral")[:, :, le] = u2q
        pool.get("gll_global_dof")[:, :, le] = all_dofs[element.global_id]
        for j, eta in enumerate(nodes):
            for i, xi in enumerate(nodes):
                lon, lat, xyz = _reference_point(
                    element,
                    float(xi),
                    float(eta),
                    ne=ne,
                )
                pool.get("gll_longitude")[i, j, le] = lon
                pool.get("gll_latitude")[i, j, le] = lat
                pool.get("gll_cartesian")[:, i, j, le] = xyz
                raw_d, raw_metdet = raw_metrics[element.global_id]
                dmap = raw_d[:, :, i, j] * metric_scale
                pool.get("metric_derivative")[:,:,i,j,le] = dmap
                jac = raw_metdet[i, j] * metric_normalization
                pool.get("metric_jacobian")[i, j, le] = jac
                pool.get("inverse_metric_jacobian")[i, j, le] = (
                    (np.float64(1.0) / raw_metdet[i, j]) / metric_normalization
                )
                # mass_matrix_mod forms mp=w_i*w_j first, then spheremp=mp*metdet.
                pool.get("spectral_mass_matrix")[i, j, le] = np.float64(weights[i] * weights[j]) * jac
                pool.get("inverse_spectral_mass_matrix")[i, j, le] = np.float64(1.0) / pool.get("spectral_mass_matrix")[i, j, le]
                # The source computes inv(D_raw) first and only then applies
                # the metric normalization.  Inverting normalized D changes
                # the last bit, so keep the original expression order.
                raw = raw_d[:, :, i, j]
                raw_det = raw[0, 0] * raw[1, 1] - raw[0, 1] * raw[1, 0]
                pool.get("inverse_metric")[0,0,i,j,le] = (raw[1, 1] / raw_det) / metric_scale
                pool.get("inverse_metric")[0,1,i,j,le] = (-raw[0, 1] / raw_det) / metric_scale
                pool.get("inverse_metric")[1,0,i,j,le] = (-raw[1, 0] / raw_det) / metric_scale
                pool.get("inverse_metric")[1,1,i,j,le] = (raw[0, 0] / raw_det) / metric_scale
                # cube_mod forms met and metinv from the unnormalized D, then
                # applies metinv/alpha.  Retain that state explicitly because
                # rebuilding it from normalized D changes low bits.
                met11 = np.float64(
                    np.float64(raw[0, 0] * raw[0, 0])
                    + np.float64(raw[1, 0] * raw[1, 0])
                )
                met12 = np.float64(
                    np.float64(raw[0, 0] * raw[0, 1])
                    + np.float64(raw[1, 0] * raw[1, 1])
                )
                met22 = np.float64(
                    np.float64(raw[0, 1] * raw[0, 1])
                    + np.float64(raw[1, 1] * raw[1, 1])
                )
                det_squared = np.float64(raw_det * raw_det)
                metinv = pool.get("inverse_metric_tensor")
                metinv[0, 0, i, j, le] = np.float64(
                    np.float64(met22 / det_squared) / metric_normalization
                )
                metinv[0, 1, i, j, le] = np.float64(
                    np.float64(-met12 / det_squared) / metric_normalization
                )
                metinv[1, 0, i, j, le] = np.float64(
                    np.float64(-met12 / det_squared) / metric_normalization
                )
                metinv[1, 1, i, j, le] = np.float64(
                    np.float64(met11 / det_squared) / metric_normalization
                )
        physics_nodes = _physics_reference_nodes(fv_nphys)
        for pj, eta in enumerate(physics_nodes):
            for pi, xi in enumerate(physics_nodes):
                col = le * fv_nphys * fv_nphys + pi + fv_nphys * pj
                lon, lat, _xyz = _physics_point(
                    element,
                    pi,
                    pj,
                    ne,
                    fv_nphys,
                )
                pool.get("physics_longitude")[col] = lon
                pool.get("physics_latitude")[col] = lat
                # The rank decomposition follows the SFC, while CAM history
                # and physics-grid global indices follow face-major element
                # numbering. Do not conflate these two orderings.
                pool.get("physics_global_column")[pi, pj, le] = (
                    (element.global_id - 1) * fv_nphys * fv_nphys
                    + pi
                    + fv_nphys * pj
                    + 1
                )
                pool.get("physics_cell_area")[col] = _cell_area(
                    element,
                    pi,
                    pj,
                    ne=ne,
                    fv_nphys=fv_nphys,
                )
                pool.get("mapping_derivative_pg3")[:, :, col] = _raw_d_at(
                    element,
                    float(xi),
                    float(eta),
                    u2q,
                    ne=ne,
                )
    peers, shared = _halo_inventory(
        rank,
        size,
        ne=ne,
        np_value=np_value,
    )
    if not peers:
        pool.get("halo_peer_rank")[0] = -1
    offset = 0
    pool.get("halo_shared_dof_offset")[0] = 0
    for ip, peer in enumerate(peers):
        values = shared[peer]
        pool.get("halo_peer_rank")[ip] = peer
        pool.get("halo_shared_dof_count")[ip] = len(values)
        pool.get("halo_shared_dof")[offset:offset+len(values)] = values
        offset += len(values)
        pool.get("halo_shared_dof_offset")[ip+1] = offset
    # Point interpolation is used for vectors; conservative scalar mapping
    # uses the separately stored subcell integration matrix.
    src, dst = nodes, _physics_reference_nodes(fv_nphys)
    w = np.empty(
        (fv_nphys, np_value),
        dtype=np.float64,
        order="F",
    )
    for i, x in enumerate(dst):
        for j in range(np_value):
            value = 1.0
            for k in range(np_value):
                if k != j:
                    value *= (x - src[k]) / (src[j] - src[k])
            w[i, j] = value
    pool.set("mapping_weights_gll_to_pg3", w)
    pool.set("mapping_weights_pg3_to_gll", np.linalg.pinv(w))
    pool.set(
        "mapping_subcell_integration",
        _subcell_integration_weights(
            nodes,
            weights,
            intervals=fv_nphys,
        ),
    )
    pool.set(
        "mapping_boundary_interpolation",
        _subcell_boundary_weights(nodes, intervals=fv_nphys),
    )
    pool.set("mapping_interpolation_matrix", _interpolation_matrix(nodes, weights))
    _initialize_physgrid_halo(pool, elements, size, ne)
