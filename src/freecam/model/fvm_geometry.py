"""Pure-Python construction of configurable CSLAM finite-volume geometry."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .grid import (
    Element,
    _cartesian_to_face_angles,
    _cross_face_cell,
    _element_corners,
    _fortran_cubedsphere_to_cart,
    _fvm_cube_boundary,
    global_elements,
)


def _f64(value: Any) -> np.float64:
    return np.float64(value)


def _i00(x: np.float64, y: np.float64) -> np.float64:
    return _f64(math.atan(_f64(x * y) / math.sqrt(_f64(1.0) + x * x + y * y)))


def _i10(x: np.float64, y: np.float64) -> np.float64:
    # GNU folds COS(ATAN(x)) to a reciprocal square root at -O2.  Preserve the
    # multiplication by that reciprocal; y / sqrt(...) rounds differently for
    # a few grid vertices and then gets amplified by the corner differences.
    reciprocal = _f64(_f64(1.0) / math.sqrt(_f64(x * x + _f64(1.0))))
    tmp = _f64(y * reciprocal)
    return _f64(-math.log(tmp + math.sqrt(tmp * tmp + _f64(1.0))))


def _i01(x: np.float64, y: np.float64) -> np.float64:
    tmp = _f64(x / math.sqrt(_f64(1.0) + y * y))
    return _f64(-math.log(tmp + math.sqrt(tmp * tmp + _f64(1.0))))


def _i20(x: np.float64, y: np.float64) -> np.float64:
    tmp = _f64(_f64(1.0) + y * y)
    tmp1 = _f64(x / math.sqrt(tmp))
    return _f64(
        y * math.log(tmp1 + math.sqrt(tmp1 * tmp1 + _f64(1.0)))
        + math.acos(_f64(x * y) / math.sqrt(_f64((_f64(1.0) + x * x) * tmp)))
    )


def _i02(x: np.float64, y: np.float64) -> np.float64:
    tmp = _f64(_f64(1.0) + x * x)
    tmp1 = _f64(y / math.sqrt(tmp))
    return _f64(
        x * math.log(tmp1 + math.sqrt(tmp1 * tmp1 + _f64(1.0)))
        + math.acos(_f64(x * y) / math.sqrt(_f64(tmp * (_f64(1.0) + y * y))))
    )


def _i11(x: np.float64, y: np.float64) -> np.float64:
    return _f64(-math.sqrt(_f64(1.0) + x * x + y * y))


def _rectangle_integral(
    function, x0: np.float64, x1: np.float64, y0: np.float64, y1: np.float64
) -> np.float64:
    # Preserve Fortran's left-associated expression exactly:
    # f(x1,y1) - f(x0,y1) + f(x0,y0) - f(x1,y0).
    value = _f64(function(x1, y1) - function(x0, y1))
    value = _f64(value + function(x0, y0))
    return _f64(value - function(x1, y0))


def _basic_coordinates(
    element: Element,
    *,
    ne: int,
    nc: int,
    irecons: int,
) -> tuple[np.float64, np.float64, np.ndarray, np.ndarray, np.ndarray]:
    corners = _element_corners(element, ne)
    dalpha = _f64(abs(_f64(corners[0][0] - corners[1][0])) / _f64(nc))
    dbeta = _f64(abs(_f64(corners[0][1] - corners[3][1])) / _f64(nc))
    acartx = np.empty(nc + 1, dtype=np.float64)
    acarty = np.empty(nc + 1, dtype=np.float64)
    for index in range(nc + 1):
        acartx[index] = math.tan(_f64(corners[0][0]) + _f64(index) * dalpha)
        acarty[index] = math.tan(_f64(corners[0][1]) + _f64(index) * dbeta)

    vertices = np.full((4, 2, nc, nc), -9.0e9, dtype=np.float64, order="F")
    area = np.empty((nc, nc), dtype=np.float64, order="F")
    centroid = np.empty((irecons - 1, nc, nc), dtype=np.float64, order="F")
    functions = (_i10, _i01, _i20, _i02, _i11)
    for j in range(nc):
        for i in range(nc):
            x0, x1 = acartx[i], acartx[i + 1]
            y0, y1 = acarty[j], acarty[j + 1]
            vertices[:, :, i, j] = (
                (x0, y0),
                (x1, y0),
                (x1, y1),
                (x0, y1),
            )
            area[i, j] = _rectangle_integral(_i00, x0, x1, y0, y1)
            for moment, function in enumerate(functions):
                centroid[moment, i, j] = _f64(
                    _rectangle_integral(function, x0, x1, y0, y1) / area[i, j]
                )
    return dalpha, dbeta, vertices, area, centroid


def _init_flux_orientation(
    face: int,
    boundary: int,
    *,
    nc: int,
    nhc: int,
) -> tuple[np.ndarray, np.ndarray]:
    halo = nc + 2 * nhc
    orientation = np.full((2, halo, halo), 99.9e9, dtype=np.float64, order="F")
    indicator = np.ones((halo, halo), dtype=np.int32, order="F")
    orientation[0, nhc : nhc + nc, nhc : nhc + nc] = _f64(face)
    orientation[1, :, :] = _f64(0.0)
    if boundary == 0:
        return orientation, indicator

    west, east, south, north = 1, 2, 3, 4
    southwest, southeast, northwest, northeast = 5, 6, 7, 8
    south_set = (south, southwest, southeast)
    north_set = (north, northwest, northeast)
    west_set = (west, southwest, northwest)
    east_set = (east, southeast, northeast)
    if face == 2:
        if boundary in north_set:
            orientation[1, :, nhc + nc :] = 1
        if boundary in south_set:
            orientation[1, :, :nhc] = 3
    elif face == 3:
        if boundary in north_set:
            orientation[1, :, nhc + nc :] = 2
        if boundary in south_set:
            orientation[1, :, :nhc] = 2
    elif face == 4:
        if boundary in north_set:
            orientation[1, :, nhc + nc :] = 3
        if boundary in south_set:
            orientation[1, :, :nhc] = 1
    elif face == 5:
        if boundary in south_set:
            orientation[1, :, :nhc] = 2
        if boundary in west_set:
            orientation[1, :nhc, :] = 3
        if boundary in east_set:
            orientation[1, nhc + nc :, :] = 1
    elif face == 6:
        if boundary in north_set:
            orientation[1, :, nhc + nc :] = 2
        if boundary in west_set:
            orientation[1, :nhc, :] = 1
        if boundary in east_set:
            orientation[1, nhc + nc :, :] = 3

    if boundary == northwest:
        orientation[1, :nhc, nhc + nc :] = 0
        indicator[:nhc, nhc + nc :] = 0
    elif boundary == southwest:
        orientation[1, :nhc, :nhc] = 0
        indicator[:nhc, :nhc] = 0
    elif boundary == northeast:
        orientation[1, nhc + nc :, nhc + nc :] = 0
        indicator[nhc + nc :, nhc + nc :] = 0
    elif boundary == southeast:
        orientation[1, nhc + nc :, :nhc] = 0
        indicator[nhc + nc :, :nhc] = 0
    return orientation, indicator


def _halo_ranges(
    boundary: int,
    *,
    nc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Store the Fortran values verbatim; the transport kernel consumes them in
    # the original one-based coordinate convention.
    jx_min = np.array((0, 0, 0), dtype=np.int32)
    jx_max = np.array((-1, -1, -1), dtype=np.int32)
    jy_min = np.array((0, 0, 0), dtype=np.int32)
    jy_max = np.array((-1, -1, -1), dtype=np.int32)
    high = nc + 2
    inside_high = nc + 1
    if boundary == 0:
        jx_min[0], jx_max[0], jy_min[0], jy_max[0] = 0, high, 0, high
    elif boundary == 1:
        jx_min[:2], jx_max[:2] = (1, 0), (high, 1)
        jy_min[:2], jy_max[:2] = (0, 0), (high, high)
    elif boundary == 2:
        jx_min[:2], jx_max[:2] = (0, inside_high), (inside_high, high)
        jy_min[:2], jy_max[:2] = (0, 0), (high, high)
    elif boundary == 4:
        jx_min[:2], jx_max[:2] = (0, 0), (high, high)
        jy_min[:2], jy_max[:2] = (0, inside_high), (inside_high, high)
    elif boundary == 3:
        jx_min[:2], jx_max[:2] = (0, 0), (high, high)
        jy_min[:2], jy_max[:2] = (1, 0), (high, 1)
    elif boundary == 5:
        jx_min[:3], jx_max[:3] = (1, 1, 0), (high, high, 1)
        jy_min[:3], jy_max[:3] = (1, 0, 1), (high, 1, high)
    elif boundary == 6:
        jx_min[:3], jx_max[:3] = (0, 0, inside_high), (inside_high, inside_high, high)
        jy_min[:3], jy_max[:3] = (1, 0, 1), (high, 1, high)
    elif boundary == 8:
        jx_min[:3], jx_max[:3] = (0, 0, inside_high), (inside_high, inside_high, high)
        jy_min[:3], jy_max[:3] = (0, inside_high, 0), (inside_high, high, inside_high)
    elif boundary == 7:
        jx_min[:3], jx_max[:3] = (1, 1, 0), (high, high, 1)
        jy_min[:3], jy_max[:3] = (0, inside_high, 0), (inside_high, high, inside_high)
    else:
        raise ValueError(f"invalid FVM cube boundary {boundary}")
    return jx_min, jx_max, jy_min, jy_max


def _rotation_matrices(orientation: np.ndarray, boundary: int) -> np.ndarray:
    halo = orientation.shape[1]
    result = np.empty((2, 2, halo, halo), dtype=np.int32, order="F")
    result[0, 0] = 1
    result[0, 1] = 0
    result[1, 0] = 0
    result[1, 1] = 1
    if boundary == 0:
        return result
    clockwise = np.array(((0, 1), (-1, 0)), dtype=np.int32, order="F")
    for j in range(halo):
        for i in range(halo):
            for _ in range(4 - int(np.rint(orientation[1, i, j]))):
                result[:, :, i, j] = clockwise @ result[:, :, i, j]
    return result


def _global_cell(
    element: Element,
    hi: int,
    hj: int,
    *,
    ne: int,
    nc: int,
    nhc: int,
) -> tuple[int, int, int] | None:
    gx = element.i * nc + hi - nhc
    gy = element.j * nc + hj - nhc
    return _cross_face_cell(element.face, gx, gy, ne * nc)


def _source_location(
    mapped: tuple[int, int, int],
    *,
    ne: int,
    nc: int,
) -> tuple[int, int, int]:
    face, gx, gy = mapped
    ei, pi = divmod(gx, nc)
    ej, pj = divmod(gy, nc)
    gid = 1 + ei + ne * ej + ne * ne * (face - 1)
    return gid - 1, pi, pj


def _get_gnomonic_point(alpha: np.float64, beta: np.float64, face: int, component: int) -> np.float64:
    xyz = _fortran_cubedsphere_to_cart(1, alpha, beta)
    converted = _cartesian_to_face_angles(face, xyz)
    return _f64(converted[component])


def _interpolation_point(
    alpha: np.float64,
    beta: np.float64,
    gnomonic: dict[int, np.float64],
    face: int,
    component: int,
    ida: int,
    ide: int,
    ns: int,
) -> tuple[np.float64, int]:
    point = _get_gnomonic_point(alpha, beta, face, component)
    reference = ida
    while point > gnomonic[reference]:
        reference += 1
        if reference > ide:
            reference = ide
            break
    if ns % 2:
        reference = max(reference, ida + 1)
        if gnomonic[reference] - point > point - gnomonic[reference - 1]:
            reference -= 1
        reference -= (ns - 1) // 2
    else:
        reference -= ns // 2
    base = min(max(reference, ida), ide - (ns - 1))
    return _f64(point - gnomonic[base]), base


def _equispace_weights(
    delta: np.float64,
    point: np.float64,
    *,
    ns: int,
) -> np.ndarray:
    weights = np.ones(ns, dtype=np.float64)
    for j in range(ns):
        for k in range(ns):
            if k != j:
                weights[j] = _f64(
                    weights[j]
                    * _f64(point - _f64(k) * delta)
                    / _f64(_f64(j) * delta - _f64(k) * delta)
                )
    return weights


def _interpolation_geometry(
    element: Element,
    boundary: int,
    dalpha: np.float64,
    dbeta: np.float64,
    *,
    ne: int,
    nc: int,
    nhc: int,
    nhe: int,
    nhr: int,
    ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    nh = nhr + nhe - 1
    bases = np.full(
        (2 * nh + nc, nhr, 2),
        99999,
        dtype=np.int32,
        order="F",
    )
    weights = np.full(
        (ns, 2 * nh + nc, nhr, 2),
        9.99e9,
        dtype=np.float64,
        order="F",
    )
    if boundary == 0:
        return bases, weights

    cube_start, cube_end = _f64(-0.25 * math.pi), _f64(0.25 * math.pi)
    gxs: dict[int, np.float64] = {}
    gxe: dict[int, np.float64] = {}
    gys: dict[int, np.float64] = {}
    gye: dict[int, np.float64] = {}

    if boundary <= 4:
        corners = _element_corners(element, ne)
        gxs[1 - nhc] = _f64(corners[0][0] - _f64(nhc - 0.5) * dalpha)
        gys[1 - nhc] = _f64(corners[0][1] - _f64(nhc - 0.5) * dbeta)
        for index in range(2 - nhc, nc + nhc + 1):
            gxs[index] = _f64(gxs[index - 1] + dalpha)
            gys[index] = _f64(gys[index - 1] + dbeta)
    else:
        gxs[1 - nhc] = _f64(cube_start - _f64(nhc - 0.5) * dalpha)
        gxe[nc + nhc] = _f64(cube_end + _f64(nhc - 0.5) * dalpha)
        gys[1 - nhc] = _f64(cube_start - _f64(nhc - 0.5) * dbeta)
        gye[nc + nhc] = _f64(cube_end + _f64(nhc - 0.5) * dbeta)
        for index in range(2 - nhc, nc + nhc + 1):
            gxs[index] = _f64(gxs[index - 1] + dalpha)
            gys[index] = _f64(gys[index - 1] + dbeta)
            reverse = nc + 1 - index
            gxe[reverse] = _f64(gxe[reverse + 1] - dalpha)
            gye[reverse] = _f64(gye[reverse + 1] - dbeta)

    interpolation: dict[tuple[int, int, int], np.float64] = {}
    temporary_base: dict[tuple[int, int, int], int] = {}
    if boundary <= 4:
        ida, ide = 1 - nhc, nc + nhc
        for halo in range(1, nhr + 1):
            for index in range(halo - nh, nc + nh - (halo - 1) + 1):
                if boundary == 1:
                    alpha, beta, face, component, values = _f64(cube_start - _f64(halo - 0.5) * dalpha), gys[index], 4, 1, gys
                    slot = 0
                elif boundary == 2:
                    alpha, beta, face, component, values = _f64(cube_end + _f64(halo - 0.5) * dalpha), gys[index], 2, 1, gys
                    slot = 0
                elif boundary == 4:
                    alpha, beta, face, component, values = gxs[index], _f64(cube_end + _f64(halo - 0.5) * dbeta), 6, 0, gxs
                    slot = 1
                else:
                    alpha, beta, face, component, values = gxs[index], _f64(cube_start - _f64(halo - 0.5) * dbeta), 5, 0, gxs
                    slot = 1
                point, base = _interpolation_point(
                    alpha, beta, values, face, component, ida, ide, ns
                )
                interpolation[index, halo, slot] = point
                temporary_base[index, halo, slot] = base
    else:
        for halo in range(1, nhr + 1):
            if boundary in (5, 6):
                indices = range(0, nc + nh - (halo - 1) + 1)
                values, ida, ide = gys, 1, nc + nhc
                beta_values = gys
            else:
                indices = range(halo - nh, nc + 2)
                values, ida, ide = gye, 1 - nhc, nc
                beta_values = gye
            for index in indices:
                if boundary in (5, 7):
                    alpha, face = _f64(cube_start - _f64(halo - 0.5) * dalpha), 4
                else:
                    alpha, face = _f64(cube_end + _f64(halo - 0.5) * dalpha), 2
                point, base = _interpolation_point(
                    alpha, beta_values[index], values, face, 1, ida, ide, ns
                )
                interpolation[index, halo, 0] = point
                temporary_base[index, halo, 0] = base

    def storage(index: int) -> int:
        return index + nh - 1

    if boundary < 5:
        slot = 0 if boundary in (1, 2) else 1
        for halo in range(1, nhr + 1):
            for index in range(halo - nh, nc + nh - (halo - 1) + 1):
                base = temporary_base[index, halo, slot]
                bases[storage(index), halo - 1, 0] = base
                weights[:, storage(index), halo - 1, 0] = _equispace_weights(
                    dbeta, interpolation[index, halo, slot], ns=ns
                )
    else:
        for halo in range(1, nhr + 1):
            if boundary in (5, 6):
                imin, imax = 0, nc + nh - (halo - 1)
                jmin, jmax = halo - nh, nc + 1
            else:
                jmin, jmax = 0, nc + nh - (halo - 1)
                imin, imax = halo - nh, nc + 1
            for index in range(imin, imax + 1):
                base = temporary_base[index, halo, 0]
                bases[storage(index), halo - 1, 0] = base
                weights[:, storage(index), halo - 1, 0] = _equispace_weights(
                    dbeta, interpolation[index, halo, 0], ns=ns
                )
            destination = list(range(jmin, jmax + 1))
            source = list(range(imax, imin - 1, -1))
            for dst, src in zip(destination, source):
                weights[:, storage(dst), halo - 1, 1] = weights[::-1, storage(src), halo - 1, 0]
                bases[storage(dst), halo - 1, 1] = nc + 1 - (ns - 1) - bases[storage(src), halo - 1, 0]
    return bases, weights


def _reconstruction_geometry(
    centroid: np.ndarray,
    vertices: np.ndarray,
    dalpha: np.float64,
    dbeta: np.float64,
    *,
    nc: int,
    nhc: int,
    nhe: int,
    nht: int,
    irecons: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    internal = nc + 2 * nhe
    stretch_count = nc + nht + 1
    moment_count = irecons - 1
    stretch = np.zeros(
        (stretch_count, internal, internal),
        dtype=np.float64,
        order="F",
    )
    vertex_weights = np.empty(
        (4, moment_count, internal, internal),
        dtype=np.float64,
        order="F",
    )
    metric = np.empty(
        (irecons - 3, internal, internal),
        dtype=np.float64,
        order="F",
    )
    metric_integral = np.empty_like(metric, order="F")
    for j in range(internal):
        hj = j + nhc - nhe
        for i in range(internal):
            hi = i + nhc - nhe
            sx, sy, sx2, sy2, sxy = centroid[:, hi, hj]
            for vertex in range(4):
                x, y = vertices[vertex, :, hi, hj]
                vertex_weights[vertex, 0, i, j] = _f64(x - sx)
                vertex_weights[vertex, 1, i, j] = _f64(y - sy)
                vertex_weights[vertex, 2, i, j] = _f64(_f64(sx * sx - sx2) + _f64((x - sx) ** 2))
                vertex_weights[vertex, 3, i, j] = _f64(_f64(sy * sy - sy2) + _f64((y - sy) ** 2))
                vertex_weights[vertex, 4, i, j] = _f64(
                    _f64((x - sx) * (y - sy)) + _f64(sx * sy - sxy)
                )
            metric[0, i, j] = _f64(sx * sx - sx2)
            metric[1, i, j] = _f64(sy * sy - sy2)
            metric[2, i, j] = _f64(sx * sy - sxy)
            metric_integral[0, i, j] = _f64(_f64(2.0) * sx * sx - sx2)
            metric_integral[1, i, j] = _f64(_f64(2.0) * sy * sy - sy2)
            metric_integral[2, i, j] = _f64(_f64(2.0) * sx * sy - sxy)
            coef = _f64(_f64(1.0) / _f64(_f64(12.0) * dalpha))
            stretch[0, i, j] = _f64(coef / _f64(_f64(1.0) + sx * sx))
            coef = _f64(_f64(1.0) / _f64(_f64(12.0) * dbeta))
            stretch[1, i, j] = _f64(coef / _f64(_f64(1.0) + sy * sy))
            coef = _f64(_f64(1.0) / _f64(_f64(12.0) * dalpha**2))
            tmp = _f64(_f64(0.5) / _f64((_f64(1.0) + sx * sx) ** 2))
            stretch[2, i, j] = _f64(coef * tmp)
            stretch[5, i, j] = _f64(-sx / _f64(_f64(1.0) + sx * sx))
            coef = _f64(_f64(1.0) / _f64(_f64(12.0) * dbeta**2))
            tmp = _f64(_f64(0.5) / _f64((_f64(1.0) + sy * sy) ** 2))
            stretch[3, i, j] = _f64(coef * tmp)
            stretch[6, i, j] = _f64(-sy / _f64(_f64(1.0) + sy * sy))
            coef = _f64(_f64(1.0) / _f64(_f64(4.0) * dalpha * dbeta))
            stretch[4, i, j] = _f64(
                coef
                / _f64(
                    _f64(_f64(1.0) + sx * sx)
                    * _f64(_f64(1.0) + sy * sy)
                )
            )
    return stretch, vertex_weights, metric, metric_integral


def generate_fvm_geometry(
    hyai: np.ndarray,
    hybi: np.ndarray,
    reference_pressure: float,
    *,
    ne: int = 3,
    nc: int = 3,
    nhc: int = 3,
    nhe: int = 1,
    nhr: int = 2,
    nht: int = 3,
    ns: int | None = None,
    irecons: int = 6,
) -> dict[str, np.ndarray]:
    """Generate every persistent FVM geometry field without native code or files."""

    # Geometry is global and independent of the runtime MPI decomposition.
    if ns is None:
        ns = nc
    if min(ne, nc, nhc, nhe, nhr, nht, ns) <= 0:
        raise ValueError("FVM geometry dimensions must be positive")
    if irecons != 6:
        raise ValueError("the current CSLAM reconstruction requires irecons=6")
    halo = nc + 2 * nhc
    internal = nc + 2 * nhe
    interpolation_span = nc + 2 * nhr
    stretch_count = nc + nht + 1
    elements = tuple(
        sorted(global_elements(1, ne), key=lambda item: item.global_id)
    )
    count = len(elements)
    result: dict[str, np.ndarray] = {
        "global_element_id": np.arange(1, count + 1, dtype=np.int32),
        "cube_boundary": np.empty(count, dtype=np.int32),
        "dp_ref": np.empty((len(hyai) - 1, count), dtype=np.float64, order="F"),
        "dp_ref_inverse": np.empty((len(hyai) - 1, count), dtype=np.float64, order="F"),
        "area_sphere": np.empty((nc, nc, count), dtype=np.float64, order="F"),
        "inverse_area_sphere": np.empty((nc, nc, count), dtype=np.float64, order="F"),
        "displacement_maximum": np.empty((halo, halo, 4, count), dtype=np.float64, order="F"),
        "flux_vector": np.empty((2, halo, halo, 4, count), dtype=np.int32, order="F"),
        "vertex_cartesian": np.full((4, 2, halo, halo, count), -9.0e9, dtype=np.float64, order="F"),
        "flux_orientation": np.empty((2, halo, halo, count), dtype=np.float64, order="F"),
        "cell_indicator": np.empty((halo, halo, count), dtype=np.int32, order="F"),
        "rotation_matrix": np.empty((2, 2, halo, halo, count), dtype=np.int32, order="F"),
        "sphere_centroid": np.empty((irecons - 1, halo, halo, count), dtype=np.float64, order="F"),
        "reconstruction_metric": np.empty((irecons - 3, internal, internal, count), dtype=np.float64, order="F"),
        "reconstruction_metric_integral": np.empty((irecons - 3, internal, internal, count), dtype=np.float64, order="F"),
        "jx_min": np.empty((3, count), dtype=np.int32, order="F"),
        "jx_max": np.empty((3, count), dtype=np.int32, order="F"),
        "jy_min": np.empty((3, count), dtype=np.int32, order="F"),
        "jy_max": np.empty((3, count), dtype=np.int32, order="F"),
        "interpolation_base": np.empty((interpolation_span, 2, nhr, count), dtype=np.int32, order="F"),
        "halo_interpolation_weight": np.empty((ns, interpolation_span, 2, nhr, count), dtype=np.float64, order="F"),
        "centroid_stretch": np.empty((stretch_count, internal, internal, count), dtype=np.float64, order="F"),
        "vertex_reconstruction_weight": np.empty((4, irecons - 1, internal, internal, count), dtype=np.float64, order="F"),
    }
    delta_pressure = np.empty(len(hyai) - 1, dtype=np.float64)
    ps0 = _f64(reference_pressure)
    for level in range(len(delta_pressure)):
        delta_pressure[level] = _f64(
            _f64(_f64(hyai[level + 1] - hyai[level]) * ps0)
            + _f64(_f64(hybi[level + 1] - hybi[level]) * ps0)
        )
    result["dp_ref"][:] = delta_pressure[:, None]
    result["dp_ref_inverse"][:] = (_f64(1.0) / delta_pressure)[:, None]

    dalpha = np.empty(count, dtype=np.float64)
    dbeta = np.empty(count, dtype=np.float64)
    for index, element in enumerate(elements):
        boundary = _fvm_cube_boundary(element, ne)
        result["cube_boundary"][index] = boundary
        da, db, vertices, area, centroid = _basic_coordinates(
            element,
            ne=ne,
            nc=nc,
            irecons=irecons,
        )
        dalpha[index], dbeta[index] = da, db
        result["vertex_cartesian"][:, :, nhc : nhc + nc, nhc : nhc + nc, index] = vertices
        result["area_sphere"][:, :, index] = area
        result["inverse_area_sphere"][:, :, index] = _f64(1.0) / area
        result["sphere_centroid"][:, nhc : nhc + nc, nhc : nhc + nc, index] = centroid
        orientation, indicator = _init_flux_orientation(
            element.face,
            boundary,
            nc=nc,
            nhc=nhc,
        )
        result["flux_orientation"][:, :, :, index] = orientation
        result["cell_indicator"][:, :, index] = indicator
        result["rotation_matrix"][:, :, :, :, index] = _rotation_matrices(orientation, boundary)
        ranges = _halo_ranges(boundary, nc=nc)
        for name, values in zip(("jx_min", "jx_max", "jy_min", "jy_max"), ranges):
            result[name][:, index] = values
        bases, weights = _interpolation_geometry(
            element,
            boundary,
            da,
            db,
            ne=ne,
            nc=nc,
            nhc=nhc,
            nhe=nhe,
            nhr=nhr,
            ns=ns,
        )
        result["interpolation_base"][:, :, :, index] = bases
        result["halo_interpolation_weight"][:, :, :, :, index] = weights

    # Reproduce fvm_init3's ghostpack/exchange/unpack for geometry arrays.
    for index, element in enumerate(elements):
        for hj in range(halo):
            for hi in range(halo):
                if nhc <= hi < nhc + nc and nhc <= hj < nhc + nc:
                    continue
                mapped = _global_cell(
                    element,
                    hi,
                    hj,
                    ne=ne,
                    nc=nc,
                    nhc=nhc,
                )
                if mapped is None:
                    continue
                source, pi, pj = _source_location(
                    mapped,
                    ne=ne,
                    nc=nc,
                )
                result["vertex_cartesian"][:, :, hi, hj, index] = result["vertex_cartesian"][:, :, nhc + pi, nhc + pj, source]
                result["flux_orientation"][0, hi, hj, index] = result["flux_orientation"][0, nhc + pi, nhc + pj, source]
                result["sphere_centroid"][:, hi, hj, index] = result["sphere_centroid"][:, nhc + pi, nhc + pj, source]

    counterclockwise = np.array(((0, -1), (1, 0)), dtype=np.int32, order="F")
    unit = np.empty((2, 4), dtype=np.int32, order="F")
    unit[:, 0] = (0, 1)
    for side in range(1, 4):
        unit[:, side] = counterclockwise @ unit[:, side - 1]
    for index, element in enumerate(elements):
        boundary = int(result["cube_boundary"][index])
        if boundary == 7:
            result["flux_orientation"][:, :nhc, nhc + nc :, index] = -1
            result["sphere_centroid"][:, :nhc, nhc + nc :, index] = -1.0e5
            result["vertex_cartesian"][:, 0, :nhc, nhc + nc :, index] = result["vertex_cartesian"][3, 0, nhc, nhc + nc - 1, index]
            result["vertex_cartesian"][:, 1, :nhc, nhc + nc :, index] = result["vertex_cartesian"][3, 1, nhc, nhc + nc - 1, index]
        elif boundary == 5:
            result["flux_orientation"][:, :nhc, :nhc, index] = -1
            result["sphere_centroid"][:, :nhc, :nhc, index] = -1.0e5
            result["vertex_cartesian"][:, 0, :nhc, :nhc, index] = result["vertex_cartesian"][0, 0, nhc, nhc, index]
            result["vertex_cartesian"][:, 1, :nhc, :nhc, index] = result["vertex_cartesian"][0, 1, nhc, nhc, index]
        elif boundary == 8:
            result["flux_orientation"][:, nhc + nc :, nhc + nc :, index] = -1
            result["sphere_centroid"][:, nhc + nc :, nhc + nc :, index] = -1.0e5
            result["vertex_cartesian"][:, 0, nhc + nc :, nhc + nc :, index] = result["vertex_cartesian"][2, 0, nhc + nc - 1, nhc + nc - 1, index]
            result["vertex_cartesian"][:, 1, nhc + nc :, nhc + nc :, index] = result["vertex_cartesian"][2, 1, nhc + nc - 1, nhc + nc - 1, index]
        elif boundary == 6:
            result["flux_orientation"][:, nhc + nc :, :nhc, index] = -1
            result["sphere_centroid"][:, nhc + nc :, :nhc, index] = -1.0e5
            result["vertex_cartesian"][:, 0, nhc + nc :, :nhc, index] = result["vertex_cartesian"][1, 0, nhc + nc - 1, nhc, index]
            result["vertex_cartesian"][:, 1, nhc + nc :, :nhc, index] = result["vertex_cartesian"][1, 1, nhc + nc - 1, nhc, index]

        displacement = result["displacement_maximum"][:, :, :, index]
        displacement[...] = 0.0
        for j in range(halo):
            for i in range(halo):
                shift = int(np.rint(result["flux_orientation"][1, i, j, index]))
                for component in range(2):
                    result["vertex_cartesian"][:, component, i, j, index] = np.roll(
                        result["vertex_cartesian"][:, component, i, j, index], -shift
                    )
                    result["flux_vector"][component, i, j, :, index] = (
                        np.roll(unit[component], -shift)
                        * result["cell_indicator"][i, j, index]
                    )
                    values = result["vertex_cartesian"][:, component, i, j, index]
                    displacement[i, j, 0] = _f64(displacement[i, j, 0] + abs(values[3] - values[0]))
                    displacement[i, j, 1] = _f64(displacement[i, j, 1] + abs(values[0] - values[1]))
                    displacement[i, j, 2] = _f64(displacement[i, j, 2] + abs(values[1] - values[2]))
                    displacement[i, j, 3] = _f64(displacement[i, j, 3] + abs(values[1] - values[0]))

        stretch, vertex_weights, metric, metric_integral = _reconstruction_geometry(
            result["sphere_centroid"][:, :, :, index],
            result["vertex_cartesian"][:, :, :, :, index],
            dalpha[index],
            dbeta[index],
            nc=nc,
            nhc=nhc,
            nhe=nhe,
            nht=nht,
            irecons=irecons,
        )
        result["centroid_stretch"][:, :, :, index] = stretch
        result["vertex_reconstruction_weight"][:, :, :, :, index] = vertex_weights
        result["reconstruction_metric"][:, :, :, index] = metric
        result["reconstruction_metric_integral"][:, :, :, index] = metric_integral
    return result
