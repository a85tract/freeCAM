"""DCMIP2016 moist baroclinic initial state in scalar operation order."""

from __future__ import annotations

import math

import numpy as np


T0E, T0P, B, KK, LAPSE = 310.0, 240.0, 2.0, 3.0, 0.005
PSURF_MOIST, MVAP = 100000.0, 0.608
MOIST_QLAT, MOIST_QP, MOIST_Q0 = 2.0 * math.pi / 9.0, 34000.0, 0.018
PERTUP, PERTEXPR, PERTLON, PERTLAT, PERTZ = 1.0, 0.1, math.pi / 9.0, 2.0 * math.pi / 9.0, 15000.0
EPS, QV_MIN = 1.0e-13, 1.0e-12
GAUSS_X = (-0.97390652851717, -0.865063366689, -0.67940956829902, -0.4333953941292, -0.14887433898163,
           0.14887433898163, 0.4333953941292, 0.679409568299, 0.86506336668898, 0.97390652851717)
GAUSS_W = (0.06667134430869, 0.1494513491506, 0.219086362516, 0.26926671931, 0.29552422471475,
           0.2955242247148, 0.26926671931, 0.21908636251598, 0.1494513491506, 0.0666713443087)


class Dcmip2016:
    def __init__(self, *, rair: float, gravity: float, rearth: float, omega: float, epsilo: float):
        self.rair, self.gravity, self.rearth, self.omega, self.epsilo = map(float, (rair, gravity, rearth, omega, epsilo))

    def moist_pressure(self, z: float, lat: float) -> float:
        t0 = 0.5 * (T0E + T0P)
        const_a = 1.0 / LAPSE
        const_b = (t0 - T0P) / (t0 * T0P)
        const_c = 0.5 * (KK + 2.0) * (T0E - T0P) / (T0E * T0P)
        const_h = self.rair * t0 / self.gravity
        scaled_z = z / (B * const_h)
        scaled_z_squared = scaled_z * scaled_z
        inttau1 = const_a * (math.exp(LAPSE * z / t0) - 1.0) + const_b * z * math.exp(-scaled_z_squared)
        inttau2 = const_c * z * math.exp(-scaled_z_squared)
        rratio = 1.0
        interior = (rratio * math.cos(lat))**KK - KK / (KK + 2.0) * (rratio * math.cos(lat))**(KK + 2.0)
        return PSURF_MOIST * math.exp(-self.gravity / self.rair * (inttau1 - inttau2 * interior))

    def virtual_temperature(self, z: float, lat: float) -> float:
        t0 = 0.5 * (T0E + T0P)
        const_b = (t0 - T0P) / (t0 * T0P)
        const_c = 0.5 * (KK + 2.0) * (T0E - T0P) / (T0E * T0P)
        const_h = self.rair * t0 / self.gravity
        scaled_z = z / (B * const_h)
        scaled_z_squared = scaled_z * scaled_z
        tau1 = 1.0 / t0 * math.exp(LAPSE * z / t0) + const_b * (1.0 - 2.0 * scaled_z_squared) * math.exp(-scaled_z_squared)
        tau2 = const_c * (1.0 - 2.0 * scaled_z_squared) * math.exp(-scaled_z_squared)
        interior = math.cos(lat)**KK - KK / (KK + 2.0) * math.cos(lat)**(KK + 2.0)
        return 1.0 / (tau1 - tau2 * interior)

    @staticmethod
    def qv(pwet: float, lat: float) -> float:
        eta = pwet / PSURF_MOIST
        if eta > 0.1:
            lat_ratio = lat / MOIST_QLAT
            lat_squared = lat_ratio * lat_ratio
            pressure_ratio = (eta - 1.0) * PSURF_MOIST / MOIST_QP
            return MOIST_Q0 * math.exp(-(lat_squared * lat_squared)) * math.exp(-(pressure_ratio * pressure_ratio))
        return QV_MIN

    def water_weight(self, z: float, lat: float, ztop: float) -> float:
        xm, xr, integral = 0.5 * (z + ztop), 0.5 * (ztop - z), 0.0
        for gx, gw in zip(GAUSS_X, GAUSS_W):
            zz = xm + gx * xr
            pwet = self.moist_pressure(zz, lat)
            integral += gw * self.gravity * pwet * self.qv(pwet, lat) / (self.rair * self.virtual_temperature(zz, lat))
        return 0.5 * (ztop - z) * integral

    def dry_weight(self, z: float, ptop: float, lat: float, ztop: float) -> float:
        xm, xr, integral = 0.5 * (z + ztop), 0.5 * (ztop - z), 0.0
        for gx, gw in zip(GAUSS_X, GAUSS_W):
            zz = xm + gx * xr
            pwet = self.moist_pressure(zz, lat)
            integral += gw * self.gravity * pwet * (1.0 - self.qv(pwet, lat)) / (self.rair * self.virtual_temperature(zz, lat))
        return 0.5 * (ztop - z) * integral + ptop

    def height_for_pressure(self, p: float, dry: bool, ptop: float, lat: float, ztop: float) -> float:
        z0, z1 = 0.0, 10000.0
        func = (lambda z: self.dry_weight(z, ptop, lat, ztop)) if dry else (lambda z: self.moist_pressure(z, lat))
        p0, p1 = func(z0), func(z1)
        z2 = z1
        for _ix in range(1000):
            z2 = z1 - (p1 - p) * (z1 - z0) / (p1 - p0)
            p2 = func(z2)
            if abs(p2 - p) / p < EPS or abs(z1 - z2) < EPS or abs(p1 - p2) < EPS:
                return z2
            z0, p0, z1, p1 = z1, p1, z2, p2
        raise RuntimeError(f"DCMIP height iteration did not converge for p={p}, lat={lat}")

    def wind(self, z: float, lat: float, lon: float) -> tuple[float, float]:
        t0 = 0.5 * (T0E + T0P)
        const_h = self.rair * t0 / self.gravity
        const_c = 0.5 * (KK + 2.0) * (T0E - T0P) / (T0E * T0P)
        scaled_z = z / (B * const_h)
        inttau2 = const_c * z * math.exp(-(scaled_z * scaled_z))
        intterm_u = math.cos(lat)**(KK - 1.0) - math.cos(lat)**(KK + 1.0)
        big_u = self.gravity / self.rearth * KK * inttau2 * intterm_u * self.virtual_temperature(z, lat)
        rcoslat = self.rearth * math.cos(lat)
        omega_rcoslat = self.omega * rcoslat
        u = -omega_rcoslat + math.sqrt(omega_rcoslat * omega_rcoslat + rcoslat * big_u)
        distance = (1.0 / PERTEXPR) * math.acos(
            math.sin(PERTLAT) * math.sin(lat)
            + math.cos(PERTLAT) * math.cos(lat) * math.cos(lon - PERTLON)
        )
        z_squared = z * z
        taper = 1.0 - 3.0 * z_squared / (PERTZ * PERTZ) + 2.0 * (z_squared * z) / (PERTZ * PERTZ * PERTZ) if z < PERTZ else 0.0
        if distance < 1.0:
            u += PERTUP * taper * math.exp(-(distance * distance))
        return u, 0.0

    def column(self, lat: float, lon: float, hyai, hybi, hyam, hybm, ps0: float):
        ptop = float(hyai[0] * ps0)
        ztop = self.height_for_pressure(ptop, False, ptop, 0.0, -1000.0)
        ps = PSURF_MOIST - self.water_weight(0.0, lat, ztop)
        nlev = len(hyam)
        u = np.empty(nlev); v = np.empty(nlev); t = np.empty(nlev); q = np.empty(nlev); dp = np.empty(nlev)
        zmid = np.empty(nlev); pdry_half = np.empty(nlev+1); pwet_half = np.empty(nlev+1); zdry_half = np.empty(nlev+1)
        for k in range(nlev):
            pdry = float(hyam[k] * ps0 + hybm[k] * ps)
            zmid[k] = self.height_for_pressure(pdry, True, ptop, lat, ztop)
            u[k], v[k] = self.wind(float(zmid[k]), lat, lon)
        pdry_half[0], pwet_half[0], zdry_half[0] = ptop, ptop, ztop
        for k in range(1, nlev+1):
            pdry_half[k] = hyai[k] * ps0 + hybi[k] * ps
            zdry_half[k] = self.height_for_pressure(float(pdry_half[k]), True, ptop, lat, ztop)
            pwet_half[k] = pdry_half[k] + self.water_weight(float(zdry_half[k]), lat, ztop)
        for k in range(nlev):
            qdry = (pwet_half[k+1] - pwet_half[k]) / (pdry_half[k+1] - pdry_half[k]) - 1.0
            q[k] = max(qdry, QV_MIN / (1.0 - QV_MIN))
            tv = self.virtual_temperature(float(zmid[k]), lat)
            t[k] = tv * (1.0 + q[k]) / (1.0 + (1.0 / self.epsilo) * q[k])
            # dyn_comp initializes dry dp from coefficient differences after
            # bc_wav_set_ic returns.  Subtracting the two interface pressures
            # is mathematically equivalent but not bitwise equivalent.
            dp[k] = (hyai[k+1] - hyai[k]) * ps0 + (hybi[k+1] - hybi[k]) * ps
        return u, v, t, ps, q, dp, zmid


def _ic_angle(value: float) -> float:
    """Reproduce the GLL grid radians -> degrees -> IC radians path."""

    rad2deg = np.float64(180.0) / np.float64(math.pi)
    deg2rad = np.float64(math.pi) / np.float64(180.0)
    return float(np.float64(np.float64(value) * rad2deg) * deg2rad)


def _synchronize_gll_initial_state(pool, comm) -> None:
    """Apply the source pmask plus edge exchange using MIN-owned GLL DOFs."""

    nlev, nconst = pool.dimensions["pver"], pool.dimensions["nconst"]
    width = 1 + nlev * (3 + nconst)
    send = np.zeros((54 * 16, width), dtype=np.float64, order="F")
    receive = np.empty_like(send, order="F")
    dofs = pool.get("gll_global_dof")
    gids = pool.get("global_element_id")
    for le in range(pool.dimensions["nelem_local"]):
        gid = int(gids[le])
        for j in range(4):
            for i in range(4):
                dof = int(dofs[i, j, le])
                local_dof = (gid - 1) * 16 + j * 4 + i + 1
                if dof != local_dof:
                    continue
                offset = 0
                send[dof - 1, offset] = pool.get("surface_pressure")[i, j, le, 0]
                offset += 1
                for name in ("zonal_wind", "meridional_wind", "air_temperature"):
                    send[dof - 1, offset:offset+nlev] = pool.get(name)[i, j, :, le, 0]
                    offset += nlev
                for constituent in range(nconst):
                    send[dof - 1, offset:offset+nlev] = pool.get("constituent_mixing_ratio")[i, j, :, le, constituent, 0]
                    offset += nlev
    comm.Allreduce(send, receive)
    for le in range(pool.dimensions["nelem_local"]):
        for j in range(4):
            for i in range(4):
                row = receive[int(dofs[i, j, le]) - 1]
                offset = 0
                pool.get("surface_pressure")[i, j, le, 0] = row[offset]
                offset += 1
                for name in ("zonal_wind", "meridional_wind", "air_temperature"):
                    pool.get(name)[i, j, :, le, 0] = row[offset:offset+nlev]
                    offset += nlev
                for constituent in range(nconst):
                    pool.get("constituent_mixing_ratio")[i, j, :, le, constituent, 0] = row[offset:offset+nlev]
                    offset += nlev


def populate_initial_state(pool, comm=None) -> None:
    model = Dcmip2016(
        rair=float(pool.get("dry_air_gas_constant")), gravity=float(pool.get("gravitational_acceleration")),
        rearth=float(pool.get("earth_radius")), omega=float(pool.get("earth_angular_velocity")),
        epsilo=float(pool.get("water_to_dry_molecular_weight_ratio")),
    )
    hyai, hybi, hyam, hybm = (pool.get(name) for name in ("hybrid_a_interface", "hybrid_b_interface", "hybrid_a_midpoint", "hybrid_b_midpoint"))
    ps0 = float(pool.get("reference_pressure"))
    shape = pool.get("gll_longitude").shape
    for le in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                result = model.column(_ic_angle(pool.get("gll_latitude")[i,j,le]), _ic_angle(pool.get("gll_longitude")[i,j,le]), hyai, hybi, hyam, hybm, ps0)
                u, v, t, ps, q, dp, _z = result
                pool.get("zonal_wind")[i,j,:,le,0] = u
                pool.get("meridional_wind")[i,j,:,le,0] = v
                pool.get("air_temperature")[i,j,:,le,0] = t
                pool.get("surface_pressure")[i,j,le,0] = ps
                pool.get("water_vapor")[i,j,:,le,0] = q
    if comm is not None and hasattr(comm, "Allreduce"):
        _synchronize_gll_initial_state(pool, comm)
    # dyn_comp constructs dp from the exchanged dry surface pressure, then
    # copies the complete state to all startup time levels.
    for le in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                ps = pool.get("surface_pressure")[i, j, le, 0]
                for k in range(pool.dimensions["pver"]):
                    dp = np.float64(hyai[k+1] - hyai[k]) * ps0 + np.float64(hybi[k+1] - hybi[k]) * ps
                    pool.get("layer_pressure_thickness")[i, j, k, le, 0] = dp
                    pool.get("constituent_mass")[i, j, k, le, 2, 0] = np.float64(pool.get("water_vapor")[i, j, k, le, 0] * dp)
                for time in range(1, 3):
                    pool.get("zonal_wind")[i,j,:,le,time] = pool.get("zonal_wind")[i,j,:,le,0]
                    pool.get("meridional_wind")[i,j,:,le,time] = pool.get("meridional_wind")[i,j,:,le,0]
                    pool.get("air_temperature")[i,j,:,le,time] = pool.get("air_temperature")[i,j,:,le,0]
                    pool.get("surface_pressure")[i,j,le,time] = ps
                    pool.get("layer_pressure_thickness")[i,j,:,le,time] = pool.get("layer_pressure_thickness")[i,j,:,le,0]
                    pool.get("constituent_mixing_ratio")[i,j,:,le,:,time] = pool.get("constituent_mixing_ratio")[i,j,:,le,:,0]
                    pool.get("constituent_mass")[i,j,:,le,:,time] = pool.get("constituent_mass")[i,j,:,le,:,0]
    ncol = pool.get("physics_longitude").size
    for col in range(ncol):
        u, v, t, ps, q, dp, z = model.column(_ic_angle(pool.get("physics_latitude")[col]), _ic_angle(pool.get("physics_longitude")[col]), hyai, hybi, hyam, hybm, ps0)
        pool.get("physics_zonal_wind")[col,:] = u
        pool.get("physics_meridional_wind")[col,:] = v
        pool.get("physics_air_temperature")[col,:] = t
        pool.get("physics_surface_pressure")[col] = ps
        pool.get("physics_layer_pressure_thickness")[col,:] = dp
        pool.get("physics_water_vapor")[col,:] = q
        le, within = divmod(col, 9)
        pi, pj = within % 3, within // 3
        pool.get("fvm_tracer")[3 + pi,3 + pj,:,le,2] = q
        pool.get("thermodynamic_level_height")[col,:] = z
    pressure = pool.get("hybrid_a_midpoint")[None,:] * ps0 + pool.get("hybrid_b_midpoint")[None,:] * pool.get("physics_surface_pressure")[:,None]
    exner = (pressure / ps0) ** (float(pool.get("dry_air_gas_constant")) / float(pool.get("dry_air_specific_heat")))
    pool.set("exner_function", exner)
    pool.set("potential_temperature", pool.get("physics_air_temperature") / exner)
    pool.set("dry_air_density", pressure / (pool.get("physics_air_temperature") * float(pool.get("dry_air_gas_constant"))))
    pool.get("column_dry_air_specific_heat")[:] = float(pool.get("dry_air_specific_heat"))
    pool.get("column_dry_air_gas_constant")[:] = float(pool.get("dry_air_gas_constant"))
    pool.set("temperature_before_kessler", pool.get("physics_air_temperature"))
