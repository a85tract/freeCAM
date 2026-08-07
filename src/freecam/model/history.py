"""NetCDF history output in global PG3 column order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset


HISTORY_FIELDS = (
    ("PS", "physics_surface_pressure"),
    ("PSDRY", "physics_surface_dry_air_pressure"),
    ("PHIS", "physics_surface_geopotential"),
    ("T", "physics_air_temperature"),
    ("U", "physics_zonal_wind"),
    ("V", "physics_meridional_wind"),
    ("DSE", "static_energy"),
    ("OMEGA", "physics_vertical_pressure_velocity"),
    ("PMID", "physics_midpoint_pressure"),
    ("PMIDDRY", "physics_dry_midpoint_pressure"),
    ("PDEL", "physics_layer_pressure_thickness"),
    ("PDELDRY", "physics_dry_layer_pressure_thickness"),
    ("RPDEL", "physics_reciprocal_layer_pressure_thickness"),
    ("RPDELDRY", "physics_reciprocal_dry_layer_pressure_thickness"),
    ("LNPMID", "physics_log_midpoint_pressure"),
    ("LNPMIDDRY", "physics_log_dry_midpoint_pressure"),
    ("EXNER", "physics_inverse_surface_exner"),
    ("ZM", "thermodynamic_level_height"),
    ("PINT", "physics_interface_pressure"),
    ("PINTDRY", "physics_dry_interface_pressure"),
    ("LNPINT", "physics_log_interface_pressure"),
    ("LNPINTDRY", "physics_log_dry_interface_pressure"),
    ("ZI", "physics_interface_geopotential_height"),
    ("Q", "physics_water_vapor"),
    ("CLDLIQ", "physics_cloud_liquid_water"),
    ("RAINQM", "physics_rain_water"),
    ("TTEND", "physics_air_temperature_tendency"),
    ("UTEND", "physics_zonal_wind_tendency"),
    ("VTEND", "physics_meridional_wind_tendency"),
)


class HistoryWriter:
    def __init__(
        self,
        output_dir: str | Path,
        case_name: str,
        comm,
        *,
        config=None,
    ):
        self.output_dir, self.case_name, self.comm = Path(output_dir), case_name, comm
        self.config = config
        self._captured_fields: dict[str, np.ndarray] = {}

    def capture(
        self,
        pool,
        fields: tuple[tuple[str, str], ...],
        *,
        reset: bool = False,
    ) -> None:
        """Retain rank-local diagnostics until the next history write."""

        captured: dict[str, np.ndarray] = {}
        for output_name, state_name in fields:
            try:
                values = pool.get(state_name)
            except KeyError:
                continue
            captured[output_name] = np.ascontiguousarray(values).copy()
        if reset:
            self._captured_fields = captured
        else:
            self._captured_fields.update(captured)

    def write(self, pool, clock) -> Path | None:
        ids = pool.get("physics_global_column").reshape(-1, order="F").copy()
        payload = {
            "global_column": ids,
            "lat": np.rad2deg(pool.get("physics_latitude")),
            "lon": np.rad2deg(pool.get("physics_longitude")),
            "area": pool.get("physics_cell_area").copy(),
        }
        for output_name, state_name in HISTORY_FIELDS:
            try:
                value = self._captured_fields.get(
                    output_name, pool.get(state_name)
                )
            except KeyError:
                # Reduced-constituent and non-Kessler suites do not
                # necessarily expose all three moist-species aliases.  The
                # FKESSLER reference still resolves the complete 26-field
                # inventory, while a generic suite writes only fields that
                # actually exist in its StatePool.
                continue
            payload[output_name] = np.ascontiguousarray(value)
        dynamic_history: dict[str, tuple[str, str]] = {}
        for state_name in sorted(pool.dynamic_fields):
            contract = pool.contracts[state_name]
            if not contract.history:
                continue
            if contract.dimensions not in {
                ("nphys_local",),
                ("nphys_local", "pver"),
                ("nphys_local", "pverp"),
            }:
                raise ValueError(
                    f"history field {state_name!r} has unsupported dimensions "
                    f"{contract.dimensions}"
                )
            values = pool.get(state_name)
            if values.dtype.kind not in {"b", "i", "u", "f"}:
                raise TypeError(
                    f"history field {state_name!r} has non-numeric dtype "
                    f"{values.dtype}"
                )
            payload[state_name] = np.ascontiguousarray(values)
            dynamic_history[state_name] = (
                contract.units,
                contract.ccpp_standard_name or contract.standard_name,
            )
        gathered = self.comm.gather(payload, root=0)
        if self.comm.rank != 0: return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        order = np.argsort(np.concatenate([item["global_column"] for item in gathered]))
        merged = {}
        for name in payload:
            joined = np.concatenate([item[name] for item in gathered], axis=0)
            merged[name] = joined[order]
        filename = f"{self.case_name}.cam.h0.{clock.year:04d}-{clock.month:02d}-{clock.day:02d}-{clock.seconds:05d}i.nc"
        path = self.output_dir / filename
        with Dataset(path, "w", format="NETCDF4") as ds:
            ds.createDimension("ncol", len(order)); ds.createDimension("time", None)
            ds.createDimension("nbnd", 2); ds.createDimension("lev", pool.dimensions["pver"])
            ds.createDimension("ilev", pool.dimensions["pverp"])
            for name in ("lat", "lon", "area"):
                coordinate = ds.createVariable(name, "f8", ("ncol",), fill_value=np.float64(-900.0))
                coordinate[:] = merged.pop(name)
            ds["lat"].setncatts({"long_name": "latitude", "units": "degrees_north"})
            ds["lon"].setncatts({"long_name": "longitude", "units": "degrees_east"})
            ds["area"].setncatts({"long_name": "physics column areas"})
            hyam, hybm = pool.get("hybrid_a_midpoint"), pool.get("hybrid_b_midpoint")
            hyai, hybi = pool.get("hybrid_a_interface"), pool.get("hybrid_b_interface")
            for name, values, dims in (
                ("lev", np.float64(1000.0) * (hyam + hybm), ("lev",)),
                ("hyam", hyam, ("lev",)), ("hybm", hybm, ("lev",)),
                ("ilev", np.float64(1000.0) * (hyai + hybi), ("ilev",)),
                ("hyai", hyai, ("ilev",)), ("hybi", hybi, ("ilev",)),
            ):
                variable = ds.createVariable(name, "f8", dims); variable[:] = values
            days = np.float64(clock.nstep * clock.dt_seconds) / np.float64(86400.0)
            time = ds.createVariable("time", "f8", ("time",)); time[:] = (days,)
            time.setncatts(
                {
                    "long_name": "time",
                    "units": getattr(
                        clock,
                        "time_units",
                        "days since 0001-01-01 00:00:00",
                    ),
                    "calendar": getattr(
                        clock,
                        "netcdf_calendar",
                        "noleap",
                    ),
                    "bounds": "time_bounds",
                }
            )
            date = ds.createVariable("date", "i4", ("time",)); date[:] = (clock.yyyymmdd,)
            datesec = ds.createVariable("datesec", "i4", ("time",)); datesec[:] = (clock.seconds,)
            bounds = ds.createVariable("time_bounds", "f8", ("time", "nbnd"))
            bounds[:] = ((np.float64(max(clock.nstep - 1, 0) * clock.dt_seconds) / np.float64(86400.0), days),)
            for name, value in (
                ("ndbase", 0),
                ("nsbase", 0),
                ("nbdate", getattr(clock, "base_yyyymmdd", 10101)),
                ("nbsec", getattr(clock, "base_seconds", 0)),
                ("mdt", clock.dt_seconds),
            ):
                scalar = ds.createVariable(name, "i4"); scalar.assignValue(value)
            ndcur = ds.createVariable("ndcur", "i4", ("time",)); ndcur[:] = (clock.nstep * clock.dt_seconds // 86400,)
            nscur = ds.createVariable("nscur", "i4", ("time",)); nscur[:] = (clock.seconds,)
            nsteph = ds.createVariable("nsteph", "i4", ("time",)); nsteph[:] = (clock.nstep,)
            # global_column is an internal gather key. HISTORY_FIELDS are the
            # exact 26 FKESSLER fields registered by sima_state_diagnostics.
            merged.pop("global_column")
            for name, value in merged.items():
                if value.ndim == 2 and value.shape[1] == pool.dimensions["pver"]:
                    dims = ("time", "lev", "ncol"); output = value.T[None, :, :]
                elif value.ndim == 2 and value.shape[1] == pool.dimensions["pverp"]:
                    dims = ("time", "ilev", "ncol"); output = value.T[None, :, :]
                else:
                    dims = ("time", "ncol"); output = value[None, :]
                variable = ds.createVariable(name, "f8", dims)
                variable[:] = output
                variable.setncattr("cell_methods", "time: point")
                if name in dynamic_history:
                    units, standard_name = dynamic_history[name]
                    variable.setncatts(
                        {
                            "units": units,
                            "standard_name": standard_name,
                        }
                    )
            ds.setncatts(
                {
                    "ne": (
                        int(self.config.ne)
                        if self.config is not None
                        else int(pool.dimensions.get("ne", 3))
                    ),
                    "np": pool.dimensions.get("np", 4),
                    "fv_nphys": pool.dimensions.get("fv_nphys", 3),
                    "pver": pool.dimensions["pver"],
                    "constituent_count": pool.dimensions.get("nconst", 3),
                    "Conventions": "CF-1.0",
                    "source": "CAM-SIMA Python-owned runtime",
                    "case": self.case_name,
                    "time_period_freq": "step_1",
                }
            )
        pool.get("history_sample_count")[...] += 1
        self._captured_fields.clear()
        return path
