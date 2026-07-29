import math
import json

from netCDF4 import Dataset
import numpy as np

from pycam_sima.model.phases import (
    check_energy_gmean,
    check_energy_timestep_initial,
)
from pycam_sima.model.scientific_data import (
    _linear_weights,
    read_musica_initial_concentrations,
    read_musica_placeholder_data,
    read_physics_topography,
    read_ridge_gravity_wave_data,
    read_tropopause_climatology,
    solar_irradiance_data_initialize,
    solar_irradiance_data_register,
    solar_irradiance_data_timestep_initial,
    stage_musica_tuvx_configuration,
)


class _Pool:
    def __init__(self):
        self.dimensions = {"nphys_local": 2}
        self.values = {
            "air_pressure_at_interface": np.asfortranarray(
                [[100.0, 500.0, 1000.0], [100.0, 500.0, 1000.0]]
            ),
            (
                "vertically_integrated_total_energy_using_dycore_energy_"
                "formula_at_start_of_physics_timestep"
            ): np.asfortranarray([1.0, 3.0]),
            (
                "vertically_integrated_total_energy_using_dycore_energy_"
                "formula_at_end_of_physics_timestep"
            ): np.asfortranarray([0.0, 2.0]),
            "physics_cell_area": np.asfortranarray([math.pi, math.pi]),
            "circle_constant": np.array(math.pi),
            "model_timestep": np.array(2.0),
            "gravitational_acceleration": np.array(10.0),
        }

    def ccpp_field_name(self, standard_name):
        return standard_name

    def get(self, name):
        return self.values[name]

    def set(self, name, value):
        self.values[name] = np.array(value)


class _TwoRankComm:
    def allgather(self, local):
        remote = np.asfortranarray(
            [
                [5.0 * math.pi, 4.0 * math.pi, 1000.0 * math.pi, 100.0 * math.pi],
                [7.0 * math.pi, 6.0 * math.pi, 1000.0 * math.pi, 100.0 * math.pi],
            ]
        )
        return [local, remote]


class _EnergyInitialPool:
    def __init__(self):
        self.dimensions = {"nphys_local": 2}
        self.values = {
            "air_temperature": np.asfortranarray(
                [[250.0, 251.0], [252.0, 253.0]]
            ),
            "air_temperature_at_start_of_physics_timestep": (
                np.asfortranarray(
                    [[100.0, 101.0], [102.0, 103.0]]
                )
            ),
            (
                "vertically_integrated_total_energy_using_physics_energy_"
                "formula_at_start_of_physics_timestep"
            ): np.zeros(2),
            (
                "vertically_integrated_total_energy_using_dycore_energy_"
                "formula_at_start_of_physics_timestep"
            ): np.zeros(2),
            (
                "vertically_integrated_total_water_at_start_of_physics_"
                "timestep"
            ): np.zeros(2),
            (
                "vertically_integrated_total_energy_using_physics_energy_"
                "formula"
            ): np.zeros(2),
            (
                "vertically_integrated_total_energy_using_dycore_energy_"
                "formula"
            ): np.zeros(2),
            "vertically_integrated_total_water": np.zeros(2),
            (
                "geopotential_height_wrt_surface_at_start_of_physics_"
                "timestep"
            ): np.zeros((2, 2)),
            "geopotential_height_wrt_surface": np.ones((2, 2)),
            (
                "cumulative_total_energy_boundary_flux_using_physics_energy_"
                "formula"
            ): np.ones(2),
            "cumulative_total_water_boundary_flux": np.ones(2),
            "is_first_timestep": np.array(False),
        }

    def ccpp_field_name(self, standard_name):
        return standard_name

    def get_ccpp(self, standard_name):
        return self.values[standard_name]

    def set(self, name, value):
        self.values[name] = np.asarray(value)


def test_check_energy_gmean_uses_area_weighted_collective():
    pool = _Pool()
    check_energy_gmean(pool, _TwoRankComm())

    assert pool.get(
        "global_mean_vertically_integrated_total_energy_using_dycore_energy_"
        "formula_at_start_of_physics_timestep"
    ).item() == 4.0
    assert pool.get(
        "global_mean_vertically_integrated_total_energy_using_dycore_energy_"
        "formula_at_end_of_physics_timestep"
    ).item() == 3.0
    assert np.isclose(
        pool.get("global_mean_surface_air_pressure").item(), 1000.0
    )
    assert np.isclose(
        pool.get(
            "global_mean_air_pressure_at_top_of_atmosphere_model"
        ).item(),
        100.0,
    )
    assert (
        pool.get(
            "global_mean_total_energy_correction_for_energy_conservation"
        ).item()
        == 1.0
    )
    assert np.isclose(
        pool.get(
            "global_mean_heating_rate_correction_for_energy_conservation"
        ).item(),
        np.float64(-1.0 / 2.0 * 10.0 / 900.0),
    )


def test_energy_timestep_initial_refreshes_temperature_before_integrating(
    monkeypatch,
):
    pool = _EnergyInitialPool()
    current = pool.get_ccpp("air_temperature").copy()

    def hydrostatic_energy(current_pool, backend=None):
        del backend
        assert np.array_equal(
            current_pool.get_ccpp(
                "air_temperature_at_start_of_physics_timestep"
            ),
            current,
        )
        return np.ones(2), np.full(2, 2.0), np.full(2, 3.0)

    monkeypatch.setattr(
        "pycam_sima.model.phases._hydrostatic_energy",
        hydrostatic_energy,
    )
    check_energy_timestep_initial(pool)

    assert np.array_equal(
        pool.get_ccpp("air_temperature_at_start_of_physics_timestep"),
        current,
    )


def test_linear_weights_match_cam_boundary_and_cyclic_rules():
    source = np.asarray((0.0, 1.0, 2.0, 3.0))
    target = np.asarray((0.0, 0.5, 3.5))

    lower, upper, weight_lower, weight_upper = _linear_weights(
        source,
        target,
        cyclic=True,
        cyclic_min=0.0,
        cyclic_max=4.0,
    )

    assert np.array_equal(lower, (3, 0, 3))
    assert np.array_equal(upper, (0, 1, 0))
    assert np.array_equal(weight_lower, (0.0, 0.5, 0.5))
    assert np.array_equal(weight_upper, (1.0, 0.5, 0.5))

    lower, upper, weight_lower, weight_upper = _linear_weights(
        source,
        np.asarray((-1.0, 1.5, 4.0)),
        cyclic=False,
    )
    assert np.array_equal(lower, (0, 1, 3))
    assert np.array_equal(upper, (0, 2, 3))
    assert np.array_equal(weight_lower, (1.0, 0.5, 1.0))
    assert np.array_equal(weight_upper, (0.0, 0.5, 0.0))


def test_tropopause_climatology_reader_regrids_and_builds_calendar(tmp_path):
    path = tmp_path / "tropopause.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("lon", 4)
        dataset.createDimension("lat", 3)
        dataset.createDimension("time", 12)
        dataset.createVariable("lon", "f4", ("lon",))[:] = (
            0.0,
            90.0,
            180.0,
            270.0,
        )
        dataset.createVariable("lat", "f4", ("lat",))[:] = (
            -90.0,
            0.0,
            90.0,
        )
        pressure = dataset.createVariable(
            "trop_p", "f4", ("time", "lat", "lon")
        )
        values = np.empty((12, 3, 4), dtype=np.float32)
        for month in range(12):
            for latitude in range(3):
                for longitude in range(4):
                    values[month, latitude, longitude] = (
                        100.0 * month + 10.0 * latitude + longitude
                    )
        pressure[:] = values

    local_pressure, calendar_days = read_tropopause_climatology(
        path,
        target_longitude=np.asarray((0.0, math.pi / 4.0)),
        target_latitude=np.asarray((-math.pi / 2.0, math.pi / 4.0)),
    )

    assert local_pressure.flags.f_contiguous
    assert local_pressure.shape == (2, 12)
    assert np.array_equal(local_pressure[0], values[:, 0, 0])
    assert np.array_equal(
        local_pressure[1],
        np.asarray(
            [15.5 + 100.0 * month for month in range(12)],
            dtype=np.float64,
        ),
    )
    assert np.array_equal(
        calendar_days,
        (16, 45, 75, 105, 136, 166, 197, 228, 258, 289, 319, 350),
    )


def test_ridge_reader_selects_rank_columns_and_preserves_fortran_order(
    tmp_path,
):
    path = tmp_path / "ridge.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("ncol", 5)
        dataset.createDimension("nrdg", 2)
        dataset.createVariable("GBXAR", "f8", ("ncol",))[:] = np.arange(5)
        dataset.createVariable("ISOVAR", "f8", ("ncol",))[:] = (
            np.arange(5) + 10.0
        )
        for offset, name in enumerate(
            ("HWDTH", "CLNGT", "MXDIS", "ANIXY", "ANGLL")
        ):
            values = np.arange(10, dtype=np.float64).reshape(2, 5)
            dataset.createVariable(name, "f8", ("nrdg", "ncol"))[:] = (
                values + 100.0 * offset
            )

    fields = read_ridge_gravity_wave_data(
        path,
        global_columns=np.asarray((5, 2), dtype=np.int64),
        earth_radius=1000.0,
        ridge_count=2,
    )

    assert np.array_equal(
        fields["grid_box_area_for_beta_ridge_gravity_wave_drag"],
        (4.0, 1.0),
    )
    assert np.array_equal(
        fields["isotropic_variance_for_beta_ridge_gravity_wave_drag"],
        (14.0, 11.0),
    )
    assert np.array_equal(
        fields["isotropic_weight_for_beta_ridge_gravity_wave_drag"],
        (0.0, 0.0),
    )
    widths = fields[
        "ridge_half_width_for_beta_ridge_gravity_wave_drag"
    ]
    assert widths.flags.f_contiguous
    assert np.array_equal(widths, ((4.0, 9.0), (1.0, 6.0)))


def test_ridge_reader_preserves_cam_area_scaling_operation_order(tmp_path):
    path = tmp_path / "ridge-area.nc"
    raw_area = np.float64(0.023583285603070627)
    earth_radius = np.float64(6371220.0)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("ncol", 1)
        dataset.createDimension("nrdg", 1)
        dataset.createVariable("GBXAR", "f8", ("ncol",))[:] = raw_area
        for name in ("HWDTH", "CLNGT", "MXDIS", "ANIXY", "ANGLL"):
            dataset.createVariable(name, "f8", ("nrdg", "ncol"))[:] = 0.0

    fields = read_ridge_gravity_wave_data(
        path,
        global_columns=np.asarray((1,), dtype=np.int64),
        earth_radius=earth_radius,
        ridge_count=1,
    )

    radius_km = earth_radius / np.float64(1000.0)
    expected = np.multiply(np.multiply(raw_area, radius_km), radius_km)
    grouped = np.multiply(raw_area, np.multiply(radius_km, radius_km))
    actual = fields["grid_box_area_for_beta_ridge_gravity_wave_drag"][0]
    assert actual.view(np.uint64) == expected.view(np.uint64)
    assert actual.view(np.uint64) != grouped.view(np.uint64)


def test_topography_reader_selects_one_based_physics_columns(tmp_path):
    path = tmp_path / "topography.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("ncol", 5)
        dataset.createVariable("PHIS", "f8", ("ncol",))[:] = (
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
        )

    values = read_physics_topography(
        path,
        global_columns=np.asfortranarray([[5, 2], [1, 4]]),
    )

    assert values.flags.f_contiguous
    assert np.array_equal(values, (50.0, 10.0, 20.0, 40.0))


def test_musica_placeholder_reader_uses_pinned_original_source(tmp_path):
    source_root = tmp_path / "cam"
    source = source_root / "src/physics/utils/musica_ccpp_dependencies.F90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
subroutine musica_ccpp_dependencies_init
  surface_albedo(:) = 0.25_kind_phys
  blackbody_temperature_at_surface(:) = 280.5_kind_phys
  extraterrestrial_radiation_flux(:) = 2.0e14_kind_phys
  photolysis_wavelength_grid_interfaces = (/ &
    120.0e-9_kind_phys, &
    130.0e-9_kind_phys, &
    140.0e-9_kind_phys &
  /)
end subroutine musica_ccpp_dependencies_init
""",
        encoding="utf-8",
    )

    fields = read_musica_placeholder_data(
        source_root,
        horizontal_dimension=4,
        wavelength_interface_dimension=3,
        wavelength_section_dimension=2,
    )

    assert np.array_equal(
        fields["photolysis_wavelength_grid_interfaces"],
        (120.0e-9, 130.0e-9, 140.0e-9),
    )
    assert np.array_equal(
        fields["extraterrestrial_radiation_flux"],
        (2.0e14, 2.0e14),
    )
    assert np.array_equal(
        fields["surface_albedo_due_to_uv_and_vis_direct"],
        np.full(4, 0.25),
    )
    assert np.array_equal(
        fields["blackbody_temperature_at_surface"],
        np.full(4, 280.5),
    )
    assert all(value.flags.f_contiguous for value in fields.values())


def test_musica_initial_concentrations_use_original_source_and_micm_data(
    tmp_path,
):
    source_root = tmp_path / "cam"
    source = source_root / "src/physics/utils/musica_ccpp_dependencies.F90"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
tuvx_species(1) = species_t(&
  "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
  0.0000060_kind_phys)
tuvx_species(2) = species_t("air", 1.0_kind_phys)
tuvx_species(3) = species_t("O2", 0.21_kind_phys)
tuvx_species(4) = species_t("O3", 4.0e-6_kind_phys)
""",
        encoding="utf-8",
    )
    mechanism = tmp_path / "mechanism"
    mechanism.mkdir()
    configuration = mechanism / "config.json"
    configuration.write_text(
        json.dumps({"camp-files": ["species.json"]}),
        encoding="utf-8",
    )
    (mechanism / "species.json").write_text(
        json.dumps(
            {
                "camp-data": [
                    {
                        "name": "Cl",
                        "__default mixing ratio [kg kg-1]": 1.0e-12,
                    },
                    {
                        "name": "Cl2",
                        "__default mixing ratio [kg kg-1]": 2.0e-12,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    values = read_musica_initial_concentrations(
        source_root,
        configuration,
    )

    assert values == {
        "cl": np.float64(1.0e-12),
        "cl2": np.float64(2.0e-12),
        "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water": (
            np.float64(6.0e-6)
        ),
        "air": np.float64(1.0),
        "o2": np.float64(0.21),
        "o3": np.float64(4.0e-6),
    }


def test_stage_musica_tuvx_configuration_resolves_cime_data_paths(tmp_path):
    mechanism = tmp_path / "chemistry-data/mechanisms/terminator"
    data = mechanism / "tuvx/data/cross_sections/O3.nc"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"netcdf")
    source = mechanism / "tuvx/config.json"
    source.write_text(
        json.dumps(
            {
                "cross sections": [
                    {
                        "file path": (
                            "musica_configurations/terminator/tuvx/data/"
                            "cross_sections/O3.nc"
                        )
                    }
                ],
                "unchanged": "O3",
            }
        ),
        encoding="utf-8",
    )

    staged = stage_musica_tuvx_configuration(
        source,
        run_dir=tmp_path / "run",
        rank=7,
    )
    values = json.loads(staged.read_text(encoding="utf-8"))

    assert staged.name == "tuvx-config-rank-000007.json"
    assert values["cross sections"][0]["file path"] == str(data.resolve())
    assert values["unchanged"] == "O3"


class _SolarPool:
    def __init__(self, path):
        self.dimensions = {
            "number_of_wavelength_samples_of_spectrum": 3,
            "number_of_wavelength_samples_of_spectrum_plus_one": 4,
        }
        self.values = {
            "filename_of_solar_irradiance_data": np.asarray(
                str(path), dtype="S512"
            ),
            "type_of_solar_irradiance_data": np.asarray(
                "FIXED", dtype="S512"
            ),
            "constant_total_solar_irradiance": np.asarray(-9999.0),
            "do_solar_radiation_heating_spectral_scaling": np.asarray(False),
            "number_of_wavelength_samples_of_spectrum": np.asarray(
                3, dtype=np.int32
            ),
            "number_of_wavelength_samples_of_spectrum_plus_one": np.asarray(
                4, dtype=np.int32
            ),
            "do_spectral_scaling_of_solar_irradiance_data": np.asarray(False),
            "solar_irradiance_file_has_spectrum_information": np.asarray(False),
            "total_solar_irradiance": np.asarray(-1.0),
            "wavelength_endpoints": np.zeros(4, dtype=np.float64, order="F"),
            "solar_irradiance": np.zeros(3, dtype=np.float64, order="F"),
        }

    def ccpp_field_name(self, standard_name):
        if standard_name not in self.values:
            raise KeyError(standard_name)
        return standard_name

    def get_ccpp(self, standard_name):
        return self.values[standard_name]

    def set(self, name, value, *, unsafe=False):
        np.copyto(self.values[name], value, casting="same_kind")


def test_fixed_solar_irradiance_host_service_reads_cam_file_contract(tmp_path):
    path = tmp_path / "solar.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 2)
        dataset.createDimension("wlen", 3)
        dataset.createVariable("wavelength", "f8", ("wlen",))[:] = (
            100.0,
            200.0,
            300.0,
        )
        dataset.createVariable("band_width", "f8", ("wlen",))[:] = (
            10.0,
            20.0,
            30.0,
        )
        dataset.createVariable("ssi", "f4", ("time", "wlen"))[:] = (
            (1000.0, 2000.0, 3000.0),
            (1000.0, 2000.0, 3000.0),
        )
        dataset.createVariable("tsi", "f4", ("time",))[:] = (
            1361.75,
            1361.75,
        )

    pool = _SolarPool(path)
    solar_irradiance_data_register(pool)
    solar_irradiance_data_initialize(pool)
    solar_irradiance_data_timestep_initial(pool)

    assert pool.values[
        "solar_irradiance_file_has_spectrum_information"
    ].item()
    assert not pool.values[
        "do_spectral_scaling_of_solar_irradiance_data"
    ].item()
    assert pool.values["total_solar_irradiance"].item() == 1361.0
    assert np.array_equal(
        pool.values["wavelength_endpoints"],
        (95.0, 190.0, 285.0, 315.0),
    )
    assert np.array_equal(
        pool.values["solar_irradiance"],
        (1.0, 2.0, 3.0),
    )
