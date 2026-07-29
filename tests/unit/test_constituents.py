import numpy as np

from pycam_sima.model.constituents import (
    constituent_lookup_keys,
    constituent_standard_name,
    water_constituent_indices,
    water_species_flags,
    water_vapor_index,
)


def test_constituent_aliases_expose_ccpp_standard_name_lookup_keys() -> None:
    assert constituent_standard_name("cloud_liquid_water") == (
        "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water"
    )
    assert constituent_lookup_keys("cloud_liquid_water") == (
        "cloud_liquid_water",
        "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
    )
    assert constituent_lookup_keys("air") == ("air",)


def test_water_properties_are_derived_from_runtime_constituent_order() -> None:
    names = (
        "O2",
        "cloud_liquid_water",
        "water_vapor",
        "CO2",
    )

    assert water_constituent_indices(names) == (1, 2)
    assert water_vapor_index(names) == 3
    flags = water_species_flags(names)
    assert flags.dtype == np.int32
    assert flags.flags.f_contiguous
    assert np.array_equal(flags, (0, 1, 1, 0))


def test_water_vapor_index_is_zero_when_the_registry_has_no_vapor() -> None:
    assert water_vapor_index(("air", "O2", "O3")) == 0
