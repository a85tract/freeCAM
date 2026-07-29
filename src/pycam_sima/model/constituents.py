"""Suite-independent constituent properties used by the Python host."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


_WATER_CONSTITUENTS = frozenset(
    {
        "cloud_ice",
        "cloud_ice_mixing_ratio_wrt_moist_air_and_condensed_water",
        "cloud_liquid_water",
        "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water",
        "rain_water",
        "rain_water_mixing_ratio_wrt_moist_air_and_condensed_water",
        "water_vapor",
        "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water",
    }
)
_WATER_VAPOR = frozenset(
    {
        "water_vapor",
        "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water",
    }
)
_STANDARD_NAMES = {
    "cl": "Cl",
    "cl2": "Cl2",
    "o2": "O2",
    "o3": "O3",
    "cloud_ice": (
        "cloud_ice_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
    "cloud_liquid_water": (
        "cloud_liquid_water_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
    "rain_water": (
        "rain_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
    "water_vapor": (
        "water_vapor_mixing_ratio_wrt_moist_air_and_condensed_water"
    ),
}


def constituent_standard_name(name: str) -> str:
    """Return the CCPP standard name for a configured constituent alias."""

    value = str(name).strip()
    return _STANDARD_NAMES.get(value.lower(), value)


def constituent_lookup_keys(name: str) -> tuple[str, ...]:
    """Return normalized short-name and standard-name lookup keys."""

    short = str(name).strip().lower()
    standard = constituent_standard_name(name).strip().lower()
    return (short,) if short == standard else (short, standard)


def is_water_constituent(name: str) -> bool:
    return str(name).strip().lower() in _WATER_CONSTITUENTS


def is_water_vapor(name: str) -> bool:
    return str(name).strip().lower() in _WATER_VAPOR


def water_constituent_indices(names: Iterable[str]) -> tuple[int, ...]:
    return tuple(
        index
        for index, name in enumerate(names)
        if is_water_constituent(name)
    )


def water_species_flags(names: Iterable[str]) -> np.ndarray:
    return np.asfortranarray(
        np.asarray(
            [int(is_water_constituent(name)) for name in names],
            dtype=np.int32,
        )
    )


def water_vapor_index(names: Iterable[str]) -> int:
    """Return the one-based ABI index, or zero when vapor is absent."""

    indexes = [
        index
        for index, name in enumerate(names, start=1)
        if is_water_vapor(name)
    ]
    if len(indexes) > 1:
        raise ValueError("constituent registry contains multiple water vapors")
    return indexes[0] if indexes else 0
