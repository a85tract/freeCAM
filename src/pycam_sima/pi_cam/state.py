"""Python-owned, rank-local PI-CAM state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Iterator, Mapping

import numpy as np

from .errors import PICAMStateError


@dataclass(frozen=True, slots=True)
class PICAMFieldContract:
    name: str
    dimensions: tuple[str, ...]
    dtype: str = "float64"
    units: str = "1"
    category: str = "prognostic"
    writable: bool = True
    restart: bool = True
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise PICAMStateError("field name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "dtype", np.dtype(self.dtype).str)
        object.__setattr__(self, "aliases", tuple(self.aliases))

    def shape(self, dimensions: Mapping[str, int]) -> tuple[int, ...]:
        try:
            return tuple(int(dimensions[name]) for name in self.dimensions)
        except KeyError as exc:
            raise PICAMStateError(
                f"field {self.name!r} uses unknown dimension {exc.args[0]!r}"
            ) from exc

    def to_payload(self) -> dict[str, object]:
        result = asdict(self)
        result["dimensions"] = list(self.dimensions)
        result["aliases"] = list(self.aliases)
        return result


class PICAMStatePool(Mapping[str, np.ndarray]):
    """One MPI rank's arrays; no array is implicitly shared across ranks."""

    def __init__(self, dimensions: Mapping[str, int]) -> None:
        self.dimensions = {str(key): int(value) for key, value in dimensions.items()}
        self._contracts: dict[str, PICAMFieldContract] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._aliases: dict[str, str] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        return self._arrays[self.canonical_name(name)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def canonical_name(self, name: str) -> str:
        if name in self._arrays:
            return name
        try:
            return self._aliases[name]
        except KeyError as exc:
            raise KeyError(f"unknown PI-CAM field {name!r}") from exc

    def contract(self, name: str) -> PICAMFieldContract:
        return self._contracts[self.canonical_name(name)]

    def create(
        self,
        contract: PICAMFieldContract,
        *,
        initial: float | int = 0,
    ) -> np.ndarray:
        if contract.name in self._arrays or contract.name in self._aliases:
            raise PICAMStateError(f"field {contract.name!r} already exists")
        array = np.full(
            contract.shape(self.dimensions),
            initial,
            dtype=np.dtype(contract.dtype),
            order="F",
        )
        self.attach(contract, array)
        return array

    def attach(self, contract: PICAMFieldContract, values: np.ndarray) -> None:
        if contract.name in self._arrays:
            raise PICAMStateError(f"field {contract.name!r} already exists")
        array = np.asarray(values)
        expected = contract.shape(self.dimensions)
        if array.shape != expected or array.dtype != np.dtype(contract.dtype):
            raise PICAMStateError(
                f"field {contract.name!r} has {array.shape}/{array.dtype}; "
                f"expected {expected}/{np.dtype(contract.dtype)}"
            )
        if array.ndim > 1 and not array.flags.f_contiguous:
            raise PICAMStateError(f"field {contract.name!r} must be Fortran contiguous")
        self._contracts[contract.name] = contract
        self._arrays[contract.name] = array
        for alias in contract.aliases:
            if alias in self._arrays or alias in self._aliases:
                raise PICAMStateError(f"duplicate field alias {alias!r}")
            self._aliases[alias] = contract.name

    def ensure_from_array(
        self,
        name: str,
        values: np.ndarray,
        *,
        category: str,
        units: str = "1",
        writable: bool = True,
    ) -> np.ndarray:
        """Attach an authoritative replay field while preserving its exact bytes."""

        array = np.asfortranarray(values)
        if name in self._arrays:
            target = self._arrays[name]
            if target.shape != array.shape or target.dtype != array.dtype:
                raise PICAMStateError(
                    f"replayed field {name!r} changed shape or dtype"
                )
            np.copyto(target, array, casting="no")
            return target
        dimension_names = tuple(f"{name}__dim{axis}" for axis in range(array.ndim))
        for dimension, extent in zip(dimension_names, array.shape):
            existing = self.dimensions.setdefault(dimension, int(extent))
            if existing != extent:
                raise PICAMStateError(f"inconsistent replay dimension {dimension!r}")
        contract = PICAMFieldContract(
            name=name,
            dimensions=dimension_names,
            dtype=array.dtype.str,
            units=units,
            category=category,
            writable=writable,
        )
        self.attach(contract, array.copy(order="F"))
        return self._arrays[name]

    def remove(self, name: str) -> np.ndarray:
        canonical = self.canonical_name(name)
        for alias, target in tuple(self._aliases.items()):
            if target == canonical:
                del self._aliases[alias]
        del self._contracts[canonical]
        return self._arrays.pop(canonical)

    def snapshot(self, *, restart_only: bool = False) -> dict[str, np.ndarray]:
        return {
            name: array.copy(order="F")
            for name, array in self._arrays.items()
            if not restart_only or self._contracts[name].restart
        }

    def restore(self, arrays: Mapping[str, np.ndarray]) -> None:
        missing = set(self._arrays) - set(arrays)
        if missing:
            raise PICAMStateError(
                "snapshot is missing fields: " + ", ".join(sorted(missing))
            )
        for name, values in arrays.items():
            if name in self._arrays:
                np.copyto(self._arrays[name], values, casting="no")
            else:
                self.ensure_from_array(name, values, category="restored")

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for array in self._arrays.values())


@dataclass(frozen=True, slots=True)
class PICAMStateSchema:
    contracts: tuple[PICAMFieldContract, ...]

    def allocate(self, dimensions: Mapping[str, int]) -> PICAMStatePool:
        pool = PICAMStatePool(dimensions)
        for contract in self.contracts:
            pool.create(contract)
        return pool

    @classmethod
    def core(cls) -> "PICAMStateSchema":
        """State owned before captured boundary and generated scheme fields arrive."""

        scalar = ()
        return cls(
            (
                PICAMFieldContract("model_step", scalar, "int64", category="time"),
                PICAMFieldContract("current_date", scalar, "int32", category="time"),
                PICAMFieldContract(
                    "current_seconds_of_day", scalar, "int32", "s", "time"
                ),
                PICAMFieldContract(
                    "model_timestep", scalar, "float64", "s", "configuration", False
                ),
                PICAMFieldContract(
                    "configured_stop_n",
                    scalar,
                    "int64",
                    category="configuration",
                    writable=False,
                ),
                PICAMFieldContract(
                    "case_name_utf8",
                    ("case_name_length",),
                    "uint8",
                    category="configuration",
                    writable=False,
                ),
                PICAMFieldContract(
                    "orbital_year",
                    scalar,
                    "int32",
                    category="configuration",
                    writable=False,
                ),
                PICAMFieldContract(
                    "mpi_rank", scalar, "int32", category="configuration", writable=False
                ),
                PICAMFieldContract(
                    "mpi_size", scalar, "int32", category="configuration", writable=False
                ),
            )
        )
