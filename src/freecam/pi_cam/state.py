"""Python-owned, rank-local PI-CAM state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Mapping

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
    # Python-owned fields join the default history stream unless a caller
    # opts out, mirroring CAM's registered fields joining the default tape.
    output: bool = True
    aliases: tuple[str, ...] = ()
    standard_name: str | None = None
    requires_contiguous: bool = True
    owner: str = "python"
    dynamic: bool = False

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise PICAMStateError("field name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "dtype", np.dtype(self.dtype).str)
        object.__setattr__(self, "aliases", tuple(self.aliases))
        if self.standard_name is not None:
            object.__setattr__(
                self, "standard_name", str(self.standard_name).strip().lower()
            )
        if self.owner not in {"python", "native"}:
            raise PICAMStateError("field owner must be 'python' or 'native'")

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

    @classmethod
    def from_payload(cls, values: Mapping[str, Any]) -> "PICAMFieldContract":
        return cls(
            name=str(values["name"]),
            dimensions=tuple(str(item) for item in values.get("dimensions", ())),
            dtype=str(values.get("dtype", "float64")),
            units=str(values.get("units", "1")),
            category=str(values.get("category", "prognostic")),
            writable=bool(values.get("writable", True)),
            restart=bool(values.get("restart", True)),
            aliases=tuple(str(item) for item in values.get("aliases", ())),
            standard_name=(
                None
                if values.get("standard_name") is None
                else str(values["standard_name"])
            ),
            requires_contiguous=bool(values.get("requires_contiguous", True)),
            owner=str(values.get("owner", "python")),
            dynamic=bool(values.get("dynamic", False)),
        )


@dataclass(frozen=True, slots=True)
class PICAMVariableSpec:
    """Serializable definition for one Python-owned runtime field."""

    name: str
    dimensions: tuple[str, ...]
    dtype: str = "float64"
    units: str = "1"
    initial: float | int = 0.0
    writable: bool = True
    restart: bool = True
    output: bool = True
    aliases: tuple[str, ...] = ()
    standard_name: str | None = None

    def __post_init__(self) -> None:
        contract = self.contract()
        object.__setattr__(self, "name", contract.name)
        object.__setattr__(self, "dimensions", contract.dimensions)
        object.__setattr__(self, "dtype", contract.dtype)
        object.__setattr__(self, "aliases", contract.aliases)
        object.__setattr__(self, "standard_name", contract.standard_name)

    def contract(self) -> PICAMFieldContract:
        return PICAMFieldContract(
            name=self.name,
            dimensions=tuple(self.dimensions),
            dtype=self.dtype,
            units=self.units,
            category="dynamic",
            writable=self.writable,
            restart=self.restart,
            output=self.output,
            aliases=tuple(self.aliases),
            standard_name=self.standard_name,
            owner="python",
            dynamic=True,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            **self.contract().to_payload(),
            "initial": self.initial,
        }

    @classmethod
    def from_payload(cls, values: Mapping[str, Any]) -> "PICAMVariableSpec":
        if int(values.get("schema_version", 1)) != 1:
            raise PICAMStateError("unsupported PI-CAM variable schema version")
        return cls(
            name=str(values["name"]),
            dimensions=tuple(str(item) for item in values.get("dimensions", ())),
            dtype=str(values.get("dtype", "float64")),
            units=str(values.get("units", "1")),
            initial=values.get("initial", 0.0),
            writable=bool(values.get("writable", True)),
            restart=bool(values.get("restart", True)),
            aliases=tuple(str(item) for item in values.get("aliases", ())),
            standard_name=(
                None
                if values.get("standard_name") is None
                else str(values["standard_name"])
            ),
        )


class PICAMStatePool(Mapping[str, np.ndarray]):
    """One MPI rank's arrays; no array is implicitly shared across ranks."""

    def __init__(self, dimensions: Mapping[str, int]) -> None:
        self.dimensions = {str(key): int(value) for key, value in dimensions.items()}
        self._contracts: dict[str, PICAMFieldContract] = {}
        self._arrays: dict[str, np.ndarray] = {}
        self._aliases: dict[str, str] = {}
        self._standard_names: dict[str, str] = {}
        self._dynamic_fields: set[str] = set()
        self._initialized_fields: set[str] = set()

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

    def ccpp_field_name(self, standard_name: str) -> str:
        token = str(standard_name).strip().lower()
        try:
            return self._standard_names[token]
        except KeyError as exc:
            raise KeyError(f"unknown PI-CAM standard name {standard_name!r}") from exc

    def is_initialized(self, name: str) -> bool:
        return self.canonical_name(name) in self._initialized_fields

    def mark_initialized(self, name: str) -> None:
        self._initialized_fields.add(self.canonical_name(name))

    @property
    def process_state_names(self) -> frozenset[str]:
        return frozenset()

    @property
    def contracts(self) -> Mapping[str, PICAMFieldContract]:
        return dict(self._contracts)

    def get(self, name: str, default: object = None) -> np.ndarray:
        """Return one canonical or aliased field.

        ``default`` is accepted for Mapping compatibility, but a missing field
        still raises unless the caller supplied a non-``None`` default.  Model
        contracts should fail closed rather than silently substitute arrays.
        """

        try:
            return self[name]
        except KeyError:
            if default is not None:
                return default  # type: ignore[return-value]
            raise


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

    def create_from_array(self, name: str, values: np.ndarray) -> np.ndarray:
        """Register a copied, rank-independent NumPy array as a dynamic field.

        This backs the concise ``state.name = np.ndarray`` UI.  Synthetic
        dimensions preserve the supplied shape without pretending that the
        array follows CAM's rank-dependent grid decomposition.
        """

        array = np.array(values, copy=True, order="F", subok=False)
        if array.ndim == 0:
            raise PICAMStateError(
                "direct NumPy state assignment requires an array with at least "
                "one dimension"
            )
        if array.dtype.hasobject:
            raise PICAMStateError("StatePool arrays cannot use object dtype")
        dimension_names = tuple(f"{name}__dim{axis}" for axis in range(array.ndim))
        added_dimensions: list[str] = []
        try:
            for dimension, extent in zip(dimension_names, array.shape):
                if dimension not in self.dimensions:
                    self.dimensions[dimension] = int(extent)
                    added_dimensions.append(dimension)
                elif self.dimensions[dimension] != int(extent):
                    raise PICAMStateError(
                        f"dynamic array dimension {dimension!r} already has "
                        f"extent {self.dimensions[dimension]}, not {extent}"
                    )
            contract = PICAMFieldContract(
                name=name,
                dimensions=dimension_names,
                dtype=array.dtype.str,
                category="dynamic",
                owner="python",
                dynamic=True,
            )
            self.attach(contract, array)
        except BaseException:
            for dimension in added_dimensions:
                self.dimensions.pop(dimension, None)
            raise
        return self._arrays[name]

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
        if (
            contract.requires_contiguous
            and array.ndim > 1
            and not array.flags.f_contiguous
        ):
            raise PICAMStateError(f"field {contract.name!r} must be Fortran contiguous")
        for alias in contract.aliases:
            if alias in self._arrays or alias in self._aliases:
                raise PICAMStateError(f"duplicate field alias {alias!r}")
        if (
            contract.standard_name is not None
            and contract.standard_name.lower() in self._standard_names
        ):
            raise PICAMStateError(
                f"duplicate CCPP standard name {contract.standard_name!r}"
            )
        self._contracts[contract.name] = contract
        self._arrays[contract.name] = array
        self._initialized_fields.add(contract.name)
        if contract.dynamic:
            self._dynamic_fields.add(contract.name)
        if contract.standard_name is not None:
            key = contract.standard_name.lower()
            self._standard_names[key] = contract.name
        for alias in contract.aliases:
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
        contract = self._contracts[canonical]
        for alias, target in tuple(self._aliases.items()):
            if target == canonical:
                del self._aliases[alias]
        del self._contracts[canonical]
        self._dynamic_fields.discard(canonical)
        self._initialized_fields.discard(canonical)
        self._standard_names = {
            standard_name: target
            for standard_name, target in self._standard_names.items()
            if target != canonical
        }
        values = self._arrays.pop(canonical)
        remaining_dimensions = {
            dimension
            for remaining in self._contracts.values()
            for dimension in remaining.dimensions
        }
        for dimension in contract.dimensions:
            if (
                dimension.startswith(f"{canonical}__dim")
                and dimension not in remaining_dimensions
            ):
                self.dimensions.pop(dimension, None)
        return values

    def remove_dynamic(self, name: str) -> np.ndarray:
        canonical = self.canonical_name(name)
        if canonical not in self._dynamic_fields:
            raise PICAMStateError(
                f"field {canonical!r} is model-owned and cannot be removed dynamically"
            )
        return self.remove(canonical)

    @property
    def dynamic_fields(self) -> frozenset[str]:
        return frozenset(self._dynamic_fields)

    def pointer_records(self) -> dict[str, tuple[int, tuple[int, ...], str]]:
        return {
            name: (int(values.ctypes.data), tuple(values.shape), values.dtype.str)
            for name, values in self._arrays.items()
        }

    def assert_pointer_stability(
        self, before: Mapping[str, tuple[int, tuple[int, ...], str]]
    ) -> None:
        after = self.pointer_records()
        changed = [
            name
            for name, record in before.items()
            if name not in after or after[name] != record
        ]
        if changed:
            raise PICAMStateError(
                "runtime process changed StatePool storage: "
                + ", ".join(changed[:8])
            )

    def snapshot(self, *, restart_only: bool = False) -> dict[str, np.ndarray]:
        return {
            name: array.copy(order="F")
            for name, array in self._arrays.items()
            if not restart_only or self._contracts[name].restart
        }

    def restore(self, arrays: Mapping[str, np.ndarray]) -> None:
        # Raw owner buffers and native grid context are reconstructed locally
        # and deliberately omitted from restart snapshots.  Inline fields in
        # ``arrays`` are copied through their strided views into those buffers.
        missing = {
            name
            for name, contract in self._contracts.items()
            if contract.restart and name not in arrays
        }
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
        # Inline native fields are strided views into the corresponding raw
        # derived-type owner buffer and therefore must not be counted twice;
        # module-parameter fields view storage the Fortran image owns.
        return sum(
            array.nbytes
            for name, array in self._arrays.items()
            if self._contracts[name].category
            not in ("native_cam_inline_state", "native_module_parameter")
        )


def active_field_slices(
    pool: PICAMStatePool, name: str
) -> tuple[np.ndarray, ...]:
    """Return field views excluding inactive CAM ``pcols`` padding."""

    canonical = pool.canonical_name(name)
    values = pool[canonical]
    dimensions = tuple(pool.contract(canonical).dimensions)
    if "pcols" not in dimensions or "chunks" not in dimensions:
        return (values,)
    pcols_axis = dimensions.index("pcols")
    chunks_axis = dimensions.index("chunks")
    ncols = None
    for candidate in ("grid.chunk_ncols", "phys_state.ncol"):
        try:
            ncols = np.asarray(pool[candidate], dtype=np.int64).reshape(-1)
        except KeyError:
            continue
        break
    if ncols is None:
        raise PICAMStateError(
            f"field {canonical!r} uses pcols/chunks but StatePool has no "
            "chunk ncol field"
        )
    moved = np.moveaxis(values, (pcols_axis, chunks_axis), (0, -1))
    if ncols.size != moved.shape[-1]:
        raise PICAMStateError(
            f"field {canonical!r} chunk dimension does not match chunk ncol metadata"
        )
    slices = tuple(
        moved[: int(count), ..., chunk]
        for chunk, count in enumerate(ncols)
        if int(count) > 0
    )
    if not slices:
        raise PICAMStateError(f"field {canonical!r} has no active local columns")
    return slices


def active_field_mask(pool: PICAMStatePool, name: str) -> np.ndarray:
    """Return a mask selecting scientifically active rank-local entries.

    CAM allocates every physics chunk with ``pcols`` rows even when the final
    chunk owns fewer columns.  A Notebook slice uses the array's normal NumPy
    axes, while this mask prevents scalar edits and statistics from touching
    those inactive padding rows.
    """

    canonical = pool.canonical_name(name)
    values = pool[canonical]
    dimensions = tuple(pool.contract(canonical).dimensions)
    if "pcols" not in dimensions or "chunks" not in dimensions:
        return np.ones(values.shape, dtype=np.bool_, order="F")
    pcols_axis = dimensions.index("pcols")
    chunks_axis = dimensions.index("chunks")
    ncols = None
    for candidate in ("grid.chunk_ncols", "phys_state.ncol"):
        try:
            ncols = np.asarray(pool[candidate], dtype=np.int64).reshape(-1)
        except KeyError:
            continue
        break
    if ncols is None:
        raise PICAMStateError(
            f"field {canonical!r} uses pcols/chunks but StatePool has no "
            "chunk ncol field"
        )
    moved_shape = np.moveaxis(values, (pcols_axis, chunks_axis), (0, -1)).shape
    if ncols.size != moved_shape[-1]:
        raise PICAMStateError(
            f"field {canonical!r} chunk dimension does not match chunk ncol metadata"
        )
    moved_mask = np.zeros(moved_shape, dtype=np.bool_, order="F")
    for chunk, count in enumerate(ncols):
        moved_mask[: int(count), ..., chunk] = True
    return np.moveaxis(moved_mask, (0, -1), (pcols_axis, chunks_axis))


def selected_active_values(
    pool: PICAMStatePool,
    name: str,
    selection: object = Ellipsis,
) -> np.ndarray:
    """Flatten active values selected with ordinary rank-local NumPy syntax."""

    values = np.asarray(pool[name][selection])
    mask = np.asarray(active_field_mask(pool, name)[selection], dtype=np.bool_)
    if values.ndim == 0:
        return values.reshape(1) if bool(mask) else np.empty(0, dtype=values.dtype)
    return values[mask]


def edit_active_field(
    pool: PICAMStatePool,
    name: str,
    *,
    selection: object = Ellipsis,
    operation: str,
    value: float | int,
) -> int:
    """Apply one scalar operation to active entries of a rank-local slice."""

    values = pool[name]
    selected = np.asarray(values[selection])
    active = np.asarray(active_field_mask(pool, name)[selection], dtype=np.bool_)
    count = int(active.sum())
    if count == 0:
        raise PICAMStateError("field selection contains no active CAM values")
    if selected.ndim == 0:
        current = selected.item()
        operations = {
            "fill": lambda: value,
            "add": lambda: current + value,
            "subtract": lambda: current - value,
            "multiply": lambda: current * value,
            "divide": lambda: current / value,
        }
        try:
            updated = operations[str(operation)]()
        except KeyError as exc:
            raise PICAMStateError(
                f"unsupported distributed field edit {operation!r}"
            ) from exc
        values[selection] = updated
        return count

    updated = np.array(selected, copy=True, order="K")
    if operation == "fill":
        updated[active] = value
    else:
        functions = {
            "add": np.add,
            "subtract": np.subtract,
            "multiply": np.multiply,
            "divide": np.divide,
        }
        try:
            function = functions[str(operation)]
        except KeyError as exc:
            raise PICAMStateError(
                f"unsupported distributed field edit {operation!r}"
            ) from exc
        function(
            updated,
            value,
            out=updated,
            where=active,
            casting="same_kind",
        )
    values[selection] = updated
    return count


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
