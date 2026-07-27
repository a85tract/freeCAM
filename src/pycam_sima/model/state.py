"""Fortran-contiguous persistent state owned by Python."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .contracts import AliasRule, FieldContract, default_alias_rules, default_contracts
from .errors import StateOwnershipError


_STATIC_CATEGORIES = {
    "configuration",
    "constants",
    "vertical_coordinate",
    "grid",
    "topology",
    "communication",
}
_UNSET = object()


@dataclass(frozen=True, slots=True)
class PointerRecord:
    address: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(slots=True)
class NativeObjectHandle:
    """Python-owned lifetime record for one opaque Fortran process object."""

    address: int
    fortran_type: str
    shape: tuple[int, ...]
    owner: Any
    destroy: Callable[[], None]
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.destroy()
            self.released = True


class StatePool:
    def __init__(
        self,
        dimensions: Mapping[str, int],
        contracts: Iterable[FieldContract] | None = None,
        alias_rules: Iterable[AliasRule] | None = None,
    ) -> None:
        self.dimensions = {name: int(value) for name, value in dimensions.items()}
        self.contracts = {
            item.standard_name: item for item in (contracts or default_contracts())
        }
        self._arrays: dict[str, np.ndarray] = {}
        self._aliases: dict[str, tuple[str, int | None, int | None]] = {}
        self._ccpp_fields: dict[str, str] = {}
        self._process_state: dict[str, NativeObjectHandle] = {}
        self._initialized_fields: set[str] = set()
        self._dynamic_fields: set[str] = set()
        self._sealed = False

        initial_contracts = tuple(self.contracts.values())
        self.contracts.clear()
        for item in initial_contracts:
            self.register_field(item, initialized=True, dynamic=False)
        rules = default_alias_rules() if alias_rules is None else alias_rules
        for rule in rules:
            self._register_alias(rule.alias, rule.target, rule.axis, rule.index)
            if rule.ccpp_standard_name is not None:
                self._register_ccpp_name(
                    rule.ccpp_standard_name, rule.alias
                )

    def _register_ccpp_name(self, standard_name: str, field_name: str) -> None:
        key = standard_name.lower()
        if key in self._ccpp_fields:
            raise StateOwnershipError(
                f"duplicate CCPP standard name {standard_name!r}"
            )
        self._ccpp_fields[key] = field_name

    def _register_alias(
        self,
        alias: str,
        target: str,
        axis: int | None,
        index: int | None,
    ) -> None:
        if alias in self.contracts or alias in self._aliases:
            raise StateOwnershipError(f"duplicate state alias {alias!r}")
        if target not in self.contracts:
            raise StateOwnershipError(f"alias {alias!r} targets unknown field {target!r}")
        if (axis is None) != (index is None):
            raise StateOwnershipError(f"alias {alias!r} must specify both axis and index")
        self._aliases[alias] = (target, axis, index)

    def canonical_name(self, name: str) -> str:
        if name in self.contracts:
            return name
        try:
            return self._aliases[name][0]
        except KeyError as exc:
            raise KeyError(f"unknown state field {name!r}") from exc

    def contract(self, name: str) -> FieldContract:
        return self.contracts[self.canonical_name(name)]

    def ccpp_field_name(self, standard_name: str) -> str:
        """Resolve one CCPP standard name to a canonical field or zero-copy alias."""

        try:
            return self._ccpp_fields[standard_name.lower()]
        except KeyError as exc:
            raise KeyError(
                f"no StatePool field provides CCPP standard name "
                f"{standard_name!r}"
            ) from exc

    def get_ccpp(self, standard_name: str, *, unsafe: bool = False) -> np.ndarray:
        return self.get(self.ccpp_field_name(standard_name), unsafe=unsafe)

    def contract_ccpp(self, standard_name: str) -> FieldContract:
        return self.contract(self.ccpp_field_name(standard_name))

    def _resolve(self, name: str) -> np.ndarray:
        if name in self._arrays:
            return self._arrays[name]
        try:
            target, axis, index = self._aliases[name]
        except KeyError as exc:
            raise KeyError(f"unknown state field {name!r}") from exc
        value = self._arrays[target]
        if index is None:
            return value
        selector = [slice(None)] * value.ndim
        selector[axis] = index
        return value[tuple(selector)]

    def get(self, name: str, *, unsafe: bool = False) -> np.ndarray:
        value = self._resolve(name)
        contract = self.contract(name)
        if self._sealed and not contract.writable and not unsafe:
            view = value.view()
            view.flags.writeable = False
            return view
        if unsafe and not value.flags.writeable:
            value.flags.writeable = True
        return value

    def set(self, name: str, value: Any, *, unsafe: bool = False) -> None:
        target = self._resolve(name)
        contract = self.contract(name)
        if self._sealed and not contract.writable and not unsafe:
            raise StateOwnershipError(
                f"{contract.standard_name!r} is read-only after initialization; use unsafe=True"
            )
        if unsafe and not target.flags.writeable:
            target.flags.writeable = True
        np.copyto(target, value, casting="same_kind")
        self._initialized_fields.add(self.canonical_name(name))

    def register_field(
        self,
        contract: FieldContract,
        *,
        initial: Any = _UNSET,
        initialized: bool | None = None,
        dynamic: bool = True,
    ) -> np.ndarray:
        """Add Python-owned canonical storage without moving existing arrays."""

        name = contract.standard_name
        if name in self.contracts or name in self._aliases:
            raise StateOwnershipError(f"duplicate state field {name!r}")
        if contract.owner != "python":
            raise StateOwnershipError(
                f"dynamic field {name!r} must be owned by Python"
            )
        missing_dimensions = [
            item for item in contract.dimensions
            if not str(item).isdigit() and item not in self.dimensions
        ]
        if missing_dimensions:
            raise StateOwnershipError(
                f"field {name!r} uses unknown dimensions "
                f"{missing_dimensions}"
            )
        aliases = tuple(contract.aliases)
        conflicts = [
            alias for alias in aliases
            if alias in self.contracts or alias in self._aliases
        ]
        if conflicts:
            raise StateOwnershipError(
                f"field {name!r} has duplicate aliases {conflicts}"
            )
        ccpp_key = (
            None
            if contract.ccpp_standard_name is None
            else contract.ccpp_standard_name.lower()
        )
        if ccpp_key is not None and ccpp_key in self._ccpp_fields:
            raise StateOwnershipError(
                f"duplicate CCPP standard name "
                f"{contract.ccpp_standard_name!r}"
            )

        try:
            values = np.zeros(
                contract.shape(self.dimensions),
                dtype=contract.dtype,
                order="F",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise StateOwnershipError(
                f"cannot allocate dynamic field {name!r}: {exc}"
            ) from exc
        if initial is not _UNSET:
            np.copyto(values, initial, casting="same_kind")

        self.contracts[name] = contract
        self._arrays[name] = values
        if ccpp_key is not None:
            self._ccpp_fields[ccpp_key] = name
        for alias in aliases:
            self._aliases[alias] = (name, None, None)
        if dynamic:
            self._dynamic_fields.add(name)
        is_initialized = initial is not _UNSET if initialized is None else initialized
        if is_initialized:
            self._initialized_fields.add(name)
        if self._sealed and (
            contract.category in _STATIC_CATEGORIES
            or not contract.writable
        ):
            values.flags.writeable = False
        return values

    def unregister_field(self, name: str) -> None:
        """Remove dynamic canonical storage without moving other arrays."""

        canonical = self.canonical_name(name)
        if canonical not in self._dynamic_fields:
            raise StateOwnershipError(
                f"field {canonical!r} is not a removable dynamic field"
            )
        if any(target == canonical for target, _axis, _index in self._aliases.values()):
            self._aliases = {
                alias: record
                for alias, record in self._aliases.items()
                if record[0] != canonical
            }
        self._ccpp_fields = {
            standard_name: field_name
            for standard_name, field_name in self._ccpp_fields.items()
            if self.canonical_name(field_name) != canonical
        }
        self._arrays.pop(canonical)
        self.contracts.pop(canonical)
        self._initialized_fields.discard(canonical)
        self._dynamic_fields.discard(canonical)

    def mark_initialized(self, name: str) -> None:
        self._initialized_fields.add(self.canonical_name(name))

    def is_initialized(self, name: str) -> bool:
        return self.canonical_name(name) in self._initialized_fields

    @property
    def initialized_fields(self) -> frozenset[str]:
        return frozenset(self._initialized_fields)

    @property
    def dynamic_fields(self) -> frozenset[str]:
        return frozenset(self._dynamic_fields)

    def restore_registration_state(
        self,
        *,
        initialized_fields: Iterable[str],
        dynamic_fields: Iterable[str],
    ) -> None:
        """Restore schema bookkeeping after allocating checkpoint contracts."""

        initialized = {self.canonical_name(name) for name in initialized_fields}
        dynamic = {self.canonical_name(name) for name in dynamic_fields}
        unknown = (initialized | dynamic) - set(self.contracts)
        if unknown:
            raise StateOwnershipError(
                f"checkpoint registration state names unknown fields "
                f"{sorted(unknown)}"
            )
        self._initialized_fields = initialized
        self._dynamic_fields = dynamic

    def seal_static(self) -> None:
        for name, contract in self.contracts.items():
            if contract.category in _STATIC_CATEGORIES or not contract.writable:
                self._arrays[name].flags.writeable = False
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def pointer_records(self) -> dict[str, PointerRecord]:
        return {
            name: PointerRecord(
                address=int(value.__array_interface__["data"][0]),
                shape=value.shape,
                dtype=value.dtype.str,
            )
            for name, value in self._arrays.items()
        }

    def assert_pointer_stability(self, before: Mapping[str, PointerRecord]) -> None:
        after = self.pointer_records()
        changed = [name for name in before if before[name] != after.get(name)]
        if changed:
            raise StateOwnershipError(f"kernel replaced or reshaped Python arrays: {changed}")

    def get_process_state(self, standard_name: str) -> NativeObjectHandle:
        """Return one opaque native object by its CCPP standard name."""

        try:
            handle = self._process_state[standard_name.lower()]
        except KeyError as exc:
            raise KeyError(
                f"no opaque process state provides CCPP standard name "
                f"{standard_name!r}"
            ) from exc
        if handle.released:
            raise StateOwnershipError(
                f"opaque process state {standard_name!r} was released"
            )
        return handle

    def set_process_state(
        self, standard_name: str, handle: NativeObjectHandle
    ) -> None:
        """Install a newly allocated opaque object without replacing one."""

        key = standard_name.lower()
        if key in self._process_state:
            raise StateOwnershipError(
                f"opaque process state {standard_name!r} already exists"
            )
        if handle.address <= 0:
            raise StateOwnershipError(
                f"opaque process state {standard_name!r} has a null address"
            )
        self._process_state[key] = handle

    def release_process_state(self) -> None:
        """Destroy all native objects through the factory that created them."""

        errors: list[str] = []
        for name, handle in reversed(tuple(self._process_state.items())):
            try:
                handle.release()
            except Exception as exc:  # pragma: no cover - native failure path.
                errors.append(f"{name}: {exc}")
        self._process_state.clear()
        if errors:
            raise StateOwnershipError(
                "failed to release opaque process state: " + "; ".join(errors)
            )

    @property
    def process_state_names(self) -> frozenset[str]:
        return frozenset(self._process_state)

    def snapshot_arrays(self, *, readonly: bool = True) -> dict[str, np.ndarray]:
        """Copy canonical storage for an isolated model-state snapshot."""

        if self._process_state:
            raise StateOwnershipError(
                "cannot checkpoint opaque Fortran process state; complete the "
                "suite finalize lifecycle before snapshotting or provide a "
                "type-specific serializer"
            )
        arrays: dict[str, np.ndarray] = {}
        for name, value in self._arrays.items():
            copied = np.array(value, dtype=value.dtype, order="F", copy=True)
            if readonly:
                copied.flags.writeable = False
            arrays[name] = copied
        return arrays

    def restore_arrays(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Restore exact canonical values without replacing owned buffers."""

        expected = set(self._arrays)
        actual = set(arrays)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise StateOwnershipError(
                f"snapshot field mismatch: missing={missing}, extra={extra}"
            )
        for name, target in self._arrays.items():
            source = np.asarray(arrays[name])
            if source.shape != target.shape or source.dtype != target.dtype:
                raise StateOwnershipError(
                    f"snapshot field {name!r} has shape/dtype "
                    f"{source.shape}/{source.dtype}, expected "
                    f"{target.shape}/{target.dtype}"
                )
            was_writeable = bool(target.flags.writeable)
            if not was_writeable:
                target.flags.writeable = True
            try:
                np.copyto(target, source, casting="no")
            finally:
                if not was_writeable:
                    target.flags.writeable = False
            self._initialized_fields.add(name)

    def validate(self, *, finite: bool = True) -> None:
        errors: list[str] = []
        for name, contract in self.contracts.items():
            value = self._arrays[name]
            expected_shape = contract.shape(self.dimensions)
            if value.shape != expected_shape:
                errors.append(f"{name}: shape {value.shape}, expected {expected_shape}")
            if value.dtype != np.dtype(contract.dtype):
                errors.append(f"{name}: dtype {value.dtype}, expected {contract.dtype}")
            if value.ndim > 1 and not value.flags.f_contiguous:
                errors.append(f"{name}: not Fortran contiguous")
            if not value.flags.owndata:
                errors.append(f"{name}: canonical storage does not own its memory")
            if contract.owner != "python":
                errors.append(f"{name}: owner is {contract.owner!r}")
            if finite and value.dtype.kind == "f" and not np.isfinite(value).all():
                errors.append(f"{name}: contains NaN or infinity")
        for alias, (target, _axis, index) in self._aliases.items():
            if index is not None and not np.shares_memory(self._resolve(alias), self._arrays[target]):
                errors.append(f"{alias}: constituent alias is not zero-copy")
        if errors:
            raise StateOwnershipError("; ".join(errors))

    def inventory(self) -> list[dict[str, Any]]:
        records = []
        for name, contract in self.contracts.items():
            value = self._arrays[name]
            records.append(
                {
                    **contract.machine_record(),
                    "shape": list(value.shape),
                    "nbytes": int(value.nbytes),
                    "address": int(value.__array_interface__["data"][0]),
                    "fortran_contiguous": bool(value.flags.f_contiguous),
                    "owns_data": bool(value.flags.owndata),
                }
            )
        return records
