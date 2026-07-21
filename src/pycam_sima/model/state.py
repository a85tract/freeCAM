"""Fortran-contiguous persistent state owned by Python."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

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


@dataclass(frozen=True, slots=True)
class PointerRecord:
    address: int
    shape: tuple[int, ...]
    dtype: str


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
        self._sealed = False

        for item in self.contracts.values():
            shape = item.shape(self.dimensions)
            self._arrays[item.standard_name] = np.zeros(shape, dtype=item.dtype, order="F")
            for alias in item.aliases:
                self._register_alias(alias, item.standard_name, None, None)
        for rule in alias_rules or default_alias_rules():
            self._register_alias(rule.alias, rule.target, rule.axis, rule.index)

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

    def snapshot_arrays(self, *, readonly: bool = True) -> dict[str, np.ndarray]:
        """Copy canonical storage for an isolated model-state snapshot."""

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
            np.copyto(target, source, casting="no")

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
