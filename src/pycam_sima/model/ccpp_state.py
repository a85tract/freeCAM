"""Compile CCPP metadata into Python-owned StatePool field requirements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import (
    FieldContract,
    component_contracts,
    default_alias_rules,
    default_contracts,
    process_contract_templates,
)
from .device_catalog import (
    CatalogArgument,
    DeviceCatalog,
    SchemeCatalogEntry,
    _dimension_standard,
)
from .errors import DeviceContractError


_INTERNAL = {"ccpp_error_code", "ccpp_error_message", "scheme_name"}
_DIMENSION_ALIASES = {
    "horizontal_dimension": "nphys_local",
    "horizontal_loop_extent": "nphys_local",
    "vertical_layer_dimension": "pver",
    "vertical_interface_dimension": "pverp",
}
_PRIMITIVE_DTYPES = {
    "real": "float64",
    "integer": "int32",
    "logical": "bool",
}


def _canonical_dimension(expression: str) -> str:
    if not expression.strip().strip(":"):
        return "__allocatable__"
    standard = _dimension_standard(expression)
    return _DIMENSION_ALIASES.get(standard, standard)


def _normalized_units(value: str) -> str:
    return " ".join(str(value or "1").strip().lower().split())


@dataclass(frozen=True, slots=True)
class FieldVariant:
    fortran_type: str
    kind: str
    dimensions: tuple[str, ...]
    units: str
    allocatable: bool
    intents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CCPPFieldRequirement:
    standard_name: str
    variants: tuple[FieldVariant, ...]
    schemes: tuple[str, ...]

    @property
    def primitive(self) -> bool:
        return all(
            item.fortran_type in {*_PRIMITIVE_DTYPES, "character"}
            and "__allocatable__" not in item.dimensions
            and (not item.allocatable or bool(item.dimensions))
            for item in self.variants
        )

    @property
    def requires_conversion(self) -> bool:
        signatures = {
            (
                item.fortran_type,
                item.kind,
                item.dimensions,
                _normalized_units(item.units),
            )
            for item in self.variants
        }
        return len(signatures) > 1

    @property
    def dtype(self) -> str:
        types = {item.fortran_type for item in self.variants}
        if len(types) != 1:
            raise DeviceContractError(
                f"{self.standard_name!r} has incompatible types {types}"
            )
        fortran_type = next(iter(types))
        if fortran_type == "character":
            lengths: list[int] = []
            for item in self.variants:
                text = item.kind.replace(" ", "").lower()
                if text.startswith("len=") and text[4:].isdigit():
                    lengths.append(int(text[4:]))
            return f"S{max(lengths, default=512)}"
        try:
            return _PRIMITIVE_DTYPES[fortran_type]
        except KeyError as exc:
            raise DeviceContractError(
                f"{self.standard_name!r} is opaque derived type "
                f"{fortran_type!r}"
            ) from exc

    @property
    def dimensions(self) -> tuple[str, ...]:
        shapes = {item.dimensions for item in self.variants}
        if len(shapes) != 1:
            raise DeviceContractError(
                f"{self.standard_name!r} has incompatible shapes "
                f"{sorted(shapes)}"
            )
        return next(iter(shapes))

    @property
    def units(self) -> str:
        units = {_normalized_units(item.units) for item in self.variants}
        if len(units) != 1:
            raise DeviceContractError(
                f"{self.standard_name!r} requires unit conversion among "
                f"{sorted(units)}"
            )
        return next(iter(units))

    @property
    def intent(self) -> str:
        intents = {
            intent
            for item in self.variants
            for intent in item.intents
        }
        if "inout" in intents or {"in", "out"} <= intents:
            return "inout"
        if "out" in intents:
            return "out"
        return "in"

    def contract(self) -> FieldContract:
        if not self.primitive:
            raise DeviceContractError(
                f"{self.standard_name!r} is opaque process state"
            )
        if self.requires_conversion:
            # Shape aliases are normalized while collecting variants, so any
            # remaining difference is a real type/kind/unit conversion.
            self.dtype
            self.dimensions
            self.units
        return FieldContract(
            standard_name=f"ccpp_{self.standard_name}",
            ccpp_standard_name=self.standard_name,
            dtype=self.dtype,
            dimensions=self.dimensions,
            intent=self.intent,
            category="ccpp_state",
            units=self.units,
            owner="python",
            lifetime="persistent",
            restart=True,
            writable=True,
        )


class CCPPStateSchema:
    """All standard-name fields and opaque process objects for one suite."""

    def __init__(
        self,
        suite: str,
        requirements: Mapping[str, CCPPFieldRequirement],
        dimension_names: Iterable[str],
        unresolved_schemes: Iterable[str] = (),
    ) -> None:
        self.suite = suite
        self.requirements = dict(requirements)
        self.dimension_names = frozenset(dimension_names)
        self.unresolved_schemes = tuple(sorted(set(unresolved_schemes)))

    @classmethod
    def from_catalog(
        cls, catalog: DeviceCatalog, suite: str
    ) -> "CCPPStateSchema":
        if suite not in {
            occurrence.suite
            for entry in catalog.entries.values()
            for occurrence in entry.occurrences
        }:
            raise ValueError(f"unknown CCPP suite {suite!r}")

        active_entries = [
            entry
            for entry in catalog.entries.values()
            if any(item.suite == suite for item in entry.occurrences)
        ]
        return cls._from_entries(suite, active_entries)

    @classmethod
    def from_scheme_names(
        cls,
        catalog: DeviceCatalog,
        suite: str,
        schemes: Iterable[str],
    ) -> "CCPPStateSchema":
        """Compile a custom suite from cataloged process names."""

        names = {str(name).lower() for name in schemes}
        active_entries = [
            entry
            for name, entry in catalog.entries.items()
            if name in names
        ]
        return cls._from_entries(
            suite,
            active_entries,
            unresolved_schemes=names - set(catalog.entries),
        )

    @classmethod
    def _from_entries(
        cls,
        suite: str,
        active_entries: Iterable[SchemeCatalogEntry],
        *,
        unresolved_schemes: Iterable[str] = (),
    ) -> "CCPPStateSchema":
        arguments: dict[str, list[tuple[str, CatalogArgument]]] = {}
        dimension_names: set[str] = {
            "nphys_local",
            "pver",
            "pverp",
            "ccpp_constant_one",
            "ccpp_constant_two",
        }
        for entry in active_entries:
            for endpoint in entry.entrypoints:
                for argument in endpoint.arguments:
                    if argument.standard_name in _INTERNAL:
                        continue
                    for expression in argument.dimensions:
                        dimension = _canonical_dimension(expression)
                        if (
                            not dimension.isdigit()
                            and dimension != "__allocatable__"
                        ):
                            dimension_names.add(dimension)
                    arguments.setdefault(
                        argument.standard_name, []
                    ).append((entry.name, argument))

        # Dimension standard names are scalar controls supplied directly from
        # StatePool.dimensions, not separately allocated arrays.
        dimension_standards = {
            argument.standard_name
            for values in arguments.values()
            for _, argument in values
            if not argument.dimensions
            and _DIMENSION_ALIASES.get(
                argument.standard_name, argument.standard_name
            )
            in dimension_names
        }
        requirements: dict[str, CCPPFieldRequirement] = {}
        for standard_name, values in arguments.items():
            if standard_name in dimension_standards:
                continue
            variants: dict[
                tuple[str, str, tuple[str, ...], str, bool], set[str]
            ] = {}
            schemes: set[str] = set()
            for scheme, argument in values:
                schemes.add(scheme)
                signature = (
                    argument.fortran_type,
                    argument.kind,
                    tuple(
                        _canonical_dimension(item)
                        for item in argument.dimensions
                    ),
                    argument.units or "1",
                    argument.allocatable,
                )
                variants.setdefault(signature, set()).add(argument.intent)
            requirements[standard_name] = CCPPFieldRequirement(
                standard_name=standard_name,
                variants=tuple(
                    FieldVariant(*signature, tuple(sorted(intents)))
                    for signature, intents in sorted(variants.items())
                ),
                schemes=tuple(sorted(schemes)),
            )
        return cls(
            suite,
            requirements,
            dimension_names,
            unresolved_schemes,
        )

    @property
    def primitive_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, requirement in self.requirements.items()
            if requirement.primitive
        )

    @property
    def opaque_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, requirement in self.requirements.items()
            if not requirement.primitive
        )

    @property
    def conversion_fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, requirement in self.requirements.items()
            if requirement.requires_conversion
        )

    def additional_contracts(
        self,
        existing: Iterable[FieldContract] | None = None,
        *,
        allow_conversions: bool = False,
    ) -> tuple[FieldContract, ...]:
        existing_contracts = tuple(existing or default_contracts())
        provided = {
            item.ccpp_standard_name.lower()
            for item in existing_contracts
            if item.ccpp_standard_name
        }
        contracts: list[FieldContract] = []
        errors: list[str] = []
        for name, requirement in sorted(self.requirements.items()):
            if name in provided or not requirement.primitive:
                continue
            if requirement.requires_conversion and not allow_conversions:
                errors.append(name)
                continue
            try:
                contracts.append(requirement.contract())
            except DeviceContractError:
                errors.append(name)
        if errors:
            raise DeviceContractError(
                f"suite {self.suite!r} requires explicit conversion policy "
                f"for fields {errors}"
            )
        return tuple(contracts)

    def pool_contract_groups(
        self,
    ) -> tuple[tuple[FieldContract, ...], tuple[FieldContract, ...]]:
        """Compile initialized templates and metadata-generated fields.

        Component fields are always present. Process-specific templates are
        included only when the active suite metadata names them. Both selected
        process fields and generated primitive metadata fields begin
        uninitialized; a producer or explicit Python initializer must populate
        them before a device may consume them.
        """

        initialized = list(component_contracts())
        process_fields: list[FieldContract] = []
        for contract in process_contract_templates():
            standard_name = contract.ccpp_standard_name
            if (
                standard_name is not None
                and standard_name.lower() in self.requirements
            ):
                process_fields.append(contract)
        available = [*initialized, *process_fields]
        alias_standard_names = {
            rule.ccpp_standard_name.lower()
            for rule in default_alias_rules()
            if rule.ccpp_standard_name is not None
        }
        generated = tuple(
            contract
            for contract in self.additional_contracts(available)
            if (
                contract.ccpp_standard_name is None
                or contract.ccpp_standard_name.lower()
                not in alias_standard_names
            )
        )
        return tuple(initialized), (*process_fields, *generated)

    def pool_contracts(self) -> tuple[FieldContract, ...]:
        """Return the complete selected schema in allocation order."""

        initialized, generated = self.pool_contract_groups()
        return (*initialized, *generated)

    def report(self) -> dict[str, Any]:
        try:
            contracts = self.pool_contracts()
            state_pool_field_count: int | None = len(contracts)
            state_pool_error: str | None = None
        except DeviceContractError as exc:
            state_pool_field_count = None
            state_pool_error = str(exc)
        return {
            "suite": self.suite,
            "field_count": len(self.requirements),
            "primitive_field_count": len(self.primitive_fields),
            "opaque_field_count": len(self.opaque_fields),
            "conversion_field_count": len(self.conversion_fields),
            "required_dimensions": sorted(self.dimension_names),
            "opaque_fields": list(self.opaque_fields),
            "conversion_fields": list(self.conversion_fields),
            "unresolved_schemes": list(self.unresolved_schemes),
            "state_pool_field_count": state_pool_field_count,
            "suite_selected_field_count": (
                None
                if state_pool_field_count is None
                else state_pool_field_count - len(component_contracts())
            ),
            "state_pool_error": state_pool_error,
        }
