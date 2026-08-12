"""Promote rank-local Fortran caller arguments into the Python StatePool."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import numpy as np

from .errors import PICAMConfigurationError, PICAMStateError
from .physics_catalog import PICAMPhysicsCatalog, PICAMPhysicsProcess
from .state import PICAMFieldContract, PICAMStatePool


_SIMPLE_DIMENSION = re.compile(r"[A-Za-z_]\w*|[1-9]\d*")
_OFFSET_DIMENSION = re.compile(r"([A-Za-z_]\w*)\s*\+\s*([1-9]\d*)")
@dataclass(frozen=True, slots=True)
class PICAMProcessArgumentBinding:
    argument: str
    field: str
    dtype: str | None
    fortran_type: str
    intent: str
    local_rank: int
    aggregate_dimensions: tuple[str, ...]
    source: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "argument": self.argument,
            "field": self.field,
            "dtype": self.dtype,
            "fortran_type": self.fortran_type,
            "intent": self.intent,
            "local_rank": self.local_rank,
            "aggregate_dimensions": list(self.aggregate_dimensions),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PICAMPromotedProcess:
    name: str
    qualified_name: str
    bindings: tuple[PICAMProcessArgumentBinding, ...]
    created_fields: tuple[str, ...]
    created_dimensions: tuple[str, ...]
    native_available: bool

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(binding.field for binding in self.bindings)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "bindings": [binding.to_payload() for binding in self.bindings],
            "created_fields": list(self.created_fields),
            "created_dimensions": list(self.created_dimensions),
            "native_available": self.native_available,
        }


class PICAMProcessContextRegistry:
    """Own promoted process arguments and their rank-local StatePool fields."""

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self.catalog = PICAMPhysicsCatalog.load_default()
        self._records: dict[str, PICAMPromotedProcess] = {}

    def __contains__(self, name: str) -> bool:
        return str(name) in self._records

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def record(self, name: str) -> PICAMPromotedProcess:
        try:
            return self._records[str(name)]
        except KeyError as exc:
            raise PICAMStateError(f"physics process {name!r} is not promoted") from exc

    def promote(
        self,
        name: str,
        *,
        bindings: Mapping[str, str] | None = None,
        initials: Mapping[str, Any] | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> PICAMPromotedProcess:
        process = self.catalog.process(name)
        if process.level != "process":
            raise PICAMConfigurationError(
                f"{name!r} is a helper routine, not a physical process"
            )
        if not process.generated_adapter:
            raise PICAMConfigurationError(
                f"{process.name!r} does not yet have a compiled pointer adapter; "
                f"adapter status is {process.adapter_status!r}"
            )
        if process.name in self._records:
            raise PICAMStateError(f"physics process {process.name!r} is already promoted")
        pool = self.driver.pool
        overrides = {str(key): int(value) for key, value in (dimensions or {}).items()}
        created_dimensions: list[str] = []
        dimensions_before = frozenset(pool.dimensions)
        for dimension, extent in overrides.items():
            if extent < 1:
                raise PICAMStateError(f"dimension {dimension!r} must be positive")
            existing = pool.dimensions.get(dimension)
            if existing is not None and existing != extent:
                raise PICAMStateError(
                    f"dimension {dimension!r}={extent} conflicts with {existing}"
                )
        requested = {str(key): str(value) for key, value in (bindings or {}).items()}
        values = dict(initials or {})
        created: list[str] = []
        try:
            for dimension, extent in overrides.items():
                pool.dimensions[dimension] = extent
            plan = self._plan(process, requested, values)
            created_dimensions.extend(
                dimension
                for dimension in pool.dimensions
                if dimension not in dimensions_before
            )
            for argument, field, contract, initial, source in plan:
                if contract is not None:
                    target = pool.create(contract, initial=0)
                    created.append(contract.name)
                    if isinstance(initial, np.ndarray):
                        if initial.shape != target.shape:
                            raise PICAMStateError(
                                f"initial value for {argument['name']!r} has shape "
                                f"{initial.shape}; expected {target.shape}"
                            )
                        np.copyto(target, np.asarray(initial, dtype=target.dtype))
                    elif initial is not None:
                        target[...] = initial
                array = np.asarray(pool[field])
                dtype = argument.get("dtype")
                if dtype is None:
                    if array.dtype != np.dtype("uint8") and array.dtype.kind != "V":
                        raise PICAMStateError(
                            f"opaque binding {argument['name']!r}->{field!r} "
                            "must use a uint8 owner buffer or a void-record array"
                        )
                    if array.dtype.kind == "V":
                        expected_rank = int(argument["rank"]) + 1
                        if array.ndim != expected_rank:
                            raise PICAMStateError(
                                f"opaque binding {argument['name']!r}->{field!r} "
                                f"has aggregate rank {array.ndim}; expected "
                                f"{expected_rank} including chunks"
                            )
                        if array.shape[-1] != int(pool.dimensions["chunks"]):
                            raise PICAMStateError(
                                f"opaque binding {argument['name']!r}->{field!r} "
                                "must have the chunks axis last"
                            )
                else:
                    expected_rank = int(argument["rank"]) + 1
                    if array.ndim != expected_rank:
                        raise PICAMStateError(
                            f"binding {argument['name']!r}->{field!r} has rank "
                            f"{array.ndim}; expected aggregate rank {expected_rank}"
                        )
                    expected_dtype = self._numpy_dtype(argument, initial)
                    if (
                        expected_dtype is not None
                        and array.dtype != expected_dtype
                    ):
                        raise PICAMStateError(
                            f"binding {argument['name']!r}->{field!r} has dtype "
                            f"{array.dtype}; expected {expected_dtype}"
                        )
                    if array.shape[-1] != int(pool.dimensions["chunks"]):
                        raise PICAMStateError(
                            f"binding {argument['name']!r}->{field!r} does not have "
                            "the rank-local chunks axis last"
                        )
                intent = str(argument.get("intent") or "inout").lower()
                if intent in {"out", "inout"} and not pool.contract(field).writable:
                    raise PICAMStateError(
                        f"binding {argument['name']!r}->{field!r} is read-only "
                        f"but the Fortran intent is {intent}"
                    )
                if array.ndim > 1 and not array.flags.f_contiguous:
                    raise PICAMStateError(
                        f"binding {argument['name']!r}->{field!r} must be "
                        "Fortran contiguous"
                    )
            has_adapter = getattr(self.driver.backend, "has_process_adapter", None)
            native = process.name in tuple(
                getattr(self.driver.backend, "direct_kernels", ())
            ) or (
                callable(has_adapter)
                and has_adapter(process.qualified_name, process.source)
            )
            record = PICAMPromotedProcess(
                name=process.name,
                qualified_name=process.qualified_name,
                bindings=tuple(
                    PICAMProcessArgumentBinding(
                        argument=str(argument["name"]),
                        field=field,
                        dtype=(
                            None
                            if argument.get("dtype") is None
                            else str(argument["dtype"])
                        ),
                        fortran_type=str(argument.get("fortran_type") or "unknown"),
                        intent=str(argument["intent"]),
                        local_rank=int(argument["rank"]),
                        aggregate_dimensions=tuple(pool.contract(field).dimensions),
                        source=source,
                    )
                    for argument, field, _, _, source in plan
                ),
                created_fields=tuple(created),
                created_dimensions=tuple(created_dimensions),
                native_available=native,
            )
            self._records[process.name] = record
            return record
        except BaseException:
            for field in reversed(created):
                if field in pool:
                    pool.remove(field)
            for dimension in reversed(tuple(pool.dimensions)):
                if dimension not in dimensions_before:
                    pool.dimensions.pop(dimension, None)
            raise

    def _plan(
        self,
        process: PICAMPhysicsProcess,
        requested: Mapping[str, str],
        initials: Mapping[str, Any],
    ) -> tuple[tuple[Mapping[str, Any], str, PICAMFieldContract | None, Any, str], ...]:
        pool = self.driver.pool
        if "chunks" not in pool.dimensions:
            raise PICAMStateError(
                "process arguments can be promoted only after CAM grid initialization"
            )
        process_arguments = tuple(
            item for item in process.arguments if not bool(item.get("procedure", False))
        )
        if process.result is not None:
            process_arguments = (
                *process_arguments,
                {**process.result, "intent": "out", "function_result": True},
            )
        arguments = {str(item["name"]): item for item in process_arguments}
        unknown = (set(requested) | set(initials)).difference(arguments)
        if unknown:
            raise PICAMConfigurationError(
                f"unknown arguments for {process.name!r}: " + ", ".join(sorted(unknown))
            )
        planned: list[
            tuple[Mapping[str, Any], str, PICAMFieldContract | None, Any, str]
        ] = []
        missing_inputs: list[str] = []
        for argument in process_arguments:
            argument_name = str(argument["name"])
            dtype = argument.get("dtype")
            field = requested.get(argument_name)
            source = "explicit"
            if field is not None and argument_name in initials:
                raise PICAMConfigurationError(
                    f"{process.name}.{argument_name} cannot specify both a "
                    "field binding and an initial value"
                )
            if field is not None:
                try:
                    field = pool.canonical_name(field)
                except KeyError as exc:
                    raise PICAMStateError(
                        f"binding {argument_name!r} references unknown field {field!r}"
                    ) from exc
                dimensions = pool.contract(field).dimensions
            elif dtype is None and argument_name in initials:
                initial_array = np.asarray(initials[argument_name])
                if initial_array.dtype.kind != "V":
                    raise PICAMStateError(
                        f"opaque initial {process.name}.{argument_name} must be "
                        "a NumPy void-record array produced by its Fortran "
                        "subsystem lifecycle"
                    )
                dimensions = self._dimensions_from_initial(
                    process,
                    argument,
                    initial_array,
                )
                source = "opaque_initial"
            elif dtype is None:
                field = self._native_owner_field(str(argument.get("fortran_type") or ""))
                if field is None:
                    raise PICAMConfigurationError(
                        f"{process.name}.{argument_name} uses "
                        f"{argument.get('fortran_type')!r}; pass bindings={{"
                        f"{argument_name!r}: <StatePool opaque owner>}} with a "
                        "valid object created by that subsystem's lifecycle"
                    )
                dimensions = pool.contract(field).dimensions
                source = "native_owner"
            else:
                try:
                    dimensions = self._aggregate_dimensions(process, argument)
                except PICAMConfigurationError:
                    if argument_name not in initials:
                        raise
                    dimensions = self._dimensions_from_initial(
                        process,
                        argument,
                        initials[argument_name],
                    )
            if field is None and argument_name not in initials:
                field = self._infer_field(argument_name, dimensions, str(dtype))
                source = "inferred" if field is not None else "promoted"
            initial = initials.get(argument_name)
            if field is None:
                if argument_name in initials:
                    source = "initialized"
                automatic = self._automatic_initial(argument_name)
                if initial is None and automatic is not None:
                    initial = automatic
                intent = str(argument.get("intent") or "inout").lower()
                if initial is None and intent in {"in", "inout"}:
                    missing_inputs.append(argument_name)
                    continue
                field = f"process_context.{process.name}.{argument_name}"
                contract = PICAMFieldContract(
                    name=field,
                    dimensions=dimensions,
                    dtype=(
                        np.asarray(initial).dtype.str
                        if dtype is None and initial is not None
                        else self._contract_dtype(argument, initial)
                    ),
                    category="process_context",
                    # These arrays replace caller-local working storage.  Even
                    # intent(in) values must remain user-editable between
                    # isolated calls, while the adapter still enforces the
                    # original Fortran intent for the call itself.
                    writable=True,
                    restart=True,
                    aliases=(f"{process.name}_{argument_name}",),
                    dynamic=True,
                )
            else:
                contract = None
            planned.append((argument, field, contract, initial, source))
        if missing_inputs:
            raise PICAMStateError(
                f"{process.name!r} needs initial values or field bindings for: "
                + ", ".join(missing_inputs)
            )
        return tuple(planned)

    def _contract_dtype(self, argument: Mapping[str, Any], initial: Any) -> str:
        dtype = str(argument.get("dtype") or "")
        if dtype != "character":
            return dtype
        length = str(argument.get("character_length") or "").strip()
        if length.isdigit():
            return f"S{int(length)}"
        if initial is None:
            raise PICAMConfigurationError(
                f"character argument {argument['name']!r} needs a bytes/string "
                "initial value so its element length is explicit"
            )
        values = np.asarray(initial)
        if values.dtype.kind not in {"S", "a"}:
            values = np.asarray(initial, dtype=f"S{max(1, len(str(initial)))}")
        return values.dtype.str

    def _numpy_dtype(
        self, argument: Mapping[str, Any], initial: Any
    ) -> np.dtype | None:
        dtype = argument.get("dtype")
        if dtype is None:
            return None
        return np.dtype(self._contract_dtype(argument, initial))

    def _native_owner_field(self, fortran_type: str) -> str | None:
        owner = {
            "type:physics_state": "phys_state",
            "type:physics_tend": "phys_tend",
            "type:cam_in_t": "cam_in",
            "type:cam_out_t": "cam_out",
        }.get(str(fortran_type).lower())
        if owner is None:
            return None
        field = f"__native_owner.{owner}"
        return field if field in self.driver.pool else None

    def _aggregate_dimensions(
        self,
        process: PICAMPhysicsProcess,
        argument: Mapping[str, Any],
    ) -> tuple[str, ...]:
        result: list[str] = []
        for axis, raw in enumerate(argument.get("dimensions", ())):
            token = str(raw).strip()
            offset = _OFFSET_DIMENSION.fullmatch(token)
            if offset is not None:
                base, amount = offset.groups()
                if base not in self.driver.pool.dimensions:
                    raise PICAMStateError(
                        f"{process.name}.{argument['name']} needs dimension {base!r}; "
                        "pass dimensions={...}"
                    )
                name = f"__{base}_plus_{amount}"
                extent = int(self.driver.pool.dimensions[base]) + int(amount)
                existing = self.driver.pool.dimensions.setdefault(name, extent)
                if existing != extent:
                    raise PICAMStateError(f"inconsistent dimension {name!r}")
            elif _SIMPLE_DIMENSION.fullmatch(token) is None:
                raise PICAMConfigurationError(
                    f"{process.name}.{argument['name']} dimension {token!r} "
                    "requires an explicit field binding or aggregate initial array"
                )
            elif token.isdigit():
                name = f"__extent_{token}"
                self.driver.pool.dimensions.setdefault(name, int(token))
            elif offset is None:
                name = token
            if name not in self.driver.pool.dimensions:
                raise PICAMStateError(
                    f"{process.name}.{argument['name']} needs dimension {name!r}; "
                    "pass dimensions={...}"
                )
            result.append(name)
        result.append("chunks")
        return tuple(result)

    def _dimensions_from_initial(
        self,
        process: PICAMPhysicsProcess,
        argument: Mapping[str, Any],
        initial: Any,
    ) -> tuple[str, ...]:
        array = np.asarray(initial)
        expected_rank = int(argument["rank"]) + 1
        if array.ndim != expected_rank:
            raise PICAMStateError(
                f"aggregate initial for {process.name}.{argument['name']} has rank "
                f"{array.ndim}; expected {expected_rank} including chunks"
            )
        if array.shape[-1] != int(self.driver.pool.dimensions["chunks"]):
            raise PICAMStateError(
                f"aggregate initial for {process.name}.{argument['name']} must "
                "have chunks as its last axis"
            )
        names: list[str] = []
        for axis, extent in enumerate(array.shape[:-1]):
            name = f"__{process.name}_{argument['name']}_axis_{axis}"
            existing = self.driver.pool.dimensions.setdefault(name, int(extent))
            if existing != int(extent):
                raise PICAMStateError(f"inconsistent dimension {name!r}")
            names.append(name)
        names.append("chunks")
        return tuple(names)

    def _infer_field(
        self, argument: str, dimensions: tuple[str, ...], dtype: str
    ) -> str | None:
        pool = self.driver.pool
        if dtype == "character":
            return None
        special = {"ncol": "grid.chunk_ncols", "lchnk": "grid.chunk_id"}
        if argument in special and special[argument] in pool:
            return special[argument]
        matches = [
            name
            for name, contract in pool.contracts.items()
            if name.rsplit(".", 1)[-1] == argument
            and contract.dimensions == dimensions
            and np.dtype(contract.dtype) == np.dtype(dtype)
        ]
        return matches[0] if len(matches) == 1 else None

    def _automatic_initial(self, argument: str) -> Any | None:
        pool = self.driver.pool
        if argument in pool.dimensions:
            return int(pool.dimensions[argument])
        if argument in {"dtime", "deltat", "ztodt", "dt"}:
            return float(self.driver.config.timestep_seconds)
        if argument == "nstep":
            return int(self.driver.clock.nstep)
        return None

    def invoke(self, name: str) -> None:
        """Execute one bound catalog process without recording plan metadata."""

        record = self.record(name)
        if not record.native_available:
            raise PICAMConfigurationError(
                f"{record.name!r} arguments are in StatePool, but the currently "
                "loaded CAM device was built without its direct adapter"
            )
        execute_catalog = getattr(self.driver.backend, "execute_catalog_process", None)
        if callable(execute_catalog) and getattr(
            self.driver.backend,
            "has_process_adapter",
            lambda _name, _source=None: False,
        )(record.qualified_name, self.catalog.process(record.name).source):
            before = self.driver.pool.pointer_records()
            process = self.catalog.process(record.name)
            arguments = tuple(
                item
                for item in process.arguments
                if not bool(item.get("procedure", False))
            )
            if process.result is not None:
                arguments = (*arguments, process.result)
            execute_catalog(
                record.name,
                record.qualified_name,
                process.source,
                self.driver.pool,
                bindings={
                    f"process_context.{record.name}.{binding.argument}": binding.field
                    for binding in record.bindings
                },
                arguments=arguments,
                fcomm=self.driver.fcomm,
            )
            self.driver.pool.assert_pointer_stability(before)
            return
        execute = getattr(self.driver.backend, "execute_promoted_process", None)
        if not callable(execute):
            raise PICAMConfigurationError(
                "the selected CAM backend cannot run promoted processes"
            )
        before = self.driver.pool.pointer_records()
        execute(
            record.name,
            self.driver.pool,
            bindings={
                f"process_context.{record.name}.{binding.argument}": binding.field
                for binding in record.bindings
            },
            fcomm=self.driver.fcomm,
        )
        self.driver.pool.assert_pointer_stability(before)

    def run(self, name: str) -> Any:
        """Execute one bound process as an isolated experimental call."""

        self.invoke(name)
        record = self.record(name)
        return self.driver._record_promoted_process(record.name)

    def dependencies(self, field: str) -> tuple[str, ...]:
        canonical = self.driver.pool.canonical_name(field)
        return tuple(
            name
            for name, record in self._records.items()
            if canonical in {binding.field for binding in record.bindings}
        )

    def remove(self, name: str) -> PICAMPromotedProcess:
        record = self.record(name)
        for field in record.created_fields:
            if field in self.driver.pool:
                self.driver.pool.remove(field)
        del self._records[record.name]
        used_dimensions = {
            dimension
            for contract in self.driver.pool.contracts.values()
            for dimension in contract.dimensions
        }
        for dimension in record.created_dimensions:
            if dimension not in used_dimensions:
                self.driver.pool.dimensions.pop(dimension, None)
        return record
