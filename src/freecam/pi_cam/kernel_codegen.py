"""Generate stable pointer-table adapters for ordinary Fortran kernels.

The generated wrapper does not reproduce a kernel body.  It converts the
generic Python pointer table into typed Fortran pointers and calls the
original routine already present in the CAM image.  A descriptor can select
one rank-local chunk at a time from an aggregate StatePool field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import yaml

from .errors import PICAMConfigurationError


_FORTRAN_TYPES = {
    np.dtype("float64").str: "real(c_double)",
    np.dtype("int32").str: "integer(c_int32_t)",
    np.dtype("int64").str: "integer(c_int64_t)",
}


@dataclass(frozen=True, slots=True)
class DirectKernelArgument:
    """One StatePool field bound to one dummy argument of the original routine.

    ``pointer`` and ``fortran_type`` record how the original declaration differs
    from a plain array of ``dtype``.  A ``pointer`` dummy cannot receive an array
    section, so the wrapper hands it a Fortran pointer associated with the same
    storage; a ``logical`` dummy has no portable C type, so the wrapper carries
    the value as ``int32`` and converts it at the call.  Both are conversions of
    shape and scalar representation, not of numerical content.
    """

    field: str
    dtype: str
    rank: int
    intent: str
    chunk_axis: int
    fixed_indices: tuple[tuple[int, int], ...] = ()
    extents: tuple[str, ...] = ()
    pointer: bool = False
    fortran_type: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DirectKernelArgument":
        dtype = np.dtype(str(payload["dtype"])).str
        if dtype not in _FORTRAN_TYPES:
            raise PICAMConfigurationError(
                f"direct kernel field {payload.get('field')!r} uses unsupported "
                f"dtype {np.dtype(dtype)}"
            )
        rank = int(payload["rank"])
        chunk_axis = int(payload.get("chunk_axis", rank))
        if rank < 1 or not 1 <= chunk_axis <= rank:
            raise PICAMConfigurationError("direct kernel chunk_axis is outside its rank")
        intent = str(payload.get("intent", "inout")).lower()
        if intent not in {"in", "out", "inout"}:
            raise PICAMConfigurationError(f"invalid direct kernel intent {intent!r}")
        raw_indices = payload.get("fixed_indices", {})
        if not isinstance(raw_indices, Mapping):
            raise PICAMConfigurationError("fixed_indices must be an axis-to-index mapping")
        fixed = tuple(sorted((int(axis), int(index)) for axis, index in raw_indices.items()))
        if any(axis < 1 or axis > rank or axis == chunk_axis or index < 1 for axis, index in fixed):
            raise PICAMConfigurationError("invalid direct kernel fixed index")
        raw_extents = payload.get("extents", ())
        if raw_extents and (
            not isinstance(raw_extents, list) or len(raw_extents) != rank
        ):
            raise PICAMConfigurationError(
                "direct kernel extents must contain one expression per axis"
            )
        extents = tuple(str(extent) for extent in raw_extents)
        if any(
            re.fullmatch(r"(?:chunks|[A-Za-z_]\w*|[1-9]\d*)", extent) is None
            for extent in extents
        ):
            raise PICAMConfigurationError(
                "direct kernel extents accept only names, positive integers, or chunks"
            )
        pointer = bool(payload.get("pointer", False))
        if pointer and chunk_axis != rank:
            raise PICAMConfigurationError(
                f"direct kernel field {payload.get('field')!r} is a pointer dummy, "
                "so its chunk axis must be last for the per-chunk slice to be "
                "contiguous"
            )
        if pointer and fixed:
            raise PICAMConfigurationError(
                f"direct kernel field {payload.get('field')!r} cannot fix an axis "
                "of a pointer dummy"
            )
        fortran_type = str(payload.get("fortran_type", "")).lower()
        if fortran_type not in {"", "logical"}:
            raise PICAMConfigurationError(
                f"direct kernel field {payload.get('field')!r} declares unsupported "
                f"fortran_type {fortran_type!r}"
            )
        if fortran_type == "logical":
            if dtype != np.dtype("int32").str or rank != 1 or chunk_axis != 1:
                raise PICAMConfigurationError(
                    f"direct kernel field {payload.get('field')!r} carries a Fortran "
                    "logical, which must be one int32 value per chunk"
                )
            if intent != "in":
                raise PICAMConfigurationError(
                    f"direct kernel field {payload.get('field')!r} carries a Fortran "
                    "logical, which the wrapper can only pass as intent(in)"
                )
        return cls(
            field=str(payload["field"]),
            dtype=dtype,
            rank=rank,
            intent=intent,
            chunk_axis=chunk_axis,
            fixed_indices=fixed,
            extents=extents,
            pointer=pointer,
            fortran_type=fortran_type,
        )

    @property
    def storage_type(self) -> str:
        return _FORTRAN_TYPES[self.dtype]

    def operation_payload(self) -> dict[str, object]:
        return {
            "field": self.field,
            "dtype": np.dtype(self.dtype).name,
            "rank": self.rank,
            "intent": self.intent,
        }


@dataclass(frozen=True, slots=True)
class DirectKernel:
    name: str
    routine: str
    symbol: str
    arguments: tuple[DirectKernelArgument, ...]
    modules: tuple[tuple[str, tuple[str, ...]], ...] = ()
    action_id: int = 0

    @property
    def operation_name(self) -> str:
        return f"direct_kernel.{self.name}"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DirectKernel":
        arguments = payload.get("arguments")
        if not isinstance(arguments, list) or not arguments:
            raise PICAMConfigurationError("direct kernel requires arguments")
        name = str(payload["name"])
        raw_modules = payload.get("modules", {})
        if not isinstance(raw_modules, Mapping):
            raise PICAMConfigurationError("direct kernel modules must be a mapping")
        parsed_arguments = tuple(
            DirectKernelArgument.from_payload(argument)
            for argument in arguments
            if isinstance(argument, Mapping)
        )
        if len(parsed_arguments) != len(arguments):
            raise PICAMConfigurationError("direct kernel argument must be a mapping")
        parsed_modules: list[tuple[str, tuple[str, ...]]] = []
        for module, symbols in raw_modules.items():
            if not isinstance(symbols, list) or not symbols:
                raise PICAMConfigurationError(
                    "direct kernel module imports must be non-empty lists"
                )
            parsed_modules.append(
                (str(module), tuple(str(symbol) for symbol in symbols))
            )
        return cls(
            name=name,
            routine=str(payload.get("routine", name)),
            symbol=str(payload.get("symbol", f"pycam_pi_cam_{name}_v1")),
            action_id=int(payload.get("action_id", 0)),
            arguments=parsed_arguments,
            modules=tuple(parsed_modules),
        )

    def operation_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "action_id": self.action_id,
            "arguments": [argument.operation_payload() for argument in self.arguments],
        }


def load_direct_kernels(path: str | Path) -> tuple[DirectKernel, ...]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text())
    if not isinstance(payload, Mapping) or int(payload.get("schema_version", 0)) != 1:
        raise PICAMConfigurationError("direct kernels require schema_version: 1")
    records = payload.get("kernels")
    if not isinstance(records, list) or not records:
        raise PICAMConfigurationError("direct kernel descriptor is empty")
    kernels = tuple(
        DirectKernel.from_payload(record)
        for record in records
        if isinstance(record, Mapping)
    )
    if len(kernels) != len(records):
        raise PICAMConfigurationError("direct kernel record must be a mapping")
    names = [kernel.name for kernel in kernels]
    symbols = [kernel.symbol for kernel in kernels]
    if len(set(names)) != len(names) or len(set(symbols)) != len(symbols):
        raise PICAMConfigurationError("direct kernel names and symbols must be unique")
    return kernels


def _shape_index(field_index: int, axis: int) -> str:
    return f"{field_index} * max_rank + {axis}"


def _chunk_slice(name: str, argument: DirectKernelArgument) -> str:
    fixed = dict(argument.fixed_indices)
    indices = []
    for axis in range(1, argument.rank + 1):
        if axis == argument.chunk_axis:
            indices.append("chunk")
        elif axis in fixed:
            indices.append(str(fixed[axis]))
        else:
            indices.append(":")
    return f"{name}({','.join(indices)})"


def _argument_reference(name: str, argument: DirectKernelArgument) -> str:
    """The actual argument handed to the original routine for this chunk."""

    if argument.pointer:
        return f"{name}_chunk"
    if argument.fortran_type == "logical":
        return f"{name}_value"
    return _chunk_slice(name, argument)


def _chunk_preamble(name: str, argument: DirectKernelArgument) -> tuple[str, ...]:
    """Statements that prepare this argument inside the chunk loop."""

    if argument.pointer:
        return (f"    {name}_chunk => {_chunk_slice(name, argument)}",)
    if argument.fortran_type == "logical":
        return (
            f"    {name}_value = ({_chunk_slice(name, argument)} /= 0_c_int32_t)",
        )
    return ()


def _kernel_function(kernel: DirectKernel) -> list[str]:
    names = [f"field_{index}" for index in range(1, len(kernel.arguments) + 1)]
    maximum_rank = max(argument.rank for argument in kernel.arguments)
    lines = [
        f"integer(c_int) function {kernel.symbol}(action_id, nfields, pointers, &",
        "     ndims, shapes, max_rank, fortran_comm, errmsg, errmsg_len) &",
        f"     bind(C, name='{kernel.symbol}') result(status)",
        "  integer(c_int), value, intent(in) :: action_id, nfields, max_rank",
        "  integer(c_int), value, intent(in) :: fortran_comm, errmsg_len",
        "  type(c_ptr), intent(in) :: pointers(*)",
        "  integer(c_int32_t), intent(in) :: ndims(*)",
        "  integer(c_int64_t), intent(in) :: shapes(*)",
        "  character(kind=c_char), intent(out) :: errmsg(*)",
        "  integer :: chunk, nchunks, index",
    ]
    for name, argument in zip(names, kernel.arguments):
        dimensions = ",".join(":" for _ in range(argument.rank))
        lines.append(f"  {argument.storage_type}, pointer :: {name}({dimensions})")
        if argument.pointer:
            # A POINTER dummy cannot take an array section, so the wrapper keeps
            # a pointer of the dummy's own rank and associates it per chunk.
            slice_dimensions = ",".join(":" for _ in range(argument.rank - 1))
            lines.append(
                f"  {argument.storage_type}, pointer :: {name}_chunk({slice_dimensions})"
            )
        if argument.fortran_type == "logical":
            lines.append(f"  logical :: {name}_value")
    lines.extend(
        [
            "  call pycam_pi_cam_set_fp_environment_v1()",
            "  do index = 1, max(1, int(errmsg_len))",
            "    errmsg(index) = c_null_char",
            "  enddo",
            "  status = 0_c_int",
            f"  if (action_id /= {kernel.action_id} .or. nfields /= {len(kernel.arguments)} &",
            f"       .or. max_rank < {maximum_rank}) then",
            "    status = 1_c_int",
            "    return",
            "  endif",
        ]
    )
    for field_index, argument in enumerate(kernel.arguments):
        lines.extend(
            [
                f"  if (ndims({field_index + 1}) /= {argument.rank}) then",
                f"    status = {10 + field_index}_c_int",
                "    return",
                "  endif",
            ]
        )
    first = kernel.arguments[0]
    lines.append(f"  nchunks = int(shapes({_shape_index(0, first.chunk_axis)}))")
    lines.extend(
        [
            "  if (nchunks < 1) then",
            "    status = 20_c_int",
            "    return",
            "  endif",
        ]
    )
    for field_index, argument in enumerate(kernel.arguments):
        shape_values = [
            f"int(shapes({_shape_index(field_index, axis)}))"
            for axis in range(1, argument.rank + 1)
        ]
        lines.extend(
            [
                f"  if (int(shapes({_shape_index(field_index, argument.chunk_axis)})) "
                "/= nchunks) then",
                f"    status = {30 + field_index}_c_int",
                "    return",
                "  endif",
            ]
        )
        for axis, extent in enumerate(argument.extents, start=1):
            expected = "nchunks" if extent == "chunks" else extent
            lines.extend(
                [
                    f"  if (int(shapes({_shape_index(field_index, axis)})) "
                    f"/= {expected}) then",
                    f"    status = {50 + field_index}_c_int",
                    "    return",
                    "  endif",
                ]
            )
        for axis, index in argument.fixed_indices:
            lines.extend(
                [
                    f"  if (int(shapes({_shape_index(field_index, axis)})) < {index}) then",
                    f"    status = {40 + field_index}_c_int",
                    "    return",
                    "  endif",
                ]
            )
        lines.append(
            f"  call c_f_pointer(pointers({field_index + 1}), "
            f"{names[field_index]}, (/ &"
        )
        for axis, value in enumerate(shape_values):
            ending = ", &" if axis + 1 < len(shape_values) else " /))"
            lines.append(f"       {value}{ending}")
    arguments = [
        _argument_reference(name, argument)
        for name, argument in zip(names, kernel.arguments)
    ]
    lines.extend(
        [
            "  do chunk = 1, nchunks",
        ]
    )
    for name, argument in zip(names, kernel.arguments):
        lines.extend(_chunk_preamble(name, argument))
    lines.append(f"    call {kernel.routine}( &")
    for index, argument in enumerate(arguments):
        ending = ", &" if index + 1 < len(arguments) else ")"
        lines.append(f"         {argument}{ending}")
    lines.extend(("  enddo", f"end function {kernel.symbol}", ""))
    return lines


def generate_direct_kernel_module(
    kernels: tuple[DirectKernel, ...],
    *,
    module_name: str = "pycam_pi_cam_direct_kernels",
) -> str:
    lines = [
        "! Generated by freecam.pi_cam.kernel_codegen; do not edit.",
        f"module {module_name}",
        "  use, intrinsic :: iso_c_binding, only: c_char, c_double, c_int, &",
        "       c_int32_t, c_int64_t, c_null_char, c_ptr, c_f_pointer",
    ]
    modules: dict[str, list[str]] = {}
    for kernel in kernels:
        for module, symbols in kernel.modules:
            current = modules.setdefault(module, [])
            current.extend(symbol for symbol in symbols if symbol not in current)
    for module, symbols in modules.items():
        lines.append(f"  use {module}, only: {', '.join(symbols)}")
    lines.extend(
        (
            "  implicit none",
            "  private",
            "  interface",
            "    subroutine pycam_pi_cam_set_fp_environment_v1() bind(C, &",
            "         name='pycam_pi_cam_set_fp_environment_v1')",
            "    end subroutine pycam_pi_cam_set_fp_environment_v1",
            "  end interface",
        )
    )
    for kernel in kernels:
        lines.append(f"  public :: {kernel.symbol}")
    lines.extend(("contains", ""))
    for kernel in kernels:
        lines.extend(_kernel_function(kernel))
    lines.append(f"end module {module_name}")
    lines.append("")
    return "\n".join(lines)
