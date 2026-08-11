"""Generate chunk-aware direct adapters for StatePool-promotable CAM routines."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable, Mapping

import numpy as np

from .kernel_codegen import DirectKernel, DirectKernelArgument
from .physics_catalog import PICAMPhysicsCatalog, PICAMPhysicsProcess


_DTYPE_NAMES = frozenset(("float64", "int32", "int64"))


def statepool_promotable(process: PICAMPhysicsProcess) -> bool:
    """Return whether the generic chunk adapter can express the full signature."""

    return bool(
        process.level == "process"
        and process.generated_adapter
        # This is the legacy external chunk-wrapper path.  Context-required
        # and private procedures use the newer source-injected device path.
        and process.adapter_status in {"candidate", "validated"}
        and process.arguments
        and all(
            str(argument.get("dtype") or "") in _DTYPE_NAMES
            and not bool(argument.get("optional"))
            and not bool(argument.get("pointer"))
            and not bool(argument.get("allocatable"))
            for argument in process.arguments
        )
    )


def promoted_direct_kernel(
    process: PICAMPhysicsProcess,
    *,
    action_id: int,
) -> DirectKernel:
    """Build one adapter descriptor using logical process-context field names."""

    if not statepool_promotable(process):
        raise ValueError(f"{process.qualified_name} is not StatePool-promotable")
    module, separator, _ = process.qualified_name.partition("::")
    if not separator:
        raise ValueError(f"{process.qualified_name!r} is not a module procedure")
    digest = sha256(process.qualified_name.encode("utf-8")).hexdigest()[:12]
    return DirectKernel(
        name=process.name,
        routine=process.routine,
        symbol=f"freecam_pi_cam_promoted_{process.name}_{digest}_v1",
        action_id=int(action_id),
        modules=((module, (process.routine,)),),
        arguments=tuple(
            DirectKernelArgument(
                field=f"process_context.{process.name}.{argument['name']}",
                dtype=np.dtype(str(argument["dtype"])).str,
                rank=int(argument["rank"]) + 1,
                intent=str(argument.get("intent") or "inout").lower(),
                chunk_axis=int(argument["rank"]) + 1,
            )
            for argument in process.arguments
        ),
    )


def generated_promoted_kernels(
    catalog: PICAMPhysicsCatalog | None = None,
    *,
    first_action_id: int = 1000,
) -> tuple[DirectKernel, ...]:
    """Generate every currently expressible physical-process adapter."""

    selected = tuple(
        process
        for process in (catalog or PICAMPhysicsCatalog.load_default()).physics_processes
        if statepool_promotable(process)
    )
    return tuple(
        promoted_direct_kernel(process, action_id=first_action_id + index)
        for index, process in enumerate(selected)
    )


def direct_kernel_payload(kernels: Iterable[DirectKernel]) -> Mapping[str, object]:
    """Render descriptors back to the reviewed YAML schema."""

    return {
        "schema_version": 1,
        "kernels": [
            {
                "name": kernel.name,
                "routine": kernel.routine,
                "symbol": kernel.symbol,
                "action_id": kernel.action_id,
                "modules": {
                    module: list(symbols) for module, symbols in kernel.modules
                },
                "arguments": [
                    dict(
                        {
                            "field": argument.field,
                            "dtype": np.dtype(argument.dtype).name,
                            "rank": argument.rank,
                            "intent": argument.intent,
                            "chunk_axis": argument.chunk_axis,
                        },
                        **(
                            {"extents": list(argument.extents)}
                            if argument.extents
                            else {}
                        ),
                        **(
                            {
                                "fixed_indices": {
                                    axis: index
                                    for axis, index in argument.fixed_indices
                                }
                            }
                            if argument.fixed_indices
                            else {}
                        ),
                    )
                    for argument in kernel.arguments
                ],
            }
            for kernel in kernels
        ],
    }
