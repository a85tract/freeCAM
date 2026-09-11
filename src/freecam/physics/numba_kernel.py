"""A kernel written in Python, compiled with Numba, bound at a hook: Fortran calls it directly.

A hook whose table entry has a ``model`` block hands a bound answerer the named
arguments as ``float64`` arrays with the callee's own extents.  For a TorchScript
model that answerer is FTorch; for a Python function it is a Numba ``cfunc`` of
the hook's plugin interface, generated here from the contract, which unpacks the
pointer and extent tables into Fortran-ordered arrays and calls the user's
compiled function.  No interpreter runs during the call, no fiber pauses, no
frame is copied: the step crosses the Python/Fortran boundary once, as with
nothing replaced.

The user writes the kernel over NumPy arrays and scalars, in the order of the
``model`` block, inputs first, then outputs, writing the outputs in place::

    def fice_kernel(t, fice, fsnow):          # t[ncol, pver] in, fice/fsnow[ncol, pver] out
        ...

Scalar inputs arrive as Python floats; every array is ``float64`` and indexed
``[column, level]`` as the Fortran callee indexes it.  The function must be
compilable by ``numba.njit`` in nopython mode.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Callable

from .errors import PhysicsError
from .native_model import NativePlugin

REPO = Path(__file__).resolve().parents[3]

#: the C signature of pycam_hooks' plugin_interface
PLUGIN_SIGNATURE = "int32(int32, CPointer(voidptr), CPointer(int64), int32, CPointer(voidptr), CPointer(int64))"


def _model_arguments(hook_name: str):
    """The hook's model block as contract arguments: (inputs, outputs)."""

    from freecam.physics.spec import load_function_spec
    from freecam.pi_cam.hooks import load_hooks

    hook = load_hooks().hook(hook_name)
    if not hook.takes_model:
        raise PhysicsError(
            f"hook {hook_name!r} has no model block in native/pi_cam/hooks.yaml: the image does not know "
            f"how to hand its arguments to a plugin")
    spec = load_function_spec(str(REPO / hook.contract))
    arguments = {item.name: item for item in spec.arguments}
    inputs = [arguments[name] for name in hook.model_inputs]
    outputs = [arguments[name] for name in hook.model_outputs]
    return hook, inputs, outputs


def adapter_source(hook_name: str, inputs, outputs, *, kernel_name: str = "kernel") -> str:
    """The Numba cfunc that unpacks the hook's tables and calls ``kernel``, as source."""

    lines = ["def adapter(n_in, in_ptrs, in_shapes, n_out, out_ptrs, out_shapes):",
             f"    if n_in != {len(inputs)} or n_out != {len(outputs)}:",
             "        return 1"]
    names: list[str] = []
    for slot, item in enumerate(inputs):
        base = 3 * slot
        if item.rank == 0:
            lines.append(f"    a{slot} = carray(in_ptrs[{slot}], (1,), float64)[0]")
        else:
            dims = ", ".join(f"in_shapes[{base + axis}]" for axis in range(item.rank))
            lines.append(f"    a{slot} = farray(in_ptrs[{slot}], ({dims},), float64)")
        names.append(f"a{slot}")
    for slot, item in enumerate(outputs):
        base = 3 * slot
        if item.rank == 0:
            lines.append(f"    o{slot} = carray(out_ptrs[{slot}], (1,), float64)")
        else:
            dims = ", ".join(f"out_shapes[{base + axis}]" for axis in range(item.rank))
            lines.append(f"    o{slot} = farray(out_ptrs[{slot}], ({dims},), float64)")
        names.append(f"o{slot}")
    lines.append(f"    {kernel_name}({', '.join(names)})")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def compile_kernel(hook_name: str, function: Callable[..., Any], *, shadow: bool = False) -> NativePlugin:
    """Compile ``function`` for hook ``hook_name`` and wrap it as a :class:`NativePlugin` for a kernel slot.

    ``function`` takes the model block's inputs then outputs, as described in the
    module docstring; it is compiled with ``numba.njit`` unless it already is a
    Numba dispatcher.  Compilation happens in this process, once per rank.
    """

    try:
        import numba
        from numba import carray, farray, float64, int32, int64, types  # noqa: F401
    except ImportError as error:                       # pragma: no cover - environment
        raise PhysicsError("a Numba kernel needs the numba package in this environment") from error

    hook, inputs, outputs = _model_arguments(hook_name)
    kernel = function if isinstance(function, numba.core.registry.CPUDispatcher) else numba.njit(function)
    source = adapter_source(hook_name, inputs, outputs)
    namespace: dict[str, Any] = {"carray": carray, "farray": farray, "float64": float64, "kernel": kernel}
    exec(compile(source, f"<plugin adapter for {hook_name}>", "exec"), namespace)
    signature = types.int32(types.int32, types.CPointer(types.voidptr), types.CPointer(types.int64),
                            types.int32, types.CPointer(types.voidptr), types.CPointer(types.int64))
    try:
        adapter = numba.cfunc(signature, nopython=True, cache=False)(namespace["adapter"])
    except Exception as error:
        raise PhysicsError(
            f"the kernel for hook {hook_name!r} did not compile under Numba: {error}") from error
    label = f"{getattr(function, '__module__', '?')}:{getattr(function, '__name__', getattr(function, 'py_func', function).__class__.__name__)}"
    return NativePlugin(adapter, label=label, kernel=hook_name, shadow=shadow,
                        inputs=[item.name for item in inputs], outputs=[item.name for item in outputs])


def call_plugin_from_python(plugin: NativePlugin, inputs: list, outputs: list) -> int:
    """Call a compiled plugin the way the hook does, from Python: for tests and offline checks.

    ``inputs`` and ``outputs`` are NumPy ``float64`` arrays (Fortran order) or
    floats in the model block's order; the outputs are written in place.
    """

    import numpy as np

    def table(items):
        arrays = [np.asarray(item, dtype=np.float64).reshape(1) if np.ndim(item) == 0 else np.asfortranarray(item, dtype=np.float64)
                  for item in items]
        pointers = (ctypes.c_void_p * len(arrays))(*[a.ctypes.data for a in arrays])
        shapes = (ctypes.c_int64 * (3 * len(arrays)))()
        for slot, a in enumerate(arrays):
            for axis, extent in enumerate(a.shape[:3]):
                shapes[3 * slot + axis] = extent
        return arrays, pointers, shapes

    in_arrays, in_ptrs, in_shapes = table(inputs)
    out_arrays, out_ptrs, out_shapes = table(outputs)
    prototype = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
                                 ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64))
    status = int(prototype(plugin.address)(len(in_arrays), in_ptrs, in_shapes, len(out_arrays), out_ptrs, out_shapes))
    for target, written in zip(outputs, out_arrays):
        if np.ndim(target):
            np.copyto(target, written)
    return status


__all__ = ["compile_kernel", "adapter_source", "call_plugin_from_python", "PLUGIN_SIGNATURE"]
