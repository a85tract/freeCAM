"""A model the image runs itself: a TorchScript file bound at a hooked kernel.

Put in a stage's kernel slot, a :class:`NativeModel` is not called from Python
at all.  The stage binds the file at the kernel's hook (``pycam_hooks_bind_model_v1``)
and runs the original Fortran stage whole; the hook, reached inside the compiled
routine, hands the kernel's arrays to the model through FTorch and writes the
answer back, so a step has one Python/Fortran crossing whatever is replaced.
The Python replacements -- a callable answering the frame at a pause -- remain
for validation, frame capture and quick experiments.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

from .errors import PhysicsError


class NativeModel:
    """A TorchScript model by path, for a kernel slot; the image loads it, not Python."""

    #: the segment runner must never see this in a slot: it is not a frame callable
    takes_frame = False

    def __init__(self, path: str | Path, *, shadow: bool = False) -> None:
        #: run the model on every call but let the original answer: bit-for-bit, cost measured
        self.shadow = bool(shadow)
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise PhysicsError(f"native model {self.path} is not a file")
        if not self.is_torchscript(self.path):
            raise PhysicsError(f"native model {self.path} is not a TorchScript archive")
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @staticmethod
    def is_torchscript(path: str | Path) -> bool:
        """Whether ``path`` is a TorchScript archive (a zip carrying the module's code and constants)."""

        path = Path(path)
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        return any(name.endswith("constants.pkl") for name in names) and any("/code/" in name for name in names)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise PhysicsError(
            f"{self.path.name} is a native model: the image answers the kernel with it; "
            f"it is not called from Python")

    def describe(self) -> dict[str, Any]:
        return {"file": self.path.name, "sha256": self.sha256, "binding": "torchscript", "shadow": self.shadow}

    def __repr__(self) -> str:
        return f"NativeModel({str(self.path)!r}{', shadow=True' if self.shadow else ''})"


#: the compiled code of every plugin made in this process, by identity: kept alive for the
#: life of the process, and how a plugin unpickled from the process registry finds its address
_PLUGIN_CODE: dict[str, Any] = {}


class NativePlugin:
    """A compiled kernel (a Numba cfunc of the hook's plugin interface) for a kernel slot.

    Built by :func:`freecam.physics.numba_kernel.compile_kernel`; the stage binds its
    address at the kernel's hook (``pycam_hooks_bind_plugin_v1``) and the image calls it
    directly on every call, inside Fortran, with the model block's arrays.

    ``identity`` names the compiled function the same way on every rank (the hook, the
    source file, the function, the mode): a stage is cloudpickled into each rank's process
    registry and the payload must hash the same on all of them (7402200), so the pickle
    carries the identity and never the address, which is this process's own.
    """

    takes_frame = False

    def __init__(self, adapter: Any, *, label: str, kernel: str, identity: str, shadow: bool = False,
                 inputs: list[str] | None = None, outputs: list[str] | None = None) -> None:
        self._adapter = adapter
        self.address = int(adapter.address)
        self.identity = str(identity)
        _PLUGIN_CODE[self.identity] = adapter
        self.label = str(label)
        self.kernel = str(kernel)
        self.shadow = bool(shadow)
        self.inputs = list(inputs or ())
        self.outputs = list(outputs or ())

    @property
    def key(self) -> str:
        """What identifies this binding to the stage: the code's address and the mode."""

        return f"numba:{self.address:#x}{':shadow' if self.shadow else ''}"

    def __getstate__(self) -> dict[str, Any]:
        # the compiled code is not picklable and the address is this process's: the pickle
        # carries the identity, identical on every rank, and the code is found again here
        state = dict(self.__dict__)
        state.pop("_adapter", None)
        state.pop("address", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        adapter = _PLUGIN_CODE.get(self.identity)
        if adapter is None:
            raise PhysicsError(
                f"{self.label}: a compiled plugin cannot cross processes by pickle; its code lives in "
                f"the process that compiled it (compile it on every rank with compile_kernel)")
        self._adapter = adapter
        self.address = int(adapter.address)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise PhysicsError(
            f"{self.label} is a compiled plugin: the image calls it at the hook; it is not called from Python")

    def describe(self) -> dict[str, Any]:
        return {"function": self.label, "binding": "numba", "kernel": self.kernel, "shadow": self.shadow,
                "inputs": len(self.inputs), "outputs": len(self.outputs)}

    def __repr__(self) -> str:
        return f"NativePlugin({self.label!r}{', shadow=True' if self.shadow else ''})"


#: the C callbacks made in this process, by identity: alive as long as the image may call them
_PYTHON_CALLBACKS: dict[str, Any] = {}
#: the tracebacks of a callback's failed calls, by identity (the hook stops the rank on the first)
_PYTHON_FAILURES: dict[str, list[str]] = {}


def model_block(kernel: str):
    """A hooked kernel's model block as contract arguments: (hook, spec, inputs, outputs)."""

    from freecam.pi_cam.hooks import load_hooks

    from .spec import load_function_spec

    hook = load_hooks().hook(kernel)
    if not hook.takes_model:
        raise PhysicsError(
            f"hook {kernel!r} has no model block in native/pi_cam/hooks.yaml: the image does not know "
            f"how to hand its arguments to a function")
    spec = load_function_spec(str(Path(__file__).resolve().parents[3] / hook.contract))
    arguments = {item.name: item for item in spec.arguments}
    return hook, spec, [arguments[name] for name in hook.model_inputs], [arguments[name] for name in hook.model_outputs]


def packed_layout(hook: Any, spec: Any, outputs: list) -> list[dict[str, Any]]:
    """Where each output sits in the hook's packed tensor: offset, width, the extents after the
    first (the last one full) and the 1-based subset indices when the model returns a subset.

    The hook flattens each output per column with the last axis fastest, as ``reshape(n, -1)``
    does, and lays the outputs side by side in the model block's order.
    """

    layout, offset = [], 0
    for item in outputs:
        dims = [int(spec.dimensions[axis]) if str(axis) in spec.dimensions else int(axis) for axis in item.native_shape[1:]]
        subset = hook.model_subset(item.name)
        packed = list(dims)
        if subset is not None:
            packed[-1] = len(subset)
        width = 1
        for extent in packed:
            width *= extent
        layout.append({"name": item.name, "offset": offset, "width": width, "dims": dims, "subset": subset, "rank": item.rank})
        offset += width
    return layout


def _python_identity(kernel: str, function: Any, shadow: bool) -> str:
    """The same string on every rank for the same function: the hook, the function's name and its pickled code."""

    import cloudpickle

    try:
        digest = hashlib.sha256(cloudpickle.dumps(function)).hexdigest()[:16]
    except Exception:    # noqa: BLE001 -- an unpicklable callable is refused when the stage is installed
        digest = "unpicklable"
    return f"{kernel}:{getattr(function, '__qualname__', type(function).__name__)}:{digest}:{'shadow' if shadow else 'live'}"


class PythonPlugin:
    """A Python callable the image calls at a kernel's hook, inside Fortran, through a C callback.

    Put in a kernel slot, the stage binds a ``ctypes`` callback of the hook's plugin
    interface at the kernel's hook and runs the original stage whole.  On every call the
    hook hands the model block's arrays to the callback; it re-enters the interpreter,
    builds NumPy views of them (Fortran order, every column the chunk holds, the contract's
    extents; a scalar as a float) and calls ``function(batch)`` with them by name.  The
    function returns the outputs by name: an array of the output's shape (a rank-1 output
    as one value a column), and for a subset output either the whole array or the compact
    one over the subset's indices; an output it leaves out stays zero.  A failure inside
    the function is written to stderr and stops the rank.

    The interpreter runs during the call, so the function's own speed is the cost; the
    callback itself is a few tens of microseconds.  For a network, prefer a TorchScript
    :class:`NativeModel`, which the image runs without Python.
    """

    takes_frame = False

    def __init__(self, function: Any, kernel: str, *, shadow: bool = False, label: str | None = None) -> None:
        if not callable(function):
            raise PhysicsError(f"a hook callback for {kernel!r} needs a callable, not {type(function).__name__}")
        self.function = function
        self.kernel = str(kernel)
        #: run the function on every call but let the original answer: bit-for-bit, cost measured
        self.shadow = bool(shadow)
        self.label = str(label or getattr(function, "__qualname__", type(function).__name__))
        self.identity = _python_identity(self.kernel, function, self.shadow)
        self._callback: Any = None
        self._address = 0

    @property
    def key(self) -> str:
        """What identifies this binding to the stage: the function and the mode."""

        return f"python:{self.identity}"

    @property
    def address(self) -> int:
        """The C address of this process's callback, made on first use."""

        if not self._address:
            self._callback, self._address = self._build()
        return self._address

    @property
    def failures(self) -> list[str]:
        """Tracebacks of the calls that failed in this process."""

        return list(_PYTHON_FAILURES.get(self.identity, ()))

    def __getstate__(self) -> dict[str, Any]:
        # the callback and its address are this process's; every rank makes its own
        state = dict(self.__dict__)
        state["_callback"] = None
        state["_address"] = 0
        return state

    def __call__(self, batch: dict[str, Any]) -> Any:
        """The function itself, for a test from Python; the image calls it at the hook."""

        return self.function(batch)

    def describe(self) -> dict[str, Any]:
        return {"function": self.label, "binding": "python", "kernel": self.kernel, "shadow": self.shadow,
                "identity": self.identity}

    def __repr__(self) -> str:
        return f"PythonPlugin({self.label!r}, {self.kernel!r}{', shadow=True' if self.shadow else ''})"

    def _build(self) -> tuple[Any, int]:
        import ctypes
        import sys
        import traceback

        import numpy as np

        hook, spec, inputs, outputs = model_block(self.kernel)
        layout = packed_layout(hook, spec, outputs) if hook.model_packed else None
        function = self.function
        failures = _PYTHON_FAILURES.setdefault(self.identity, [])
        double = ctypes.POINTER(ctypes.c_double)

        def view(pointer: int, shape: list[int]) -> np.ndarray:
            # a Fortran-ordered view of the hook's array: C order over the reversed extents, transposed
            if not shape:
                return np.ctypeslib.as_array(ctypes.cast(pointer, double), shape=(1,))
            return np.ctypeslib.as_array(ctypes.cast(pointer, double), shape=tuple(reversed(shape))).T

        def write(target: np.ndarray, value: Any, rank: int) -> None:
            if rank == 0:
                target[0] = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
            else:
                target[...] = np.asarray(value, dtype=np.float64).reshape(target.shape)

        def write_packed(packed: np.ndarray, entry: dict[str, Any], value: Any) -> None:
            columns = packed.shape[0]
            array = np.asarray(value, dtype=np.float64)
            if entry["rank"] == 1:
                array = array.reshape(columns, 1)
            elif array.shape[0] != columns:
                raise PhysicsError(f"{entry['name']}: {array.shape[0]} columns returned, the hook holds {columns}")
            subset = entry["subset"]
            if subset is not None:
                if array.shape[-1] == entry["dims"][-1]:
                    array = array[..., [index - 1 for index in subset]]
                elif array.shape[-1] != len(subset):
                    raise PhysicsError(
                        f"{entry['name']}: the last axis is {array.shape[-1]}, neither the contract's "
                        f"{entry['dims'][-1]} nor the subset's {len(subset)}")
            packed[:, entry["offset"]:entry["offset"] + entry["width"]] = array.reshape(columns, -1)

        def adapter(n_in: int, in_ptrs: Any, in_shapes: Any, n_out: int, out_ptrs: Any, out_shapes: Any) -> int:
            try:
                if n_in != len(inputs) or n_out != (1 if layout is not None else len(outputs)):
                    raise PhysicsError(f"hook {hook.kernel}: {n_in} inputs and {n_out} outputs offered, "
                                       f"{len(inputs)} and {1 if layout is not None else len(outputs)} expected")
                batch: dict[str, Any] = {}
                for slot, item in enumerate(inputs):
                    shape = [int(in_shapes[3 * slot + axis]) for axis in range(item.rank)]
                    array = view(in_ptrs[slot], shape)
                    batch[item.name] = float(array[0]) if item.rank == 0 else array
                result = function(batch) or {}
                if not isinstance(result, dict):
                    raise PhysicsError(f"hook {hook.kernel}: the function returned a {type(result).__name__}, not outputs by name")
                unknown = sorted(set(result) - {item.name for item in outputs})
                if unknown:
                    raise PhysicsError(f"hook {hook.kernel}: {unknown} are not outputs of the model block")
                if layout is not None:
                    packed = view(out_ptrs[0], [int(out_shapes[0]), int(out_shapes[1])])
                    for entry in layout:
                        if entry["name"] in result:
                            write_packed(packed, entry, result[entry["name"]])
                else:
                    for slot, item in enumerate(outputs):
                        if item.name in result:
                            shape = [int(out_shapes[3 * slot + axis]) for axis in range(item.rank)]
                            write(view(out_ptrs[slot], shape), result[item.name], item.rank)
                return 0
            except Exception:    # noqa: BLE001 -- nothing may propagate through the C boundary
                failures.append(traceback.format_exc())
                sys.stderr.write(f"pycam hook callback {self.label} at {self.kernel} failed:\n{failures[-1]}")
                sys.stderr.flush()
                return 1

        prototype = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
                                     ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64))
        callback = prototype(adapter)
        _PYTHON_CALLBACKS[self.identity] = callback
        return callback, int(ctypes.cast(callback, ctypes.c_void_p).value)


#: the name a notebook reads: a Python function stands at the kernel's hook
HookCallback = PythonPlugin

__all__ = ["HookCallback", "NativeModel", "NativePlugin", "PythonPlugin", "model_block", "packed_layout"]
