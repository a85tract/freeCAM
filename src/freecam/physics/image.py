"""Module storage of a loaded Fortran image, read and written by symbol.

Both the standalone harness and the rank-side snapshot tool see a CAM image
the same way: a ctypes handle and a table of module-variable symbols from the
function spec.  Reading every entry into plain Python values (with exact hex
for reals and a content hash) is what lets a snapshot taken in an initialized
model be compared bit for bit with what the standalone image holds after its
own initialization.
"""

from __future__ import annotations

import ctypes
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np

from .errors import PhysicsError
from .spec import ModuleStateEntry

_CTYPES: dict[str, Any] = {
    "float64": ctypes.c_double,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "S16": ctypes.c_char * 16,
}


def module_view(library: ctypes.CDLL, symbol: str, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """A live, writable NumPy view of one module variable in the image."""

    ctype = _CTYPES.get(dtype)
    if ctype is None:
        raise PhysicsError(f"module state dtype {dtype!r} is not supported")
    count = 1
    for extent in shape:
        count *= int(extent)
    try:
        storage = (ctype * count).in_dll(library, symbol)
    except ValueError as error:
        raise PhysicsError(f"image has no symbol {symbol!r}") from error
    view = np.frombuffer(storage, dtype=np.dtype("S16") if dtype == "S16" else np.dtype(dtype), count=count)
    return view.reshape(shape, order="F") if shape else view.reshape(())


def _values(view: np.ndarray, dtype: str) -> list[Any]:
    flat = view.reshape(-1, order="F")
    if dtype == "S16":
        return [item.decode("ascii", errors="replace") for item in flat.tolist()]
    if dtype == "float64":
        return [float(item) for item in flat.tolist()]
    return [int(item) for item in flat.tolist()]


def read_entry(library: ctypes.CDLL, entry: ModuleStateEntry) -> dict[str, Any]:
    """One entry's current contents, exactly: values, hex for reals, a hash."""

    view = module_view(library, entry.symbol, entry.dtype, entry.shape)
    raw = np.ascontiguousarray(view).tobytes()
    record: dict[str, Any] = {
        "symbol": entry.symbol,
        "dtype": entry.dtype,
        "shape": list(entry.shape),
        "write": entry.write,
        "values": _values(view, entry.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if entry.dtype == "float64":
        record["hex"] = [float(item).hex() for item in record["values"]]
    return record


def read_module_state(library: ctypes.CDLL, entries: Iterable[ModuleStateEntry]) -> dict[str, dict[str, Any]]:
    return {entry.symbol: read_entry(library, entry) for entry in entries}


def state_digest(state: Mapping[str, Mapping[str, Any]]) -> str:
    """One hash over every entry's hash, in symbol order."""

    digest = hashlib.sha256()
    for symbol in sorted(state):
        digest.update(symbol.encode())
        digest.update(str(state[symbol]["sha256"]).encode())
    return digest.hexdigest()


__all__ = ["module_view", "read_entry", "read_module_state", "state_digest"]


# -- the standalone image itself ---------------------------------------------

import json
import os
from pathlib import Path
import sys

from freecam.core.fortran_adapter import PointerTableAdapter

from .spec import FunctionSpec, load_function_spec

EXPECTED_MXCSR = 0x9FC0


class ModuleStateMismatch(PhysicsError):
    """The image's module state does not match the model snapshot."""


class MathLibraryNotPreloaded(PhysicsError):
    """The process did not start with the image's Intel math library preloaded."""


def preloaded_libraries() -> tuple[str, ...]:
    return tuple(item for item in os.environ.get("LD_PRELOAD", "").replace(":", " ").split() if item)


def require_math_library(math_library: Path) -> None:
    """The image's exp/log/pow must bind to Intel's libimf, as the model's do.

    A Python process already holds glibc's libm in its global symbol scope,
    which the dynamic loader searches before a dlopen'd image's own
    dependencies; without the preload the routine's transcendental calls
    bind to glibc and results differ in the last bits.  Only a preload puts
    libimf ahead of libm, so the process must have been started with it.
    """

    wanted = math_library.resolve()
    for item in preloaded_libraries():
        try:
            if Path(item).resolve() == wanted:
                return
        except OSError:
            continue
    raise MathLibraryNotPreloaded(
        f"start this process with LD_PRELOAD={wanted} (the standalone harness "
        "does this for its worker); without it the routine's math binds to glibc"
    )


def reexec_with_math_library(manifest: str | Path) -> None:
    """Restart the current tool under the manifest's libimf if it is missing."""

    math_library = Path(str(json.loads(Path(manifest).read_text())["intel_math_library"]))
    try:
        require_math_library(math_library)
    except MathLibraryNotPreloaded:
        env = dict(os.environ)
        env["LD_PRELOAD"] = str(math_library)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _first_difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any] | None:
    a = np.ascontiguousarray(reference).reshape(-1)
    b = np.ascontiguousarray(candidate).reshape(-1)
    if a.dtype != b.dtype or a.shape != b.shape:
        return {"reason": f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"}
    bits = a.dtype.itemsize
    view = np.dtype(f"u{bits}") if a.dtype.kind in "fiu" else a.dtype
    unequal = np.nonzero(a.view(view) != b.view(view))[0] if a.dtype.kind in "fiu" else np.nonzero(a != b)[0]
    if unequal.size == 0:
        return None
    index = int(unequal[0])
    position = np.unravel_index(index, np.asarray(reference).shape, order="F") if np.asarray(reference).ndim else ()
    record: dict[str, Any] = {"flat_index": index, "index": [int(item) for item in position], "count": int(unequal.size)}
    for label, value in (("reference", a[index]), ("candidate", b[index])):
        record[label] = value.decode() if isinstance(value, bytes) else (float(value) if a.dtype.kind == "f" else int(value))
        if a.dtype.kind == "f":
            record[f"{label}_hex"] = float(value).hex()
    return record


def bitwise_equal(reference: np.ndarray, candidate: np.ndarray) -> bool:
    return _first_difference(reference, candidate) is None


class StandaloneImage:
    """One physics routine, loaded from its standalone image in this process.

    Loading checks the manifest's hash and the fixed load address, installs
    the production floating-point environment, and nothing else; the routine
    is callable only after :meth:`initialize` has set and verified every
    module variable the spec lists against a snapshot from a real model.
    """

    def __init__(self, manifest: str | Path, *, spec: FunctionSpec | None = None) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.spec = spec or load_function_spec(str(self.manifest["function"]))
        library = Path(str(self.manifest["library"]))
        if not library.is_file():
            raise PhysicsError(f"standalone image not found: {library}")
        digest = hashlib.sha256(library.read_bytes()).hexdigest()
        if digest != self.manifest["library_sha256"]:
            raise PhysicsError(f"{library} hash {digest[:12]} differs from its manifest")
        self.library_path = library
        self.math_library = Path(str(self.manifest["intel_math_library"]))
        require_math_library(self.math_library)
        self.library = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL | os.RTLD_NOW)
        base = int(str(self.manifest["load_start"]), 16)
        mapped = [line for line in Path("/proc/self/maps").read_text().splitlines() if str(library) in line]
        loaded = min(int(line.split("-")[0], 16) for line in mapped)
        if loaded != base:
            raise PhysicsError(f"image loaded at {loaded:#x}, manifest says {base:#x}")
        self.library.pycam_pi_cam_get_mxcsr_v1.restype = ctypes.c_uint32
        self.library.pycam_pi_cam_set_fp_environment_v1()
        if int(self.library.pycam_pi_cam_get_mxcsr_v1()) != EXPECTED_MXCSR:
            raise PhysicsError("could not install the production floating-point environment")
        self._adapter = PointerTableAdapter(
            self.library, {"call": self.manifest["wrapper"]["operation"]}, library_name=str(library)
        )
        self._parameter_values: dict[str, Any] = {}
        self._baseline: dict[str, Any] = {}
        self.initialized = False

    # -- module state ----------------------------------------------------

    def _entry(self, symbol: str) -> ModuleStateEntry:
        for entry in self.spec.module_state:
            if entry.symbol == symbol:
                return entry
        raise PhysicsError(f"{symbol} is not in the spec's module state")

    def _write(self, entry: ModuleStateEntry, values: Any) -> None:
        view = module_view(self.library, entry.symbol, entry.dtype, entry.shape)
        flat = view.reshape(-1, order="F") if entry.shape else view.reshape(1)
        items = list(values) if isinstance(values, (list, tuple)) else [values]
        if len(items) != flat.size:
            raise PhysicsError(f"{entry.symbol} takes {flat.size} values, got {len(items)}")
        if entry.dtype == "S16":
            flat[...] = np.array([str(item).ljust(16).encode("ascii") for item in items], dtype="S16")
        else:
            flat[...] = np.asarray(items, dtype=np.dtype(entry.dtype))

    def initialize(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        """Run the initializers, write the snapshot's values, verify everything."""

        entries = snapshot.get("entries", snapshot)
        missing = [entry.symbol for entry in self.spec.module_state if entry.symbol not in entries]
        if missing:
            raise PhysicsError("snapshot lacks module state for: " + ", ".join(missing))
        for name in self.spec.initializers:
            getattr(self.library, name)()
        for entry in self.spec.module_state:
            recorded = entries[entry.symbol]
            if entry.write == "declared":
                self._write(entry, entry.value if isinstance(entry.value, list) or not entry.shape else [entry.value])
            elif entry.write in ("snapshot", "parameter"):
                self._write(entry, recorded["values"])
            if entry.expected is not None:
                expected = entry.expected if isinstance(entry.expected, list) else [entry.expected]
                actual = list(recorded["values"])[: len(expected)]
                if entry.dtype == "S16":
                    actual = [str(item).strip() for item in actual]
                if actual != expected:
                    raise ModuleStateMismatch(
                        f"{entry.symbol}: the model holds {actual!r}, the spec expects {expected!r}"
                    )
        verification = self.verify(entries)
        for name, parameter in self.spec.parameters.items():
            values = {symbol: entries[symbol]["values"][0] for symbol in parameter.symbols}
            if len(set(values.values())) != 1:
                raise ModuleStateMismatch(f"parameter {name} copies disagree in the snapshot: {values}")
            value = next(iter(values.values()))
            if value != parameter.default:
                raise ModuleStateMismatch(
                    f"parameter {name}: the model holds {value!r}, the spec default is {parameter.default!r}"
                )
            self._baseline[name] = value
        self._parameter_values = dict(self._baseline)
        self.initialized = True
        return verification

    def verify(self, entries: Mapping[str, Any]) -> dict[str, Any]:
        current = read_module_state(self.library, self.spec.module_state)
        mismatched = {
            symbol: {"image": current[symbol]["values"], "snapshot": entries[symbol]["values"]}
            for symbol in current
            if current[symbol]["sha256"] != entries[symbol]["sha256"]
        }
        if mismatched:
            raise ModuleStateMismatch(
                "module state differs from the model snapshot: "
                + "; ".join(f"{symbol} image={item['image']} model={item['snapshot']}" for symbol, item in mismatched.items())
            )
        return {"entries": len(current), "digest": state_digest(current), "all_equal": True}

    # -- parameters --------------------------------------------------------

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._parameter_values)

    def set_parameters(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Write every storage copy of each parameter, verifying before and after."""

        self._require_initialized()
        written: dict[str, dict[str, Any]] = {}
        for name, value in values.items():
            parameter = self.spec.parameters.get(name)
            if parameter is None:
                raise PhysicsError(f"{self.spec.function} has no parameter {name!r}")
            if parameter.values is not None and value not in parameter.values:
                raise PhysicsError(f"parameter {name} takes one of {list(parameter.values)}, got {value!r}")
            coerced = int(value) if parameter.dtype in ("int32", "int64") else float(value)
            if parameter.dtype in ("int32", "int64") and coerced != value:
                raise PhysicsError(f"parameter {name} is an integer, got {value!r}")
            for symbol in parameter.symbols:
                entry = self._entry(symbol)
                view = module_view(self.library, symbol, entry.dtype, entry.shape)
                before = view[()].item() if not entry.shape else view.reshape(-1)[0].item()
                if before != self._parameter_values[name]:
                    raise ModuleStateMismatch(
                        f"{symbol} holds {before!r} but {name} was last set to {self._parameter_values[name]!r}"
                    )
                view[...] = coerced
                after = view[()].item() if not entry.shape else view.reshape(-1)[0].item()
                if after != coerced:
                    raise ModuleStateMismatch(f"{symbol} reads back {after!r} after writing {coerced!r}")
                written[symbol] = {"previous": before, "value": after}
            self._parameter_values[name] = coerced
        return written

    def restore_parameters(self) -> dict[str, dict[str, Any]]:
        changed = {name: value for name, value in self._baseline.items() if self._parameter_values[name] != value}
        return self.set_parameters(changed)

    # -- calling -------------------------------------------------------------

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise PhysicsError(f"{self.spec.function} image is not initialized")

    def empty_pool(self, nchunks: int = 1) -> dict[str, np.ndarray]:
        """Zeroed native arrays for every argument, chunk axis last."""

        from .column import empty_pool

        return empty_pool(self.spec, nchunks)

    def call(self, pool: Mapping[str, np.ndarray]) -> None:
        """Run the routine on every chunk of ``pool`` (one wrapper call)."""

        self._require_initialized()
        self._adapter.call("call", pool, fcomm=0)


__all__ += ["MathLibraryNotPreloaded", "ModuleStateMismatch", "StandaloneImage", "bitwise_equal", "reexec_with_math_library", "require_math_library"]
