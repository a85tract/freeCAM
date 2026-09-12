"""The radiation process as one replaceable unit: capture its inputs and outputs, or answer it.

``radiation_tend`` is the most expensive physics process of the admitted
configuration and the classic target of a learned emulator: a column function
from temperature, humidity, pressure, clouds, aerosols, gases, geometry and
surface properties to heating rates and surface fluxes.  The
:class:`~freecam.physics.radiation.Radiation` stage, which transcribes the
driver statement for statement, offers the whole computing branch of a
radiative step as a *process slot*:

- :class:`RadiationProcessCapture` records what the driver had in hand before
  the branch and what the branch produced, per chunk and radiative step, and
  saves them per rank: the dataset a process-level emulator is trained on.
- :class:`RadiationReplay` answers the branch with the outputs a capture
  recorded for the same step and chunk: the gate that proves the write-back
  path is exact (a replay of a bit-for-bit capture must itself be bit-for-bit).
- :class:`RadiationProcessModel` answers the branch with a user function --
  the trained emulator -- taking the inputs by name and returning the outputs.

What the branch consumes and produces is fixed by the driver, not by any
model: the names below are the contract.  The stage keeps everything around
the branch as the original has it: the quiet-step conversions, ``radheat``
building the tendency and the net flux, the copy into ``cam_out%netsw``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .errors import PhysicsError

#: scalar inputs the stage records with every call
SCALAR_INPUTS = ("nstep", "lchnk", "ncol", "dt", "calday", "dosw", "dolw")
#: array inputs available before the computing branch (chunk arrays, F-ordered as CAM holds them)
ARRAY_INPUTS = (
    "coszrs", "clat", "clon",
    "state_t", "state_pmid", "state_pint", "state_pdel", "state_lnpint", "state_lnpmid", "state_q",
    "cld", "cldfsnow", "dei", "mu", "lambdac", "iciwp", "iclwp", "des", "icswp", "dgnumwet", "qaerwat",
    "cam_in_lwup", "cam_in_asdir", "cam_in_asdif", "cam_in_aldir", "cam_in_aldif",
)
#: the RRTMG state the driver builds from the state and the radiative constituents (gas profiles)
RSTATE_INPUTS = (
    "rstate_h2ovmr", "rstate_o3vmr", "rstate_co2vmr", "rstate_ch4vmr", "rstate_o2vmr", "rstate_n2ovmr",
    "rstate_cfc11vmr", "rstate_cfc12vmr", "rstate_cfc22vmr", "rstate_ccl4vmr",
    "rstate_pmidmb", "rstate_pintmb", "rstate_tlay", "rstate_tlev",
)
#: what the computing branch leaves for the rest of the driver and the model: the heating rates
#: (energy units, before the driver scales them for storage) and the surface and top fluxes
OUTPUTS = ("qrs", "qrl", "fsns", "fsnt", "flns", "flnt", "fsds", "sols", "soll", "solsd", "solld", "flwds")

#: The table the Fortran slot (pycam_rad_process, asked by the radiation runner at the top of the
#: radiative branch) hands a compiled plugin: name and rank, in order.  A test pins this to the
#: module's own list.  Scalars travel as one-element doubles; the constituent array is rank 3, as
#: are the modal aerosol fields (columns, levels, modes); the RRTMG state's profiles have the
#: RRTMG level count.  A rank-0 input reaches the kernel as a float.
TABLE_INPUTS = (
    ("nstep", 0), ("lchnk", 0), ("ncol", 0), ("calday", 0), ("dosw", 0), ("dolw", 0),
    ("coszrs", 1), ("clat", 1), ("clon", 1),
    ("state_t", 2), ("state_pmid", 2), ("state_pint", 2), ("state_pdel", 2), ("state_lnpint", 2), ("state_lnpmid", 2),
    ("state_q", 3),
    ("cld", 2), ("cldfsnow", 2), ("dei", 2), ("mu", 2), ("lambdac", 2), ("iciwp", 2), ("iclwp", 2), ("des", 2), ("icswp", 2),
    ("dgnumwet", 3), ("qaerwat", 3),
    ("cam_in_lwup", 1), ("cam_in_asdir", 1), ("cam_in_asdif", 1), ("cam_in_aldir", 1), ("cam_in_aldif", 1),
    ("rstate_h2ovmr", 2), ("rstate_o3vmr", 2), ("rstate_co2vmr", 2), ("rstate_ch4vmr", 2), ("rstate_o2vmr", 2),
    ("rstate_n2ovmr", 2), ("rstate_cfc11vmr", 2), ("rstate_cfc12vmr", 2), ("rstate_cfc22vmr", 2), ("rstate_ccl4vmr", 2),
    ("rstate_pmidmb", 2), ("rstate_pintmb", 2), ("rstate_tlay", 2), ("rstate_tlev", 2),
)
TABLE_OUTPUTS = tuple((name, 2 if name in ("qrs", "qrl") else 1) for name in OUTPUTS)


class RadiationProcessCapture:
    """Record the radiation branch's inputs and outputs on every radiative step of every chunk."""

    #: the stage runs the original branch and hands this object what it saw
    records = True
    answers = False

    def __init__(self, *, every: int = 1) -> None:
        #: record every ``every``-th radiative step of this rank (all of them by default): a month
        #: of every call is 270 GB over 512 ranks, every eighth step a training set of 25 GB
        self.every = max(1, int(every))
        self.inputs: list[dict[str, Any]] = []
        self.outputs: list[dict[str, np.ndarray]] = []
        self._steps_seen: list[int] = []
        self.skipped = 0

    def wants(self, nstep: int) -> bool:
        """Whether this radiative step is recorded: the first, then every ``every``-th distinct step."""

        if not self._steps_seen or self._steps_seen[-1] != int(nstep):
            self._steps_seen.append(int(nstep))
        return (len(self._steps_seen) - 1) % self.every == 0

    def record(self, inputs: dict[str, Any], outputs: dict[str, np.ndarray]) -> None:
        if not self.wants(int(inputs["nstep"])):
            self.skipped += 1
            return
        self.inputs.append({k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in inputs.items()})
        self.outputs.append({k: np.array(v, copy=True) for k, v in outputs.items()})

    @property
    def calls(self) -> int:
        return len(self.outputs)

    def save(self, path: str | Path) -> Path:
        arrays: dict[str, np.ndarray] = {}
        meta = []
        for index, (before, after) in enumerate(zip(self.inputs, self.outputs)):
            meta.append({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else bool(v) if isinstance(v, (bool, np.bool_)) else v)
                         for k, v in before.items() if not isinstance(v, np.ndarray)})
            for name, value in before.items():
                if isinstance(value, np.ndarray):
                    arrays[f"in/{index}/{name}"] = value
            for name, value in after.items():
                arrays[f"out/{index}/{name}"] = value
        arrays["meta"] = np.array(json.dumps(meta))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, **arrays)
        return target

    def describe(self) -> dict[str, Any]:
        described: dict[str, Any] = {"kind": "capture", "calls": self.calls}
        if self.every > 1:
            described.update(every=self.every, skipped=self.skipped)
        return described

    def __repr__(self) -> str:
        return f"RadiationProcessCapture(calls={self.calls}{f', every={self.every}' if self.every > 1 else ''})"


#: the replay tables this process loaded, by capture directory: a stage is cloudpickled into each
#: rank's process registry at install and the payload must hash the same on every rank (7417389), so
#: the pickle of a replay carries the directory alone and the rank's own table is found again here
_REPLAY_TABLES: dict[str, tuple[Path, dict[tuple[int, int], dict[str, np.ndarray]]]] = {}


class RadiationReplay:
    """Answer the radiation branch with the outputs a capture recorded for the same step and chunk."""

    records = False
    answers = True

    def __init__(self, directory: str | Path, rank: int) -> None:
        self.directory = str(directory)
        path = Path(directory) / f"radiation_tend.rank-{int(rank):04d}.npz"
        if not path.is_file():
            raise PhysicsError(f"no radiation capture for rank {rank} at {path}")
        archive = np.load(path, allow_pickle=True)
        meta = json.loads(str(archive["meta"]))
        table: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for index, record in enumerate(meta):
            key = (int(record["nstep"]), int(record["lchnk"]))
            table[key] = {name: np.asarray(archive[f"out/{index}/{name}"]) for name in OUTPUTS
                          if f"out/{index}/{name}" in archive.files}
        _REPLAY_TABLES[self.directory] = (path, table)
        self.calls = 0

    @property
    def path(self) -> Path:
        return _REPLAY_TABLES[self.directory][0]

    @property
    def _by_call(self) -> dict[tuple[int, int], dict[str, np.ndarray]]:
        return _REPLAY_TABLES[self.directory][1]

    def __getstate__(self) -> dict[str, Any]:
        return {"directory": self.directory, "calls": self.calls}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if self.directory not in _REPLAY_TABLES:
            raise PhysicsError(
                f"a radiation replay of {self.directory} cannot cross processes by pickle; its table lives in "
                f"the process that loaded it")

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        key = (int(inputs["nstep"]), int(inputs["lchnk"]))
        try:
            answer = self._by_call[key]
        except KeyError:
            raise PhysicsError(f"the radiation capture at {self.path} has no record for step {key[0]}, chunk {key[1]}") from None
        self.calls += 1
        return answer

    def describe(self) -> dict[str, Any]:
        return {"kind": "replay", "file": self.path.name, "calls": self.calls, "records": len(self._by_call)}

    def __repr__(self) -> str:
        return f"RadiationReplay({str(self.path)!r})"


class RadiationProcessModel:
    """Answer the radiation branch with a function: the inputs by name in, the outputs by name out."""

    records = False
    answers = True

    def __init__(self, function: Callable[[dict[str, Any]], dict[str, np.ndarray]], *, label: str) -> None:
        if not callable(function):
            raise PhysicsError(f"a radiation process model must be callable, got {type(function).__name__}")
        self.function = function
        self.label = str(label)
        self.calls = 0
        self.seconds = 0.0          # wall time inside the function, this rank; the first call carries its compile
        self.first_seconds = 0.0

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        import time

        started = time.perf_counter()
        answer = self.function(inputs)
        elapsed = time.perf_counter() - started
        missing = [name for name in OUTPUTS if name not in answer]
        if missing:
            raise PhysicsError(f"the radiation process model {self.label} returned no {missing}")
        if self.calls == 0:
            self.first_seconds = elapsed
        self.calls += 1
        self.seconds += elapsed
        return answer

    def describe(self) -> dict[str, Any]:
        return {"kind": "model", "function": self.label, "calls": self.calls,
                "seconds": self.seconds, "first_call_seconds": self.first_seconds}

    def __repr__(self) -> str:
        return f"RadiationProcessModel({self.label!r})"


def load_process_model(spec: str, *, rank: int):
    """``replay:DIR`` -> :class:`RadiationReplay`; ``MODULE:FUNCTION`` or ``path.py:FUNCTION`` -> :class:`RadiationProcessModel`."""

    import importlib
    import importlib.util
    import sys

    if spec.startswith("replay:"):
        return RadiationReplay(spec[len("replay:"):], rank)
    module_name, sep, function_name = spec.rpartition(":")
    if not sep or not module_name or not function_name:
        raise PhysicsError(f"--radiation-model takes replay:DIR, MODULE:FUNCTION or path.py:FUNCTION, got {spec!r}")
    if module_name.endswith(".py"):
        path = Path(module_name).expanduser().resolve()
        if not path.is_file():
            raise PhysicsError(f"--radiation-model: {path} is not a file")
        loader_spec = importlib.util.spec_from_file_location(f"freecam_radiation_model_{path.stem}", path)
        module = importlib.util.module_from_spec(loader_spec)
        sys.modules[loader_spec.name] = module
        loader_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if function is None:
        raise PhysicsError(f"--radiation-model: {spec} names no function")
    return RadiationProcessModel(function, label=f"{Path(module_name).name}:{function_name}")


__all__ = ["ARRAY_INPUTS", "OUTPUTS", "RSTATE_INPUTS", "SCALAR_INPUTS", "RadiationProcessCapture",
           "RadiationProcessModel", "RadiationReplay", "load_process_model"]


# -- the slot inside the image -----------------------------------------------------------------

def compile_radiation_plugin(function: Callable[..., Any], *, shadow: bool = False):
    """Compile a kernel over the Fortran slot's table (TABLE_INPUTS then TABLE_OUTPUTS, positional)
    into a plugin the image calls at the top of the radiative branch: no Python in the step.

    The kernel takes the 46 inputs (scalars as floats, arrays ``[column, ...]`` Fortran-ordered) and
    the 12 output arrays, writing the outputs in place.
    """

    from .numba_kernel import compile_table_kernel

    return compile_table_kernel(TABLE_INPUTS, TABLE_OUTPUTS, function, kernel="radiation_process", shadow=shadow)


def bind_radiation_process(library: Any, address: int, *, shadow: bool = False) -> None:
    """Bind a compiled plugin at the image's radiation process slot (``pycam_rad_process_bind_v1``)."""

    import ctypes

    entry = getattr(library, "pycam_rad_process_bind_v1", None)
    if entry is None:
        raise PhysicsError("this image has no radiation process slot (pycam_rad_process_bind_v1): built before it")
    entry.restype = ctypes.c_int32
    entry.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    status = int(entry(ctypes.c_void_p(int(address)), 1 if shadow else 0))
    if status != 0:
        raise PhysicsError(f"the radiation process slot refused the plugin (status {status})")


def unbind_radiation_process(library: Any) -> None:
    entry = getattr(library, "pycam_rad_process_unbind_v1", None)
    if entry is not None:
        entry.restype = __import__("ctypes").c_int32
        entry.argtypes = []
        entry()


def read_radiation_process_counts(library: Any) -> dict[str, Any] | None:
    """This rank's plugin calls at the slot, the wall seconds inside them, and the first call alone."""

    import ctypes

    entry = getattr(library, "pycam_rad_process_counts_v1", None)
    if entry is None:
        return None
    entry.restype = ctypes.c_int32
    entry.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    calls, seconds, first = ctypes.c_int64(0), ctypes.c_double(0.0), ctypes.c_double(0.0)
    if entry(ctypes.byref(calls), ctypes.byref(seconds), ctypes.byref(first)) != 0:
        return None
    return {"calls": int(calls.value), "seconds": float(seconds.value), "first_call_seconds": float(first.value)}


__all__ += ["TABLE_INPUTS", "TABLE_OUTPUTS", "compile_radiation_plugin", "bind_radiation_process",
            "unbind_radiation_process", "read_radiation_process_counts"]
