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


class RadiationProcessCapture:
    """Record the radiation branch's inputs and outputs on every radiative step of every chunk."""

    #: the stage runs the original branch and hands this object what it saw
    records = True
    answers = False

    def __init__(self) -> None:
        self.inputs: list[dict[str, Any]] = []
        self.outputs: list[dict[str, np.ndarray]] = []

    def record(self, inputs: dict[str, Any], outputs: dict[str, np.ndarray]) -> None:
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
        return {"kind": "capture", "calls": self.calls}

    def __repr__(self) -> str:
        return f"RadiationProcessCapture(calls={self.calls})"


class RadiationReplay:
    """Answer the radiation branch with the outputs a capture recorded for the same step and chunk."""

    records = False
    answers = True

    def __init__(self, directory: str | Path, rank: int) -> None:
        self.path = Path(directory) / f"radiation_tend.rank-{int(rank):04d}.npz"
        if not self.path.is_file():
            raise PhysicsError(f"no radiation capture for rank {rank} at {self.path}")
        archive = np.load(self.path, allow_pickle=True)
        meta = json.loads(str(archive["meta"]))
        self._by_call: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for index, record in enumerate(meta):
            key = (int(record["nstep"]), int(record["lchnk"]))
            self._by_call[key] = {name: np.asarray(archive[f"out/{index}/{name}"]) for name in OUTPUTS
                                  if f"out/{index}/{name}" in archive.files}
        self.calls = 0

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

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        answer = self.function(inputs)
        missing = [name for name in OUTPUTS if name not in answer]
        if missing:
            raise PhysicsError(f"the radiation process model {self.label} returned no {missing}")
        self.calls += 1
        return answer

    def describe(self) -> dict[str, Any]:
        return {"kind": "model", "function": self.label, "calls": self.calls}

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
