"""The cloud macro/microphysics stage's two compute blocks as contracts: what each has in memory before its
arithmetic, and what it leaves behind.

tphysbc stage 7 (``physpkg.F90:2188-2393``) is, under the admitted configuration, two compute blocks with
bookkeeping between and after them: the macrophysics driver (``macrop_driver_tend``, the block ``mmacro_pcond``
lives in) and the microphysics driver with the aerosol activation that feeds it (``microp_aero_run``,
``micro_mg_cam_tend``, the tendency sum).  Each block reads the state, the surface, the convection carries and
its physics-buffer fields, and writes a tendency object, the macrophysics detrainment and its buffer fields.
The bookkeeping -- tendency scaling and application, the energy checks, the precipitation means, the tracer
mass fixer -- is not in either block.

:class:`BlockContract` names a block's inputs and outputs.  :class:`BlockCapture` records them from the original
drivers; :class:`BlockReplay` answers a block from a capture (the Python driver's bit-for-bit gate);
:class:`BlockModel` answers it with any Python callable.  The Python driver
(:meth:`freecam.physics.cloud_macro_microphysics.CloudMacroMicrophysics._tend_python_driver`) reads the inputs
from memory, calls whatever stands in the block's slot, writes the outputs where the driver leaves them, and
does the bookkeeping through the same Fortran calls the glue makes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..pi_cam.pbuf import MACROP_FIELDS
from ..pi_cam.tables import load_table
from .errors import PhysicsError

REPO = Path(__file__).resolve().parents[3]
MICRO_TABLE = REPO / "native/pi_cam/pbuf_fields_micro.yaml"
AERO_TABLE = REPO / "native/pi_cam/pbuf_fields_aero.yaml"

#: the scalars every block is given
SCALARS = ("nstep", "lchnk", "ncol", "dt")
#: the physics_state components the drivers read, by name in the state pool (``phys_state.<name>``)
STATE_FIELDS = ("t", "q", "pmid", "pdel", "rpdel", "pint", "lnpmid", "lnpint", "zm", "zi", "omega", "ps", "phis")
#: the surface input the macrophysics driver takes from cam_in, in tphysbc's argument names
CAM_IN_FIELDS = ("landfrac", "ocnfrac", "snowhland", "ts", "sst")
#: tphysbc's convection carries the macrophysics driver takes (control patch 0039's buffers)
FORCING_FIELDS = ("dlf", "dlf2", "wtdlf", "cmfmc", "cmfmc2", "zdu")
#: the tendency object as a block leaves it: arrays and the flags physics_update reads
TENDENCY_OUTPUTS = ("ptend_s", "ptend_q", "ptend_ls", "ptend_lq")


@dataclass(frozen=True)
class BufferField:
    """One physics-buffer field a block reads or writes: its registered name, the module integer holding
    its index, whether the driver takes the older time sample, and the pointer's rank and kind."""

    name: str
    symbol: str
    time_sliced: bool
    rank: int = 2
    dtype: str = "float64"


def _macro_buffers() -> tuple[BufferField, ...]:
    return tuple(BufferField(name, f"macrop_driver_mp_{symbol}_", bool(sliced)) for name, symbol, sliced in MACROP_FIELDS)


def _table_buffers(path: Path) -> tuple[BufferField, ...]:
    rows = load_table(path)["fields"]
    return tuple(BufferField(str(r["name"]), str(r["symbol"]), bool(r["time_sliced"]), int(r.get("rank", 2)),
                             str(r.get("dtype", "float64"))) for r in rows)


def _union(*groups: tuple[BufferField, ...]) -> tuple[BufferField, ...]:
    """The fields of several drivers by name; a field two drivers read differently is refused, not guessed."""

    by_name: dict[str, BufferField] = {}
    for group in groups:
        for field in group:
            other = by_name.get(field.name)
            if other is None:
                by_name[field.name] = field
            elif (other.time_sliced, other.rank, other.dtype) != (field.time_sliced, field.rank, field.dtype):
                raise PhysicsError(f"buffer field {field.name} is read two ways: {other} and {field}")
    return tuple(by_name.values())


@dataclass(frozen=True)
class BlockContract:
    """A compute block by name: its inputs and outputs by name, and the buffer fields among them."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    buffers: tuple[BufferField, ...]
    #: the physics_ptend name the driver gives its object
    ptend_name: str

    @property
    def file_stem(self) -> str:
        return f"cloud_{self.name}"

    @property
    def buffer_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.buffers)


MACRO_BUFFERS = _macro_buffers()
MICRO_BUFFERS = _union(_table_buffers(AERO_TABLE), _table_buffers(MICRO_TABLE))

#: macrop_driver_tend: the block mmacro_pcond lives in
MACRO_BLOCK = BlockContract(
    "macro",
    inputs=SCALARS + tuple(f"state_{n}" for n in STATE_FIELDS) + tuple(f"cam_in_{n}" for n in CAM_IN_FIELDS)
    + FORCING_FIELDS + tuple(f.name for f in MACRO_BUFFERS),
    outputs=TENDENCY_OUTPUTS + ("det_s", "det_ice") + tuple(f.name for f in MACRO_BUFFERS),
    buffers=MACRO_BUFFERS, ptend_name="macrop")
#: microp_aero_run, micro_mg_cam_tend and the tendency sum: one block, the activation inside it
MICRO_BLOCK = BlockContract(
    "micro",
    inputs=SCALARS + tuple(f"state_{n}" for n in STATE_FIELDS) + tuple(f.name for f in MICRO_BUFFERS),
    outputs=TENDENCY_OUTPUTS + tuple(f.name for f in MICRO_BUFFERS),
    buffers=MICRO_BUFFERS, ptend_name="cldwat")
BLOCKS = {block.name: block for block in (MACRO_BLOCK, MICRO_BLOCK)}


# -- recording the original blocks -------------------------------------------------------------------------------

class BlockCapture:
    """Record a block's inputs (as seen before it ran) and outputs (as it left them) on every call.

    Inputs are kept in single precision -- they are training data -- and outputs exactly: a replay writes them
    back and the run must stay bit-for-bit.  The tendency's constituent array is saved for the flagged
    constituents only (the others are zero by construction) and expanded again on load.
    """

    records = True
    answers = False

    def __init__(self, block: BlockContract, *, inputs: bool = True) -> None:
        self.block = block
        self.keep_inputs = bool(inputs)
        self.inputs: list[dict[str, Any]] = []
        self.outputs: list[dict[str, Any]] = []

    def begin(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Copy the inputs now: the block writes into some of this storage."""

        copied: dict[str, Any] = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                if self.keep_inputs:
                    copied[name] = np.array(value, dtype=np.float32 if value.dtype.kind == "f" else value.dtype, copy=True)
            else:
                copied[name] = value
        return copied

    def finish(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        self.inputs.append(inputs)
        self.outputs.append({k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in outputs.items()})

    @property
    def calls(self) -> int:
        return len(self.outputs)

    def save(self, path: str | Path) -> Path:
        arrays: dict[str, np.ndarray] = {}
        meta = []
        for index, (before, after) in enumerate(zip(self.inputs, self.outputs)):
            meta.append({k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)
                             else bool(v) if isinstance(v, (bool, np.bool_)) else v)
                         for k, v in before.items() if not isinstance(v, np.ndarray)})
            for name, value in before.items():
                if isinstance(value, np.ndarray):
                    arrays[f"in/{index}/{name}"] = value
            lq = np.asarray(after["ptend_lq"]).reshape(-1)
            for name, value in after.items():
                if name == "ptend_q":
                    arrays[f"out/{index}/{name}"] = np.ascontiguousarray(np.asarray(value)[:, :, lq != 0])
                else:
                    arrays[f"out/{index}/{name}"] = np.asarray(value)
        arrays["meta"] = np.array(json.dumps(meta))
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        return target

    def describe(self) -> dict[str, Any]:
        return {"kind": "capture", "block": self.block.name, "calls": self.calls}

    def __repr__(self) -> str:
        return f"BlockCapture({self.block.name!r}, calls={self.calls})"


class CloudBlockCapture:
    """Both blocks recorded: what ``--cloud-block-capture`` installs."""

    records = True
    answers = False

    def __init__(self, *, inputs: bool = True) -> None:
        self.macro = BlockCapture(MACRO_BLOCK, inputs=inputs)
        self.micro = BlockCapture(MICRO_BLOCK, inputs=inputs)

    def of(self, block: BlockContract) -> BlockCapture:
        return self.macro if block is MACRO_BLOCK else self.micro

    def save(self, directory: str | Path, rank: int) -> list[str]:
        return [str(capture.save(Path(directory) / f"{capture.block.file_stem}.rank-{int(rank):04d}.npz").name)
                for capture in (self.macro, self.micro)]

    def describe(self) -> dict[str, Any]:
        return {"kind": "capture", "macro_calls": self.macro.calls, "micro_calls": self.micro.calls}


# -- answering a block -------------------------------------------------------------------------------------------

#: the replay tables this process loaded, by (directory, block): a stage is cloudpickled into each rank's process
#: registry and the payload must hash the same on every rank, so the pickle carries the directory and the rank's
#: own table is found again here (radiation_process._REPLAY_TABLES's precedent)
_REPLAY_TABLES: dict[tuple[str, str], tuple[Path, dict[tuple[int, int], dict[str, np.ndarray]]]] = {}


def _expand_q(q_flagged: np.ndarray, lq: np.ndarray) -> np.ndarray:
    """The saved (ncol, pver, flagged) tendency back to (ncol, pver, pcnst), zeros where the flag is off."""

    lq = np.asarray(lq).reshape(-1)
    full = np.zeros(q_flagged.shape[:2] + (lq.shape[0],), dtype=q_flagged.dtype, order="F")
    full[:, :, lq != 0] = q_flagged
    return full


class BlockReplay:
    """Answer a block with the outputs a capture recorded for the same step and chunk."""

    records = False
    answers = True
    block_model = True

    def __init__(self, directory: str | Path, rank: int, block: BlockContract) -> None:
        self.directory = str(directory)
        self.block_name = block.name
        path = Path(directory) / f"{block.file_stem}.rank-{int(rank):04d}.npz"
        if not path.is_file():
            raise PhysicsError(f"no {block.name} block capture for rank {rank} at {path}")
        archive = np.load(path, allow_pickle=True)
        meta = json.loads(str(archive["meta"]))
        table: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        for index, record in enumerate(meta):
            key = (int(record["nstep"]), int(record["lchnk"]))
            prefix = f"out/{index}/"
            answer = {name[len(prefix):]: np.asarray(archive[name]) for name in archive.files if name.startswith(prefix)}
            if "ptend_q" in answer:
                answer["ptend_q"] = _expand_q(answer["ptend_q"], answer["ptend_lq"])
            table[key] = answer
        _REPLAY_TABLES[(self.directory, self.block_name)] = (path, table)
        self.calls = 0

    @property
    def block(self) -> BlockContract:
        return BLOCKS[self.block_name]

    @property
    def path(self) -> Path:
        return _REPLAY_TABLES[(self.directory, self.block_name)][0]

    @property
    def _by_call(self) -> dict[tuple[int, int], dict[str, np.ndarray]]:
        return _REPLAY_TABLES[(self.directory, self.block_name)][1]

    def __getstate__(self) -> dict[str, Any]:
        return {"directory": self.directory, "block_name": self.block_name, "calls": self.calls}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        if (self.directory, self.block_name) not in _REPLAY_TABLES:
            try:
                from mpi4py import MPI
                rank = int(MPI.COMM_WORLD.Get_rank())
            except Exception as error:      # noqa: BLE001 -- no MPI here: the table cannot be found
                raise PhysicsError(f"a block replay of {self.directory} cannot cross processes by pickle without MPI "
                                   f"to name the rank whose capture to load") from error
            BlockReplay.__init__(self, self.directory, rank, BLOCKS[self.block_name])
            self.calls = int(state.get("calls", 0))

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        key = (int(inputs["nstep"]), int(inputs["lchnk"]))
        try:
            answer = self._by_call[key]
        except KeyError:
            raise PhysicsError(f"the {self.block_name} block capture at {self.path} has no record for step {key[0]}, "
                               f"chunk {key[1]}") from None
        self.calls += 1
        return answer

    def describe(self) -> dict[str, Any]:
        return {"kind": "block-replay", "block": self.block_name, "file": self.path.name, "calls": self.calls,
                "records": len(self._by_call)}

    def __repr__(self) -> str:
        return f"BlockReplay({self.block_name!r}, {str(self.path)!r})"


class BlockModel:
    """Answer a block with a function over its inputs: the Python driver calls it and writes what it returns.

    The answer must carry the tendency (``ptend_s``, ``ptend_q``, ``ptend_ls``, ``ptend_lq``; the flags are
    configuration constants a capture shows) and, for the macrophysics block, the detrainment; buffer fields it
    leaves out keep the values they had.
    """

    records = False
    answers = True
    block_model = True

    def __init__(self, function: Callable[[dict[str, Any]], dict[str, np.ndarray]], *, label: str, block: BlockContract) -> None:
        if not callable(function):
            raise PhysicsError(f"a cloud block model must be callable, got {type(function).__name__}")
        self.function = function
        self.label = str(label)
        self.block_name = block.name
        self.calls = 0
        self.seconds = 0.0
        self.first_seconds = 0.0

    @property
    def block(self) -> BlockContract:
        return BLOCKS[self.block_name]

    def __call__(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        import time

        started = time.perf_counter()
        answer = self.function(inputs)
        elapsed = time.perf_counter() - started
        required = TENDENCY_OUTPUTS + (("det_s", "det_ice") if self.block is MACRO_BLOCK else ())
        missing = [name for name in required if name not in answer]
        if missing:
            raise PhysicsError(f"the {self.block_name} block model {self.label} returned no {missing}")
        if self.calls == 0:
            self.first_seconds = elapsed
        self.calls += 1
        self.seconds += elapsed
        return answer

    def describe(self) -> dict[str, Any]:
        return {"kind": "block-model", "block": self.block_name, "function": self.label, "calls": self.calls,
                "seconds": self.seconds, "first_call_seconds": self.first_seconds}

    def __repr__(self) -> str:
        return f"BlockModel({self.block_name!r}, {self.label!r})"


def load_block_model(spec: str, *, block: BlockContract, rank: int):
    """``replay:DIR`` -> :class:`BlockReplay`; ``MODULE:FUNCTION`` or ``path.py:FUNCTION`` -> :class:`BlockModel`."""

    if spec.startswith("replay:"):
        return BlockReplay(spec[len("replay:"):], rank, block)
    from .radiation_process import RadiationProcessModel, load_process_model

    model = load_process_model(spec, rank=rank)
    if not isinstance(model, RadiationProcessModel):
        raise PhysicsError(f"a cloud block model is replay:DIR, MODULE:FUNCTION or path.py:FUNCTION, got {spec!r}")
    return BlockModel(model.function, label=model.label, block=block)


__all__ = ["BLOCKS", "BlockCapture", "BlockContract", "BlockModel", "BlockReplay", "BufferField", "CAM_IN_FIELDS",
           "CloudBlockCapture", "FORCING_FIELDS", "MACRO_BLOCK", "MACRO_BUFFERS", "MICRO_BLOCK", "MICRO_BUFFERS",
           "SCALARS", "STATE_FIELDS", "TENDENCY_OUTPUTS", "load_block_model"]
