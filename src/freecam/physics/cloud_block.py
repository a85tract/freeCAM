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

import hashlib
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
    #: the index itself when it was read from an array-valued registry rather than a scalar module integer
    index: int | None = None


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


#: buffer fields the blocks only read, fetched inside routines the drivers call (the field tables cover the
#: drivers' own fetches): cldfrc's convective cloud fractions; the ice nucleation's dry diameters and the
#: activation's eddy diffusivity.  Inputs to a model, never outputs.
MACRO_INPUT_BUFFERS = (BufferField("SH_FRAC", "cloud_fraction_mp_sh_frac_idx_", False),
                       BufferField("DP_FRAC", "cloud_fraction_mp_dp_frac_idx_", False))
MICRO_INPUT_BUFFERS = (BufferField("DGNUM", "nucleate_ice_cam_mp_dgnum_idx_", False, 3),
                       BufferField("KVH", "microp_aero_mp_kvh_idx_", False))


@dataclass(frozen=True)
class BlockContract:
    """A compute block by name: its inputs and outputs by name, and the buffer fields among them."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    buffers: tuple[BufferField, ...]
    #: the physics_ptend name the driver gives its object
    ptend_name: str
    #: buffer fields read but never written (in ``inputs``, not in ``outputs``)
    input_buffers: tuple[BufferField, ...] = ()
    #: whether the block also writes the cloud-borne aerosol fields, registered per constituent at run time
    #: (modal_aero_data's ``qqcw``): the activation inside the microphysics block updates them in place
    cloud_borne: bool = False
    #: whether the block also writes the water tracers' surface precipitation fields (water_tracer_vars'
    #: ``wtrc_srfpcp_indices``, one per precipitation type and tracer set): the microphysics driver's
    #: ``wtrc_output_precip`` fills the stratiform ones
    tracer_precipitation: bool = False

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
    + FORCING_FIELDS + tuple(f.name for f in MACRO_BUFFERS) + tuple(f.name for f in MACRO_INPUT_BUFFERS),
    outputs=TENDENCY_OUTPUTS + ("det_s", "det_ice") + tuple(f.name for f in MACRO_BUFFERS),
    buffers=MACRO_BUFFERS, ptend_name="macrop", input_buffers=MACRO_INPUT_BUFFERS)
#: microp_aero_run, micro_mg_cam_tend and the tendency sum: one block, the activation inside it -- which also
#: rewrites the cloud-borne aerosol fields (dropmixnuc, ndrop.F90), named per constituent at run time
MICRO_BLOCK = BlockContract(
    "micro",
    inputs=SCALARS + tuple(f"state_{n}" for n in STATE_FIELDS) + tuple(f.name for f in MICRO_BUFFERS)
    + tuple(f.name for f in MICRO_INPUT_BUFFERS),
    outputs=TENDENCY_OUTPUTS + tuple(f.name for f in MICRO_BUFFERS),
    buffers=MICRO_BUFFERS, ptend_name="cldwat", input_buffers=MICRO_INPUT_BUFFERS, cloud_borne=True,
    tracer_precipitation=True)
BLOCKS = {block.name: block for block in (MACRO_BLOCK, MICRO_BLOCK)}


def cloud_borne_fields(library: Any, pcnst: int) -> tuple[BufferField, ...]:
    """The cloud-borne aerosol fields of this image: modal_aero_data's ``qqcw`` registry, one buffer field per
    constituent that has a cloud-borne phase, named as the module names them (``cnst_name_cw``)."""

    from .image import module_view

    indices = np.asarray(module_view(library, "modal_aero_data_mp_qqcw_", "int32", (int(pcnst),)))
    names = np.asarray(module_view(library, "modal_aero_data_mp_cnst_name_cw_", "S16", (int(pcnst),)))
    fields = []
    for m in range(int(pcnst)):
        index = int(indices[m])
        if index <= 0:
            continue
        name = names[m].tobytes().decode("ascii", "replace").strip() or f"QQCW_{m + 1}"
        fields.append(BufferField(name, f"modal_aero_data_mp_qqcw_[{m}]", False, 2, "float64", index))
    return tuple(fields)


#: water_types.F90: the water types a surface precipitation field is registered for, by index
WATER_TYPES = 7
WATER_TYPE_NAMES = {4: "strain", 5: "stsnow", 6: "cvrain", 7: "cvsnow"}
#: water_tracer_vars.F90: WTRC_MAX_CNST / pwtype tracer sets in the index array
TRACER_SETS = 700 // WATER_TYPES


def tracer_precipitation_fields(library: Any) -> tuple[BufferField, ...]:
    """The water tracers' surface precipitation fields of this image: water_tracer_vars' ``wtrc_srfpcp_indices``
    (a ``(pwtype, sets)`` array in Fortran order, -1 where nothing is registered), named by type and set."""

    from .image import module_view

    flat = np.asarray(module_view(library, "water_tracer_vars_mp_wtrc_srfpcp_indices_", "int32", (WATER_TYPES * TRACER_SETS,)))
    fields = []
    for iwset in range(TRACER_SETS):
        for itype in range(WATER_TYPES):
            index = int(flat[iwset * WATER_TYPES + itype])
            if index <= 0:
                continue
            kind = WATER_TYPE_NAMES.get(itype + 1, f"type{itype + 1}")
            fields.append(BufferField(f"WTRC_P_{kind}_{iwset + 1}", f"water_tracer_vars_mp_wtrc_srfpcp_indices_[{itype},{iwset}]",
                                      False, 1, "float64", index))
    return tuple(fields)


def dynamic_fields(library: Any, pcnst: int, block: BlockContract) -> tuple[BufferField, ...]:
    """The buffer fields a block writes that only the image can name: registered per constituent or tracer."""

    fields: tuple[BufferField, ...] = ()
    if block.cloud_borne:
        fields += cloud_borne_fields(library, pcnst)
    if block.tracer_precipitation:
        fields += tracer_precipitation_fields(library)
    return fields


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
        """Copy the inputs now: the block writes into some of this storage.  Beside the single-precision
        copies, an exact digest of every input's live lanes goes into the record, so a replay can tell the
        first field and step at which its run has left the captured one."""

        ncol = int(inputs["ncol"])
        copied: dict[str, Any] = {}
        for name, value in inputs.items():
            if isinstance(value, np.ndarray):
                if self.keep_inputs:
                    copied[name] = np.array(value, dtype=np.float32 if value.dtype.kind == "f" else value.dtype, copy=True)
            else:
                copied[name] = value
        copied["digests"] = input_digests(inputs, ncol)
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
_REPLAY_TABLES: dict[tuple[str, str], tuple[Path, dict[tuple[int, int], dict[str, np.ndarray]], dict[tuple[int, int], dict[str, str]]]] = {}


def input_digests(inputs: dict[str, Any], ncol: int) -> dict[str, str]:
    """An exact digest of every array input's live lanes, by name."""

    return {name: hashlib.sha256(np.ascontiguousarray(np.asarray(value)[:ncol]).tobytes()).hexdigest()[:16]
            for name, value in inputs.items() if isinstance(value, np.ndarray)}


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
        digests: dict[tuple[int, int], dict[str, str]] = {}
        for index, record in enumerate(meta):
            key = (int(record["nstep"]), int(record["lchnk"]))
            prefix = f"out/{index}/"
            answer = {name[len(prefix):]: np.asarray(archive[name]) for name in archive.files if name.startswith(prefix)}
            if "ptend_q" in answer:
                answer["ptend_q"] = _expand_q(answer["ptend_q"], answer["ptend_lq"])
            table[key] = answer
            if isinstance(record.get("digests"), dict):
                digests[key] = dict(record["digests"])
        _REPLAY_TABLES[(self.directory, self.block_name)] = (path, table, digests)
        self.calls = 0
        #: inputs whose live lanes did not hash as the capture's, by (step, chunk): where this run left the captured one
        self.mismatches: list[tuple[int, int, tuple[str, ...]]] = []

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
        # the pickle carries the directory and the count; the table and the mismatch list are this process's own
        self.mismatches = []
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
        expected = _REPLAY_TABLES[(self.directory, self.block_name)][2].get(key)
        if expected:
            seen = input_digests(inputs, int(inputs["ncol"]))
            differing = tuple(name for name, digest in expected.items() if name in seen and seen[name] != digest)
            if differing:
                self.mismatches.append((key[0], key[1], differing))
                if len(self.mismatches) <= 8:
                    print(f"[cloud replay] {self.block_name} block, step {key[0]} chunk {key[1]}: {len(differing)} inputs differ "
                          f"from the capture: {', '.join(differing[:12])}", flush=True)
        self.calls += 1
        return answer

    def describe(self) -> dict[str, Any]:
        described = {"kind": "block-replay", "block": self.block_name, "file": self.path.name, "calls": self.calls,
                     "records": len(self._by_call)}
        if self.mismatches:
            first = self.mismatches[0]
            described.update(input_mismatches=len(self.mismatches), first_mismatch={"nstep": first[0], "lchnk": first[1], "inputs": list(first[2][:12])})
        return described

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


class OriginalBlock:
    """Put in a block's slot: the Python driver runs, and the block is the original driver called in place.

    The measure of the driver itself -- its views, its bookkeeping calls -- with nothing replaced and nothing
    recorded; the run must stay bit-for-bit.
    """

    records = False
    answers = False
    block_model = True

    def __init__(self, block: BlockContract) -> None:
        self.block_name = block.name

    @property
    def block(self) -> BlockContract:
        return BLOCKS[self.block_name]

    def describe(self) -> dict[str, Any]:
        return {"kind": "original-in-driver", "block": self.block_name}

    def __repr__(self) -> str:
        return f"OriginalBlock({self.block_name!r})"


class VerifiedOriginalBlock(OriginalBlock):
    """Put in a block's slot with a capture: the original driver runs in place, and what it left behind is compared,
    field by field and exactly, with what the capture recorded for the same step and chunk.

    The other half of the replay's input digests: where a replay says which of a block's inputs first differ from
    the captured run, this says which of a block's outputs the original produced differently from the capture on
    the same inputs -- a dependence the contract does not name.
    """

    def __init__(self, directory: str | Path, rank: int, block: BlockContract) -> None:
        super().__init__(block)
        self.replay = BlockReplay(directory, rank, block)
        self.mismatches: list[tuple[int, int, tuple[str, ...]]] = []
        self.compared = 0
        self._recorded: dict[str, np.ndarray] | None = None

    def begin(self, inputs: dict[str, Any]) -> None:
        """Before the original runs: hash the inputs as they lie and fetch the record (the block writes into
        some of this storage, so the inputs must be looked at now, not after)."""

        self._recorded = self.replay(inputs)

    def compare(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        recorded = self._recorded if self._recorded is not None else self.replay(inputs)
        self._recorded = None
        ncol = int(inputs["ncol"])
        differing = []
        for name, value in outputs.items():
            if name not in recorded:
                continue
            mine = np.asarray(value)[:ncol] if np.ndim(value) else np.asarray(value)
            theirs = np.asarray(recorded[name])[:ncol] if np.ndim(recorded[name]) else np.asarray(recorded[name])
            if mine.shape != theirs.shape or not np.array_equal(mine, theirs):
                differing.append(name)
        self.compared += 1
        if differing:
            key = (int(inputs["nstep"]), int(inputs["lchnk"]))
            self.mismatches.append((key[0], key[1], tuple(differing)))
            if len(self.mismatches) <= 8:
                print(f"[cloud verify] {self.block_name} block, step {key[0]} chunk {key[1]}: {len(differing)} outputs differ "
                      f"from the capture: {', '.join(differing[:12])}", flush=True)

    def describe(self) -> dict[str, Any]:
        described = {"kind": "original-verified", "block": self.block_name, "compared": self.compared,
                     "input_mismatches": len(self.replay.mismatches), "output_mismatches": len(self.mismatches)}
        if self.mismatches:
            first = self.mismatches[0]
            described["first_output_mismatch"] = {"nstep": first[0], "lchnk": first[1], "outputs": list(first[2][:12])}
        if self.replay.mismatches:
            first = self.replay.mismatches[0]
            described["first_input_mismatch"] = {"nstep": first[0], "lchnk": first[1], "inputs": list(first[2][:12])}
        return described

    def __repr__(self) -> str:
        return f"VerifiedOriginalBlock({self.block_name!r}, {self.replay.directory!r})"


CENSUS = REPO / "native/pi_cam/pbuf_census.yaml"


def census_fields(library: Any) -> tuple[BufferField, ...]:
    """Every buffer field of the census this image registers: name, resolved index, rank and kind.  A field with
    two time samples is served whole (both planes, rank one higher), so a write to either plane shows."""

    from .image import module_view

    fields = []
    for row in load_table(CENSUS)["rows"]:
        index = None
        for symbol in row["symbols"]:
            try:
                index = int(module_view(library, symbol, "int32", ()))
            except Exception:       # noqa: BLE001 -- a module this image does not link
                continue
            break
        if index is None or index <= 0:
            continue
        fields.append(BufferField(str(row["name"]), str(row["symbols"][0]), False, int(row["rank"]), str(row["dtype"]), index))
    return tuple(fields)


class CensusBlock(OriginalBlock):
    """Put in a block's slot: the original driver runs in place, and every buffer field of the census is compared
    before and after it -- the diagnostic that names what a block writes beyond its contract.

    ``unlisted`` counts, by name, the census fields the block changed that its contract does not list among the
    written; ``both_planes`` the listed time-rotated fields whose two planes both changed.
    """

    def __init__(self, block: BlockContract) -> None:
        super().__init__(block)
        self.compared = 0
        self.unlisted: dict[str, int] = {}
        self.both_planes: dict[str, int] = {}

    @staticmethod
    def snapshot(views: dict[str, np.ndarray], ncol: int) -> dict[str, np.ndarray]:
        return {name: np.array(view[:ncol], copy=True) for name, view in views.items()}

    def compare(self, before: dict[str, np.ndarray], views: dict[str, np.ndarray], ncol: int, written: set[str], step: int, lchnk: int) -> None:
        changed_unlisted = []
        for name, old in before.items():
            new = np.asarray(views[name][:ncol])
            if np.array_equal(old, new):
                continue
            if name in written:
                if old.ndim == 3 and old.shape[2] == 2 and all(not np.array_equal(old[:, :, k], new[:, :, k]) for k in range(2)):
                    self.both_planes[name] = self.both_planes.get(name, 0) + 1
                continue
            changed_unlisted.append(name)
            self.unlisted[name] = self.unlisted.get(name, 0) + 1
        self.compared += 1
        if changed_unlisted and self.compared <= 4:
            print(f"[cloud census] {self.block_name} block, step {step} chunk {lchnk}: wrote {len(changed_unlisted)} fields its contract "
                  f"does not list: {', '.join(sorted(changed_unlisted)[:20])}", flush=True)

    def describe(self) -> dict[str, Any]:
        return {"kind": "original-census", "block": self.block_name, "compared": self.compared,
                "unlisted": dict(sorted(self.unlisted.items())), "both_planes": dict(sorted(self.both_planes.items()))}

    def __repr__(self) -> str:
        return f"CensusBlock({self.block_name!r})"


def load_block_model(spec: str, *, block: BlockContract, rank: int):
    """``original`` -> :class:`OriginalBlock`; ``census`` -> :class:`CensusBlock`; ``verify:DIR`` ->
    :class:`VerifiedOriginalBlock`; ``replay:DIR`` -> :class:`BlockReplay`; ``MODULE:FUNCTION`` or
    ``path.py:FUNCTION`` -> :class:`BlockModel`."""

    if spec == "original":
        return OriginalBlock(block)
    if spec == "census":
        return CensusBlock(block)
    if spec.startswith("verify:"):
        return VerifiedOriginalBlock(spec[len("verify:"):], rank, block)
    if spec.startswith("replay:"):
        return BlockReplay(spec[len("replay:"):], rank, block)
    from .radiation_process import RadiationProcessModel, load_process_model

    model = load_process_model(spec, rank=rank)
    if not isinstance(model, RadiationProcessModel):
        raise PhysicsError(f"a cloud block model is replay:DIR, MODULE:FUNCTION or path.py:FUNCTION, got {spec!r}")
    return BlockModel(model.function, label=model.label, block=block)


__all__ = ["BLOCKS", "BlockCapture", "BlockContract", "BlockModel", "BlockReplay", "BufferField", "CAM_IN_FIELDS",
           "CloudBlockCapture", "FORCING_FIELDS", "MACRO_BLOCK", "MACRO_BUFFERS", "MACRO_INPUT_BUFFERS", "MICRO_BLOCK",
           "MICRO_BUFFERS", "MICRO_INPUT_BUFFERS", "OriginalBlock", "SCALARS", "STATE_FIELDS", "TENDENCY_OUTPUTS",
           "CENSUS", "CensusBlock", "VerifiedOriginalBlock", "census_fields", "cloud_borne_fields", "dynamic_fields",
           "input_digests", "load_block_model", "tracer_precipitation_fields"]
