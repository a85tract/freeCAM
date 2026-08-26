"""Decode physics-function captures written by pycam_function_capture.

A capture stream holds, per rank, a sequence of records: a 'before' record
with every actual argument as the routine received it and an 'after' record
with every argument as it left it, for one chunk on one timestep.  This
module turns those streams into arrays keyed by argument tag, preserving the
chunk layout exactly, so a standalone image can be handed the same inputs and
its outputs compared bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
import struct
from typing import Iterator, Sequence

import numpy as np

MAGIC = b"PYCAM_FUNCTION1 "
STREAM_NAME = re.compile(r"\.rank-(?P<rank>\d{6})\.stream\.bin$")
KIND_REAL, KIND_INT, KIND_LOGICAL, KIND_POINTER = 1, 2, 3, 4


@dataclass
class CaptureEntry:
    tag: str
    kind: int
    rank: int
    dims: tuple[int, int]
    associated: bool
    values: np.ndarray


@dataclass
class CaptureRecord:
    function: str
    phase: str
    nstep: int
    lchnk: int
    ncol: int
    mpi_rank: int
    dt: float
    entries: dict[str, CaptureEntry] = field(default_factory=dict)


class _Reader:
    def __init__(self, raw: bytes, path: Path) -> None:
        self.raw = raw
        self.path = path
        self.offset = 0
        self.order = ">"

    def take(self, count: int) -> bytes:
        end = self.offset + count
        if end > len(self.raw):
            raise RuntimeError(f"{self.path}: truncated at byte {self.offset}")
        chunk = self.raw[self.offset:end]
        self.offset = end
        return chunk

    def ints(self, count: int) -> tuple[int, ...]:
        return struct.unpack(f"{self.order}{count}i", self.take(4 * count))

    def reals(self, count: int) -> np.ndarray:
        return np.frombuffer(self.take(8 * count), dtype=f"{self.order}f8").astype(np.float64)

    @property
    def done(self) -> bool:
        return self.offset >= len(self.raw)


def _detect_order(reader: _Reader) -> None:
    """The production build writes big-endian; accept either, by the header."""

    start = reader.offset
    reader.take(16 + 32 + 8)
    for order in (">", "<"):
        reader.order = order
        version = struct.unpack(f"{order}i", reader.raw[reader.offset:reader.offset + 4])[0]
        if version == 1:
            reader.offset = start
            return
    raise RuntimeError(f"{reader.path}: cannot determine byte order of capture header")


def iter_records(path: Path) -> Iterator[CaptureRecord]:
    """Yield every record in one rank's capture stream, in file order."""

    reader = _Reader(path.read_bytes(), path)
    first = True
    while not reader.done:
        if first:
            _detect_order(reader)
            first = False
        magic = reader.take(16)
        if magic != MAGIC:
            raise RuntimeError(f"{path}: bad magic {magic!r} at byte {reader.offset - 16}")
        name = reader.take(32).decode("ascii").strip()
        phase = reader.take(8).decode("ascii").strip()
        version, nstep, lchnk, ncol, mpi_rank, nargs = reader.ints(6)
        if version != 1:
            raise RuntimeError(f"{path}: unsupported capture version {version}")
        (dt,) = struct.unpack(f"{reader.order}d", reader.take(8))
        record = CaptureRecord(name, phase, nstep, lchnk, ncol, mpi_rank, dt)
        for _ in range(nargs):
            tag = reader.take(32).decode("ascii").strip()
            kind, rank, dim0, dim1 = reader.ints(4)
            associated = True
            if kind == KIND_POINTER:
                associated = reader.ints(1)[0] != 0
            if kind in (KIND_INT, KIND_LOGICAL):
                values = np.asarray(reader.ints(1), dtype=np.int32)
            elif not associated:
                values = np.zeros((0, 0), dtype=np.float64, order="F")
            else:
                count = {0: 1, 1: dim0, 2: dim0 * dim1}[rank]
                flat = reader.reals(count)
                shape = {0: (), 1: (dim0,), 2: (dim0, dim1)}[rank]
                values = np.asarray(flat.reshape(shape, order="F"), order="F")
            record.entries[tag] = CaptureEntry(tag, kind, rank, (dim0, dim1), associated, values)
        yield record


def stream_paths(prefix: Path) -> list[Path]:
    paths = sorted(prefix.parent.glob(prefix.name + ".rank-*.stream.bin"))
    if not paths:
        raise RuntimeError(f"no capture streams match {prefix}.rank-*.stream.bin")
    return paths


def pair_records(records: Sequence[CaptureRecord]) -> list[tuple[CaptureRecord, CaptureRecord]]:
    """Match each 'before' with the 'after' that follows it for the same call."""

    pairs: list[tuple[CaptureRecord, CaptureRecord]] = []
    pending: CaptureRecord | None = None
    for record in records:
        if record.phase == "before":
            if pending is not None:
                raise RuntimeError(f"{pending.function} before-record at step {pending.nstep} chunk {pending.lchnk} has no after")
            pending = record
        elif record.phase == "after":
            if pending is None or (pending.function, pending.nstep, pending.lchnk) != (record.function, record.nstep, record.lchnk):
                raise RuntimeError(f"{record.function} after-record at step {record.nstep} chunk {record.lchnk} has no before")
            pairs.append((pending, record))
            pending = None
        else:
            raise RuntimeError(f"unknown capture phase {record.phase!r}")
    if pending is not None:
        raise RuntimeError(f"{pending.function} before-record at step {pending.nstep} has no after")
    return pairs


def lane_sha256(values: np.ndarray, ncol: int) -> str:
    """Hash the live lanes of one chunk argument, in Fortran order.

    The padding lanes ``ncol..pcols-1`` are never written by the routine and
    hold whatever the caller's storage held, so they are excluded; everything
    else -- every level, every constituent -- is hashed as float64 bytes.
    Used by the Python driver layer's trace and by the tool that compares it
    against a capture, so both hash the same bytes.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        return hashlib.sha256(array.tobytes()).hexdigest()
    live = np.asfortranarray(array[: int(ncol)])
    return hashlib.sha256(live.tobytes(order="F")).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


__all__ = [
    "CaptureEntry",
    "CaptureRecord",
    "KIND_INT",
    "KIND_LOGICAL",
    "KIND_POINTER",
    "KIND_REAL",
    "array_sha256",
    "iter_records",
    "pair_records",
    "stream_paths",
]
