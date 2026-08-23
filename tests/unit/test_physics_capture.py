"""The capture stream decoder, against a stream written the Fortran way."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np
import pytest

from freecam.physics.capture import (
    KIND_INT,
    KIND_LOGICAL,
    KIND_POINTER,
    KIND_REAL,
    iter_records,
    pair_records,
)


def _record(order: str, function: str, phase: str, *, nstep: int, lchnk: int, ncol: int, entries) -> bytes:
    out = b"PYCAM_FUNCTION1 " + function.ljust(32).encode() + phase.ljust(8).encode()
    out += struct.pack(f"{order}6i", 1, nstep, lchnk, ncol, 7, len(entries))
    out += struct.pack(f"{order}d", 1800.0)
    for tag, kind, values, associated in entries:
        array = np.asarray(values)
        dims = (1, 1) if array.ndim == 0 else ((array.shape[0], 1) if array.ndim == 1 else array.shape)
        out += tag.ljust(32).encode() + struct.pack(f"{order}4i", kind, array.ndim, *dims)
        if kind == KIND_POINTER:
            out += struct.pack(f"{order}i", int(associated))
            if not associated:
                continue
        if kind in (KIND_INT, KIND_LOGICAL):
            out += struct.pack(f"{order}i", int(array))
        else:
            out += np.asarray(array, dtype=f"{order}f8", order="F").tobytes(order="F")
    return out


@pytest.mark.parametrize("order", [">", "<"])
def test_records_round_trip_in_either_byte_order(tmp_path: Path, order: str) -> None:
    t = np.arange(16 * 3, dtype=np.float64).reshape(16, 3, order="F") + 0.5
    stream = b"".join(
        [
            _record(order, "dadadj", "before", nstep=2, lchnk=5, ncol=11, entries=[
                ("lchnk", KIND_INT, 5, True), ("t", KIND_REAL, t, True), ("snowh", KIND_REAL, t[:, 0], True),
                ("tke", KIND_POINTER, np.zeros((16, 4)), False), ("do_cldice", KIND_LOGICAL, 1, True),
            ]),
            _record(order, "dadadj", "after", nstep=2, lchnk=5, ncol=11, entries=[
                ("lchnk", KIND_INT, 5, True), ("t", KIND_REAL, 2.0 * t, True), ("snowh", KIND_REAL, t[:, 0], True),
                ("tke", KIND_POINTER, t, True), ("do_cldice", KIND_LOGICAL, 0, True),
            ]),
        ]
    )
    path = tmp_path / "probe.rank-000007.stream.bin"
    path.write_bytes(stream)

    records = list(iter_records(path))
    assert [(r.function, r.phase, r.nstep, r.lchnk, r.ncol, r.mpi_rank, r.dt) for r in records] == [
        ("dadadj", "before", 2, 5, 11, 7, 1800.0), ("dadadj", "after", 2, 5, 11, 7, 1800.0),
    ]
    before, after = records
    assert np.array_equal(before.entries["t"].values, t)
    assert before.entries["t"].values.flags.f_contiguous
    assert np.array_equal(after.entries["t"].values, 2.0 * t)
    assert before.entries["lchnk"].values.tolist() == [5]
    assert before.entries["do_cldice"].values.tolist() == [1] and after.entries["do_cldice"].values.tolist() == [0]
    assert before.entries["tke"].associated is False and before.entries["tke"].values.size == 0
    assert after.entries["tke"].associated is True and np.array_equal(after.entries["tke"].values, t)
    assert before.entries["snowh"].values.shape == (16,)

    pairs = pair_records(records)
    assert len(pairs) == 1 and pairs[0][0] is before and pairs[0][1] is after


def test_unpaired_records_are_rejected(tmp_path: Path) -> None:
    stream = _record(">", "dadadj", "before", nstep=1, lchnk=1, ncol=1, entries=[("lchnk", KIND_INT, 1, True)])
    path = tmp_path / "probe.rank-000000.stream.bin"
    path.write_bytes(stream)
    with pytest.raises(RuntimeError, match="has no after"):
        pair_records(list(iter_records(path)))
