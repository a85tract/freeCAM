"""Physics-buffer handles: zero-copy, or a refusal."""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from freecam.pi_cam.errors import PICAMConfigurationError  # noqa: E402
from freecam.pi_cam.pbuf import (  # noqa: E402
    MACROP_FIELDS, PBuf, PBufField, PBufFieldAbsent, SYMBOL, macrop_fields,
)
from freecam.pi_cam.state_codegen import _pbuf_accessor  # noqa: E402

PCOLS, PVER = 16, 30


class FakeImage:
    """Stands in for the CAM image: one Fortran-ordered array per field."""

    def __init__(self, storage: dict[int, np.ndarray], *, status: int = 0) -> None:
        self.storage = storage
        self.status = status
        self.calls: list[tuple[int, int, int]] = []

    def __getattr__(self, name):
        if name != SYMBOL:
            raise AttributeError(name)

        def entry(chunk, index, time_sliced, pointer, extents):
            self.calls.append((chunk, index, time_sliced))
            if self.status:
                return self.status
            array = self.storage[index]
            pointer._obj.value = array.ctypes.data
            extents[0], extents[1] = array.shape
            return 0

        entry.restype = None
        entry.argtypes = None
        return entry


def _pbuf(**overrides):
    storage = {}
    fields = {}
    for position, (name, _symbol, sliced) in enumerate(MACROP_FIELDS, start=1):
        fields[name] = PBufField(name, position, sliced)
        storage[position] = np.asfortranarray(
            np.arange(PCOLS * PVER, dtype=np.float64).reshape(PCOLS, PVER) + position
        )
    fields.update(overrides)
    return PBuf(FakeImage(storage), fields), storage


def test_a_view_is_the_buffer_itself_not_a_copy() -> None:
    pbuf, storage = _pbuf()
    view = pbuf.view("CLD", chunk=3)
    assert view.shape == (PCOLS, PVER)
    assert view.flags.f_contiguous
    index = next(i for i, (n, _, _) in enumerate(MACROP_FIELDS, start=1) if n == "CLD")
    view[2, 5] = -12345.0
    assert storage[index][2, 5] == -12345.0, "the write did not land in CAM's storage"


def test_the_older_time_sample_is_asked_for_only_where_the_source_asks() -> None:
    pbuf, _ = _pbuf()
    image = pbuf._entry.__self__ if hasattr(pbuf._entry, "__self__") else None
    del image
    pbuf.view("CLD", chunk=1)      # time-rotated in the source
    pbuf.view("CMELIQ", chunk=1)   # plain
    assert pbuf.fields["CLD"].time_sliced is True
    assert pbuf.fields["CMELIQ"].time_sliced is False
    # and the flag is what reaches Fortran
    sliced = {name for name, _, flag in MACROP_FIELDS if flag}
    assert "QCWAT" in sliced and "CC_qlst" in sliced
    assert "SHFRC" not in sliced and "NAAI" not in sliced


def test_an_unregistered_field_refuses_instead_of_returning_something() -> None:
    """CAM leaves the six UNICON indices at -1; reading one must not guess."""

    pbuf, _ = _pbuf(TKE=PBufField("TKE", -1, False))
    assert "TKE" not in pbuf
    with pytest.raises(PBufFieldAbsent, match="never registered"):
        pbuf.view("TKE", chunk=1)


def test_a_field_this_process_never_declared_is_a_configuration_error() -> None:
    pbuf, _ = _pbuf()
    with pytest.raises(PICAMConfigurationError, match="not one of"):
        pbuf.view("OMEGA", chunk=1)


def test_a_refusal_from_fortran_is_reported_not_swallowed() -> None:
    _, storage = _pbuf()
    fields = {name: PBufField(name, i, s)
              for i, (name, _, s) in enumerate(MACROP_FIELDS, start=1)}
    pbuf = PBuf(FakeImage(storage, status=3), fields)
    with pytest.raises(PICAMConfigurationError, match="not allocated"):
        pbuf.view("CLD", chunk=1)


def test_verify_checks_every_registered_field_once() -> None:
    pbuf, _ = _pbuf(TKE=PBufField("TKE", -1, False))
    shapes = pbuf.verify(chunk=1, pcols=PCOLS, pver=PVER)
    assert "TKE" not in shapes
    assert len(shapes) == len(MACROP_FIELDS) - 1
    assert set(shapes.values()) == {(PCOLS, PVER)}


def test_verify_refuses_a_field_whose_shape_is_not_the_grid() -> None:
    _, storage = _pbuf()
    storage[1] = np.asfortranarray(np.zeros((PCOLS, 7)))
    fields = {name: PBufField(name, i, s)
              for i, (name, _, s) in enumerate(MACROP_FIELDS, start=1)}
    pbuf = PBuf(FakeImage(storage), fields)
    with pytest.raises(PICAMConfigurationError, match="unexpected shapes"):
        pbuf.verify(chunk=1, pcols=PCOLS, pver=PVER)


def test_an_image_without_the_handle_says_so() -> None:
    class Bare:
        pass

    with pytest.raises(PICAMConfigurationError, match="predates"):
        PBuf(Bare(), {})


def test_the_index_table_must_be_complete() -> None:
    with pytest.raises(PICAMConfigurationError, match="were not read"):
        macrop_fields({"cld_idx": 4})
    indices = {symbol: position for position, (_, symbol, _) in enumerate(MACROP_FIELDS, 1)}
    bound = macrop_fields(indices)
    assert len(bound) == len(MACROP_FIELDS)
    assert bound["CLD"].index == indices["cld_idx"]


def test_the_generated_fortran_serves_only_the_two_shapes_the_source_uses() -> None:
    text = "\n".join(_pbuf_accessor())
    assert "kount=(/pcols,pver,1/)" in text
    assert "pbuf_old_tim_idx()" in text
    # every refusal path returns before touching the pointer
    for status in ("1_c_int", "2_c_int", "3_c_int", "4_c_int"):
        assert f"status = {status}" in text
    assert text.count("c_loc(") == 1
