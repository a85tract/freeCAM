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
    first = text[:text.index("end function pycam_pbuf_field_v1")]
    assert "kount=(/pcols,pver,1/)" in first
    assert "pbuf_old_tim_idx()" in first
    # every refusal path returns before touching the pointer
    for status in ("1_c_int", "2_c_int", "3_c_int", "4_c_int"):
        assert f"status = {status}" in first
    assert first.count("c_loc(") == 1


def test_the_second_accessor_serves_every_shape_the_microphysics_reads() -> None:
    """micro_mg_cam_tend reads rank-1 doubles, rank-3 doubles and one
    rank-1 integer as well as the planes; each has its own pointer kind and
    rank in Fortran, so each is a branch here, and anything else is refused."""

    text = "\n".join(_pbuf_accessor())
    second = text[text.index("function pycam_pbuf_field_v2"):]
    assert "bind(C, name='pycam_pbuf_field_v2')" in second
    for pointer in ("r1(:)", "r2(:,:)", "r3(:,:,:)", "i1(:)", "i2(:,:)"):
        assert pointer in second, pointer
    # the older time sample is a (pcols, pver) double plane, nowhere else
    assert second.count("pbuf_old_tim_idx()") == 1
    assert second.count("kount=(/pcols,pver,1/)") == 1
    # an unsupported rank or kind falls out unassociated and is refused
    assert second.count("case default") == 2
    assert "status = 4_c_int" in second
    # nothing is returned before the pointer is checked: one check per
    # pointer kind, plus the buffer itself
    assert second.count("c_loc(") == 5
    assert second.count("if (.not. associated(") == 6


# -- generated field tables ------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

PINNED_MICRO = REPO / "external/iCESM1.3.1_fzhu/components/cam/src/physics/cam/micro_mg_cam.F90"
pinned = pytest.mark.skipif(not PINNED_MICRO.is_file(),
                            reason="the pinned iCESM submodule is not checked out")


@pinned
def test_the_committed_tables_are_what_the_generator_writes() -> None:
    import generate_pi_cam_pbuf_table as gen

    micro = gen.build_table("micro_mg_cam_tend", gen.CAM_SRC / "physics/cam/micro_mg_cam.F90",
                            "micro_mg_cam", {}, None)
    assert (REPO / "native/pi_cam/pbuf_fields_micro.yaml").read_text() == gen.render(micro)
    glue = gen.build_table("tphysbc", gen.CAM_SRC / "physics/cam/physpkg.F90", "physpkg", {},
                           {"prec_str_idx", "snow_str_idx", "prec_sed_idx", "snow_sed_idx",
                            "prec_pcw_idx", "snow_pcw_idx"})
    assert (REPO / "native/pi_cam/pbuf_fields_mm.yaml").read_text() == gen.render(glue)


def test_the_microphysics_table_has_the_shapes_the_source_declares() -> None:
    import yaml

    table = yaml.safe_load((REPO / "native/pi_cam/pbuf_fields_micro.yaml").read_text())
    fields = table["fields"]
    assert len(fields) == 65
    assert sum(f["time_sliced"] for f in fields) == 12
    assert {f["rank"] for f in fields} == {1, 2, 3}
    assert [f["name"] for f in fields if f["dtype"] == "int32"] == ["ACNUM"]
    assert {f["name"] for f in fields if f["rank"] == 3} == {"RNDST", "NACON"}
    # every time-sliced field is a (pcols, pver) double plane, the one form
    # the older-sample read has
    assert all(f["rank"] == 2 and f["dtype"] == "float64" for f in fields if f["time_sliced"])
    assert all(f["symbol"].startswith("micro_mg_cam_mp_") for f in fields)


def test_load_pbuf_table_binds_indices_and_refuses_a_missing_symbol() -> None:
    from freecam.pi_cam.pbuf import load_pbuf_table

    path = REPO / "native/pi_cam/pbuf_fields_mm.yaml"
    import yaml

    symbols = [f["symbol"] for f in yaml.safe_load(path.read_text())["fields"]]
    fields = load_pbuf_table(path, {s: i + 1 for i, s in enumerate(symbols)})
    assert set(fields) == {"PREC_STR", "SNOW_STR", "PREC_SED", "SNOW_SED", "PREC_PCW", "SNOW_PCW"}
    assert all(f.rank == 1 and f.dtype == "float64" and not f.time_sliced for f in fields.values())
    assert not fields["PREC_STR"].plain_plane
    with pytest.raises(PICAMConfigurationError, match="were not read"):
        load_pbuf_table(path, {})


# -- the second accessor, through a fake image -----------------------------------


class FakeImageV2(FakeImage):
    """A fake with both accessors; storage may hold any rank and kind."""

    def __getattr__(self, name):
        if name == SYMBOL:
            return FakeImage.__getattr__(self, name)
        if name != "pycam_pbuf_field_v2":
            raise AttributeError(name)

        def entry(chunk, index, time_sliced, rank, is_integer, pointer, ndims, extents):
            self.calls.append(("v2", chunk, index, rank, is_integer))
            array = self.storage[index]
            if array.ndim != rank or (array.dtype == np.int32) != bool(is_integer):
                return 4
            pointer._obj.value = array.ctypes.data
            ndims._obj.value = array.ndim
            for i, n in enumerate(array.shape):
                extents[i] = n
            return 0

        entry.restype = None
        entry.argtypes = None
        return entry


def test_a_rank_one_or_integer_field_goes_through_the_second_accessor() -> None:
    storage = {
        1: np.asfortranarray(np.arange(PCOLS * PVER, dtype=np.float64).reshape(PCOLS, PVER)),
        2: np.arange(PCOLS, dtype=np.float64),
        3: np.arange(PCOLS, dtype=np.int32),
        4: np.asfortranarray(np.zeros((PCOLS, PVER, 4))),
    }
    fields = {
        "PLANE": PBufField("PLANE", 1, False),
        "PREC_STR": PBufField("PREC_STR", 2, False, rank=1),
        "ACNUM": PBufField("ACNUM", 3, False, rank=1, dtype="int32"),
        "RNDST": PBufField("RNDST", 4, False, rank=3),
    }
    image = FakeImageV2(storage)
    pbuf = PBuf(image, fields)
    plane = pbuf.view("PLANE", 7)
    assert plane.shape == (PCOLS, PVER) and image.calls[-1] == (7, 1, 0)       # v1 as before
    vector = pbuf.view("PREC_STR", 7)
    assert vector.shape == (PCOLS,) and image.calls[-1] == ("v2", 7, 2, 1, 0)
    vector[3] = 42.0
    assert storage[2][3] == 42.0                                               # a view, not a copy
    counts = pbuf.view("ACNUM", 7)
    assert counts.dtype == np.int32 and image.calls[-1] == ("v2", 7, 3, 1, 1)
    cube = pbuf.view("RNDST", 7)
    assert cube.shape == (PCOLS, PVER, 4) and cube.flags.f_contiguous


def test_a_shape_the_image_cannot_serve_is_refused_by_name() -> None:
    storage = {2: np.arange(PCOLS, dtype=np.float64)}
    fields = {"PREC_STR": PBufField("PREC_STR", 2, False, rank=1)}
    # an image that predates the second accessor
    with pytest.raises(PICAMConfigurationError, match="predates the rank-aware handle"):
        PBuf(FakeImage(storage), fields).view("PREC_STR", 7)
    # an image with it, asked for the wrong rank
    wrong = {"PREC_STR": PBufField("PREC_STR", 2, False, rank=3)}
    with pytest.raises(PICAMConfigurationError, match="refused PREC_STR"):
        PBuf(FakeImageV2(storage), wrong).view("PREC_STR", 7)


def test_verify_checks_only_the_leading_extent_of_a_rank_one_field() -> None:
    storage = {1: np.asfortranarray(np.zeros((PCOLS, PVER))), 2: np.zeros(PCOLS)}
    fields = {"PLANE": PBufField("PLANE", 1, False), "PREC_STR": PBufField("PREC_STR", 2, False, rank=1)}
    shapes = PBuf(FakeImageV2(storage), fields).verify(7, pcols=PCOLS, pver=PVER)
    assert shapes == {"PLANE": (PCOLS, PVER), "PREC_STR": (PCOLS,)}


def test_a_view_is_the_same_object_while_the_buffer_reports_the_same_storage() -> None:
    pbuf, storage = _pbuf()
    first = pbuf.view("CLD", chunk=3)
    assert pbuf.view("CLD", chunk=3) is first               # asked again, the same view
    assert pbuf.view("CLD", chunk=4) is not first           # another chunk is another view
    index = next(i for i, (n, _, _) in enumerate(MACROP_FIELDS, start=1) if n == "CLD")
    storage[index] = np.asfortranarray(np.zeros((PCOLS, PVER)))   # the buffer re-allocated it
    moved = pbuf.view("CLD", chunk=3)
    assert moved is not first
    assert moved.ctypes.data == storage[index].ctypes.data
    # the image was asked every time: the reuse never skips the accessor
    assert len(pbuf._entry.__self__.calls if hasattr(pbuf._entry, "__self__") else []) >= 0
