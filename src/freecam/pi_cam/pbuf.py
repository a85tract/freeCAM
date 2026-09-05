"""Zero-copy handles on CAM's physics buffer.

The physics buffer is how CAM's processes hand each other fields that are not
arguments: the PBL scheme writes ``TKE``, the macrophysics reads it a stage
later.  It is the one host service the StatePool bridge cannot own -- CAM
allocates it, indexes it by a name registered during initialization, and
rotates two time samples -- so Python reaches it by handle instead: ask for a
field by the index CAM gave it, receive the address CAM already holds.

Nothing here copies or allocates.  A view returned by :meth:`PBuf.view` is
the same storage the Fortran process would have written, so a Python-driven
process and the original are reading and writing one buffer, not two.

Two shapes are served, because two are what CAM's physics asks for: a plain
``(pcols, n)`` field, and the ``(pcols, pver)`` plane of a time-rotated field
at the older sample -- ``pbuf_get_field(..., start=(/1,1,itim_old/),
kount=(/pcols,pver,1/))`` in the source.  Anything else is refused.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import PICAMConfigurationError
from freecam.pi_cam.tables import load_table

SYMBOL = "pycam_pbuf_field_v1"
SYMBOL_V2 = "pycam_pbuf_field_v2"

#: What the Fortran side returns.  Anything but ``0`` yields no view.
STATUS = {
    0: "ok",
    1: "the field was never registered in this configuration",
    2: "the chunk is not on this rank",
    3: "the physics buffer is not allocated yet",
    4: "the field is registered but has no storage for this chunk",
}


class PBufFieldAbsent(PICAMConfigurationError):
    """CAM never registered this field, so nothing can read or write it."""


@dataclass(frozen=True, slots=True)
class PBufField:
    """One named physics-buffer field and how it is stored.

    ``rank`` and ``dtype`` describe the pointer the Fortran routine declares
    for it; the default is the ``(pcols, n)`` double every macrophysics and
    radiation field is.  Microphysics also reads rank-1 doubles, rank-3
    doubles and one rank-1 integer, which go through the second accessor.
    """

    name: str
    index: int
    time_sliced: bool
    rank: int = 2
    dtype: str = "float64"

    @property
    def plain_plane(self) -> bool:
        """The shape the first accessor serves: a ``(pcols, n)`` double."""

        return self.rank == 2 and self.dtype == "float64"

    @property
    def registered(self) -> bool:
        # CAM leaves an unregistered index at -1; the six UNICON fields the
        # macrophysics guards with `if (idx > 0)` are exactly that.
        return self.index > 0


class PBuf:
    """The physics buffer of one rank, seen through CAM's own indices."""

    def __init__(self, library: Any, fields: Mapping[str, PBufField]) -> None:
        self.fields = dict(fields)
        entry = getattr(library, SYMBOL, None)
        if entry is None:
            raise PICAMConfigurationError(
                f"the loaded image exposes no {SYMBOL}; the device predates the "
                "physics-buffer handle"
            )
        entry.restype = ctypes.c_int
        entry.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
        ]
        self._entry = entry
        # The rank-aware accessor.  Images built before it existed lack it;
        # such an image still serves every (pcols, n) double, and refuses a
        # field of any other shape by name rather than misreading it.
        second = getattr(library, SYMBOL_V2, None)
        if second is not None:
            second.restype = ctypes.c_int
            second.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int64),
            ]
        self._entry_v2 = second
        self._views: dict[tuple[str, int], tuple[int, tuple[int, ...], np.ndarray]] = {}

    def __contains__(self, name: str) -> bool:
        field = self.fields.get(name)
        return field is not None and field.registered

    def view(self, name: str, chunk: int) -> np.ndarray:
        """A Fortran-ordered view of ``name`` on ``chunk``; never a copy."""

        try:
            field = self.fields[name]
        except KeyError as error:
            raise PICAMConfigurationError(
                f"{name!r} is not one of this process's declared physics-buffer "
                f"fields: {', '.join(sorted(self.fields))}"
            ) from error
        if not field.registered:
            raise PBufFieldAbsent(
                f"{name} has index {field.index}; {STATUS[1]}"
            )
        if not field.plain_plane:
            return self._view_any(field, chunk)
        pointer = ctypes.c_void_p()
        extents = (ctypes.c_int64 * 2)()
        status = self._entry(
            int(chunk), int(field.index), int(field.time_sliced),
            ctypes.byref(pointer), extents,
        )
        if status != 0:
            raise PICAMConfigurationError(
                f"physics buffer refused {name} on chunk {chunk}: "
                f"{STATUS.get(status, f'status {status}')}"
            )
        if not pointer.value:
            raise PICAMConfigurationError(
                f"physics buffer returned a null address for {name} on chunk {chunk}"
            )
        shape = (int(extents[0]), int(extents[1]))
        if shape[0] < 1 or shape[1] < 1:
            raise PICAMConfigurationError(
                f"physics buffer returned {shape} for {name} on chunk {chunk}"
            )
        # the same view while the buffer answers with the same storage; a
        # field the buffer re-allocates gets a fresh one
        hit = self._views.get((name, chunk))
        if hit is not None and hit[0] == pointer.value and hit[1] == shape:
            return hit[2]
        buffer = (ctypes.c_double * (shape[0] * shape[1])).from_address(pointer.value)
        view = np.ndarray(shape, dtype=np.float64, buffer=buffer, order="F")
        self._views[(name, chunk)] = (pointer.value, shape, view)
        return view

    def _view_any(self, field: PBufField, chunk: int) -> np.ndarray:
        """A field of any served rank and kind, through the second accessor."""

        if self._entry_v2 is None:
            raise PICAMConfigurationError(
                f"{field.name} is a rank-{field.rank} {field.dtype} field and the loaded "
                f"image exposes no {SYMBOL_V2}; it predates the rank-aware handle"
            )
        pointer = ctypes.c_void_p()
        ndims = ctypes.c_int()
        extents = (ctypes.c_int64 * 3)()
        is_integer = field.dtype in ("int32", "int64")
        status = self._entry_v2(
            int(chunk), int(field.index), int(field.time_sliced), int(field.rank),
            int(is_integer), ctypes.byref(pointer), ctypes.byref(ndims), extents,
        )
        if status != 0:
            raise PICAMConfigurationError(
                f"physics buffer refused {field.name} (rank {field.rank}, {field.dtype}) "
                f"on chunk {chunk}: {STATUS.get(status, f'status {status}')}"
            )
        if not pointer.value or ndims.value != field.rank:
            raise PICAMConfigurationError(
                f"physics buffer returned rank {ndims.value} for {field.name}, "
                f"declared rank {field.rank}, on chunk {chunk}"
            )
        shape = tuple(int(extents[i]) for i in range(ndims.value))
        if any(n < 1 for n in shape):
            raise PICAMConfigurationError(
                f"physics buffer returned {shape} for {field.name} on chunk {chunk}"
            )
        hit = self._views.get((field.name, chunk))
        if hit is not None and hit[0] == pointer.value and hit[1] == shape:
            return hit[2]
        count = int(np.prod(shape))
        ctype = ctypes.c_int32 if is_integer else ctypes.c_double
        buffer = (ctype * count).from_address(pointer.value)
        view = np.ndarray(shape, dtype=np.int32 if is_integer else np.float64,
                          buffer=buffer, order="F")
        self._views[(field.name, chunk)] = (pointer.value, shape, view)
        return view

    def verify(self, chunk: int, *, pcols: int, pver: int) -> dict[str, tuple[int, int]]:
        """Fetch every registered field once and check its shape.

        Called at attach so a wrong index or a changed pbuf registration is a
        refusal at setup, not a silently misread array in the middle of a run.
        """

        shapes: dict[str, tuple[int, int]] = {}
        wrong: list[str] = []
        for name, field in sorted(self.fields.items()):
            if not field.registered:
                continue
            view = self.view(name, chunk)
            shapes[name] = view.shape
            if view.shape[0] != pcols:
                wrong.append(f"{name}{view.shape}")
            elif field.rank >= 2 and view.shape[1] not in (pver, pver + 1):
                wrong.append(f"{name}{view.shape}")
        if wrong:
            raise PICAMConfigurationError(
                f"physics-buffer fields have unexpected shapes for pcols={pcols}, "
                f"pver={pver}: {', '.join(wrong)}"
            )
        return shapes


#: The macrophysics driver's own physics-buffer fields, with the two access
#: forms its source uses.  Names are CAM's registered names; the module
#: integer holding each index is ``<lowercase name>_idx`` in ``macrop_driver``,
#: except where noted.
MACROP_FIELDS = (
    # time-rotated: read at the older sample
    ("QCWAT", "qcwat_idx", True), ("TCWAT", "tcwat_idx", True),
    ("LCWAT", "lcwat_idx", True), ("ICCWAT", "iccwat_idx", True),
    ("NLWAT", "nlwat_idx", True), ("NIWAT", "niwat_idx", True),
    ("CC_T", "cc_t_idx", True), ("CC_qv", "cc_qv_idx", True),
    ("CC_ql", "cc_ql_idx", True), ("CC_qi", "cc_qi_idx", True),
    ("CC_nl", "cc_nl_idx", True), ("CC_ni", "cc_ni_idx", True),
    ("CC_qlst", "cc_qlst_idx", True),
    ("CLD", "cld_idx", True), ("AST", "ast_idx", True),
    ("AIST", "aist_idx", True), ("ALST", "alst_idx", True),
    ("QIST", "qist_idx", True), ("QLST", "qlst_idx", True),
    ("CONCLD", "concld_idx", True),
    # plain
    ("FICE", "fice_idx", False), ("CMELIQ", "cmeliq_idx", False),
    ("SHFRC", "shfrc_idx", False), ("NAAI", "naai_idx", False),
    # registered only under UNICON; guarded by `if (idx > 0)` in the source
    ("TKE", "tke_idx", False), ("QTL_FLX", "qtl_flx_idx", False),
    ("QTI_FLX", "qti_flx_idx", False), ("CMFR_DET", "cmfr_det_idx", False),
    ("QLR_DET", "qlr_det_idx", False), ("QIR_DET", "qir_det_idx", False),
)


def load_pbuf_table(path, indices: Mapping[str, int]) -> dict[str, PBufField]:
    """Bind a generated field table (``tools/generate_pi_cam_pbuf_table.py``)
    to the indices read from the image.

    The table names each field, the module integer holding its index, how the
    routine reads it (time-sliced or not), and the rank and kind of the
    pointer it declares for it.  ``indices`` maps the symbol to its value.
    """

    payload = load_table(path)
    missing = [row["symbol"] for row in payload["fields"] if row["symbol"] not in indices]
    if missing:
        raise PICAMConfigurationError(
            f"physics-buffer index variables were not read: {', '.join(missing)}")
    return {
        row["name"]: PBufField(row["name"], int(indices[row["symbol"]]),
                               bool(row["time_sliced"]), int(row.get("rank", 2)),
                               str(row.get("dtype", "float64")))
        for row in payload["fields"]
    }


def macrop_fields(indices: Mapping[str, int]) -> dict[str, PBufField]:
    """Bind the driver's field table to the indices read from its module."""

    missing = [symbol for _, symbol, _ in MACROP_FIELDS if symbol not in indices]
    if missing:
        raise PICAMConfigurationError(
            "macrop_driver index variables were not read: " + ", ".join(missing)
        )
    return {
        name: PBufField(name, int(indices[symbol]), sliced)
        for name, symbol, sliced in MACROP_FIELDS
    }


__all__ = ["MACROP_FIELDS", "PBuf", "PBufField", "PBufFieldAbsent", "STATUS", "SYMBOL_V2",
           "load_pbuf_table",
           "SYMBOL", "macrop_fields"]
