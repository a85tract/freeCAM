"""Replace the CESM coupler with an explicit CAM boundary provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import BinaryIO, Callable, Mapping

import numpy as np

from freecam.core.fortran_runtime import prepare_fortran_runtime

from .errors import BoundaryReplayError
from .state import PICAMStatePool


def _array_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return sha256(memoryview(contiguous).cast("B")).hexdigest()


def _first_difference(candidate: np.ndarray, reference: np.ndarray) -> str:
    """Return one compact, reproducible element-level BFB diagnostic."""

    candidate_bytes = np.ascontiguousarray(candidate).view(np.uint8).reshape(-1)
    reference_bytes = np.ascontiguousarray(reference).view(np.uint8).reshape(-1)
    differing_bytes = np.flatnonzero(candidate_bytes != reference_bytes)
    if differing_bytes.size == 0:
        return "logical values differ but contiguous bytes match"
    itemsize = candidate.dtype.itemsize
    flat_index = int(differing_bytes[0]) // itemsize
    index = np.unravel_index(flat_index, candidate.shape, order="C")
    actual = np.asarray(candidate[index])
    expected = np.asarray(reference[index])
    return (
        f"first index {tuple(int(value) for value in index)}: "
        f"{actual.item()!r} [0x{actual.tobytes().hex()}] != "
        f"{expected.item()!r} [0x{expected.tobytes().hex()}]"
    )


class CAMBoundaryProvider(ABC):
    """One source of surface/coupler imports and one sink for CAM exports."""

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        del rank, size, config_fingerprint

    @abstractmethod
    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        """Populate the rank-local ``cam_in.*`` fields for one coupling step."""

    @abstractmethod
    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        """Consume or validate the rank-local ``cam_out.*`` fields."""

    def finalize(self) -> None:
        return None

    def has_fresh_import(self, step: int, rank: int) -> bool:
        """Whether source CAM called ``atm_import`` at this action boundary."""

        del step, rank
        return True


def prepare_cesm_online_run(
    seed_run: str | Path,
    destination: str | Path,
) -> Path:
    """Create the private run directory used by the exact online provider.

    A coupled CESM run directory contains both startup inputs and potentially
    gigabytes of history/restart output.  The provider needs the former only.
    This helper performs the same filtered copy used by the scientific
    validation jobs while keeping that machinery out of notebooks.
    """

    source = Path(seed_run).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not (source / "drv_in").is_file():
        raise FileNotFoundError(
            f"CESM online-provider seed lacks drv_in: {source}"
        )
    if not (source / "SEMapping.nc").is_file():
        raise FileNotFoundError(
            f"CESM online-provider seed lacks SEMapping.nc: {source}"
        )
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(
            f"CESM online-provider run directory is not empty: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    def ignore_outputs(directory: str, names: list[str]) -> set[str]:
        del directory
        ignored: set[str] = set()
        for name in names:
            if name == "SEMapping.nc":
                continue
            if (
                name == "timing"
                or name.startswith("rpointer.")
                or name.endswith(".bin")
                or ".log" in name
                or name.endswith(".nc")
            ):
                ignored.add(name)
        return ignored

    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=ignore_outputs,
    )
    (target / "timing" / "checkpoints").mkdir(parents=True, exist_ok=True)
    return target


@dataclass(frozen=True, slots=True)
class BoundaryManifest:
    rank_count: int
    step_count: int
    config_fingerprint: str | None = None
    schema_version: int = 1
    file_pattern: str = "step-{step:06d}/rank-{rank:04d}-{direction}.npz"
    storage: str = "per_step_v1"
    held_import_steps: tuple[int, ...] = ()

    @classmethod
    def from_path(cls, path: Path) -> "BoundaryManifest":
        payload = json.loads(path.read_text())
        if int(payload.get("schema_version", 1)) != 1:
            raise BoundaryReplayError("unsupported CAM boundary manifest schema")
        return cls(
            rank_count=int(payload["rank_count"]),
            step_count=int(payload["step_count"]),
            config_fingerprint=payload.get("config_fingerprint"),
            schema_version=1,
            file_pattern=str(
                payload.get(
                    "file_pattern",
                    "step-{step:06d}/rank-{rank:04d}-{direction}.npz",
                )
            ),
            storage=str(payload.get("storage", "per_step_v1")),
            held_import_steps=tuple(
                int(step) for step in payload.get("held_import_steps", ())
            ),
        )


@dataclass(slots=True)
class _NpyStepReader:
    """Read one contiguous step from a rank-local ``.npy`` archive."""

    path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    data_offset: int
    handle: BinaryIO

    @classmethod
    def open(cls, path: Path) -> "_NpyStepReader":
        inspected = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if inspected.ndim < 1:
                raise BoundaryReplayError(
                    f"boundary array has no step dimension: {path}"
                )
            if not inspected.flags.c_contiguous:
                raise BoundaryReplayError(
                    f"boundary array must store contiguous step records: {path}"
                )
            shape = tuple(int(value) for value in inspected.shape)
            dtype = np.dtype(inspected.dtype)
            data_offset = int(inspected.offset)
        finally:
            mapping = getattr(inspected, "_mmap", None)
            if mapping is not None:
                mapping.close()
        return cls(
            path=path,
            shape=shape,
            dtype=dtype,
            data_offset=data_offset,
            handle=path.open("rb", buffering=0),
        )

    def read(self, step: int) -> np.ndarray:
        step_shape = self.shape[1:]
        step_nbytes = int(np.prod(step_shape, dtype=np.int64)) * int(
            self.dtype.itemsize
        )
        payload = os.pread(
            self.handle.fileno(),
            step_nbytes,
            self.data_offset + int(step) * step_nbytes,
        )
        if len(payload) != step_nbytes:
            raise BoundaryReplayError(
                f"short boundary read for {self.path} step {step}: "
                f"{len(payload)} != {step_nbytes} bytes"
            )
        return np.frombuffer(payload, dtype=self.dtype).reshape(step_shape)

    def close(self) -> None:
        self.handle.close()


@dataclass(frozen=True, slots=True)
class OnlineBoundaryContext:
    """Read-only metadata passed to one rank-local online surface update."""

    step: int
    rank: int
    size: int
    model_step: int
    date: int
    seconds: int
    timestep_seconds: int


@dataclass(slots=True)
class OnlineBoundaryFields:
    """The writable next x2a state and read-only previous CAM export."""

    x2a: np.ndarray
    a2x: np.ndarray


class HeldSurfaceModel:
    """Keep the bootstrap surface state fixed while CAM advances.

    This is an explicit technical control, not a replacement for interactive
    CLM, CICE, DOCN, and coupler flux calculations.
    """

    def __call__(
        self,
        fields: OnlineBoundaryFields,
        context: OnlineBoundaryContext,
    ) -> None:
        del fields, context


_CESM_INITIALIZATION_ACTIONS = tuple(range(501, 533))
_CESM_FINALIZATION_ACTIONS = tuple(range(601, 611))
_CESM_PLAN_ACTIONS = (
    (101, None, None),
    (102, "lnd", "run"),
    (103, "lnd", "run"),
    (104, "ice", "run"),
    (105, "ice", "run"),
    (106, "rof", "run"),
    (107, "rof", "run"),
    (108, "ice", "run"),
    (109, "lnd", "run"),
    (110, "rof", "run"),
    (111, "ocn", "run"),
    (112, "ocn", "next"),
    (113, "ocn", "next"),
    (114, None, None),
    (115, None, None),
    (116, "lnd", "run"),
    (117, "lnd", "run"),
    (118, "rof", "run"),
    (119, "rof", "run"),
    (120, "ice", "run"),
    (121, "ice", "run"),
    (122, None, None),
    (123, "atm", "run"),
    (124, "atm", "run"),
    (125, "atm", "run"),
    (126, "atm", "run"),
    (127, "atm", "run"),
    (128, "restart", "alarm"),
    (129, None, None),
)
_CESM_ALARM_BITS = {
    ("atm", "run"): 0,
    ("lnd", "run"): 1,
    ("ice", "run"): 2,
    ("rof", "run"): 3,
    ("ocn", "run"): 4,
    ("ocn", "next"): 5,
    ("restart", "alarm"): 6,
}


@dataclass(slots=True)
class _CESMMCTBuffer:
    allocation_id: int
    scope: str
    field_names: tuple[str, ...]
    values: np.ndarray
    owner: object | None = None


_MCTAllocatorCallback = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int64,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int32),
)
_MCTReleaseCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
)

_FortranHeapAllocatorCallback = ctypes.CFUNCTYPE(
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int32),
)
_FortranHeapReleaseCallback = ctypes.CFUNCTYPE(
    None,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int32),
)


class _CESMMCTRegistry:
    """Own the live MCT arrays allocated by the original CESM kernels."""

    def __init__(self) -> None:
        self.scope = "runtime"
        self.buffers: dict[int, _CESMMCTBuffer] = {}
        self.addresses: dict[int, int] = {}
        self.error: BaseException | None = None
        self.allocations = 0
        self.releases = 0
        self.callback = _MCTAllocatorCallback(self._allocate)
        self.release_callback = _MCTReleaseCallback(self._release)

    @contextmanager
    def in_scope(self, scope: str):
        previous = self.scope
        self.scope = str(scope)
        try:
            yield
        finally:
            self.scope = previous

    def _allocate(
        self,
        kind: int,
        allocation_id: int,
        item_bytes: int,
        attribute_count: int,
        point_count: int,
        field_names: bytes | None,
        status: ctypes.POINTER(ctypes.c_int32),
    ) -> int | None:
        try:
            if int(kind) not in {1, 2} or int(item_bytes) not in {4, 8}:
                raise BoundaryReplayError(
                    f"unsupported CESM MCT storage kind={kind}, bytes={item_bytes}"
                )
            names = tuple(
                item.strip()
                for item in (field_names or b"").decode("utf-8").split(":")
                if item.strip()
            )
            if len(names) != int(attribute_count):
                raise BoundaryReplayError(
                    "CESM MCT allocator field-count mismatch: "
                    f"declared {attribute_count}, received {len(names)}"
                )
            dtype_kind = "i" if int(kind) == 1 else "f"
            values = np.empty(
                (int(attribute_count), int(point_count)),
                dtype=np.dtype(f"{dtype_kind}{int(item_bytes)}"),
                order="F",
            )
            allocation = int(allocation_id)
            if allocation in self.buffers:
                raise BoundaryReplayError(
                    f"duplicate CESM MCT allocation id {allocation}"
                )
            buffer = _CESMMCTBuffer(allocation, self.scope, names, values)
            self.buffers[allocation] = buffer
            self.addresses[int(values.ctypes.data)] = allocation
            self.allocations += 1
            status[0] = 0
            return int(values.ctypes.data)
        except BaseException as exc:
            self.error = exc
            traceback.print_exc(file=sys.stderr)
            status[0] = 1
            return None

    def _release(
        self,
        address: int | None,
        status: ctypes.POINTER(ctypes.c_int32),
    ) -> None:
        try:
            if address is None:
                raise BoundaryReplayError("CESM released a null MCT address")
            allocation = self.addresses.pop(int(address))
            del self.buffers[allocation]
            self.releases += 1
            status[0] = 0
        except BaseException as exc:
            self.error = exc
            status[0] = 1

    def raise_if_failed(self) -> None:
        if self.error is not None:
            error = self.error
            self.error = None
            raise BoundaryReplayError("CESM MCT allocation callback failed") from error

    def component_exchange(self, attributes: int) -> _CESMMCTBuffer:
        matches = tuple(
            buffer
            for buffer in self.buffers.values()
            if buffer.scope == "initialize:atm_initialize"
            and buffer.values.dtype == np.dtype("float64")
            and buffer.values.ndim == 2
            and buffer.values.shape[0] == int(attributes)
        )
        if len(matches) != 1:
            detail = tuple(
                (buffer.allocation_id, buffer.scope, buffer.values.shape)
                for buffer in self.buffers.values()
                if buffer.values.ndim == 2
                and buffer.values.shape[0] == int(attributes)
            )
            raise BoundaryReplayError(
                "cannot uniquely identify the ATM component MCT buffer with "
                f"{attributes} attributes; matches={detail}"
            )
        return matches[0]


@dataclass(slots=True)
class _CESMFortranHeapBuffer:
    allocation_id: int
    source_id: str
    scope: str
    storage: np.ndarray
    values: np.ndarray


class _CESMFortranHeapRegistry:
    """Own aligned storage requested by the instrumented Intel runtime.

    The original allocatable descriptors remain inside CESM.  Only their raw,
    rank-local backing bytes are supplied here so the online provider uses the
    same Python-owned allocation path as the previously validated coupled run.
    """

    def __init__(self) -> None:
        self.scope = "runtime"
        self.buffers: dict[int, _CESMFortranHeapBuffer] = {}
        self.addresses: dict[int, int] = {}
        self.error: BaseException | None = None
        self.allocations = 0
        self.releases = 0
        self.peak_bytes = 0
        self.live_bytes = 0
        self.callback = _FortranHeapAllocatorCallback(self._allocate)
        self.release_callback = _FortranHeapReleaseCallback(self._release)

    @contextmanager
    def in_scope(self, scope: str):
        previous = self.scope
        self.scope = str(scope)
        try:
            yield
        finally:
            self.scope = previous

    @staticmethod
    def _alignment(value: int) -> int:
        alignment = max(int(value), ctypes.sizeof(ctypes.c_void_p))
        if alignment & (alignment - 1):
            alignment = 1 << (alignment - 1).bit_length()
        return alignment

    def _allocate(
        self,
        source_id: bytes | None,
        allocation_id: int,
        byte_count: int,
        alignment: int,
        status: ctypes.POINTER(ctypes.c_int32),
    ) -> int | None:
        try:
            identifier = int(allocation_id)
            requested_count = int(byte_count)
            count = max(requested_count, 1)
            if identifier < 1 or identifier in self.buffers:
                raise BoundaryReplayError(
                    f"invalid or duplicate CESM heap allocation id {identifier}"
                )
            if requested_count < 0:
                raise BoundaryReplayError(
                    f"invalid CESM heap allocation size {requested_count}"
                )
            aligned = self._alignment(alignment)
            storage = np.empty(count + aligned - 1, dtype=np.uint8)
            offset = (-int(storage.ctypes.data)) % aligned
            values = storage[offset : offset + count]
            address = int(values.ctypes.data)
            buffer = _CESMFortranHeapBuffer(
                identifier,
                (source_id or b"").decode("utf-8", errors="replace"),
                self.scope,
                storage,
                values,
            )
            self.buffers[identifier] = buffer
            self.addresses[address] = identifier
            self.allocations += 1
            self.live_bytes += count
            self.peak_bytes = max(self.peak_bytes, self.live_bytes)
            status[0] = 0
            return address
        except BaseException as exc:
            self.error = exc
            traceback.print_exc(file=sys.stderr)
            status[0] = 1
            return None

    def _release(
        self,
        address: int | None,
        status: ctypes.POINTER(ctypes.c_int32),
    ) -> None:
        try:
            if address is None:
                raise BoundaryReplayError("CESM released a null heap address")
            identifier = self.addresses.pop(int(address))
            buffer = self.buffers.pop(identifier)
            self.live_bytes -= int(buffer.values.nbytes)
            self.releases += 1
            status[0] = 0
        except BaseException as exc:
            self.error = exc
            status[0] = 1

    def raise_if_failed(self) -> None:
        if self.error is not None:
            error = self.error
            self.error = None
            raise BoundaryReplayError("CESM heap allocation callback failed") from error


class CESMOnlineBoundaryProvider(CAMBoundaryProvider):
    """Run the original CESM surface components and coupler online.

    The complete CESM component/coupler image lives in the same 512 MPI
    processes as FreeCAM.  Python executes the original coupling actions up to
    ``component_run_begin``, actively queries the native x2a/a2x addresses,
    and builds zero-copy NumPy views over those rank-local MCT arrays.  FreeCAM
    writes a2x through that view, then Python explicitly advances the original
    ATM loop boundary and, once complete, invokes the remaining coupler
    kernels.  No shadow CAM is run and no Fortran routine calls back into
    Python.
    """

    def __init__(
        self,
        *,
        library: str | Path,
        run_dir: str | Path,
        verify_shadow_atmosphere: bool = False,
        python_owned_internal: bool = False,
        oracle: str | Path | None = None,
    ) -> None:
        self.library = Path(library).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        if verify_shadow_atmosphere:
            raise BoundaryReplayError(
                "shadow-atmosphere verification was removed with the shadow CAM"
            )
        self.verify_shadow_atmosphere = False
        if python_owned_internal:
            raise BoundaryReplayError(
                "reverse Fortran-to-Python allocation callbacks are no longer "
                "supported; coupled internal arrays remain native and Python "
                "actively queries zero-copy exchange views"
            )
        self.python_owned_internal = False
        self.oracle = (
            None if oracle is None else Path(oracle).expanduser().resolve()
        )
        self._rank: int | None = None
        self._size: int | None = None
        self._world: object | None = None
        self._native: ctypes.CDLL | None = None
        self._registry: _CESMMCTRegistry | None = None
        self._heap_registry: _CESMFortranHeapRegistry | None = None
        self._x2a: _CESMMCTBuffer | None = None
        self._a2x: _CESMMCTBuffer | None = None
        self._initial_x2a: np.ndarray | None = None
        self._initial_a2x: np.ndarray | None = None
        self._primed_a2x: np.ndarray | None = None
        self._oracle_x2a: np.ndarray | None = None
        self._oracle_a2x: np.ndarray | None = None
        self._next_import_step = 0
        self._last_export_step = -1
        self._active_coupling_step = False
        self._fresh_import = True
        self._alarm_mask = 0
        self._remaining_actions: list[int] = []
        self._coupling_steps = 0
        self._shadow_matches = 0
        self._oracle_import_matches = 0
        self._oracle_export_matches = 0
        self._finalized = False

    @classmethod
    def from_seed_run(
        cls,
        *,
        library: str | Path,
        seed_run: str | Path,
        run_dir: str | Path,
        verify_shadow_atmosphere: bool = False,
        python_owned_internal: bool = False,
        oracle: str | Path | None = None,
    ) -> "CESMOnlineBoundaryProvider":
        """Prepare a private CESM run directory and return an online provider.

        The seed contributes only configuration and input files.  History,
        restart, log, and timing output is deliberately not copied into the
        live provider directory.
        """

        prepared = prepare_cesm_online_run(seed_run, run_dir)
        return cls(
            library=library,
            run_dir=prepared,
            verify_shadow_atmosphere=verify_shadow_atmosphere,
            python_owned_internal=python_owned_internal,
            oracle=oracle,
        )

    @contextmanager
    def _provider_directory(self):
        previous = Path.cwd()
        os.chdir(self.run_dir)
        try:
            yield
        finally:
            os.chdir(previous)

    @staticmethod
    def _check(status: ctypes.c_int32, operation: str) -> None:
        if status.value:
            raise BoundaryReplayError(
                f"original CESM {operation} returned status {status.value}"
            )

    def _collective(self, operation: str, function: Callable[[], None]) -> None:
        assert self._world is not None
        local_error: str | None = None
        try:
            with self._provider_directory():
                function()
            if self._registry is not None:
                self._registry.raise_if_failed()
            if self._heap_registry is not None:
                self._heap_registry.raise_if_failed()
        except BaseException:
            local_error = traceback.format_exc()
        failure_count = int(self._world.allreduce(int(local_error is not None)))
        if not failure_count:
            return
        errors = self._world.allgather(local_error)
        first = next((item for item in errors if item), "unknown error")
        raise BoundaryReplayError(
            f"original CESM online provider failed during {operation}:\n{first}"
        )

    def _call_initialize_action(self, action_id: int) -> None:
        assert self._native is not None
        assert self._world is not None
        status = ctypes.c_int32()
        self._native.pycesm_full_initialize_action_v1(
            ctypes.c_int32(action_id),
            ctypes.c_int32(self._world.py2f()),
            ctypes.byref(status),
        )
        self._check(status, f"initialization action {action_id}")

    def _call_plan_action(self, action_id: int) -> None:
        assert self._native is not None
        status = ctypes.c_int32()
        self._native.pycesm_full_action_v1(
            ctypes.c_int32(action_id), ctypes.byref(status)
        )
        self._check(status, f"coupling action {action_id}")

    def _call_nested_action(self, action_id: int) -> bool:
        assert self._native is not None
        complete = ctypes.c_int32()
        status = ctypes.c_int32()
        self._native.pycesm_full_nested_action_v1(
            ctypes.c_int32(action_id),
            ctypes.byref(complete),
            ctypes.byref(status),
        )
        self._check(status, f"ATM nested action {action_id}")
        return bool(complete.value)

    def _call_external_atm_iteration(self) -> bool:
        assert self._native is not None
        complete = ctypes.c_int32()
        status = ctypes.c_int32()
        self._native.pycesm_full_external_atm_iteration_v1(
            ctypes.byref(complete), ctypes.byref(status)
        )
        self._check(status, "external ATM iteration")
        return bool(complete.value)

    def _query_exchange_buffer(
        self,
        exchange_id: int,
        *,
        expected_attributes: int,
        label: str,
    ) -> _CESMMCTBuffer:
        """Actively query one native MCT buffer and expose a zero-copy view."""

        assert self._native is not None
        address = ctypes.c_void_p()
        nattr = ctypes.c_int32()
        npoint = ctypes.c_int32()
        status = ctypes.c_int32()
        self._native.pycesm_full_exchange_buffer_v1(
            ctypes.c_int32(exchange_id),
            ctypes.byref(address),
            ctypes.byref(nattr),
            ctypes.byref(npoint),
            ctypes.byref(status),
        )
        self._check(status, f"query {label} exchange buffer")
        if not address.value or nattr.value <= 0 or npoint.value <= 0:
            raise BoundaryReplayError(
                f"native CESM returned an invalid {label} exchange buffer"
            )
        if nattr.value != expected_attributes:
            raise BoundaryReplayError(
                f"native CESM {label} has {nattr.value} attributes; "
                f"expected {expected_attributes}"
            )
        storage_type = ctypes.c_double * (nattr.value * npoint.value)
        storage = storage_type.from_address(int(address.value))
        values = np.ctypeslib.as_array(storage).reshape(
            (nattr.value, npoint.value), order="F"
        )
        return _CESMMCTBuffer(
            allocation_id=-int(exchange_id),
            scope="native:atm_exchange",
            field_names=tuple(
                f"{label}_{index}" for index in range(nattr.value)
            ),
            values=values,
            owner=storage,
        )

    @staticmethod
    def _action_enabled(
        action_id: int,
        alarm: str | None,
        kind: str | None,
        mask: int,
    ) -> bool:
        del action_id
        if alarm is None:
            return True
        bit = _CESM_ALARM_BITS[(alarm, str(kind))]
        return bool(int(mask) & (1 << bit))

    def _load_oracle(self, rank: int) -> None:
        if self.oracle is None:
            return
        manifest_path = self.oracle / "manifest.json"
        if not manifest_path.is_file():
            raise BoundaryReplayError(f"missing exact-provider oracle {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("rank_count", -1)) != self._size:
            raise BoundaryReplayError("exact-provider oracle rank count differs")
        pattern = str(manifest.get("file_pattern", "rank-{rank:04d}.npz"))
        path = self.oracle / pattern.format(rank=rank)
        with np.load(path, allow_pickle=False) as payload:
            self._oracle_x2a = np.asarray(payload["x2a_rattr"]).copy()
            self._oracle_a2x = np.asarray(payload["a2x_rattr"]).copy()

    def _verify_oracle(self, direction: str, step: int, values: np.ndarray) -> None:
        reference = self._oracle_x2a if direction == "import" else self._oracle_a2x
        if reference is None:
            return
        if not 0 <= step < reference.shape[0]:
            raise BoundaryReplayError(
                f"exact-provider oracle lacks {direction} step {step}"
            )
        expected = reference[step]
        if not np.array_equal(values, expected):
            raise BoundaryReplayError(
                f"online CESM {direction} differs from oracle at step {step}: "
                + _first_difference(values, expected)
            )
        if direction == "import":
            self._oracle_import_matches += 1
        else:
            self._oracle_export_matches += 1

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        del config_fingerprint
        if size != 512:
            raise BoundaryReplayError(
                f"the admitted PI-atm CESM provider requires 512 ranks, got {size}"
            )
        if not self.library.is_file():
            raise BoundaryReplayError(f"missing coupled CESM library {self.library}")
        if not (self.run_dir / "drv_in").is_file():
            raise BoundaryReplayError(
                f"coupled CESM provider run directory lacks drv_in: {self.run_dir}"
            )
        from mpi4py import MPI

        self._rank = int(rank)
        self._size = int(size)
        self._world = MPI.COMM_WORLD
        self._load_oracle(rank)
        prepare_fortran_runtime()
        # Keep the coupled image local.  RTLD_DEEPBIND is deliberately not used:
        # this historical CESM image also contains MPI symbols, and deep binding
        # would select a second, uninitialized MPI runtime inside this process.
        self._native = ctypes.CDLL(str(self.library), mode=ctypes.RTLD_LOCAL)
        self._registry = None
        self._heap_registry = None
        self._native.pycesm_full_initialize_action_v1.argtypes = (
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        )
        self._native.pycesm_full_initialize_action_v1.restype = None
        self._native.pycesm_full_action_v1.argtypes = (
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        )
        self._native.pycesm_full_action_v1.restype = None
        self._native.pycesm_full_nested_action_v1.argtypes = (
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        )
        self._native.pycesm_full_nested_action_v1.restype = None
        try:
            exchange_buffer = self._native.pycesm_full_exchange_buffer_v1
            external_iteration = (
                self._native.pycesm_full_external_atm_iteration_v1
            )
        except AttributeError as exc:
            raise BoundaryReplayError(
                "coupled CESM library lacks the callback-free external ATM ABI"
            ) from exc
        exchange_buffer.argtypes = (
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        )
        exchange_buffer.restype = None
        external_iteration.argtypes = (
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        )
        external_iteration.restype = None
        self._native.pycesm_full_step_begin_v1.argtypes = (
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        )
        self._native.pycesm_full_step_begin_v1.restype = None
        self._native.pycesm_full_step_end_v1.argtypes = (
            ctypes.POINTER(ctypes.c_int32),
        )
        self._native.pycesm_full_step_end_v1.restype = None
        self._native.pycesm_full_finalize_action_v1.argtypes = (
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        )
        self._native.pycesm_full_finalize_action_v1.restype = None

        for action_id in _CESM_INITIALIZATION_ACTIONS:
            self._collective(
                f"initialization action {action_id}",
                lambda action_id=action_id: self._call_initialize_action(action_id),
            )
            if action_id == 512:
                self._x2a = self._query_exchange_buffer(
                    1, expected_attributes=41, label="x2a"
                )
                self._a2x = self._query_exchange_buffer(
                    2, expected_attributes=50, label="a2x"
                )
                self._initial_x2a = self._x2a.values.copy(order="F")
                self._initial_a2x = self._a2x.values.copy(order="F")
            elif action_id == 529:
                assert self._a2x is not None
                self._primed_a2x = self._a2x.values.copy(order="F")
        assert self._x2a is not None and self._a2x is not None

    def _begin_coupling_step(self) -> None:
        assert self._native is not None
        native_step = ctypes.c_int32()
        ymd = ctypes.c_int32()
        seconds = ctypes.c_int32()
        mask = ctypes.c_int32()

        def begin() -> None:
            status = ctypes.c_int32()
            self._native.pycesm_full_step_begin_v1(
                ctypes.byref(native_step),
                ctypes.byref(ymd),
                ctypes.byref(seconds),
                ctypes.byref(mask),
                ctypes.byref(status),
            )
            self._check(status, "step begin")

        self._collective("step begin", begin)
        self._alarm_mask = int(mask.value)
        actions = [
            action_id
            for action_id, alarm, kind in _CESM_PLAN_ACTIONS
            if self._action_enabled(action_id, alarm, kind, self._alarm_mask)
        ]
        if 125 not in actions:
            raise BoundaryReplayError("PI-atm provider ATM alarm is unexpectedly off")
        split = actions.index(125)
        for action_id in actions[:split]:
            self._collective(
                f"coupling action {action_id}",
                lambda action_id=action_id: self._call_plan_action(action_id),
            )
        self._collective(
            "ATM component begin", lambda: self._call_nested_action(201)
        )
        self._collective("ATM import begin", lambda: self._call_nested_action(202))
        self._remaining_actions = actions[split + 1 :]
        self._active_coupling_step = True
        self._fresh_import = True

    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if rank != self._rank or self._x2a is None:
            raise BoundaryReplayError("CESM online provider is not initialized")
        if step != self._next_import_step:
            raise BoundaryReplayError(
                f"CESM online provider expected import {self._next_import_step}, got {step}"
            )
        if step == 0:
            assert self._initial_x2a is not None
            values = self._initial_x2a
            self._fresh_import = True
        elif step == 1:
            values = self._x2a.values
            self._fresh_import = True
        else:
            if not self._active_coupling_step:
                self._begin_coupling_step()
                self._fresh_import = True
            else:
                # The source ATM component imports x2a once, outside its
                # internal do-while loop.  A continuing loop reuses the same
                # already-imported CAM boundary state.
                self._fresh_import = False
            values = self._x2a.values
        self._verify_oracle("import", step, values)
        pool.ensure_from_array(
            "cam_in.x2a_rattr", values, category="boundary_import"
        )
        if "cam_out.a2x_rattr" not in pool:
            assert self._a2x is not None
            pool.ensure_from_array(
                "cam_out.a2x_rattr",
                np.zeros_like(self._a2x.values, order="F"),
                category="boundary_export",
            )
        self._next_import_step += 1

    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if rank != self._rank or self._a2x is None:
            raise BoundaryReplayError("CESM online provider is not initialized")
        if step != self._next_import_step - 1:
            raise BoundaryReplayError(
                f"CESM online provider export {step} does not follow its import"
            )
        values = np.asarray(pool["cam_out.a2x_rattr"])
        self._verify_oracle("export", step, values)
        self._a2x.values[...] = values
        if step < 2:
            self._fresh_import = True
            self._last_export_step = int(step)
            return
        if not self._active_coupling_step:
            raise BoundaryReplayError("CAM exported without an active CESM step")
        complete = False

        def iterate() -> None:
            nonlocal complete
            complete = self._call_external_atm_iteration()

        self._collective("external ATM iteration", iterate)
        assert self._world is not None
        complete_count = int(self._world.allreduce(int(complete)))
        if complete_count not in (0, self._size):
            raise BoundaryReplayError(
                "external ATM loop completion differs across MPI ranks"
            )
        complete = complete_count == self._size
        if complete:
            self._collective("ATM component end", lambda: self._call_nested_action(210))
            for action_id in self._remaining_actions:
                self._collective(
                    f"coupling action {action_id}",
                    lambda action_id=action_id: self._call_plan_action(action_id),
                )

            def end() -> None:
                assert self._native is not None
                status = ctypes.c_int32()
                self._native.pycesm_full_step_end_v1(ctypes.byref(status))
                self._check(status, "step end")

            self._collective("step end", end)
            self._active_coupling_step = False
            self._remaining_actions = []
            self._coupling_steps += 1
        self._last_export_step = int(step)

    def has_fresh_import(self, step: int, rank: int) -> bool:
        if rank != self._rank or step != self._next_import_step - 1:
            raise BoundaryReplayError("CESM online import schedule is inconsistent")
        return self._fresh_import

    def finalize(self) -> None:
        if self._finalized or self._native is None:
            return
        if self._active_coupling_step:
            raise BoundaryReplayError(
                "cannot finalize CESM provider in the middle of a coupling step"
            )
        for action_id in _CESM_FINALIZATION_ACTIONS:
            def invoke(action_id: int = action_id) -> None:
                assert self._native is not None
                status = ctypes.c_int32()
                self._native.pycesm_full_finalize_action_v1(
                    ctypes.c_int32(action_id), ctypes.byref(status)
                )
                self._check(status, f"finalization action {action_id}")

            self._collective(f"finalization action {action_id}", invoke)
        self._finalized = True

    @property
    def diagnostics(self) -> Mapping[str, object]:
        return {
            "provider": type(self).__name__,
            "library": str(self.library),
            "run_dir": str(self.run_dir),
            "imports": self._next_import_step,
            "exports": self._last_export_step + 1,
            "coupling_steps": self._coupling_steps,
            "shadow_a2x_bfb_steps": self._shadow_matches,
            "shadow_atmosphere": False,
            "oracle_x2a_bfb_steps": self._oracle_import_matches,
            "oracle_a2x_bfb_steps": self._oracle_export_matches,
            "active_coupling_step": self._active_coupling_step,
            "mct_live_buffers": (
                0 if self._registry is None else len(self._registry.buffers)
            ),
            "python_owned_internal": self.python_owned_internal,
            "reverse_allocator_callbacks": False,
            "exchange_storage": "native-mct-zero-copy-view",
            "fortran_heap_live_allocations": (
                0
                if self._heap_registry is None
                else len(self._heap_registry.buffers)
            ),
            "fortran_heap_peak_bytes": (
                0 if self._heap_registry is None else self._heap_registry.peak_bytes
            ),
        }


class OnlineBoundaryProvider(CAMBoundaryProvider):
    """Generate x2a in memory from the previous rank-local CAM export.

    Only one bootstrap x2a/a2x state is read for each rank.  Every later x2a
    state is generated by ``update(fields, context)``.  The callback may edit
    ``fields.x2a`` in place; ``fields.a2x`` is a read-only view of the previous
    CAM export.  It must return ``None``.
    """

    def __init__(
        self,
        bootstrap: str | Path,
        update: Callable[[OnlineBoundaryFields, OnlineBoundaryContext], None],
        *,
        require_finite: bool = True,
    ) -> None:
        if not callable(update):
            raise TypeError("online boundary update must be callable")
        self.bootstrap = Path(bootstrap).expanduser().resolve()
        self.update = update
        self.require_finite = bool(require_finite)
        manifest_path = self.bootstrap / "manifest.json"
        if not manifest_path.is_file():
            raise BoundaryReplayError(
                f"missing online boundary bootstrap manifest {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("schema_version", 0)) != 1:
            raise BoundaryReplayError("unsupported online boundary bootstrap schema")
        if manifest.get("storage") != "rank_bootstrap_v1":
            raise BoundaryReplayError(
                "online boundary bootstrap must use rank_bootstrap_v1 storage"
            )
        self.rank_count = int(manifest["rank_count"])
        self.file_pattern = str(
            manifest.get("file_pattern", "rank-{rank:04d}.npz")
        )
        self.source_config_fingerprint = manifest.get("source_config_fingerprint")
        self._rank: int | None = None
        self._size: int | None = None
        self._current_x2a: np.ndarray | None = None
        self._a2x_template: np.ndarray | None = None
        self._previous_a2x: np.ndarray | None = None
        self._next_import_step = 0
        self._last_export_step = -1
        self._generated_imports = 0

    @classmethod
    def held(
        cls,
        bootstrap: str | Path,
        *,
        require_finite: bool = True,
    ) -> "OnlineBoundaryProvider":
        """Construct the explicit fixed-surface technical control."""

        return cls(
            bootstrap,
            HeldSurfaceModel(),
            require_finite=require_finite,
        )

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        del config_fingerprint
        if size != self.rank_count:
            raise BoundaryReplayError(
                f"online bootstrap has {self.rank_count} ranks, runtime has {size}"
            )
        if not 0 <= rank < size:
            raise BoundaryReplayError(f"invalid runtime rank {rank}")
        path = self.bootstrap / self.file_pattern.format(rank=rank)
        if not path.is_file():
            raise BoundaryReplayError(f"missing online boundary bootstrap {path}")
        with np.load(path, allow_pickle=False) as payload:
            try:
                x2a = np.asarray(payload["x2a_rattr"], dtype=np.float64)
                a2x = np.asarray(payload["a2x_rattr"], dtype=np.float64)
            except KeyError as exc:
                raise BoundaryReplayError(
                    f"online boundary bootstrap {path} lacks {exc.args[0]!r}"
                ) from exc
        if x2a.ndim != 2 or a2x.ndim != 2:
            raise BoundaryReplayError(
                f"online boundary bootstrap {path} must contain two rank-2 arrays"
            )
        # The original MCT a2x vector may contain non-finite sentinels in
        # inactive rows. It is opaque feedback to the surface provider; only
        # x2a, which CAM will consume, is required to be finite.
        if self.require_finite and not np.isfinite(x2a).all():
            raise BoundaryReplayError(
                f"online boundary bootstrap {path} contains non-finite x2a"
            )
        self._rank = int(rank)
        self._size = int(size)
        self._current_x2a = x2a.copy(order="F")
        self._a2x_template = a2x.copy(order="F")
        self._previous_a2x = None
        self._next_import_step = 0
        self._last_export_step = -1
        self._generated_imports = 0

    @staticmethod
    def _scalar(pool: PICAMStatePool, name: str, default: int) -> int:
        try:
            return int(np.asarray(pool[name]).item())
        except KeyError:
            return int(default)

    def _context(
        self,
        step: int,
        pool: PICAMStatePool,
    ) -> OnlineBoundaryContext:
        assert self._rank is not None
        assert self._size is not None
        return OnlineBoundaryContext(
            step=int(step),
            rank=self._rank,
            size=self._size,
            model_step=self._scalar(pool, "model_step", step),
            date=self._scalar(pool, "current_date", 0),
            seconds=self._scalar(pool, "current_seconds_of_day", 0),
            timestep_seconds=self._scalar(pool, "model_timestep", 0),
        )

    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if self._rank != rank or self._current_x2a is None:
            raise BoundaryReplayError(
                "online boundary provider is not initialized for this rank"
            )
        if step != self._next_import_step:
            raise BoundaryReplayError(
                "online boundary expected import step "
                f"{self._next_import_step}, got {step}"
            )
        if step > 0:
            if self._last_export_step != step - 1 or self._previous_a2x is None:
                raise BoundaryReplayError(
                    f"online boundary step {step} has no preceding CAM export"
                )
            candidate = self._current_x2a.copy(order="F")
            previous = self._previous_a2x.view()
            previous.flags.writeable = False
            result = self.update(
                OnlineBoundaryFields(x2a=candidate, a2x=previous),
                self._context(step, pool),
            )
            if result is not None:
                raise BoundaryReplayError(
                    "online boundary update must edit fields.x2a and return None"
                )
            if candidate.shape != self._current_x2a.shape:
                raise BoundaryReplayError("online boundary update changed x2a shape")
            if candidate.dtype != self._current_x2a.dtype:
                raise BoundaryReplayError("online boundary update changed x2a dtype")
            if self.require_finite and not np.isfinite(candidate).all():
                raise BoundaryReplayError(
                    f"online boundary update produced non-finite x2a at step {step}"
                )
            self._current_x2a = candidate
            self._generated_imports += 1
        pool.ensure_from_array(
            "cam_in.x2a_rattr",
            self._current_x2a,
            category="boundary_import",
        )
        if "cam_out.a2x_rattr" not in pool:
            assert self._a2x_template is not None
            pool.ensure_from_array(
                "cam_out.a2x_rattr",
                np.zeros_like(self._a2x_template, order="F"),
                category="boundary_export",
            )
        self._next_import_step += 1

    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if self._rank != rank:
            raise BoundaryReplayError(
                "online boundary provider is not initialized for this rank"
            )
        if step != self._next_import_step - 1:
            raise BoundaryReplayError(
                f"online boundary export step {step} does not follow its import"
            )
        try:
            values = pool["cam_out.a2x_rattr"]
        except KeyError as exc:
            raise BoundaryReplayError("CAM did not provide cam_out.a2x_rattr") from exc
        assert self._a2x_template is not None
        if (
            values.shape != self._a2x_template.shape
            or values.dtype != self._a2x_template.dtype
        ):
            raise BoundaryReplayError(
                "CAM export shape or dtype changed during online coupling"
            )
        self._previous_a2x = np.asarray(values).copy(order="F")
        self._last_export_step = int(step)

    def has_fresh_import(self, step: int, rank: int) -> bool:
        if self._rank != rank:
            raise BoundaryReplayError(
                "online boundary provider is not initialized for this rank"
            )
        if step != self._next_import_step - 1:
            raise BoundaryReplayError(
                f"online boundary schedule is inconsistent at step {step}"
            )
        return True

    def finalize(self) -> None:
        self._current_x2a = None
        self._a2x_template = None
        self._previous_a2x = None

    @property
    def diagnostics(self) -> Mapping[str, object]:
        return {
            "provider": type(self).__name__,
            "surface_model": type(self.update).__name__,
            "bootstrap": str(self.bootstrap),
            "imports": self._next_import_step,
            "generated_imports": self._generated_imports,
            "exports": self._last_export_step + 1,
        }


class ReplayBoundaryProvider(CAMBoundaryProvider):
    """Strictly replay per-step, per-rank imports captured from PI-atm."""

    def __init__(self, root: str | Path, *, verify_exports: bool = True) -> None:
        self.root = Path(root)
        self.verify_exports = bool(verify_exports)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise BoundaryReplayError(f"missing boundary manifest {manifest_path}")
        self.manifest = BoundaryManifest.from_path(manifest_path)
        self._rank: int | None = None
        self._bundle: dict[str, np.ndarray | _NpyStepReader] | None = None

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        if size != self.manifest.rank_count:
            raise BoundaryReplayError(
                f"boundary capture has {self.manifest.rank_count} ranks, runtime has {size}"
            )
        if self.manifest.config_fingerprint not in {None, config_fingerprint}:
            raise BoundaryReplayError("boundary capture belongs to a different config")
        if not 0 <= rank < size:
            raise BoundaryReplayError(f"invalid runtime rank {rank}")
        self._rank = rank
        if self.manifest.storage in {
            "rank_bundle_v1",
            "rank_memmap_v1",
            "rank_pread_v1",
        }:
            if self.manifest.storage == "rank_bundle_v1":
                path = self.root / self.manifest.file_pattern.format(rank=rank)
                if not path.is_file():
                    raise BoundaryReplayError(f"missing boundary rank bundle {path}")
                self._bundle = self._load(path)
            else:
                paths = {
                    "x2a_rattr": self.root
                    / self.manifest.file_pattern.format(
                        rank=rank, direction="import"
                    ),
                    "a2x_rattr": self.root
                    / self.manifest.file_pattern.format(
                        rank=rank, direction="export"
                    ),
                }
                missing = [path for path in paths.values() if not path.is_file()]
                if missing:
                    raise BoundaryReplayError(
                        f"missing boundary rank memory map {missing[0]}"
                    )
                self._bundle = {
                    name: _NpyStepReader.open(path) for name, path in paths.items()
                }
            expected = self.manifest.step_count
            if any(values.shape[0] != expected for values in self._bundle.values()):
                raise BoundaryReplayError(
                    f"boundary rank bundle does not contain {expected} steps"
                )
        elif self.manifest.storage != "per_step_v1":
            raise BoundaryReplayError(
                f"unsupported boundary storage {self.manifest.storage!r}"
            )

    def _path(self, step: int, rank: int, direction: str) -> Path:
        if not 0 <= step < self.manifest.step_count:
            raise BoundaryReplayError(
                f"boundary step {step} is outside 0..{self.manifest.step_count - 1}"
            )
        path = self.root / self.manifest.file_pattern.format(
            step=step, rank=rank, direction=direction
        )
        if not path.is_file():
            raise BoundaryReplayError(f"missing boundary payload {path}")
        return path

    @staticmethod
    def _load(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            return {name: payload[name].copy(order="F") for name in payload.files}

    def _fields(self, step: int, rank: int, direction: str) -> dict[str, np.ndarray]:
        if not 0 <= step < self.manifest.step_count:
            raise BoundaryReplayError(
                f"boundary step {step} is outside 0..{self.manifest.step_count - 1}"
            )
        if self.manifest.storage in {
            "rank_bundle_v1",
            "rank_memmap_v1",
            "rank_pread_v1",
        }:
            if self._bundle is None or self._rank != rank:
                raise BoundaryReplayError("boundary rank bundle is not initialized")
            name = "x2a_rattr" if direction == "import" else "a2x_rattr"
            try:
                stored = self._bundle[name]
            except KeyError as exc:
                raise BoundaryReplayError(
                    f"boundary rank bundle lacks {name!r}"
                ) from exc
            values = (
                stored.read(step)
                if isinstance(stored, _NpyStepReader)
                else stored[step]
            )
            return {name: values}
        return self._load(self._path(step, rank, direction))

    def finalize(self) -> None:
        if self._bundle is not None:
            for values in self._bundle.values():
                if isinstance(values, _NpyStepReader):
                    values.close()
                else:
                    mapping = getattr(values, "_mmap", None)
                    if mapping is not None:
                        mapping.close()
        self._bundle = None
        self._rank = None

    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if self._rank != rank:
            raise BoundaryReplayError("boundary provider was initialized for another rank")
        fields = self._fields(step, rank, "import")
        for name, values in fields.items():
            canonical = name if name.startswith("cam_in.") else f"cam_in.{name}"
            pool.ensure_from_array(canonical, values, category="boundary_import")
        # Allocate the candidate export array from the oracle contract before
        # the native atm_export kernel writes it.  Its reference bytes are not
        # copied, so the later comparison still detects every differing bit.
        expected_exports = self._fields(step, rank, "export")
        for name, values in expected_exports.items():
            canonical = name if name.startswith("cam_out.") else f"cam_out.{name}"
            if canonical not in pool:
                pool.ensure_from_array(
                    canonical,
                    np.zeros_like(values, order="F"),
                    category="boundary_export",
                )

    def has_fresh_import(self, step: int, rank: int) -> bool:
        if self._rank != rank:
            raise BoundaryReplayError("boundary provider was initialized for another rank")
        if not 0 <= step < self.manifest.step_count:
            raise BoundaryReplayError(
                f"boundary step {step} is outside 0..{self.manifest.step_count - 1}"
            )
        return step not in self.manifest.held_import_steps

    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        if not self.verify_exports:
            return
        expected = self._fields(step, rank, "export")
        differences: list[str] = []
        for name, reference in expected.items():
            canonical = name if name.startswith("cam_out.") else f"cam_out.{name}"
            try:
                candidate = pool[canonical]
            except KeyError:
                differences.append(f"{canonical}: missing")
                continue
            if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
                differences.append(
                    f"{canonical}: {candidate.shape}/{candidate.dtype} != "
                    f"{reference.shape}/{reference.dtype}"
                )
            elif not np.array_equal(candidate, reference):
                differences.append(
                    f"{canonical}: {_array_hash(candidate)} != {_array_hash(reference)}; "
                    + _first_difference(candidate, reference)
                )
        if differences:
            raise BoundaryReplayError(
                f"rank {rank} step {step} CAM export is not bitwise identical: "
                + "; ".join(differences[:8])
            )


class InMemoryBoundaryProvider(CAMBoundaryProvider):
    """Programmatic boundary provider for a future coupler or experiment."""

    def __init__(
        self,
        imports: Mapping[tuple[int, int], Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        self.imports = {
            (int(step), int(rank)): {
                name: np.asfortranarray(values).copy(order="F")
                for name, values in fields.items()
            }
            for (step, rank), fields in (imports or {}).items()
        }
        self.exports: dict[tuple[int, int], dict[str, np.ndarray]] = {}
        self._size: int | None = None

    def initialize(self, *, rank: int, size: int, config_fingerprint: str) -> None:
        del rank, config_fingerprint
        self._size = size

    def set_imports(
        self, step: int, rank: int, fields: Mapping[str, np.ndarray]
    ) -> None:
        self.imports[(int(step), int(rank))] = {
            name: np.asfortranarray(values).copy(order="F")
            for name, values in fields.items()
        }

    def import_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        try:
            fields = self.imports[(step, rank)]
        except KeyError as exc:
            raise BoundaryReplayError(
                f"no in-memory CAM imports for step {step}, rank {rank}"
            ) from exc
        for name, values in fields.items():
            canonical = name if name.startswith("cam_in.") else f"cam_in.{name}"
            pool.ensure_from_array(canonical, values, category="boundary_import")

    def export_fields(self, step: int, rank: int, pool: PICAMStatePool) -> None:
        self.exports[(step, rank)] = {
            name: values.copy(order="F")
            for name, values in pool.items()
            if name.startswith("cam_out.")
        }


def write_boundary_payload(
    root: str | Path,
    *,
    step: int,
    rank: int,
    direction: str,
    fields: Mapping[str, np.ndarray],
) -> Path:
    """Write one atomic capture payload used by source instrumentation."""

    if direction not in {"import", "export"}:
        raise BoundaryReplayError("direction must be import or export")
    root_path = Path(root)
    destination = root_path / f"step-{step:06d}" / f"rank-{rank:04d}-{direction}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **{name: np.asfortranarray(value) for name, value in fields.items()})
    temporary.replace(destination)
    return destination
