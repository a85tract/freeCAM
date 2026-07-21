"""Bit-preserving snapshots for Dask fan-out and MPI restart."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import numpy as np

from .clock import NoLeapClock
from .config import ModelConfig
from .errors import ConfigurationError, StateTransitionError
from .scheme_plan import KesslerSchemePlan
from .state import StatePool

if TYPE_CHECKING:
    from .driver import CAMDriver


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """One MPI rank's immutable Python-owned model state."""

    rank: int
    size: int
    config: Mapping[str, Any]
    dimensions: Mapping[str, int]
    arrays: Mapping[str, np.ndarray]
    pool_sealed: bool
    clock: Mapping[str, int]
    driver_state: str
    scheme_plan: Mapping[str, Any]
    last_phase: str | None
    last_scheme: str | None
    last_scheme_group: str | None
    native_calls: int

    @classmethod
    def capture(cls, driver: CAMDriver) -> "ModelSnapshot":
        if driver.pool is None or driver.clock is None:
            raise StateTransitionError("cannot snapshot an uninitialized model")
        if driver.state.value == "FINALIZED":
            raise StateTransitionError("cannot snapshot a finalized model")
        return cls(
            rank=int(driver.comm.rank),
            size=int(driver.comm.size),
            config=driver.config.as_dict(),
            dimensions=dict(driver.pool.dimensions),
            arrays=driver.pool.snapshot_arrays(readonly=True),
            pool_sealed=driver.pool.sealed,
            clock=asdict(driver.clock),
            driver_state=driver.state.value,
            scheme_plan=driver.scheme_plan.to_payload(),
            last_phase=driver._last_phase,
            last_scheme=driver._last_scheme,
            last_scheme_group=driver._last_scheme_group,
            native_calls=int(driver.backend.call_count),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "size": self.size,
            "config": dict(self.config),
            "dimensions": dict(self.dimensions),
            "pool_sealed": self.pool_sealed,
            "clock": dict(self.clock),
            "driver_state": self.driver_state,
            "scheme_plan": dict(self.scheme_plan),
            "last_phase": self.last_phase,
            "last_scheme": self.last_scheme,
            "last_scheme_group": self.last_scheme_group,
            "native_calls": self.native_calls,
            "array_names": sorted(self.arrays),
        }

    @classmethod
    def from_storage(
        cls,
        metadata: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
    ) -> "ModelSnapshot":
        expected = set(metadata["array_names"])
        if set(arrays) != expected:
            raise ConfigurationError(
                "checkpoint array inventory does not match its metadata"
            )
        immutable: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            copied = np.array(value, dtype=value.dtype, order="F", copy=True)
            copied.flags.writeable = False
            immutable[name] = copied
        return cls(
            rank=int(metadata["rank"]),
            size=int(metadata["size"]),
            config=dict(metadata["config"]),
            dimensions={
                name: int(value)
                for name, value in metadata["dimensions"].items()
            },
            arrays=immutable,
            pool_sealed=bool(metadata["pool_sealed"]),
            clock={name: int(value) for name, value in metadata["clock"].items()},
            driver_state=str(metadata["driver_state"]),
            scheme_plan=dict(metadata["scheme_plan"]),
            last_phase=metadata.get("last_phase"),
            last_scheme=metadata.get("last_scheme"),
            last_scheme_group=metadata.get("last_scheme_group"),
            native_calls=int(metadata.get("native_calls", 0)),
        )

    def new_pool(self) -> StatePool:
        """Create branch-private arrays with the same bit patterns."""

        pool = StatePool(self.dimensions)
        pool.restore_arrays(self.arrays)
        if self.pool_sealed:
            pool.seal_static()
        pool.validate(finite=True)
        return pool


@dataclass(frozen=True, slots=True)
class CheckpointBundle:
    """Immutable serialized checkpoint retained by a Dask Future."""

    files: tuple[tuple[str, bytes], ...]

    @classmethod
    def from_directory(cls, path: str | Path) -> "CheckpointBundle":
        root = Path(path)
        manifest = root / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"checkpoint manifest is absent: {manifest}")
        names = ["manifest.json"]
        payload = json.loads(manifest.read_text())
        names.extend(
            f"rank-{int(record['rank']):03d}.npz"
            for record in payload["ranks"]
        )
        return cls(tuple((name, (root / name).read_bytes()) for name in names))

    @property
    def nbytes(self) -> int:
        return sum(len(content) for _name, content in self.files)

    def materialize(self, path: str | Path) -> Path:
        root = Path(path)
        if root.exists():
            raise FileExistsError(f"refusing to replace checkpoint directory: {root}")
        root.mkdir(parents=True)
        for name, content in self.files:
            (root / name).write_bytes(content)
        return root


def restore_driver(
    snapshot: ModelSnapshot,
    *,
    run_dir: str | Path,
    comm: Any,
    kernel_library: str | Path | None = None,
    history_dir: str | Path | None = None,
    expected_config: ModelConfig | None = None,
) -> CAMDriver:
    """Restore a fresh driver without rerunning Python initialization."""

    from .driver import CAMDriver, DriverState

    if int(comm.rank) != snapshot.rank or int(comm.size) != snapshot.size:
        raise ConfigurationError(
            "checkpoint communicator mismatch: "
            f"snapshot rank/size={snapshot.rank}/{snapshot.size}, "
            f"runtime={comm.rank}/{comm.size}"
        )
    config = ModelConfig.from_mapping(snapshot.config)
    if expected_config is not None and config.as_dict() != expected_config.as_dict():
        raise ConfigurationError("checkpoint configuration differs from requested config")
    driver = CAMDriver(
        config,
        run_dir=run_dir,
        comm=comm,
        kernel_library=kernel_library,
        history_dir=history_dir,
        scheme_plan=KesslerSchemePlan.from_payload(snapshot.scheme_plan),
    )
    driver.pool = snapshot.new_pool()
    driver.clock = NoLeapClock(**snapshot.clock)
    driver.state = DriverState(snapshot.driver_state)
    driver._last_phase = snapshot.last_phase
    driver._last_scheme = snapshot.last_scheme
    driver._last_scheme_group = snapshot.last_scheme_group
    driver.backend.call_count = snapshot.native_calls
    return driver


def write_checkpoint(driver: CAMDriver, path: str | Path) -> Path:
    """Collectively save all rank-local state into one checkpoint directory."""

    root = Path(path).resolve()
    comm = driver.comm
    exists = root.exists() if comm.rank == 0 else None
    if comm.bcast(exists, root=0):
        raise FileExistsError(f"refusing to replace checkpoint directory: {root}")
    setup_failure: str | None = None
    if comm.rank == 0:
        try:
            root.mkdir(parents=True)
        except BaseException as exc:
            setup_failure = f"{type(exc).__name__}: {exc}"
    setup_failure = comm.bcast(setup_failure, root=0)
    if setup_failure:
        raise OSError(f"cannot create checkpoint {root}: {setup_failure}")

    snapshot = ModelSnapshot.capture(driver)
    filename = root / f"rank-{comm.rank:03d}.npz"
    temporary = root / f".rank-{comm.rank:03d}.npz.tmp"
    failure: str | None = None
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **snapshot.arrays)
        os.replace(temporary, filename)
    except BaseException as exc:
        failure = f"rank {comm.rank}: {type(exc).__name__}: {exc}"
    failures = comm.allgather(failure)
    messages = [message for message in failures if message]
    if messages:
        raise OSError("checkpoint rank write failed: " + "; ".join(messages))

    records = comm.gather(snapshot.metadata(), root=0)
    manifest_failure: str | None = None
    if comm.rank == 0:
        try:
            manifest = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "mpi_size": int(comm.size),
                "ranks": records,
            }
            temporary_manifest = root / ".manifest.json.tmp"
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True)
            )
            os.replace(temporary_manifest, root / "manifest.json")
        except BaseException as exc:
            manifest_failure = f"{type(exc).__name__}: {exc}"
    manifest_failure = comm.bcast(manifest_failure, root=0)
    if manifest_failure:
        raise OSError(f"checkpoint manifest write failed: {manifest_failure}")
    return root


def read_checkpoint(path: str | Path, comm: Any) -> ModelSnapshot:
    """Collectively load the calling rank's part of a checkpoint."""

    root = Path(path).resolve()
    payload: dict[str, Any] | None = None
    error: str | None = None
    if comm.rank == 0:
        try:
            payload = json.loads((root / "manifest.json").read_text())
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error:
        raise ConfigurationError(f"cannot read checkpoint {root}: {error}")
    payload = comm.bcast(payload, root=0)
    assert payload is not None
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ConfigurationError(
            f"unsupported checkpoint schema {payload.get('schema_version')!r}"
        )
    if int(payload.get("mpi_size", -1)) != int(comm.size):
        raise ConfigurationError(
            f"checkpoint requires {payload.get('mpi_size')} ranks, got {comm.size}"
        )
    records = {int(record["rank"]): record for record in payload["ranks"]}
    if set(records) != set(range(comm.size)):
        raise ConfigurationError("checkpoint rank inventory is incomplete")
    with np.load(root / f"rank-{comm.rank:03d}.npz", allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return ModelSnapshot.from_storage(records[comm.rank], arrays)
