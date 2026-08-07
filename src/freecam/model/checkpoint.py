"""Bit-preserving snapshots for Dask fan-out and MPI restart."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from .clock import NoLeapClock
from .ccpp_suite import CCPPSuitePlan
from .config import ModelConfig
from .contracts import (
    FieldContract,
    model_alias_rules,
    model_ccpp_field_aliases,
)
from .errors import ConfigurationError, StateTransitionError
from .state import StatePool

if TYPE_CHECKING:
    from .driver import CAMDriver


CHECKPOINT_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """One MPI rank's immutable Python-owned model state."""

    rank: int
    size: int
    config: Mapping[str, Any]
    dimensions: Mapping[str, int]
    contracts: tuple[Mapping[str, Any], ...]
    initialized_fields: tuple[str, ...]
    dynamic_fields: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]
    pool_sealed: bool
    clock: Mapping[str, Any]
    driver_state: str
    scheme_plan: Mapping[str, Any]
    last_phase: str | None
    last_scheme: str | None
    last_scheme_group: str | None
    native_calls: int
    plugin_inventory: tuple[Mapping[str, Any], ...]
    python_process_inventory: tuple[Mapping[str, Any], ...]
    omitted_process_states: tuple[str, ...]
    boundary_index: int
    after_coupler_prepared: bool

    @classmethod
    def capture(
        cls,
        driver: CAMDriver,
        *,
        allow_recreatable_process_state: bool = False,
    ) -> "ModelSnapshot":
        if driver.pool is None or driver.clock is None:
            raise StateTransitionError("cannot snapshot an uninitialized model")
        if driver.state.value == "FINALIZED":
            raise StateTransitionError("cannot snapshot a finalized model")
        if hasattr(driver, "plugins"):
            driver.plugins.assert_checkpointable()
        if allow_recreatable_process_state:
            omitted_process_states = (
                driver.pool.recreatable_process_state_names()
            )
            arrays = driver.pool.snapshot_array_values(readonly=True)
        else:
            omitted_process_states = ()
            arrays = driver.pool.snapshot_arrays(readonly=True)
        return cls(
            rank=int(driver.comm.rank),
            size=int(driver.comm.size),
            config=driver.config.as_dict(),
            dimensions=dict(driver.pool.dimensions),
            contracts=tuple(
                driver.pool.contracts[name].machine_record()
                for name in sorted(driver.pool.contracts)
            ),
            initialized_fields=tuple(
                sorted(driver.pool.initialized_fields)
            ),
            dynamic_fields=tuple(sorted(driver.pool.dynamic_fields)),
            arrays=arrays,
            pool_sealed=driver.pool.sealed,
            clock=asdict(driver.clock),
            driver_state=driver.state.value,
            scheme_plan=driver.scheme_plan.to_payload(),
            last_phase=driver._last_phase,
            last_scheme=driver._last_scheme,
            last_scheme_group=driver._last_scheme_group,
            native_calls=int(driver.backend.call_count),
            plugin_inventory=(
                ()
                if not hasattr(driver, "plugins")
                else driver.plugins.inventory()
            ),
            python_process_inventory=(
                ()
                if not hasattr(driver, "python_processes")
                else driver.python_processes.inventory()
            ),
            omitted_process_states=omitted_process_states,
            boundary_index=int(getattr(driver, "_boundary_index", 0)),
            after_coupler_prepared=bool(
                getattr(driver, "_after_coupler_prepared", False)
            ),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "size": self.size,
            "config": dict(self.config),
            "dimensions": dict(self.dimensions),
            "contracts": [dict(item) for item in self.contracts],
            "initialized_fields": list(self.initialized_fields),
            "dynamic_fields": list(self.dynamic_fields),
            "pool_sealed": self.pool_sealed,
            "clock": dict(self.clock),
            "driver_state": self.driver_state,
            "scheme_plan": dict(self.scheme_plan),
            "last_phase": self.last_phase,
            "last_scheme": self.last_scheme,
            "last_scheme_group": self.last_scheme_group,
            "native_calls": self.native_calls,
            "plugin_inventory": [
                dict(item) for item in self.plugin_inventory
            ],
            "python_process_inventory": [
                dict(item) for item in self.python_process_inventory
            ],
            "omitted_process_states": list(self.omitted_process_states),
            "boundary_index": self.boundary_index,
            "after_coupler_prepared": self.after_coupler_prepared,
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
            config=ModelConfig.from_mapping(metadata["config"]).as_dict(),
            dimensions={
                name: int(value)
                for name, value in metadata["dimensions"].items()
            },
            contracts=tuple(
                dict(item) for item in metadata.get("contracts", ())
            ),
            initialized_fields=tuple(
                str(item)
                for item in metadata.get(
                    "initialized_fields", metadata["array_names"]
                )
            ),
            dynamic_fields=tuple(
                str(item) for item in metadata.get("dynamic_fields", ())
            ),
            arrays=immutable,
            pool_sealed=bool(metadata["pool_sealed"]),
            clock={
                name: (
                    str(value)
                    if name == "calendar"
                    else None
                    if value is None
                    else int(value)
                )
                for name, value in metadata["clock"].items()
            },
            driver_state=str(metadata["driver_state"]),
            scheme_plan=dict(metadata["scheme_plan"]),
            last_phase=metadata.get("last_phase"),
            last_scheme=metadata.get("last_scheme"),
            last_scheme_group=metadata.get("last_scheme_group"),
            native_calls=int(metadata.get("native_calls", 0)),
            plugin_inventory=tuple(
                dict(item)
                for item in metadata.get("plugin_inventory", ())
            ),
            python_process_inventory=tuple(
                dict(item)
                for item in metadata.get("python_process_inventory", ())
            ),
            omitted_process_states=tuple(
                str(item)
                for item in metadata.get("omitted_process_states", ())
            ),
            boundary_index=int(metadata.get("boundary_index", 0)),
            after_coupler_prepared=bool(
                metadata.get("after_coupler_prepared", False)
            ),
        )

    def new_pool(self) -> StatePool:
        """Create branch-private arrays with the same bit patterns."""

        contracts = (
            None
            if not self.contracts
            else tuple(
                FieldContract.from_mapping(item)
                for item in self.contracts
            )
        )
        config = ModelConfig.from_mapping(self.config)
        pool = StatePool(
            self.dimensions,
            contracts=contracts,
            alias_rules=model_alias_rules(config.constituent_names),
            ccpp_aliases=model_ccpp_field_aliases(
                config.constituent_names
            ),
            constituent_names=config.constituent_names,
            advected_constituent_indices=(
                config.advected_constituent_indices
            ),
        )
        pool.restore_arrays(self.arrays)
        pool.restore_registration_state(
            initialized_fields=self.initialized_fields,
            dynamic_fields=self.dynamic_fields,
        )
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

    @classmethod
    def from_rank_payloads(
        cls,
        payloads: Sequence[tuple[Mapping[str, Any], bytes]],
    ) -> "CheckpointBundle":
        """Build a complete checkpoint in memory without filesystem staging."""

        ordered = sorted(payloads, key=lambda item: int(item[0]["rank"]))
        if not ordered:
            raise ConfigurationError(
                "in-memory checkpoint requires at least one rank"
            )
        ranks = [int(metadata["rank"]) for metadata, _content in ordered]
        if ranks != list(range(len(ordered))):
            raise ConfigurationError(
                "in-memory checkpoint rank inventory is incomplete"
            )
        if any(int(metadata["size"]) != len(ordered) for metadata, _ in ordered):
            raise ConfigurationError(
                "in-memory checkpoint communicator sizes are inconsistent"
            )
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "mpi_size": len(ordered),
            "ranks": [dict(metadata) for metadata, _content in ordered],
        }
        files: list[tuple[str, bytes]] = [
            (
                "manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            )
        ]
        files.extend(
            (f"rank-{rank:03d}.npz", bytes(content))
            for rank, (_metadata, content) in enumerate(ordered)
        )
        return cls(tuple(files))

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

    def rank_payloads(self) -> tuple[tuple[Mapping[str, Any], bytes], ...]:
        """Return rank-local metadata and arrays without touching the filesystem."""

        files = dict(self.files)
        try:
            manifest = json.loads(files["manifest.json"])
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConfigurationError(
                "in-memory checkpoint manifest is invalid"
            ) from exc
        if manifest.get("schema_version") not in {
            1, 2, CHECKPOINT_SCHEMA_VERSION
        }:
            raise ConfigurationError(
                "unsupported in-memory checkpoint schema "
                f"{manifest.get('schema_version')!r}"
            )
        mpi_size = int(manifest.get("mpi_size", -1))
        if mpi_size <= 0:
            raise ConfigurationError(
                "in-memory checkpoint MPI size must be positive"
            )
        records = {
            int(record["rank"]): record for record in manifest.get("ranks", ())
        }
        if set(records) != set(range(mpi_size)):
            raise ConfigurationError(
                "in-memory checkpoint rank inventory is incomplete"
            )
        payloads: list[tuple[Mapping[str, Any], bytes]] = []
        for rank in range(mpi_size):
            filename = f"rank-{rank:03d}.npz"
            try:
                content = files[filename]
            except KeyError as exc:
                raise ConfigurationError(
                    f"in-memory checkpoint lacks {filename}"
                ) from exc
            payloads.append((records[rank], content))
        return tuple(payloads)


def serialize_snapshot(snapshot: ModelSnapshot) -> tuple[Mapping[str, Any], bytes]:
    """Serialize one rank's snapshot into an immutable in-memory payload."""

    stream = BytesIO()
    np.savez(stream, **snapshot.arrays)
    return snapshot.metadata(), stream.getvalue()


def deserialize_snapshot(
    metadata: Mapping[str, Any],
    content: bytes,
) -> ModelSnapshot:
    """Restore one rank's snapshot from an in-memory payload."""

    with np.load(BytesIO(content), allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    return ModelSnapshot.from_storage(metadata, arrays)


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
        scheme_plan=CCPPSuitePlan.from_payload(snapshot.scheme_plan),
    )
    driver.pool = snapshot.new_pool()
    _rebind_runtime_local_fields(driver)
    driver.clock = NoLeapClock(**snapshot.clock)
    driver.state = DriverState(snapshot.driver_state)
    driver._last_phase = snapshot.last_phase
    driver._last_scheme = snapshot.last_scheme
    driver._last_scheme_group = snapshot.last_scheme_group
    driver.backend.call_count = snapshot.native_calls
    driver._boundary_index = snapshot.boundary_index
    driver._after_coupler_prepared = snapshot.after_coupler_prepared
    driver._suite_lifecycle_initialized = (
        driver.state is not DriverState.INITIALIZED
    )
    driver.plugins.restore_inventory(snapshot.plugin_inventory)
    driver.python_processes.restore_inventory(
        snapshot.python_process_inventory
    )
    return driver


def _rebind_runtime_local_fields(driver: CAMDriver) -> None:
    """Replace serialized process-local handles with this runtime's values."""

    communicator = (
        int(driver.comm.py2f())
        if hasattr(driver.comm, "py2f")
        else 0
    )
    values = {
        "mpi_communicator": communicator,
        "mpi_root": 0,
        "flag_for_mpi_root": int(driver.comm.rank) == 0,
    }
    for standard_name, value in values.items():
        try:
            field_name = driver.pool.ccpp_field_name(standard_name)
        except KeyError:
            continue
        driver.pool.set(field_name, value)


def write_checkpoint(
    driver: CAMDriver,
    path: str | Path,
    *,
    allow_recreatable_process_state: bool = False,
) -> Path:
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

    snapshot = ModelSnapshot.capture(
        driver,
        allow_recreatable_process_state=allow_recreatable_process_state,
    )
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
    if payload.get("schema_version") not in {
        1, 2, CHECKPOINT_SCHEMA_VERSION
    }:
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
