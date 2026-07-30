from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pycam_sima.model import (
    BranchSpec,
    CAMDriver,
    CCPPSuitePlan,
    CheckpointBundle,
    FieldEdit,
    ModelSnapshot,
    SchemeMove,
    read_checkpoint,
)
from pycam_sima.model.checkpoint import (
    deserialize_snapshot,
    serialize_snapshot,
    write_checkpoint,
)
from pycam_sima.model.clock import NoLeapClock
from pycam_sima.model.config import ModelConfig
from pycam_sima.model.driver import DriverState
from pycam_sima.model.grid import dimensions_for_rank
from pycam_sima.model.state import NativeObjectHandle, StatePool


ROOT = Path(__file__).resolve().parents[2]
KESSLER_SUITE = (
    ROOT
    / "external/CAM-SIMA/src/physics/ncar_ccpp/suites/suite_kessler.xml"
)


def _scheme_plan() -> CCPPSuitePlan:
    return CCPPSuitePlan.from_xml(KESSLER_SUITE)


class _LocalCheckpointComm:
    rank = 0
    size = 1

    def bcast(self, value, root=0):
        return value

    def gather(self, value, root=0):
        return [value]

    def allgather(self, value):
        return [value]

    def barrier(self):
        return None


def test_snapshot_arrays_create_isolated_fortran_contiguous_branches() -> None:
    pool = StatePool(dimensions_for_rank(0, 24))
    pool.set("air_temperature", 240.0)
    pool.seal_static()
    arrays = pool.snapshot_arrays()

    first = StatePool(pool.dimensions)
    first.restore_arrays(arrays)
    first.seal_static()
    second = StatePool(pool.dimensions)
    second.restore_arrays(arrays)
    second.seal_static()

    first.get("air_temperature")[0, 0, 0, 0, 0] = 250.0
    assert second.get("air_temperature")[0, 0, 0, 0, 0] == 240.0
    assert arrays["air_temperature"][0, 0, 0, 0, 0] == 240.0
    assert not arrays["air_temperature"].flags.writeable
    assert first.get("air_temperature").flags.f_contiguous
    assert not np.shares_memory(
        first.get("air_temperature"), second.get("air_temperature")
    )


def test_driver_snapshot_recreates_private_pool() -> None:
    pool = StatePool(dimensions_for_rank(0, 24))
    pool.set("air_temperature", 231.5)
    pool.seal_static()
    driver = SimpleNamespace(
        pool=pool,
        clock=NoLeapClock(nstep=3, seconds=5400),
        state=DriverState.RUNNING,
        comm=SimpleNamespace(rank=0, size=24),
        config=ModelConfig(),
        scheme_plan=_scheme_plan(),
        _last_phase="physics_timestep_initial",
        _last_scheme="physics_before_coupler.kessler_diagnostics",
        _last_scheme_group="physics_before_coupler",
        backend=SimpleNamespace(call_count=17),
    )

    snapshot = ModelSnapshot.capture(driver)
    restored = snapshot.new_pool()
    restored.get("air_temperature")[0, 0, 0, 0, 0] += 1.0

    assert snapshot.clock["nstep"] == 3
    assert snapshot.native_calls == 17
    assert pool.get("air_temperature")[0, 0, 0, 0, 0] == 231.5
    assert restored.get("air_temperature")[0, 0, 0, 0, 0] == 232.5


def test_pool_fork_omits_only_recreatable_opaque_process_state() -> None:
    pool = StatePool(dimensions_for_rank(0, 24))
    pool.set("air_temperature", 231.5)
    handle = NativeObjectHandle(
        address=12345,
        fortran_type="ccpp_constituent_prop_ptr_t",
        shape=(3,),
        owner=object(),
        destroy=lambda: None,
        recreatable=True,
    )
    pool.set_process_state("ccpp_constituent_properties", handle)
    driver = SimpleNamespace(
        pool=pool,
        clock=NoLeapClock(nstep=3, seconds=5400),
        state=DriverState.RUNNING,
        comm=SimpleNamespace(rank=0, size=24),
        config=ModelConfig(),
        scheme_plan=_scheme_plan(),
        _last_phase="physics_timestep_initial",
        _last_scheme="physics_before_coupler.geopotential_temp",
        _last_scheme_group="physics_before_coupler",
        backend=SimpleNamespace(call_count=17),
    )

    with pytest.raises(Exception, match="cannot checkpoint opaque"):
        ModelSnapshot.capture(driver)
    snapshot = ModelSnapshot.capture(
        driver, allow_recreatable_process_state=True
    )
    restored = deserialize_snapshot(*serialize_snapshot(snapshot))
    restored_pool = restored.new_pool()

    assert snapshot.omitted_process_states == (
        "ccpp_constituent_properties",
    )
    assert restored.omitted_process_states == snapshot.omitted_process_states
    assert not restored_pool.process_state_names
    assert np.array_equal(
        restored_pool.get("air_temperature"),
        pool.get("air_temperature"),
    )

    handle.recreatable = False
    with pytest.raises(Exception, match="type-specific serializer"):
        ModelSnapshot.capture(
            driver, allow_recreatable_process_state=True
        )


def test_checkpoint_bundle_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(
        '{"ranks": [{"rank": 0}, {"rank": 1}]}'
    )
    (source / "rank-000.npz").write_bytes(b"rank zero")
    (source / "rank-001.npz").write_bytes(b"rank one")

    bundle = CheckpointBundle.from_directory(source)
    restored = bundle.materialize(tmp_path / "restored")

    assert bundle.nbytes > 0
    assert (restored / "manifest.json").read_bytes() == (
        source / "manifest.json"
    ).read_bytes()
    assert (restored / "rank-001.npz").read_bytes() == b"rank one"


def test_checkpoint_bundle_rank_payload_round_trip_without_disk() -> None:
    pool = StatePool(dimensions_for_rank(0, 24))
    pool.set("air_temperature", 240.0)
    pool.seal_static()
    driver = SimpleNamespace(
        pool=pool,
        clock=NoLeapClock(nstep=6, seconds=10800),
        state=DriverState.RUNNING,
        comm=SimpleNamespace(rank=0, size=1),
        config=ModelConfig(),
        scheme_plan=_scheme_plan(),
        _last_phase="physics_timestep_initial",
        _last_scheme=None,
        _last_scheme_group=None,
        backend=SimpleNamespace(call_count=31),
    )

    snapshot = ModelSnapshot.capture(driver)
    payload = serialize_snapshot(snapshot)
    bundle = CheckpointBundle.from_rank_payloads((payload,))
    restored = deserialize_snapshot(*bundle.rank_payloads()[0])

    assert bundle.nbytes > 0
    assert restored.metadata() == snapshot.metadata()
    for name in snapshot.arrays:
        assert np.array_equal(restored.arrays[name], snapshot.arrays[name])


def test_collective_checkpoint_round_trip_preserves_bits(tmp_path: Path) -> None:
    pool = StatePool(dimensions_for_rank(0, 24))
    values = np.arange(pool.get("physics_air_temperature").size, dtype=np.float64)
    pool.set(
        "physics_air_temperature",
        values.reshape(pool.get("physics_air_temperature").shape),
    )
    pool.seal_static()
    comm = _LocalCheckpointComm()
    driver = SimpleNamespace(
        pool=pool,
        clock=NoLeapClock(nstep=4, seconds=7200),
        state=DriverState.RUNNING,
        comm=comm,
        config=ModelConfig(),
        scheme_plan=_scheme_plan(),
        _last_phase="physics_timestep_initial",
        _last_scheme=None,
        _last_scheme_group=None,
        backend=SimpleNamespace(call_count=21),
    )

    checkpoint = write_checkpoint(driver, tmp_path / "checkpoint")
    restored = read_checkpoint(checkpoint, comm)

    assert restored.clock["nstep"] == 4
    assert restored.native_calls == 21
    assert np.array_equal(
        restored.arrays["physics_air_temperature"],
        pool.get("physics_air_temperature"),
    )


def test_continue_run_restores_clock_and_state_without_startup(
    tmp_path: Path,
) -> None:
    comm = _LocalCheckpointComm()
    config = ModelConfig(
        mpi_size=1,
        calendar="360_DAY",
        start_date="2000-02-30",
        analytic_ic_type="resting_isothermal",
        dt_seconds=60,
        history_enabled=False,
    )
    pool = StatePool(dimensions_for_rank(0, 1))
    pool.set("air_temperature", 271.25)
    pool.seal_static()
    source = SimpleNamespace(
        pool=pool,
        clock=NoLeapClock(
            year=2000,
            month=2,
            day=30,
            seconds=120,
            nstep=2,
            dt_seconds=60,
            calendar="360_DAY",
        ),
        state=DriverState.RUNNING,
        comm=comm,
        config=config,
        scheme_plan=_scheme_plan(),
        _last_phase="physics_timestep_initial",
        _last_scheme=None,
        _last_scheme_group=None,
        backend=SimpleNamespace(call_count=9),
    )
    checkpoint = write_checkpoint(source, tmp_path / "restart")
    continued = config.with_overrides(
        run_type="continue",
        restart_path=str(checkpoint),
        stop_n=3,
    )

    driver = CAMDriver(
        continued,
        run_dir=tmp_path,
        comm=comm,
        history_dir=tmp_path / "continued-history",
    ).start()

    assert driver.config.run_type == "continue"
    assert driver.clock.calendar == "360_DAY"
    assert driver.clock.iso_stamp == "2000-02-30-00120"
    assert driver.clock.nstep == 2
    assert np.array_equal(
        driver.pool.get("air_temperature"),
        pool.get("air_temperature"),
    )
    assert driver.backend.call_count == 9


def test_branch_spec_round_trip_and_isolated_edits() -> None:
    pool = StatePool(dimensions_for_rank(0, 24))
    pool.set("air_temperature", 240.0)
    pool.seal_static()
    driver = SimpleNamespace(
        pool=pool,
        scheme_plan=_scheme_plan(),
    )
    spec = BranchSpec(
        name="warm-no-kessler",
        steps=2,
        disable_schemes=("kessler",),
        scheme_moves=(
            SchemeMove("kessler", to_group="physics_after_coupler"),
        ),
        field_edits=(FieldEdit("air_temperature", "add", 1.5),),
    )

    restored = BranchSpec.from_mapping(spec.as_dict())
    restored.apply(driver)

    assert restored == spec
    assert not driver.scheme_plan.scheme("kessler").enabled
    assert driver.scheme_plan.scheme("kessler").group == "physics_after_coupler"
    assert np.all(pool.get("air_temperature") == 241.5)
