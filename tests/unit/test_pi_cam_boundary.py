import json

import numpy as np
import pytest

from freecam.pi_cam import (
    BoundaryReplayError,
    CESMOnlineBoundaryProvider,
    HeldSurfaceModel,
    OnlineBoundaryProvider,
    PICAMStatePool,
    ReplayBoundaryProvider,
    prepare_cesm_online_run,
    write_boundary_payload,
)
from freecam.pi_cam.boundary import _CESMFortranHeapRegistry, _CESMMCTRegistry


def test_replay_boundary_loads_import_and_compares_export_bitwise(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "rank_count": 1, "step_count": 1})
    )
    imported = np.arange(6.0).reshape((2, 3), order="F")
    exported = np.arange(2.0)
    write_boundary_payload(
        tmp_path, step=0, rank=0, direction="import", fields={"sst": imported}
    )
    write_boundary_payload(
        tmp_path, step=0, rank=0, direction="export", fields={"tbot": exported}
    )
    provider = ReplayBoundaryProvider(tmp_path)
    pool = PICAMStatePool({})
    provider.initialize(rank=0, size=1, config_fingerprint="unused")
    provider.import_fields(0, 0, pool)
    pool.ensure_from_array("cam_out.tbot", exported, category="boundary_export")

    provider.export_fields(0, 0, pool)
    assert np.array_equal(pool["cam_in.sst"], imported)

    pool["cam_out.tbot"][0] += np.spacing(exported[0])
    with pytest.raises(BoundaryReplayError, match="not bitwise identical"):
        provider.export_fields(0, 0, pool)


def test_rank_bundle_replay_keeps_all_steps_in_one_rank_file(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rank_count": 1,
                "step_count": 2,
                "storage": "rank_bundle_v1",
                "file_pattern": "rank-{rank:04d}.npz",
                "held_import_steps": [1],
            }
        )
    )
    imports = np.arange(12.0).reshape((2, 2, 3), order="F")
    exports = np.arange(8.0).reshape((2, 2, 2), order="F")
    np.savez(
        tmp_path / "rank-0000.npz",
        x2a_rattr=imports,
        a2x_rattr=exports,
    )
    provider = ReplayBoundaryProvider(tmp_path)
    pool = PICAMStatePool({})
    provider.initialize(rank=0, size=1, config_fingerprint="unused")
    assert provider.has_fresh_import(0, 0)
    assert not provider.has_fresh_import(1, 0)
    provider.import_fields(1, 0, pool)
    np.copyto(pool["cam_out.a2x_rattr"], exports[1])

    provider.export_fields(1, 0, pool)
    assert np.array_equal(pool["cam_in.x2a_rattr"], imports[1])


def _online_bootstrap(tmp_path, *, ranks=1):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "storage": "rank_bootstrap_v1",
                "rank_count": ranks,
                "file_pattern": "rank-{rank:04d}.npz",
            }
        )
    )
    for rank in range(ranks):
        np.savez(
            tmp_path / f"rank-{rank:04d}.npz",
            x2a_rattr=np.full((3, 2), 10.0 + rank),
            a2x_rattr=np.zeros((4, 2)),
        )


def test_online_boundary_generates_unbounded_x2a_from_previous_cam_export(
    tmp_path,
) -> None:
    _online_bootstrap(tmp_path)
    calls = []

    def update(fields, context):
        assert not fields.a2x.flags.writeable
        fields.x2a += fields.a2x[0] + context.step
        calls.append((context.step, context.rank, context.timestep_seconds))

    provider = OnlineBoundaryProvider(tmp_path, update)
    pool = PICAMStatePool({})
    pool.ensure_from_array(
        "model_timestep", np.asarray(1800, dtype=np.int32), category="control"
    )
    provider.initialize(rank=0, size=1, config_fingerprint="unused")

    for step in range(75):
        provider.import_fields(step, 0, pool)
        assert provider.has_fresh_import(step, 0)
        pool["cam_out.a2x_rattr"][...] = step + 1
        provider.export_fields(step, 0, pool)

    # There is only one bootstrap state, but the callback generated 74 more.
    assert len(calls) == 74
    assert calls[0] == (1, 0, 1800)
    assert calls[-1][0] == 74
    assert np.all(pool["cam_in.x2a_rattr"] > 10.0)


def test_held_surface_online_boundary_is_explicit_and_constant(tmp_path) -> None:
    _online_bootstrap(tmp_path)
    provider = OnlineBoundaryProvider(tmp_path, HeldSurfaceModel())
    pool = PICAMStatePool({})
    provider.initialize(rank=0, size=1, config_fingerprint="unused")

    provider.import_fields(0, 0, pool)
    initial = pool["cam_in.x2a_rattr"].copy()
    pool["cam_out.a2x_rattr"].fill(99.0)
    provider.export_fields(0, 0, pool)
    provider.import_fields(1, 0, pool)

    assert np.array_equal(pool["cam_in.x2a_rattr"], initial)


def test_online_boundary_preserves_opaque_a2x_sentinels(tmp_path) -> None:
    _online_bootstrap(tmp_path)
    with np.load(tmp_path / "rank-0000.npz", allow_pickle=False) as payload:
        x2a = payload["x2a_rattr"].copy()
        a2x = payload["a2x_rattr"].copy()
    a2x[0, :] = np.inf
    np.savez(tmp_path / "rank-0000.npz", x2a_rattr=x2a, a2x_rattr=a2x)
    provider = OnlineBoundaryProvider.held(tmp_path)
    pool = PICAMStatePool({})
    provider.initialize(rank=0, size=1, config_fingerprint="unused")
    provider.import_fields(0, 0, pool)
    pool["cam_out.a2x_rattr"][0, :] = np.inf

    provider.export_fields(0, 0, pool)
    provider.import_fields(1, 0, pool)

    assert np.isfinite(pool["cam_in.x2a_rattr"]).all()


def test_online_boundary_rejects_callback_return_values(tmp_path) -> None:
    _online_bootstrap(tmp_path)
    provider = OnlineBoundaryProvider(tmp_path, lambda fields, context: fields.x2a)
    pool = PICAMStatePool({})
    provider.initialize(rank=0, size=1, config_fingerprint="unused")
    provider.import_fields(0, 0, pool)
    provider.export_fields(0, 0, pool)

    with pytest.raises(BoundaryReplayError, match="return None"):
        provider.import_fields(1, 0, pool)


def test_cesm_mct_registry_owns_fortran_contiguous_component_buffers() -> None:
    import ctypes

    registry = _CESMMCTRegistry()
    status = ctypes.c_int32(-1)
    with registry.in_scope("initialize:atm_initialize"):
        address = registry._allocate(
            2,
            1,
            8,
            41,
            27,
            ":".join(f"x2a_{index}" for index in range(41)).encode(),
            ctypes.pointer(status),
        )
    assert status.value == 0
    assert address
    buffer = registry.component_exchange(41)
    assert buffer.values.shape == (41, 27)
    assert buffer.values.flags.f_contiguous
    registry._release(address, ctypes.pointer(status))
    assert not registry.buffers


def test_cesm_heap_registry_returns_aligned_rank_local_storage() -> None:
    import ctypes

    registry = _CESMFortranHeapRegistry()
    status = ctypes.c_int32(-1)
    with registry.in_scope("initialize:atm"):
        address = registry._allocate(
            b"cam_state.F90:42",
            7,
            513,
            64,
            ctypes.pointer(status),
        )
    assert status.value == 0
    assert address
    assert int(address) % 64 == 0
    assert registry.buffers[7].values.nbytes == 513
    assert registry.buffers[7].scope == "initialize:atm"
    registry._release(address, ctypes.pointer(status))
    assert not registry.buffers
    assert registry.live_bytes == 0


def test_exact_cesm_provider_is_pickle_safe_before_mpi_initialization(
    tmp_path,
) -> None:
    import cloudpickle

    provider = CESMOnlineBoundaryProvider(
        library=tmp_path / "libcesm.so",
        run_dir=tmp_path / "run",
        oracle=tmp_path / "oracle",
    )
    restored = cloudpickle.loads(cloudpickle.dumps(provider, protocol=5))
    assert restored.library == (tmp_path / "libcesm.so").resolve()
    assert restored.run_dir == (tmp_path / "run").resolve()
    assert restored.oracle == (tmp_path / "oracle").resolve()
    assert restored.python_owned_internal
    assert not restored.verify_shadow_atmosphere


def test_exact_cesm_provider_prepares_only_startup_inputs(tmp_path) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "drv_in").write_text("driver input\n")
    (seed / "SEMapping.nc").write_bytes(b"mapping")
    (seed / "lnd_in").write_text("land input\n")
    (seed / "history.nc").write_bytes(b"history")
    (seed / "restart.bin").write_bytes(b"restart")
    (seed / "cesm.log.test").write_text("log\n")
    (seed / "rpointer.drv").write_text("restart pointer\n")
    (seed / "timing").mkdir()
    (seed / "timing" / "old").write_text("old timing\n")

    prepared = prepare_cesm_online_run(seed, tmp_path / "provider-run")

    assert (prepared / "drv_in").is_file()
    assert (prepared / "SEMapping.nc").read_bytes() == b"mapping"
    assert (prepared / "lnd_in").is_file()
    assert not (prepared / "history.nc").exists()
    assert not (prepared / "restart.bin").exists()
    assert not (prepared / "cesm.log.test").exists()
    assert not (prepared / "rpointer.drv").exists()
    assert (prepared / "timing" / "checkpoints").is_dir()
