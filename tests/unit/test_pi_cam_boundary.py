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
from freecam.pi_cam.boundary import (
    _CESMMCTBuffer,
    _CESMFortranHeapRegistry,
    _CESMMCTRegistry,
    _NpyStepReader,
)


class _OneRankWorld:
    def allreduce(self, value):
        return value

    def allgather(self, value):
        return [value]


class _SuccessfulStepEnd:
    @staticmethod
    def pycesm_full_step_end_v1(status) -> None:
        status._obj.value = 0


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


def test_rank_pread_replay_reads_only_the_requested_step(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rank_count": 1,
                "step_count": 2,
                "storage": "rank_pread_v1",
                "file_pattern": "rank-{rank:04d}-{direction}.npy",
            }
        )
    )
    imports = np.arange(12.0).reshape((2, 2, 3))
    exports = np.arange(8.0).reshape((2, 2, 2))
    np.save(tmp_path / "rank-0000-import.npy", imports)
    np.save(tmp_path / "rank-0000-export.npy", exports)
    provider = ReplayBoundaryProvider(tmp_path)
    pool = PICAMStatePool({})

    provider.initialize(rank=0, size=1, config_fingerprint="unused")

    assert isinstance(provider._bundle["x2a_rattr"], _NpyStepReader)
    provider.import_fields(1, 0, pool)
    np.copyto(pool["cam_out.a2x_rattr"], exports[1])
    provider.export_fields(1, 0, pool)
    assert np.array_equal(pool["cam_in.x2a_rattr"], imports[1])
    provider.finalize()
    assert provider._bundle is None


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
    assert not restored.python_owned_internal
    assert not restored.verify_shadow_atmosphere


def test_exact_cesm_provider_rejects_reverse_allocator_callbacks(tmp_path) -> None:
    with pytest.raises(BoundaryReplayError, match="reverse Fortran-to-Python"):
        CESMOnlineBoundaryProvider(
            library=tmp_path / "libcesm.so",
            run_dir=tmp_path / "run",
            python_owned_internal=True,
        )


def test_exact_cesm_provider_rejects_shadow_atmosphere(tmp_path) -> None:
    with pytest.raises(BoundaryReplayError, match="shadow CAM"):
        CESMOnlineBoundaryProvider(
            library=tmp_path / "libcesm.so",
            run_dir=tmp_path / "run",
            verify_shadow_atmosphere=True,
        )


def test_exact_cesm_provider_preserves_first_atm_internal_loop(tmp_path, monkeypatch) -> None:
    provider = CESMOnlineBoundaryProvider(
        library=tmp_path / "libcesm.so",
        run_dir=tmp_path,
    )
    provider._rank = 0
    provider._size = 1
    provider._world = _OneRankWorld()
    provider._native = _SuccessfulStepEnd()
    provider._initial_x2a = np.zeros((2, 3), order="F")
    provider._x2a = _CESMMCTBuffer(
        allocation_id=-1,
        scope="test",
        field_names=("x0", "x1"),
        values=np.ones((2, 3), order="F"),
    )
    provider._a2x = _CESMMCTBuffer(
        allocation_id=-2,
        scope="test",
        field_names=("a0", "a1"),
        values=np.zeros((2, 3), order="F"),
    )
    pool = PICAMStatePool({})
    starts: list[int] = []
    nested: list[int] = []

    def begin() -> None:
        starts.append(provider._next_import_step)
        provider._active_coupling_step = True
        provider._remaining_actions = []

    completions = iter((False, True))
    monkeypatch.setattr(provider, "_begin_coupling_step", begin)
    monkeypatch.setattr(
        provider, "_call_external_atm_iteration", lambda: next(completions)
    )
    monkeypatch.setattr(
        provider,
        "_call_nested_action",
        lambda action_id: nested.append(action_id) or False,
    )

    # The first two records belong to CAM startup and do not lease a CESM
    # coupling step.
    provider.import_fields(0, 0, pool)
    provider.export_fields(0, 0, pool)
    provider.import_fields(1, 0, pool)
    provider.export_fields(1, 0, pool)
    assert starts == []

    # Boundary 2 starts the first real CESM step.  Its first ATM iteration is
    # incomplete, so boundary 3 must reuse the same import and same native
    # component invocation.
    provider.import_fields(2, 0, pool)
    assert provider.has_fresh_import(2, 0)
    provider.export_fields(2, 0, pool)
    assert provider._active_coupling_step

    provider.import_fields(3, 0, pool)
    assert not provider.has_fresh_import(3, 0)
    provider.export_fields(3, 0, pool)

    assert starts == [2]
    assert nested == [210]
    assert provider._coupling_steps == 1
    assert not provider._active_coupling_step


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


class _CountingWorld:
    """A one-rank world that offers the buffer reduction and records aborts."""

    def __init__(self) -> None:
        self.reductions = 0
        self.gathers = 0
        self.aborts: list[int] = []

    def Allreduce(self, sendbuf, recvbuf) -> None:
        self.reductions += 1
        recvbuf[...] = sendbuf

    def allreduce(self, value):
        raise AssertionError("the pickling reduction must not be used when Allreduce exists")

    def allgather(self, value):
        self.gathers += 1
        return [value]

    def Abort(self, code: int) -> None:
        self.aborts.append(code)


class _RecordingCoupler:
    """The native coupler as the provider drives it, recording the order."""

    def __init__(self, *, refuse: int | None = None, mask: int = 0b1111111) -> None:
        self.calls: list[object] = []
        self.refuse = refuse
        self.mask = mask

    def pycesm_full_step_begin_v1(self, native_step, ymd, seconds, mask, status) -> None:
        self.calls.append("begin")
        native_step._obj.value = 1
        ymd._obj.value = 10101
        seconds._obj.value = 1800
        mask._obj.value = self.mask
        status._obj.value = 0

    def pycesm_full_action_v1(self, action_id, status) -> None:
        self.calls.append(action_id.value)
        status._obj.value = 4 if action_id.value == self.refuse else 0

    def pycesm_full_nested_action_v1(self, action_id, complete, status) -> None:
        self.calls.append(action_id.value)
        complete._obj.value = 0
        status._obj.value = 0

    def pycesm_full_step_end_v1(self, status) -> None:
        self.calls.append("end")
        status._obj.value = 0


def _grouped_provider(tmp_path, native) -> tuple[CESMOnlineBoundaryProvider, _CountingWorld]:
    provider = CESMOnlineBoundaryProvider(library=tmp_path / "libcesm.so", run_dir=tmp_path)
    provider._rank = 0
    provider._size = 1
    world = _CountingWorld()
    provider._world = world
    provider._native = native
    return provider, world


def test_the_coupling_step_opens_with_one_reduction_after_the_whole_sequence(tmp_path) -> None:
    """The original driver runs the land, ice and ocean actions back to back,
    each rank skipping the components it is not part of, so they overlap on
    disjoint ranks.  A check after every action lined them up behind each
    other; the check now comes once, after the whole opening sequence."""

    native = _RecordingCoupler()
    provider, world = _grouped_provider(tmp_path, native)

    provider._begin_coupling_step()

    assert native.calls[0] == "begin"
    assert native.calls[1:] == [*range(101, 125), 201, 202]
    assert world.reductions == 1
    assert world.gathers == 0
    assert provider._remaining_actions == [126, 127, 128, 129]
    assert provider._active_coupling_step


def test_an_alarm_that_is_off_drops_its_actions_from_the_sequence(tmp_path) -> None:
    # every alarm but rof (bit 3)
    native = _RecordingCoupler(mask=0b1110111)
    provider, _ = _grouped_provider(tmp_path, native)

    provider._begin_coupling_step()

    assert 106 not in native.calls and 118 not in native.calls
    assert 102 in native.calls and 104 in native.calls


def test_a_protocol_refusal_stops_the_sequence_and_is_reported_collectively(tmp_path) -> None:
    """A status is decided from replicated state, so every rank stops at the
    same call and meets at the one reduction; nobody aborts."""

    native = _RecordingCoupler(refuse=111)
    provider, world = _grouped_provider(tmp_path, native)

    with pytest.raises(BoundaryReplayError, match="coupling step begin"):
        provider._begin_coupling_step()

    assert native.calls[-1] == 111
    assert world.reductions == 1
    assert world.gathers == 1
    assert world.aborts == []
    assert not provider._active_coupling_step


def test_a_rank_local_failure_inside_the_sequence_aborts_instead_of_hanging_the_others(
    tmp_path,
) -> None:
    """An allocation callback failing is one rank's problem.  The other ranks
    may already be inside the next call's MPI collective, so the rank aborts
    the job as shr_sys_abort would rather than leave them waiting."""

    from freecam.pi_cam.boundary import CESMRankLocalError

    class FailingRegistry:
        def raise_if_failed(self) -> None:
            raise CESMRankLocalError("CESM MCT allocation callback failed")

    native = _RecordingCoupler()
    provider, world = _grouped_provider(tmp_path, native)
    provider._registry = FailingRegistry()

    with pytest.raises(CESMRankLocalError):
        provider._begin_coupling_step()

    assert world.aborts == [1]
    assert world.reductions == 0
    assert native.calls == ["begin"]


def test_the_export_carries_the_loop_vote_in_its_reduction_and_closes_in_one_group(
    tmp_path, monkeypatch
) -> None:
    native = _RecordingCoupler()
    provider, world = _grouped_provider(tmp_path, native)
    provider._initial_x2a = np.zeros((2, 3), order="F")
    provider._x2a = _CESMMCTBuffer(
        allocation_id=-1, scope="test", field_names=("x0", "x1"), values=np.ones((2, 3), order="F")
    )
    provider._a2x = _CESMMCTBuffer(
        allocation_id=-2, scope="test", field_names=("a0", "a1"), values=np.zeros((2, 3), order="F")
    )
    pool = PICAMStatePool({})

    def begin() -> None:
        provider._active_coupling_step = True
        provider._remaining_actions = [126, 129]

    monkeypatch.setattr(provider, "_begin_coupling_step", begin)
    monkeypatch.setattr(provider, "_call_external_atm_iteration", lambda: True)

    for step in (0, 1):
        provider.import_fields(step, 0, pool)
        provider.export_fields(step, 0, pool)
    provider.import_fields(2, 0, pool)
    world.reductions = 0
    native.calls.clear()

    provider.export_fields(2, 0, pool)

    # one reduction votes on the loop's completion; one closes the step
    assert world.reductions == 2
    assert native.calls == [210, 126, 129, "end"]
    assert provider._coupling_steps == 1
    assert not provider._active_coupling_step


def test_a_world_without_the_buffer_reduction_still_agrees(tmp_path) -> None:
    native = _RecordingCoupler()
    provider, _ = _grouped_provider(tmp_path, native)
    provider._world = _OneRankWorld()

    provider._begin_coupling_step()

    assert native.calls[-1] == 202
