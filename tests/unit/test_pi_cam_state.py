import numpy as np
import pytest
from types import SimpleNamespace

from freecam.pi_cam import (
    PICAMFieldContract,
    PICAMStateError,
    PICAMStatePool,
)
from freecam.pi_cam.native import _NativeStateBridge


def test_pi_cam_state_is_rank_local_python_owned_fortran_storage() -> None:
    pool = PICAMStatePool({"column": 27, "level": 30})
    values = pool.create(
        PICAMFieldContract(
            "air_temperature",
            ("column", "level"),
            aliases=("T",),
        )
    )

    assert values.shape == (27, 30)
    assert values.flags.f_contiguous
    assert pool["T"].ctypes.data == values.ctypes.data


def test_replay_field_cannot_change_shape_between_steps() -> None:
    pool = PICAMStatePool({})
    pool.ensure_from_array("cam_in.sst", np.ones((2, 3)), category="boundary")

    with pytest.raises(PICAMStateError, match="changed shape"):
        pool.ensure_from_array(
            "cam_in.sst", np.ones((3, 2)), category="boundary"
        )


def test_generated_native_contract_preallocates_rank_local_arrays() -> None:
    bridge = object.__new__(_NativeStateBridge)
    bridge.dimension_defaults = {
        "pcnst": 57,
        "physics_columns_per_element": 9,
    }
    bridge.derived_shell = False
    bridge.fields = (
        {
            "name": "phys_state.t",
            "dtype": "float64",
            "dimensions": ["pcols", "pver", "chunks"],
            "active_by_default": True,
        },
        {
            "name": "cam_in.optional",
            "dtype": "float64",
            "dimensions": ["pcols", "chunks"],
            "active_by_default": False,
        },
    )
    pool = PICAMStatePool({})

    bridge.preallocate(
        pool,
        SimpleNamespace(pcols=16, pver=30, resolution="ne16"),
        rank=0,
        size=512,
    )

    assert pool["phys_state.t"].shape == (16, 30, 2)
    assert pool["phys_state.t"].flags.f_contiguous
    assert "cam_in.optional" not in pool


def test_inline_native_field_can_be_a_strided_view_without_double_counting() -> None:
    pool = PICAMStatePool({"column": 2, "chunks": 3, "owner_bytes": 96})
    raw = np.zeros(96, dtype=np.uint8)
    pool.attach(
        PICAMFieldContract(
            "__native_owner.state",
            ("owner_bytes",),
            "uint8",
            category="native_cam_owner",
            restart=False,
        ),
        raw,
    )
    field = np.ndarray(
        (2, 3),
        dtype=np.float64,
        buffer=raw,
        offset=8,
        strides=(8, 32),
    )
    pool.attach(
        PICAMFieldContract(
            "state.temperature",
            ("column", "chunks"),
            category="native_cam_inline_state",
            requires_contiguous=False,
        ),
        field,
    )

    field[:, 1] = (280.0, 281.0)
    assert np.frombuffer(raw, dtype=np.float64, count=2, offset=40).tolist() == [
        280.0,
        281.0,
    ]
    assert pool.nbytes == raw.nbytes


def test_python_initialization_uses_exact_native_chunk_and_sentinel_context() -> None:
    bridge = object.__new__(_NativeStateBridge)
    bridge.derived_shell = True
    bridge._initialization_context = {
        "chunk_begin": 101,
        "chunk_end": 102,
        "chunk_ncols": (16, 11),
        "inf_bits": int(np.asarray(np.inf, dtype=np.float64).view(np.int64)),
        "posinf_bits": int(np.asarray(np.inf, dtype=np.float64).view(np.int64)),
    }
    bridge.initializers = {
        "state.lchnk": "chunk_id",
        "state.ncol": "chunk_ncols",
        "state.psetcols": "pcols",
        "state.temperature": "inf",
        "state.factor": 0.1,
    }
    pool = PICAMStatePool({"pcols": 16, "chunks": 2})
    for name, dtype in (
        ("state.lchnk", "int32"),
        ("state.ncol", "int32"),
        ("state.psetcols", "int32"),
    ):
        pool.create(PICAMFieldContract(name, ("chunks",), dtype))
    pool.create(PICAMFieldContract("state.temperature", ("pcols", "chunks")))
    pool.create(PICAMFieldContract("state.factor", ("pcols", "chunks")))

    bridge._initialize_python_state(pool, pcols=16)

    assert pool["state.lchnk"].tolist() == [101, 102]
    assert pool["state.ncol"].tolist() == [16, 11]
    assert pool["state.psetcols"].tolist() == [16, 16]
    assert np.isposinf(pool["state.temperature"]).all()
    assert np.array_equal(pool["state.factor"], np.full((16, 2), 0.1))
    assert pool["grid.chunk_id"].flags.f_contiguous


def test_restart_snapshot_restores_inline_view_into_python_owner_buffer() -> None:
    pool = PICAMStatePool({"owner_bytes": 64, "chunks": 2})
    raw = np.zeros(64, dtype=np.uint8)
    pool.attach(
        PICAMFieldContract(
            "__native_owner.state",
            ("owner_bytes",),
            "uint8",
            category="native_cam_owner",
            restart=False,
        ),
        raw,
    )
    inline = np.ndarray((2,), dtype=np.int32, buffer=raw, strides=(32,))
    pool.attach(
        PICAMFieldContract(
            "state.ncol",
            ("chunks",),
            "int32",
            category="native_cam_inline_state",
            requires_contiguous=False,
        ),
        inline,
    )
    inline[...] = (16, 11)
    snapshot = pool.snapshot(restart_only=True)
    inline[...] = 0

    pool.restore(snapshot)

    assert inline.tolist() == [16, 11]
    assert "__native_owner.state" not in snapshot
