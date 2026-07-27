from pathlib import Path

import pytest

from pycam_sima.model import ModelConfig
from pycam_sima.notebook.pool_session import PooledWorkerSession
from pycam_sima.notebook.pool_worker import (
    slot_for_world_rank,
    validate_pool_layout,
)


def _session(tmp_path: Path) -> PooledWorkerSession:
    library = tmp_path / "libpycam_sima_kernels.so"
    library.touch()
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "atm_in").write_text("&cam_initfiles_nl /\n")
    env_script = tmp_path / "machine-env.sh"
    env_script.write_text("export PYCAM_POOL_TEST=1\n")
    return PooledWorkerSession(
        ModelConfig(),
        run_root=tmp_path / "runs",
        initial_run_dir=initial,
        library=library,
        ranks_per_model=3,
        model_slots=4,
        env_script=env_script,
    )


def test_pool_layout_is_dynamic_and_complete() -> None:
    validate_pool_layout(12, ranks_per_model=3, model_slots=4)
    assert [slot_for_world_rank(rank, 3) for rank in range(12)] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
    ]
    with pytest.raises(ValueError, match="requires 4 slots x 3 ranks"):
        validate_pool_layout(11, ranks_per_model=3, model_slots=4)


def test_pool_session_overrides_model_rank_count_without_fixed_24(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    assert session.ranks_per_model == 3
    assert session.model_slots == 4
    assert session.world_size == 12
    assert session.ranks == 12
    assert session.config.mpi_size == 3


def test_advance_models_emits_one_concurrent_pool_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    session._models = {"control": 1, "warm": 3}
    requests = []

    def request(payload):
        requests.append(payload)
        return {
            "1": {"step": 7, "model_name": "control"},
            "3": {"step": 7, "model_name": "warm"},
        }

    monkeypatch.setattr(session, "_request", request)
    result = session.advance_models(("control", "warm"), count=2)

    assert result["control"]["step"] == 7
    assert result["warm"]["step"] == 7
    assert requests == [
        {
            "op": "model_commands",
            "commands": [
                {
                    "slot": 1,
                    "name": "control",
                    "command": {"op": "step", "count": 2},
                },
                {
                    "slot": 3,
                    "name": "warm",
                    "command": {"op": "step", "count": 2},
                },
            ],
        }
    ]


def test_call_models_batches_different_operations_by_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    session._models = {"control": 1, "warm": 3}
    requests = []

    def request(payload):
        requests.append(payload)
        return {
            "1": {"step": 3, "model_name": "control"},
            "3": {"model_name": "warm", "mean": 241.0},
        }

    monkeypatch.setattr(session, "_request", request)
    result = session.call_models(
        (
            ("control", "run_scheme", ("kessler",), {"group": "before"}),
            (
                "warm",
                "get_field_stats",
                ("air_temperature",),
                {"rank": 0},
            ),
        )
    )

    assert result["control"]["step"] == 3
    assert result["warm"]["mean"] == 241.0
    assert requests == [
        {
            "op": "model_commands",
            "commands": [
                {
                    "slot": 1,
                    "name": "control",
                    "command": {
                        "op": "run_scheme",
                        "scheme": "kessler",
                        "group": "before",
                        "model_name": "control",
                    },
                },
                {
                    "slot": 3,
                    "name": "warm",
                    "command": {
                        "op": "get_field_stats",
                        "field": "air_temperature",
                        "rank": 0,
                        "model_name": "warm",
                    },
                },
            ],
        }
    ]


def test_pool_model_call_never_uses_checkpoint_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    session._models = {"base": 2}
    requests = []
    monkeypatch.setattr(
        session,
        "_request",
        lambda payload: requests.append(payload) or {"step": 1},
    )

    session.call("base", "run_scheme", "kessler")
    assert requests == [
        {
            "op": "model_command",
            "slot": 2,
            "name": "base",
            "command": {
                "op": "run_scheme",
                "scheme": "kessler",
                "group": None,
                "model_name": "base",
            },
        }
    ]
    with pytest.raises(ValueError, match="not a model command"):
        session.call("base", "restore_memory_checkpoint")
