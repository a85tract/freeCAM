import json

import numpy as np
import pytest

from freecam.pi_cam import (
    BoundaryReplayError,
    PICAMStatePool,
    ReplayBoundaryProvider,
    write_boundary_payload,
)


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
