import json
from pathlib import Path
import subprocess
import sys

import numpy as np

def test_extractor_keeps_only_one_rank_local_boundary_state(tmp_path: Path) -> None:
    replay = tmp_path / "replay"
    replay.mkdir()
    source_x2a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    source_a2x = np.arange(40, dtype=np.float64).reshape(2, 5, 4)
    np.savez(
        replay / "rank-0000.npz",
        x2a_rattr=source_x2a,
        a2x_rattr=source_a2x,
    )
    (replay / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "storage": "rank_bundle_v1",
                "rank_count": 1,
                "step_count": 2,
                "file_pattern": "rank-{rank:04d}.npz",
                "config_fingerprint": "test-fingerprint",
            }
        )
    )
    output = tmp_path / "bootstrap"

    project = Path(__file__).parents[2]
    subprocess.run(
        (
            sys.executable,
            str(project / "tools/extract_pi_cam_online_bootstrap.py"),
            "--replay",
            str(replay),
            "--output",
            str(output),
            "--step",
            "1",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())

    with np.load(output / "rank-0000.npz", allow_pickle=False) as payload:
        assert np.array_equal(payload["x2a_rattr"], source_x2a[1])
        assert np.array_equal(payload["a2x_rattr"], source_a2x[1])
    assert manifest["storage"] == "rank_bootstrap_v1"
    assert manifest["source_config_fingerprint"] == "test-fingerprint"
    assert manifest["source_step"] == 1
