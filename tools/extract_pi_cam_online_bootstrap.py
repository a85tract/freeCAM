#!/usr/bin/env python3
"""Extract one rank-local x2a/a2x state from a replay capture.

The result is a small bootstrap, not another boundary trajectory.  An online
provider reads it once and generates every later x2a state in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def _read_step(path: Path, name: str, step: int) -> np.ndarray:
    """Read one leading-axis slice without loading a complete rank bundle."""

    with ZipFile(path) as archive, archive.open(f"{name}.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream
            )
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream
            )
        if not shape or not 0 <= step < shape[0]:
            raise ValueError(f"{path}:{name} has no leading-axis step {step}")
        if fortran_order:
            # A leading-axis slice is not contiguous in a Fortran-order bundle.
            with np.load(path, allow_pickle=False) as payload:
                return np.array(payload[name][step], copy=True, order="F")
        count = int(np.prod(shape[1:], dtype=np.int64))
        stream.seek(step * count * dtype.itemsize, 1)
        data = stream.read(count * dtype.itemsize)
        if len(data) != count * dtype.itemsize:
            raise ValueError(f"truncated array {path}:{name}")
        return np.frombuffer(data, dtype=dtype).reshape(shape[1:]).copy(order="F")


def extract(replay: Path, output: Path, *, step: int = 0) -> dict[str, object]:
    manifest = json.loads((replay / "manifest.json").read_text())
    if manifest.get("storage") != "rank_bundle_v1":
        raise ValueError("source replay must use rank_bundle_v1 storage")
    rank_count = int(manifest["rank_count"])
    step_count = int(manifest["step_count"])
    if not 0 <= step < step_count:
        raise ValueError(f"step must be in 0..{step_count - 1}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"completed bootstrap already exists: {output}")
    import_shapes: set[tuple[int, ...]] = set()
    export_shapes: set[tuple[int, ...]] = set()
    total_bytes = 0
    for rank in range(rank_count):
        source = replay / str(manifest["file_pattern"]).format(rank=rank)
        destination = output / f"rank-{rank:04d}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as payload:
                x2a = np.array(payload["x2a_rattr"], copy=False)
                a2x = np.array(payload["a2x_rattr"], copy=False)
        else:
            x2a = _read_step(source, "x2a_rattr", step)
            a2x = _read_step(source, "a2x_rattr", step)
            np.savez(destination, x2a_rattr=x2a, a2x_rattr=a2x)
        import_shapes.add(tuple(int(value) for value in x2a.shape))
        export_shapes.add(tuple(int(value) for value in a2x.shape))
        total_bytes += destination.stat().st_size
    result: dict[str, object] = {
        "schema_version": 1,
        "storage": "rank_bootstrap_v1",
        "rank_count": rank_count,
        "file_pattern": "rank-{rank:04d}.npz",
        "source_step": int(step),
        "source_replay": str(replay.resolve()),
        "source_config_fingerprint": manifest.get("config_fingerprint"),
        "rank_local_shapes": {
            "import": [list(shape) for shape in sorted(import_shapes)],
            "export": [list(shape) for shape in sorted(export_shapes)],
        },
        "bootstrap_bytes": total_bytes,
    }
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--step", default=0, type=int)
    args = parser.parse_args()
    print(json.dumps(extract(args.replay, args.output, step=args.step), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
