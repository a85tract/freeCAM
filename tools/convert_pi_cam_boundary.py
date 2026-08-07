#!/usr/bin/env python3
"""Convert raw Fortran PI-CAM boundary captures into replayable rank files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import struct

import numpy as np

from freecam.pi_cam import PICAMConfig


MAGIC = b"PYCAM_BOUNDARY1 "
NAME = re.compile(
    r"\.step-(?P<step>\d{6})\.rank-(?P<rank>\d{6})\."
    r"(?P<direction>import|export)\.bin$"
)


def read_capture(path: Path) -> tuple[dict[str, int], np.ndarray]:
    raw = path.read_bytes()
    if raw[:16] != MAGIC:
        raise RuntimeError(f"{path}: invalid PI-CAM boundary magic")
    if len(raw) < 44:
        raise RuntimeError(f"{path}: truncated PI-CAM boundary header")
    header_values = None
    byte_order = ""
    for candidate in (">", "<"):
        values = struct.unpack(candidate + "7i", raw[16:44])
        version, step, rank, size, direction, nattr, nlocal = values
        if version == 1 and direction in {1, 2} and min(nattr, nlocal) >= 0:
            header_values = values
            byte_order = candidate
            break
    if header_values is None:
        raise RuntimeError(f"{path}: invalid PI-CAM boundary header")
    version, step, rank, size, direction, nattr, nlocal = header_values
    values = np.frombuffer(raw, dtype=byte_order + "f8", offset=44)
    if values.size != nattr * nlocal:
        raise RuntimeError(
            f"{path}: expected {nattr * nlocal} values, found {values.size}"
        )
    return (
        {
            "step": step,
            "rank": rank,
            "size": size,
            "direction": direction,
            "nattr": nattr,
            "nlocal": nlocal,
        },
        np.asarray(
            values.reshape((nattr, nlocal), order="F"), dtype=np.float64
        ).copy(order="F"),
    )


def convert(
    prefix: Path,
    output: Path,
    config: PICAMConfig,
    *,
    workers: int = 32,
) -> dict[str, object]:
    paths = sorted(prefix.parent.glob(prefix.name + ".step-*.rank-*.*.bin"))
    if not paths:
        raise RuntimeError(f"no raw boundary files match {prefix}")
    observed: set[tuple[int, int, str]] = set()
    sizes: set[int] = set()
    steps: set[int] = set()
    ranks: set[int] = set()
    dimensions: dict[str, set[tuple[int, int]]] = {"import": set(), "export": set()}
    captured_imports: dict[tuple[int, int], np.ndarray] = {}
    captured_exports: dict[tuple[int, int], np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        captures = executor.map(read_capture, paths)
        for path, (header, values) in zip(paths, captures):
            match = NAME.search(path.name)
            if match is None:
                raise RuntimeError(f"unexpected capture file name {path}")
            direction = match.group("direction")
            key = (header["step"], header["rank"], direction)
            if key in observed:
                raise RuntimeError(f"duplicate boundary capture {key}")
            observed.add(key)
            sizes.add(header["size"])
            steps.add(header["step"])
            ranks.add(header["rank"])
            dimensions[direction].add((header["nattr"], header["nlocal"]))
            destination = (
                captured_imports if direction == "import" else captured_exports
            )
            destination[(header["step"], header["rank"])] = values
    if len(sizes) != 1:
        raise RuntimeError(f"captures disagree about MPI size: {sorted(sizes)}")
    rank_count = sizes.pop()
    export_steps = {step for step, _ in captured_exports}
    step_count = max(export_steps) + 1
    expected_exports = {
        (step, rank) for step in range(step_count) for rank in range(rank_count)
    }
    missing_exports = sorted(expected_exports - set(captured_exports))
    if missing_exports:
        raise RuntimeError(
            "boundary capture is missing a CAM export; first missing entry "
            f"{missing_exports[0]}"
        )

    held_import_steps: list[int] = []
    for step in range(step_count):
        present = [
            (step, rank) in captured_imports for rank in range(rank_count)
        ]
        if any(present) and not all(present):
            raise RuntimeError(
                f"boundary import step {step} is present on only some MPI ranks"
            )
        if not any(present):
            held_import_steps.append(step)

    # A CESM coupling call can execute more than one internal CAM action while
    # holding the same x2a import fixed.  The capture therefore records the
    # import once and multiple exports.  Materialize the held input for every
    # replay action so the Python driver has one complete rank file per step.
    reconstructed_imports = 0
    output.mkdir(parents=True, exist_ok=True)
    for rank in range(rank_count):
        current: np.ndarray | None = None
        rank_imports: list[np.ndarray] = []
        rank_exports: list[np.ndarray] = []
        for step in range(step_count):
            captured = captured_imports.get((step, rank))
            if captured is not None:
                current = captured
            if current is None:
                raise RuntimeError(
                    f"rank {rank} has no boundary import at or before step {step}"
                )
            if captured is None:
                reconstructed_imports += 1
            rank_imports.append(current)
            rank_exports.append(captured_exports[(step, rank)])
        destination = output / f"rank-{rank:04d}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                x2a_rattr=np.stack(rank_imports, axis=0),
                a2x_rattr=np.stack(rank_exports, axis=0),
            )
        temporary.replace(destination)
    manifest = {
        "schema_version": 1,
        "case_name": config.case_name,
        "config_fingerprint": config.fingerprint,
        "rank_count": rank_count,
        "step_count": step_count,
        "storage": "rank_bundle_v1",
        "file_pattern": "rank-{rank:04d}.npz",
        "fields": {
            "import": {"x2a_rattr": "rank-local MCT x2a real attributes"},
            "export": {"a2x_rattr": "rank-local MCT a2x real attributes"},
        },
        "rank_local_shapes": {
            name: [list(shape) for shape in sorted(shapes)]
            for name, shapes in dimensions.items()
        },
        "raw_file_count": len(paths),
        "reconstructed_held_imports": reconstructed_imports,
        "held_import_steps": held_import_steps,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    result = convert(
        args.capture_prefix,
        args.output,
        PICAMConfig.from_yaml(args.config),
        workers=args.workers,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.summary is None:
        print(text, end="")
    else:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text)
        print(args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
