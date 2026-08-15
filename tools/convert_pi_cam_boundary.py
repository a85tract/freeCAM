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


MAGIC_V1 = b"PYCAM_BOUNDARY1 "
MAGIC_V2 = b"PYCAM_BOUNDARY2 "
NAME = re.compile(
    r"\.step-(?P<step>\d{6})\.rank-(?P<rank>\d{6})\."
    r"(?P<direction>import|export)\.bin$"
)
STREAM_NAME = re.compile(r"\.rank-(?P<rank>\d{6})\.stream\.bin$")


def _decode_header(
    raw: bytes,
    *,
    version: int,
    path: Path,
) -> tuple[tuple[int, ...], str]:
    if len(raw) != 28:
        raise RuntimeError(f"{path}: truncated PI-CAM boundary header")
    for byte_order in (">", "<"):
        values = struct.unpack(byte_order + "7i", raw)
        found_version, _, _, _, direction, nattr, nlocal = values
        if (
            found_version == version
            and direction in {1, 2}
            and min(nattr, nlocal) >= 0
        ):
            return values, byte_order
    raise RuntimeError(f"{path}: invalid PI-CAM boundary header")


def read_capture(path: Path) -> tuple[dict[str, int], np.ndarray]:
    raw = path.read_bytes()
    if raw[:16] != MAGIC_V1:
        raise RuntimeError(f"{path}: invalid PI-CAM boundary magic")
    if len(raw) < 44:
        raise RuntimeError(f"{path}: truncated PI-CAM boundary header")
    header_values, byte_order = _decode_header(
        raw[16:44], version=1, path=path
    )
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


def iter_stream_captures(
    path: Path,
):
    """Yield records from one rank-local stream without loading it all."""

    with path.open("rb") as stream:
        record = 0
        while True:
            magic = stream.read(16)
            if not magic:
                return
            record += 1
            if len(magic) != 16:
                raise RuntimeError(
                    f"{path}: truncated magic in stream record {record}"
                )
            if magic != MAGIC_V2:
                raise RuntimeError(
                    f"{path}: invalid magic in stream record {record}"
                )
            header_values, byte_order = _decode_header(
                stream.read(28), version=2, path=path
            )
            _, step, rank, size, direction, nattr, nlocal = header_values
            byte_count = 8 * nattr * nlocal
            raw_values = stream.read(byte_count)
            if len(raw_values) != byte_count:
                raise RuntimeError(
                    f"{path}: truncated values in stream record {record}"
                )
            values = np.frombuffer(raw_values, dtype=byte_order + "f8")
            yield (
                {
                    "step": step,
                    "rank": rank,
                    "size": size,
                    "direction": direction,
                    "nattr": nattr,
                    "nlocal": nlocal,
                },
                np.asarray(
                    values.reshape((nattr, nlocal), order="F"),
                    dtype=np.float64,
                ).copy(order="F"),
            )


def _write_rank_bundle(
    output: Path,
    *,
    rank: int,
    step_count: int,
    captured_imports: dict[int, np.ndarray],
    captured_exports: dict[int, np.ndarray],
) -> tuple[int, list[int]]:
    missing_exports = [
        step for step in range(step_count) if step not in captured_exports
    ]
    if missing_exports:
        raise RuntimeError(
            "boundary capture is missing a CAM export; first missing entry "
            f"({missing_exports[0]}, {rank})"
        )
    present_imports = set(captured_imports)
    held_import_steps = [
        step for step in range(step_count) if step not in present_imports
    ]
    first_import = min(present_imports, default=None)
    if first_import != 0:
        raise RuntimeError(
            f"rank {rank} has no boundary import at or before step 0"
        )
    import_shape = captured_imports[first_import].shape
    export_shape = captured_exports[0].shape
    import_destination = output / f"rank-{rank:04d}-import.npy"
    export_destination = output / f"rank-{rank:04d}-export.npy"
    import_temporary = import_destination.with_suffix(".npy.tmp")
    export_temporary = export_destination.with_suffix(".npy.tmp")
    rank_imports = np.lib.format.open_memmap(
        import_temporary,
        mode="w+",
        dtype=np.float64,
        shape=(step_count, *import_shape),
    )
    rank_exports = np.lib.format.open_memmap(
        export_temporary,
        mode="w+",
        dtype=np.float64,
        shape=(step_count, *export_shape),
    )
    current: np.ndarray | None = None
    reconstructed_imports = 0
    for step in range(step_count):
        captured = captured_imports.get(step)
        if captured is not None:
            if captured.shape != import_shape:
                raise RuntimeError(
                    f"rank {rank} import shape changed at step {step}"
                )
            current = captured
        if current is None:  # pragma: no cover - guarded by first_import
            raise RuntimeError(
                f"rank {rank} has no boundary import at or before step {step}"
            )
        exported = captured_exports[step]
        if exported.shape != export_shape:
            raise RuntimeError(
                f"rank {rank} export shape changed at step {step}"
            )
        if captured is None:
            reconstructed_imports += 1
        rank_imports[step] = current
        rank_exports[step] = exported
    rank_imports.flush()
    rank_exports.flush()
    del rank_imports, rank_exports
    import_temporary.replace(import_destination)
    export_temporary.replace(export_destination)
    return reconstructed_imports, held_import_steps


def _convert_streams(
    paths: list[Path],
    output: Path,
    config: PICAMConfig,
) -> dict[str, object]:
    expected_files: dict[int, Path] = {}
    for path in paths:
        match = STREAM_NAME.search(path.name)
        if match is None:
            raise RuntimeError(f"unexpected stream capture file name {path}")
        rank = int(match.group("rank"))
        if rank in expected_files:
            raise RuntimeError(f"duplicate stream capture for MPI rank {rank}")
        expected_files[rank] = path

    output.mkdir(parents=True, exist_ok=True)
    rank_count: int | None = None
    step_count: int | None = None
    canonical_held_steps: list[int] | None = None
    dimensions: dict[str, set[tuple[int, int]]] = {
        "import": set(),
        "export": set(),
    }
    reconstructed_imports = 0
    record_count = 0
    for rank, path in sorted(expected_files.items()):
        captured_imports: dict[int, np.ndarray] = {}
        captured_exports: dict[int, np.ndarray] = {}
        observed: set[tuple[int, int]] = set()
        observed_size: int | None = None
        for header, values in iter_stream_captures(path):
            record_count += 1
            if header["rank"] != rank:
                raise RuntimeError(
                    f"{path}: header rank {header['rank']} does not match file"
                )
            if observed_size is None:
                observed_size = header["size"]
            elif observed_size != header["size"]:
                raise RuntimeError(f"{path}: MPI size changes between records")
            direction = header["direction"]
            key = (header["step"], direction)
            if key in observed:
                raise RuntimeError(
                    f"duplicate stream boundary capture {(rank, *key)}"
                )
            observed.add(key)
            name = "import" if direction == 1 else "export"
            dimensions[name].add((header["nattr"], header["nlocal"]))
            destination = (
                captured_imports if direction == 1 else captured_exports
            )
            destination[header["step"]] = values
        if observed_size is None or not captured_exports:
            raise RuntimeError(f"{path}: stream contains no CAM exports")
        if rank_count is None:
            rank_count = observed_size
        elif rank_count != observed_size:
            raise RuntimeError("stream captures disagree about MPI size")
        rank_step_count = max(captured_exports) + 1
        if step_count is None:
            step_count = rank_step_count
        elif step_count != rank_step_count:
            raise RuntimeError(
                f"rank {rank} has {rank_step_count} steps, expected {step_count}"
            )
        reconstructed, held_steps = _write_rank_bundle(
            output,
            rank=rank,
            step_count=rank_step_count,
            captured_imports=captured_imports,
            captured_exports=captured_exports,
        )
        reconstructed_imports += reconstructed
        if canonical_held_steps is None:
            canonical_held_steps = held_steps
        elif canonical_held_steps != held_steps:
            raise RuntimeError(
                f"rank {rank} has a different set of held import steps"
            )

    assert rank_count is not None
    assert step_count is not None
    expected_ranks = set(range(rank_count))
    if set(expected_files) != expected_ranks:
        missing = sorted(expected_ranks - set(expected_files))
        extra = sorted(set(expected_files) - expected_ranks)
        raise RuntimeError(
            f"stream rank files do not match MPI size; missing={missing[:1]}, "
            f"extra={extra[:1]}"
        )
    return {
        "schema_version": 1,
        "case_name": config.case_name,
        "config_fingerprint": config.fingerprint,
        "rank_count": rank_count,
        "step_count": step_count,
        "storage": "rank_pread_v1",
        "source_storage": "rank_stream_v2",
        "file_pattern": "rank-{rank:04d}-{direction}.npy",
        "fields": {
            "import": {"x2a_rattr": "rank-local MCT x2a real attributes"},
            "export": {"a2x_rattr": "rank-local MCT a2x real attributes"},
        },
        "rank_local_shapes": {
            name: [list(shape) for shape in sorted(shapes)]
            for name, shapes in dimensions.items()
        },
        "raw_file_count": len(paths),
        "raw_record_count": record_count,
        "reconstructed_held_imports": reconstructed_imports,
        "held_import_steps": canonical_held_steps or [],
    }


def convert(
    prefix: Path,
    output: Path,
    config: PICAMConfig,
    *,
    workers: int = 32,
) -> dict[str, object]:
    stream_paths = sorted(
        prefix.parent.glob(prefix.name + ".rank-*.stream.bin")
    )
    if stream_paths:
        manifest = _convert_streams(stream_paths, output, config)
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return manifest
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
