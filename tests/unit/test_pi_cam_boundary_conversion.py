import importlib.util
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "convert_pi_cam_boundary",
    PROJECT / "tools/convert_pi_cam_boundary.py",
)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)
MAGIC_V1 = CONVERTER.MAGIC_V1
MAGIC_V2 = CONVERTER.MAGIC_V2
convert = CONVERTER.convert
read_capture = CONVERTER.read_capture


def _record(
    magic: bytes,
    *,
    version: int,
    step: int,
    rank: int,
    size: int,
    direction: int,
    values: np.ndarray,
) -> bytes:
    header = struct.pack(
        "<7i",
        version,
        step,
        rank,
        size,
        direction,
        values.shape[0],
        values.shape[1],
    )
    return magic + header + np.asarray(values, dtype="<f8").tobytes(order="F")


def test_read_capture_preserves_legacy_v1_format(tmp_path) -> None:
    values = np.arange(6.0).reshape((2, 3), order="F")
    path = tmp_path / "pi_cam.step-000000.rank-000000.import.bin"
    path.write_bytes(
        _record(
            MAGIC_V1,
            version=1,
            step=0,
            rank=0,
            size=1,
            direction=1,
            values=values,
        )
    )

    header, loaded = read_capture(path)

    assert header["direction"] == 1
    assert np.array_equal(loaded, values)


def test_stream_v2_conversion_writes_one_replay_bundle_per_rank(tmp_path) -> None:
    prefix = tmp_path / "raw" / "pi_cam"
    prefix.parent.mkdir()
    for rank in range(2):
        imported = np.arange(6.0).reshape((2, 3), order="F") + rank * 10
        export0 = np.arange(8.0).reshape((2, 4), order="F") + rank * 20
        export1 = export0 + 100
        payload = b"".join(
            (
                _record(
                    MAGIC_V2,
                    version=2,
                    step=0,
                    rank=rank,
                    size=2,
                    direction=1,
                    values=imported,
                ),
                _record(
                    MAGIC_V2,
                    version=2,
                    step=0,
                    rank=rank,
                    size=2,
                    direction=2,
                    values=export0,
                ),
                _record(
                    MAGIC_V2,
                    version=2,
                    step=1,
                    rank=rank,
                    size=2,
                    direction=2,
                    values=export1,
                ),
            )
        )
        (prefix.parent / f"pi_cam.rank-{rank:06d}.stream.bin").write_bytes(
            payload
        )

    output = tmp_path / "replay"
    manifest = convert(
        prefix,
        output,
        SimpleNamespace(case_name="test", fingerprint="abc"),
    )

    assert manifest["storage"] == "rank_pread_v1"
    assert manifest["source_storage"] == "rank_stream_v2"
    assert manifest["rank_count"] == 2
    assert manifest["step_count"] == 2
    assert manifest["raw_file_count"] == 2
    assert manifest["raw_record_count"] == 6
    assert manifest["held_import_steps"] == [1]
    assert manifest["reconstructed_held_imports"] == 2
    imported = np.load(
        output / "rank-0001-import.npy", mmap_mode="r", allow_pickle=False
    )
    exported = np.load(
        output / "rank-0001-export.npy", mmap_mode="r", allow_pickle=False
    )
    assert np.array_equal(imported[0], imported[1])
    assert np.array_equal(exported[1], export1)
