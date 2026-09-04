"""The shared YAML table loader: one parse per file, invalidated when the file changes."""

from __future__ import annotations

import os
import time

from freecam.pi_cam import tables


def test_same_file_is_parsed_once(tmp_path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text("fields:\n  - name: a\n    symbol: s\n")
    first = tables.load_table(path)
    second = tables.load_table(str(path))
    assert first is second
    assert first == {"fields": [{"name": "a", "symbol": "s"}]}


def test_a_changed_file_is_read_again(tmp_path) -> None:
    path = tmp_path / "t.yaml"
    path.write_text("value: 1\n")
    assert tables.load_table(path) == {"value": 1}
    path.write_text("value: 22\n")
    stamp = time.time() + 5
    os.utime(path, (stamp, stamp))
    assert tables.load_table(path) == {"value": 22}


def test_kernel_descriptors_load_through_the_cache() -> None:
    from freecam.pi_cam.kernel_codegen import load_direct_kernels
    from freecam.physics.macrophysics import Macrophysics

    once = load_direct_kernels(Macrophysics.DESCRIPTORS)
    again = load_direct_kernels(Macrophysics.DESCRIPTORS)
    assert [k.name for k in once] == [k.name for k in again]
