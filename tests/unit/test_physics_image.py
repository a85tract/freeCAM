"""The standalone image runtime: bit comparison, and initialization against a snapshot."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from freecam.physics.image import ModuleStateMismatch, _first_difference, bitwise_equal

PROJECT = Path(__file__).resolve().parents[2]
DADADJ_MANIFEST = PROJECT / "build/pi_cam_standalone/dadadj/manifest.json"


def test_bitwise_comparison_reports_the_first_differing_element() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3, order="F")
    b = a.copy()
    assert bitwise_equal(a, b)
    b[1, 2] = np.nextafter(b[1, 2], np.inf)
    report = _first_difference(a, b)
    assert report["index"] == [1, 2] and report["count"] == 1
    assert report["reference_hex"] == float(a[1, 2]).hex()
    # Identical NaN payloads are equal bit for bit; differing payloads are not.
    nan_a = np.array([np.nan]); nan_b = np.array([np.nan])
    assert bitwise_equal(nan_a, nan_b)
    assert not bitwise_equal(np.array([1.0]), np.array([-1.0]))


def test_math_library_must_be_preloaded(monkeypatch, tmp_path: Path) -> None:
    from freecam.physics.image import MathLibraryNotPreloaded, require_math_library

    library = tmp_path / "libimf.so"
    library.write_bytes(b"")
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    with pytest.raises(MathLibraryNotPreloaded, match="LD_PRELOAD="):
        require_math_library(library)
    monkeypatch.setenv("LD_PRELOAD", f"/elsewhere/libfoo.so:{library}")
    require_math_library(library)


def _preloaded(manifest: Path) -> bool:
    if not manifest.is_file():
        return False
    from freecam.physics.image import MathLibraryNotPreloaded, require_math_library

    try:
        require_math_library(Path(json.loads(manifest.read_text())["intel_math_library"]))
    except (MathLibraryNotPreloaded, KeyError):
        return False
    return True


@pytest.mark.skipif(not _preloaded(DADADJ_MANIFEST), reason="dadadj image not built, or its libimf not preloaded")
def test_image_initializes_verifies_and_transacts_parameters() -> None:
    from freecam.physics.image import StandaloneImage, read_module_state

    image = StandaloneImage(DADADJ_MANIFEST)
    snapshot_path = PROJECT / "validation/pi_cam_dadadj_module_state.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else {
        "entries": read_module_state(image.library, image.spec.module_state)
    }
    verification = image.initialize(snapshot)
    assert verification["all_equal"] and image.parameters == {"nlvdry": 3}

    # A -> B -> A leaves every copy exactly where it began.
    image.set_parameters({"nlvdry": 6})
    assert image.parameters == {"nlvdry": 6}
    image.restore_parameters()
    assert image.parameters == {"nlvdry": 3}
    with pytest.raises(Exception, match="no parameter"):
        image.set_parameters({"rhminl": 0.9})
    with pytest.raises(Exception, match="integer"):
        image.set_parameters({"nlvdry": 2.5})

    # Verification is bit for bit: a snapshot that disagrees is refused.
    wrong = json.loads(json.dumps(snapshot))
    wrong["entries"]["cam_control_mod_mp_nlvdry_"]["sha256"] = "0" * 64
    wrong["entries"]["cam_control_mod_mp_nlvdry_"]["values"] = [4]
    with pytest.raises(ModuleStateMismatch, match="cam_control_mod_mp_nlvdry_"):
        StandaloneImage(DADADJ_MANIFEST).initialize(wrong)

    pool = image.empty_pool()
    assert pool["dadadj.t"].shape == (16, 30, 1) and pool["dadadj.t"].flags.f_contiguous
    assert pool["dadadj.pint"].shape == (16, 31, 1) and pool["dadadj.ncol"].dtype == np.int32
