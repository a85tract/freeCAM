#!/usr/bin/env python3
"""Apply the validated PI-CAM Python-control surface to a clean CAM checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[1]
PATCHES = (
    "native/pi_cam/patches/0001-capture-cam-boundary.patch",
    "native/pi_cam/control_patches/0010-python-atm-phase-control.patch",
    "native/pi_cam/control_patches/0011-python-cam-phase-control.patch",
    "native/pi_cam/control_patches/0013-python-physics-phase-control.patch",
    "native/pi_cam/control_patches/0015-python-after-coupler-scheme-control.patch",
    "native/pi_cam/control_patches/0017-python-before-coupler-scheme-control.patch",
    "native/pi_cam/control_patches/0024-startup-boundary-capture.patch",
    "native/pi_cam/control_patches/0025-python-owned-atm-mct-state.patch",
    "native/pi_cam/control_patches/0030-order-independent-scheme-actions.patch",
    "native/pi_cam/control_patches/0033-single-cam-online-coupler.patch",
    "native/pi_cam/control_patches/0039-macro-tend-boundary.patch",
    "native/pi_cam/control_patches/0041-rad-tend-boundary.patch",
)

# Modules this repository owns that are added to the CAM source tree.  They
# are additions, never replacements: no patch edits a numerical object to
# call them, so the oracle's machine code is untouched and they are reached
# only from Python.
SUPPORT_SOURCES = (
    ("native/pi_cam/support/pycam_macro_kernels.F90",
     "src/physics/cam/pycam_macro_kernels.F90"),
    ("native/pi_cam/support/pycam_macro_handles.F90",
     "src/physics/cam/pycam_macro_handles.F90"),
    ("native/pi_cam/support/pycam_rad_kernels.F90",
     "src/physics/cam/pycam_rad_kernels.F90"),
    ("native/pi_cam/support/pycam_rad_handles.F90",
     "src/physics/cam/pycam_rad_handles.F90"),
    ("native/pi_cam/support/pycam_micro_kernels.F90",
     "src/physics/cam/pycam_micro_kernels.F90"),
    ("native/pi_cam/support/pycam_mm_kernels.F90",
     "src/physics/cam/pycam_mm_kernels.F90"),
)


def _install_support_sources(cam_root: Path) -> tuple[Path, ...]:
    installed = []
    for relative, destination in SUPPORT_SOURCES:
        target = cam_root / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
        installed.append(target)
    return tuple(installed)


def _cam_root(source_root: Path) -> Path:
    source_root = source_root.resolve()
    component = source_root / "components/cam"
    root = component if component.is_dir() else source_root
    required = root / "src/cpl/atm_comp_mct.F90"
    if not required.is_file():
        raise FileNotFoundError(f"not a CAM source checkout: {root}")
    return root


def _apply_to(cam_root: Path) -> tuple[Path, ...]:
    _install_support_sources(cam_root)
    applied: list[Path] = []
    for relative in PATCHES:
        patch = REPO / relative
        command = ["git", "apply", "--unidiff-zero", "--verbose", str(patch)]
        checked = subprocess.run(
            [*command[:2], "--check", *command[2:]],
            cwd=cam_root,
            check=True,
            capture_output=True,
            text=True,
        )
        diagnostic = checked.stdout + checked.stderr
        if "Skipped patch" in diagnostic:
            raise RuntimeError(
                f"git skipped PI-CAM patch instead of applying it: {patch}"
            )
        applied_result = subprocess.run(
            command,
            cwd=cam_root,
            check=True,
            capture_output=True,
            text=True,
        )
        diagnostic = applied_result.stdout + applied_result.stderr
        if "Skipped patch" in diagnostic:
            raise RuntimeError(
                f"git skipped PI-CAM patch instead of applying it: {patch}"
            )
        applied.append(patch)
    return tuple(applied)


def apply_patches(source_root: Path, *, check: bool = False) -> tuple[Path, ...]:
    """Apply the production patch subset in its validated dependency order."""

    cam_root = _cam_root(source_root)
    if not check:
        return _apply_to(cam_root)
    relative_sources = (
        Path("src/cpl/atm_comp_mct.F90"),
        Path("src/control/cam_comp.F90"),
        Path("src/physics/cam/physpkg.F90"),
    )
    with tempfile.TemporaryDirectory(prefix="pycam-pi-cam-patches-") as temporary:
        scratch = Path(temporary)
        for relative in relative_sources:
            destination = scratch / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cam_root / relative, destination)
        return _apply_to(scratch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="iCESM source root or its components/cam checkout",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate that every patch applies without modifying the checkout",
    )
    args = parser.parse_args()
    for patch in apply_patches(args.source_root, check=args.check):
        print(patch.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
