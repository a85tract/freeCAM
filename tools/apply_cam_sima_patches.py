#!/usr/bin/env python3
"""Apply the small, pinned CAM-SIMA fixes required by the scientific gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _patch_command(
    cam_root: Path,
    patch_file: Path,
    *,
    dry_run: bool,
    reverse: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        "patch",
        "--batch",
        "--silent",
        "--ignore-whitespace",
        "-p1",
        "-d",
        str(cam_root),
    ]
    if dry_run:
        command.append("--dry-run")
    if reverse:
        command.append("--reverse")
    else:
        command.append("--forward")
    command.extend(["-i", str(patch_file)])
    return subprocess.run(command, text=True, capture_output=True)


def patch_state(cam_root: Path, patch_file: Path) -> str:
    if _patch_command(
        cam_root, patch_file, dry_run=True, reverse=False
    ).returncode == 0:
        return "missing"
    if _patch_command(
        cam_root, patch_file, dry_run=True, reverse=True
    ).returncode == 0:
        return "applied"
    raise RuntimeError(
        f"{patch_file.name} matches neither the pinned source nor the "
        "expected patched source"
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that every pinned patch is already applied",
    )
    args = parser.parse_args()

    cam_root = repo / "external/CAM-SIMA"
    patch_files = sorted((repo / "validation/patches").glob("*.patch"))
    if not patch_files:
        raise SystemExit("no CAM-SIMA patches found")

    records: list[dict[str, str]] = []
    missing: list[str] = []
    for patch_file in patch_files:
        state = patch_state(cam_root, patch_file)
        if state == "missing" and not args.check:
            result = _patch_command(
                cam_root, patch_file, dry_run=False, reverse=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"failed to apply {patch_file.name}: "
                    f"{result.stderr or result.stdout}"
                )
            state = "applied"
        elif state == "missing":
            missing.append(patch_file.name)
        records.append({"patch": patch_file.name, "state": state})

    print(
        json.dumps(
            {
                "cam_root": str(cam_root),
                "patches": records,
                "all_applied": not missing,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if missing:
        raise SystemExit(
            "required CAM-SIMA patches are missing: " + ", ".join(missing)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
