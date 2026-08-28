#!/usr/bin/env python3
"""Prepare a writable, patched PI-CAM source tree from the pinned iCESM external."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping


REPO = Path(__file__).resolve().parents[1]
PINNED_SOURCE = REPO / "external/iCESM1.3.1_fzhu"
DEFAULT_OUTPUT = REPO / "build/iCESM1.3.1_PI_cam_only"
PINNED_REVISIONS: Mapping[str, str] = {
    ".": "556ad1c112af6606299f8270ad6f8b9b16540332",
    "cime": "763b4a09d54c08bc0cc840af293e4ae79fa48fc6",
    "components/cam": "09c8a19c463122fad488e6fd1bbc0fcd0f410130",
    "components/cice": "8609c71127dba2e25ded6cf4f1eed6e6cd9c77e4",
    "components/clm": "5cc5a9838a05739dc4b97e2fce247a30967b9527",
    "components/pop": "dc1c67eb2bdbbcd4d54273a28c98e107316ee865",
    "components/rtm": "88fed403904f6fca99ab68117f11f29fb0ce726f",
}


def _revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_pinned_externals(source: Path, *, checkout: bool = True) -> dict[str, str]:
    """Populate iCESM externals and reject every source revision mismatch."""

    source = source.resolve()
    if not (source / "Externals.cfg").is_file():
        raise RuntimeError(
            f"pinned iCESM source is absent at {source}; run "
            "git submodule update --init external/iCESM1.3.1_fzhu"
        )
    missing = [
        relative
        for relative in PINNED_REVISIONS
        if not (
            source / ".git"
            if relative == "."
            else source / relative / ".git"
        ).exists()
    ]
    if missing and checkout:
        command = source / "manage_externals/checkout_externals"
        if not command.is_file():
            raise RuntimeError(f"iCESM checkout_externals is absent: {command}")
        subprocess.run([str(command)], cwd=source, check=True)
        missing = [
            relative
            for relative in PINNED_REVISIONS
            if not (
                source / ".git"
                if relative == "."
                else source / relative / ".git"
            ).exists()
        ]
    if missing:
        raise RuntimeError(
            "iCESM external components are absent: "
            + ", ".join(missing)
            + "; run external/iCESM1.3.1_fzhu/manage_externals/checkout_externals"
        )

    revisions: dict[str, str] = {}
    for relative, expected in PINNED_REVISIONS.items():
        path = source if relative == "." else source / relative
        actual = _revision(path)
        if actual != expected:
            raise RuntimeError(
                f"iCESM source revision mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
        revisions[relative] = actual
    return revisions


def prepare_source(source: Path, output: Path, *, force: bool = False) -> Path:
    """Copy pristine pinned sources, then apply the maintained PI-CAM patches."""

    source = source.resolve()
    output = output.resolve()
    revisions = ensure_pinned_externals(source)
    if output.exists():
        if not force:
            raise RuntimeError(f"prepared source already exists: {output}; use --force")
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        output,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "*.pyo"),
    )

    patch_tool = REPO / "tools/apply_pi_cam_source_patches.py"
    # The applier reports what it applied, and that report is the record.  A
    # second list kept by hand here drifts: it named ten patches while twelve
    # were being applied, silently omitting the macrophysics and radiation
    # stage boundaries from every prepared tree's provenance.
    with tempfile.TemporaryDirectory(prefix="pycam-prepare-") as temporary:
        report_path = Path(temporary) / "applied.json"
        subprocess.run(
            [
                sys.executable,
                str(patch_tool),
                "--source-root",
                str(output),
                "--report",
                str(report_path),
            ],
            cwd=REPO,
            check=True,
        )
        report = json.loads(report_path.read_text())
    manifest = {
        "schema_version": 1,
        "source_external": str(source),
        "output": str(output),
        "revisions": revisions,
        "applied_patches": {
            name: _sha256(REPO / name) for name in report["patches"]
        },
        "installed_support_sources": {
            name: _sha256(REPO / name) for name in report["support_sources"]
        },
    }
    manifest_path = output / ".pycam-source.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PINNED_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify pinned top-level and component revisions without copying",
    )
    parser.add_argument(
        "--no-checkout",
        action="store_true",
        help="fail instead of running checkout_externals when components are absent",
    )
    args = parser.parse_args()
    if args.check:
        payload = ensure_pinned_externals(args.source, checkout=not args.no_checkout)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    manifest = prepare_source(args.source, args.output, force=args.force)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
