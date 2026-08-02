#!/usr/bin/env python3
"""Capture the exact external iCESM source and PI-atm input revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Iterable


REPOSITORIES = (
    ".",
    "cime",
    "components/cam",
    "components/clm",
    "components/cice",
    "components/rtm",
    "components/pop",
)


def _run(*args: str, cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_files(paths: Iterable[Path]) -> dict[str, str]:
    return {
        str(path): _sha256(path)
        for path in sorted(paths)
        if path.is_file()
    }


def capture(source_root: Path, mapping_root: Path) -> dict[str, object]:
    repositories: dict[str, object] = {}
    for relative in REPOSITORIES:
        root = source_root / relative
        if not root.exists():
            continue
        repositories[relative] = {
            "commit": _run("git", "rev-parse", "HEAD", cwd=root),
            "tracked_diff_sha256": hashlib.sha256(
                _run("git", "diff", "--binary", "--", cwd=root).encode()
            ).hexdigest(),
            "tracked_changes": tuple(
                line
                for line in _run(
                    "git", "status", "--short", "--untracked-files=no", cwd=root
                ).splitlines()
                if line
            ),
        }

    mapping_names = (
        "domain.lnd.ne16np4_gx1v6.231103.nc",
        "domain.ocn.gx1v6.231103.nc",
        "map_ne16np4_TO_gx1v6_aave.231103.nc",
        "map_ne16np4_TO_gx1v6_blin.231103.nc",
        "map_ne16np4_TO_gx1v6_patc.231103.nc",
        "map_gx1v6_TO_ne16np4_aave.231103.nc",
        "map_ne16np4_TO_r05_nomask_aave.231103.nc",
        "map_r05_nomask_TO_ne16np4_aave.231103.nc",
        "map_r05_nomask_TO_gx1v6_aave.231103.nc",
        "map_r05_nomask_to_gx1v6_nnsm_e1000r300.231103.nc",
    )
    return {
        "schema_version": 1,
        "source_root": str(source_root.resolve()),
        "repositories": repositories,
        "mapping_root": str(mapping_root.resolve()),
        "mapping_sha256": _hash_files(mapping_root / name for name in mapping_names),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mapping-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = capture(args.source_root, args.mapping_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
