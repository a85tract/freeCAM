#!/usr/bin/env python3
"""Build a boundary-capture CESM executable from the pristine numerical image."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

from build_pi_cam_devices import (
    _compile_command,
    _compile_to,
    _link_command,
    _replace_archive,
    _replace_library,
    _run,
    _xml,
)


REPO = Path(__file__).resolve().parents[1]
PATCHES = (
    REPO / "native/pi_cam/patches/0001-capture-cam-boundary.patch",
    REPO / "native/pi_cam/control_patches/0024-startup-boundary-capture.patch",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(case: Path, source_root: Path, output: Path) -> dict[str, object]:
    case = case.resolve()
    source_root = source_root.resolve()
    output = output.resolve()
    build_root = Path(_xml(case, "EXEROOT")).resolve()
    work = output.parent / "capture_objects"
    if work.exists():
        shutil.rmtree(work)
    source = work / "src/cpl/atm_comp_mct.F90"
    source.parent.mkdir(parents=True)
    shutil.copy2(
        source_root / "components/cam/src/cpl/atm_comp_mct.F90",
        source,
    )
    for patch in PATCHES:
        _run(
            ["git", "apply", "--unidiff-zero", "--verbose", str(patch)],
            cwd=work,
        )

    _, compile_command = _compile_command(build_root, "atm_comp_mct.F90")
    object_path = work / "atm_comp_mct.o"
    compiled = _compile_to(
        compile_command,
        "atm_comp_mct.F90",
        source,
        object_path,
        work,
        build_root / "atm/obj",
    )
    archive = output.parent / "libatm_boundary_capture.a"
    _replace_archive(
        build_root / "lib/libatm.a",
        archive,
        (object_path,),
    )
    _, original_link = _link_command(build_root)
    link = _replace_library(original_link, archive)
    link[link.index("-o") + 1] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(link, cwd=build_root / "cpl/obj")
    return {
        "schema_version": 1,
        "case": str(case),
        "source_root": str(source_root),
        "output": str(output),
        "output_sha256": _sha256(output),
        "patches": {str(path.relative_to(REPO)): _sha256(path) for path in PATCHES},
        "compile_command": compiled,
        "link_command": link,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO / "external/iCESM1.3.1_fzhu",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.case, args.source_root, args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
