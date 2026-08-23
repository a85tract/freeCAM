#!/usr/bin/env python3
"""Build a CESM executable that captures physics-function arguments at call sites.

Starts from the pristine source and the oracle's own numerical archive, like
the boundary-capture build: the capture module and the two call-site patches
are compiled with the recovered production commands, the affected objects
are replaced in a copy of the oracle libatm.a, and the executable is relinked
into its own build directory.  Two 512-rank runs then prove it bit-for-bit
against the oracle, with capture on and with it off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi_cam_build_common import (  # noqa: E402
    compile_command,
    compile_to,
    link_command,
    replace_archive,
    replace_library,
    run,
    sha256,
    xml,
)

REPO = Path(__file__).resolve().parents[1]
CAPTURE_MODULE = REPO / "native/pi_cam/standalone/capture_support.F90"
# Each patch names the routine it brackets and the source file it edits.
PATCHES = (
    ("mmacro_pcond", REPO / "native/pi_cam/patches/0002-capture-mmacro-pcond.patch", "src/physics/cam/macrop_driver.F90"),
    ("dadadj", REPO / "native/pi_cam/patches/0003-capture-dadadj.patch", "src/physics/cam/physpkg.F90"),
)


def build(case: Path, source_root: Path, output: Path) -> dict[str, object]:
    case = case.resolve()
    source_root = source_root.resolve()
    output = output.resolve()
    build_root = Path(xml(case, "EXEROOT")).resolve()
    module_include = build_root / "atm/obj"
    work = output.parent / "capture_objects"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for _, _, relative in PATCHES:
        target = work / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / "components/cam" / relative, target)
    for _, patch, _ in PATCHES:
        run(["git", "apply", "--unidiff-zero", "--verbose", str(patch)], cwd=work)

    compiled: dict[str, list[str]] = {}
    objects: list[Path] = []
    # The capture module first: the patched files `use` it, and ifort writes
    # its .mod into the working directory, which -I. already covers.
    _, command = compile_command(build_root, "macrop_driver.F90")
    module_object = work / "pycam_function_capture.o"
    compiled["pycam_function_capture"] = compile_to(
        command, "macrop_driver.F90", CAPTURE_MODULE, module_object, work, module_include
    )
    objects.append(module_object)
    for _, _, relative in PATCHES:
        name = Path(relative).name
        _, command = compile_command(build_root, name)
        object_path = work / name.replace(".F90", ".o")
        compiled[name] = compile_to(command, name, work / relative, object_path, work, module_include)
        objects.append(object_path)

    archive = output.parent / "libatm_function_capture.a"
    replace_archive(
        build_root / "lib/libatm.a", archive, tuple(objects), additions=(module_object.name,)
    )
    _, original_link = link_command(build_root)
    link = replace_library(original_link, archive)
    link[link.index("-o") + 1] = str(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run(link, cwd=build_root / "cpl/obj")
    return {
        "schema_version": 1,
        "generator": "tools/build_pi_cam_function_capture.py",
        "case": str(case),
        "source_root": str(source_root),
        "numerical_archive": {"path": str(build_root / "lib/libatm.a"), "sha256": sha256(build_root / "lib/libatm.a")},
        "output": str(output),
        "output_sha256": sha256(output),
        "functions": [name for name, _, _ in PATCHES],
        "capture_module": {"path": str(CAPTURE_MODULE.relative_to(REPO)), "sha256": sha256(CAPTURE_MODULE)},
        "patches": {str(patch.relative_to(REPO)): sha256(patch) for _, patch, _ in PATCHES},
        "objects": {path.name: sha256(path) for path in objects},
        "archive": {"path": str(archive), "sha256": sha256(archive)},
        "compile_commands": compiled,
        "link_command": link,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO / "external/iCESM1.3.1_fzhu")
    parser.add_argument("--output", type=Path, default=REPO / "build/pi_cam_function_capture/pi_cam_function_capture.exe")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    manifest = build(args.case, args.source_root, args.output)
    manifest_path = args.manifest or args.output.with_name("manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{args.output} ({manifest['output_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
