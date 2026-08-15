#!/usr/bin/env python3
"""Link the PIC iCESM PI-atm build into one loadable native driver device."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _read_log(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            return stream.read()
    return path.read_text(errors="replace")


def _build_logs_newest_first(build_root: Path) -> tuple[Path, ...]:
    paths = sorted(
        (*build_root.glob("cesm.bldlog.*"), *build_root.glob("cesm.bldlog.*.gz")),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"no CESM build log below {build_root}")
    return tuple(paths)


def _commands(log_text: str) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in log_text.splitlines() if line.startswith("ftn ")]
    compile_lines = [line for line in lines if "cesm_driver.F90" in line and " -c " in line]
    link_lines = [line for line in lines if " -o " in line and "cesm.exe" in line]
    if not compile_lines or not link_lines:
        raise RuntimeError("could not recover compile/link commands from CESM build log")
    return shlex.split(compile_lines[-1]), shlex.split(link_lines[-1])


def _commands_from_build_logs(build_root: Path) -> tuple[Path, list[str], list[str]]:
    """Use the newest full-link log, skipping later incremental-build logs."""

    for path in _build_logs_newest_first(build_root):
        try:
            compile_command, link_command = _commands(_read_log(path))
        except RuntimeError:
            continue
        return path, compile_command, link_command
    raise RuntimeError(f"no complete CESM compile/link log below {build_root}")


def _compile_command_from_build_logs(
    build_root: Path,
    source_name: str,
) -> tuple[Path, list[str]]:
    """Recover the exact compile command for one original CESM source file."""

    for path in _build_logs_newest_first(build_root):
        lines = [
            line.strip()
            for line in _read_log(path).splitlines()
            if line.startswith("ftn ")
            and source_name in line
            and " -c " in line
        ]
        if lines:
            return path, shlex.split(lines[-1])
    raise RuntimeError(
        f"could not recover compile command for {source_name} below {build_root}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_dependency(line: str) -> str:
    """Remove the ASLR load address while retaining the resolved library path."""

    return re.sub(r"\s+\(0x[0-9a-fA-F]+\)$", "", line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPOSITORY_ROOT / "native/cesm/cesm_full_driver_adapter.F90",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "build/cesm/pi_atm/libpycesm_full.so",
    )
    parser.add_argument(
        "--control-source",
        type=Path,
        help=(
            "patched cesm_comp_mod.F90 to compile before linking; defaults to "
            "the case SRCROOT"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "validation/pi_atm_full_native.json",
    )
    parser.add_argument(
        "--source-patch",
        type=Path,
        action="append",
        dest="source_patches",
        help="source patch recorded in the build manifest; may be repeated",
    )
    args = parser.parse_args()

    case = args.case.resolve()
    query = subprocess.run(
        [str(case / "xmlquery"), "EXEROOT", "--value"],
        cwd=case,
        check=True,
        capture_output=True,
        text=True,
    )
    build_root = Path(query.stdout.strip()).resolve()
    source_query = subprocess.run(
        [str(case / "xmlquery"), "SRCROOT", "--value"],
        cwd=case,
        check=True,
        capture_output=True,
        text=True,
    )
    source_root = Path(source_query.stdout.strip()).resolve()
    object_root = build_root / "cpl" / "obj"
    log_path, compile_command, link_command = _commands_from_build_logs(build_root)

    control_source = (
        args.control_source.resolve()
        if args.control_source is not None
        else source_root / "cime/src/drivers/mct/main/cesm_comp_mod.F90"
    )
    if not control_source.is_file():
        raise FileNotFoundError(f"missing patched CESM control source: {control_source}")
    control_log, control_compile_command = _compile_command_from_build_logs(
        build_root,
        "cesm_comp_mod.F90",
    )
    control_source_index = next(
        index
        for index, value in enumerate(control_compile_command)
        if value.endswith("cesm_comp_mod.F90")
    )
    control_compile_command[control_source_index] = str(control_source)
    if "-fPIC" not in control_compile_command:
        control_compile_command.insert(1, "-fPIC")
    subprocess.run(control_compile_command, cwd=object_root, check=True)

    adapter = args.adapter.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    adapter_object = object_root / f"{adapter.stem}.o"

    source_index = next(
        index for index, value in enumerate(compile_command) if value.endswith("cesm_driver.F90")
    )
    compile_command[source_index] = str(adapter)
    if "-fPIC" not in compile_command:
        compile_command.insert(1, "-fPIC")
    subprocess.run(compile_command, cwd=object_root, check=True)

    executable_index = link_command.index("-o") + 1
    link_command[executable_index] = str(output)
    link_command.insert(1, "-shared")
    link_command.insert(2, "-fPIC")
    link_command = [item for item in link_command if item != "cesm_driver.o"]
    first_object = next(
        index for index, item in enumerate(link_command) if item.endswith("cesm_comp_mod.o")
    )
    link_command.insert(first_object + 1, adapter_object.name)
    subprocess.run(link_command, cwd=object_root, check=True)

    dynamic_symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    required = (
        "pycesm_full_initialize_v1",
        "pycesm_full_step_v1",
        "pycesm_full_advance_v1",
        "pycesm_full_action_v1",
        "pycesm_full_step_begin_v1",
        "pycesm_full_step_end_v1",
        "pycesm_full_finalize_v1",
    )
    missing = [name for name in required if name not in dynamic_symbols]
    if missing:
        raise RuntimeError(f"shared library is missing symbols: {missing}")

    source_patches = tuple(
        path.resolve()
        for path in (
            args.source_patches
            or (
                REPOSITORY_ROOT
                / "native/cesm/patches/0001-external-mpi-and-one-step.patch",
                REPOSITORY_ROOT
                / "native/cesm/patches/0003-pic-build.patch",
                REPOSITORY_ROOT
                / "native/cesm/patches/0007-live-action-dispatch.patch",
                REPOSITORY_ROOT
                / "native/cesm/patches/0008-python-step-control.patch",
            )
        )
    )
    missing_patches = tuple(path for path in source_patches if not path.is_file())
    if missing_patches:
        raise FileNotFoundError(f"missing source patches: {missing_patches}")
    dependencies = subprocess.run(
        ["ldd", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    unresolved = [line.strip() for line in dependencies if "not found" in line]
    if unresolved:
        raise RuntimeError(f"shared library has unresolved dependencies: {unresolved}")

    manifest = {
        "schema_version": 1,
        "case": str(case),
        "build_root": str(build_root),
        "build_log": str(log_path),
        "control_build_log": str(control_log),
        "control_source": str(control_source),
        "adapter": str(adapter),
        "library": str(output),
        "library_sha256": _sha256(output),
        "library_bytes": output.stat().st_size,
        "source_patches": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in source_patches
        ],
        "symbols": list(required),
        "dependencies": [_stable_dependency(line) for line in dependencies],
        "compile_command": compile_command,
        "control_compile_command": control_compile_command,
        "link_command": link_command,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
