#!/usr/bin/env python3
"""Build the callback-free coupled-component image used by online FreeCAM.

The result is one ET_DYN image loaded by each of FreeCAM's 512 MPI ranks.
Python calls its bind(C) kernels explicitly; the image never calls Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from _pi_cam_coupler_build import (
    REPOSITORY_ROOT,
    _commands_from_build_logs,
    _compile_command_from_build_logs,
    _read_log,
    _sha256,
    _stable_dependency,
)


def _component_compile_command(
    build_root: Path, source_name: str
) -> tuple[Path, list[str]]:
    """Recover component compile commands from all CESM build logs."""

    logs = sorted(
        (*build_root.glob("*.bldlog.*"), *build_root.glob("*.bldlog.*.gz")),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in logs:
        lines = [
            line.strip()
            for line in _read_log(path).splitlines()
            if line.startswith("ftn ") and source_name in line and " -c " in line
        ]
        if lines:
            import shlex

            return path, shlex.split(lines[-1])
    raise RuntimeError(
        f"could not recover compile command for {source_name} below {build_root}"
    )


def _xml_value(case: Path, name: str) -> str:
    result = subprocess.run(
        [str(case / "xmlquery"), name, "--value"],
        cwd=case,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _compile_to(
    command: list[str],
    *,
    source_suffix: str,
    source: Path,
    output: Path,
    cwd: Path,
    module_include: Path,
) -> list[str]:
    result = list(command)
    source_index = next(
        index for index, value in enumerate(result) if value.endswith(source_suffix)
    )
    result[source_index] = str(source)
    result = [value for value in result if value != "-fPIC"]
    # The staging directory must stay first so the adapter sees the newly
    # generated cesm_comp_mod.mod.  The original CPL object directory supplies
    # every other module file without recompiling the numerical objects.
    first_local_include = result.index("-I.")
    result.insert(first_local_include + 1, f"-I{module_include}")
    result.extend(("-o", str(output)))
    subprocess.run(result, cwd=cwd, check=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--control-source", required=True, type=Path)
    parser.add_argument("--component-source", type=Path)
    parser.add_argument("--cam-control-source", type=Path)
    parser.add_argument("--physics-control-source", type=Path)
    parser.add_argument("--atm-source", type=Path)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=REPOSITORY_ROOT / "native/pi_cam/cesm/cesm_full_driver_adapter.F90",
    )
    parser.add_argument(
        "--main-source",
        type=Path,
        default=REPOSITORY_ROOT / "native/pi_cam/cesm/native_coupler_main.c",
    )
    parser.add_argument(
        "--allocator-interposer",
        type=Path,
        default=(
            REPOSITORY_ROOT / "native/pi_cam/cesm/intel_allocator_interposer.c"
        ),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=REPOSITORY_ROOT / "build/cesm/pi_atm/callback_free_build",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "build/cesm/pi_atm/production-components/libpycesm_external_atm.so"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "validation/pi_cam_external_atm_build.json",
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
    build_root = Path(_xml_value(case, "EXEROOT")).resolve()
    source_root = Path(_xml_value(case, "SRCROOT")).resolve()
    object_root = build_root / "cpl/obj"
    link_log, driver_compile, link_command = _commands_from_build_logs(build_root)
    control_log, control_compile = _compile_command_from_build_logs(
        build_root, "cesm_comp_mod.F90"
    )
    component_log, component_compile = _compile_command_from_build_logs(
        build_root, "component_mod.F90"
    )
    atm_log, atm_compile = _component_compile_command(
        build_root, "atm_comp_mct.F90"
    )
    cam_control_log, cam_control_compile = _component_compile_command(
        build_root, "cam_comp.F90"
    )
    physics_control_log, physics_control_compile = _component_compile_command(
        build_root, "physpkg.F90"
    )
    mct_log, mct_compile = _component_compile_command(
        build_root, "m_AttrVect.F90"
    )

    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    control_source = args.control_source.resolve()
    component_source = (
        args.component_source.resolve()
        if args.component_source is not None
        else source_root / "cime/src/drivers/mct/main/component_mod.F90"
    )
    cam_control_source = (
        args.cam_control_source.resolve()
        if args.cam_control_source is not None
        else source_root / "components/cam/src/control/cam_comp.F90"
    )
    physics_control_source = (
        args.physics_control_source.resolve()
        if args.physics_control_source is not None
        else source_root / "components/cam/src/physics/cam/physpkg.F90"
    )
    atm_source = (
        args.atm_source.resolve()
        if args.atm_source is not None
        else source_root / "components/cam/src/cpl/atm_comp_mct.F90"
    )
    adapter = args.adapter.resolve()
    main_source = args.main_source.resolve()
    allocator_interposer = args.allocator_interposer.resolve()
    mct_source = source_root / "cime/src/externals/mct/mct/m_AttrVect.F90"
    for source in (
        control_source,
        component_source,
        physics_control_source,
        cam_control_source,
        atm_source,
        adapter,
        main_source,
        allocator_interposer,
        mct_source,
    ):
        if not source.is_file():
            raise FileNotFoundError(source)

    control_object = build_dir / "cesm_comp_mod.o"
    component_object = build_dir / "component_mod.o"
    physics_control_object = build_dir / "physpkg.o"
    cam_control_object = build_dir / "cam_comp.o"
    atm_object = build_dir / "atm_comp_mct.o"
    adapter_object = build_dir / "cesm_full_driver_adapter.o"
    main_object = build_dir / "pycesm_embedded_main.o"
    allocator_object = build_dir / "intel_allocator_interposer.o"
    mct_object = build_dir / "m_AttrVect.o"
    compiled_mct = _compile_to(
        mct_compile,
        source_suffix="m_AttrVect.F90",
        source=mct_source,
        output=mct_object,
        cwd=build_dir,
        module_include=build_root / "mct/obj",
    )
    compiled_component = _compile_to(
        component_compile,
        source_suffix="component_mod.F90",
        source=component_source,
        output=component_object,
        cwd=build_dir,
        module_include=object_root,
    )
    compiled_physics_control = _compile_to(
        physics_control_compile,
        source_suffix="physpkg.F90",
        source=physics_control_source,
        output=physics_control_object,
        cwd=build_dir,
        module_include=build_root / "atm/obj",
    )
    compiled_cam_control = _compile_to(
        cam_control_compile,
        source_suffix="cam_comp.F90",
        source=cam_control_source,
        output=cam_control_object,
        cwd=build_dir,
        module_include=build_root / "atm/obj",
    )
    compiled_atm = _compile_to(
        atm_compile,
        source_suffix="atm_comp_mct.F90",
        source=atm_source,
        output=atm_object,
        cwd=build_dir,
        module_include=build_root / "atm/obj",
    )
    compiled_control = _compile_to(
        control_compile,
        source_suffix="cesm_comp_mod.F90",
        source=control_source,
        output=control_object,
        cwd=build_dir,
        module_include=object_root,
    )
    compiled_adapter = _compile_to(
        driver_compile,
        source_suffix="cesm_driver.F90",
        source=adapter,
        output=adapter_object,
        cwd=build_dir,
        module_include=object_root,
    )

    c_compile = [
        "cc",
        "-c",
        "-O2",
        "-Wall",
        "-Wextra",
        str(main_source),
        "-o",
        str(main_object),
    ]
    subprocess.run(c_compile, cwd=build_dir, check=True)
    allocator_compile = [
        "cc",
        "-c",
        "-O2",
        "-fno-omit-frame-pointer",
        "-Wall",
        "-Wextra",
        str(allocator_interposer),
        "-o",
        str(allocator_object),
    ]
    subprocess.run(allocator_compile, cwd=build_dir, check=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    executable_index = link_command.index("-o") + 1
    link_command[executable_index] = str(output)
    link_command = [
        value
        for value in link_command
        if value not in {"cesm_driver.o", "cesm_comp_mod.o", "component_mod.o"}
    ]
    first_driver_object = next(
        index
        for index, value in enumerate(link_command)
        if value.endswith("component_type_mod.o")
    )
    link_command[first_driver_object:first_driver_object] = [
        str(mct_object),
        str(component_object),
        str(physics_control_object),
        str(cam_control_object),
        str(atm_object),
        str(control_object),
        str(adapter_object),
        str(allocator_object),
        str(main_object),
    ]
    link_command.extend(
        (
            "-nofor-main",
            "-Wl,--export-dynamic",
            "-ldl",
            "-lm",
        )
    )
    subprocess.run(link_command, cwd=object_root, check=True)
    # Preserve the original non-PIC text/data layout while making the image
    # loadable through ctypes.CDLL.  This is the same header-only conversion
    # used by the validated coupled image; no numerical object is recompiled
    # as PIC or relinked with a different floating-point policy.
    subprocess.run(
        ("elfedit", "--output-type", "dyn", str(output)),
        cwd=object_root,
        check=True,
    )

    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    required = (
        "pycesm_full_initialize_v1",
        "pycesm_full_initialize_action_v1",
        "pycesm_full_action_v1",
        "pycesm_full_nested_action_v1",
        "pycesm_full_external_atm_iteration_v1",
        "pycesm_full_exchange_buffer_v1",
        "pycesm_full_cam_action_v1",
        "pycesm_full_physics_action_v1",
        "pycesm_full_step_begin_v1",
        "pycesm_full_step_begin_python_v1",
        "pycesm_full_step_end_v1",
        "pycesm_full_finalize_action_v1",
        "pycesm_full_finalize_v1",
        "pycesm_mct_set_allocator_v1",
        "pycesm_mct_clear_allocator_v1",
        "pycesm_fortran_heap_set_allocator_v1",
        "pycesm_fortran_heap_clear_allocator_v1",
        "for_allocate",
        "for_alloc_allocatable",
        "for_deallocate",
        "for_dealloc_allocatable",
    )
    missing = tuple(symbol for symbol in required if symbol not in symbols)
    if missing:
        raise RuntimeError(f"embedded executable is missing exported symbols: {missing}")

    dependencies = subprocess.run(
        ["ldd", str(output)], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    unresolved = tuple(line.strip() for line in dependencies if "not found" in line)
    if unresolved:
        raise RuntimeError(f"embedded executable has unresolved dependencies: {unresolved}")

    source_patches = tuple(
        path.resolve()
        for path in (
            args.source_patches
            or tuple(
                sorted(
                    (
                        REPOSITORY_ROOT / "native/pi_cam/control_patches"
                    ).glob("00*.patch")
                )
            )
        )
    )
    missing_patches = tuple(path for path in source_patches if not path.is_file())
    if missing_patches:
        raise FileNotFoundError(f"missing source patches: {missing_patches}")

    manifest = {
        "schema_version": 1,
        "case": str(case),
        "build_root": str(build_root),
        "link_log": str(link_log),
        "control_log": str(control_log),
        "component_log": str(component_log),
        "atm_log": str(atm_log),
        "cam_control_log": str(cam_control_log),
        "physics_control_log": str(physics_control_log),
        "mct_log": str(mct_log),
        "control_source": str(control_source),
        "control_source_sha256": _sha256(control_source),
        "component_source": str(component_source),
        "component_source_sha256": _sha256(component_source),
        "atm_source": str(atm_source),
        "atm_source_sha256": _sha256(atm_source),
        "cam_control_source": str(cam_control_source),
        "cam_control_source_sha256": _sha256(cam_control_source),
        "physics_control_source": str(physics_control_source),
        "physics_control_source_sha256": _sha256(physics_control_source),
        "adapter": str(adapter),
        "adapter_sha256": _sha256(adapter),
        "main_source": str(main_source),
        "main_source_sha256": _sha256(main_source),
        "mct_source": str(mct_source),
        "mct_source_sha256": _sha256(mct_source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "numerical_object_policy": "original_nonpic_executable_objects",
        "python_control_policy": "python_calls_bind_c_only",
        "reverse_callback_policy": "disabled_not_registered",
        "shadow_atmosphere": False,
        "symbols": list(required),
        "source_patches": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in source_patches
        ],
        "dependencies": [_stable_dependency(line) for line in dependencies],
        "control_compile_command": compiled_control,
        "component_compile_command": compiled_component,
        "atm_compile_command": compiled_atm,
        "cam_control_compile_command": compiled_cam_control,
        "physics_control_compile_command": compiled_physics_control,
        "mct_compile_command": compiled_mct,
        "adapter_compile_command": compiled_adapter,
        "c_compile_command": c_compile,
        "allocator_compile_command": allocator_compile,
        "link_command": link_command,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
