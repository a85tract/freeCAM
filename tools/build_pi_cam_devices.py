#!/usr/bin/env python3
"""Build the PI-CAM device without recompiling numerical CAM as PIC.

The production Intel CAM objects are non-PIC.  Recompiling all of CAM with
``-fPIC`` changes register allocation and already fails the PI-atm bitwise
gate.  This builder preserves those objects, replaces only the three Python
control surfaces with non-PIC builds, links a fixed-address executable image,
and changes its ELF type from ET_EXEC to ET_DYN.  Python can then use dlopen
without changing the numerical machine instructions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pycam_sima.pi_cam.state_codegen import (  # noqa: E402
    generate_fortran_include,
    instrument_cam_comp,
    load_state_bridge,
)


REPO = Path(__file__).resolve().parents[1]
IMAGE_BASE = 0x30000000
IMAGE_WINDOW_BYTES = 0x20000000
ABI_SYMBOLS = (
    "pycam_pi_cam_initialize_v1",
    "pycam_pi_cam_action_v1",
    "pycam_pi_cam_finalize_v1",
    "pycam_pi_cam_state_count_v1",
    "pycam_pi_cam_state_metadata_v1",
    "pycam_pi_cam_state_transfer_v1",
)


def _run(command: list[str] | tuple[str, ...], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _xml(case: Path, name: str) -> str:
    return subprocess.run(
        [str(case / "xmlquery"), name, "--value"],
        cwd=case,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            return stream.read()
    return path.read_text(errors="replace")


def _logs(build: Path, stem: str) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (*build.glob(f"{stem}.*"), *build.glob(f"{stem}.*.gz")),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    )


def _compile_command(build: Path, source: str) -> tuple[Path, list[str]]:
    for path in _logs(build, "atm.bldlog"):
        for line in reversed(_text(path).splitlines()):
            if line.startswith("ftn ") and source in line and " -c " in line:
                return path, shlex.split(line)
    raise RuntimeError(f"cannot recover compile command for {source}")


def _link_command(build: Path) -> tuple[Path, list[str]]:
    for path in _logs(build, "cesm.bldlog"):
        for line in reversed(_text(path).splitlines()):
            if line.startswith("ftn ") and "cesm.exe" in line and " -o " in line:
                return path, shlex.split(line)
    raise RuntimeError("cannot recover the CESM link command")


def _without_output(command: list[str]) -> list[str]:
    result: list[str] = []
    skip = False
    for value in command:
        if skip:
            skip = False
            continue
        if value == "-o":
            skip = True
            continue
        if value.startswith("-o") and len(value) > 2:
            continue
        result.append(value)
    return result


def _compile_to(
    command: list[str],
    source_name: str,
    source: Path,
    output: Path,
    cwd: Path,
    module_include: Path,
) -> list[str]:
    result = _without_output(command)
    source_index = next(
        index for index, value in enumerate(result) if value.endswith(source_name)
    )
    result[source_index] = str(source)
    # ``-I.`` now refers to the private replacement-module directory.  The
    # case object directory remains the fallback for the hundreds of
    # unchanged CAM modules.
    local_include = result.index("-I.")
    result.insert(local_include + 1, f"-I{module_include}")
    # Preserve the production non-PIC numerical code-generation boundary.
    result = [value for value in result if value != "-fPIC"]
    result.extend(("-o", str(output)))
    _run(result, cwd=cwd)
    return result


def _replace_archive(source: Path, destination: Path, objects: tuple[Path, ...]) -> None:
    shutil.copy2(source, destination)
    members = set(
        subprocess.run(
            ["ar", "t", str(destination)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    for replacement in objects:
        if replacement.name not in members:
            raise RuntimeError(f"{source} lacks {replacement.name}")
        _run(["ar", "r", str(destination), str(replacement)], cwd=destination.parent)
    _run(["ranlib", str(destination)], cwd=destination.parent)


def _replace_library(command: list[str], archive: Path) -> list[str]:
    return [str(archive) if value == "-latm" else value for value in command]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_range(path: Path) -> tuple[int, int]:
    output = subprocess.run(
        ["readelf", "-lW", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    ranges: list[tuple[int, int]] = []
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "LOAD":
            start = int(fields[2], 16)
            ranges.append((start, start + int(fields[5], 16)))
    if not ranges:
        raise RuntimeError("CAM image has no loadable segment")
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def _zero_calls(path: Path) -> tuple[str, ...]:
    output = subprocess.run(
        ["objdump", "-d", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return tuple(
        line.strip()
        for line in output.splitlines()
        if re.search(r"\bcall\s+0\s+<", line)
    )


def _runtime_library(executable: Path, name: str) -> Path:
    """Return the concrete library selected by the production link."""

    output = subprocess.run(
        ["ldd", str(executable)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    prefix = f"{name}.so"
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0].startswith(prefix) and len(fields) >= 3:
            return Path(fields[2]).resolve()
    raise RuntimeError(f"cannot resolve {name} from {executable}")


def _operations() -> dict[str, dict[str, object]]:
    operations: dict[str, dict[str, object]] = {
        "initialize": {
            "symbol": ABI_SYMBOLS[0],
            "action_id": 0,
            "arguments": [
                {"field": "configured_stop_n", "dtype": "int64", "rank": 0, "intent": "in"},
                {"field": "case_name_utf8", "dtype": "uint8", "rank": 1, "intent": "in"},
                {"field": "orbital_year", "dtype": "int32", "rank": 0, "intent": "in"},
            ],
        },
        "finalize": {"symbol": ABI_SYMBOLS[2], "action_id": 0, "arguments": []},
        "initial_priming": {
            "symbol": ABI_SYMBOLS[1],
            "action_id": 200,
            "arguments": [
                {"field": "cam_in.x2a_rattr", "dtype": "float64", "rank": 2, "intent": "in"},
                {"field": "cam_out.a2x_rattr", "dtype": "float64", "rank": 2, "intent": "out"},
            ],
        },
        "source_step": {
            "symbol": ABI_SYMBOLS[1],
            "action_id": 500,
            "arguments": [
                {"field": "cam_in.x2a_rattr", "dtype": "float64", "rank": 2, "intent": "in"},
                {"field": "cam_out.a2x_rattr", "dtype": "float64", "rank": 2, "intent": "out"},
            ],
        },
        "source_step_held_import": {
            "symbol": ABI_SYMBOLS[1],
            "action_id": 501,
            "arguments": [
                {"field": "cam_in.x2a_rattr", "dtype": "float64", "rank": 2, "intent": "in"},
                {"field": "cam_out.a2x_rattr", "dtype": "float64", "rank": 2, "intent": "out"},
            ],
        },
        "boundary_import": {
            "symbol": ABI_SYMBOLS[1],
            "action_id": 202,
            "arguments": [
                {"field": "cam_in.x2a_rattr", "dtype": "float64", "rank": 2, "intent": "in"},
            ],
        },
        "boundary_export": {
            "symbol": ABI_SYMBOLS[1],
            "action_id": 432,
            "arguments": [
                {"field": "cam_out.a2x_rattr", "dtype": "float64", "rank": 2, "intent": "out"},
            ],
        },
    }
    names = (
        "prepare", "chem_emissions", "tracers_chemistry",
        "vertical_diffusion_tend", "rayleigh_friction_tend",
        "aero_model_drydep", "charge_fix", "gw_tend", "qbo_relax",
        "iondrag_calc", "physics_dme_adjust", "finish", "stepon_run2",
        "stepon_run3", "wshist", "restart", "wrapup",
        "advance_timestep", "stepon_run1", "prepare_cam_run1", "bc_init",
        "check_energy_fix", "dadadj", "convect_deep_tend",
        "convect_shallow_tend", "sslt_rebin_adv", "macro_microphysics",
        "aero_model_wetdep", "physics_diagnostics", "radiation_tend",
        "cam_export",
    )
    for action_id, name in zip(range(401, 432), names):
        operations[name] = {
            "symbol": ABI_SYMBOLS[1], "action_id": action_id, "arguments": []
        }
    return operations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path,
        default=Path("/glade/work/ruitong/iCESM1.3.1_PI_cam_only"),
    )
    parser.add_argument(
        "--adapter", type=Path,
        default=REPO / "native/pi_cam/pi_cam_adapter.F90",
    )
    parser.add_argument(
        "--floating-environment", type=Path,
        default=REPO / "native/pi_cam/floating_environment.c",
    )
    parser.add_argument(
        "--state-bridge", type=Path,
        default=REPO / "native/pi_cam/state_bridge.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "build/pi_cam/libpycam_pi_cam.so",
    )
    parser.add_argument(
        "--capture-executable", type=Path,
        default=REPO / "build/pi_cam/pi_cam_capture.exe",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=REPO / "build/pi_cam/native_cam_manifest.json",
    )
    args = parser.parse_args()

    case = args.case.resolve()
    source_root = args.source_root.resolve()
    build = Path(_xml(case, "EXEROOT")).resolve()
    output = args.output.resolve()
    work = output.parent / "nonpic_objects"
    for directory in (output.parent, work):
        directory.mkdir(parents=True, exist_ok=True)

    state_bridge = load_state_bridge(args.state_bridge.resolve(), source_root)
    state_include = work / "pycam_pi_cam_state_bridge.inc"
    state_include.write_text(generate_fortran_include(state_bridge))
    generated_cam_comp = work / "cam_comp.F90"
    original_cam_comp = source_root / "components/cam/src/control/cam_comp.F90"
    generated_cam_comp.write_text(
        instrument_cam_comp(original_cam_comp.read_text(), state_include.name)
    )

    sources = {
        "physpkg.F90": source_root / "components/cam/src/physics/cam/physpkg.F90",
        "cam_comp.F90": generated_cam_comp,
        "atm_comp_mct.F90": source_root / "components/cam/src/cpl/atm_comp_mct.F90",
    }
    compile_logs: dict[str, str] = {}
    compile_commands: dict[str, list[str]] = {}
    objects: dict[str, Path] = {}
    for source_name in (
        "physpkg.F90", "cam_comp.F90", "atm_comp_mct.F90",
    ):
        log, command = _compile_command(build, source_name)
        destination = work / f"{Path(source_name).stem}.o"
        compile_commands[source_name] = _compile_to(
            command,
            source_name,
            sources[source_name],
            destination,
            work,
            build / "atm/obj",
        )
        compile_logs[source_name] = str(log)
        objects[source_name] = destination

    _, adapter_command = _compile_command(build, "cam_comp.F90")
    adapter_object = work / "pi_cam_adapter.o"
    adapter_compile = _compile_to(
        adapter_command,
        "cam_comp.F90",
        args.adapter.resolve(),
        adapter_object,
        work,
        build / "atm/obj",
    )
    floating_environment_object = work / "floating_environment.o"
    floating_environment_compile = [
        "cc", "-c", "-O2", str(args.floating_environment.resolve()),
        "-o", str(floating_environment_object),
    ]
    _run(floating_environment_compile, cwd=work)

    atm_archive = output.parent / "libatm_nonpic_python_control.a"
    _replace_archive(
        build / "lib/libatm.a",
        atm_archive,
        (
            objects["physpkg.F90"], objects["cam_comp.F90"],
            objects["atm_comp_mct.F90"],
        ),
    )

    link_log, original_link = _link_command(build)
    patched_link = _replace_library(original_link, atm_archive)
    capture_executable = args.capture_executable.resolve()
    capture_link = list(patched_link)
    capture_link.insert(capture_link.index("-o"), str(floating_environment_object))
    capture_link[capture_link.index("-o") + 1] = str(capture_executable)
    _run(capture_link, cwd=build / "cpl/obj")
    # Record the exact Intel math runtime used by the standalone executable.
    # The persistent launcher preloads this exact dependency before Python
    # loads the fixed CAM image, preserving the source executable's math ABI.
    imf_shared = _runtime_library(capture_executable, "libimf")

    first_library = next(
        index for index, value in enumerate(patched_link) if value.startswith("-L")
    )
    # Preserve the BFB executable's complete pre-library link context.  These
    # CESM control objects are not executed by Python, but including them makes
    # static archive extraction and duplicate-symbol selection identical to
    # the validated capture executable.  Omitting them produced a CAM image
    # whose cam_init export matched but whose first cam_run1 did not.
    driver_objects = [
        str((build / "cpl/obj" / value).resolve())
        for value in patched_link[:first_library]
        if value.endswith(".o")
    ]
    libraries = patched_link[first_library:]
    fixed_executable = output.with_suffix(".exec")
    fixed_link = [
        "ftn", "-nostartfiles", "-nofor-main",
        (
            f"-Wl,-e,0,-Ttext-segment=0x{IMAGE_BASE:x},--export-dynamic,"
            f"--allow-multiple-definition,-Bsymbolic-functions,-soname,{output.name}"
        ),
        *(f"-Wl,-u,{symbol}" for symbol in ABI_SYMBOLS),
        *driver_objects,
        *libraries,
        # Keep the original executable object/library order intact.  The ABI
        # objects come last and resolve against symbols already selected by
        # that link, without changing the source archive boundary.
        str(adapter_object),
        str(floating_environment_object),
        "-Wl,--unresolved-symbols=ignore-all",
        "-o", str(fixed_executable),
    ]
    _run(fixed_link, cwd=output.parent)
    shutil.copy2(fixed_executable, output)
    _run(["elfedit", "--output-type", "dyn", str(output)], cwd=output.parent)

    symbols = subprocess.run(
        ["nm", "-D", "--defined-only", str(output)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    missing = tuple(symbol for symbol in ABI_SYMBOLS if symbol not in symbols)
    if missing:
        raise RuntimeError(f"CAM image lacks ABI symbols: {missing}")
    zero_calls = _zero_calls(output)
    if zero_calls:
        raise RuntimeError(f"CAM image contains unresolved direct calls: {zero_calls[:8]}")
    start, end = _load_range(output)
    if start != IMAGE_BASE or end > IMAGE_BASE + IMAGE_WINDOW_BYTES:
        raise RuntimeError(f"CAM image load range 0x{start:x}-0x{end:x} is invalid")

    manifest = {
        "schema_version": 1,
        "execution_model": "fixed-address-nonpic-cam-image",
        "library": str(output),
        "library_sha256": _sha256(output),
        "library_bytes": output.stat().st_size,
        "load_start": start,
        "load_end": end,
        "case": str(case),
        "build_root": str(build),
        "source_root": str(source_root),
        "link_log": str(link_log),
        "compile_logs": compile_logs,
        "adapter": str(args.adapter.resolve()),
        "floating_environment": str(args.floating_environment.resolve()),
        "floating_environment_compile_command": floating_environment_compile,
        "intel_math_library": str(imf_shared),
        "operations": _operations(),
        "state_bridge": state_bridge.manifest(),
        "state_bridge_description": str(args.state_bridge.resolve()),
        "state_bridge_include": str(state_include),
        "compile_commands": compile_commands,
        "adapter_compile_command": adapter_compile,
        "capture_executable": str(capture_executable),
        "capture_executable_sha256": _sha256(capture_executable),
        "capture_link_command": capture_link,
        "driver_link_objects": driver_objects,
        "fixed_link_command": fixed_link,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
