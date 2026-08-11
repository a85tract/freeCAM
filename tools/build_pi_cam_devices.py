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

from freecam.pi_cam.state_codegen import (  # noqa: E402
    generate_fortran_include,
    instrument_cam_comp,
    load_state_bridge,
)
from freecam.pi_cam.kernel_codegen import (  # noqa: E402
    generate_direct_kernel_module,
    load_direct_kernels,
)


REPO = Path(__file__).resolve().parents[1]
LEAF_PATCHES = (
    REPO
    / "native/pi_cam/control_patches/0026-python-before-coupler-leaf-control.patch",
    REPO / "native/pi_cam/control_patches/0027-python-leaf-dispatch.patch",
    REPO
    / "native/pi_cam/control_patches/0028-python-after-coupler-leaf-control.patch",
    REPO / "native/pi_cam/control_patches/0029-python-run4-leaf-control.patch",
    REPO / "native/pi_cam/control_patches/0031-order-independent-leaf-actions.patch",
)
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
PREPARE_INITIALIZE_SYMBOL = "pycam_pi_cam_prepare_initialize_v1"
STATE_CONTEXT_SYMBOL = "pycam_pi_cam_state_context_v1"
STATE_BIND_SYMBOL = "pycam_pi_cam_bind_state_v1"
LEAF_ACTION_SYMBOL = "pycam_pi_cam_leaf_action_v1"
LEAF_OPERATION_NAMES = (
    "leaf_modal_aero_prepare",
    "leaf_aero_model_wetdep",
    "leaf_carma_wetdep_tend",
    "leaf_convect_deep_tend_2",
    "leaf_diag_phys_writeout",
    "leaf_cloud_diagnostics_calc",
    "leaf_tropopause_output",
    "leaf_cam_export",
    "leaf_diag_export",
    "leaf_tracers_timestep_tend",
    "leaf_aoa_tracers_timestep_tend",
    "leaf_chem_timestep_tend",
    "leaf_aero_model_drydep",
    "leaf_carma_timestep_tend",
    "leaf_carma_accumulate_stats",
    "leaf_pbuf_deallocate",
    "leaf_pbuf_update_tim_idx",
    "leaf_diag_deallocate",
    "leaf_cam_run4_wrapup",
    "leaf_cam_run4_step_cost",
    "leaf_cam_run4_flush",
)
LEAF_OPERATION_IDS = (*range(450, 459), *range(460, 469), *range(470, 473))


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
    *,
    pic: bool = False,
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
    if pic:
        result.append("-fPIC")
    result.extend(("-o", str(output)))
    _run(result, cwd=cwd)
    return result


def _replace_archive(
    source: Path,
    destination: Path,
    objects: tuple[Path, ...],
    *,
    additions: tuple[str, ...] = (),
) -> None:
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
        if replacement.name not in members and replacement.name not in additions:
            raise RuntimeError(f"{source} lacks {replacement.name}")
        _run(["ar", "r", str(destination), str(replacement)], cwd=destination.parent)
    _run(["ranlib", str(destination)], cwd=destination.parent)


def _global_text_symbols(path: Path) -> tuple[str, ...]:
    output = subprocess.run(
        ["nm", "-g", "--defined-only", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result: list[str] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[-2].upper() in {"T", "W"}:
            result.append(fields[-1])
    return tuple(result)


def _global_defined_symbols(path: Path) -> dict[str, str]:
    output = subprocess.run(
        ["nm", "-g", "--defined-only", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result: dict[str, str] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            result[fields[-1]] = fields[-2].upper()
    return result


def _addon_module_object(
    generated: Path,
    production: Path,
    destination: Path,
) -> None:
    """Retain only new strong procedures beside a production CAM object.

    Existing procedures are renamed so the add-on cannot replace production
    machine code.  Existing module storage is weak, so all references from
    the new procedures resolve to the already-linked production storage.
    The add-on is linked into a separate lazy-loaded device, so it cannot move
    production code or change the default archive extraction order.
    """

    generated_symbols = _global_defined_symbols(generated)
    production_symbols = _global_defined_symbols(production)
    shared = set(generated_symbols) & set(production_symbols)
    command = ["objcopy"]
    for symbol in sorted(shared):
        if generated_symbols[symbol] in {"T", "W"}:
            command.extend(
                ("--redefine-sym", f"{symbol}=__pycam_leaf_unused_{symbol}")
            )
        else:
            command.extend(("--weaken-symbol", symbol))
    command.extend((str(generated), str(destination)))
    _run(command, cwd=destination.parent)


def _renamed_object(
    source: Path,
    destination: Path,
    symbols: tuple[str, ...],
    prefix: str,
) -> None:
    shutil.copy2(source, destination)
    command = ["objcopy"]
    for symbol in symbols:
        command.extend(("--redefine-sym", f"{symbol}={prefix}{symbol}"))
    command.append(str(destination))
    _run(command, cwd=destination.parent)


def _hybrid_module_object(
    generated: Path,
    numerical: Path,
    destination: Path,
    shell_symbols: tuple[str, ...],
) -> None:
    """Keep generated storage lifecycle code and original numerical code.

    Recompiling a large CAM module after changing only its storage descriptors
    can alter floating-point instruction selection in unrelated procedures.
    This merger retains the generated allocation/binding entry points while
    resolving every numerical module procedure to the production object.
    """

    generated_symbols = _global_text_symbols(generated)
    numerical_symbols = set(_global_text_symbols(numerical))
    absent = sorted(set(shell_symbols) - set(generated_symbols))
    if absent:
        raise RuntimeError(f"generated state shell lacks symbols {absent}")
    generated_numerics = tuple(
        symbol
        for symbol in generated_symbols
        if symbol in numerical_symbols and symbol not in shell_symbols
    )
    original_shells = tuple(
        symbol for symbol in shell_symbols if symbol in numerical_symbols
    )
    generated_copy = destination.with_name(destination.stem + "_storage.o")
    numerical_copy = destination.with_name(destination.stem + "_numerical.o")
    _renamed_object(
        generated,
        generated_copy,
        generated_numerics,
        "__pycam_generated_unused_",
    )
    _renamed_object(
        numerical,
        numerical_copy,
        original_shells,
        "__pycam_original_unused_",
    )
    _run(
        [
            "ld",
            "-r",
            "--allow-multiple-definition",
            str(generated_copy),
            str(numerical_copy),
            "-o",
            str(destination),
        ],
        cwd=destination.parent,
    )


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


def _direct_kernel_call_proof(path: Path, symbol: str, routine: str) -> str:
    """Prove that a generated wrapper calls, rather than copies, its routine."""

    output = subprocess.run(
        ["objdump", "-d", f"--disassemble={symbol}", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    # External procedures use ``routine_`` while an ifort module procedure is
    # emitted as ``module_mp_routine_``.  Both are the original routine; the
    # generated adapter must contain a machine-level call to one of those
    # symbols rather than a copied implementation.
    target = re.compile(
        rf"<(?:[^>]*_mp_)?{re.escape(routine.lower())}_(?:@plt)?>"
    )
    for line in output.splitlines():
        if "call" in line and target.search(line.lower()):
            return line.strip()
    raise RuntimeError(f"{symbol} does not call original routine {routine}_")


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


def _operations(state_bridge, direct_kernels=(), *, zero_copy_state: bool = False) -> dict[str, dict[str, object]]:
    operations: dict[str, dict[str, object]] = {
        "prepare_initialize": {
            "symbol": PREPARE_INITIALIZE_SYMBOL,
            "action_id": 0,
            "arguments": [
                {"field": "configured_stop_n", "dtype": "int64", "rank": 0, "intent": "in"},
                {"field": "case_name_utf8", "dtype": "uint8", "rank": 1, "intent": "in"},
                {"field": "orbital_year", "dtype": "int32", "rank": 0, "intent": "in"},
            ],
        },
        "initialize": {
            "symbol": ABI_SYMBOLS[0],
            "action_id": 0,
            "arguments": [],
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
                {"field": "cam_in.x2a_rattr", "dtype": "float64", "rank": 2, "intent": "in"},
                {"field": "cam_out.a2x_rattr", "dtype": "float64", "rank": 2, "intent": "out"},
                {"field": "model_step", "dtype": "int64", "rank": 0, "intent": "in"},
                {"field": "current_date", "dtype": "int32", "rank": 0, "intent": "in"},
                {"field": "current_seconds_of_day", "dtype": "int32", "rank": 0, "intent": "in"},
            ],
        },
    }
    if zero_copy_state:
        operations["bind_state"] = {
            "symbol": STATE_BIND_SYMBOL,
            "action_id": 0,
            "arguments": [
                *[
                    {
                        "field": f"__native_owner.{owner.name}",
                        "dtype": "uint8",
                        "rank": 1,
                        "intent": "inout",
                    }
                    for owner in state_bridge.owners
                ],
                *[
                    {
                        "field": field.name,
                        "dtype": "float64" if field.dtype == "real" else "int32",
                        "rank": field.python_rank,
                        "intent": "inout",
                    }
                    for field in state_bridge.fields
                    if field.active_by_default
                    and (field.allocatable or field.pointer)
                ],
            ],
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
    if zero_copy_state:
        for action_id, name in zip(LEAF_OPERATION_IDS, LEAF_OPERATION_NAMES):
            operations[name] = {
                "symbol": LEAF_ACTION_SYMBOL,
                "action_id": action_id,
                "arguments": [],
            }
    for kernel in direct_kernels:
        if kernel.operation_name in operations:
            raise RuntimeError(
                f"direct kernel operation {kernel.operation_name!r} is duplicated"
            )
        operations[kernel.operation_name] = kernel.operation_payload()
    return operations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--source-root", type=Path,
        default=REPO / "build/iCESM1.3.1_PI_cam_only",
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
        "--direct-kernels", type=Path,
        default=REPO / "native/pi_cam/direct_kernels_promoted.yaml",
    )
    parser.add_argument(
        "--zero-copy-state",
        action="store_true",
        help="require pointer-shell CAM objects and bind Python state in place",
    )
    parser.add_argument(
        "--numerical-build",
        type=Path,
        help=(
            "production BFB build root supplying unchanged physics_types.o "
            "and camsrfexch.o when --zero-copy-state is used"
        ),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    state_bridge = load_state_bridge(args.state_bridge.resolve(), source_root)
    direct_kernels = load_direct_kernels(args.direct_kernels.resolve())
    promoted_direct_kernels = tuple(
        kernel
        for kernel in direct_kernels
        if kernel.symbol.startswith("freecam_pi_cam_promoted_")
    )
    main_direct_kernels = tuple(
        kernel for kernel in direct_kernels if kernel not in promoted_direct_kernels
    )
    all_abi_symbols = (
        PREPARE_INITIALIZE_SYMBOL,
        STATE_CONTEXT_SYMBOL,
        *ABI_SYMBOLS,
        *((STATE_BIND_SYMBOL,) if args.zero_copy_state else ()),
        *(
            (
                f"pycam_pi_cam_layout_{owner.name}_v1"
                for owner in state_bridge.owners
            )
            if args.zero_copy_state
            else ()
        ),
        *(kernel.symbol for kernel in main_direct_kernels),
    )
    state_include = work / "pycam_pi_cam_state_bridge.inc"
    generated_state_include = generate_fortran_include(state_bridge)
    if args.zero_copy_state:
        # The generated SourceMods change persistent numerical components from
        # allocatable storage to pointers associated with Python arrays.
        generated_state_include = generated_state_include.replace(
            "allocated(", "associated("
        )
    state_include.write_text(generated_state_include)
    generated_cam_comp = work / "cam_comp.F90"
    original_cam_comp = source_root / "components/cam/src/control/cam_comp.F90"
    generated_cam_comp.write_text(
        instrument_cam_comp(
            original_cam_comp.read_text(),
            state_include.name,
            split_initialization=args.zero_copy_state,
        )
    )
    generated_direct_kernels = work / "pi_cam_direct_kernels.F90"
    generated_direct_kernels.write_text(
        generate_direct_kernel_module(main_direct_kernels)
    )
    generated_promoted_kernels = work / "pi_cam_promoted_kernels.F90"
    generated_promoted_kernels.write_text(
        generate_direct_kernel_module(
            promoted_direct_kernels,
            module_name="freecam_pi_cam_promoted_kernels",
        )
    )

    generated_physpkg = case / "SourceMods/src.cam/physpkg.F90"
    sources = {
        "physpkg.F90": (
            generated_physpkg
            if args.zero_copy_state and generated_physpkg.is_file()
            else source_root / "components/cam/src/physics/cam/physpkg.F90"
        ),
        "cam_comp.F90": generated_cam_comp,
        "atm_comp_mct.F90": source_root / "components/cam/src/cpl/atm_comp_mct.F90",
    }
    compile_logs: dict[str, str] = {}
    compile_commands: dict[str, list[str]] = {}
    objects: dict[str, Path] = {}
    leaf_addon_objects: tuple[Path, ...] = ()
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

    if args.zero_copy_state:
        if args.numerical_build is None:
            raise RuntimeError("--zero-copy-state requires --numerical-build")
        numerical_build = args.numerical_build.resolve()
        generated_objects = build / "atm/obj"
        numerical_objects = numerical_build / "atm/obj"
        generated_sources = case / "SourceMods/src.cam"
        freshly_generated: dict[str, Path] = {}
        for source_name in ("physics_types.F90", "camsrfexch.F90"):
            source = generated_sources / source_name
            if not source.is_file():
                raise RuntimeError(f"Python state SourceMod is absent: {source}")
            log, command = _compile_command(build, source_name)
            destination = work / f"{Path(source_name).stem}_generated.o"
            compile_commands[f"generated/{source_name}"] = _compile_to(
                command,
                source_name,
                source,
                destination,
                work,
                build / "atm/obj",
            )
            compile_logs[f"generated/{source_name}"] = str(log)
            freshly_generated[source_name] = destination
        physics_shells = (
            "physics_types_mp_physics_type_alloc_",
            "physics_types_mp_physics_state_alloc_",
            "physics_types_mp_physics_state_dealloc_",
            "physics_types_mp_physics_tend_alloc_",
            "physics_types_mp_physics_tend_dealloc_",
            "physics_types_mp_pycam_bind_phys_state_",
            "physics_types_mp_pycam_bind_phys_tend_",
        )
        surface_shells = (
            "camsrfexch_mp_hub2atm_alloc_",
            "camsrfexch_mp_atm2hub_alloc_",
            "camsrfexch_mp_hub2atm_deallocate_",
            "camsrfexch_mp_atm2hub_deallocate_",
            "camsrfexch_mp_pycam_bind_cam_in_",
            "camsrfexch_mp_pycam_bind_cam_out_",
        )
        physpkg_shells = (
            "physpkg_mp_phys_run1_prepare_",
            "physpkg_mp_phys_run1_schemes_",
            "physpkg_mp_phys_run1_scheme_action_",
            "physpkg_mp_phys_run2_prepare_",
            "physpkg_mp_phys_run2_schemes_",
            "physpkg_mp_phys_run2_scheme_action_",
            "physpkg_mp_phys_run2_finish_",
            "physpkg_mp_phys_final_",
        )
        hybrid_physics = work / "physics_types.o"
        hybrid_surface = work / "camsrfexch.o"
        # ``sources['physpkg.F90']`` is regenerated from the current prepared
        # source immediately above.  Do not silently replace that object with
        # the stale object left in a previously built case: doing so discards
        # newly admitted control shells while making the manifest look fresh.
        generated_physpkg_object = work / "physpkg_generated.o"
        shutil.copy2(objects["physpkg.F90"], generated_physpkg_object)
        # Keep the production archive-member name: ``_replace_archive`` uses
        # it to replace exactly ``physpkg.o`` in libatm.a.
        hybrid_physpkg = work / "physpkg.o"
        _hybrid_module_object(
            freshly_generated["physics_types.F90"],
            numerical_objects / "physics_types.o",
            hybrid_physics,
            physics_shells,
        )
        _hybrid_module_object(
            freshly_generated["camsrfexch.F90"],
            numerical_objects / "camsrfexch.o",
            hybrid_surface,
            surface_shells,
        )
        _hybrid_module_object(
            generated_physpkg_object,
            numerical_objects / "physpkg.o",
            hybrid_physpkg,
            physpkg_shells,
        )
        objects["physics_types.F90"] = hybrid_physics
        objects["camsrfexch.F90"] = hybrid_surface
        objects["physpkg.F90"] = hybrid_physpkg

        # Compile the deeper cam_run1 controls as an after-library add-on.
        # They are intentionally absent from the default source/archive so
        # enabling the API cannot move any BFB production code or storage.
        leaf_root = work / "leaf_sources"
        if leaf_root.exists():
            shutil.rmtree(leaf_root)
        leaf_physpkg = leaf_root / "src/physics/cam/physpkg.F90"
        leaf_cam_comp = leaf_root / "src/control/cam_comp.F90"
        leaf_physpkg.parent.mkdir(parents=True, exist_ok=True)
        leaf_cam_comp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources["physpkg.F90"], leaf_physpkg)
        shutil.copy2(original_cam_comp, leaf_cam_comp)
        for patch in LEAF_PATCHES:
            _run(
                ["git", "apply", "--unidiff-zero", "--verbose", str(patch)],
                cwd=leaf_root,
            )
        leaf_cam_comp.write_text(
            instrument_cam_comp(
                leaf_cam_comp.read_text(),
                state_include.name,
                split_initialization=True,
            )
        )
        leaf_physpkg_full = work / "physpkg_leaf_full.o"
        leaf_cam_comp_full = work / "cam_comp_leaf_full.o"
        for source_name, source, destination in (
            ("physpkg.F90", leaf_physpkg, leaf_physpkg_full),
            ("cam_comp.F90", leaf_cam_comp, leaf_cam_comp_full),
        ):
            log, command = _compile_command(build, source_name)
            key = f"leaf/{source_name}"
            compile_commands[key] = _compile_to(
                command,
                source_name,
                source,
                destination,
                work,
                build / "atm/obj",
                pic=True,
            )
            compile_logs[key] = str(log)
        leaf_physpkg_addon = work / "physpkg_leaf_addon.o"
        leaf_cam_comp_addon = work / "cam_comp_leaf_addon.o"
        _addon_module_object(
            leaf_physpkg_full,
            objects["physpkg.F90"],
            leaf_physpkg_addon,
        )
        _addon_module_object(
            leaf_cam_comp_full,
            objects["cam_comp.F90"],
            leaf_cam_comp_addon,
        )
        leaf_addon_objects = (leaf_cam_comp_addon, leaf_physpkg_addon)

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
    leaf_adapter_object: Path | None = None
    leaf_adapter_compile: list[str] | None = None
    public_adapter_compile: list[str] | None = None
    if leaf_addon_objects:
        public_adapter_source = work / "pi_cam_adapter_public.F90"
        public_marker = "  public :: pycam_pi_cam_state_transfer_v1\n"
        public_source = args.adapter.resolve().read_text()
        if public_source.count(public_marker) != 1:
            raise RuntimeError("cannot expose PI-CAM adapter leaf context")
        public_adapter_source.write_text(
            public_source.replace(
                public_marker,
                public_marker
                + "  public :: cam_in, cam_out, configured_stop_n\n",
            )
        )
        public_adapter_compile = _compile_to(
            adapter_command,
            "cam_comp.F90",
            public_adapter_source,
            work / "pi_cam_adapter_public.o",
            work,
            build / "atm/obj",
        )
        leaf_adapter_object = work / "pi_cam_leaf_adapter.o"
        leaf_adapter_compile = _compile_to(
            adapter_command,
            "cam_comp.F90",
            REPO / "native/pi_cam/pi_cam_leaf_adapter.F90",
            leaf_adapter_object,
            work,
            build / "atm/obj",
            pic=True,
        )
    direct_kernel_object = work / "pi_cam_direct_kernels.o"
    direct_kernel_compile = _compile_to(
        adapter_command,
        "cam_comp.F90",
        generated_direct_kernels,
        direct_kernel_object,
        work,
        build / "atm/obj",
    )
    promoted_kernel_object: Path | None = None
    promoted_kernel_compile: list[str] | None = None
    if promoted_direct_kernels:
        promoted_kernel_object = work / "pi_cam_promoted_kernels.o"
        promoted_kernel_compile = _compile_to(
            adapter_command,
            "cam_comp.F90",
            generated_promoted_kernels,
            promoted_kernel_object,
            work,
            build / "atm/obj",
            pic=True,
        )
    floating_environment_object = work / "floating_environment.o"
    floating_environment_compile = [
        "cc", "-c", "-O2", str(args.floating_environment.resolve()),
        "-o", str(floating_environment_object),
    ]
    _run(floating_environment_compile, cwd=work)

    atm_archive = output.parent / "libatm_nonpic_python_control.a"
    base_atm_archive = build / "lib/libatm.a"
    replacement_objects: tuple[Path, ...] = (
        objects["physpkg.F90"],
        objects["cam_comp.F90"],
        objects["atm_comp_mct.F90"],
    )
    if args.zero_copy_state:
        # Every production numerical object must come from the oracle build.
        # Merely recompiling an unchanged consumer against a pointer-shell
        # .mod can change Intel alias/vectorization decisions and produce ULP
        # differences.  Replace only control/ABI/storage objects; all physics
        # and dynamics machine code remains byte-for-byte production code.
        base_atm_archive = numerical_build / "lib/libatm.a"
        replacement_objects = (
            objects["physpkg.F90"],
            objects["cam_comp.F90"],
            objects["atm_comp_mct.F90"],
            objects["physics_types.F90"],
            objects["camsrfexch.F90"],
            generated_objects / "pycam_python_state_registry.o",
        )
    _replace_archive(
        base_atm_archive,
        atm_archive,
        replacement_objects,
        additions=("pycam_python_state_registry.o",) if args.zero_copy_state else (),
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
        *(f"-Wl,-u,{symbol}" for symbol in all_abi_symbols),
        *driver_objects,
        *libraries,
        # Keep the original executable object/library order intact.  The ABI
        # objects come last and resolve against symbols already selected by
        # that link, without changing the source archive boundary.
        str(adapter_object),
        str(direct_kernel_object),
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
    missing = tuple(symbol for symbol in all_abi_symbols if symbol not in symbols)
    if missing:
        raise RuntimeError(f"CAM image lacks ABI symbols: {missing}")
    zero_calls = _zero_calls(output)
    if zero_calls:
        raise RuntimeError(f"CAM image contains unresolved direct calls: {zero_calls[:8]}")
    start, end = _load_range(output)
    if start != IMAGE_BASE or end > IMAGE_BASE + IMAGE_WINDOW_BYTES:
        raise RuntimeError(f"CAM image load range 0x{start:x}-0x{end:x} is invalid")
    leaf_library: Path | None = None
    leaf_link: list[str] | None = None
    if leaf_adapter_object is not None:
        leaf_library = output.with_name(output.stem + "_leaf.so")
        leaf_link = [
            "ftn",
            "-shared",
            "-Wl,-z,notext,--allow-multiple-definition,--allow-shlib-undefined",
            str(leaf_adapter_object),
            *(str(path) for path in leaf_addon_objects),
            "-o",
            str(leaf_library),
        ]
        _run(leaf_link, cwd=output.parent)
        leaf_symbols = subprocess.run(
            ["nm", "-D", "--defined-only", str(leaf_library)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if LEAF_ACTION_SYMBOL not in leaf_symbols:
            raise RuntimeError("PI-CAM leaf add-on lacks its action ABI")
    promoted_kernel_library: Path | None = None
    promoted_kernel_link: list[str] | None = None
    if promoted_kernel_object is not None:
        promoted_kernel_library = output.with_name(
            output.stem + "_promoted_kernels.so"
        )
        promoted_kernel_link = [
            "ftn",
            "-shared",
            "-Wl,-z,notext,--allow-shlib-undefined",
            str(promoted_kernel_object),
            "-o",
            str(promoted_kernel_library),
        ]
        _run(promoted_kernel_link, cwd=output.parent)
        promoted_symbols = subprocess.run(
            ["nm", "-D", "--defined-only", str(promoted_kernel_library)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        missing_promoted = tuple(
            kernel.symbol
            for kernel in promoted_direct_kernels
            if kernel.symbol not in promoted_symbols
        )
        if missing_promoted:
            raise RuntimeError(
                f"promoted-kernel add-on lacks ABI symbols: {missing_promoted}"
            )
    direct_call_proof = {
        kernel.name: _direct_kernel_call_proof(output, kernel.symbol, kernel.routine)
        for kernel in main_direct_kernels
    }
    if promoted_kernel_library is not None:
        direct_call_proof.update(
            {
                kernel.name: _direct_kernel_call_proof(
                    promoted_kernel_library, kernel.symbol, kernel.routine
                )
                for kernel in promoted_direct_kernels
            }
        )

    state_manifest = state_bridge.manifest()
    if args.zero_copy_state:
        state_manifest["ownership"] = "python-owned-zero-copy-derived-shell"
        state_manifest["symbols"]["context"] = STATE_CONTEXT_SYMBOL
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
        "numerical_archive": (
            {
                "path": str(base_atm_archive),
                "sha256": _sha256(base_atm_archive),
                "policy": "oracle-production-objects-with-generated-control-and-storage",
            }
            if args.zero_copy_state
            else None
        ),
        "source_root": str(source_root),
        "process_device_generation": str(
            (REPO / "validation/pi_cam_in_module_adapter_generation.json").resolve()
        ),
        "process_device_validation": str(
            (REPO / "validation/pi_cam_in_module_adapter_validation.json").resolve()
        ),
        "process_device_loading": str(
            (REPO / "validation/pi_cam_process_device_loading.json").resolve()
        ),
        "control_source": str(sources["physpkg.F90"].resolve()),
        "control_source_sha256": _sha256(sources["physpkg.F90"].resolve()),
        "hybrid_control_object_sha256": _sha256(objects["physpkg.F90"]),
        "leaf_addon_objects": [
            {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in leaf_addon_objects
        ],
        "leaf_device": (
            {
                "library": str(leaf_library),
                "library_sha256": _sha256(leaf_library),
                "operations": list(LEAF_OPERATION_NAMES),
                "load_policy": "lazy-after-initialize",
            }
            if leaf_library is not None
            else None
        ),
        "promoted_kernel_device": (
            {
                "library": str(promoted_kernel_library),
                "library_sha256": _sha256(promoted_kernel_library),
                "operations": [
                    kernel.operation_name for kernel in promoted_direct_kernels
                ],
                "load_policy": "lazy-after-initialize",
            }
            if promoted_kernel_library is not None
            else None
        ),
        "link_log": str(link_log),
        "compile_logs": compile_logs,
        "adapter": str(args.adapter.resolve()),
        "floating_environment": str(args.floating_environment.resolve()),
        "floating_environment_compile_command": floating_environment_compile,
        "intel_math_library": str(imf_shared),
        "operations": _operations(
            state_bridge,
            direct_kernels,
            zero_copy_state=args.zero_copy_state,
        ),
        "direct_kernels": {
            "schema_version": 1,
            "description": str(args.direct_kernels.resolve()),
            "generated_source": str(generated_direct_kernels),
            "generated_promoted_source": str(generated_promoted_kernels),
            "kernels": [
                {
                    "name": kernel.name,
                    "routine": kernel.routine,
                    "symbol": kernel.symbol,
                    "original_call_proof": direct_call_proof[kernel.name],
                    "arguments": [
                        argument.operation_payload() for argument in kernel.arguments
                    ],
                }
                for kernel in direct_kernels
            ],
        },
        "state_bridge": state_manifest,
        "state_bridge_description": str(args.state_bridge.resolve()),
        "state_bridge_include": str(state_include),
        "compile_commands": compile_commands,
        "adapter_compile_command": adapter_compile,
        "public_adapter_compile_command": public_adapter_compile,
        "leaf_adapter_compile_command": leaf_adapter_compile,
        "leaf_link_command": leaf_link,
        "direct_kernel_compile_command": direct_kernel_compile,
        "promoted_kernel_compile_command": promoted_kernel_compile,
        "promoted_kernel_link_command": promoted_kernel_link,
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
