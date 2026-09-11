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
import os
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
from freecam.pi_cam.hooks import HOOKS, load_hooks  # noqa: E402
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
    REPO / "native/pi_cam/control_patches/0040-macro-tend-leaf-dispatch.patch",
    REPO / "native/pi_cam/control_patches/0042-rad-tend-leaf-dispatch.patch",
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
# The physics buffer is reached by handle, not by StatePool field.
PBUF_FIELD_SYMBOL = "pycam_pbuf_field_v1"
# Fortran modules this repository adds to the source tree beside CAM's own.
# They are compiled into the fixed image as additions -- no numerical object
# is replaced -- and are reached only from Python or from the control
# objects the builder already replaces.
SUPPORT_MODULES = ("pycam_macro_kernels.F90", "pycam_macro_handles.F90",
                   "pycam_rad_kernels.F90", "pycam_rad_handles.F90",
                   "pycam_micro_kernels.F90", "pycam_mm_kernels.F90",
                   "pycam_aero_kernels.F90",
                   "pycam_micro_handles.F90", "pycam_aero_handles.F90",
                   "pycam_mm_handles.F90",
                   # the hooks: kernels reached inside compiled routines by symbol
                   # redirection; the runners of hooked kernels use this module
                   "pycam_hooks.F90",
                   # the pausable runners: hosts, then each process's units (a
                   # driver before the glue that binds it), then its runner
                   "pycam_stage_hosts.F90",
                   "pycam_dadadj_glue.F90", "pycam_dadadj_runner.F90",
                   "pycam_shcu_driver.F90", "pycam_shcu_glue.F90", "pycam_shcu_runner.F90",
                   "pycam_radt_driver.F90", "pycam_radt_glue.F90", "pycam_radt_runner.F90",
                   # the deepest unit first: each unit's binder is used by the unit that calls it
                   "pycam_zmdeep_zm.F90", "pycam_zmdeep_deep.F90", "pycam_zmdeep_glue.F90", "pycam_zmdeep_runner.F90",
                   "pycam_zmtran_zm2.F90", "pycam_zmtran_deep2.F90", "pycam_zmtran_glue.F90", "pycam_zmtran_runner.F90",
                   "pycam_vdiff_driver.F90", "pycam_vdiff_glue.F90", "pycam_vdiff_runner.F90",
                   "pycam_gwd_driver.F90", "pycam_gwd_glue.F90", "pycam_gwd_runner.F90",
                   "pycam_awet_driver.F90", "pycam_awet_glue.F90", "pycam_awet_runner.F90",
                   "pycam_adry_driver.F90", "pycam_adry_glue.F90", "pycam_adry_runner.F90",
                   "pycam_chem_driver.F90", "pycam_chem_glue.F90", "pycam_chem_runner.F90")
#: Modules whose control patch changes accessibility alone (a `public` statement
#: naming module state the pausable runners' hoisted drivers read).  They are
#: compiled from the prepared source for their .mod files only, into the working
#: directory the support modules read first; the object is discarded and the
#: oracle's stays in the archive, so no numerical machine code is recompiled.
INTERFACE_MODULES = ("zm_conv_intr.F90", "vertical_diffusion.F90", "gw_drag.F90",
                     "../../chemistry/mozart/chemistry.F90", "../../chemistry/modal_aero/aero_model.F90")
MACRO_BIND_HOSTS_SYMBOL = "pycam_macro_bind_hosts_v1"
RAD_BIND_HOSTS_SYMBOL = "pycam_rad_bind_hosts_v1"
MICRO_BIND_HOSTS_SYMBOL = "pycam_micro_bind_hosts_v1"
MM_BIND_HOSTS_SYMBOL = "pycam_mm_bind_hosts_v1"
AERO_BIND_HOSTS_SYMBOL = "pycam_aero_bind_hosts_v1"
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
    "leaf_macro_tend_pre",
    "leaf_macro_tend_post",
    "leaf_rad_tend_pre",
    "leaf_rad_tend_post",
)
LEAF_OPERATION_IDS = (
    *range(450, 459), *range(460, 469), *range(470, 473), *range(480, 484)
)


from pi_cam_build_common import (  # noqa: E402
    run as _run,
    xml as _xml,
    text as _text,
    logs as _logs,
    compile_command as _compile_command,
    link_command as _link_command,
    without_output as _without_output,
    compile_to as _compile_to,
    replace_archive as _replace_archive,
    global_text_symbols as _global_text_symbols,
    global_defined_symbols as _global_defined_symbols,
    addon_module_object as _addon_module_object,
    renamed_object as _renamed_object,
    hybrid_module_object as _hybrid_module_object,
    replace_library as _replace_library,
    sha256 as _sha256,
    load_range as _load_range,
    zero_calls as _zero_calls,
    direct_kernel_call_proof as _direct_kernel_call_proof,
    runtime_library as _runtime_library,
)


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


def _text_sha256(path: Path, work: Path) -> str:
    """The bytes of an object's .text section: the machine code a redirection must leave alone."""

    dump = work / f"{path.name}.text.bin"
    _run(["objcopy", "-O", "binary", "--only-section=.text", str(path), str(dump)], cwd=work)
    return _sha256(dump)


def _refuse_duplicate_globals(objects: list[Path]) -> None:
    """Two support objects defining one global symbol is a build error.

    Linked through the archive, the second definition would be dropped without
    a word and every reference would reach the first: two runners whose fiber
    bodies shared the bare C name ``fiber_body`` ran each other's state machine
    (gate 7343594).
    """

    owners: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in objects:
        output = subprocess.run(["nm", "-g", "--defined-only", str(path)], check=True, capture_output=True, text=True).stdout
        for line in output.splitlines():
            parts = line.split()
            if len(parts) != 3 or parts[1] not in ("T", "D", "B", "R"):
                continue
            symbol = parts[2]
            if symbol in owners and owners[symbol] != path:
                duplicates.append(f"{symbol} ({owners[symbol].name}, {path.name})")
            owners.setdefault(symbol, path)
    if duplicates:
        raise RuntimeError("global symbols defined by more than one support object: " + "; ".join(sorted(duplicates)))


def _relocations_naming(path: Path, symbol: str) -> int:
    output = subprocess.run(["readelf", "-rW", str(path)], check=True, capture_output=True, text=True).stdout
    return sum(1 for line in output.splitlines() if line.split() and symbol in line.split())


def _load_tool(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse_trampoline_collisions(trampoline_object: Path, support_objects) -> None:
    """A trampoline symbol defined strongly anywhere else would win or lose the link silently.

    The image links with --allow-multiple-definition, where the first strong
    definition wins without a diagnostic; a counting trampoline must therefore
    never share a symbol with a support module (a replacement hook included).
    """

    def strong_globals(path: Path) -> set[str]:
        out = subprocess.run(["nm", "-g", "--defined-only", str(path)], check=True,
                             capture_output=True, text=True).stdout
        return {line.split()[-1] for line in out.splitlines() if line.split() and line.split()[-2] in ("T", "D", "B", "R")}

    trampolines = strong_globals(trampoline_object) - {"pycam_kcount_kernel_count"}
    for support in support_objects:
        clash = trampolines & strong_globals(Path(support))
        if clash:
            raise RuntimeError(f"counting trampolines collide with {Path(support).name}: {sorted(clash)}")


def _definition_address(target: Path, symbol: str) -> str:
    symbols = subprocess.run(["nm", str(target)], check=True, capture_output=True, text=True).stdout
    definition = next((line.split() for line in symbols.splitlines()
                       if line.split() and line.split()[-1] == symbol and line.split()[-2] == "T"), None)
    if definition is None:
        raise RuntimeError(f"{target.name} does not define {symbol}; weaken-definition needs the definition")
    return definition[0]


def _apply_redirections(archive: Path, out_dir: Path, plans: dict[str, list[dict]]) -> tuple[list[dict], list[Path]]:
    """Redirected copies of archive objects; one extraction and one objcopy per object.

    Every operation on one object is applied in a single objcopy invocation so a
    later redirection can never clobber an earlier one.  ``redefine`` renames a
    reference (the hooks' rename-references mode); ``weaken-alias`` weakens a
    definition and adds a second, global name at the same address -- the strong
    definition elsewhere (a replacement hook or a counting trampoline) then wins
    every reference, and forwards to the alias, never to itself.  The .text bytes
    are proved unchanged either way.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    objects: list[Path] = []
    for object_name, operations in sorted(plans.items()):
        target = out_dir / object_name
        if target.exists():
            target.unlink()
        _run(["ar", "x", str(archive), object_name], cwd=out_dir)
        before_text = _text_sha256(target, out_dir)
        flags: list[str] = []
        for op in operations:
            if op["kind"] == "redefine":
                if _relocations_naming(target, op["old"]) == 0:
                    raise RuntimeError(f"{object_name} has no relocation naming {op['old']}; nothing to redirect")
                flags.append(f"--redefine-sym={op['old']}={op['new']}")
            elif op["kind"] == "weaken-alias":
                address = _definition_address(target, op["symbol"])
                if op.get("alias"):
                    flags.append(f"--add-symbol={op['alias']}=.text:0x{address},global,function")
                flags.append(f"--weaken-symbol={op['symbol']}")
            elif op["kind"] == "alias-at":
                # a second global name at an existing definition's address; the
                # definition itself is left exactly as it is
                address = _definition_address(target, op["symbol"])
                flags.append(f"--add-symbol={op['alias']}=.text:0x{address},global,function")
            else:
                raise RuntimeError(f"unknown redirection kind {op['kind']!r}")
        _run(["objcopy", *flags, str(target)], cwd=out_dir)
        after_text = _text_sha256(target, out_dir)
        if after_text != before_text:
            raise RuntimeError(f"{object_name}: the redirection changed .text ({before_text[:12]} -> {after_text[:12]})")
        object_sha = _sha256(target)
        for op in operations:
            named = op["new"] if op["kind"] == "redefine" else (
                op["alias"] if op["kind"] == "alias-at" else op["symbol"])
            records.append({**op["record"], "mode": op["record"].get("mode", op["kind"]),
                            "object": object_name, "relocations": _relocations_naming(target, named),
                            "text_sha256": after_text, "object_sha256": object_sha})
        objects.append(target)
    return records, objects


def _hook_plans(table) -> dict[str, list[dict]]:
    """The hook table's redirections as per-object operation plans."""

    plans: dict[str, list[dict]] = {}
    for hook in table.hooks:
        for caller in hook.callers:
            record = {"kernel": hook.kernel, "routine": caller.routine, "mode": hook.redirect,
                      "callee_symbol": hook.callee_symbol, "hook_symbol": hook.symbol,
                      "original_symbol": hook.original_symbol}
            if hook.redirect == "rename-references":
                op = {"kind": "redefine", "old": hook.callee_symbol, "new": hook.symbol, "record": record}
            else:
                op = {"kind": "weaken-alias", "symbol": hook.callee_symbol,
                      "alias": hook.original_symbol, "record": record}
            plans.setdefault(caller.object, []).append(op)
    return plans


def _kcount_plans(rows: list[dict]) -> dict[str, list[dict]]:
    """The counting trampolines' weaken-and-alias operations, one per kernel."""

    plans: dict[str, list[dict]] = {}
    for row in rows:
        record = {"kernel": row["routine"], "qualified": row["qualified"], "mode": "count-weaken-alias",
                  "callee_symbol": row["symbol"], "hook_symbol": row["symbol"], "original_symbol": row["alias"],
                  "index": row["index"]}
        plans.setdefault(row["object"], []).append(
            {"kind": "weaken-alias", "symbol": row["symbol"], "alias": row["alias"], "record": record})
    return plans


def _torch_lib_dir() -> Path:
    """The libtorch directory of this interpreter's torch package, which FTorch links against."""

    try:
        import torch  # noqa: WPS433 - the build resolves the runtime it links
    except ImportError as exc:
        raise RuntimeError("--torch-lib was not given and torch is not importable here") from exc
    return Path(torch.__file__).resolve().parent / "lib"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument(
        "--ftorch-root", type=Path, default=REPO / "build/ftorch",
        help="an FTorch installation (include/ftorch with the .mod files, lib64 with libftorch.so): "
             "the hooks module runs bound TorchScript models through it",
    )
    parser.add_argument(
        "--torch-lib", type=Path, default=None,
        help="the libtorch directory FTorch was built against (default: the torch package of this interpreter)",
    )
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
    parser.add_argument(
        "--kcount-scope", default=os.environ.get("FREECAM_KCOUNT_SCOPE", ""),
        help="build a counting image: 'all', 'batch-a', or comma-separated process ids / routine names; "
             "empty (the default) builds the ordinary image with the replacement hooks",
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
    # The interface-only modules before anything that `use`s them: their .mod
    # files land in the working directory, which -I. covers ahead of the case
    # objects; the objects are not linked.
    for source_name in INTERFACE_MODULES:
        # a name is under components/cam/src/physics/cam; a relative entry reaches the chemistry tree
        source = (source_root / "components/cam/src/physics/cam" / source_name).resolve()
        if not source.is_file():
            raise RuntimeError(f"interface module is absent from the prepared source: {source}")
        log, command = _compile_command(build, source.name)
        destination = work / f"{source.stem}_interface_only.o"
        compile_commands[f"interface/{source.name}"] = _compile_to(
            command, source.name, source, destination, work, build / "atm/obj",
        )
        compile_logs[f"interface/{source.name}"] = str(log)
    # FTorch, which the hooks module uses to run bound TorchScript models: its
    # module files at compile time, its library (and libtorch) at link time
    ftorch_root = args.ftorch_root.resolve()
    ftorch_include = ftorch_root / "include/ftorch"
    ftorch_lib = next((d for d in (ftorch_root / "lib64", ftorch_root / "lib") if (d / "libftorch.so").is_file()), None)
    if not (ftorch_include / "ftorch.mod").is_file() or ftorch_lib is None:
        raise RuntimeError(f"FTorch is not installed under {ftorch_root}: build it first (see docs/installation.md)")
    torch_lib = args.torch_lib.resolve() if args.torch_lib is not None else _torch_lib_dir()
    ftorch_link = [f"-L{ftorch_lib}", "-lftorch", f"-Wl,-rpath,{ftorch_lib}", f"-Wl,-rpath,{torch_lib}"]
    cxx_runtime = ftorch_root / "cxx_runtime_dir"          # written by tools/build_ftorch.sh
    if cxx_runtime.is_file() and cxx_runtime.read_text().strip():
        ftorch_link.append(f"-Wl,-rpath,{cxx_runtime.read_text().strip()}")
    # The support modules first: physpkg and cam_comp `use` them, and ifort
    # writes their .mod files into the working directory, which -I. covers.
    support_objects: list[Path] = []
    for source_name in SUPPORT_MODULES:
        source = source_root / "components/cam/src/physics/cam" / source_name
        if not source.is_file():
            raise RuntimeError(f"support module is absent from the prepared source: {source}")
        log, command = _compile_command(build, "macrop_driver.F90")
        if source_name == "pycam_hooks.F90":
            # the hooks call FTorch (bound TorchScript models): its module files
            command = [*command, f"-I{ftorch_include}"]
        destination = work / f"{Path(source_name).stem}.o"
        compile_commands[source_name] = _compile_to(
            command, "macrop_driver.F90", source, destination, work, build / "atm/obj",
        )
        compile_logs[source_name] = str(log)
        support_objects.append(destination)
    _refuse_duplicate_globals(support_objects)
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
    stage7_runner_object: Path | None = None
    stage7_runner_compile: list[str] | None = None
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
        # The stage-7 segment runner is compiled like the leaf adapter: outside
        # the tree, against the public adapter (for cam_in) and the tree's
        # modules (the mm handles, physpkg's neighbours), and linked into the
        # leaf library.  It replaces no object: it is a new module that calls
        # the originals.
        stage7_runner_object = work / "pi_cam_stage7_runner.o"
        stage7_runner_compile = _compile_to(
            adapter_command,
            "cam_comp.F90",
            REPO / "native/pi_cam/support/pycam_stage7_runner.F90",
            stage7_runner_object,
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
    # the fiber the hooked stages run on: a second stack the hooks yield from
    fiber_source = REPO / "native/pi_cam/pycam_fiber.c"
    fiber_object = work / "pycam_fiber.o"
    fiber_compile = ["cc", "-c", "-O2", str(fiber_source), "-o", str(fiber_object)]
    _run(fiber_compile, cwd=work)
    # the kernel execution counters: the table and its context are linked into
    # every image so the ABI is uniform; trampolines exist only in a counting image
    kcount_source = REPO / "native/pi_cam/pycam_kcount.c"
    kcount_object = work / "pycam_kcount.o"
    kcount_compile = ["cc", "-c", "-O2", str(kcount_source), "-o", str(kcount_object)]
    _run(kcount_compile, cwd=work)

    atm_archive = output.parent / "libatm_nonpic_python_control.a"
    base_atm_archive = build / "lib/libatm.a"
    # The support modules go into the archive as new members as well as onto
    # the fixed link line: the capture executable is a pure CESM link against
    # this archive, and the replaced physpkg and cam_comp objects reference
    # them.  An added member is never a replaced numerical object.
    support_additions = tuple(path.name for path in support_objects)
    replacement_objects: tuple[Path, ...] = (
        objects["physpkg.F90"],
        objects["cam_comp.F90"],
        objects["atm_comp_mct.F90"],
        *support_objects,
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
            *support_objects,
        )
    hook_table = load_hooks()
    # count-only observation: a scope selects kernels whose symbol a counting
    # trampoline takes (weaken-and-alias on the defining object).  '' keeps
    # today's image (hooks only); 'batch-a' keeps the hooks and adds the
    # trampolines that do not collide with a hook-owned symbol; anything else
    # ('all', process ids, routine names) drops the replacement hooks -- a
    # counting image counts, it does not replace.
    kcount_scope = args.kcount_scope.strip()
    # the replacement hooks stay linked and redirected in every image (their
    # module strongly defines the weaken-definition kernels' symbols, so an
    # image without the redirection would resolve those symbols ambiguously)
    hooks_active = True
    plans = _hook_plans(hook_table)
    kcount_rows: list[dict] = []
    kcount_chained: list[dict] = []
    trampoline_source_sha = None
    kcount_observability_sha = None
    if kcount_scope:
        kcount_gen = _load_tool("generate_pi_cam_kcount_trampolines")
        observability = kcount_gen.load_observability()
        kcount_observability_sha = observability["content_hash"]
        selection = kcount_gen.select_kernels(
            observability, "cldfrc_fice" if kcount_scope == "batch-a" else kcount_scope)
        hooks_by_symbol = {hook.callee_symbol: hook for hook in hook_table.hooks
                           if hook.redirect == "weaken-definition"}
        for kernel in selection:
            hook = hooks_by_symbol.get(kernel["symbol"])
            if hook is None:
                kcount_rows.append(kernel)
                continue
            # the hook owns this kernel's symbol: chain the count between the
            # hook and the original -- the trampoline takes the hook's alias
            # name and jumps to a second alias at the same address, so every
            # call (hooked, armed or not) is counted exactly once
            chained = dict(kernel)
            chained["symbol"] = hook.original_symbol
            chained["chained_from"] = kernel["symbol"]
            kcount_rows.append(chained)
            kcount_chained.append({"qualified": kernel["qualified"], "kernel_symbol": kernel["symbol"],
                                   "trampoline_symbol": hook.original_symbol,
                                   "reason": "the replacement hook owns the kernel's symbol; the counting "
                                             "trampoline sits between the hook and the original"})
            for ops in plans.values():
                for op in ops:
                    if op["kind"] == "weaken-alias" and op.get("alias") == hook.original_symbol:
                        op["alias"] = None      # the trampoline provides that name now
        if not kcount_rows:
            raise RuntimeError(f"kcount scope {kcount_scope!r} selects no instrumentable kernel")
        trampoline_rows = kcount_gen.trampoline_table(kcount_rows)
        trampoline_asm = generated_objects / "pycam_kcount_trampolines.S"
        generated_objects.mkdir(parents=True, exist_ok=True)
        trampoline_asm.write_text(kcount_gen.render(observability, kcount_rows, kcount_scope))
        trampoline_source_sha = _sha256(trampoline_asm)
        trampoline_object = generated_objects / "pycam_kcount_trampolines.o"
        _run(["cc", "-c", str(trampoline_asm), "-o", str(trampoline_object)], cwd=work)
        _refuse_trampoline_collisions(trampoline_object, support_objects)
        plain_rows = [row for row, kernel in zip(trampoline_rows, kcount_rows) if "chained_from" not in kernel]
        for object_name, ops in _kcount_plans(plain_rows).items():
            plans.setdefault(object_name, []).extend(ops)
        for row, kernel in zip(trampoline_rows, kcount_rows):
            if "chained_from" not in kernel:
                continue
            # the alias the chained trampoline jumps to sits at the real
            # definition's address; the hook op already weakened that symbol
            record = {"kernel": row["routine"], "qualified": row["qualified"], "mode": "count-alias-at",
                      "callee_symbol": kernel["chained_from"], "hook_symbol": row["symbol"],
                      "original_symbol": row["alias"], "index": row["index"]}
            plans.setdefault(row["object"], []).append(
                {"kind": "alias-at", "symbol": kernel["chained_from"], "alias": row["alias"], "record": record})
    else:
        trampoline_rows = []
        trampoline_object = None
    redirections, redirected_objects = _apply_redirections(base_atm_archive, generated_objects / "redirected", plans)
    hook_redirections = [r for r in redirections if r["mode"] in ("rename-references", "weaken-definition")]
    count_redirections = [r for r in redirections if r["mode"] in ("count-weaken-alias", "count-alias-at")]
    if not hooks_active:
        hook_redirections = []
    replacement_objects = (*replacement_objects, *redirected_objects)
    _replace_archive(
        base_atm_archive,
        atm_archive,
        replacement_objects,
        additions=(
            *(("pycam_python_state_registry.o",) if args.zero_copy_state else ()),
            *support_additions,
        ),
    )

    link_log, original_link = _link_command(build)
    patched_link = _replace_library(original_link, atm_archive)
    capture_executable = args.capture_executable.resolve()
    capture_link = list(patched_link)
    capture_link.insert(capture_link.index("-o"), str(floating_environment_object))
    capture_link.insert(capture_link.index("-o"), str(fiber_object))
    capture_link.insert(capture_link.index("-o"), str(kcount_object))
    if trampoline_object is not None:
        # the chained trampolines own names the hook module references
        capture_link.insert(capture_link.index("-o"), str(trampoline_object))
    for flag in ftorch_link:
        capture_link.insert(capture_link.index("-o"), flag)
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
        *(str(path) for path in support_objects),
        str(direct_kernel_object),
        str(floating_environment_object),
        str(fiber_object),
        str(kcount_object),
        *((str(trampoline_object),) if trampoline_object is not None else ()),
        *ftorch_link,
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
            *([str(stage7_runner_object)] if stage7_runner_object is not None else []),
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
        if stage7_runner_object is not None and "pycam_stage7_start_v1" not in leaf_symbols:
            raise RuntimeError("PI-CAM leaf library lacks the stage-7 segment runner's entries")
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
        "fiber": {"source": str(fiber_source), "source_sha256": _sha256(fiber_source), "compile_command": fiber_compile,
                  "stack_bytes": hook_table.fiber_stack_bytes},
        "hooks": {"table": str(HOOKS), "table_sha256": hook_table.sha256, "redirections": hook_redirections,
                  "active": hooks_active},
        "kernel_counts": {
            "scope": kcount_scope or None,
            "observability_sha256": kcount_observability_sha,
            "table": {"slots": 64, "max_kernels": 1024, "dtype": "int64",
                      "threading": "single-threaded per rank only; refuse OpenMP threads"},
            "source": str(kcount_source), "source_sha256": _sha256(kcount_source),
            "compile_command": kcount_compile,
            "trampolines_sha256": trampoline_source_sha,
            "instrumented": [{k: row[k] for k in ("index", "qualified", "routine", "symbol", "alias",
                                                  "object", "coverage", "processes")}
                             for row in trampoline_rows],
            "chained_through_hooks": kcount_chained,
            "redirections": count_redirections,
        },
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
        "stage7_runner_compile_command": stage7_runner_compile,
        "leaf_link_command": leaf_link,
        "direct_kernel_compile_command": direct_kernel_compile,
        "promoted_kernel_compile_command": promoted_kernel_compile,
        "promoted_kernel_link_command": promoted_kernel_link,
        "capture_executable": str(capture_executable),
        "capture_executable_sha256": _sha256(capture_executable),
        "ftorch": {"root": str(ftorch_root), "library": str(ftorch_lib / "libftorch.so"),
                   "library_sha256": _sha256(ftorch_lib / "libftorch.so"), "torch_lib": str(torch_lib)},
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
