#!/usr/bin/env python3
"""Link one physics routine from the oracle's own objects into a standalone image.

The image hosts a single CAM routine outside the model: no cam_init, no MPI,
no StatePool.  It is linked from the exact archive members the oracle build
produced (numerics are never recompiled), one generated pointer-table
wrapper for the routine, and a small set of link-time stubs the function's
reviewed YAML declares.  The link is closed -- no archives, no unresolved
symbols tolerated -- and every claim about the result is audited and
recorded in the manifest: which members went in and their hashes, that the
stub set is exactly what the objects need, that the wrapper calls the
original routine, where the image loads, and that a plain Python process can
load it, set the floating-point environment, run the initializers, and see
the abort bridge fire.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pi_cam_build_common import (  # noqa: E402
    compile_command,
    compile_to,
    direct_kernel_call_proof,
    load_range,
    run,
    runtime_library,
    sha256,
    xml,
    zero_calls,
)
from freecam.physics.spec import FunctionSpec, load_function_spec  # noqa: E402
from freecam.pi_cam.kernel_codegen import (  # noqa: E402
    DirectKernel,
    DirectKernelArgument,
    generate_direct_kernel_module,
)

REPO = Path(__file__).resolve().parents[1]
STANDALONE = REPO / "native" / "pi_cam" / "standalone"
FLOATING_ENVIRONMENT = REPO / "native" / "pi_cam" / "floating_environment.c"
IMAGE_WINDOW_BYTES = 0x08000000
EXIT_ABORT = 86
EXIT_STUB_CALLED = 87
ABORT_PROBE_SYMBOL = "freecam_standalone_abort_probe_v1"
PROBE_MESSAGE = "freecam standalone abort probe"

# Every inert stub has a shape the C side must know; an inert symbol the
# YAML names that is not listed here is a build error, not a guess.
INERT_KINDS = {
    "cam_history_mp_outfld_": "FREECAM_INERT_VOID",
    "cam_history_mp_hist_fld_active_": "FREECAM_INERT_FALSE",
    "shr_assert_mod_mp_shr_assert_in_domain_0d_double_": "FREECAM_INERT_VOID",
    "spmd_utils_mp_masterproc_": "FREECAM_INERT_DATA_INT32",
    "mpishorthand_mp_mpicom_": "FREECAM_INERT_DATA_INT32",
    "mpishorthand_mp_mpir8_": "FREECAM_INERT_DATA_INT32",
    "mpishorthand_mp_mpichar_": "FREECAM_INERT_DATA_INT32",
}
# Symbols the compiler runtime and libc provide; never candidates for stubs.
RUNTIME_SYMBOL = re.compile(r"^(?:for_|for__|_intel_|__intel_|__svml|_svml|__kmpc|_mm_|f_|d_|__libm|__stack_chk)")
# Inputs the link may legitimately open besides our own objects.
RUNTIME_PATH = re.compile(r"^(?:/opt/cray/|/glade/u/apps/|/usr/lib|/lib|/usr/x86_64|/opt/intel)")
FORBIDDEN_INPUT = re.compile(r"libatm|libcsm_share|libpio|libmct|libgptl|libmpeu|cam_history|phys_grid|physics_buffer")
ITEMSIZE = {"float64": 8, "int32": 4, "int64": 8, "S16": 16}


def standalone_kernel(spec: FunctionSpec) -> DirectKernel:
    """The one-kernel descriptor: every argument, with its extents declared."""

    arguments = []
    fields = set(DirectKernelArgument.__dataclass_fields__)
    for item in spec.arguments:
        extra: dict[str, object] = {}
        if item.pointer:
            if "pointer" not in fields:
                raise RuntimeError(
                    f"{spec.function}.{item.name} is a POINTER dummy; the wrapper "
                    "generator in this checkout cannot express it"
                )
            extra["pointer"] = True
        if item.carrier == "logical":
            if "fortran_type" not in fields:
                raise RuntimeError(
                    f"{spec.function}.{item.name} is a logical; the wrapper "
                    "generator in this checkout cannot express it"
                )
            extra["fortran_type"] = "logical"
        arguments.append(
            DirectKernelArgument(
                field=f"{spec.function}.{item.name}",
                dtype=np.dtype(item.dtype).str,
                rank=item.rank + 1,
                intent=item.intent,
                chunk_axis=item.rank + 1,
                extents=(*item.native_shape, "chunks"),
                **extra,
            )
        )
    axes = tuple(
        axis for axis in ("pcols", "pver", "pverp")
        if any(axis in item.native_shape for item in spec.arguments)
    )
    modules: list[tuple[str, tuple[str, ...]]] = []
    if spec.module:
        modules.append((spec.module, (spec.routine,)))
    if axes:
        modules.append(("ppgrid", axes))
    return DirectKernel(
        name=spec.function,
        routine=spec.routine,
        symbol=f"freecam_standalone_{spec.function}_v1",
        arguments=tuple(arguments),
        modules=tuple(modules),
        action_id=1,
    )


def stub_list(spec: FunctionSpec) -> str:
    """The stub_list.h the C stub source instantiates, from the YAML."""

    lines = ["/* Generated by build_pi_cam_standalone_function.py from the function YAML. */"]
    for symbol in spec.image.stubs["inert"]:
        kind = INERT_KINDS.get(symbol)
        if kind is None:
            raise RuntimeError(f"inert stub {symbol!r} has no known shape; add it to INERT_KINDS after review")
        lines.append(f"{kind}({symbol})")
    for symbol in spec.image.stubs["fail_closed"]:
        lines.append(f"FREECAM_FAIL_CLOSED({symbol})")
    for symbol in spec.image.stubs["abort"]:
        lines.append(f"FREECAM_ABORT({symbol})")
    return "\n".join(lines) + "\n"


def _nm(path: Path, *flags: str) -> list[tuple[str, str, str]]:
    output = subprocess.run(["nm", *flags, str(path)], check=True, capture_output=True, text=True).stdout
    # Rows are "[address] [size] type name"; the size column exists only
    # with -S.  Returned as (size, type, name).
    rows: list[tuple[str, str, str]] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            rows.append(("", fields[0], fields[1]))
        elif len(fields) == 3:
            rows.append(("", fields[1], fields[2]))
        elif len(fields) >= 4:
            rows.append((fields[1], fields[-2], fields[-1]))
    return rows


def undefined_symbols(objects: list[Path]) -> set[str]:
    undefined: set[str] = set()
    defined: set[str] = set()
    for path in objects:
        for _, kind, name in _nm(path, "-g"):
            if kind == "U":
                undefined.add(name)
            elif kind not in ("U", "w", "v"):
                defined.add(name)
    return undefined - defined


def audit_stub_set(spec: FunctionSpec, objects: list[Path]) -> dict[str, object]:
    """The stubs must be exactly what the objects need and nothing more."""

    needed = {
        name for name in undefined_symbols(objects)
        if name.endswith("_") and not RUNTIME_SYMBOL.match(name)
    }
    declared = set(spec.image.stub_symbols)
    missing = sorted(needed - declared)
    unused = sorted(declared - needed)
    if missing or unused:
        raise RuntimeError(
            f"stub set disagrees with the objects: missing={missing} unused={unused}"
        )
    return {"cesm_symbols_needed": sorted(needed), "stubs": sorted(declared)}


def parse_link_trace(text: str, own: set[Path]) -> dict[str, list[str]]:
    """Every input the link opened, split into our objects and the runtime."""

    ours: list[str] = []
    runtime: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("ld:", "ftn", "/usr/bin/ld:", "GNU ld")):
            continue
        candidate = line.split("(")[0]
        path = Path(candidate)
        if path in own:
            ours.append(str(path))
            continue
        if FORBIDDEN_INPUT.search(line):
            raise RuntimeError(f"link opened a forbidden input: {line}")
        runtime.append(line)
    return {"own_objects": ours, "runtime_inputs": runtime}


def audit_image_symbols(image: Path, spec: FunctionSpec, wrapper_symbol: str) -> dict[str, object]:
    defined = {name: (kind, int(size, 16) if size else 0) for size, kind, name in _nm(image, "-S", "-D", "--defined-only")}
    required = [wrapper_symbol, ABORT_PROBE_SYMBOL, "pycam_pi_cam_set_fp_environment_v1", "pycam_pi_cam_get_mxcsr_v1", *spec.initializers]
    missing = [name for name in required if name not in defined]
    if missing:
        raise RuntimeError(f"image lacks symbols: {missing}")
    sizes: dict[str, int] = {}
    for entry in spec.module_state:
        if entry.symbol not in defined:
            raise RuntimeError(f"image lacks module state symbol {entry.symbol}")
        expected = ITEMSIZE[entry.dtype]
        for extent in entry.shape:
            expected *= int(extent)
        actual = defined[entry.symbol][1]
        if actual != expected:
            raise RuntimeError(
                f"module state {entry.symbol} is {actual} bytes in the image, spec expects {expected}"
            )
        sizes[entry.symbol] = actual
    cam_history = sorted(name for name in defined if name.startswith("cam_history_mp_"))
    expected_history = sorted(name for name in spec.image.stub_symbols if name.startswith("cam_history_mp_"))
    if cam_history != expected_history:
        raise RuntimeError(f"cam_history symbols in image {cam_history} are not exactly the stubs {expected_history}")
    return {"module_state_sizes": sizes, "cam_history_symbols": cam_history}


def readelf_dynamic(image: Path) -> dict[str, list[str]]:
    output = subprocess.run(["readelf", "-d", str(image)], check=True, capture_output=True, text=True).stdout
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", output)
    rpath = re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r(?:un)?path: \[([^\]]+)\]", output)
    for name in needed:
        if FORBIDDEN_INPUT.search(name):
            raise RuntimeError(f"image needs a CESM library: {name}")
    return {"needed": needed, "rpath": rpath}


_SMOKE = textwrap.dedent(
    """
    import ctypes, json, os, sys
    path, base, initializers = sys.argv[1], int(sys.argv[2], 16), sys.argv[3:]
    library = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL | os.RTLD_NOW)
    mapped = [line for line in open("/proc/self/maps") if path in line]
    loaded_base = min(int(line.split("-")[0], 16) for line in mapped)
    fortran_runtime = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libifcore" in line})
    math_runtime = sorted({line.split()[-1] for line in open("/proc/self/maps") if "libimf" in line})
    library.pycam_pi_cam_get_mxcsr_v1.restype = ctypes.c_uint32
    before = library.pycam_pi_cam_get_mxcsr_v1()
    library.pycam_pi_cam_set_fp_environment_v1()
    after = library.pycam_pi_cam_get_mxcsr_v1()
    for name in initializers:
        getattr(library, name)()
    print(json.dumps({
        "loaded_base": hex(loaded_base), "base_matches": loaded_base == base,
        "mxcsr_before": hex(before), "mxcsr_after": hex(after),
        "fortran_runtime": fortran_runtime, "math_runtime": math_runtime,
        "ld_preload": os.environ.get("LD_PRELOAD"), "initializers_returned": list(initializers),
        "mpi_initialized": False,
    }))
    """
)
_PROBE = textwrap.dedent(
    """
    import ctypes, os, sys
    library = ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_LOCAL | os.RTLD_NOW)
    message = sys.argv[2].encode()
    library.freecam_standalone_abort_probe_v1.argtypes = [ctypes.c_char_p, ctypes.c_int]
    library.freecam_standalone_abort_probe_v1(message, len(message))
    print("probe returned", flush=True)
    sys.exit(99)
    """
)


def _child_env(math_library: Path) -> dict[str, str]:
    # A Python process already holds glibc's libm in its global scope, so the
    # image's exp/log/pow would bind there instead of to Intel's libimf, which
    # is what the model binds.  Preloading the image's own libimf, as the
    # freeCAM session does, restores the model's math binding.
    env = {key: value for key, value in os.environ.items() if key != "LD_PRELOAD"}
    env["LD_PRELOAD"] = str(math_library)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def smoke_load(image: Path, spec: FunctionSpec, math_library: Path) -> dict[str, object]:
    """A bare Python child loads the image, sets FP state, runs initializers."""

    result = subprocess.run(
        [sys.executable, "-c", _SMOKE, str(image), hex(spec.image.base_address), *spec.initializers],
        capture_output=True, text=True, env=_child_env(math_library), check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"smoke load failed ({result.returncode}):\n{result.stderr}")
    record = json.loads(result.stdout.strip().splitlines()[-1])
    if not record["base_matches"]:
        raise RuntimeError(f"image loaded at {record['loaded_base']}, expected {hex(spec.image.base_address)}")
    if record["mxcsr_after"] != "0x9fc0":
        raise RuntimeError(f"floating-point environment is {record['mxcsr_after']}, expected 0x9fc0")
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE, str(image), PROBE_MESSAGE],
        capture_output=True, text=True, env=_child_env(math_library), check=False,
    )
    expected = f"FREECAM_FORTRAN_ABORT: {PROBE_MESSAGE}"
    if probe.returncode != EXIT_ABORT or expected not in probe.stderr:
        raise RuntimeError(
            f"abort probe exited {probe.returncode} (expected {EXIT_ABORT}); stderr:\n{probe.stderr}"
        )
    record["abort_probe"] = {"exit_status": probe.returncode, "stderr_line": expected}
    return record


def resolve_archives(build_root: Path, names: Iterable[str]) -> dict[str, Path]:
    """Locate each named oracle archive under the case's build root.

    Most members come from the atmosphere archive.  A routine may also need a
    numerical object from CSM share (uwshcu's shr_spfn_erfc); that archive
    lives under a compiler/mpi-specific path, so it is searched rather than
    spelled out, and a missing or ambiguous match fails the build.
    """

    known = {"atm": [build_root / "lib" / "libatm.a"]}
    archives: dict[str, Path] = {}
    for name in names:
        candidates = known.get(name) or sorted(build_root.rglob(f"lib{name}.a"))
        found = [path for path in candidates if path.is_file()]
        if not found:
            raise RuntimeError(f"archive {name!r} not found under {build_root}")
        if len({sha256(path) for path in found}) != 1:
            raise RuntimeError(
                f"archive {name!r} is ambiguous under {build_root}: " + ", ".join(map(str, found))
            )
        archives[name] = found[0]
    return archives


def build(spec: FunctionSpec, case: Path, output_root: Path) -> dict[str, object]:
    build_root = Path(xml(case, "EXEROOT")).resolve()
    archives = resolve_archives(build_root, spec.image.archives)
    archive = archives["atm"]
    module_include = build_root / "atm" / "obj"
    target = output_root / spec.function
    work = target / "objects"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    members: list[dict[str, str]] = []
    member_objects: list[Path] = []
    for item in spec.image.archive_members:
        source = archives[item.archive]
        run(["ar", "x", str(source), item.member], cwd=work)
        path = work / item.member
        digest = sha256(path)
        listed = subprocess.run(
            ["ar", "p", str(source), item.member], check=True, capture_output=True
        ).stdout
        if hashlib.sha256(listed).hexdigest() != digest:
            raise RuntimeError(f"extracted {item.member} differs from the {item.archive} archive member")
        members.append({"name": item.member, "archive": item.archive, "sha256": digest})
        member_objects.append(path)

    kernel = standalone_kernel(spec)
    wrapper_source = work / f"freecam_standalone_{spec.function}.F90"
    wrapper_source.write_text(
        generate_direct_kernel_module((kernel,), module_name=f"freecam_standalone_{spec.function}")
    )
    routine_file = Path(spec.source).name
    _, production = compile_command(build_root, routine_file)
    wrapper_object = work / "wrapper.o"
    wrapper_compile = compile_to(production, routine_file, wrapper_source, wrapper_object, work, module_include)
    probe_object = work / "abort_probe.o"
    probe_compile = compile_to(
        production, routine_file, STANDALONE / "abort_probe.F90", probe_object, work, module_include
    )
    (work / "stub_list.h").write_text(stub_list(spec))
    stubs_object = work / "cam_stubs.o"
    stubs_compile = ["cc", "-c", "-O2", f"-I{work}", str(STANDALONE / "cam_stubs.c"), "-o", str(stubs_object)]
    run(stubs_compile, cwd=work)
    fp_object = work / "floating_environment.o"
    fp_compile = ["cc", "-c", "-O2", str(FLOATING_ENVIRONMENT), "-o", str(fp_object)]
    run(fp_compile, cwd=work)

    objects = [wrapper_object, probe_object, stubs_object, fp_object, *member_objects]
    stub_audit = audit_stub_set(spec, [wrapper_object, probe_object, *member_objects])

    image = target / f"libfreecam_{spec.function}.so"
    executable = image.with_suffix(".exec")
    link_map = target / "link.map"
    link = [
        # The Intel runtime stays shared, as in the validated images; linked
        # statically it brings weak references that become calls to address 0.
        "ftn", "-nostartfiles", "-nofor-main", "-shared-intel",
        f"-Wl,-e,0,-Ttext-segment=0x{spec.image.base_address:x},--export-dynamic,"
        f"-Bsymbolic-functions,-soname,{image.name}",
        f"-Wl,--trace,-Map={link_map}",
        f"-Wl,-u,{kernel.symbol}", f"-Wl,-u,{ABORT_PROBE_SYMBOL}",
        *(str(path) for path in objects),
        "-o", str(executable),
    ]
    trace = subprocess.run(link, cwd=target, check=True, capture_output=True, text=True)
    trace_audit = parse_link_trace(trace.stdout + trace.stderr, set(objects))
    shutil.copy2(executable, image)
    run(["elfedit", "--output-type", "dyn", str(image)], cwd=target)

    unresolved = zero_calls(image)
    if unresolved:
        raise RuntimeError(f"image contains unresolved direct calls: {unresolved[:8]}")
    start, end = load_range(image)
    if start != spec.image.base_address or end > spec.image.base_address + IMAGE_WINDOW_BYTES:
        raise RuntimeError(f"image load range 0x{start:x}-0x{end:x} is outside its window")
    call_proof = direct_kernel_call_proof(image, kernel.symbol, spec.routine)
    symbol_audit = audit_image_symbols(image, spec, kernel.symbol)
    dynamic = readelf_dynamic(image)
    math_library = runtime_library(image, "libimf")
    smoke = smoke_load(image, spec, math_library)

    return {
        "schema_version": 1,
        "execution_model": "fixed-address-nonpic-standalone-function",
        "function": spec.function,
        "qualified_name": spec.qualified_name,
        "spec": str(spec.path),
        "spec_sha256": sha256(spec.path) if spec.path else None,
        "library": str(image),
        "library_sha256": sha256(image),
        "library_bytes": image.stat().st_size,
        "load_start": hex(start),
        "load_end": hex(end),
        "numerical_archives": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in archives.items()
        },
        "members": members,
        "wrapper": {
            "symbol": kernel.symbol,
            "action_id": kernel.action_id,
            "source": str(wrapper_source),
            "source_sha256": sha256(wrapper_source),
            "operation": kernel.operation_payload(),
        },
        "stubs": {
            "source": str(STANDALONE / "cam_stubs.c"),
            "source_sha256": sha256(STANDALONE / "cam_stubs.c"),
            "list": stub_list(spec),
            **stub_audit,
        },
        "abort_probe": {"source_sha256": sha256(STANDALONE / "abort_probe.F90")},
        "original_call_proof": call_proof,
        "link": {"command": link, "map": str(link_map), **trace_audit},
        "compile_commands": {
            "wrapper": wrapper_compile, "abort_probe": probe_compile,
            "stubs": stubs_compile, "floating_environment": fp_compile,
        },
        "dynamic": dynamic,
        "intel_math_library": str(math_library),
        "intel_math_library_sha256": sha256(math_library),
        "symbols": symbol_audit,
        "smoke_load": smoke,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", required=True)
    parser.add_argument("--case", type=Path, required=True, help="oracle CESM case root (EXEROOT holds the archive)")
    parser.add_argument("--output-root", type=Path, default=REPO / "build" / "pi_cam_standalone")
    parser.add_argument("--evidence", type=Path, default=None, help="copy of the manifest for validation/")
    args = parser.parse_args()
    spec = load_function_spec(args.function)
    manifest = build(spec, args.case.resolve(), args.output_root.resolve())
    target = args.output_root.resolve() / spec.function
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if args.evidence is not None:
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{spec.function}: {manifest['library']} ({manifest['library_sha256'][:12]})")
    print(f"  call proof: {manifest['original_call_proof']}")
    print(f"  stubs: {len(manifest['stubs']['stubs'])}  members: {len(manifest['members'])}")
    print(f"  smoke: base={manifest['smoke_load']['loaded_base']} mxcsr={manifest['smoke_load']['mxcsr_after']} abort_probe exit={manifest['smoke_load']['abort_probe']['exit_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
