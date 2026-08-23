"""Shared helpers for building PI-CAM native images from the oracle build.

Every tool that compiles a CAM source with the production command, swaps an
object into a copy of the oracle archive, links a fixed-address image, or
proves what that image contains goes through these functions, so the
evidence they produce means the same thing everywhere.  Nothing here
recompiles numerical code on its own initiative: callers pass the exact
source and the recovered production command.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def run(command: list[str] | tuple[str, ...], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def xml(case: Path, name: str) -> str:
    return subprocess.run(
        [str(case / "xmlquery"), name, "--value"],
        cwd=case,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            return stream.read()
    return path.read_text(errors="replace")


def logs(build: Path, stem: str) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (*build.glob(f"{stem}.*"), *build.glob(f"{stem}.*.gz")),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    )


def compile_command(build: Path, source: str) -> tuple[Path, list[str]]:
    for path in logs(build, "atm.bldlog"):
        for line in reversed(text(path).splitlines()):
            if line.startswith("ftn ") and source in line and " -c " in line:
                return path, shlex.split(line)
    raise RuntimeError(f"cannot recover compile command for {source}")


def link_command(build: Path) -> tuple[Path, list[str]]:
    for path in logs(build, "cesm.bldlog"):
        for line in reversed(text(path).splitlines()):
            if line.startswith("ftn ") and "cesm.exe" in line and " -o " in line:
                return path, shlex.split(line)
    raise RuntimeError("cannot recover the CESM link command")


def without_output(command: list[str]) -> list[str]:
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


def compile_to(
    command: list[str],
    source_name: str,
    source: Path,
    output: Path,
    cwd: Path,
    module_include: Path,
    *,
    pic: bool = False,
) -> list[str]:
    result = without_output(command)
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
    run(result, cwd=cwd)
    return result


def replace_archive(
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
        run(["ar", "r", str(destination), str(replacement)], cwd=destination.parent)
    run(["ranlib", str(destination)], cwd=destination.parent)


def global_text_symbols(path: Path) -> tuple[str, ...]:
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


def global_defined_symbols(path: Path) -> dict[str, str]:
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


def addon_module_object(
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

    generated_symbols = global_defined_symbols(generated)
    production_symbols = global_defined_symbols(production)
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
    run(command, cwd=destination.parent)


def renamed_object(
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
    run(command, cwd=destination.parent)


def hybrid_module_object(
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

    generated_symbols = global_text_symbols(generated)
    numerical_symbols = set(global_text_symbols(numerical))
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
    renamed_object(
        generated,
        generated_copy,
        generated_numerics,
        "__pycam_generated_unused_",
    )
    renamed_object(
        numerical,
        numerical_copy,
        original_shells,
        "__pycam_original_unused_",
    )
    run(
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


def replace_library(command: list[str], archive: Path) -> list[str]:
    return [str(archive) if value == "-latm" else value for value in command]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_range(path: Path) -> tuple[int, int]:
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


def zero_calls(path: Path) -> tuple[str, ...]:
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


def direct_kernel_call_proof(path: Path, symbol: str, routine: str) -> str:
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


def runtime_library(executable: Path, name: str) -> Path:
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


__all__ = [
    "addon_module_object",
    "compile_command",
    "compile_to",
    "direct_kernel_call_proof",
    "global_defined_symbols",
    "global_text_symbols",
    "hybrid_module_object",
    "link_command",
    "load_range",
    "logs",
    "renamed_object",
    "replace_archive",
    "replace_library",
    "run",
    "runtime_library",
    "sha256",
    "text",
    "without_output",
    "xml",
    "zero_calls",
]
